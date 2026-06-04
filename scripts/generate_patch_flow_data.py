"""Generate PatchFlow training data on GPU and save as .pt shards.

Each shard is a dict saved with torch.save:
    {
        "dirty": (N_shard, 1, 128, 128) float32  -- peak-normalised
        "clean": (N_shard, 1, 128, 128) float32  -- peak-normalised
        "psf":   (N_shard, 1, 128, 128) float32  -- PSF cutout, peak=1
        "sigma": (N_shard,)             float32  -- normalised noise estimate
    }

Usage:
    pixi run -e gpu python scripts/generate_patch_flow_data.py \
        --out_dir data/patch_flow \
        --n_total 200000 \
        --batch_size 64 \
        --device cuda:0
"""
from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np
import torch

from mad_clean.data.extended_sky_torch import (
    BEAM_SIGMA_PX, SIGMA_MAX_PX,
    render_gaussian_blob_batch,
    render_shell_batch,
    render_filament_batch,
    fft_convolve_batch,
)
from mad_clean.data.psf_bank import load_g55_psf_bank

FIELD_SIZE   = 512
CUTOUT_SIZE  = 128
HALF         = CUTOUT_SIZE // 2
SIGMA_NOISE  = 1e-4
SNR_MIN      = 5.0
MORPHS       = ("point", "blob", "shell", "filament")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--out_dir",    type=str, default="data/patch_flow")
    p.add_argument("--n_total",    type=int, default=200_000)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--n_workers_psf", type=int, default=4,
                   help="Number of distractor sources per scene.")
    p.add_argument("--device",     type=str, default="cuda")
    p.add_argument("--seed",       type=int, default=42)
    return p.parse_args()


# ---------------------------------------------------------------------------
# Batched scene generation
# ---------------------------------------------------------------------------

def _rand(shape, lo, hi, device):
    if isinstance(shape, int):
        shape = (shape,)
    return torch.empty(shape, device=device).uniform_(lo, hi)


def prepare_psf(psf_np: np.ndarray, device: torch.device):
    """Load PSF to GPU, return (psf_full, psf_cut, beam_area, psf_shift, psf_fft).

    psf_fft has the shift baked in via roll-before-FFT so callers need neither
    rfft2(psf) nor torch.roll on the (B,H,W) output.
    """
    psf_full = torch.from_numpy(psf_np).float().to(device)
    py, px = (psf_full == psf_full.max()).nonzero(as_tuple=False)[0]
    py, px = int(py), int(px)
    psf_shift = (-py, -px)
    pr0, pr1 = py - HALF, py - HALF + CUTOUT_SIZE
    pc0, pc1 = px - HALF, px - HALF + CUTOUT_SIZE
    psf_cut   = _safe_crop_t(psf_full, pr0, pr1, pc0, pc1, device)
    beam_area = float(psf_cut.sum())
    H, W = psf_full.shape
    psf_rolled = torch.roll(psf_full, shifts=psf_shift, dims=(0, 1))
    psf_fft = torch.fft.rfft2(psf_rolled.unsqueeze(0), s=(H, W))
    return psf_full, psf_cut, beam_area, psf_shift, psf_fft


def generate_batch(
    B:         int,
    psf_full:  torch.Tensor,        # (H_psf, W_psf) already on device
    psf_cut:   torch.Tensor,        # (S, S) already on device
    beam_area: float,
    psf_shift: tuple[int, int],
    device:    torch.device,
    rng:       torch.Generator,
    psf_fft:   torch.Tensor | None = None,  # precomputed rfft2(roll(psf))
) -> dict[str, torch.Tensor]:
    """Generate B (dirty, clean, psf, sigma) samples. Returns CPU tensors."""
    H = W = FIELD_SIZE
    S = CUTOUT_SIZE

    # --- Centred source parameters (B,) ---
    morph_idx = torch.randint(0, 4, (B,), device=device, generator=rng)
    flux_log  = _rand(B, math.log(1e-4), math.log(1e-1), device)
    flux      = torch.exp(flux_log)
    margin    = HALF + 4
    cx = _rand(B, margin, W - margin, device)
    cy = _rand(B, margin, H - margin, device)

    # Render centred sources per morphology
    centred = torch.zeros(B, H, W, device=device)

    # Point sources (morph_idx == 0)
    pt_mask = morph_idx == 0
    if pt_mask.any():
        n_pt  = int(pt_mask.sum())
        rows  = cy[pt_mask].round().long().clamp(0, H - 1)
        cols  = cx[pt_mask].round().long().clamp(0, W - 1)
        flat  = rows * W + cols                               # (n_pt,)
        pt_imgs = torch.zeros(n_pt, H * W, device=device)
        pt_imgs.scatter_(1, flat.unsqueeze(1), flux[pt_mask].unsqueeze(1))
        centred[pt_mask] = pt_imgs.view(n_pt, H, W)

    # Blobs (morph_idx == 1)
    bl_mask = morph_idx == 1
    if bl_mask.any():
        n = int(bl_mask.sum())
        sig_maj = _rand(n, BEAM_SIGMA_PX, SIGMA_MAX_PX, device)
        sig_min = _rand(n, BEAM_SIGMA_PX, SIGMA_MAX_PX, device).clamp(max=sig_maj)
        pa = _rand(n, 0.0, math.pi, device)
        centred[bl_mask] = render_gaussian_blob_batch(
            H, cx[bl_mask], cy[bl_mask], flux[bl_mask],
            sig_maj, sig_min, pa, device)

    # Shells (morph_idx == 2)
    sh_mask = morph_idx == 2
    if sh_mask.any():
        n = int(sh_mask.sum())
        radius    = _rand(n, BEAM_SIGMA_PX, SIGMA_MAX_PX, device)
        thickness = _rand(n, BEAM_SIGMA_PX * 0.5, BEAM_SIGMA_PX, device) + \
                    _rand(n, 0.0, 1.0, device) * (radius * 0.5).clamp(min=0.0)
        centred[sh_mask] = render_shell_batch(
            H, cx[sh_mask], cy[sh_mask], flux[sh_mask],
            radius, thickness, device)

    # Filaments (morph_idx == 3)
    fi_mask = morph_idx == 3
    if fi_mask.any():
        n = int(fi_mask.sum())
        length = _rand(n, 2 * BEAM_SIGMA_PX, 2 * SIGMA_MAX_PX, device)
        width  = _rand(n, BEAM_SIGMA_PX, SIGMA_MAX_PX, device)
        pa     = _rand(n, 0.0, math.pi, device)
        centred[fi_mask] = render_filament_batch(
            H, cx[fi_mask], cy[fi_mask], flux[fi_mask],
            length, width, pa, device)

    # --- Distractor scene: fully vectorised, no Python loop ---
    N_DIST = 20
    dr   = torch.randint(16, H - 16, (N_DIST, B), device=device, generator=rng)
    dc   = torch.randint(16, W - 16, (N_DIST, B), device=device, generator=rng)
    df   = _rand((N_DIST, B), 1e-4, 1e-1, device)
    flat = (dr * W + dc).T.contiguous()                       # (B, N_DIST)
    distractors = torch.zeros(B, H * W, device=device)
    distractors.scatter_add_(1, flat, df.T.contiguous())
    distractors = distractors.view(B, H, W)

    sky = distractors + centred   # (B, H, W)

    # --- SNR floor per sample ---
    dirty_centred = fft_convolve_batch(centred, psf_fft=psf_fft)   # (B, H, W)
    conv_peak = dirty_centred.abs().amax(dim=(1, 2))               # (B,)
    snr_thresh = SNR_MIN * SIGMA_NOISE
    scale = (snr_thresh / conv_peak.clamp(min=1e-30)).clamp(min=1.0)
    needs_scale = conv_peak < snr_thresh
    centred[needs_scale] = centred[needs_scale] * scale[needs_scale, None, None]
    sky = distractors + centred

    # --- Dirty image ---
    dirty_full = fft_convolve_batch(sky, psf_fft=psf_fft)          # (B, H, W)
    noise = torch.randn(B, H, W, device=device) * SIGMA_NOISE
    dirty_full = dirty_full + noise

    # --- Crop cutouts (vectorised) ---
    cr = cy.round().long().clamp(HALF, H - HALF - 1)  # (B,)
    cc = cx.round().long().clamp(HALF, W - HALF - 1)  # (B,)

    # Build row/col index grids for all samples at once
    offsets = torch.arange(S, device=device) - HALF          # (S,)
    row_idx = (cr[:, None] + offsets[None, :]).clamp(0, H-1) # (B, S)
    col_idx = (cc[:, None] + offsets[None, :]).clamp(0, W-1) # (B, S)

    # Expand to (B, S, S) for gather
    ri = row_idx[:, :, None].expand(B, S, S)                 # (B, S, S)
    ci = col_idx[:, None, :].expand(B, S, S)                 # (B, S, S)
    flat_idx = ri * W + ci                                    # (B, S, S)

    dirty_cuts = dirty_full.flatten(1).gather(1, flat_idx.flatten(1)).view(B, S, S)
    clean_cuts = centred.flatten(1).gather(1, flat_idx.flatten(1)).view(B, S, S)

    # --- Beam area + peak normalisation ---
    if beam_area > 0:
        dirty_cuts = dirty_cuts / beam_area

    dirty_peak = dirty_cuts.abs().amax(dim=(1, 2)).clamp(min=1e-30)  # (B,)
    dirty_cuts = dirty_cuts / dirty_peak[:, None, None]
    clean_cuts = clean_cuts / dirty_peak[:, None, None]

    # --- Noise estimate ---
    sigma = dirty_cuts.flatten(1).abs().median(dim=1).values * 1.4826  # (B,)

    return {
        "dirty": dirty_cuts.unsqueeze(1),                          # (B, 1, S, S) on device
        "clean": clean_cuts.unsqueeze(1),                          # (B, 1, S, S) on device
        "psf":   psf_cut.unsqueeze(0).expand(B, -1, -1).unsqueeze(1).contiguous(),  # (B, 1, S, S)
        "sigma": sigma,                                            # (B,) on device
    }


def _safe_crop_t(arr, r0, r1, c0, c1, device):
    H, W = arr.shape
    out = torch.zeros(r1 - r0, c1 - c0, dtype=torch.float32, device=device)
    sr0 = max(0, r0); sr1 = min(H, r1)
    sc0 = max(0, c0); sc1 = min(W, c1)
    if sr1 > sr0 and sc1 > sc0:
        out[sr0 - r0:sr1 - r0, sc0 - c0:sc1 - c0] = arr[sr0:sr1, sc0:sc1]
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    torch.manual_seed(args.seed)
    rng = torch.Generator(device=device)
    rng.manual_seed(args.seed)

    psf_bank = load_g55_psf_bank(_REPO_ROOT)
    rng_np   = np.random.default_rng(args.seed)

    B         = args.batch_size
    n_total   = args.n_total
    n_batches = math.ceil(n_total / B)

    print(f"Generating {n_total} samples in {n_batches} batches of {B} on {device}")
    print("Warming up CUDA ...", flush=True)
    torch.zeros(1, device=device)
    print(f"CUDA ready: {torch.cuda.get_device_name(device)}", flush=True)

    # Preload all PSFs to GPU with FFTs precomputed (shift baked in)
    print(f"Preloading {len(psf_bank)} PSFs to GPU ...", flush=True)
    psf_entries = [prepare_psf(psf_bank[idx][0], device) for idx in range(len(psf_bank))]
    print(f"  {len(psf_entries)} PSFs ready.", flush=True)

    generated = 0
    shard_idx = 0

    t0 = time.time()
    for i in range(n_batches):
        psf_full, psf_cut, beam_area, psf_shift, psf_fft = \
            psf_entries[int(rng_np.integers(len(psf_entries)))]
        if i < 3 or i % 50 == 0:
            print(f"  batch {i+1}/{n_batches} starting ...", flush=True)
        batch = generate_batch(B, psf_full, psf_cut, beam_area, psf_shift, device, rng, psf_fft)
        if i < 3 or i % 50 == 0:
            elapsed = time.time() - t0
            rate = (i + 1) * B / max(elapsed, 1e-6)
            eta = (n_batches - i - 1) / max(rate / B, 1e-6)
            print(f"  batch {i+1}/{n_batches} done  samples={generated + B}  "
                  f"rate={rate:.0f}/s  eta={eta/60:.1f}min", flush=True)

        # Save each batch directly -- no in-memory accumulation
        shard = {
            "dirty": batch["dirty"].cpu(),
            "clean": batch["clean"].cpu(),
            "psf":   batch["psf"].cpu(),
            "sigma": batch["sigma"].cpu(),
        }
        path = out_dir / f"shard_{shard_idx:04d}.pt"
        torch.save(shard, path)
        shard_idx += 1
        generated += B

    elapsed = time.time() - t0
    print(f"Done. {generated} samples in {shard_idx} shards ({elapsed/60:.1f}min).")


if __name__ == "__main__":
    main()
