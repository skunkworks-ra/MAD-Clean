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
    return torch.empty(shape, device=device).uniform_(lo, hi)


def generate_batch(
    B:       int,
    psf_np:  np.ndarray,
    device:  torch.device,
    rng:     torch.Generator,
) -> dict[str, torch.Tensor]:
    """Generate B (dirty, clean, psf, sigma) samples. Returns CPU tensors."""
    H = W = FIELD_SIZE
    S = CUTOUT_SIZE

    psf_full = torch.from_numpy(psf_np).float().to(device)   # (H_psf, W_psf)
    Hp, Wp   = psf_full.shape

    # PSF cutout (same for all samples in batch -- same PSF)
    py, px = (psf_full == psf_full.max()).nonzero(as_tuple=False)[0]
    py, px = int(py), int(px)
    pr0, pr1 = py - HALF, py - HALF + S
    pc0, pc1 = px - HALF, px - HALF + S
    psf_cut = _safe_crop_t(psf_full, pr0, pr1, pc0, pc1, device)  # (S, S)
    beam_area = float(psf_cut.sum())

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
        rows = cy[pt_mask].round().long().clamp(0, H - 1)
        cols = cx[pt_mask].round().long().clamp(0, W - 1)
        pt_imgs = torch.zeros(pt_mask.sum(), H, W, device=device)
        for i, (r, c, f) in enumerate(zip(rows, cols, flux[pt_mask])):
            pt_imgs[i, r, c] = f
        centred[pt_mask] = pt_imgs

    # Blobs (morph_idx == 1)
    bl_mask = morph_idx == 1
    if bl_mask.any():
        n = int(bl_mask.sum())
        sig_maj = _rand(n, BEAM_SIGMA_PX, SIGMA_MAX_PX, device)
        sig_min = torch.stack([_rand(1, BEAM_SIGMA_PX, float(s), device)[0]
                               for s in sig_maj])
        pa = _rand(n, 0.0, math.pi, device)
        centred[bl_mask] = render_gaussian_blob_batch(
            H, cx[bl_mask], cy[bl_mask], flux[bl_mask],
            sig_maj, sig_min, pa, device)

    # Shells (morph_idx == 2)
    sh_mask = morph_idx == 2
    if sh_mask.any():
        n = int(sh_mask.sum())
        radius    = _rand(n, BEAM_SIGMA_PX, SIGMA_MAX_PX, device)
        thickness = torch.stack([
            _rand(1, BEAM_SIGMA_PX * 0.5, max(BEAM_SIGMA_PX, float(r) * 0.5), device)[0]
            for r in radius])
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

    # --- Distractor scene (simple: random point sources) ---
    n_dist = int(torch.randint(5, 31, (1,), device=device, generator=rng))
    distractors = torch.zeros(B, H, W, device=device)
    for _ in range(n_dist):
        dr = torch.randint(16, H - 16, (B,), device=device, generator=rng)
        dc = torch.randint(16, W - 16, (B,), device=device, generator=rng)
        df = _rand(B, 1e-4, 1e-1, device)
        for b in range(B):
            distractors[b, dr[b], dc[b]] += df[b]

    sky = distractors + centred   # (B, H, W)

    # --- SNR floor per sample ---
    dirty_centred = fft_convolve_batch(centred, psf_full)          # (B, H, W)
    conv_peak = dirty_centred.abs().amax(dim=(1, 2))               # (B,)
    snr_thresh = SNR_MIN * SIGMA_NOISE
    scale = (snr_thresh / conv_peak.clamp(min=1e-30)).clamp(min=1.0)
    needs_scale = conv_peak < snr_thresh
    centred[needs_scale] = centred[needs_scale] * scale[needs_scale, None, None]
    sky = distractors + centred

    # --- Dirty image ---
    dirty_full = fft_convolve_batch(sky, psf_full)                 # (B, H, W)
    noise = torch.randn(B, H, W, device=device) * SIGMA_NOISE
    dirty_full = dirty_full + noise

    # --- Crop cutouts ---
    cr = cy.round().long().clamp(HALF, H - HALF - 1)
    cc = cx.round().long().clamp(HALF, W - HALF - 1)

    dirty_cuts = torch.zeros(B, S, S, device=device)
    clean_cuts = torch.zeros(B, S, S, device=device)
    for b in range(B):
        r0, r1 = int(cr[b]) - HALF, int(cr[b]) - HALF + S
        c0, c1 = int(cc[b]) - HALF, int(cc[b]) - HALF + S
        dirty_cuts[b] = dirty_full[b, r0:r1, c0:c1]
        clean_cuts[b] = centred[b,  r0:r1, c0:c1]

    # --- Beam area + peak normalisation ---
    if beam_area > 0:
        dirty_cuts = dirty_cuts / beam_area

    dirty_peak = dirty_cuts.abs().amax(dim=(1, 2)).clamp(min=1e-30)  # (B,)
    dirty_cuts = dirty_cuts / dirty_peak[:, None, None]
    clean_cuts = clean_cuts / dirty_peak[:, None, None]

    # --- Noise estimate ---
    sigma = dirty_cuts.flatten(1).abs().median(dim=1).values * 1.4826  # (B,)

    return {
        "dirty": dirty_cuts.unsqueeze(1).cpu(),   # (B, 1, S, S)
        "clean": clean_cuts.unsqueeze(1).cpu(),   # (B, 1, S, S)
        "psf":   psf_cut.unsqueeze(0).expand(B, -1, -1).unsqueeze(1).cpu(),  # (B, 1, S, S)
        "sigma": sigma.cpu(),                     # (B,)
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

    generated = 0
    shard_idx = 0
    shard_dirty, shard_clean, shard_psf, shard_sigma = [], [], [], []

    t0 = time.time()
    for i in range(n_batches):
        psf_np, _ = psf_bank.sample(rng_np)
        batch = generate_batch(B, psf_np, device, rng)

        shard_dirty.append(batch["dirty"])
        shard_clean.append(batch["clean"])
        shard_psf.append(batch["psf"])
        shard_sigma.append(batch["sigma"])
        generated += B

        # Save shard every 10k samples
        if generated % 10_000 == 0 or i == n_batches - 1:
            shard = {
                "dirty": torch.cat(shard_dirty),
                "clean": torch.cat(shard_clean),
                "psf":   torch.cat(shard_psf),
                "sigma": torch.cat(shard_sigma),
            }
            path = out_dir / f"shard_{shard_idx:04d}.pt"
            torch.save(shard, path)
            elapsed = time.time() - t0
            rate = generated / elapsed
            print(f"  shard {shard_idx:04d}  samples={generated}  "
                  f"rate={rate:.0f}/s  -> {path}")
            shard_dirty, shard_clean, shard_psf, shard_sigma = [], [], [], []
            shard_idx += 1

    print(f"Done. {generated} samples in {len(list(out_dir.glob('shard_*.pt')))} shards.")


if __name__ == "__main__":
    main()
