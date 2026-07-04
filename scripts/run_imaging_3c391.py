"""MDN-Asp imaging loop for 3C391.

Runs under the radiosharp pixi env which has casatasks/casatools:
  /home/pjaganna/Software/radiosharp/.pixi/envs/default/bin/python \
      scripts/run_imaging_3c391.py [options]

Loop
----
1. tclean niter=0 (major cycle) → .residual + .psf images
2. minor_cycle(): find residual peak, crop 128×128 centred on peak,
   MDN forward pass, subtract PSF response, repeat until threshold
3. Accumulate model_update into the CASA .model image
4. tclean niter=0 (calcres=True) → updated .residual
5. Repeat until residual_peak < threshold

Stopping criterion is global: measured once from the initial dirty image
and PSF, never updated.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

# --- locate MAD-Clean repo ---
_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from astropy.io import fits as astrofits

try:
    from casatasks import tclean
    from casatools import image as iatool
    _HAS_CASA = True
except ImportError:
    _HAS_CASA = False

from mad_clean.minor_cycle import minor_cycle, _safe_crop, _crop_psf_centred
from mad_clean.models.mdn_asp import MDNAsp, make_cond

_PSF_CROP = 128  # size used only for sidelobe measurement


def _measure_noise_rms(residual: np.ndarray, mask: np.ndarray, n_sigma: float = 3.0, n_iter: int = 3) -> float:
    """Sigma-clipped MAD noise estimate from within-beam pixels.

    Clips emission iteratively so bright sources don't bias the estimate.
    Uses pixels inside the primary beam mask only -- mosaic gridder zeros
    everything outside, which would collapse the MAD to zero.
    """
    pixels = residual[mask].ravel()
    for _ in range(n_iter):
        rms = 1.4826 * float(np.median(np.abs(pixels)))
        if rms <= 0:
            break
        pixels = pixels[np.abs(pixels) < n_sigma * rms]
    return 1.4826 * float(np.median(np.abs(pixels))) if len(pixels) > 0 else 0.0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--vis",    default="/home/pjaganna/Data/imaging/3c391_ctm_mosaic_spw0.ms")
    p.add_argument("--solver", choices=["mdn", "flow"], default="mdn",
                   help="Minor-cycle solver: 'mdn' (Aspen Gaussian) or 'flow' "
                        "(PSFCondFlow pixel window).")
    p.add_argument("--ckpt",   default="results/mdn_asp_v1/best.pt")
    p.add_argument("--flow_ckpt", default="models/psf_condflow_real.best.pt",
                   help="PSFCondFlow checkpoint (used when --solver flow).")
    p.add_argument("--flow_base_channels", type=int,   default=32)
    p.add_argument("--flow_asinh_a",       type=float, default=1e-2)
    p.add_argument("--speckle_frac",       type=float, default=0.01,
                   help="Zero flow-window pixels below this fraction of the "
                        "window peak (sub-sidelobe speckle cut).")
    p.add_argument("--floor_frac",         type=float, default=0.05,
                   help="PRE-rescale amplitude cut in normalised flow space: zero "
                        "every pixel below this fraction of the per-draw normalised "
                        "peak BEFORE multiplying by the window peak. Kills the "
                        "constant relative floor at a consistent level across "
                        "windows (symmetric) so the rescale can't amplify it into "
                        "speckle and seams.")
    p.add_argument("--conf_k",             type=float, default=3.0,
                   help="Keep flow pixels with posterior median >= conf_k * MAD "
                        "across draws (real-emission gate; rejects speckle).")
    p.add_argument("--out_dir", default="results/3c391_mdn_asp")
    p.add_argument("--imsize",  type=int, default=512)
    p.add_argument("--cell",    default="2.5arcsec")
    p.add_argument("--gridder", default="mosaic",
                   help="tclean gridder: mosaic | standard | wproject.")
    p.add_argument("--wprojplanes", type=int, default=-1,
                   help="w-projection planes (used when --gridder wproject).")
    p.add_argument("--mask_radius_px", type=int, default=200,
                   help="Circular mask radius in pixels from image centre (used if --mask not set).")
    p.add_argument("--mask", type=str, default=None,
                   help="Path to a CASA mask image. Overrides --mask_radius_px for tclean calls.")
    p.add_argument("--loop_gain",   type=float, default=0.1)
    p.add_argument("--max_major",   type=int,   default=20)
    p.add_argument("--max_minor",   type=int,   default=10000,
                   help="Max flow passes per minor cycle (tile sweep) — safety cap.")
    p.add_argument("--tile_stride", type=int,   default=64,
                   help="Tile-sweep stride in px (tile=128; 64 = 50%% overlap, "
                        "feathered overlap-add).")
    p.add_argument("--inner_max",   type=int,   default=3,
                   help="Flow passes per tile per MAJOR cycle (depth limit). "
                        "Small = light peel per cycle, major cycles peel deep and "
                        "correct over-model; large = clean tiles deep in one minor "
                        "cycle (risks over-commit with the approximate bookkeeping).")
    p.add_argument("--config_idx",  type=int,   default=2,
                   help="VLA config index: 0=A 1=B 2=C 3=D. 3C391 is C-config.")
    p.add_argument("--divergence_tol", type=float, default=0.20,
                   help="Stop if residual peak rises by more than this fraction between major cycles (default 0.20).")
    p.add_argument("--global_threshold", type=float, default=None,
                   help="Override computed threshold (Jy/beam). Skips noise/PSF floor estimation.")
    p.add_argument("--device", default="cpu")
    # --- FITS bypass (skip tclean; for datasets with pre-computed dirty images) ---
    p.add_argument("--dirty_fits", type=str, default=None,
                   help="Path to dirty image FITS (skips tclean when set).")
    p.add_argument("--psf_fits",   type=str, default=None,
                   help="Path to PSF FITS (skips tclean when set; required with --dirty_fits).")
    # --- Wavelet NPE probe ---
    p.add_argument("--probe_wavelet_npe", action="store_true",
                   help="Run one-shot wavelet NPE inference on the dirty image and exit.")
    p.add_argument("--wavelet_ckpt",    type=str, default=None)
    p.add_argument("--wavelet_hidden",  type=int, default=128)
    p.add_argument("--wavelet_n_layers",type=int, default=8)
    p.add_argument("--wavelet_base_channels", type=int, default=32)
    p.add_argument("--wavelet_context_dim",   type=int, default=256)
    p.add_argument("--n_posterior",     type=int, default=32)
    p.add_argument("--label",           type=str, default="",
                   help="Label for the probe figure title.")
    p.add_argument("--patch_row", type=int, default=None,
                   help="Row centre for probe patch (default: global dirty peak).")
    p.add_argument("--patch_col", type=int, default=None,
                   help="Col centre for probe patch (default: global dirty peak).")
    return p.parse_args(argv)


# ---------------------------------------------------------------------------
# CASA image helpers
# ---------------------------------------------------------------------------

def read_casa_image(imagepath: str) -> np.ndarray:
    """Read a CASA image and return a 2D float32 numpy array (ny, nx)."""
    ia = iatool()
    ia.open(imagepath)
    chunk = ia.getchunk()      # (nx, ny, npol, nchan) in CASA Fortran order
    ia.close()
    arr = chunk.squeeze()      # (nx, ny) after squeezing stokes/chan
    return arr.T.astype(np.float32)   # transpose to (ny, nx) row-major


def write_casa_image(imagepath: str, data: np.ndarray) -> None:
    """Write a 2D numpy array (ny, nx) into an existing CASA image via putchunk."""
    ia = iatool()
    ia.open(imagepath)
    # CASA chunk shape is (nx, ny, npol, nchan); data is (ny, nx) row-major
    arr = data.T.astype(np.float32)          # (nx, ny)
    arr = arr[:, :, np.newaxis, np.newaxis]  # (nx, ny, 1, 1)
    ia.putchunk(arr)
    ia.close()


# ---------------------------------------------------------------------------
# PSF sidelobe measurement
# ---------------------------------------------------------------------------

def measure_psf_sidelobe(psf: np.ndarray) -> float:
    """Return |first negative sidelobe| of the PSF."""
    return float(abs(psf.min()))


def crop_psf_centred(psf: np.ndarray, size: int) -> np.ndarray:
    py, px = np.unravel_index(int(np.argmax(psf)), psf.shape)
    half = size // 2
    r0, c0 = py - half, px - half
    H, W = psf.shape
    out = np.zeros((size, size), dtype=np.float32)
    sr0, sr1 = max(0, r0), min(H, r0 + size)
    sc0, sc1 = max(0, c0), min(W, c0 + size)
    dr0, dc0 = sr0 - r0, sc0 - c0
    out[dr0:dr0+(sr1-sr0), dc0:dc0+(sc1-sc0)] = psf[sr0:sr1, sc0:sc1]
    return out


# ---------------------------------------------------------------------------
# Circular mask
# ---------------------------------------------------------------------------

def make_mask(imsize: int, radius_px: int) -> np.ndarray:
    cy, cx = imsize // 2, imsize // 2
    Y, X = np.ogrid[:imsize, :imsize]
    return ((X - cx)**2 + (Y - cy)**2) <= radius_px**2


# ---------------------------------------------------------------------------
# FITS loader (for datasets with pre-computed dirty images)
# ---------------------------------------------------------------------------

def load_fits_image(path: str) -> np.ndarray:
    """Load a FITS image and return a 2D float32 array (ny, nx)."""
    with astrofits.open(path, memmap=False) as hdul:
        arr = np.squeeze(np.asarray(hdul[0].data)).astype(np.float32)
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D after squeeze, got {arr.shape} from {path}")
    return arr


# ---------------------------------------------------------------------------
# Wavelet NPE probe
# ---------------------------------------------------------------------------

def probe_wavelet_npe(
    residual:   np.ndarray,   # (H, W) Jy/beam
    psf:        np.ndarray,   # (H, W) peak = 1
    sigma:      float,        # noise RMS in Jy/beam
    config_idx: int,
    args,
    out_path:   Path,
    mask:       np.ndarray | None = None,
) -> None:
    """Extract the brightest 128x128 patch, run CoeffFlow, save 3-panel figure."""
    from mad_clean.models.coeff_flow import CoeffFlow
    from mad_clean.wavelet.starlet import StarletCodec

    device = torch.device(args.device)

    ckpt = torch.load(args.wavelet_ckpt, map_location=device, weights_only=False)
    codec = StarletCodec.from_state_dict(ckpt["codec"])
    flow = CoeffFlow(
        theta_dim=codec.theta_dim,
        base_channels=args.wavelet_base_channels,
        context_dim=args.wavelet_context_dim,
        hidden=args.wavelet_hidden,
        n_layers=args.wavelet_n_layers,
    ).to(device)
    flow.load_state_dict(ckpt["model"])
    flow.eval()

    # Patch centre: explicit override or brightest pixel within mask
    if args.patch_row is not None and args.patch_col is not None:
        peak_row, peak_col = args.patch_row, args.patch_col
        print(f"[probe] User-specified patch centre: row={peak_row} col={peak_col}  "
              f"val={residual[peak_row, peak_col]:.4e}  sigma={sigma:.4e}")
    else:
        search = np.where(mask, residual, 0.0) if mask is not None else residual.copy()
        peak_idx = int(np.argmax(search))
        peak_row, peak_col = divmod(peak_idx, residual.shape[1])
        print(f"[probe] Peak at row={peak_row} col={peak_col}  "
              f"val={residual[peak_row, peak_col]:.4e}  sigma={sigma:.4e}")

    res_cut = _safe_crop(residual, peak_row - 64, peak_row + 64,
                         peak_col - 64, peak_col + 64)
    psf_cut = _crop_psf_centred(psf, 128)

    # Normalise to training noise level (1e-4) so the model sees familiar amplitudes
    _TRAIN_SIGMA = 1e-4
    scale = _TRAIN_SIGMA / max(sigma, 1e-12)
    res_norm = res_cut * np.float32(scale)

    sig_t = torch.tensor([_TRAIN_SIGMA], dtype=torch.float32)
    cfg_t = torch.tensor([config_idx],   dtype=torch.long)
    cond  = make_cond(sig_t, cfg_t).to(device)                    # (1, 5)
    img_t = torch.from_numpy(
        np.stack([res_norm, psf_cut], axis=0)[None]               # (1, 2, 128, 128)
    ).to(device)

    with torch.no_grad():
        theta_s = flow.sample(img_t, cond, n=args.n_posterior)    # (1, n, D)
    sky_s = codec.decode(theta_s.squeeze(0).cpu())                 # (n, 128, 128)

    # Denormalise back to Jy/pixel, then rescale amplitude to dirty_peak * loop_gain
    sky_s_real   = sky_s / scale
    sky_med_real = sky_s_real.median(dim=0).values                # (128, 128)
    sky_draw     = sky_s_real[0]                                   # (128, 128)
    target_amp   = float(np.abs(res_cut).max()) * args.loop_gain
    amp_scale    = target_amp / float(sky_med_real.abs().max().clamp_min(1e-12))
    sky_med_real = sky_med_real * amp_scale
    sky_draw     = sky_draw     * amp_scale

    # 3-panel figure: dirty | posterior median | posterior draw
    fig, axes = plt.subplots(1, 3, figsize=(9, 3.2))
    panels = [
        (res_cut,                    "dirty patch (Jy/beam)"),
        (sky_med_real.numpy(),       "posterior median (Jy/pixel)"),
        (sky_draw.numpy(),           "posterior draw (Jy/pixel)"),
    ]
    for ax, (panel, title) in zip(axes, panels):
        m = float(np.abs(panel).max()) or 1e-12
        im = ax.imshow(panel, origin="lower", vmin=-m, vmax=m)
        ax.set_title(title, fontsize=9)
        ax.set_xticks([]); ax.set_yticks([])
        fig.colorbar(im, ax=ax, fraction=0.046)
    label = args.label or ""
    fig.suptitle(label, fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"[probe] Figure saved to {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv=None):
    args = parse_args(argv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    S = args.imsize

    # --- acquire dirty image + PSF (FITS bypass or tclean) ---
    if args.dirty_fits and args.psf_fits:
        print(f"[img] Loading dirty from {args.dirty_fits}")
        print(f"[img] Loading PSF  from {args.psf_fits}")
        residual = load_fits_image(args.dirty_fits)
        psf      = load_fits_image(args.psf_fits)
        S = residual.shape[0]
        mask = make_mask(S, args.mask_radius_px)
    else:
        if not _HAS_CASA:
            raise RuntimeError("casatasks not available; pass --dirty_fits and --psf_fits instead.")
        imgname = str(out_dir / "3c391_mdn_asp")
        mask = make_mask(S, args.mask_radius_px)
        casa_mask = args.mask if args.mask else f"circle[[{S//2}pix,{S//2}pix],{args.mask_radius_px}pix]"
        print(f"[img] CASA mask: {casa_mask}")
        print("[img] Initial tclean (niter=0, calcpsf=True) ...")
        tclean(
            vis=args.vis,
            imagename=imgname,
            field='', spw='',
            specmode='mfs',
            gridder=args.gridder,
            wprojplanes=args.wprojplanes,
            imsize=[S, S],
            cell=[args.cell, args.cell],
            stokes='I',
            weighting='briggs', robust=0.5,
            niter=0,
            mask=casa_mask,
            calcpsf=True,
            calcres=True,
            pbcor=False,
        )
        psf      = read_casa_image(imgname + ".psf")
        residual = read_casa_image(imgname + ".residual")

    # If probe mode, skip the imaging loop entirely
    if args.probe_wavelet_npe:
        if not args.wavelet_ckpt:
            raise ValueError("--wavelet_ckpt required with --probe_wavelet_npe")
        finite_mask = np.isfinite(residual)
        sigma_rms   = _measure_noise_rms(np.nan_to_num(residual), finite_mask & mask)
        probe_wavelet_npe(
            residual=np.nan_to_num(residual),
            psf=np.nan_to_num(psf),
            sigma=sigma_rms,
            config_idx=args.config_idx,
            args=args,
            out_path=out_dir / f"probe_{args.label or 'result'}.png",
            mask=finite_mask & mask,
        )
        return

    # --- MDN imaging loop (unchanged below) ---
    if not _HAS_CASA:
        raise RuntimeError("MDN imaging loop requires casatasks.")

    # --- load solver model ---
    if args.solver == "flow":
        from mad_clean.models.psf_condflow import PSFCondFlow
        from mad_clean.minor_cycle import minor_cycle_flow
        ck = torch.load(args.flow_ckpt, map_location="cpu", weights_only=True)
        cfg = ck.get("config", {})
        flow = PSFCondFlow(
            base=cfg.get("base_channels", args.flow_base_channels),
            asinh_a=cfg.get("asinh_a", args.flow_asinh_a),
        )
        flow.load_state_dict(ck["model"])
        flow.to(device).eval()
        mdn = None
        print(f"[img] Loaded PSFCondFlow checkpoint: {args.flow_ckpt} "
              f"(step {ck.get('step', '?')})")
    else:
        ckpt_mdn = torch.load(args.ckpt, map_location="cpu", weights_only=False)
        state = ckpt_mdn.get("model", ckpt_mdn.get("model_state_dict", ckpt_mdn))
        mdn = MDNAsp(base_channels=32, hidden=256, n_components=5, cond_dim=5)
        mdn.load_state_dict(state)
        mdn.to(device).eval()
        print(f"[img] Loaded MDN checkpoint: {args.ckpt}")

    psf_cut  = crop_psf_centred(psf, _PSF_CROP)

    sidelobe_level = measure_psf_sidelobe(psf_cut)
    dirty_peak     = float(np.where(mask, residual, 0.0).max())
    sigma_rms      = _measure_noise_rms(residual, mask)
    psf_floor      = dirty_peak * sidelobe_level
    noise_floor    = 3.0 * sigma_rms
    threshold      = max(psf_floor, noise_floor)

    print(f"[img] dirty_peak={dirty_peak:.4f} Jy/beam")
    print(f"[img] PSF sidelobe level={sidelobe_level:.3f}  PSF floor={psf_floor:.4f} Jy/beam")
    print(f"[img] Off-source RMS={sigma_rms:.4e} Jy/beam  3-sigma={noise_floor:.4e} Jy/beam")
    if args.global_threshold is not None:
        noise_floor = args.global_threshold
        threshold   = args.global_threshold
        print(f"[img] Global stop threshold={threshold:.4e} Jy/beam  (user override)")
    else:
        print(f"[img] Global stop threshold={threshold:.4e} Jy/beam  "
              f"(limited by {'noise' if noise_floor >= psf_floor else 'PSF sidelobe'})")

    log = {
        "dirty_peak": dirty_peak,
        "sidelobe_level": sidelobe_level,
        "sigma_rms": sigma_rms,
        "psf_floor": psf_floor,
        "noise_floor": noise_floor,
        "threshold": threshold,
        "major_cycles": [],
    }

    # --- major cycle loop ---
    for major in range(args.max_major):
        residual = read_casa_image(imgname + ".residual")
        res_peak = float(np.where(mask, residual, 0.0).max())
        # Global stopping criterion, RECOMPUTED each major cycle on the full-image
        # residual (across all tiles): as cleaning lowers the residual the noise
        # falls, so 3-sigma falls with it and each major cycle cleans deeper.
        # (The sidelobe x peak term is NOT used for stopping — noise only.)
        if args.global_threshold is not None:
            sigma_rms = args.global_threshold / 3.0
            threshold = args.global_threshold
        else:
            sigma_rms = _measure_noise_rms(residual, mask)
            threshold = 3.0 * sigma_rms
        print(f"\n[img] Major cycle {major}  residual_peak={res_peak:.4e}  "
              f"sigma={sigma_rms:.4e}  3-sigma_stop={threshold:.4e}")

        if res_peak < threshold:
            print("[img] Residual peak below 3-sigma — stopping.")
            break

        # --- minor cycle (peak-centred cutouts, not fixed tile grid) ---
        masked_residual = np.where(mask, residual, 0.0).astype(np.float32)
        if args.solver == "flow":
            model_update, commits, stop_reason = minor_cycle_flow(
                residual=masked_residual,
                psf=psf,
                sigma=sigma_rms,
                model=flow,
                threshold=threshold,          # this cycle's recomputed 3-sigma
                loop_gain=args.loop_gain,
                speckle_frac=args.speckle_frac,
                conf_k=args.conf_k,
                floor_frac=args.floor_frac,
                stride=args.tile_stride,
                inner_max=args.inner_max,
                max_components=args.max_minor,
                image_mask=mask,
                device=args.device,
            )
        else:
            model_update, commits, stop_reason = minor_cycle(
                residual=masked_residual,
                psf=psf,
                sigma=sigma_rms,
                config_idx=args.config_idx,
                model=mdn,
                loop_gain=args.loop_gain,
                sidelobe_level=sidelobe_level,
                divergence_tol=args.divergence_tol,
                device=args.device,
            )
        if not commits:
            print(f"[img] No components accepted (minor_cycle_stop={stop_reason}) — stopping.")
            break

        total_flux = sum(c.flux for c in commits)
        print(f"[img]   {len(commits)} components committed  total_flux_commit={total_flux:.4f} Jy  minor_cycle_stop={stop_reason}")

        # --- add model_update to existing CASA model image ---
        model_img = read_casa_image(imgname + ".model")
        model_total_before = float(model_img.sum())
        model_img += model_update
        model_total_after = float(model_img.sum())
        print(f"[img]   model total flux: {model_total_before:.4f} → {model_total_after:.4f} Jy")
        write_casa_image(imgname + ".model", model_img)

        # --- major cycle: recompute residual from updated model ---
        # Read back model to confirm it survived the write before tclean
        _check = read_casa_image(imgname + ".model")
        print(f"[img]   model sum before tclean: {_check.sum():.6f} Jy")

        tclean(
            vis=args.vis,
            imagename=imgname,
            field='', spw='',
            specmode='mfs',
            gridder=args.gridder,
            wprojplanes=args.wprojplanes,
            imsize=[S, S],
            cell=[args.cell, args.cell],
            stokes='I',
            weighting='briggs', robust=0.5,
            niter=0,
            mask=casa_mask,
            calcpsf=False,
            calcres=True,
            restart=True,
            pbcor=False,
        )

        _check2 = read_casa_image(imgname + ".model")
        print(f"[img]   model sum after tclean:  {_check2.sum():.6f} Jy")

        new_residual = read_casa_image(imgname + ".residual")
        new_peak = float(np.where(mask, new_residual, 0.0).max())

        log["major_cycles"].append({
            "major": major,
            "residual_peak_before": res_peak,
            "threshold": threshold,
            "n_components": len(commits),
            "residual_peak_after": new_peak,
        })
        with open(out_dir / "log.json", "w") as f:
            json.dump(log, f, indent=2)

        print(f"[img]   residual_peak after major cycle: {new_peak:.4e}")

        if new_peak > res_peak * (1.0 + args.divergence_tol):
            print(f"[img] Residual peak rose by >{args.divergence_tol*100:.0f}% ({res_peak:.4e} → {new_peak:.4e}) — stopping.")
            break

    print(f"\n[img] Done. Log at {out_dir}/log.json")
    print(f"[img] Model image: {imgname}.model")
    print(f"[img] Final residual: {imgname}.residual")


if __name__ == "__main__":
    main()
