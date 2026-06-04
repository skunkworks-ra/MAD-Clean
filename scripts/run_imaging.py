"""MDN-Asp imaging loop (general purpose).

Runs under the radiosharp pixi env which has casatasks/casatools:
  /home/pjaganna/Software/radiosharp/.pixi/envs/default/bin/python \
      scripts/run_imaging.py --vis <ms> --imagename <name> [options]

Loop
----
1. tclean niter=0 (major cycle) -> .residual + .psf images
2. minor_cycle(): find residual peak, crop 128x128 centred on peak,
   MDN forward pass, subtract PSF response, repeat until threshold
3. Accumulate model_update into the CASA .model image
4. tclean niter=0 (calcres=True) -> updated .residual
5. Repeat until residual_peak < threshold

Uncertainty image (optional)
-----------------------------
If --n_posterior_samples > 0, after the loop draw N samples from the MDN
mixture posterior for each committed Aspen, render each sample's flux std
as a Gaussian at the same position/shape, and take the pixel-wise maximum
across all components. Written as <imagename>.uncertainty (CASA image).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

# --- locate MAD-Clean repo ---
_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import torch
from casatasks import tclean
from casatools import image as iatool

from mad_clean.minor_cycle import minor_cycle, refit_pass, render_aspen
from mad_clean.models.mdn_asp import MDNAsp
from mad_clean.data.cutout_dataset import unstandardise_log_flux
from mad_clean.data.extended_sky import BEAM_SIGMA_PX, SIGMA_MAX_PX

_PSF_CROP = 128


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv=None):
    p = argparse.ArgumentParser(description="MDN-Asp major/minor cycle imaging loop.")

    # Required
    p.add_argument("--vis",       required=True, help="Input Measurement Set.")
    p.add_argument("--imagename", required=True, help="Output image name stem (no extension).")

    # Paths
    p.add_argument("--ckpt",    default="results/mdn_asp_v1/best.pt")
    p.add_argument("--out_dir", default="results/mdn_asp")

    # tclean imaging params
    p.add_argument("--imsize",    type=int,   default=512)
    p.add_argument("--cell",      default="2.5arcsec")
    p.add_argument("--field",     default="",    help="tclean field selection. Empty string = all fields; pass e.g. '2~8' or '3C391*' to restrict to science targets.")
    p.add_argument("--spw",       default="",    help="tclean spw selection.")
    p.add_argument("--gridder",      default="mosaic")
    p.add_argument("--wprojplanes",  type=int, default=-1,
                   help="Number of w-projection planes. -1 = auto (tclean default). Only used when --gridder=wproject.")
    p.add_argument("--specmode",  default="mfs")
    p.add_argument("--robust",    type=float, default=0.5)

    # Mask
    p.add_argument("--mask", type=str, default=None,
                   help="CASA mask image path. If not set, uses a circle of --mask_radius_px.")
    p.add_argument("--mask_radius_px", type=int, default=200,
                   help="Circular mask radius in pixels from image centre (used if --mask not set).")

    # Loop control
    p.add_argument("--loop_gain",        type=float, default=0.1)
    p.add_argument("--max_major",        type=int,   default=20)
    p.add_argument("--max_components",   type=int,   default=1000,
                   help="Max minor-cycle components per major cycle.")
    p.add_argument("--refit",            action="store_true",
                   help="Run one refit pass after each minor cycle.")
    p.add_argument("--sigma_max_px",     type=float, default=None,
                   help="Override MDN sigma_maj clip ceiling (px). Default: SIGMA_MAX_PX (~11.2px).")
    p.add_argument("--config_idx",       type=int,   default=2,
                   help="VLA config index passed to MDN conditioning: 0=A 1=B 2=C 3=D.")
    p.add_argument("--divergence_tol",   type=float, default=0.20,
                   help="Stop if residual peak rises by more than this fraction between major cycles.")
    p.add_argument("--global_threshold", type=float, default=None,
                   help="Override computed stopping threshold (Jy/beam).")

    # Posterior uncertainty
    p.add_argument("--n_posterior_samples", type=int, default=0,
                   help="Draw N samples from MDN mixture to build uncertainty image. 0 = off.")

    p.add_argument("--device", default="cpu")
    return p.parse_args(argv)


# ---------------------------------------------------------------------------
# CASA image helpers
# ---------------------------------------------------------------------------

def read_casa_image(imagepath: str) -> np.ndarray:
    """Read a CASA image -> 2D float32 (ny, nx) row-major."""
    ia = iatool()
    ia.open(imagepath)
    chunk = ia.getchunk()   # (nx, ny, npol, nchan) Fortran order
    ia.close()
    return chunk.squeeze().T.astype(np.float32)


def write_casa_image(imagepath: str, data: np.ndarray) -> None:
    """Write (ny, nx) float32 array into an existing CASA image."""
    ia = iatool()
    ia.open(imagepath)
    arr = data.T.astype(np.float32)[:, :, np.newaxis, np.newaxis]
    ia.putchunk(arr)
    ia.close()


def copy_and_write_casa_image(src: str, dst: str, data: np.ndarray) -> None:
    """Copy a CASA image (for coordinate system) then overwrite pixel data."""
    import shutil
    shutil.copytree(src, dst)
    write_casa_image(dst, data)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_mask(imsize: int, radius_px: int) -> np.ndarray:
    cy, cx = imsize // 2, imsize // 2
    Y, X = np.ogrid[:imsize, :imsize]
    return ((X - cx)**2 + (Y - cy)**2) <= radius_px**2


def measure_psf_sidelobe(psf: np.ndarray) -> float:
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


def measure_noise_rms(residual: np.ndarray, mask: np.ndarray,
                      n_sigma: float = 3.0, n_iter: int = 3) -> float:
    """Sigma-clipped MAD from within-beam pixels."""
    pixels = residual[mask].ravel()
    for _ in range(n_iter):
        rms = 1.4826 * float(np.median(np.abs(pixels)))
        if rms <= 0:
            break
        pixels = pixels[np.abs(pixels) < n_sigma * rms]
    return 1.4826 * float(np.median(np.abs(pixels))) if len(pixels) > 0 else 0.0


# ---------------------------------------------------------------------------
# Posterior uncertainty image
# ---------------------------------------------------------------------------

def build_uncertainty_image(all_commits: list, shape: tuple[int, int],
                             n_samples: int, device: torch.device) -> np.ndarray:
    """Pixel-wise max of per-Aspen flux std, rendered at each component's position.

    For each committed Aspen draw N samples from its MDN mixture posterior,
    compute the std of the resulting flux values, render that std as a Gaussian
    at the Aspen's position and shape, then take the pixel-wise maximum across
    all components. This gives a conservative per-pixel uncertainty map in Jy.
    """
    uncertainty = np.zeros(shape, dtype=np.float32)

    for commit in all_commits:
        mp = commit.params   # MixParams: logits (1,K), mu (1,K,6), log_std (1,K,6)
        logits  = mp.logits.squeeze(0)          # (K,)
        mu      = mp.mu.squeeze(0)              # (K, 6)
        log_std = mp.log_std.squeeze(0)         # (K, 6)

        # Sample component indices from the categorical
        probs = torch.softmax(logits, dim=0)
        idx = torch.multinomial(probs.unsqueeze(0).expand(n_samples, -1),
                                num_samples=1).squeeze(1)  # (N,)

        # Sample from selected Gaussian components
        mu_sel      = mu[idx]       # (N, 6)
        std_sel     = torch.exp(log_std[idx])  # (N, 6)
        samples     = mu_sel + std_sel * torch.randn_like(std_sel)  # (N, 6)

        # Decode flux dimension (index 2 is log_flux_std)
        log_flux_std_samples = samples[:, 2].numpy()
        log_flux_vals = np.array([unstandardise_log_flux(float(z)) for z in log_flux_std_samples])
        # Clip to training flux range [1e-4, 1e-1] Jy with generous margin
        log_flux_vals = np.clip(log_flux_vals, np.log(1e-6), np.log(1.0))
        flux_samples = np.exp(log_flux_vals)

        # Robust IQR-based std -- immune to posterior tail outliers
        p16, p84 = np.percentile(flux_samples, [16.0, 84.0])
        flux_std = float((p84 - p16) / 2.0)
        if flux_std <= 0 or not np.isfinite(flux_std):
            continue

        # Render the flux std as a Gaussian at this Aspen's position/shape
        component_uncertainty = render_aspen(
            commit.cx, commit.cy,
            flux_std,
            commit.sig_maj, commit.sig_min, commit.pa,
            shape,
        )
        uncertainty = np.maximum(uncertainty, component_uncertainty)

    return uncertainty


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv=None):
    args = parse_args(argv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    imgname = str(out_dir / args.imagename)
    S = args.imsize

    # --- load MDN ---
    ckpt  = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    state = ckpt.get("model", ckpt.get("model_state_dict", ckpt))
    mdn   = MDNAsp(base_channels=32, hidden=256, n_components=5, cond_dim=5)
    mdn.load_state_dict(state)
    mdn.to(device).eval()
    print(f"[img] Loaded checkpoint: {args.ckpt}")

    if args.mask:
        casa_mask = args.mask
        mask = read_casa_image(args.mask).astype(bool)
    else:
        casa_mask = ""
        mask = np.ones((S, S), dtype=bool)
    print(f"[img] CASA mask: {casa_mask if casa_mask else '(none)'}")

    # --- initial tclean: PSF + dirty residual ---
    print("[img] Initial tclean (niter=0, calcpsf=True) ...")
    tclean(
        vis=args.vis, imagename=imgname,
        field=args.field, spw=args.spw,
        specmode=args.specmode, gridder=args.gridder,
        wprojplanes=args.wprojplanes,
        imsize=[S, S], cell=[args.cell, args.cell],
        stokes='I', weighting='briggs', robust=args.robust,
        niter=0, mask=casa_mask,
        calcpsf=True, calcres=True, pbcor=False,
    )

    psf      = read_casa_image(imgname + ".psf")
    residual = read_casa_image(imgname + ".residual")
    psf_cut  = crop_psf_centred(psf, _PSF_CROP)

    sidelobe_level = measure_psf_sidelobe(psf_cut)
    dirty_peak     = float(np.where(mask, residual, 0.0).max())
    sigma_rms      = measure_noise_rms(residual, mask)
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
        "vis": args.vis, "imagename": args.imagename,
        "dirty_peak": dirty_peak, "sidelobe_level": sidelobe_level,
        "sigma_rms": sigma_rms, "psf_floor": psf_floor,
        "noise_floor": noise_floor, "threshold": threshold,
        "major_cycles": [],
    }

    all_commits = []   # accumulated across all major cycles for uncertainty image

    # --- major cycle loop ---
    for major in range(args.max_major):
        residual = read_casa_image(imgname + ".residual")
        # Re-estimate noise from the current residual so the minor-cycle floor
        # tracks the true noise level as cleaning progresses.
        sigma_rms = measure_noise_rms(residual, mask)
        res_peak = float(np.where(mask, residual, 0.0).max())
        threshold = max(res_peak * sidelobe_level, noise_floor)
        print(f"\n[img] Major cycle {major}  residual_peak={res_peak:.4e}  "
              f"sigma_rms={sigma_rms:.4e}  threshold={threshold:.4e}  "
              f"(psf_floor={res_peak*sidelobe_level:.4e}  noise_floor={noise_floor:.4e})")

        if res_peak < threshold:
            print("[img] Below threshold -- stopping.")
            break

        masked_residual = np.where(mask, residual, 0.0).astype(np.float32)
        model_update, commits, stop_reason = minor_cycle(
            residual=masked_residual, psf=psf,
            sigma=sigma_rms, config_idx=args.config_idx, model=mdn,
            loop_gain=args.loop_gain, sidelobe_level=sidelobe_level,
            abs_noise_floor=threshold,
            divergence_tol=args.divergence_tol, image_mask=mask,
            sigma_max_px=args.sigma_max_px,
            max_components=args.max_components,
            device=args.device,
        )
        if not commits:
            print(f"[img] No components accepted (minor_cycle_stop={stop_reason}) -- stopping.")
            break

        all_commits.extend(commits)
        total_flux = sum(c.flux for c in commits)
        print(f"[img]   {len(commits)} components committed  "
              f"total_flux_commit={total_flux:.4f} Jy  minor_cycle_stop={stop_reason}")

        model_img = read_casa_image(imgname + ".model")
        model_total_before = float(model_img.sum())
        model_img += model_update
        model_total_after = float(model_img.sum())
        print(f"[img]   model total flux: {model_total_before:.4f} -> {model_total_after:.4f} Jy")
        write_casa_image(imgname + ".model", model_img)

        _check = read_casa_image(imgname + ".model")
        print(f"[img]   model sum before tclean: {_check.sum():.6f} Jy")

        tclean(
            vis=args.vis, imagename=imgname,
            field=args.field, spw=args.spw,
            specmode=args.specmode, gridder=args.gridder,
        wprojplanes=args.wprojplanes,
            imsize=[S, S], cell=[args.cell, args.cell],
            stokes='I', weighting='briggs', robust=args.robust,
            niter=0, mask=casa_mask,
            calcpsf=False, calcres=True, restart=True, pbcor=False,
        )

        _check2 = read_casa_image(imgname + ".model")
        print(f"[img]   model sum after tclean:  {_check2.sum():.6f} Jy")

        new_residual = read_casa_image(imgname + ".residual")

        if args.refit and commits:
            model_img = read_casa_image(imgname + ".model")
            masked_new_residual = np.where(mask, new_residual, 0.0).astype(np.float32)
            model_img = refit_pass(
                commits=commits,
                post_residual=masked_new_residual,
                psf=psf,
                model_update=model_img,
                model=mdn,
                device=args.device,
            )
            write_casa_image(imgname + ".model", model_img)
            tclean(
                vis=args.vis, imagename=imgname,
                field=args.field, spw=args.spw,
                specmode=args.specmode, gridder=args.gridder,
        wprojplanes=args.wprojplanes,
                imsize=[S, S], cell=[args.cell, args.cell],
                stokes='I', weighting='briggs', robust=args.robust,
                niter=0, mask=casa_mask,
                calcpsf=False, calcres=True, restart=True, pbcor=False,
            )
            new_residual = read_casa_image(imgname + ".residual")
            print(f"  [refit] residual_peak after refit: "
                  f"{float(np.where(mask, new_residual, 0.0).max()):.4e}")

        new_peak = float(np.where(mask, new_residual, 0.0).max())

        log["major_cycles"].append({
            "major": major, "residual_peak_before": res_peak,
            "threshold": threshold, "n_components": len(commits),
            "residual_peak_after": new_peak,
        })
        with open(out_dir / "log.json", "w") as f:
            json.dump(log, f, indent=2)

        print(f"[img]   residual_peak after major cycle: {new_peak:.4e}")

        if new_peak > res_peak * (1.0 + args.divergence_tol):
            print(f"[img] Residual peak rose by >{args.divergence_tol*100:.0f}% "
                  f"({res_peak:.4e} -> {new_peak:.4e}) -- stopping.")
            break

    # --- posterior uncertainty image ---
    if args.n_posterior_samples > 0 and all_commits:
        print(f"\n[img] Building uncertainty image from {args.n_posterior_samples} posterior samples "
              f"over {len(all_commits)} components ...")
        uncertainty = build_uncertainty_image(
            all_commits, shape=(S, S),
            n_samples=args.n_posterior_samples,
            device=device,
        )
        unc_path = imgname + ".uncertainty"
        copy_and_write_casa_image(imgname + ".model", unc_path, uncertainty)
        print(f"[img] Uncertainty image: {unc_path}  "
              f"(peak={uncertainty.max():.4e}  median={np.median(uncertainty[mask]):.4e} Jy)")

    print(f"\n[img] Done. Log at {out_dir}/log.json")
    print(f"[img] Model image:    {imgname}.model")
    print(f"[img] Final residual: {imgname}.residual")


if __name__ == "__main__":
    main()
