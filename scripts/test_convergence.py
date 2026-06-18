"""Convergence test for the MDN-Asp minor cycle.

Generates a controlled synthetic field (3 point sources + 1 blob), makes a
dirty image with a real G55 PSF, runs minor_cycle with the trained checkpoint,
and plots residual RMS and peak per iteration.

Usage
-----
  pixi run -e gpu python scripts/test_convergence.py \\
      --ckpt results/mdn_asp_v1/best.pt \\
      --out_dir results/convergence_v1
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.signal import fftconvolve

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from mad_clean.data.extended_sky import BEAM_SIGMA_PX, render_gaussian_blob
from mad_clean.data.psf_bank import load_g55_psf_bank
from mad_clean.minor_cycle import minor_cycle, render_aspen
from mad_clean.models.mdn_asp import MDNAsp


def parse_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt",       type=str, default="results/mdn_asp_v1/best.pt")
    p.add_argument("--out_dir",    type=str, default="results/convergence_v1")
    p.add_argument("--repo_root",  type=str, default=".")
    p.add_argument("--field_size", type=int, default=256)
    p.add_argument("--sigma_noise", type=float, default=1e-4)
    p.add_argument("--loop_gain",  type=float, default=0.1)
    p.add_argument("--n_sigma_stop", type=float, default=3.0)
    p.add_argument("--sidelobe_level", type=float, default=0.2)
    p.add_argument("--max_components", type=int, default=200)
    p.add_argument("--n_major",        type=int, default=10)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--extended_only", action="store_true",
                   help="Use a single bright extended blob instead of points+blob.")
    p.add_argument("--device", type=str,
                   default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    S = args.field_size

    # ---- PSF ----
    psf_bank = load_g55_psf_bank(
        repo_root=args.repo_root, target_size=S, rotation_augment=False,
    )
    psf, _ = psf_bank.sample(rng)
    psf = psf.astype(np.float32)
    print(f"[conv] PSF shape={psf.shape}  peak={psf.max():.4f}")

    # ---- True sky ----
    sky = np.zeros((S, S), dtype=np.float32)
    true_sources = []
    margin = 80

    if args.extended_only:
        blob_img, blob_tgt = render_gaussian_blob(
            size=S,
            cx=float(rng.uniform(margin, S - margin)),
            cy=float(rng.uniform(margin, S - margin)),
            flux_jy=0.05,
            sig_maj=float(rng.uniform(5.0, 9.0)),
            sig_min=float(rng.uniform(2.0, 5.0)),
            pa=float(rng.uniform(0.0, math.pi)),
            rng=rng,
        )
        sky += blob_img
        true_sources.append({"kind": "blob", "cx": blob_tgt.x, "cy": blob_tgt.y,
                              "flux": math.exp(blob_tgt.log_flux),
                              "sig_maj": math.exp(blob_tgt.log_sig_maj),
                              "sig_min": math.exp(blob_tgt.log_sig_min),
                              "pa": blob_tgt.pa})
        print(f"[conv] blob   cx={blob_tgt.x:.1f} cy={blob_tgt.y:.1f} "
              f"flux={math.exp(blob_tgt.log_flux):.3f} "
              f"sig_maj={math.exp(blob_tgt.log_sig_maj):.2f} "
              f"sig_min={math.exp(blob_tgt.log_sig_min):.2f} "
              f"pa={blob_tgt.pa:.2f}")
    else:
        point_fluxes = [0.05, 0.03, 0.02]
        for flux in point_fluxes:
            cx = float(rng.uniform(margin, S - margin))
            cy = float(rng.uniform(margin, S - margin))
            r, c = int(round(cy)), int(round(cx))
            sky[r, c] += np.float32(flux)
            true_sources.append({"kind": "point", "cx": cx, "cy": cy, "flux": flux,
                                  "sig_maj": BEAM_SIGMA_PX, "sig_min": BEAM_SIGMA_PX, "pa": 0.0})
            print(f"[conv] point  cx={cx:.1f} cy={cy:.1f} flux={flux:.3f}")

        blob_img, blob_tgt = render_gaussian_blob(
            size=S,
            cx=float(rng.uniform(margin, S - margin)),
            cy=float(rng.uniform(margin, S - margin)),
            flux_jy=0.04,
            sig_maj=float(rng.uniform(3.0, 7.0)),
            sig_min=float(rng.uniform(BEAM_SIGMA_PX, 3.0)),
            pa=float(rng.uniform(0.0, math.pi)),
            rng=rng,
        )
        sky += blob_img
        true_sources.append({"kind": "blob", "cx": blob_tgt.x, "cy": blob_tgt.y,
                              "flux": math.exp(blob_tgt.log_flux),
                              "sig_maj": math.exp(blob_tgt.log_sig_maj),
                              "sig_min": math.exp(blob_tgt.log_sig_min),
                              "pa": blob_tgt.pa})
        print(f"[conv] blob   cx={blob_tgt.x:.1f} cy={blob_tgt.y:.1f} "
              f"flux={math.exp(blob_tgt.log_flux):.3f} "
              f"sig_maj={math.exp(blob_tgt.log_sig_maj):.2f} "
              f"sig_min={math.exp(blob_tgt.log_sig_min):.2f} "
              f"pa={blob_tgt.pa:.2f}")

    # ---- Dirty image ----
    dirty = fftconvolve(sky, psf, mode="same").astype(np.float32)
    noise = rng.normal(0.0, args.sigma_noise, dirty.shape).astype(np.float32)
    residual = dirty + noise
    print(f"[conv] dirty peak={dirty.max():.4f}  noise sigma={args.sigma_noise:.1e}")

    # ---- Load model ----
    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    state = ckpt.get("model", ckpt.get("model_state_dict", ckpt))
    mdn = MDNAsp(base_channels=32, hidden=256, n_components=5, cond_dim=5)
    mdn.load_state_dict(state)
    mdn.to(args.device).eval()
    print(f"[conv] Loaded checkpoint: {args.ckpt}")

    # ---- Major cycle loop ----
    # Each iteration: minor_cycle -> accumulate model -> PSF convolve model
    # -> subtract from original dirty to get new residual.
    # Run minor_cycle once; replay commits to get per-component RMS trajectory
    model_update, commits = minor_cycle(
        residual=residual.copy(),
        psf=psf,
        sigma=args.sigma_noise,
        config_idx=3,
        model=mdn,
        loop_gain=args.loop_gain,
        n_sigma_stop=args.n_sigma_stop,
        sidelobe_level=args.sidelobe_level,
        max_components=args.max_components,
        device=args.device,
    )
    print(f"[conv] minor_cycle done: {len(commits)} components committed.")

    if commits:
        c0 = commits[0]
        print(f"[conv] First commit: flux_full={c0.flux/args.loop_gain:.4f}  "
              f"sig_maj={c0.sig_maj:.3f}  sig_min={c0.sig_min:.3f}  pa={c0.pa:.3f}")

    rms_per_major   = [float(np.std(residual))]
    peak_per_major  = [float(np.abs(residual).max())]
    res_track = dirty + noise
    for commit in commits:
        aspen_img = render_aspen(
            commit.cx, commit.cy, commit.flux,
            commit.sig_maj, commit.sig_min, commit.pa, (S, S),
        )
        psf_resp = fftconvolve(aspen_img, psf, mode="same").astype(np.float32)
        res_track -= psf_resp
        rms_per_major.append(float(np.std(res_track)))
        peak_per_major.append(float(np.abs(res_track).max()))

    final_residual = res_track
    print(f"[conv] Initial RMS={rms_per_major[0]:.2e}  Final RMS={rms_per_major[-1]:.2e}")

    # ---- Save truth render for comparison ----
    truth_model = np.zeros((S, S), dtype=np.float32)
    for src in true_sources:
        truth_model += render_aspen(
            src["cx"], src["cy"], src["flux"],
            src["sig_maj"], src["sig_min"], src["pa"],
            (S, S),
        )

    # ---- Plots ----
    _plot_images(dirty, model_update, final_residual, truth_model, psf,
                 out_dir / "images.png")
    _plot_convergence(rms_per_major, peak_per_major, args.sigma_noise,
                      out_dir / "convergence.png", xlabel="committed components")

    print(f"[conv] Initial RMS={rms_per_major[0]:.2e}  "
          f"Final RMS={rms_per_major[-1]:.2e}  "
          f"Noise floor={args.sigma_noise:.2e}")
    print(f"[conv] Results in {out_dir}/")


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _plot_images(dirty, model, residual, truth, psf, fname):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    vmax = float(np.abs(dirty).max())
    fig, axes = plt.subplots(1, 5, figsize=(18, 4))
    titles = ["dirty", "truth model", "recovered model", "final residual", "PSF"]
    imgs   = [dirty, truth, model, residual, psf]
    for ax, title, im in zip(axes, titles, imgs):
        v = float(np.abs(im).max())
        ax.imshow(im, origin="lower", cmap="RdBu_r", vmin=-v, vmax=v)
        ax.set_title(title)
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(fname, dpi=150)
    plt.close(fig)


def _plot_convergence(rms, peak, noise_floor, fname, xlabel="major cycles"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    iters = np.arange(len(rms))
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.semilogy(iters, rms,  lw=1.5, label="residual RMS")
    ax.semilogy(iters, peak, lw=1.5, label="residual peak", ls="--")
    ax.axhline(noise_floor, color="gray", lw=1, ls=":", label=f"noise floor ({noise_floor:.0e})")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Jy/beam")
    ax.set_title("MDN-Asp convergence")
    ax.legend()
    fig.tight_layout()
    fig.savefig(fname, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
