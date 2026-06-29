"""Offline test of the data-support despeckle gate.

Question
--------
The flow model carries scattered ~1-beam "dots" that confidence (conf_k) and a
size gate cannot remove without also killing real point sources. The proposed
discriminator is DATA FIDELITY: keep a modelled pixel only where the input data
(the dirty image) actually supports emission there.

By definition of the major cycle:  dirty = residual + PSF (conv) model.
So we reconstruct the dirty exactly from the three images a finished run leaves,
matched-filter it (beam-smooth), threshold at k*sigma, and keep model pixels
only inside that support mask.

A hallucinated dot sits on flat noise -> no support -> removed.
A real point source is a PSF bump -> support -> kept.
The extended shell has support everywhere -> kept.

This is the SAME operation we would add to minor_cycle_flow (built there from the
per-tile residual `sub`); here we run it post-hoc on a whole field so we can look
before wiring it in.

Usage
-----
pixi run -e gpu python scripts/test_support_gate.py \
    --run_dir results/3c391_psf_condflow_v5 --k 3.0
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter, label
from scipy.signal import fftconvolve

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from casatools import image as iatool


def read_casa(path: str) -> np.ndarray:
    ia = iatool(); ia.open(path)
    arr = ia.getchunk().squeeze().T.astype(np.float32)
    ia.close()
    return arr


def write_casa_copy(template: str, out: str, data: np.ndarray) -> None:
    """Copy a CASA image dir and overwrite its pixels with `data` (ny,nx)."""
    if Path(out).exists():
        shutil.rmtree(out)
    shutil.copytree(template, out)
    ia = iatool(); ia.open(out)
    arr = data.T.astype(np.float32)[:, :, np.newaxis, np.newaxis]
    ia.putchunk(arr); ia.close()


def robust_sigma(x: np.ndarray, n_iter: int = 3, clip: float = 3.0) -> float:
    v = x[np.isfinite(x)]
    for _ in range(n_iter):
        s = 1.4826 * np.median(np.abs(v - np.median(v)))
        if s <= 0:
            break
        v = v[np.abs(v - np.median(v)) < clip * s]
    return 1.4826 * float(np.median(np.abs(v - np.median(v)))) if v.size else 0.0


def beam_sigma_px(image_path: str, cell_arcsec: float) -> float:
    ia = iatool(); ia.open(image_path)
    rb = ia.restoringbeam(); ia.close()
    fwhm_arcsec = 0.5 * (rb["major"]["value"] + rb["minor"]["value"])
    fwhm_px = fwhm_arcsec / cell_arcsec
    return fwhm_px / 2.3548


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run_dir", default="results/3c391_psf_condflow_v5")
    p.add_argument("--stem", default="3c391_mdn_asp")
    p.add_argument("--cell", type=float, default=2.5, help="arcsec/px")
    p.add_argument("--k", type=float, default=3.0, help="support threshold in sigma")
    args = p.parse_args()

    rd = Path(args.run_dir); stem = args.stem
    model    = read_casa(str(rd / f"{stem}.model"))
    residual = read_casa(str(rd / f"{stem}.residual"))
    psf      = read_casa(str(rd / f"{stem}.psf"))
    model    = np.nan_to_num(model)
    residual = np.nan_to_num(residual)

    # Exact dirty reconstruction: dirty = residual + PSF (conv) model
    dirty = residual + fftconvolve(model, psf, mode="same").astype(np.float32)

    # Matched filter ~ beam-smooth the dirty, then threshold
    sig_px = beam_sigma_px(str(rd / f"{stem}.image"), args.cell)
    support = gaussian_filter(dirty, sig_px).astype(np.float32)

    # Support noise from a SOURCE-FREE outer annulus (SNR is central), so the
    # estimate is not inflated by the emission we are trying to gate.
    H, W = support.shape
    cy, cx = H // 2, W // 2
    Y, X = np.ogrid[:H, :W]
    r = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2)
    pb = (residual != 0.0)                       # primary-beam footprint
    R = r[pb].max() if pb.any() else min(H, W) / 2
    annulus = pb & (r > 0.78 * R) & (r < 0.97 * R)
    sig_s = robust_sigma(support[annulus]) if annulus.any() else robust_sigma(support[pb])
    thr = args.k * sig_s
    mask = support > thr

    gated = np.where(mask, model, 0.0).astype(np.float32)

    # ---- metrics ----
    def n_blobs(m):
        lab, n = label(m > 0)
        return n

    flux_before = float(model.sum())
    flux_after  = float(gated.sum())
    removed_map = (model > 0) & (~mask)
    blobs_before = n_blobs(model)
    blobs_after  = n_blobs(gated)

    # size of removed blobs (are they small dots?)
    lab, n = label((model > 0) & removed_map)
    sizes = np.bincount(lab.ravel())[1:] if n else np.array([])
    # size of surviving blobs
    labk, nk = label(gated > 0)
    sizes_k = np.bincount(labk.ravel())[1:] if nk else np.array([])

    print(f"beam sigma = {sig_px:.2f} px   support sigma = {sig_s:.3e}   "
          f"k={args.k}  threshold = {thr:.3e}")
    print(f"model flux:   {flux_before:.4f} -> {flux_after:.4f} Jy  "
          f"({100*(flux_before-flux_after)/max(flux_before,1e-12):.2f}% removed)")
    print(f"connected blobs: {blobs_before} -> {blobs_after} "
          f"(removed {blobs_before-blobs_after})")
    if sizes.size:
        print(f"removed blob sizes (px): median={np.median(sizes):.0f}  "
              f"max={sizes.max():.0f}  n_removed_blobs={sizes.size}  "
              f"frac<=beam(31px)={100*np.mean(sizes<=31):.1f}%")
    if sizes_k.size:
        print(f"kept    blob sizes (px): median={np.median(sizes_k):.0f}  "
              f"max={sizes_k.max():.0f}  n_kept_blobs={sizes_k.size}  "
              f"frac<=beam(31px)={100*np.mean(sizes_k<=31):.1f}%")

    # ---- CASA image for CARTA ----
    out_casa = str(rd / f"{stem}.model_gated_k{args.k:g}")
    write_casa_copy(str(rd / f"{stem}.model"), out_casa, gated)
    print(f"gated model -> {out_casa}")

    # ---- diagnostic PNG (for verification) ----
    fig, ax = plt.subplots(1, 3, figsize=(13, 4.4))
    vmax = np.percentile(model[model > 0], 99.5) if (model > 0).any() else 1e-3
    for a, (img, t) in zip(ax, [(model, "model (before)"),
                                (gated, f"model gated k={args.k:g}"),
                                (np.where(removed_map, model, 0), "removed")]):
        im = a.imshow(img, origin="lower", vmin=0, vmax=vmax, cmap="inferno")
        a.set_title(t, fontsize=10); a.set_xticks([]); a.set_yticks([])
        fig.colorbar(im, ax=a, fraction=0.046)
    fig.tight_layout()
    png = rd / f"support_gate_k{args.k:g}.png"
    fig.savefig(png, dpi=120); plt.close(fig)
    print(f"diagnostic PNG -> {png}")


if __name__ == "__main__":
    main()
