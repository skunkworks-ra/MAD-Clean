"""Loss landscape exploration for the wavelet NPE residual loss.

Part 1 (existing): image panels showing residual under shift and outside-footprint
perturbations.

Part 2 (new): penalty mechanism ranking. For each of three regularisation
mechanisms at a sweep of lambda values, computes two ratios across the N
brightest patches:

    ratio_shift   = (L_residual + lambda * penalty) for shifted sky
                  / (L_residual + lambda * penalty) for truth sky

    ratio_outside = (L_residual + lambda * penalty) for outside-flux sky
                  / (L_residual + lambda * penalty) for truth sky

A good mechanism raises ratio_outside toward ratio_shift (closes the gap)
without collapsing ratio_shift (so position is still well-constrained).

Mechanisms tested:
  L1_outside  -- lambda * sky_pred[~footprint].abs().sum()
                 Requires the truth footprint mask; valid at training time.
  TV          -- lambda * total_variation(sky_pred)
                 No mask needed; penalises spatial roughness everywhere.
  PSF_weight  -- residual loss reweighted by 1 / (PSF_response + eps),
                 upweighting regions where the PSF has low sensitivity.

Outputs (in --out_dir):
  loss_shift.png        -- image panels, shift perturbation
  loss_outside.png      -- image panels, outside-footprint perturbation
  mechanism_shift.png   -- ratio_shift vs lambda, one line per mechanism
  mechanism_outside.png -- ratio_outside vs lambda, one line per mechanism

Usage:
    pixi run python scripts/loss_exploration_wavelet.py \
        --stacks_dir /mnt/Data/Data/corpus_stacks/train \
        --n_patches 6 \
        --shift_pixels 3 \
        --outside_eps 1e-5 \
        --out_dir results/loss_exploration
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import TwoSlopeNorm

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import torch


# ---------------------------------------------------------------------------
# Core helpers
# ---------------------------------------------------------------------------

def _fft_convolve(sky: torch.Tensor, psf: torch.Tensor) -> torch.Tensor:
    """Zero-padded FFT convolution. Both (H, W). Returns (H, W)."""
    H, W = sky.shape
    fH, fW = H + H - 1, W + W - 1
    sf = torch.fft.rfft2(sky.unsqueeze(0), s=(fH, fW))
    pf = torch.fft.rfft2(psf.unsqueeze(0), s=(fH, fW))
    out = torch.fft.irfft2(sf * pf, s=(fH, fW)).squeeze(0)
    y0, x0 = (fH - H) // 2, (fW - W) // 2
    return out[y0:y0 + H, x0:x0 + W]


def _residual_loss(dirty: torch.Tensor, sky: torch.Tensor,
                   psf: torch.Tensor, sigma: float) -> float:
    return float(((dirty - _fft_convolve(sky, psf)) ** 2).sum() / sigma ** 2)


def _source_footprint(sky: torch.Tensor, threshold: float = 0.01) -> torch.Tensor:
    return sky > threshold * sky.max().clamp_min(1e-12)


def _shift_sky(sky: torch.Tensor, dy: int, dx: int) -> torch.Tensor:
    return torch.roll(torch.roll(sky, dy, dims=0), dx, dims=1)


def _psf_response(psf: torch.Tensor, sky_shape: tuple[int, int]) -> torch.Tensor:
    """Sum of PSF sensitivity at each sky pixel (column sum of PSF matrix).

    Approximated as the PSF convolved with an all-ones image -- gives the
    total PSF weight landing on each pixel when any pixel emits.
    """
    ones = torch.ones(sky_shape)
    return _fft_convolve(ones, psf).clamp_min(0)


# ---------------------------------------------------------------------------
# Penalty functions
# ---------------------------------------------------------------------------

def _penalty_l1_outside(sky: torch.Tensor,
                         footprint: torch.Tensor) -> float:
    return float(sky[~footprint].abs().sum())


def _penalty_tv(sky: torch.Tensor) -> float:
    dy = (sky[1:, :] - sky[:-1, :]).abs().sum()
    dx = (sky[:, 1:] - sky[:, :-1]).abs().sum()
    return float(dy + dx)


def _loss_psf_weighted(dirty: torch.Tensor, sky: torch.Tensor,
                        psf: torch.Tensor, sigma: float,
                        eps: float = 1e-3) -> float:
    """Residual loss reweighted by 1 / (PSF_response + eps).

    Upweights pixels where the PSF has low sensitivity so outside-footprint
    flux that hides cheaply in the standard loss becomes expensive here.
    """
    resp = _psf_response(psf, sky.shape)
    weight = 1.0 / (resp / resp.max().clamp_min(1e-12) + eps)
    resid = (dirty - _fft_convolve(sky, psf)) ** 2
    return float((resid * weight).sum() / sigma ** 2)


# ---------------------------------------------------------------------------
# Total loss for a (mechanism, lambda) pair
# ---------------------------------------------------------------------------

def _total_loss(dirty, sky, psf, sigma, footprint,
                mechanism: str, lam: float) -> float:
    base = _residual_loss(dirty, sky, psf, sigma)
    if mechanism == "none":
        return base
    if mechanism == "L1_outside":
        return base + lam * _penalty_l1_outside(sky, footprint)
    if mechanism == "TV":
        return base + lam * _penalty_tv(sky)
    if mechanism == "PSF_weight":
        # Replace base residual entirely with the weighted version
        return lam * _loss_psf_weighted(dirty, sky, psf, sigma) + (1 - lam) * base
    raise ValueError(mechanism)


# ---------------------------------------------------------------------------
# Colour helpers
# ---------------------------------------------------------------------------

def _sym_norm(data: np.ndarray) -> TwoSlopeNorm:
    vmax = float(np.abs(data).max()) or 1.0
    return TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)


def _pos_norm(data: np.ndarray) -> plt.Normalize:
    vmax = float(np.abs(data).max()) or 1.0
    return plt.Normalize(vmin=0, vmax=vmax)


def _colorbar(fig, im, ax):
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    cb.ax.tick_params(labelsize=6)
    cb.ax.yaxis.set_major_formatter(
        matplotlib.ticker.FormatStrFormatter("%.2g"))
    return cb


# ---------------------------------------------------------------------------
# Part 1: image panels
# ---------------------------------------------------------------------------

def _image_panel_figure(top_idx, dirty_all, sky_all, psf_all, field_id,
                         sky_flux, sigmas, pert_key, pert_label,
                         shift_pixels, outside_eps, out_dir):
    ncols = 6
    n = len(top_idx)
    fig = plt.figure(figsize=(ncols * 2.8, n * 2.8))
    gs  = gridspec.GridSpec(n, ncols, figure=fig, hspace=0.45, wspace=0.38)

    col_titles = ["dirty", "sky truth", f"sky {pert_label}",
                  "perturbation", "resid (truth)", f"resid ({pert_label})"]

    for row, idx in enumerate(top_idx):
        dirty = torch.from_numpy(np.array(dirty_all[idx], dtype=np.float32))
        sky   = torch.from_numpy(np.array(sky_all[idx],   dtype=np.float32))
        psf   = torch.from_numpy(np.array(psf_all[int(field_id[idx])], dtype=np.float32))
        sigma = sigmas[row]
        fp    = _source_footprint(sky)

        if pert_key == "shift":
            sky_p = _shift_sky(sky, shift_pixels, shift_pixels)
        else:
            noise = torch.zeros_like(sky)
            noise[~fp] = outside_eps
            sky_p = sky + noise

        l_truth = _residual_loss(dirty, sky,   psf, sigma)
        l_pert  = _residual_loss(dirty, sky_p, psf, sigma)

        res_t = (dirty - _fft_convolve(sky,   psf)).numpy()
        res_p = (dirty - _fft_convolve(sky_p, psf)).numpy()

        panels = [
            (dirty.numpy(),          "RdBu_r",  _sym_norm(dirty.numpy())),
            (sky.numpy(),            "inferno", _pos_norm(sky.numpy())),
            (sky_p.numpy(),          "inferno", _pos_norm(sky.numpy())),
            ((sky_p - sky).numpy(),  "RdBu_r",  _sym_norm((sky_p - sky).numpy())),
            (res_t,                  "RdBu_r",  _sym_norm(res_t)),
            (res_p,                  "RdBu_r",  _sym_norm(res_t)),
        ]

        subtitles = col_titles.copy()
        subtitles[4] = f"resid truth  L={l_truth:.0f}"
        subtitles[5] = f"resid {pert_label}  L={l_pert:.0f}  ×{l_pert/max(l_truth,1e-9):.2f}"

        for c, (data, cmap, norm) in enumerate(panels):
            ax = fig.add_subplot(gs[row, c])
            im = ax.imshow(data, origin="lower", cmap=cmap, norm=norm,
                           interpolation="nearest")
            if row == 0:
                ax.set_title(subtitles[c], fontsize=7, pad=3)
            else:
                if c >= 4:
                    ax.set_title(subtitles[c], fontsize=7, pad=3)
            ax.axis("off")
            _colorbar(fig, im, ax)

        print(f"  [{pert_key}] patch {idx:4d}  L_truth={l_truth:8.0f}  "
              f"L_pert={l_pert:8.0f}  ratio={l_pert/max(l_truth,1e-9):.3f}")

    fig.suptitle(
        f"Residual loss: truth vs {pert_label}  "
        f"(×ratio on last column = how much more the perturbation costs)",
        fontsize=9,
    )
    out_path = out_dir / f"loss_{pert_key}.png"
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


# ---------------------------------------------------------------------------
# Part 2: mechanism ranking
# ---------------------------------------------------------------------------

MECHANISMS = ["L1_outside", "TV", "PSF_weight"]
MECHANISM_LABELS = {
    "L1_outside": "L1 outside footprint",
    "TV":         "Total variation",
    "PSF_weight": "PSF-weighted residual",
}
COLORS = ["#e41a1c", "#377eb8", "#4daf4a"]
MARKERS = ["o", "s", "^"]

# Lambda sweeps are per-mechanism because their natural scales differ
LAMBDA_SWEEPS = {
    "L1_outside": [0, 1e3, 1e4, 1e5, 1e6, 1e7],
    "TV":         [0, 1e3, 1e4, 1e5, 1e6, 1e7],
    "PSF_weight": [0, 0.1, 0.3, 0.5, 0.7, 1.0],
}


def _mechanism_ranking_figure(top_idx, dirty_all, sky_all, psf_all, field_id,
                                sigmas, shift_pixels, outside_eps, out_dir):
    """Two plots: ratio_shift and ratio_outside vs lambda, per mechanism."""

    # Collect patch data
    patches = []
    for idx, sigma in zip(top_idx, sigmas):
        dirty = torch.from_numpy(np.array(dirty_all[idx], dtype=np.float32))
        sky   = torch.from_numpy(np.array(sky_all[idx],   dtype=np.float32))
        psf   = torch.from_numpy(np.array(psf_all[int(field_id[idx])], dtype=np.float32))
        fp    = _source_footprint(sky)
        sky_shift   = _shift_sky(sky, shift_pixels, shift_pixels)
        sky_outside = sky.clone()
        sky_outside[~fp] += outside_eps
        patches.append((dirty, sky, psf, fp, sigma, sky_shift, sky_outside))

    for pert_key, pert_label, sky_getter in [
        ("shift",   f"shift {shift_pixels}px",        lambda p: p[5]),
        ("outside", f"outside eps={outside_eps:.0e}", lambda p: p[6]),
    ]:
        fig, ax = plt.subplots(figsize=(8, 5))

        for mech, color, marker in zip(MECHANISMS, COLORS, MARKERS):
            lambdas = LAMBDA_SWEEPS[mech]
            median_ratios = []
            all_ratios = []  # (n_lambda, n_patches)

            for lam in lambdas:
                ratios = []
                for p in patches:
                    dirty, sky, psf, fp, sigma, _, _ = p
                    sky_p = sky_getter(p)
                    l_truth = _total_loss(dirty, sky,   psf, sigma, fp, mech, lam)
                    l_pert  = _total_loss(dirty, sky_p, psf, sigma, fp, mech, lam)
                    ratios.append(l_pert / max(l_truth, 1e-12))
                median_ratios.append(float(np.median(ratios)))
                all_ratios.append(ratios)

            lam_labels = [f"{l:.0e}" if l > 0 else "0" for l in lambdas]
            x = np.arange(len(lambdas))

            # Median line
            ax.plot(x, median_ratios, color=color, marker=marker,
                    label=MECHANISM_LABELS[mech], linewidth=2, markersize=7)

            # Per-patch scatter (faint)
            for patch_ratios in zip(*all_ratios):
                ax.plot(x, list(patch_ratios), color=color,
                        alpha=0.18, linewidth=0.8, marker=marker,
                        markersize=3)

        ax.axhline(1.0, color="black", linewidth=0.8, linestyle="--",
                   label="ratio = 1 (no extra cost)")
        ax.set_xticks(x)
        ax.set_xticklabels(lam_labels, fontsize=8)
        ax.set_xlabel("lambda", fontsize=10)
        ax.set_ylabel(f"(L_truth + penalty) ratio:  {pert_label} / truth", fontsize=9)
        ax.set_title(
            f"Mechanism ranking — {pert_label}\n"
            f"Higher ratio = perturbation costs more = better constraint\n"
            f"Median across {len(patches)} patches; faint lines = individual patches",
            fontsize=9,
        )
        ax.legend(fontsize=8, loc="upper left")
        ax.grid(True, alpha=0.3)

        out_path = out_dir / f"mechanism_{pert_key}.png"
        fig.tight_layout()
        fig.savefig(out_path, dpi=140, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved {out_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run(args):
    import matplotlib.ticker
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    dirty_all = np.load(Path(args.stacks_dir) / "dirty.npy",    mmap_mode="r")
    sky_all   = np.load(Path(args.stacks_dir) / "sky.npy",      mmap_mode="r")
    psf_all   = np.load(Path(args.stacks_dir) / "psf.npy",      mmap_mode="r")
    field_id  = np.load(Path(args.stacks_dir) / "field_id.npy", mmap_mode="r")

    sky_flux = sky_all.sum(axis=(1, 2))
    top_idx  = np.argsort(sky_flux)[::-1][:args.n_patches]
    print(f"Selected patch indices: {top_idx.tolist()}")
    print(f"Sky fluxes: {sky_flux[top_idx].tolist()}")

    sigmas = []
    for i in top_idx:
        d = dirty_all[i]
        sigmas.append(1.4826 * float(np.median(np.abs(d))))
    print(f"Per-patch sigma: {[f'{s:.2e}' for s in sigmas]}")

    # Part 1: image panels
    for pert_key, pert_label in [
        ("shift",   f"shift {args.shift_pixels}px"),
        ("outside", f"outside eps={args.outside_eps:.0e}"),
    ]:
        _image_panel_figure(
            top_idx, dirty_all, sky_all, psf_all, field_id,
            sky_flux, sigmas, pert_key, pert_label,
            args.shift_pixels, args.outside_eps, out_dir,
        )

    # Part 2: mechanism ranking
    print("\nComputing mechanism ranking...")
    _mechanism_ranking_figure(
        top_idx, dirty_all, sky_all, psf_all, field_id,
        sigmas, args.shift_pixels, args.outside_eps, out_dir,
    )


def parse_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--stacks_dir",   required=True)
    p.add_argument("--n_patches",    type=int,   default=6)
    p.add_argument("--shift_pixels", type=int,   default=3)
    p.add_argument("--outside_eps",  type=float, default=1e-5)
    p.add_argument("--out_dir",      default="results/loss_exploration")
    return p.parse_args(argv)


if __name__ == "__main__":
    run(parse_args())
