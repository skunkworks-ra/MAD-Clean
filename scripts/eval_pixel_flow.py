"""Validation suite for pixel-space flow checkpoints (CoeffFlow, GPUSkyGenerator).

Matches training val exactly: GPUSkyGenerator + sky-weighted NLL.

    pixi run -e gpu python scripts/eval_pixel_flow.py \
        --checkpoint models/pixel_basis_best.pt \
        --psf_npy /mnt/Data/Data/corpus_stacks/train/psf.npy

This eval is the trusted instrument: all decisions are judged by the held-out
reconstruction here, never by the overfit cross-assignment gate or the NLL.

Tests:
  1. weighted_nll   — reported for continuity, NOT a decision metric (it is the
                      density ranking that stays green while recon is broken).
  2. posterior_grid — dirty | true sky | 3 draws | std for 8 scenes. PNG.
  3. recon_metrics  — rel-L2(median, truth) [headline], total flux ratio
                      (blob mode), peak ratio (collapse mode). Descriptive, no
                      threshold. Saves peak scatter + rel-L2 histogram.

Outputs to --out_dir (default results/eval_pixel_flow/).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import torch

from mad_clean.data.gpu_sky_generator import GPUSkyGenerator
from mad_clean.models.coeff_flow import CoeffFlow

IMAGE_SIZE = 128
THETA_DIM  = IMAGE_SIZE * IMAGE_SIZE


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Evaluate a pixel-space flow checkpoint.")
    p.add_argument("--checkpoint",       type=str, required=True)
    p.add_argument("--psf_npy",          type=str, required=True)
    p.add_argument("--out_dir",          type=str, default="results/eval_pixel_flow")
    p.add_argument("--n_scenes",         type=int, default=1000)
    p.add_argument("--n_grid",           type=int, default=8)
    p.add_argument("--n_posterior",      type=int, default=8)
    p.add_argument("--sky_weight_floor", type=float, default=0.1)
    p.add_argument("--batch_size",       type=int, default=256)
    p.add_argument("--device", type=str,
                   default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--base_channels", type=int, default=32)
    p.add_argument("--context_dim",   type=int, default=256)
    p.add_argument("--hidden",        type=int, default=128)
    p.add_argument("--n_layers",      type=int, default=8)
    return p.parse_args(argv)


def load_model(args, device):
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=True)
    flow = CoeffFlow(
        theta_dim=THETA_DIM,
        base_channels=args.base_channels,
        context_dim=args.context_dim,
        hidden=args.hidden,
        n_layers=args.n_layers,
    ).to(device)
    flow.load_state_dict(ckpt["model"])
    flow.eval()
    step    = ckpt.get("step", -1)
    val_nll = ckpt.get("val_nll_per_dim", float("nan"))
    print(f"[eval] step={step}  stored val_nll/dim={val_nll:.4f}")
    return flow, step


def make_theta_and_weights(sky, floor):
    sky_scale = sky.abs().flatten(1).max(dim=1).values.clamp_min(1e-12)
    theta = (sky / sky_scale.view(-1, 1, 1)).flatten(1)
    sky_w = floor + theta.clamp(min=0)
    sky_w = sky_w / sky_w.mean(dim=-1, keepdim=True).clamp_min(1e-12)
    return theta, sky_w, sky_scale


# ---------------------------------------------------------------------------
# Test 1: weighted NLL (matches training val exactly)
# ---------------------------------------------------------------------------

def weighted_nll(flow, gen, n_scenes, batch_size, floor, device):
    nlls = []
    remaining = n_scenes
    with torch.no_grad():
        while remaining > 0:
            b = min(batch_size, remaining)
            img, cond, sky = gen.sample(b)
            theta, sky_w, _ = make_theta_and_weights(sky, floor)
            nll = (-flow.log_prob(theta, img, cond, dim_weights=sky_w).mean()
                   / THETA_DIM).item()
            nlls.append(nll)
            remaining -= b
    result = float(np.mean(nlls))
    print(f"[eval] Weighted NLL/dim = {result:.4f}  (training best: stored in checkpoint)")
    return result


# ---------------------------------------------------------------------------
# Test 2: posterior sample grid
# ---------------------------------------------------------------------------

def posterior_grid(flow, gen, n_grid, n_draws, floor, device, path):
    with torch.no_grad():
        img, cond, sky = gen.sample(n_grid)
        _, _, sky_scale = make_theta_and_weights(sky, floor)

        all_draws = []
        for _ in range(n_draws):
            s = flow.sample(img, cond, n=1).squeeze(1)
            s_img = (s * sky_scale.view(-1, 1)).view(n_grid, IMAGE_SIZE, IMAGE_SIZE)
            all_draws.append(s_img.cpu())

    draws_t  = torch.stack(all_draws, dim=1)        # (N, n_draws, H, W)
    post_med = draws_t.median(dim=1).values
    post_std = draws_t.std(dim=1)
    img_cpu  = img.cpu()
    sky_cpu  = sky.cpu()

    # dirty | true sky | posterior median | posterior std
    import matplotlib.colors as mcolors
    import matplotlib.ticker as ticker
    n_cols = 4
    fig, axes = plt.subplots(n_grid, n_cols, figsize=(2.5 * n_cols, 2.5 * n_grid))
    axes = np.atleast_2d(axes)
    titles = ["dirty (input)", "true sky", f"post median ({n_draws} draws)", "post std"]
    for c, t in enumerate(titles):
        axes[0, c].set_title(t, fontsize=8)

    for b in range(n_grid):
        dirty_np = img_cpu[b, 0].numpy()
        sky_np   = sky_cpu[b].numpy()
        med_np   = post_med[b].numpy()
        std_np   = post_std[b].numpy()
        vmax = float(sky_cpu[b].max().clamp(min=1e-12))
        v = float(np.abs(dirty_np).max()) or 1.0
        panels = [
            (dirty_np, "RdBu_r",  mcolors.TwoSlopeNorm(vmin=-v, vcenter=0, vmax=v)),
            (sky_np,   "inferno", plt.Normalize(0, vmax)),
            (med_np,   "inferno", plt.Normalize(0, vmax)),
            (std_np,   "magma",   plt.Normalize(0, float(std_np.max()) or 1e-12)),
        ]
        for c, (panel, cmap, norm) in enumerate(panels):
            ax = axes[b, c]
            im = ax.imshow(panel, origin="lower", cmap=cmap, norm=norm)
            ax.axis("off")
            cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
            cb.ax.tick_params(labelsize=6)
            cb.ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.2g"))

    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"[eval] Posterior grid saved to {path}")


# ---------------------------------------------------------------------------
# Test 3: held-out reconstruction metrics (the trusted instrument)
#
# Three image-domain numbers on unseen scenes, each tied to a failure mode we
# have actually observed:
#   rel_l2      — ||median - truth|| / ||truth||. The headline "does the median
#                 track the sky" number. ~1.4 for a blank prediction; grows
#                 unbounded for blob over-prediction.
#   flux_ratio  — total median flux / total true flux. Catches the BLOB mode
#                 (saturated filled sources) that peak ratio is blind to.
#   peak_ratio  — median-draw peak / true peak. Catches the COLLAPSE-to-zero
#                 mode (ratio -> 0).
# Descriptive only — no pass/fail threshold. Read the distributions + the grid.
# ---------------------------------------------------------------------------

def recon_metrics(flow, gen, n_scenes, n_draws, floor, device, scatter_path,
                  hist_path):
    true_peaks, pred_peaks = [], []
    flux_ratios, rel_l2s = [], []
    remaining = n_scenes
    with torch.no_grad():
        while remaining > 0:
            b = min(64, remaining)
            img, cond, sky = gen.sample(b)
            _, _, sky_scale = make_theta_and_weights(sky, floor)
            sky_cpu = sky.cpu()

            draws = []
            for _ in range(n_draws):
                s = flow.sample(img, cond, n=1).squeeze(1)
                s_img = (s * sky_scale.view(-1, 1)).view(b, IMAGE_SIZE, IMAGE_SIZE)
                draws.append(s_img.cpu())
            draws_t  = torch.stack(draws, dim=1)            # (b, n_draws, H, W)
            med_img  = draws_t.median(dim=1).values         # (b, H, W)

            true_peaks.append(sky_cpu.flatten(1).max(dim=1).values.numpy())
            pred_peaks.append(
                draws_t.flatten(2).max(dim=2).values.numpy())   # (b, n_draws)

            true_flux = sky_cpu.flatten(1).sum(dim=1).clamp_min(1e-12)
            flux_ratios.append(
                (med_img.flatten(1).sum(dim=1) / true_flux).numpy())

            num = (med_img - sky_cpu).flatten(1).norm(dim=1)
            den = sky_cpu.flatten(1).norm(dim=1).clamp_min(1e-12)
            rel_l2s.append((num / den).numpy())
            remaining -= b

    true_peaks  = np.concatenate(true_peaks)
    pred_peaks  = np.concatenate(pred_peaks, axis=0)
    flux_ratios = np.concatenate(flux_ratios)
    rel_l2s     = np.concatenate(rel_l2s)
    median_pred = np.median(pred_peaks, axis=1)
    peak_ratio  = median_pred / np.clip(true_peaks, 1e-12, None)

    def _qtiles(x):
        return (float(np.percentile(x, 25)), float(np.median(x)),
                float(np.percentile(x, 75)))
    l2_q   = _qtiles(rel_l2s)
    flux_q = _qtiles(flux_ratios)
    peak_q = _qtiles(peak_ratio)
    print(f"[eval] rel_L2(median, truth):  median={l2_q[1]:.3f}  "
          f"IQR=[{l2_q[0]:.3f}, {l2_q[2]:.3f}]   (lower better; ~1.4 = blank)")
    print(f"[eval] total flux ratio:       median={flux_q[1]:.3f}  "
          f"IQR=[{flux_q[0]:.3f}, {flux_q[2]:.3f}]  (1.0 = exact; >>1 = blob)")
    print(f"[eval] peak flux ratio:        median={peak_q[1]:.3f}  "
          f"IQR=[{peak_q[0]:.3f}, {peak_q[2]:.3f}]  (1.0 = exact; ->0 = collapse)")

    # Peak scatter (unchanged), now alongside a rel-L2 histogram.
    fig, ax = plt.subplots(figsize=(5, 5))
    lim = max(true_peaks.max(), median_pred.max()) * 1.1
    ax.scatter(true_peaks, median_pred, s=10, alpha=0.6)
    ax.plot([0, lim], [0, lim], "k--", lw=1, label="1:1")
    ax.set_xlabel("True peak flux")
    ax.set_ylabel("Predicted peak (median draw)")
    ax.set_xlim(0, lim); ax.set_ylim(0, lim)
    ax.legend()
    fig.tight_layout(); fig.savefig(scatter_path, dpi=100); plt.close(fig)
    print(f"[eval] Peak flux scatter saved to {scatter_path}")

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.hist(np.clip(rel_l2s, 0, 3), bins=40, color="steelblue", alpha=0.85)
    ax.axvline(l2_q[1], color="k", ls="--", lw=1, label=f"median {l2_q[1]:.2f}")
    ax.axvline(1.0, color="grey", ls=":", lw=1, label="1.0")
    ax.set_xlabel("relative L2 (median vs truth)")
    ax.set_ylabel("held-out scenes")
    ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(hist_path, dpi=100); plt.close(fig)
    print(f"[eval] rel-L2 histogram saved to {hist_path}")

    return {
        "n_scenes": n_scenes, "n_draws": n_draws,
        "rel_l2_median": l2_q[1], "rel_l2_iqr": [l2_q[0], l2_q[2]],
        "flux_ratio_median": flux_q[1], "flux_ratio_iqr": [flux_q[0], flux_q[2]],
        "peak_ratio_median": peak_q[1], "peak_ratio_iqr": [peak_q[0], peak_q[2]],
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(args):
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    flow, step = load_model(args, device)

    torch.manual_seed(99999)   # held-out seed
    gen = GPUSkyGenerator(
        psf_npy=args.psf_npy, device=device,
        image_size=IMAGE_SIZE, sigma_noise=1e-4,
        n_sources=(1, 8), extended_fraction=0.5,
    )
    print(f"[eval] GPUSkyGenerator: {len(gen.psf_bank)} PSFs")

    # Reported for continuity, NOT a decision metric: this is the same density
    # ranking that stays green while the held-out reconstruction is broken.
    nll = weighted_nll(flow, gen, args.n_scenes, args.batch_size,
                       args.sky_weight_floor, device)

    posterior_grid(flow, gen, args.n_grid, n_draws=3,
                   floor=args.sky_weight_floor, device=device,
                   path=out_dir / "posterior_grid.png")

    recon = recon_metrics(flow, gen, args.n_scenes, args.n_posterior,
                          args.sky_weight_floor, device,
                          scatter_path=out_dir / "peak_flux_scatter.png",
                          hist_path=out_dir / "rel_l2_hist.png")

    summary = {"step": step, "weighted_nll_per_dim_untrusted": nll,
               "recon": recon}
    with open(out_dir / "eval_summary.json", "w") as fh:
        json.dump(summary, fh, indent=2)
    print(f"[eval] Done. Summary at {out_dir}/eval_summary.json")


if __name__ == "__main__":
    run(parse_args())
