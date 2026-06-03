"""Coverage diagnostic for the trained MDN-Asp checkpoint.

Generates a fresh held-out set (seed_offset=20_000_000, never seen during
training or validation), runs MDN forward passes, and produces:

  results/<out_dir>/
    coverage.json        -- empirical 68 % / 95 % coverage per dimension
    coverage_bar.png     -- bar chart of coverage fractions vs ideal
    marginals_point.png  -- posterior marginals for a bright isolated point
    marginals_blob.png   -- posterior marginals for an isolated blob
    marginals_faint.png  -- posterior marginals for a faint near-threshold point

Usage
-----
  pixi run -e gpu python scripts/eval_coverage.py \\
      --ckpt results/mdn_asp_v1/best.pt \\
      --out_dir results/coverage_v1
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from mad_clean.data.cutout_dataset import CutoutDataset
from mad_clean.data.psf_bank import load_g55_psf_bank
from mad_clean.models.mdn_asp import MDNAsp, N_EMITTED, encode_pa


# Dimension names in 6D target space
DIM_NAMES = ["x_off", "y_off", "log_flux", "log_sig_maj", "log_sig_min", "PA"]

# Seed offset guaranteed never seen in training (0) or validation (10_000_000)
EVAL_SEED_OFFSET = 20_000_000


def parse_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt",       type=str, default="results/mdn_asp_v1/best.pt")
    p.add_argument("--out_dir",    type=str, default="results/coverage_v1")
    p.add_argument("--repo_root",  type=str, default=".")
    p.add_argument("--n_samples",  type=int, default=2000,
                   help="Number of held-out cutouts (default 2000).")
    p.add_argument("--n_posterior_draws", type=int, default=500,
                   help="MDN posterior samples per cutout for credible intervals.")
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--device", type=str,
                   default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--morphologies", type=str, default="point,blob")
    p.add_argument("--extended_fraction", type=float, default=0.05)
    p.add_argument("--snr_min", type=float, default=5.0)
    return p.parse_args(argv)


# ---------------------------------------------------------------------------
# Marginal CDF of a 1-D diagonal Gaussian mixture at a scalar value
# ---------------------------------------------------------------------------

def mixture_marginal_cdf(
    value: float,
    log_w: np.ndarray,   # (K,) log mixture weights
    mu:    np.ndarray,   # (K,) component means for this dim
    std:   np.ndarray,   # (K,) component stds  for this dim
) -> float:
    """sum_k w_k * Phi((value - mu_k) / sigma_k)"""
    from scipy.special import ndtr  # standard normal CDF, numerically stable
    w = np.exp(log_w - np.max(log_w))
    w /= w.sum()
    z = (value - mu) / np.clip(std, 1e-12, None)
    return float(np.dot(w, ndtr(z)))


# ---------------------------------------------------------------------------
# PA marginal coverage -- work in (sin2pa, cos2pa) space, use sample-based
# intervals on the decoded scalar PA.
# ---------------------------------------------------------------------------

def pa_credible_coverage(
    truth_pa: float,
    samples_pa: np.ndarray,  # (S,) posterior PA samples
    level: float,
) -> bool:
    """Check whether truth_pa falls within the central `level` HPD interval
    of samples_pa, handling the pi-periodicity by searching in (-pi/2, pi/2]."""
    lo = np.quantile(samples_pa, (1 - level) / 2)
    hi = np.quantile(samples_pa, 1 - (1 - level) / 2)
    return bool(lo <= truth_pa <= hi)


# ---------------------------------------------------------------------------
# Build a MixParams-like object from raw tensors (CPU numpy)
# ---------------------------------------------------------------------------

def forward_batch(model, img_t, cond_t, device):
    img_t  = img_t.to(device)
    cond_t = cond_t.to(device)
    with torch.no_grad():
        params = model(img_t, cond_t)
    return (
        params.logits.cpu(),
        params.mu.cpu(),
        params.log_std.cpu(),
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv=None):
    args = parse_args(argv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    # ---- load checkpoint ----
    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    # Checkpoint format: {"step": ..., "model": state_dict, "val_loss": ...}
    model_state = ckpt.get("model", ckpt.get("model_state_dict", ckpt))
    model = MDNAsp(base_channels=32, hidden=256, n_components=5, cond_dim=5)
    model.load_state_dict(model_state)
    model.to(device).eval()
    print(f"[eval] Loaded checkpoint: {args.ckpt}")

    # ---- held-out dataset ----
    psf_bank = load_g55_psf_bank(
        repo_root=args.repo_root, target_size=128, rotation_augment=True,
    )
    morph_keys = [m.strip() for m in args.morphologies.split(",")]
    morph_balance = {m: 1.0 for m in morph_keys}
    ds = CutoutDataset(
        psf_bank=psf_bank,
        field_size=512,
        cutout_size=128,
        sigma_noise=1e-4,
        n_sources_per_field=(5, 20),
        extended_fraction=args.extended_fraction,
        rng_seed=EVAL_SEED_OFFSET,
        length=args.n_samples,
        morphology_balance=morph_balance,
        snr_min=args.snr_min,
    )

    # ---- collect all forward passes ----
    all_logits, all_mu, all_log_std, all_targets = [], [], [], []
    all_images = []

    loader = torch.utils.data.DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        collate_fn=_collate,
    )

    for img_t, cond_t, tgt_t in loader:
        logits, mu, log_std = forward_batch(model, img_t, cond_t, device)
        all_logits.append(logits)
        all_mu.append(mu)
        all_log_std.append(log_std)
        all_targets.append(tgt_t)
        all_images.append(img_t)

    logits   = torch.cat(all_logits,   dim=0).numpy()   # (N, K)
    mu       = torch.cat(all_mu,       dim=0).numpy()   # (N, K, 7)
    log_std  = torch.cat(all_log_std,  dim=0).numpy()   # (N, K, 7)
    targets  = torch.cat(all_targets,  dim=0).numpy()   # (N, 6)
    images   = torch.cat(all_images,   dim=0)            # (N, 2, 128, 128)

    # Wrap target PA into (-pi/2, pi/2]: PA and PA+pi describe the same ellipse.
    # The decoder always outputs in this range; the generator uses [0, pi].
    targets[:, 5] = (targets[:, 5] + np.pi / 2) % np.pi - np.pi / 2

    N = len(targets)
    K = logits.shape[1]
    print(f"[eval] Collected {N} held-out samples, K={K} components.")

    log_w = logits - np.log(np.sum(np.exp(logits - logits.max(axis=1, keepdims=True)),
                                   axis=1, keepdims=True))  # (N, K) log-softmax

    std = np.exp(log_std)  # (N, K, 7)

    # ---- coverage: dims 0-4 via marginal CDF; dim 5 (PA) via samples ----
    # Mapping from 6D target dim index to 7D emitted dim index:
    #   0:x, 1:y, 2:log_flux, 3:log_sig_maj, 4:log_sig_min -> same index 0-4
    #   5:PA -> emitted dims 5 (sin2pa) and 6 (cos2pa) — handled separately

    in68  = np.zeros((N, 6), dtype=bool)
    in95  = np.zeros((N, 6), dtype=bool)

    # Draw PA samples once for all cutouts using vectorised ops
    # (B, K) probs -> (B, S) component indices -> (B, S) PA samples
    probs_np = np.exp(log_w)  # (N, K) — already log-softmax, so exp is fine
    probs_np = probs_np / probs_np.sum(axis=1, keepdims=True)  # renorm for float precision
    S = args.n_posterior_draws
    comp_idx = np.array([
        np.random.choice(K, size=S, replace=True, p=probs_np[i])
        for i in range(N)
    ])  # (N, S)

    # PA samples: pick sin2pa/cos2pa means + noise from chosen component
    # sin2pa = dim 5, cos2pa = dim 6 in emitted space
    for d_6 in range(5):  # dims 0-4 map identically to emitted dims 0-4
        for i in range(N):
            cdf = mixture_marginal_cdf(
                targets[i, d_6],
                log_w[i],
                mu[i, :, d_6],
                std[i, :, d_6],
            )
            in68[i, d_6] = 0.16 <= cdf <= 0.84
            in95[i, d_6] = 0.025 <= cdf <= 0.975

    # PA (dim 5): sample-based
    pa_samples = np.zeros((N, S), dtype=np.float32)
    for i in range(N):
        k_idx = comp_idx[i]                          # (S,)
        mu_sin  = mu[i, k_idx, 5]                   # (S,)
        mu_cos  = mu[i, k_idx, 6]
        std_sin = std[i, k_idx, 5]
        std_cos = std[i, k_idx, 6]
        s2pa = mu_sin + std_sin * np.random.randn(S)
        c2pa = mu_cos + std_cos * np.random.randn(S)
        pa_samples[i] = np.arctan2(s2pa, c2pa) * 0.5  # decode to (-pi/2, pi/2]

    for i in range(N):
        in68[i, 5] = pa_credible_coverage(targets[i, 5], pa_samples[i], 0.68)
        in95[i, 5] = pa_credible_coverage(targets[i, 5], pa_samples[i], 0.95)

    # ---- recover morphology labels by replaying the dataset RNG ----
    morph_keys  = [m.strip() for m in args.morphologies.split(",")]
    morph_probs = np.array([1.0 / len(morph_keys)] * len(morph_keys))
    kinds = np.array([
        morph_keys[int(np.random.default_rng(EVAL_SEED_OFFSET + i).choice(
            len(morph_keys), p=morph_probs
        ))]
        for i in range(N)
    ])

    # ---- aggregate and per-morphology coverage ----
    def _coverage_dict(mask):
        c68 = in68[mask].mean(axis=0)
        c95 = in95[mask].mean(axis=0)
        return {name: {"cov68": float(c68[j]), "cov95": float(c95[j])}
                for j, name in enumerate(DIM_NAMES)}

    result = {"all": _coverage_dict(np.ones(N, dtype=bool))}
    for k in morph_keys:
        result[k] = _coverage_dict(kinds == k)

    def _print_coverage(label, d):
        print(f"\n  [{label}]")
        print(f"  {'dim':<15}  {'68%':>6}  {'95%':>6}")
        for name, v in d.items():
            print(f"  {name:<15}  {v['cov68']:>6.1%}  {v['cov95']:>6.1%}")

    print("\n[eval] Coverage (ideal: 68% / 95%)")
    for label, d in result.items():
        n_label = int((kinds == label).sum()) if label != "all" else N
        _print_coverage(f"{label}  n={n_label}", d)

    with open(out_dir / "coverage.json", "w") as f:
        json.dump(result, f, indent=2)

    # ---- coverage bar chart (all + per-morphology) ----
    _plot_coverage(result, out_dir / "coverage_bar.png")

    # ---- spot-check marginals: pick 3 representative cutouts ----
    # Bright isolated point: high standardised log_flux, small log_sig_maj
    log_sig_maj = targets[:, 3]
    log_flux_std = targets[:, 2]
    beam_floor = np.log(1.4)  # BEAM_SIGMA_PX = 1.4 (extended_sky.py)

    # Bright point: high flux, scale near beam floor
    is_point = log_sig_maj < (beam_floor + 0.05)
    bright_point_idx = int(
        np.where(is_point)[0][np.argmax(log_flux_std[is_point])]
    ) if is_point.any() else 0

    # Blob: scale clearly above beam floor
    is_blob = log_sig_maj > (beam_floor + 0.5)
    blob_idx = int(
        np.where(is_blob)[0][np.argmax(log_flux_std[is_blob])]
    ) if is_blob.any() else 1

    # Faint point: point source, lowest flux
    faint_point_idx = int(
        np.where(is_point)[0][np.argmin(log_flux_std[is_point])]
    ) if is_point.any() else 2

    spot_cases = [
        ("point_bright", bright_point_idx),
        ("blob",         blob_idx),
        ("point_faint",  faint_point_idx),
    ]

    for label, idx in spot_cases:
        fname = out_dir / f"marginals_{label}.png"
        _plot_marginals(
            idx=idx,
            targets=targets,
            log_w=log_w[idx],
            mu=mu[idx],
            std=std[idx],
            pa_samples=pa_samples[idx],
            image=images[idx],
            fname=fname,
        )
        print(f"[eval] Saved {fname}")

    print(f"[eval] Done. Results in {out_dir}/")


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def _plot_coverage(result: dict, fname: Path) -> None:
    """result: {"all": {dim: {cov68, cov95}}, "point": ..., "blob": ...}"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    strata = list(result.keys())
    dims   = list(result[strata[0]].keys())
    colors68 = ["steelblue", "seagreen", "mediumpurple", "goldenrod"]
    colors95 = ["darkorange", "firebrick", "darkcyan", "saddlebrown"]

    n_strata = len(strata)
    fig, axes = plt.subplots(1, n_strata, figsize=(5 * n_strata, 4), sharey=True)
    if n_strata == 1:
        axes = [axes]

    for ax, stratum, c68, c95 in zip(axes, strata, colors68, colors95):
        d = result[stratum]
        vals68 = [d[dim]["cov68"] for dim in dims]
        vals95 = [d[dim]["cov95"] for dim in dims]
        x = np.arange(len(dims))
        w = 0.35
        ax.bar(x - w/2, vals68, w, label="68%", color=c68)
        ax.bar(x + w/2, vals95, w, label="95%", color=c95)
        ax.axhline(0.68, color=c68, lw=1, ls="--", alpha=0.5)
        ax.axhline(0.95, color=c95, lw=1, ls="--", alpha=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels(dims, rotation=25, ha="right", fontsize=8)
        ax.set_ylim(0, 1.05)
        ax.set_title(stratum)
        ax.legend(fontsize=8)

    axes[0].set_ylabel("Empirical coverage fraction")
    fig.suptitle("MDN-Asp posterior coverage — held-out set", fontsize=11)
    fig.tight_layout()
    fig.savefig(fname, dpi=150)
    plt.close(fig)


def _plot_marginals(
    *,
    idx: int,
    targets: np.ndarray,
    log_w:   np.ndarray,   # (K,)
    mu:      np.ndarray,   # (K, 7)
    std:     np.ndarray,   # (K, 7)
    pa_samples: np.ndarray,  # (S,)
    image:   torch.Tensor,   # (2, 128, 128)
    fname:   Path,
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.special import ndtr

    w = np.exp(log_w - log_w.max())
    w /= w.sum()

    n_cols = 7  # residual + 6 parameter marginals
    fig, axes = plt.subplots(1, n_cols, figsize=(n_cols * 2.8, 3))

    # Residual cutout
    residual = image[0].numpy()
    vmax = np.abs(residual).max()
    axes[0].imshow(residual, origin="lower", cmap="RdBu_r",
                   vmin=-vmax, vmax=vmax)
    axes[0].set_title("residual")
    axes[0].axis("off")

    # Marginals for dims 0-4
    dim_labels = DIM_NAMES  # 6 entries
    for d in range(5):
        ax = axes[d + 1]
        truth = targets[idx, d]
        # Grid over ±4 sigma of the widest component
        span = 4 * std[:, d].max()
        center = float(np.dot(w, mu[:, d]))
        grid = np.linspace(center - span, center + span, 400)
        # Mixture pdf
        pdf = np.zeros_like(grid)
        for k in range(len(w)):
            z = (grid - mu[k, d]) / max(std[k, d], 1e-12)
            pdf += w[k] * (1.0 / (std[k, d] * np.sqrt(2 * np.pi))) * np.exp(-0.5 * z**2)
        ax.plot(grid, pdf, lw=1.5, color="steelblue")
        ax.axvline(truth, color="red", lw=1.5, ls="--", label="truth")
        ax.set_title(dim_labels[d], fontsize=9)
        ax.set_yticks([])
        if d == 0:
            ax.legend(fontsize=7)

    # PA marginal from samples
    ax = axes[6]
    truth_pa = targets[idx, 5]
    ax.hist(pa_samples, bins=40, density=True, color="steelblue", alpha=0.7)
    ax.axvline(truth_pa, color="red", lw=1.5, ls="--")
    ax.set_title("PA", fontsize=9)
    ax.set_yticks([])

    fig.suptitle(f"Sample {idx} — truth (red dashed)", fontsize=10)
    fig.tight_layout()
    fig.savefig(fname, dpi=150)
    plt.close(fig)


def _collate(batch):
    res, psf, cond, tgt = zip(*batch)
    img = torch.stack([torch.stack(list(res)), torch.stack(list(psf))], dim=1)
    return img, torch.stack(list(cond)), torch.stack(list(tgt))


if __name__ == "__main__":
    main()
