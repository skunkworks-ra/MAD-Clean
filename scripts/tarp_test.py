#!/usr/bin/env python3
"""
scripts/tarp_test.py — TARP calibration test for DPS posterior.

Tests whether DPS posterior samples are statistically calibrated against
held-out ground truth using the TARP (Testing Approximate Reliability of
Posteriors) framework (Lemos et al. 2023).

A calibrated posterior satisfies ECP(α) = α for all α ∈ [0,1]:
  - ECP(α) < α → overconfident (credible intervals too narrow)
  - ECP(α) > α → underconfident (credible intervals too wide)

Two calibration checks are run:
  1. Integrated flux — most relevant for catalogue science
  2. Per-pixel values (subsampled) — tests spatial structure

NOTE: The test set is the last --n_test examples from the data file.
For publication-quality results, generate a fresh held-out set via
scripts/simulate.py and pass it as --data.

Usage:
    pixi run -e gpu python scripts/tarp_test.py \\
        --data       crumb_data/flow_pairs_vla.npz \\
        --prior      models/prior_model.pt \\
        --out        logs/tarp \\
        --n_test     100 \\
        --n_samples  20 \\
        --dps_weight 1.0 \\
        --device     cuda

Reference: Lemos et al. (2023) arXiv:2302.03026
"""

import argparse
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch


# ---------------------------------------------------------------------------
# TARP utilities
# ---------------------------------------------------------------------------

def compute_ranks(samples: np.ndarray, truth: np.ndarray) -> np.ndarray:
    """
    Compute rank statistics for scalar summaries.

    For each example i:
        rank_i = fraction of samples < truth_i

    Under a calibrated posterior, ranks ~ Uniform[0, 1].

    Parameters
    ----------
    samples : (N, S) — N examples, S posterior draws each
    truth   : (N,)   — true values, one per example

    Returns
    -------
    ranks : (N,) in [0, 1]
    """
    return np.mean(samples < truth[:, None], axis=1)


def compute_ecp(ranks: np.ndarray, n_alpha: int = 51) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute the Expected Coverage Probability (ECP) curve.

    ECP(α) = fraction of test cases where the true value falls within the
    central α-credible interval, i.e. rank ∈ [(1-α)/2, (1+α)/2].

    Parameters
    ----------
    ranks   : (N,) rank statistics in [0, 1]
    n_alpha : number of α levels to evaluate

    Returns
    -------
    alphas : (n_alpha,) confidence levels in [0, 1]
    ecps   : (n_alpha,) empirical coverage probabilities
    """
    alphas = np.linspace(0.0, 1.0, n_alpha)
    ecps = np.array([
        float(np.mean((ranks >= (1.0 - a) / 2.0) & (ranks <= (1.0 + a) / 2.0)))
        for a in alphas
    ])
    return alphas, ecps


def ecp_area_metric(alphas: np.ndarray, ecps: np.ndarray) -> float:
    """
    Signed area between ECP curve and diagonal.
    Positive → underconfident; negative → overconfident.
    Near zero → well calibrated.
    """
    return float(np.trapz(ecps - alphas, alphas))


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_ecp(
    alphas_dict : dict,   # {label: (alphas, ecps)}
    out_path    : Path,
) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot([0, 1], [0, 1], "k--", lw=1.5, label="Ideal (calibrated)", zorder=5)

    colors = ["#3a86ff", "#d62728", "#2ca02c", "#9467bd"]
    for (label, (alphas, ecps)), color in zip(alphas_dict.items(), colors):
        area = ecp_area_metric(alphas, ecps)
        ax.plot(alphas, ecps, color=color, lw=2,
                label=f"{label}  (area={area:+.3f})")

    ax.fill_between([0, 1], [0, 1], alpha=0.05, color="gray")
    ax.set_xlabel("Confidence level α", fontsize=11)
    ax.set_ylabel("Expected coverage probability (ECP)", fontsize=11)
    ax.set_title("TARP calibration test", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved → {out_path}")


def plot_rank_histogram(ranks_dict: dict, out_path: Path) -> None:
    n = len(ranks_dict)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4), squeeze=False)
    for ax, (label, ranks) in zip(axes[0], ranks_dict.items()):
        ax.hist(ranks, bins=20, color="#3a86ff", edgecolor="white", alpha=0.8,
                density=True)
        ax.axhline(1.0, color="black", linestyle="--", lw=1.5, label="Ideal (uniform)")
        ax.set_xlabel("Rank", fontsize=10)
        ax.set_ylabel("Density", fontsize=10)
        ax.set_title(label, fontsize=10, fontweight="bold")
        ax.legend(fontsize=8)
        ax.set_xlim(0, 1)
        ax.grid(alpha=0.3)
    plt.suptitle("Rank histograms (uniform = calibrated)", fontsize=11)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved → {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(
        description="TARP calibration test for DPS posterior.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--data",       default="crumb_data/flow_pairs_vla.npz")
    p.add_argument("--prior",      default="models/prior_model.pt")
    p.add_argument("--out",        default="logs/tarp")
    p.add_argument("--n_test",     type=int,   default=100,
                   help="Test examples — taken from end of data file.")
    p.add_argument("--n_samples",  type=int,   default=20,
                   help="Posterior draws per example (≥20 for reliable ECP).")
    p.add_argument("--dps_weight", type=float, default=0.05,
                   help="DPS likelihood gradient scale ζ.")
    p.add_argument("--noise_std",  type=float, default=0.05)
    p.add_argument("--n_pixels",   type=int,   default=500,
                   help="Pixels subsampled per example for pixel-level TARP.")
    p.add_argument("--device",     default="cpu", choices=["cpu", "cuda"])
    args = p.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Load data ──────────────────────────────────────────────────────────────
    data     = np.load(args.data)
    clean_all = data["clean"].astype(np.float32)
    dirty_all = data["dirty"].astype(np.float32)
    psf_norm  = data.get("psf_norm", data.get("psf")).astype(np.float32)
    N = len(clean_all)

    # Use last n_test examples as held-out set
    n_test = min(args.n_test, N // 5)
    if n_test < 20:
        print(f"WARNING: only {n_test} test examples. ECP curve will be noisy.")
    clean_test = clean_all[-n_test:]
    dirty_test = dirty_all[-n_test:]
    print(f"TARP test set: last {n_test}/{N} examples  "
          f"shape={clean_test.shape[1]}×{clean_test.shape[2]}")

    # ── Build DPS solver ───────────────────────────────────────────────────────
    if args.device == "cuda" and not torch.cuda.is_available():
        print("WARNING: CUDA not available — falling back to CPU (will be slow).")
        device = "cpu"
    else:
        device = args.device

    from mad_clean.training.flow import FlowModel
    from mad_clean.solvers import DPSSolver

    prior  = FlowModel.load(args.prior, device=device)
    solver = DPSSolver(
        prior,
        psf_norm   = psf_norm,
        noise_std  = args.noise_std,
        n_steps    = 50,
        n_samples  = args.n_samples,
        dps_weight = args.dps_weight,
        device     = device,
    )
    print(f"Solver: {solver}")

    # ── Collect samples and compute rank statistics ────────────────────────────
    flux_samples  = []   # (n_test, n_samples) — integrated flux per sample
    flux_truth    = []   # (n_test,)

    pixel_samples = []   # (n_test * n_pixels, n_samples)
    pixel_truth   = []   # (n_test * n_pixels,)

    rng = np.random.default_rng(42)

    for i in range(n_test):
        dirty      = torch.from_numpy(dirty_test[i]).float()
        true_clean = clean_test[i]

        # Individual posterior samples: (S, H, W)
        samples = solver.sample_all(dirty).cpu().numpy()  # (S, H, W)

        # --- Integrated flux ---
        flux_per_sample = np.clip(samples, 0, None).sum(axis=(1, 2))   # (S,)
        true_flux       = float(np.clip(true_clean, 0, None).sum())
        flux_samples.append(flux_per_sample)
        flux_truth.append(true_flux)

        # --- Per-pixel (subsampled) ---
        H, W   = true_clean.shape
        n_pix  = min(args.n_pixels, H * W)
        pix_idx = rng.choice(H * W, size=n_pix, replace=False)
        # samples: (S, H*W) → select pixels
        samples_flat = samples.reshape(len(samples), -1)[:, pix_idx]  # (S, n_pix)
        truth_flat   = true_clean.ravel()[pix_idx]                       # (n_pix,)
        pixel_samples.append(samples_flat.T)    # (n_pix, S)
        pixel_truth.append(truth_flat)

        if (i + 1) % 10 == 0:
            print(f"  {i + 1}/{n_test} evaluated")

    # ── Rank statistics ────────────────────────────────────────────────────────
    flux_samples_arr = np.stack(flux_samples)    # (n_test, S)
    flux_truth_arr   = np.array(flux_truth)       # (n_test,)
    flux_ranks       = compute_ranks(flux_samples_arr, flux_truth_arr)

    pixel_samples_arr = np.concatenate(pixel_samples, axis=0)   # (n_test*n_pix, S)
    pixel_truth_arr   = np.concatenate(pixel_truth,   axis=0)   # (n_test*n_pix,)
    pixel_ranks       = compute_ranks(pixel_samples_arr, pixel_truth_arr)

    # ── ECP curves ────────────────────────────────────────────────────────────
    flux_alphas,  flux_ecps  = compute_ecp(flux_ranks)
    pixel_alphas, pixel_ecps = compute_ecp(pixel_ranks)

    flux_area  = ecp_area_metric(flux_alphas,  flux_ecps)
    pixel_area = ecp_area_metric(pixel_alphas, pixel_ecps)

    print(f"\nFlux ECP area:  {flux_area:+.4f}  "
          f"({'underconfident' if flux_area > 0 else 'overconfident' if flux_area < 0 else 'calibrated'})")
    print(f"Pixel ECP area: {pixel_area:+.4f}  "
          f"({'underconfident' if pixel_area > 0 else 'overconfident' if pixel_area < 0 else 'calibrated'})")

    # Mean rank std (should be ~0.289 for Uniform[0,1]; lower = overconfident)
    print(f"Flux rank std:  {flux_ranks.std():.3f}  (ideal: {1/12**0.5:.3f})")
    print(f"Pixel rank std: {pixel_ranks.std():.3f}  (ideal: {1/12**0.5:.3f})")

    # ── Plots ──────────────────────────────────────────────────────────────────
    plot_ecp(
        {
            "Flux (integrated)": (flux_alphas,  flux_ecps),
            "Pixel (subsampled)": (pixel_alphas, pixel_ecps),
        },
        out_dir / "ecp_curve.png",
    )
    plot_rank_histogram(
        {
            "Flux ranks":  flux_ranks,
            "Pixel ranks": pixel_ranks,
        },
        out_dir / "rank_histograms.png",
    )

    # ── Save numerical results ─────────────────────────────────────────────────
    np.savez(
        out_dir / "tarp_results.npz",
        flux_alphas=flux_alphas,   flux_ecps=flux_ecps,
        pixel_alphas=pixel_alphas, pixel_ecps=pixel_ecps,
        flux_ranks=flux_ranks,     pixel_ranks=pixel_ranks,
        flux_area=flux_area,       pixel_area=pixel_area,
        dps_weight=args.dps_weight, n_test=n_test, n_samples=args.n_samples,
    )
    print(f"\nSaved → {out_dir}/tarp_results.npz")
    print(f"Done. Results in {out_dir}/")


if __name__ == "__main__":
    main()
