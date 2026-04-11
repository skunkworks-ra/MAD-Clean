#!/usr/bin/env python3
"""
scripts/inspect_z_codes.py — Diagnostic: z-code statistics from FISTA(clean, D).

Encodes clean images through the Variant B dictionary WITHOUT a PSF (clean encoding).
Reports whether the z-space is suitable for a latent flow prior:

  - Sparsity: what fraction of z entries are non-zero?
  - Distribution: z value histogram — should be learnable (not degenerate)
  - Reconstruction: ‖D ⊛ z - clean‖² / ‖clean‖² — is the dictionary expressive?

This is a READ-ONLY diagnostic. No training. No changes to any model.

Usage:
    pixi run python scripts/inspect_z_codes.py \\
        --data     crumb_data/flow_pairs_vla.npz \\
        --model_b  models/cdl_filters_conv.npz \\
        --n        100 \\
        --out      logs/z_diagnostic
"""

import argparse
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from mad_clean.filters import FilterBank
from mad_clean.solvers import ConvSolver


def main() -> None:
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--data",    default="crumb_data/flow_pairs_vla.npz")
    p.add_argument("--model_b", default="models/cdl_filters_conv.npz")
    p.add_argument("--n",       type=int, default=100,
                   help="Number of clean images to encode.")
    p.add_argument("--lmbda",   type=float, default=0.01,
                   help="FISTA sparsity penalty (match training value).")
    p.add_argument("--fista_iter", type=int, default=100)
    p.add_argument("--out",     default="logs/z_diagnostic")
    p.add_argument("--device",  default="cpu", choices=["cpu", "cuda"])
    args = p.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Load data and model ────────────────────────────────────────────────────
    data  = np.load(args.data)
    clean = data["clean"].astype(np.float32)
    N     = min(args.n, len(clean))
    print(f"Encoding {N} clean images  shape={clean.shape[1]}×{clean.shape[2]}")

    fb = FilterBank.load(args.model_b, device=args.device)
    # No PSF — encode clean images directly: min ‖D⊛z - clean‖² + λ‖z‖₁
    solver = ConvSolver(fb, lmbda=args.lmbda, n_iter=args.fista_iter, psf=None)
    print(f"Solver: {solver}")
    print(f"Dictionary: K={fb.K} atoms, F={fb.F}px")

    # ── Encode ─────────────────────────────────────────────────────────────────
    all_z         = []   # list of (K, H, W) tensors
    recon_errors  = []   # ‖D⊛z - clean‖ / ‖clean‖
    sparsities    = []   # fraction of z entries with |z| > 1e-4

    for i in range(N):
        x_clean  = torch.from_numpy(clean[i]).float().to(args.device)   # (H, W)
        z        = solver.encode_island(x_clean)                         # (K, H, W)

        # Reconstruct: D ⊛ z
        z_fft    = torch.fft.rfft2(z)
        H, W     = x_clean.shape
        pad_h    = H - fb.F
        pad_w    = W - fb.F
        atoms_p  = torch.nn.functional.pad(fb.atoms, (0, pad_w, 0, pad_h))
        af       = torch.fft.rfft2(atoms_p)
        recon    = torch.fft.irfft2((af * z_fft).sum(dim=0), s=(H, W))  # (H, W)

        err      = float((recon - x_clean).norm() / (x_clean.norm() + 1e-8))
        sparsity = float((z.abs() < 1e-4).float().mean())

        recon_errors.append(err)
        sparsities.append(sparsity)
        all_z.append(z.cpu())

        if (i + 1) % 20 == 0:
            print(f"  {i+1}/{N}  recon_err={err:.3f}  sparsity={sparsity:.3f}")

    # ── Summary statistics ─────────────────────────────────────────────────────
    recon_errors = np.array(recon_errors)
    sparsities   = np.array(sparsities)

    print(f"\n── Z-code diagnostic ──────────────────────────────")
    print(f"  Sparsity (fraction |z|<1e-4):  "
          f"mean={sparsities.mean():.3f}  std={sparsities.std():.3f}")
    print(f"  Reconstruction error ‖D⊛z-x‖/‖x‖:  "
          f"mean={recon_errors.mean():.3f}  std={recon_errors.std():.3f}  "
          f"max={recon_errors.max():.3f}")

    # Flatten all non-zero z values for distribution analysis
    z_all   = torch.stack(all_z)                         # (N, K, H, W)
    z_flat  = z_all.numpy().ravel()
    z_nz    = z_flat[np.abs(z_flat) > 1e-4]             # non-zero entries only

    print(f"  Non-zero z values:  n={len(z_nz):,}  "
          f"mean={z_nz.mean():.4f}  std={z_nz.std():.4f}  "
          f"min={z_nz.min():.4f}  max={z_nz.max():.4f}")
    print(f"  K={fb.K} channels  total z entries per image: "
          f"{z_all.shape[1]*z_all.shape[2]*z_all.shape[3]:,}")

    # Verdict
    print(f"\n── Verdict ─────────────────────────────────────────")
    if recon_errors.mean() < 0.15:
        print(f"  Reconstruction: GOOD (mean error {recon_errors.mean():.3f} < 0.15)")
    elif recon_errors.mean() < 0.30:
        print(f"  Reconstruction: MARGINAL (mean error {recon_errors.mean():.3f})")
        print(f"  → Consider retraining Variant B on clean images")
    else:
        print(f"  Reconstruction: POOR (mean error {recon_errors.mean():.3f} > 0.30)")
        print(f"  → Variant B must be retrained on clean images before Phase 2")

    if sparsities.mean() > 0.90:
        print(f"  Sparsity: GOOD for flow prior ({sparsities.mean():.1%} zeros)")
    elif sparsities.mean() > 0.70:
        print(f"  Sparsity: ACCEPTABLE ({sparsities.mean():.1%} zeros) — "
              f"flow prior may need more epochs")
    else:
        print(f"  Sparsity: TOO DENSE ({sparsities.mean():.1%} zeros) — "
              f"increase λ or retrain dictionary")

    # ── Plots ──────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    # 1. Z value distribution (non-zero only)
    axes[0].hist(z_nz, bins=80, color="#3a86ff", edgecolor="none", alpha=0.8,
                 density=True)
    axes[0].set_xlabel("z value (non-zero entries)", fontsize=10)
    axes[0].set_ylabel("Density", fontsize=10)
    axes[0].set_title("Non-zero z distribution\n(symmetric → flow-learnable)", fontsize=9)
    axes[0].axvline(0, color="k", lw=1)
    axes[0].grid(alpha=0.3)

    # 2. Reconstruction error distribution
    axes[1].hist(recon_errors, bins=30, color="#d62728", edgecolor="none", alpha=0.8)
    axes[1].axvline(recon_errors.mean(), color="k", lw=1.5,
                    label=f"mean={recon_errors.mean():.3f}")
    axes[1].set_xlabel("Relative reconstruction error ‖D⊛z-x‖/‖x‖", fontsize=10)
    axes[1].set_ylabel("Count", fontsize=10)
    axes[1].set_title("Reconstruction quality\n(< 0.15 = good)", fontsize=9)
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.3)

    # 3. Sparsity distribution
    axes[2].hist(sparsities, bins=30, color="#2ca02c", edgecolor="none", alpha=0.8)
    axes[2].axvline(sparsities.mean(), color="k", lw=1.5,
                    label=f"mean={sparsities.mean():.3f}")
    axes[2].set_xlabel("Fraction of z entries with |z| < 1e-4", fontsize=10)
    axes[2].set_ylabel("Count", fontsize=10)
    axes[2].set_title("Z-code sparsity\n(> 0.90 = good for flow prior)", fontsize=9)
    axes[2].legend(fontsize=8)
    axes[2].grid(alpha=0.3)

    plt.suptitle(
        f"Z-code diagnostic — Variant B ({fb.K} atoms, F={fb.F}px)  n={N}",
        fontsize=11, fontweight="bold",
    )
    plt.tight_layout()
    plot_path = out_dir / "z_diagnostic.png"
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"\nSaved → {plot_path}")

    # Save raw stats
    np.savez(out_dir / "z_stats.npz",
             recon_errors=recon_errors, sparsities=sparsities,
             z_nz_sample=z_nz[:100_000])   # cap to 100k for size
    print(f"Saved → {out_dir}/z_stats.npz")


if __name__ == "__main__":
    main()
