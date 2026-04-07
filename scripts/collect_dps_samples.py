#!/usr/bin/env python3
"""
scripts/collect_dps_samples.py — Harvest DPS posterior samples for Phase 2 training.

Runs DPSSolver on a subset of training images and saves the posterior samples.
The output is used as training data for AmortisedPosteriorTrainer (Phase 2):
the conditional flow q_φ(x_clean | dirty) is distilled from these samples.

Output .npz keys:
    dirty    : (N, H, W) float32 — conditioning dirty images
    samples  : (N, M, H, W) float32 — M posterior draws per image
    psf_norm : (H_p, W_p) float32 — PSF (passed through for downstream use)

Usage:
    pixi run -e gpu python scripts/collect_dps_samples.py \\
        --data       crumb_data/flow_pairs_vla.npz \\
        --prior      models/prior_model.pt \\
        --out        crumb_data/dps_samples.npz \\
        --n_collect  500 \\
        --n_samples  8 \\
        --dps_weight 1.0 \\
        --device     cuda
"""

import argparse
from pathlib import Path

import numpy as np
import torch


def main() -> None:
    p = argparse.ArgumentParser(
        description="Collect DPS posterior samples for amortised posterior training.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--data",       default="crumb_data/flow_pairs_vla.npz")
    p.add_argument("--prior",      default="models/prior_model.pt")
    p.add_argument("--out",        default="crumb_data/dps_samples.npz")
    p.add_argument("--n_collect",  type=int,   default=500,
                   help="Number of training images to collect samples for.")
    p.add_argument("--n_samples",  type=int,   default=8,
                   help="Posterior draws per image (M).")
    p.add_argument("--dps_weight", type=float, default=1.0)
    p.add_argument("--noise_std",  type=float, default=0.05)
    p.add_argument("--seed",       type=int,   default=42)
    p.add_argument("--device",     default="cpu", choices=["cpu", "cuda"])
    args = p.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        print("WARNING: CUDA not available — falling back to CPU (will be slow).")
        args.device = "cpu"

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # ── Load data ──────────────────────────────────────────────────────────────
    data      = np.load(args.data)
    dirty_all = data["dirty"].astype(np.float32)
    psf_norm  = data["psf_norm"].astype(np.float32)
    N = len(dirty_all)

    # Use first n_collect examples (training set; test set is at the end)
    n_collect = min(args.n_collect, N)
    dirty_src = dirty_all[:n_collect]
    print(f"Collecting {args.n_samples} DPS samples for {n_collect}/{N} training images")

    # ── Build DPS solver ───────────────────────────────────────────────────────
    from mad_clean.training.flow import FlowModel
    from mad_clean.solvers import DPSSolver

    prior  = FlowModel.load(args.prior, device=args.device)
    solver = DPSSolver(
        prior,
        psf_norm   = psf_norm,
        noise_std  = args.noise_std,
        n_steps    = 50,
        n_samples  = args.n_samples,
        dps_weight = args.dps_weight,
        device     = args.device,
    )
    print(f"Solver: {solver}")

    # ── Collect samples ────────────────────────────────────────────────────────
    H, W    = dirty_src.shape[1], dirty_src.shape[2]
    M       = args.n_samples
    samples_out = np.zeros((n_collect, M, H, W), dtype=np.float32)

    for i in range(n_collect):
        dirty_t = torch.from_numpy(dirty_src[i]).float()
        # sample_all returns (M, H, W)
        samps = solver.sample_all(dirty_t, n_samples=M).cpu().numpy()
        samples_out[i] = samps

        if (i + 1) % 50 == 0:
            print(f"  {i + 1}/{n_collect} done  "
                  f"(flux_rec sample check: "
                  f"{np.clip(samps, 0, None).sum() / (np.clip(dirty_src[i], 0, None).sum() + 1e-8):.2f})")

    # ── Save ───────────────────────────────────────────────────────────────────
    np.savez(
        out_path,
        dirty    = dirty_src,
        samples  = samples_out,
        psf_norm = psf_norm,
    )
    size_mb = out_path.stat().st_size / 1e6
    print(f"\nSaved → {out_path}  ({size_mb:.1f} MB)")
    print(f"  dirty:   {dirty_src.shape}")
    print(f"  samples: {samples_out.shape}")


if __name__ == "__main__":
    main()
