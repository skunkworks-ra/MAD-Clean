#!/usr/bin/env python3
"""
scripts/collect_z_codes.py — encode clean images through a trained VAE.

Produces z codes for LatentPriorTrainer (Phase 2 Step 3).

Usage
-----
    pixi run -e gpu python scripts/collect_z_codes.py \
        --data crumb_data/flow_pairs_vla.npz \
        --vae  models/vae_d128.pt \
        --out  crumb_data/z_codes_d128.npz \
        --device cuda

Output
------
    z_codes : (N, d)  float32 — latent mean μ for each clean image
    d       : int             — latent dimension (scalar)

Notes
-----
Uses μ (encoder mean), not a stochastic sample. Deterministic encoding
gives stable, reproducible z codes for flow prior training. Stochastic
samples would add noise to the training targets with no benefit.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

from mad_clean.training import VAEModel


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Encode clean images through a trained VAE to collect z codes.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--data",   required=True, help="Path to .npz with 'clean' key")
    p.add_argument("--vae",    required=True, help="Path to trained VAEModel .pt file")
    p.add_argument("--out",    required=True, help="Output .npz path for z codes")
    p.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    p.add_argument("--batch_size", type=int, default=64,
                   help="Encoding batch size (no gradient — can be larger than training)")
    return p


def main() -> None:
    args = build_parser().parse_args()

    data_path = Path(args.data)
    if not data_path.exists():
        print(f"ERROR: data file not found: {data_path}", file=sys.stderr)
        sys.exit(1)

    vae_path = Path(args.vae)
    if not vae_path.exists():
        print(f"ERROR: VAE checkpoint not found: {vae_path}", file=sys.stderr)
        sys.exit(1)

    data  = np.load(data_path)
    clean = data["clean"].astype(np.float32)   # (N, H, W)
    N     = len(clean)
    print(f"Loaded {N} clean images  shape={clean.shape[1]}×{clean.shape[2]}")

    vae = VAEModel.load(args.vae, device=args.device)
    vae._net.eval()
    d   = vae.latent_dim

    z_codes = np.empty((N, d), dtype=np.float32)

    print(f"Encoding {N} images → z ∈ ℝ^{d}  (batch={args.batch_size}) ...")
    with torch.no_grad():
        for start in range(0, N, args.batch_size):
            end  = min(start + args.batch_size, N)
            xb   = torch.from_numpy(clean[start:end, None]).float().to(args.device)
            # Match VAETrainer's per-image peak normalisation
            peak = xb.amax(dim=(2, 3), keepdim=True).clamp(min=1e-8)
            xb   = xb / peak
            mu, _  = vae._net.encode(xb)
            z_codes[start:end] = mu.cpu().numpy()

            if (start // args.batch_size) % 5 == 0:
                print(f"  {end}/{N}", flush=True)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_path, z_codes=z_codes, d=d)
    print(f"Saved z_codes ({N}, {d}) → {out_path}")

    # Quick sanity check on the z distribution
    print(f"\nZ-code statistics:")
    print(f"  mean per-dim:  {z_codes.mean(axis=0).mean():.4f}  (expect ≈ 0 if KL-regularised)")
    print(f"  std  per-dim:  {z_codes.std(axis=0).mean():.4f}  (expect ≈ 1 if KL-regularised)")
    print(f"  overall range: [{z_codes.min():.3f}, {z_codes.max():.3f}]")


if __name__ == "__main__":
    main()
