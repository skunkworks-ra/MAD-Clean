#!/usr/bin/env python3
"""
scripts/deconvolve.py — CLI for MAD-CLEAN or Hogbom deconvolution.

Variant hogbom:
    mad-deconvolve --variant hogbom \\
        --dirty dirty.fits --psf psf.fits --out_dir results/

Variant A (patch OMP):
    mad-deconvolve --variant A \\
        --dirty dirty.fits --psf psf.fits \\
        --atoms models/cdl_filters_patch.npz --out_dir results/

Variant B (convolutional FISTA):
    mad-deconvolve --variant B \\
        --dirty dirty.fits --psf psf.fits \\
        --atoms models/cdl_filters_conv.npz --out_dir results/
"""

import argparse
import sys
from pathlib import Path

import numpy as np

from mad_clean import FilterBank, PatchSolver, ConvSolver, MADClean
from mad_clean import hogbom_clean
from mad_clean.io import load_image_data, save_fits


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Run MAD-CLEAN or Hogbom deconvolution.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--variant",      required=True, choices=["hogbom", "A", "B"])
    p.add_argument("--dirty",        required=True,
                   help="Dirty image — FITS path or .npy")
    p.add_argument("--psf",          required=True,
                   help="PSF — FITS path or .npy. Peak must be at image centre (CASA convention).")
    p.add_argument("--atoms",        default=None,
                   help="FilterBank .npz file produced by scripts/train.py [required for A/B]")
    p.add_argument("--out_dir",      required=True,
                   help="Directory for output FITS and .npy files")
    p.add_argument("--device",       default="cpu", choices=["cpu", "cuda"])

    p.add_argument("--gain",         type=float, default=0.1,  help="Loop gain")
    p.add_argument("--n_max",        type=int,   default=500,  help="Maximum iterations")
    p.add_argument("--epsilon_frac", type=float, default=0.01,
                   help="[A/B] Convergence as fraction of initial residual RMS")
    p.add_argument("--threshold",    type=float, default=None,
                   help="[hogbom] Stop when |peak| < threshold. "
                        "Default: auto (0.1 × dirty peak).")

    p.add_argument("--psf_energy_frac", type=float, default=0.90,
                   help="PSF patch energy fraction (default 0.90)")
    p.add_argument("--refresh_every",   type=int,   default=100,
                   help="[A/B] FFT residual refresh interval")

    p.add_argument("--n_nonzero", type=int,   default=5,   help="[A] OMP sparsity")
    p.add_argument("--stride",    type=int,   default=8,   help="[A] Patch tiling stride")

    p.add_argument("--lmbda",     type=float, default=0.1, help="[B] FISTA L1 penalty")
    p.add_argument("--fista_iter",type=int,   default=100, help="[B] FISTA iterations")

    return p


def main() -> None:
    args = build_parser().parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.variant == "hogbom":
        import torch
        dirty_np = load_image_data(args.dirty)
        psf_np   = load_image_data(args.psf)
        dirty_t  = torch.from_numpy(dirty_np).float()
        psf_t    = torch.from_numpy(psf_np  ).float()

        result = hogbom_clean(
            dirty_t, psf_t,
            gain        = args.gain,
            threshold   = args.threshold,
            n_iter      = args.n_max,
            energy_frac = args.psf_energy_frac,
            device      = args.device,
        )
        model_np    = result["model"].cpu().numpy().astype(np.float32)
        residual_np = result["residual"].cpu().numpy().astype(np.float32)
        save_fits(model_np,    out_dir / "hogbom_model.fits")
        save_fits(residual_np, out_dir / "hogbom_residual.fits")
        print(f"Done  variant=hogbom  n_iter={result['n_iter']}  "
              f"converged={result['converged']}  peak_flux={result['peak_flux']:.4e}")
        return

    if args.atoms is None:
        print("ERROR: --atoms is required for variants A and B", file=sys.stderr)
        sys.exit(1)

    atoms_path = Path(args.atoms)
    if not atoms_path.exists():
        print(f"ERROR: atoms file not found: {atoms_path}", file=sys.stderr)
        sys.exit(1)

    fb = FilterBank.load(atoms_path, device=args.device)

    if args.variant == "A":
        solver = PatchSolver(fb, n_nonzero=args.n_nonzero, stride=args.stride)
    else:
        solver = ConvSolver(fb, lmbda=args.lmbda, n_iter=args.fista_iter)

    mc = MADClean(
        fb, solver,
        gamma         = args.gain,
        epsilon_frac  = args.epsilon_frac,
        n_max         = args.n_max,
        refresh_every = args.refresh_every,
        energy_frac   = args.psf_energy_frac,
        device        = args.device,
    )

    result = mc.deconvolve(args.dirty, args.psf, out_dir=out_dir)
    print(f"Done  variant={args.variant}  n_iter={result['n_iter']}  "
          f"converged={result['converged']}  peak_flux={result['peak_flux']:.4e}")


if __name__ == "__main__":
    main()
