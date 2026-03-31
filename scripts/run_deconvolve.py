#!/usr/bin/env python3
"""
run_deconvolve.py — CLI wrapper for MAD-CLEAN deconvolution.

Variant hogbom (classical Hogbom CLEAN):
    python scripts/run_deconvolve.py --variant hogbom \\
        --dirty dirty.fits --psf psf.fits --out_dir results/ \\
        [--gain 0.1] [--n_max 1000] [--threshold 0.0] [--device cpu]

Variant A  (patch OMP):
    python scripts/run_deconvolve.py --variant A \\
        --dirty dirty.fits --psf psf.fits \\
        --atoms models/cdl_filters_patch.npy \\
        --out_dir results/ \\
        [--gamma 0.1] [--n_max 500] [--epsilon_frac 0.01] \\
        [--n_nonzero 5] [--stride 8] [--device cpu]

Variant B  (convolutional FISTA):
    python scripts/run_deconvolve.py --variant B \\
        --dirty dirty.fits --psf psf.fits \\
        --atoms models/cdl_filters_conv.npy \\
        --out_dir results/ \\
        [--lmbda 0.1] [--fista_iter 100] [--device cuda]

Inputs: FITS files or .npy arrays for dirty and PSF.
"""

import argparse
import sys
from pathlib import Path

import numpy as np

# Make project root importable when running as a script
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from filters    import FilterBank            # noqa: E402
from solvers    import PatchSolver, ConvSolver  # noqa: E402
from deconvolver import MADClean             # noqa: E402
from hogbom     import hogbom_clean          # noqa: E402
from io         import load_image_data, save_fits  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Run MAD-CLEAN or Hogbom deconvolution.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--variant",      required=True, choices=["hogbom", "A", "B"])
    p.add_argument("--dirty",        required=True,
                   help="Dirty image — FITS path or .npy")
    p.add_argument("--psf",          required=True,
                   help="PSF — FITS path or .npy. Peak must be at image centre (CASA convention)")
    p.add_argument("--atoms",        default=None,
                   help="FilterBank .npy file produced by run_train.py [required for A/B]")
    p.add_argument("--out_dir",      required=True,
                   help="Directory for output FITS and .npy files")
    p.add_argument("--device",       default="cpu", choices=["cpu", "cuda"])

    # Shared loop params
    p.add_argument("--gain",         type=float, default=0.1,
                   help="Loop gain")
    p.add_argument("--n_max",        type=int,   default=500,
                   help="Maximum iterations")
    p.add_argument("--epsilon_frac", type=float, default=0.01,
                   help="[A/B] Convergence as fraction of initial residual RMS")
    p.add_argument("--threshold",    type=float, default=0.0,
                   help="[hogbom] Stop when |peak| < threshold")

    # PSF patch params (all variants)
    p.add_argument("--psf_energy_frac", type=float, default=0.90,
                   help="PSF patch energy fraction (default 0.90)")
    p.add_argument("--min_sidelobes",   type=int,   default=2,
                   help="Minimum PSF sidelobes in patch (default 2)")

    # MADClean params (A/B)
    p.add_argument("--refresh_every",  type=int,   default=100,
                   help="[A/B] FFT residual refresh interval (default 100)")

    # Variant A
    p.add_argument("--n_nonzero", type=int,   default=5,
                   help="[A] OMP sparsity — max active atoms per patch")
    p.add_argument("--stride",    type=int,   default=8,
                   help="[A] Patch tiling stride in pixels")

    # Variant B
    p.add_argument("--lmbda",     type=float, default=0.1,
                   help="[B] FISTA L1 penalty")
    p.add_argument("--fista_iter",type=int,   default=100,
                   help="[B] FISTA iterations per island at inference")

    return p


def main() -> None:
    args = build_parser().parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Variant hogbom ────────────────────────────────────────────────────────
    if args.variant == "hogbom":
        import torch
        dirty_np = load_image_data(args.dirty)
        psf_np   = load_image_data(args.psf)
        dirty_t  = torch.from_numpy(dirty_np).float()
        psf_t    = torch.from_numpy(psf_np  ).float()

        result = hogbom_clean(
            dirty_t, psf_t,
            gain          = args.gain,
            threshold     = args.threshold,
            n_iter        = args.n_max,
            energy_frac   = args.psf_energy_frac,
            min_sidelobes = args.min_sidelobes,
            device        = args.device,
        )
        model_np    = result["model"].cpu().numpy().astype(np.float32)
        residual_np = result["residual"].cpu().numpy().astype(np.float32)
        save_fits(model_np,    out_dir / "hogbom_model.fits")
        save_fits(residual_np, out_dir / "hogbom_residual.fits")
        print(f"Done  variant=hogbom  n_iter={result['n_iter']}  "
              f"converged={result['converged']}  peak_flux={result['peak_flux']:.4e}")
        return

    # ── Variants A / B ────────────────────────────────────────────────────────
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
        min_sidelobes = args.min_sidelobes,
        device        = args.device,
    )

    result = mc.deconvolve(args.dirty, args.psf, out_dir=out_dir)
    print(f"Done  variant={args.variant}  n_iter={result['n_iter']}  "
          f"converged={result['converged']}  peak_flux={result['peak_flux']:.4e}")


if __name__ == "__main__":
    main()
