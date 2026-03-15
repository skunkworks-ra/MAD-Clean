#!/usr/bin/env python3
"""
run_deconvolve.py — CLI wrapper for MAD-CLEAN deconvolution.

Variant A  (patch OMP):
    python scripts/run_deconvolve.py --variant A \\
        --dirty dirty.fits --psf psf.fits \\
        --atoms models/cdl_filters_patch.npy \\
        --out_dir results/ \\
        [--gamma 0.1] [--n_max 500] [--epsilon_frac 0.01] [--detect_sigma 3.0] \\
        [--n_nonzero 5] [--stride 8] [--device cpu]

Variant B  (convolutional FISTA):
    python scripts/run_deconvolve.py --variant B \\
        --dirty dirty.fits --psf psf.fits \\
        --atoms models/cdl_filters_conv.npy \\
        --out_dir results/ \\
        [--lmbda 0.1] [--fista_iter 100] [--device cuda]

Inputs: FITS files or .npy arrays for dirty and PSF.
Outputs written to out_dir/: mad_clean_{A|B}_model.fits, mad_clean_{A|B}_residual.fits,
    mad_clean_{A|B}_rms_curve.npy
"""

import argparse
import sys
from pathlib import Path

# Make project root importable when running as a script
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from filters    import FilterBank       # noqa: E402
from detection  import IslandDetector   # noqa: E402
from solvers    import PatchSolver, ConvSolver  # noqa: E402
from deconvolver import MADClean        # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Run MAD-CLEAN deconvolution (Variant A or B).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--variant",      required=True, choices=["A", "B"])
    p.add_argument("--dirty",        required=True,
                   help="Dirty image — FITS path or .npy")
    p.add_argument("--psf",          required=True,
                   help="PSF — FITS path or .npy. Peak must be at image centre (CASA convention)")
    p.add_argument("--atoms",        required=True,
                   help="FilterBank .npy file produced by run_train.py")
    p.add_argument("--out_dir",      required=True,
                   help="Directory for output FITS and .npy files")
    p.add_argument("--device",       default="cpu", choices=["cpu", "cuda"])

    # Outer loop
    p.add_argument("--gamma",        type=float, default=0.1,
                   help="CLEAN loop gain")
    p.add_argument("--n_max",        type=int,   default=500,
                   help="Maximum major-cycle iterations")
    p.add_argument("--epsilon_frac", type=float, default=0.01,
                   help="Convergence as fraction of initial residual RMS")
    p.add_argument("--detect_sigma", type=float, default=3.0,
                   help="Island detection threshold in units of residual RMS")

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

    # ── load filter bank ──────────────────────────────────────────────────────
    atoms_path = Path(args.atoms)
    if not atoms_path.exists():
        print(f"ERROR: atoms file not found: {atoms_path}", file=sys.stderr)
        sys.exit(1)

    fb       = FilterBank.load(atoms_path, device=args.device)
    detector = IslandDetector(sigma_thresh=args.detect_sigma, device=args.device)

    if args.variant == "A":
        solver = PatchSolver(fb, n_nonzero=args.n_nonzero, stride=args.stride)
    else:
        solver = ConvSolver(fb, lmbda=args.lmbda, n_iter=args.fista_iter)

    mc = MADClean(
        fb, solver, detector,
        gamma        = args.gamma,
        epsilon_frac = args.epsilon_frac,
        n_max        = args.n_max,
        device       = args.device,
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    result = mc.deconvolve(args.dirty, args.psf, out_dir=out_dir)
    print(f"Done  n_iter={result['n_iter']}  "
          f"final RMS={result['rms_curve'][-1]:.4e}")


if __name__ == "__main__":
    main()
