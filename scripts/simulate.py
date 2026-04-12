#!/usr/bin/env python3
"""
scripts/simulate.py — simulate dirty radio observations from clean sky images.

Usage
-----
# With a real PSF (.fits, .npy, or .npz):
mad-simulate --data crumb_data/crumb_preprocessed.npz \\
    --psf models/psf.fits --noise_std 0.05 --out crumb_data/flow_pairs.npz

# With a synthetic Gaussian PSF (FWHM in pixels):
mad-simulate --data crumb_data/crumb_preprocessed.npz \\
    --psf_fwhm 3.0 --noise_std 0.05 --out crumb_data/flow_pairs.npz
"""

import argparse
import sys
from pathlib import Path

from mad_clean.data import SimulateObservations


def main() -> None:
    p = argparse.ArgumentParser(
        description="Simulate dirty radio observations from clean sky images.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--data",      required=True,
                   help="Path to clean images .npz (must have 'images' key)")
    p.add_argument("--out",       required=True,
                   help="Output .npz path (clean, dirty, psf, noise_std)")
    p.add_argument("--psf",       default=None,
                   help="PSF file path (.fits, .npy, or .npz). Mutually exclusive with --psf_fwhm.")
    p.add_argument("--psf_fwhm",  type=float, default=None,
                   help="Synthetic Gaussian PSF FWHM in pixels. Mutually exclusive with --psf.")
    p.add_argument("--noise_std",     type=float, default=0.05,
                   help="Gaussian noise std added to each dirty image")
    p.add_argument("--seed",          type=int,   default=42)
    p.add_argument("--no_normalise",  action="store_true",
                   help="Skip per-image peak normalisation. Use when training was done "
                        "on raw flux (Option C / physical units).")
    args = p.parse_args()

    if args.psf is None and args.psf_fwhm is None:
        p.error("Provide either --psf (file) or --psf_fwhm (Gaussian FWHM in pixels)")
    if args.psf is not None and args.psf_fwhm is not None:
        p.error("--psf and --psf_fwhm are mutually exclusive")

    sim = SimulateObservations(
        psf_fwhm    = args.psf_fwhm,
        psf_path    = args.psf,
        noise_std   = args.noise_std,
        seed        = args.seed,
        normalise   = not args.no_normalise,
    )
    sim.run(args.data, out=args.out)


if __name__ == "__main__":
    main()
