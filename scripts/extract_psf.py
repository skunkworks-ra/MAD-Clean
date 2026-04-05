#!/usr/bin/env python3
"""
extract_psf.py — Extract a PSF from a CASA image table and save as .npz.

Requires casatools — run with the data-analyst pixi environment:
    pixi run -e data-analyst python scripts/extract_psf.py \\
        --psf /path/to/dirty.psf \\
        --out models/psf.npz \\
        [--size 150]

The extracted PSF is peak-normalised (peak = 1.0, CASA convention).
--size crops or zero-pads the PSF to a square of that side length.
It should match the training image size (default 150).
"""

import argparse
import sys
from pathlib import Path

import numpy as np


def extract(psf_path: str, out_path: str, size: int) -> None:
    try:
        import casatools
    except ImportError:
        print(
            "ERROR: casatools not found.\n"
            "Run with the data-analyst pixi environment:\n"
            "  pixi run -e data-analyst python scripts/extract_psf.py ...",
            file=sys.stderr,
        )
        sys.exit(1)

    ia = casatools.image()
    ia.open(psf_path)
    data = ia.getchunk()
    ia.close()

    # Squeeze degenerate axes (Stokes, Freq) → (H, W)
    arr = np.squeeze(data).astype(np.float32)
    if arr.ndim != 2:
        print(f"ERROR: expected 2D after squeeze, got shape {arr.shape}", file=sys.stderr)
        sys.exit(1)

    print(f"Raw PSF  shape={arr.shape}  peak={arr.max():.6f}  sum={arr.sum():.6f}")

    # Peak-normalise: CASA convention is peak=1 at beam centre
    peak = float(arr.max())
    if peak < 1e-12:
        print("ERROR: PSF peak is essentially zero — check the image.", file=sys.stderr)
        sys.exit(1)
    arr /= peak

    # Crop or zero-pad to (size, size)
    H, W = arr.shape
    if H != size or W != size:
        # Centre-crop if larger
        if H > size:
            ch = (H - size) // 2
            arr = arr[ch : ch + size, :]
        if W > size:
            cw = (W - size) // 2
            arr = arr[:, cw : cw + size]
        # Zero-pad if smaller
        H, W = arr.shape
        if H < size or W < size:
            out = np.zeros((size, size), dtype=np.float32)
            ph = (size - H) // 2
            pw = (size - W) // 2
            out[ph : ph + H, pw : pw + W] = arr
            arr = out

    out_p = Path(out_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_p, psf=arr)
    print(
        f"Saved  → {out_p}\n"
        f"  shape={arr.shape}  peak={arr.max():.6f}  sum={arr.sum():.6f}"
    )


def main() -> None:
    p = argparse.ArgumentParser(
        description="Extract VLA PSF from CASA image table → .npz (peak-normalised).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--psf",  required=True, help="CASA image table path (e.g. dirty.psf)")
    p.add_argument("--out",  required=True, help="Output .npz path")
    p.add_argument("--size", type=int, default=150,
                   help="Crop/pad PSF to this square size (must match training image size)")
    args = p.parse_args()

    extract(args.psf, args.out, args.size)


if __name__ == "__main__":
    main()
