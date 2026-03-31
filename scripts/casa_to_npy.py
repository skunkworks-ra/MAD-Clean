#!/usr/bin/env python3
"""
casa_to_npy.py — Export a CASA image to a numpy .npy file.

Squeezes degenerate Stokes/Frequency axes to produce a (H, W) float32 array.

Usage:
    /home/pjaganna/Software/data-analyst/.pixi/envs/default/bin/python \
        /home/pjaganna/Software/MAD-clean/scripts/casa_to_npy.py \
        --image SNR_G55_10s.MultiScale_spw3.image \
        --out models/SNR_G55.npy
"""

import argparse
import numpy as np
import casatools
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--image", required=True, help="Path to CASA image table")
    p.add_argument("--out",   required=True, help="Output .npy path")
    args = p.parse_args()

    ia = casatools.image()
    ia.open(args.image)
    data = ia.getchunk()
    ia.close()

    arr = np.squeeze(data).astype(np.float32)
    print(f"Shape after squeeze: {arr.shape}")
    print(f"min={arr.min():.4e}  max={arr.max():.4e}  mean={arr.mean():.4e}  std={arr.std():.4e}")

    np.save(args.out, arr)
    print(f"Saved → {args.out}")


if __name__ == "__main__":
    main()
