"""Patchify corpus FITS files into PatchCorpusDataset stacks.

Reads dirty.fits / model.fits / psf.fits for each field, centre-crops
the 768-px dirty and PSF to the central 512-px model region, tiles into
128x128 non-overlapping patches, and writes the memmap stacks expected
by PatchCorpusDataset.

Usage:
    pixi run python scripts/patchify_fits.py \
        --fits_dir /mnt/Data/Data/corpus_fits \
        --out_dir  /mnt/Data/Data/corpus_stacks \
        --patch_size 128 \
        --val_fraction 0.1
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
from astropy.io import fits

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

PATCH = 128
MODEL_SIZE = 512   # model image is always 512x512
DIRTY_SIZE = 768   # dirty/psf are 768x768; centre 512 aligns with model
CROP_OFF   = (DIRTY_SIZE - MODEL_SIZE) // 2  # 128


def centre_crop(arr2d, size=MODEL_SIZE):
    """Crop centre size×size from a 2-D array."""
    h, w = arr2d.shape
    r0 = (h - size) // 2
    c0 = (w - size) // 2
    return arr2d[r0:r0 + size, c0:c0 + size]


def tile_patches(field_2d, patch=PATCH):
    """Tile a square field into non-overlapping patch×patch blocks.
    Returns (N, patch, patch) where N = (field/patch)^2."""
    n = field_2d.shape[0] // patch
    patches = []
    for r in range(n):
        for c in range(n):
            patches.append(field_2d[r*patch:(r+1)*patch, c*patch:(c+1)*patch])
    return np.stack(patches, axis=0)


def squeeze_model(data):
    """Squeeze CASA cube to 2-D Stokes-I MFS image."""
    while data.ndim > 2:
        data = data[0]
    return data.astype(np.float32)


def config_from_header(header):
    """Try to extract VLA config (0=A,1=B,2=C,3=D) from FITS header."""
    for key in ("TELESCOP", "OBSERVER", "OBJECT", "INSTRUME"):
        val = str(header.get(key, "")).upper()
        for i, cfg in enumerate(["VLA-A", "VLA-B", "VLA-C", "VLA-D"]):
            if cfg in val:
                return i
        for i, cfg in enumerate(["-A", "-B", "-C", "-D"]):
            if cfg in val:
                return i
    return 2   # default C-config


def find_fields(fits_dir):
    dirty_files = sorted(Path(fits_dir).glob("*dirty.fits"))
    fields = []
    for df in dirty_files:
        stem = df.stem.replace("_dirty", "")
        mf = df.parent / f"{stem}_model.fits"
        pf = df.parent / f"{stem}_psf.fits"
        if mf.exists() and pf.exists():
            fields.append((stem, df, mf, pf))
        else:
            print(f"[skip] {stem}: missing model or psf")
    return fields


def build_stacks(fields, out_dir, patch=PATCH):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    n_fields = len(fields)
    n_per_field = (MODEL_SIZE // patch) ** 2   # 16
    N = n_fields * n_per_field

    dirty_stack  = np.zeros((N, patch, patch), dtype=np.float32)
    sky_stack    = np.zeros((N, patch, patch), dtype=np.float32)
    psf_stack    = np.zeros((n_fields, patch, patch), dtype=np.float32)
    field_id     = np.zeros(N, dtype=np.int32)
    config_idx   = np.zeros(n_fields, dtype=np.int32)
    manifest_fields = []

    for fi, (stem, df, mf, pf) in enumerate(fields):
        with fits.open(df) as hdul:
            dirty2d = centre_crop(hdul[0].data.squeeze().astype(np.float32))
            cfg = config_from_header(hdul[0].header)
        with fits.open(mf) as hdul:
            sky2d = squeeze_model(hdul[0].data)
            if sky2d.shape != (MODEL_SIZE, MODEL_SIZE):
                sky2d = centre_crop(sky2d, MODEL_SIZE)
            sky2d = np.clip(sky2d, 0, None)
        with fits.open(pf) as hdul:
            psf2d = centre_crop(hdul[0].data.squeeze().astype(np.float32))

        # PSF: centre 128×128 patch, normalise to peak=1.
        pc = MODEL_SIZE // 2
        psf_patch = psf2d[pc - patch//2:pc + patch//2,
                          pc - patch//2:pc + patch//2].copy()
        peak = psf_patch.max()
        if peak > 0:
            psf_patch /= peak

        dirty_patches = tile_patches(dirty2d, patch)
        sky_patches   = tile_patches(sky2d,   patch)

        start = fi * n_per_field
        end   = start + n_per_field
        dirty_stack[start:end] = dirty_patches
        sky_stack[start:end]   = sky_patches
        psf_stack[fi]          = psf_patch
        field_id[start:end]    = fi
        config_idx[fi]         = cfg

        sky_peak = float(sky2d.max())
        dirty_peak = float(np.abs(dirty2d).max())
        print(f"  field {fi:03d} {stem}: cfg={cfg} "
              f"dirty_peak={dirty_peak:.4f} sky_peak={sky_peak:.4f}")

        manifest_fields.append({
            "field_idx": fi, "stem": stem,
            "vla_config": ["A", "B", "C", "D"][cfg],
            "config_idx": int(cfg),
            "n_patches": n_per_field,
            "patch_start": int(start), "patch_end": int(end),
        })

    np.save(out_dir / "dirty.npy",     dirty_stack)
    np.save(out_dir / "sky.npy",       sky_stack)
    np.save(out_dir / "psf.npy",       psf_stack)
    np.save(out_dir / "field_id.npy",  field_id)
    np.save(out_dir / "config_idx.npy",config_idx)

    manifest = {
        "n_fields": n_fields, "n_patches": N,
        "patches_per_field": n_per_field,
        "patch_size": patch, "stride": patch,
        "fields": manifest_fields,
    }
    with open(out_dir / "manifest.json", "w") as fh:
        json.dump(manifest, fh, indent=2)

    print(f"[done] {N} patches from {n_fields} fields -> {out_dir}")
    return manifest


def parse_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--fits_dir",      required=True)
    p.add_argument("--out_dir",       required=True)
    p.add_argument("--patch_size",    type=int, default=128)
    p.add_argument("--val_fraction",  type=float, default=0.1,
                   help="Fraction of fields held out for validation.")
    p.add_argument("--seed",          type=int, default=42)
    return p.parse_args(argv)


def run(args):
    fields = find_fields(args.fits_dir)
    print(f"[patchify] found {len(fields)} fields in {args.fits_dir}")

    rng = np.random.default_rng(args.seed)
    idx = rng.permutation(len(fields))
    n_val = max(1, int(len(fields) * args.val_fraction))
    val_idx   = set(idx[:n_val].tolist())
    train_idx = set(idx[n_val:].tolist())

    train_fields = [fields[i] for i in sorted(train_idx)]
    val_fields   = [fields[i] for i in sorted(val_idx)]

    print(f"[patchify] train={len(train_fields)} val={len(val_fields)}")

    train_dir = Path(args.out_dir) / "train"
    val_dir   = Path(args.out_dir) / "val"

    print("[patchify] building train stacks...")
    build_stacks(train_fields, train_dir, args.patch_size)
    print("[patchify] building val stacks...")
    build_stacks(val_fields,   val_dir,   args.patch_size)


if __name__ == "__main__":
    run(parse_args())
