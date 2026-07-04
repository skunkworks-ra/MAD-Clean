"""Extract source-centred training windows from the realistic (T-RECS) FITS
corpus into matched (dirty, psf, sky) memmap stacks for PSFCondFlow.

Why source-centred windows (not the fixed 128 patch grid, not single-source):
in real deconvolution a window holds whatever sources fall in it, neighbours
included.  The flow cleans a *window*, and one thresholded pass recovers every
source above the (per-window, sub-sidelobe) threshold; the outer major cycle
peels dynamic range across cycles.  So we train on honest multi-source windows
centred on real sources, not on artificially isolated patches.  This recovers
the full ~2200-source supply instead of the 234 windows the fixed grid yields.
See project_psf_condflow_validated.

Grid: model.fits is (FREQ, STOKES, 512, 512); dirty/psf are 768 (padded to
avoid aliasing) whose central 512 matches the model grid.  Truth = channel 0,
Stokes I (corpus built for MFS; MTMFS via flow comes later).  PSF is one per
field (shift-invariant here), centred, cropped to the window size.

Output stacks (same layout as casa_sim corpus_stacks, so PatchCorpusDataset
reads them unchanged):
    dirty.npy    (N, S, S) float32  signed window of the dirty image
    sky.npy      (N, S, S) float32  non-negative truth window (model ch0, Stokes I)
    psf.npy      (F, S, S) float32  one centred PSF per field, peak=1
    field_id.npy (N,)      int32    field index per window
    manifest.json

    pixi run -e gpu python scripts/extract_corpus_islands.py \
        --fits_dir /mnt/Data/Data/corpus_fits \
        --out /mnt/Data/Data/corpus_island_stacks/train
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
from astropy.io import fits
from scipy import ndimage


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Extract source-centred islands.")
    p.add_argument("--fits_dir", type=str, required=True)
    p.add_argument("--out",      type=str, required=True)
    p.add_argument("--window",   type=int, default=128,
                   help="Output window size (px). Matches PSFCondFlow size.")
    p.add_argument("--n_sigma",  type=float, default=5.0,
                   help="Detection floor in units of the field's dirty-image "
                        "noise (robust MAD sigma). A source is kept if its model "
                        "peak exceeds n_sigma*sigma_dirty — i.e. it is actually "
                        "detectable in the dirty. 5 is the conventional choice.")
    p.add_argument("--min_pix",  type=int, default=1,
                   help="Minimum connected area (px). 1 keeps single-pixel point "
                        "sources (the model is a delta-component image).")
    p.add_argument("--central", action="store_true",
                   help="Extract ONE window per field, centred on the brightest "
                        "source within --center_radius of the field centre (the "
                        "primary extended source the field is built around), "
                        "instead of one window per detected source.")
    p.add_argument("--center_radius", type=float, default=80.0,
                   help="Search radius (px, model grid) around the field centre "
                        "for the primary source in --central mode.")
    return p.parse_args(argv)


def _central_crop(img: np.ndarray, size: int) -> np.ndarray:
    """Centre-crop a 2D image to (size, size)."""
    h, w = img.shape
    r0 = (h - size) // 2
    c0 = (w - size) // 2
    return img[r0:r0 + size, c0:c0 + size]


def _window_at(img: np.ndarray, cy: int, cx: int, size: int) -> np.ndarray:
    """Crop a (size,size) window centred on (cy,cx), clamped to image bounds."""
    h, w = img.shape
    half = size // 2
    r0 = int(np.clip(cy - half, 0, h - size))
    c0 = int(np.clip(cx - half, 0, w - size))
    return img[r0:r0 + size, c0:c0 + size]


def main(argv=None):
    args = parse_args(argv)
    fits_dir = Path(args.fits_dir)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    S = args.window

    model_files = sorted(fits_dir.glob("corpus_field_*_model.fits"))
    print(f"[extract] {len(model_files)} fields with model.fits in {fits_dir}")

    dirties, skies, psfs, field_ids = [], [], [], []
    structure = np.ones((3, 3), dtype=bool)   # 8-connectivity
    n_skipped = 0

    for fi, mf in enumerate(model_files):
        stem = re.match(r"(corpus_field_\d+)_model\.fits", mf.name).group(1)
        df = fits_dir / f"{stem}_dirty.fits"
        pf = fits_dir / f"{stem}_psf.fits"
        if not (df.exists() and pf.exists()):
            n_skipped += 1
            continue

        model = fits.getdata(mf)                       # (FREQ, STOKES, 512, 512)
        sky_full = np.asarray(model[0, 0], dtype=np.float32)        # ch0, Stokes I
        sky_full = np.clip(sky_full, 0.0, None)        # truth is non-negative
        N512 = sky_full.shape[0]

        dirty_full = _central_crop(np.squeeze(fits.getdata(df)).astype(np.float32), N512)
        psf_full   = _central_crop(np.squeeze(fits.getdata(pf)).astype(np.float32), N512)
        psf_win    = _central_crop(psf_full, S)        # centred PSF, peak at centre

        if float(sky_full.max()) <= 0.0:
            continue
        # Detection floor = n_sigma * robust noise of THIS field's dirty image.
        sigma = 1.4826 * float(np.median(np.abs(dirty_full - np.median(dirty_full))))
        thr = args.n_sigma * sigma
        labels, nlab = ndimage.label(sky_full > thr, structure=structure)
        if nlab == 0:
            continue
        idx = np.arange(1, nlab + 1)
        sizes = ndimage.sum(np.ones_like(labels), labels, index=idx)
        peaks = ndimage.maximum(sky_full, labels, index=idx)
        fluxes = ndimage.sum(sky_full, labels, index=idx)
        centroids = ndimage.center_of_mass(sky_full, labels, index=idx)
        ctr = N512 / 2.0

        if args.central:
            # ONE window per field, centred on the brightest source within
            # center_radius of the field centre — the primary (extended) source
            # the field was built around.  Neighbours fall in the crop naturally.
            best_k, best_flux = -1, -1.0
            for k in range(nlab):
                if sizes[k] < args.min_pix or peaks[k] < thr:
                    continue
                cy, cx = centroids[k]
                if (cy - ctr) ** 2 + (cx - ctr) ** 2 > args.center_radius ** 2:
                    continue
                if fluxes[k] > best_flux:
                    best_flux, best_k = fluxes[k], k
            if best_k < 0:
                continue                                   # no source near centre
            cy, cx = (int(round(centroids[best_k][0])),
                      int(round(centroids[best_k][1])))
            skies.append(_window_at(sky_full,   cy, cx, S))
            dirties.append(_window_at(dirty_full, cy, cx, S))
            field_ids.append(fi)
            psfs.append(psf_win)
        else:
            for k in range(nlab):
                if sizes[k] < args.min_pix or peaks[k] < thr:
                    continue
                cy, cx = int(round(centroids[k][0])), int(round(centroids[k][1]))
                skies.append(_window_at(sky_full,   cy, cx, S))
                dirties.append(_window_at(dirty_full, cy, cx, S))
                field_ids.append(fi)
            psfs.append(psf_win)

    dirty = np.stack(dirties).astype(np.float32)
    sky   = np.stack(skies).astype(np.float32)
    psf   = np.stack(psfs).astype(np.float32)
    field_id = np.asarray(field_ids, dtype=np.int32)

    np.save(out_dir / "dirty.npy", dirty)
    np.save(out_dir / "sky.npy", sky)
    np.save(out_dir / "psf.npy", psf)
    np.save(out_dir / "field_id.npy", field_id)
    manifest = {
        "n_islands": int(dirty.shape[0]),
        "n_fields": int(psf.shape[0]),
        "window": S,
        "n_sigma": args.n_sigma,
        "min_pix": args.min_pix,
        "source": str(fits_dir),
        "note": "source-centred windows, neighbours included; truth=ch0 Stokes I",
    }
    with open(out_dir / "manifest.json", "w") as fh:
        json.dump(manifest, fh, indent=2)

    print(f"[extract] islands={dirty.shape[0]}  fields={psf.shape[0]}  "
          f"skipped_fields(missing dirty/psf)={n_skipped}")
    print(f"[extract] dirty {dirty.shape}  sky {sky.shape}  psf {psf.shape}")
    print(f"[extract] wrote stacks to {out_dir}")


if __name__ == "__main__":
    main()
