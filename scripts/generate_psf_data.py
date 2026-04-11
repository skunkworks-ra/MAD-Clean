#!/usr/bin/env python3
"""
Generate synthetic PSF-conditioned training data in parallel.

One .npz file per PSF written to crumb_data/psf_pairs/.
Each file contains all clean/dirty pairs for that PSF.
An index.npz records the train/test split and PSF parameters.

Usage
-----
    pixi run -e gpu python scripts/generate_psf_data.py
    pixi run -e gpu python scripts/generate_psf_data.py --n_workers 8
"""

import argparse
import numpy as np
from astropy.modeling.models import Gaussian2D
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path


# ── PSF family ────────────────────────────────────────────────────────────────

def make_psf(fwhm_maj: float, fwhm_min: float, pa_deg: float,
             size: int = 150) -> np.ndarray:
    sigma_maj = fwhm_maj / (2 * np.sqrt(2 * np.log(2)))
    sigma_min = fwhm_min / (2 * np.sqrt(2 * np.log(2)))
    cy, cx    = size // 2, size // 2
    y, x      = np.mgrid[0:size, 0:size]
    g = Gaussian2D(
        amplitude = 1.0,
        x_mean    = cx, y_mean   = cy,
        x_stddev  = sigma_maj, y_stddev = sigma_min,
        theta     = np.deg2rad(pa_deg),
    )
    psf = g(x, y).astype(np.float32)
    psf /= psf.max()
    return psf


def build_psf_family() -> list[dict]:
    psfs = []
    for fwhm_maj in [1.5, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0]:
        for ellipticity in [1.0, 1.5, 2.0, 3.0]:
            for pa in [0, 30, 60, 90, 120, 150]:
                psfs.append(dict(
                    fwhm_maj = fwhm_maj,
                    fwhm_min = fwhm_maj / ellipticity,
                    pa       = pa,
                ))
    return psfs


# ── Per-PSF worker ────────────────────────────────────────────────────────────

def _generate_one(args: tuple) -> str:
    """Worker: generate all dirty images for one PSF and save to disk."""
    p_idx, params, cleans, out_dir, noise_std, seed = args

    psf   = make_psf(params["fwhm_maj"], params["fwhm_min"], params["pa"])
    H, W  = cleans.shape[1], cleans.shape[2]
    rng   = np.random.default_rng(seed + p_idx)

    psf_fft = np.fft.rfft2(np.fft.ifftshift(psf))
    dirty   = np.fft.irfft2(
        np.fft.rfft2(cleans, axes=(1, 2)) * psf_fft[None],
        s=(H, W), axes=(1, 2),
    ).astype(np.float32)
    dirty  += rng.standard_normal(dirty.shape).astype(np.float32) * noise_std

    import torch
    out_path = Path(out_dir) / f"psf_{p_idx:04d}.pt"
    torch.save({
        "dirty" : torch.from_numpy(dirty),
        "clean" : torch.from_numpy(cleans),
        "psf"   : torch.from_numpy(psf),
        "params": torch.tensor([params["fwhm_maj"], params["fwhm_min"], params["pa"]]),
    }, out_path)
    return f"PSF {p_idx:4d}  fwhm={params['fwhm_maj']:.1f}×{params['fwhm_min']:.1f}  pa={params['pa']:3d}°  → {out_path.name}"


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--crumb",      default="crumb_data/crumb_preprocessed.npz")
    p.add_argument("--out_dir",    default="crumb_data/psf_pairs")
    p.add_argument("--noise_std",  type=float, default=0.05)
    p.add_argument("--n_workers",  type=int,   default=4)
    p.add_argument("--test_frac",  type=float, default=0.2,
                   help="Fraction of PSFs held out for testing")
    p.add_argument("--seed",       type=int,   default=42)
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load clean images once — shared read-only across workers
    print(f"Loading clean images from {args.crumb} …")
    cleans = np.load(args.crumb)["images"].astype(np.float32)
    N_img  = len(cleans)
    print(f"  {N_img} images  shape={cleans.shape[1]}×{cleans.shape[2]}")

    psf_params = build_psf_family()
    N_psf      = len(psf_params)

    # 80/20 train/test split on PSFs
    rng          = np.random.default_rng(args.seed)
    shuffled_idx = rng.permutation(N_psf)
    n_test       = max(1, int(N_psf * args.test_frac))
    test_set     = set(shuffled_idx[:n_test].tolist())
    train_mask   = np.array([i not in test_set for i in range(N_psf)], dtype=bool)

    print(f"PSF family: {N_psf} PSFs  ({train_mask.sum()} train / {(~train_mask).sum()} test)")
    print(f"Generating with {args.n_workers} workers …\n")

    worker_args = [
        (i, psf_params[i], cleans, str(out_dir), args.noise_std, args.seed)
        for i in range(N_psf)
    ]

    with ProcessPoolExecutor(max_workers=args.n_workers) as pool:
        futures = {pool.submit(_generate_one, a): a[0] for a in worker_args}
        for fut in as_completed(futures):
            print(f"  ✓ {fut.result()}", flush=True)

    # Save index
    index_path = out_dir / "index.npz"
    np.savez(
        index_path,
        n_psf      = np.int32(N_psf),
        n_img      = np.int32(N_img),
        train_mask = train_mask,           # (N_psf,) bool
        noise_std  = np.float32(args.noise_std),
    )
    print(f"\nIndex saved → {index_path}")
    print(f"Total pairs: {N_psf * N_img:,}  "
          f"({train_mask.sum() * N_img:,} train / {(~train_mask).sum() * N_img:,} test)")


if __name__ == "__main__":
    main()
