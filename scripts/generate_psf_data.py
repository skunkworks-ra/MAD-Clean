#!/usr/bin/env python3
"""
Generate synthetic PSF family + simulated dirty/clean pairs.

PSFs: 2D elliptical Gaussians with varying FWHM, PA, and ellipticity.
Data: each clean CRUMB image convolved with each PSF + noise.

Output: crumb_data/psf_pairs.npz
  clean    : (N, 150, 150)  float32  Jy/pixel  [0, 1]
  dirty    : (N, 150, 150)  float32  Jy/beam
  psf      : (N, 150, 150)  float32  peak=1
  psf_idx  : (N,)           int32    which PSF was used
  noise_std: float32

80/20 train/test split by PSF — held-out PSFs are never seen during training.
Split indices saved as train_mask (N,) bool.
"""

import numpy as np
from astropy.modeling.models import Gaussian2D
from pathlib import Path

# ── PSF family ────────────────────────────────────────────────────────────────

def make_psf(fwhm_maj: float, fwhm_min: float, pa_deg: float,
             size: int = 150) -> np.ndarray:
    """
    2D elliptical Gaussian PSF, peak=1, centred at image centre.
    pa_deg: position angle in degrees (0 = aligned with x-axis).
    """
    sigma_maj = fwhm_maj / (2 * np.sqrt(2 * np.log(2)))
    sigma_min = fwhm_min / (2 * np.sqrt(2 * np.log(2)))
    pa_rad    = np.deg2rad(pa_deg)

    cy, cx = size // 2, size // 2
    y, x   = np.mgrid[0:size, 0:size]

    g = Gaussian2D(
        amplitude  = 1.0,
        x_mean     = cx,
        y_mean     = cy,
        x_stddev   = sigma_maj,
        y_stddev   = sigma_min,
        theta      = pa_rad,
    )
    psf = g(x, y).astype(np.float32)
    psf /= psf.max()
    return psf


def make_psf_family(seed: int = 0) -> list[dict]:
    """
    Return a list of PSF parameter dicts covering a range of
    FWHM (1–8 px), ellipticity (1.0–3.0), and PA (0–165°).
    """
    rng  = np.random.default_rng(seed)
    psfs = []

    fwhm_majors  = [1.5, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0]
    ellipticities = [1.0, 1.5, 2.0, 3.0]
    pas           = [0, 30, 60, 90, 120, 150]

    for fwhm_maj in fwhm_majors:
        for ell in ellipticities:
            for pa in pas:
                fwhm_min = fwhm_maj / ell
                psfs.append(dict(fwhm_maj=fwhm_maj, fwhm_min=fwhm_min, pa=pa))

    rng.shuffle(psfs)
    return psfs


# ── Convolution ───────────────────────────────────────────────────────────────

def convolve(clean: np.ndarray, psf: np.ndarray) -> np.ndarray:
    """dirty = psf(peak=1) ⊛ clean  via FFT."""
    H, W    = clean.shape
    psf_fft = np.fft.rfft2(np.fft.ifftshift(psf))
    dirty   = np.fft.irfft2(np.fft.rfft2(clean) * psf_fft, s=(H, W))
    return dirty.astype(np.float32)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    noise_std = 0.05
    seed      = 42
    rng       = np.random.default_rng(seed)

    # Load clean images (already peak-normalised [0,1])
    crumb = np.load("crumb_data/crumb_preprocessed.npz")
    cleans = crumb["images"].astype(np.float32)          # (2100, 150, 150)
    N_img  = len(cleans)
    print(f"Loaded {N_img} clean images  shape=150×150")

    # Build PSF family
    psf_params = make_psf_family(seed=seed)
    N_psf      = len(psf_params)
    print(f"PSF family: {N_psf} PSFs")

    # 80/20 split on PSFs
    n_test_psf  = max(1, N_psf // 5)
    test_psf_idx = set(range(N_psf - n_test_psf, N_psf))
    print(f"  Train PSFs: {N_psf - n_test_psf}   Test PSFs: {n_test_psf}")

    # Build arrays
    all_clean   = []
    all_dirty   = []
    all_psf     = []
    all_psf_idx = []
    all_train   = []

    for p_idx, params in enumerate(psf_params):
        psf    = make_psf(params["fwhm_maj"], params["fwhm_min"], params["pa"])
        is_train = p_idx not in test_psf_idx

        for c_idx, clean in enumerate(cleans):
            dirty = convolve(clean, psf)
            dirty += rng.standard_normal(dirty.shape).astype(np.float32) * noise_std

            all_clean.append(clean)
            all_dirty.append(dirty)
            all_psf.append(psf)
            all_psf_idx.append(p_idx)
            all_train.append(is_train)

        print(f"  PSF {p_idx+1:3d}/{N_psf}  fwhm={params['fwhm_maj']:.1f}×{params['fwhm_min']:.1f}  "
              f"pa={params['pa']:3d}°  {'train' if is_train else 'TEST'}", flush=True)

    clean_arr   = np.stack(all_clean).astype(np.float32)
    dirty_arr   = np.stack(all_dirty).astype(np.float32)
    psf_arr     = np.stack(all_psf).astype(np.float32)
    psf_idx_arr = np.array(all_psf_idx, dtype=np.int32)
    train_mask  = np.array(all_train,   dtype=bool)

    N_total = len(clean_arr)
    print(f"\nTotal pairs: {N_total}  ({train_mask.sum()} train / {(~train_mask).sum()} test)")
    print(f"dirty range: [{dirty_arr.min():.3f}, {dirty_arr.max():.3f}]")

    out = Path("crumb_data/psf_pairs.npz")
    np.savez(
        out,
        clean      = clean_arr,
        dirty      = dirty_arr,
        psf        = psf_arr,
        psf_idx    = psf_idx_arr,
        train_mask = train_mask,
        noise_std  = np.float32(noise_std),
    )
    print(f"Saved → {out}  ({out.stat().st_size / 1e9:.2f} GB)")


if __name__ == "__main__":
    main()
