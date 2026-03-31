#!/usr/bin/env python3
"""
simulate_observations.py — simulate dirty radio observations from clean sky images.

Applies a PSF (point spread function) to each clean image via FFT convolution
and adds Gaussian noise. Produces a unified training npz usable by all variants:

    Variant A/B: use data["clean"]
    Variant C:   use data["dirty"] + data["clean"]

Usage
-----
# With a real PSF (FITS or .npy):
python scripts/simulate_observations.py \\
    --data crumb_data/crumb_preprocessed.npz \\
    --psf  models/psf.fits \\
    --noise_std 0.05 \\
    --out  crumb_data/flow_pairs.npz

# With a synthetic Gaussian PSF (FWHM in pixels):
python scripts/simulate_observations.py \\
    --data crumb_data/crumb_preprocessed.npz \\
    --psf_fwhm 3.0 \\
    --noise_std 0.05 \\
    --out  crumb_data/flow_pairs.npz
"""

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _make_gaussian_psf(fwhm: float, size: int) -> np.ndarray:
    """
    Build a 2D Gaussian PSF of given FWHM (pixels), centred at (size//2, size//2).

    The PSF is normalised to sum to 1.
    """
    sigma = fwhm / (2.0 * np.sqrt(2.0 * np.log(2.0)))
    cy, cx = size // 2, size // 2
    ys = np.arange(size, dtype=np.float32) - cy
    xs = np.arange(size, dtype=np.float32) - cx
    yy, xx = np.meshgrid(ys, xs, indexing="ij")
    psf = np.exp(-(xx ** 2 + yy ** 2) / (2.0 * sigma ** 2)).astype(np.float32)
    psf /= psf.sum()
    return psf


def _load_psf(path: str | Path, target_shape: tuple) -> np.ndarray:
    """
    Load PSF from FITS or .npy file. Crop or pad to target_shape (H, W).
    Normalise to sum=1.
    """
    path = Path(path)
    if path.suffix.lower() in (".fits", ".fit"):
        from astropy.io import fits
        with fits.open(path) as hdul:
            data = hdul[0].data.astype(np.float32)
            # FITS may have extra axes (e.g. Stokes, frequency)
            while data.ndim > 2:
                data = data[0]
    elif path.suffix.lower() == ".npy":
        data = np.load(path).astype(np.float32)
    else:
        raise ValueError(f"Unsupported PSF format: {path.suffix} — use .fits or .npy")

    H, W = target_shape
    h, w = data.shape

    # Centre-crop if larger
    if h > H or w > W:
        ch = (h - H) // 2
        cw = (w - W) // 2
        data = data[ch:ch + H, cw:cw + W]

    # Zero-pad if smaller
    if data.shape != (H, W):
        out = np.zeros((H, W), dtype=np.float32)
        ph = (H - data.shape[0]) // 2
        pw = (W - data.shape[1]) // 2
        out[ph:ph + data.shape[0], pw:pw + data.shape[1]] = data
        data = out

    psf_sum = data.sum()
    if psf_sum > 1e-12:
        data /= psf_sum
    return data


def _convolve_psf(images: np.ndarray, psf: np.ndarray) -> np.ndarray:
    """
    Convolve each image in (N, H, W) with the PSF (H, W) via FFT.

    The PSF peak is assumed to be at (H//2, W//2). We ifftshift before
    taking the FFT so the peak moves to (0, 0), consistent with the
    circular-convolution convention used by MADClean.
    """
    H, W = images.shape[1], images.shape[2]
    psf_shifted = np.fft.ifftshift(psf)
    psf_fft     = np.fft.rfft2(psf_shifted, s=(H, W))
    imgs_fft    = np.fft.rfft2(images, axes=(1, 2))
    dirty_fft   = imgs_fft * psf_fft[None, :, :]
    dirty       = np.fft.irfft2(dirty_fft, s=(H, W), axes=(1, 2))
    return dirty.astype(np.float32)


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
                   help="PSF file path (.fits or .npy). Mutually exclusive with --psf_fwhm")
    p.add_argument("--psf_fwhm",  type=float, default=None,
                   help="Synthetic Gaussian PSF FWHM in pixels. Mutually exclusive with --psf")
    p.add_argument("--noise_std", type=float, default=0.05,
                   help="Gaussian noise std added to each dirty image (in normalised units)")
    p.add_argument("--seed",      type=int,   default=42)
    args = p.parse_args()

    if args.psf is None and args.psf_fwhm is None:
        p.error("Provide either --psf (file) or --psf_fwhm (Gaussian FWHM in pixels)")
    if args.psf is not None and args.psf_fwhm is not None:
        p.error("--psf and --psf_fwhm are mutually exclusive")

    # ── load clean images ──────────────────────────────────────────────────
    data_path = Path(args.data)
    if not data_path.exists():
        print(f"ERROR: data file not found: {data_path}", file=sys.stderr)
        sys.exit(1)

    raw    = np.load(data_path)
    clean  = raw["images"].astype(np.float32)
    N, H, W = clean.shape
    print(f"Loaded {N} clean images  shape={H}×{W}")

    # ── per-image normalise clean images (zero mean, unit std) ────────────
    img_mean = clean.mean(axis=(1, 2), keepdims=True)
    img_std  = clean.std(axis=(1, 2),  keepdims=True) + 1e-8
    clean_n  = (clean - img_mean) / img_std

    # ── build PSF ─────────────────────────────────────────────────────────
    if args.psf_fwhm is not None:
        psf = _make_gaussian_psf(args.psf_fwhm, size=max(H, W))
        psf = psf[:H, :W]
        print(f"Synthetic Gaussian PSF  FWHM={args.psf_fwhm}px  shape={psf.shape}")
    else:
        psf = _load_psf(args.psf, target_shape=(H, W))
        print(f"Loaded PSF from {args.psf}  shape={psf.shape}")

    # ── simulate dirty images ─────────────────────────────────────────────
    dirty_n = _convolve_psf(clean_n, psf)

    rng      = np.random.default_rng(args.seed)
    noise    = rng.standard_normal(dirty_n.shape).astype(np.float32) * args.noise_std
    dirty_n += noise

    print(f"Dirty images  noise_std={args.noise_std}  "
          f"dirty range=[{dirty_n.min():.3f}, {dirty_n.max():.3f}]  "
          f"clean range=[{clean_n.min():.3f}, {clean_n.max():.3f}]")

    # ── save ──────────────────────────────────────────────────────────────
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        out_path,
        clean     = clean_n,
        dirty     = dirty_n,
        psf       = psf,
        noise_std = np.float32(args.noise_std),
    )
    print(f"Saved → {out_path}  keys: clean {clean_n.shape}, dirty {dirty_n.shape}, psf {psf.shape}")


if __name__ == "__main__":
    main()
