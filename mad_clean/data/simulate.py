"""
mad_clean.data.simulate
=======================
SimulateObservations — simulate dirty radio observations from clean sky images.

Applies a PSF (point spread function) to each clean image via FFT convolution
and adds Gaussian noise. Produces a unified training npz usable by all variants:

    Variant A:   use data["clean"]
    Variant B:   use data["dirty"] + data["psf"]
    Variant C:   use data["dirty"] + data["clean"]

Usage
-----
    sim = SimulateObservations(psf_fwhm=3.0, noise_std=0.05)
    sim.run("crumb_data/crumb_preprocessed.npz", out="crumb_data/flow_pairs.npz")

    # Or with a real PSF file:
    sim = SimulateObservations(psf_path="models/psf.fits", noise_std=0.05)
    sim.run("crumb_data/crumb_preprocessed.npz", out="crumb_data/flow_pairs.npz")
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ..normalise import ImageNormaliser

__all__ = ["SimulateObservations"]


def _make_gaussian_psf(fwhm: float, size: int) -> np.ndarray:
    """
    Build a 2D Gaussian PSF of given FWHM (pixels), centred at (size//2, size//2).
    Peak-normalised to 1.0 (CASA convention: dirty ≈ PSF-blurred clean).
    """
    sigma = fwhm / (2.0 * np.sqrt(2.0 * np.log(2.0)))
    cy, cx = size // 2, size // 2
    ys = np.arange(size, dtype=np.float32) - cy
    xs = np.arange(size, dtype=np.float32) - cx
    yy, xx = np.meshgrid(ys, xs, indexing="ij")
    psf = np.exp(-(xx ** 2 + yy ** 2) / (2.0 * sigma ** 2)).astype(np.float32)
    psf /= psf.max()
    return psf


def _load_psf(path: str | Path, target_shape: tuple) -> np.ndarray:
    """
    Load PSF from .npz, .npy, or FITS file. Crop or pad to target_shape (H, W).
    Peak-normalised to 1.0 (CASA convention).
    """
    path = Path(path)
    if path.suffix.lower() == ".npz":
        data = np.load(path)["psf"].astype(np.float32)
    elif path.suffix.lower() == ".npy":
        data = np.load(path).astype(np.float32)
    elif path.suffix.lower() in (".fits", ".fit"):
        from astropy.io import fits
        with fits.open(path) as hdul:
            data = hdul[0].data.astype(np.float32)
            while data.ndim > 2:
                data = data[0]
    else:
        raise ValueError(f"Unsupported PSF format: {path.suffix} — use .npz, .npy, or .fits")

    H, W = target_shape
    h, w = data.shape

    if h > H or w > W:
        ch = (h - H) // 2
        cw = (w - W) // 2
        data = data[ch:ch + H, cw:cw + W]

    if data.shape != (H, W):
        out = np.zeros((H, W), dtype=np.float32)
        ph = (H - data.shape[0]) // 2
        pw = (W - data.shape[1]) // 2
        out[ph:ph + data.shape[0], pw:pw + data.shape[1]] = data
        data = out

    peak = data.max()
    if peak > 1e-12:
        data /= peak
    return data


def _next_pow2(n: int) -> int:
    """Smallest power of 2 that is ≥ n."""
    return 1 << (n - 1).bit_length()


def _convolve_psf(images: np.ndarray, psf: np.ndarray) -> np.ndarray:
    """
    Convolve each image in (N, H, W) with the PSF (H, W) via padded FFT.

    Padding to the next power-of-2 ≥ 2×max(H,W) converts circular convolution
    to linear convolution — no wrap-around artifacts at image boundaries.
    ifftshift moves the PSF peak from (H//2, W//2) to (0, 0) before the FFT,
    consistent with MADClean's PSF convention (peak=1 at centre).
    Output is cropped back to (H, W).
    """
    H, W  = images.shape[1], images.shape[2]
    pad   = _next_pow2(2 * max(H, W))

    psf_shifted = np.fft.ifftshift(psf)
    psf_fft     = np.fft.rfft2(psf_shifted, s=(pad, pad))
    imgs_fft    = np.fft.rfft2(images, s=(pad, pad), axes=(1, 2))
    dirty_fft   = imgs_fft * psf_fft[None, :, :]
    dirty_pad   = np.fft.irfft2(dirty_fft, s=(pad, pad), axes=(1, 2))
    return dirty_pad[:, :H, :W].astype(np.float32)


class SimulateObservations:
    """
    Simulate dirty radio observations from clean sky images.

    Parameters
    ----------
    psf_fwhm  : float | None   Synthetic Gaussian PSF FWHM in pixels.
                               Mutually exclusive with psf_path.
    psf_path  : str | Path | None  Real PSF file (.fits, .npy, or .npz).
                               Mutually exclusive with psf_fwhm.
    noise_std : float          Gaussian noise std in normalised units (default 0.05)
    seed      : int            RNG seed (default 42)
    """

    def __init__(
        self,
        psf_fwhm  : float | None       = None,
        psf_path  : str | Path | None  = None,
        noise_std : float              = 0.05,
        seed      : int                = 42,
        normalise : bool               = True,
    ):
        """
        Parameters
        ----------
        normalise : if True (default), apply per-image peak normalisation before
                    convolution — clean_n = clean / peak, range [0, 1].
                    Set False when training used raw physical units (Option C / SBI).
        """
        if psf_fwhm is None and psf_path is None:
            raise ValueError("Provide either psf_fwhm or psf_path")
        if psf_fwhm is not None and psf_path is not None:
            raise ValueError("psf_fwhm and psf_path are mutually exclusive")

        self.psf_fwhm  = psf_fwhm
        self.psf_path  = Path(psf_path) if psf_path is not None else None
        self.noise_std = noise_std
        self.seed      = seed
        self.normalise = normalise

    def run(self, data_path: str | Path, out: str | Path) -> None:
        """
        Load clean images from data_path, simulate dirty images, save to out.

        Parameters
        ----------
        data_path : path to .npz with 'images' key — (N, H, W) float32
        out       : output .npz path — keys: clean, dirty, psf, noise_std
        """
        data_path = Path(data_path)
        out_path  = Path(out)

        raw    = np.load(data_path)
        clean  = raw["images"].astype(np.float32)
        N, H, W = clean.shape
        print(f"Loaded {N} clean images  shape={H}×{W}")

        if self.normalise:
            # Per-image peak normalisation: map each image to [0, 1].
            peak    = clean.max(axis=(1, 2), keepdims=True) + 1e-8
            clean_n = clean / peak
        else:
            # Raw physical units — no normalisation (SBI / Option C training).
            clean_n = clean
        print(f"Normalisation: {'peak (range [0,1])' if self.normalise else 'none (raw flux)'}")

        if self.psf_fwhm is not None:
            psf = _make_gaussian_psf(self.psf_fwhm, size=max(H, W))
            psf = psf[:H, :W]
            print(f"Synthetic Gaussian PSF  FWHM={self.psf_fwhm}px  shape={psf.shape}")
        else:
            psf = _load_psf(self.psf_path, target_shape=(H, W))
            print(f"Loaded PSF from {self.psf_path}  shape={psf.shape}")

        # PSF conventions (CASA standard):
        #   psf      — peak=1 → dirty in Jy/beam, model in Jy/pixel
        #   psf_norm — sum=1  → stored for reference only
        # For a point source: dirty.peak = clean.peak × psf.peak = clean.peak.
        normaliser = ImageNormaliser().fit(psf)
        psf_norm   = (psf / normaliser.area).astype(np.float32)

        dirty   = _convolve_psf(clean_n, psf)   # peak=1 PSF → Jy/beam
        rng     = np.random.default_rng(self.seed)
        dirty  += rng.standard_normal(dirty.shape).astype(np.float32) * self.noise_std
        dirty_n = dirty

        print(f"PSF peak={psf.max():.4f}  area={normaliser.area:.2f}")
        print(f"Dirty  std={dirty_n.std():.3f}  "
              f"range=[{dirty_n.min():.3f}, {dirty_n.max():.3f}]")
        print(f"Clean  std={clean_n.std():.3f}  "
              f"range=[{clean_n.min():.3f}, {clean_n.max():.3f}]")

        out_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            out_path,
            clean     = clean_n,
            dirty     = dirty_n,
            psf       = psf,
            psf_norm  = psf_norm,
            psf_area  = np.float32(normaliser.area),
            noise_std = np.float32(self.noise_std),
        )
        print(f"Saved → {out_path}  keys: clean {clean_n.shape}, dirty {dirty_n.shape}, "
              f"psf {psf.shape}  psf_norm (sum={psf_norm.sum():.4f})")
