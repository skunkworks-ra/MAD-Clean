"""
mad_clean.normalise
===================
Physically motivated normalisation for dirty/clean image pairs.

Area normalisation
------------------
Divides dirty images by the PSF total flux (sum over all pixels).

Since dirty = PSF ⊛ clean, this approximately restores the clean flux
scale:  dirty / psf.sum() ≈ clean  (ideal, noise-free case).

This is physically motivated: the PSF area sets the amplitude relationship
between dirty and clean.  Per-image std normalisation is arbitrary and not
flux-conserving — it destroys the dirty/clean amplitude relationship.

Usage
-----
    normaliser = ImageNormaliser().fit(psf)
    dirty_n    = normaliser.transform(dirty)

    # At inference, invert to recover flux-calibrated output:
    model_flux = normaliser.inverse_transform(model_n)

    # Save/restore the area factor alongside the data:
    np.savez("data.npz", ..., psf_area=normaliser.area)
    normaliser = ImageNormaliser(area=float(data["psf_area"]))
"""

from __future__ import annotations

import numpy as np

__all__ = ["ImageNormaliser"]


class ImageNormaliser:
    """
    PSF-area normalisation for dirty images.

    The normalisation factor is the scalar PSF total flux (psf.sum()).
    This is a global constant for the dataset — all images are divided by
    the same value, preserving the relative flux relationships across images.

    Parameters
    ----------
    area : float | None
        Pre-computed PSF area.  If provided, fit() is not needed.
        Useful for restoring a saved normaliser from npz metadata.
    """

    def __init__(self, area: float | None = None):
        if area is not None:
            self._area = float(area)

    # ------------------------------------------------------------------
    @property
    def area(self) -> float:
        """PSF total flux used as normalisation divisor."""
        if not hasattr(self, "_area"):
            raise RuntimeError("ImageNormaliser has not been fit. Call fit(psf) first.")
        return self._area

    # ------------------------------------------------------------------
    def fit(self, psf: np.ndarray) -> "ImageNormaliser":
        """
        Compute the normalisation area from the PSF.

        Parameters
        ----------
        psf : np.ndarray (H, W) float32 — peak-normalised PSF (peak = 1)

        Returns
        -------
        self — for method chaining
        """
        self._area = float(np.asarray(psf, dtype=np.float64).sum())
        return self

    # ------------------------------------------------------------------
    def transform(self, dirty: np.ndarray) -> np.ndarray:
        """
        Normalise dirty images: dirty / area.

        Parameters
        ----------
        dirty : np.ndarray (N, H, W) float32

        Returns
        -------
        np.ndarray float32 — dirty images in clean-flux units
        """
        return (dirty / self.area).astype(np.float32)

    # ------------------------------------------------------------------
    def inverse_transform(self, dirty_n: np.ndarray) -> np.ndarray:
        """
        Invert normalisation: dirty_n * area.

        Parameters
        ----------
        dirty_n : np.ndarray (N, H, W) float32 — normalised dirty images

        Returns
        -------
        np.ndarray float32 — dirty images in original flux units
        """
        return (dirty_n * self.area).astype(np.float32)

    # ------------------------------------------------------------------
    def __repr__(self) -> str:
        area_str = f"{self._area:.4f}" if hasattr(self, "_area") else "not fit"
        return f"ImageNormaliser(area={area_str})"
