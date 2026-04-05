"""
mad_clean.normalise
===================
Canonical normalisation for dirty/clean image pairs.

Fixes the double-normalisation bug: previously simulate_observations.py
normalised before saving, then FlowTrainer normalised again, and eval
scripts had to add a third pass.  Now one class owns the transform and its
inverse — call it once at data generation time, once at inference.

Usage
-----
    normaliser = ImageNormaliser().fit(clean)
    clean_n, dirty_n = normaliser.fit_transform(clean, dirty)

    # At inference, invert to get flux-calibrated output:
    model_flux = normaliser.inverse_transform(model_n)
"""

from __future__ import annotations

import numpy as np

__all__ = ["ImageNormaliser"]


class ImageNormaliser:
    """
    Per-image normalisation using clean image statistics.

    Normalises both clean and dirty arrays using the per-image mean and std
    of the clean array.  This preserves the relative amplitude difference
    (blurring + noise) between dirty and clean — the model learns the
    dirty→clean mapping in a well-scaled space.

    Parameters
    ----------
    None at construction.  Call fit() before transform().
    """

    def fit(self, clean: np.ndarray) -> "ImageNormaliser":
        """
        Compute per-image mean and std from a clean image array.

        Parameters
        ----------
        clean : np.ndarray (N, H, W) float32

        Returns
        -------
        self — for method chaining
        """
        self._mean = clean.mean(axis=(1, 2), keepdims=True)   # (N, 1, 1)
        self._std  = clean.std(axis=(1, 2),  keepdims=True) + 1e-8
        return self

    def transform(self, images: np.ndarray) -> np.ndarray:
        """
        Apply normalisation: (images - mean) / std.

        Parameters
        ----------
        images : np.ndarray (N, H, W) float32 — same N as fit() array

        Returns
        -------
        np.ndarray float32, normalised
        """
        return ((images - self._mean) / self._std).astype(np.float32)

    def inverse_transform(self, images: np.ndarray) -> np.ndarray:
        """
        Invert normalisation: images * std + mean.

        Parameters
        ----------
        images : np.ndarray (N, H, W) float32 — normalised array

        Returns
        -------
        np.ndarray float32, in original flux units
        """
        return (images * self._std + self._mean).astype(np.float32)

    def fit_transform(
        self,
        clean: np.ndarray,
        dirty: np.ndarray | None = None,
    ) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
        """
        Fit on clean, then transform clean (and optionally dirty).

        Parameters
        ----------
        clean : np.ndarray (N, H, W)
        dirty : np.ndarray (N, H, W) or None

        Returns
        -------
        clean_n            — if dirty is None
        (clean_n, dirty_n) — if dirty is provided
        """
        self.fit(clean)
        clean_n = self.transform(clean)
        if dirty is not None:
            return clean_n, self.transform(dirty)
        return clean_n
