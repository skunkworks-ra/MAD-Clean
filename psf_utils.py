"""
mad_clean.psf_utils
===================
PSF analysis utilities used by Hogbom CLEAN and MADClean.

Public API
----------
compute_psf_patch(psf, energy_frac) -> (psf_patch, (half_h, half_w))
    Compute a trimmed PSF patch that captures a target fraction of total PSF
    power (squared L2 norm), centred on the PSF peak.
"""

from __future__ import annotations

import torch

__all__ = ["compute_psf_patch"]


def compute_psf_patch(
    psf         : torch.Tensor,
    energy_frac : float = 0.90,
) -> tuple[torch.Tensor, tuple[int, int]]:
    """
    Compute a centred PSF patch that captures a target fraction of total power.

    Parameters
    ----------
    psf         : (H, W) float32 tensor, peak at image centre (H//2, W//2)
    energy_frac : fraction of total PSF power (L2) to capture (default 0.90)

    Returns
    -------
    psf_patch : torch.Tensor  (2*half_h+1, 2*half_w+1) centred on PSF peak
    halves    : (half_h, half_w) int tuple — patch extent in each direction

    Algorithm
    ---------
    Expand a square window outward from the peak pixel until it contains
    >= energy_frac of total PSF power (squared L2 norm), then clip to image
    bounds.  Falls back to maximum half-size if PSF power is zero.
    """
    H, W = psf.shape
    cy, cx = H // 2, W // 2

    # Maximum half that keeps the centred patch inside the image.
    # For even H: cy=32, H-1-cy=31 → max_half_h=31 so cy+31+1=64 fits exactly.
    max_half_h = min(cy, H - 1 - cy)
    max_half_w = min(cx, W - 1 - cx)
    max_half   = min(max_half_h, max_half_w)

    total_power = float((psf ** 2).sum())

    half = max_half  # fallback: full image if PSF is all zeros
    if total_power > 0.0:
        for h in range(0, max_half + 1):
            patch_power = float(
                (psf[cy - h : cy + h + 1,
                     cx - h : cx + h + 1] ** 2).sum()
            )
            if patch_power / total_power >= energy_frac:
                half = h
                break

    half_h = min(half, max_half_h)
    half_w = min(half, max_half_w)

    psf_patch = psf[cy - half_h : cy + half_h + 1,
                    cx - half_w : cx + half_w + 1].clone()

    return psf_patch, (half_h, half_w)
