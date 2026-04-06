"""
mad_clean.psf_utils
===================
PSF analysis utilities used by Hogbom CLEAN and MADClean.

Public API
----------
compute_psf_patch(psf, energy_frac) -> (psf_patch, (half_h, half_w))
    Compute a trimmed PSF patch that captures a target fraction of total PSF
    power (squared L2 norm), centred on the PSF peak.

psf_sidelobe_analysis(psf, n_sidelobes) -> dict
    Radial profile, null radii, and cumulative PSF flux at each sidelobe
    boundary. Informs physically motivated normalisation scale for dirty images.
"""

from __future__ import annotations

import numpy as np
import torch

__all__ = ["compute_psf_patch", "psf_sidelobe_analysis"]


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


def psf_sidelobe_analysis(
    psf         : np.ndarray,
    n_sidelobes : int = 5,
) -> dict:
    """
    Compute PSF radial profile, null radii, and cumulative flux fractions.

    For a peak-normalised PSF (peak=1), finds the radii at which the azimuthal-
    average profile crosses zero (sidelobe nulls) and reports the cumulative
    integral of the PSF within each boundary.  The main-lobe integral (within
    the 1st null) is the physically motivated normalisation scale for dirty
    images: dirty = PSF * clean, so a unit-flux source produces dirty values
    whose sum equals that integral.

    Note: the VLA PSF is not circularly symmetric (Y-configuration).  The
    radial average smooths over the three arms, so null radii are averages
    across all azimuths.  Cumulative flux integrals are computed over the full
    2D PSF within each circular aperture, not from the radial average.

    Parameters
    ----------
    psf         : (H, W) float32 numpy array, peak-normalised (peak = 1),
                  PSF centre at (H//2, W//2)
    n_sidelobes : number of null crossings to find (default 5)

    Returns
    -------
    dict with keys
        radii          : 1D array of integer radii in pixels (0 … max_r)
        radial_profile : azimuthal-average PSF value at each radius
        null_radii     : list of radii (pixels) where profile crosses zero
        cumulative_flux: list of cumulative PSF sums within each null radius
        flux_fractions : cumulative_flux / total_psf_flux for each null
        total_flux     : sum of all PSF values (full-image integral)
    """
    psf  = np.asarray(psf, dtype=np.float64)
    H, W = psf.shape
    cy, cx = H // 2, W // 2
    max_r  = min(cy, cx, H - 1 - cy, W - 1 - cx)

    # Build pixel-distance map from centre.
    ys = np.arange(H, dtype=np.float64) - cy
    xs = np.arange(W, dtype=np.float64) - cx
    yy, xx = np.meshgrid(ys, xs, indexing="ij")
    dist = np.sqrt(yy ** 2 + xx ** 2)

    # Azimuthal average: bin by integer radius.
    radii   = np.arange(0, max_r + 1, dtype=int)
    profile = np.zeros(len(radii), dtype=np.float64)
    for r in radii:
        mask = (dist >= r - 0.5) & (dist < r + 0.5)
        if mask.any():
            profile[r] = psf[mask].mean()

    # Find zero crossings (nulls) in the radial profile.
    null_radii = []
    for i in range(1, len(profile) - 1):
        if profile[i - 1] * profile[i] < 0:   # sign change
            # Linear interpolation to sub-pixel null position.
            null_r = i - 1 + profile[i - 1] / (profile[i - 1] - profile[i])
            null_radii.append(float(null_r))
        if len(null_radii) == n_sidelobes:
            break

    # Cumulative PSF flux within each null (circular aperture, full 2D).
    total_flux     = float(psf.sum())
    cumulative_flux = []
    flux_fractions  = []
    for nr in null_radii:
        mask   = dist <= nr
        c_flux = float(psf[mask].sum())
        cumulative_flux.append(c_flux)
        flux_fractions.append(c_flux / total_flux if total_flux != 0 else 0.0)

    return {
        "radii"          : radii,
        "radial_profile" : profile,
        "null_radii"     : null_radii,
        "cumulative_flux": cumulative_flux,
        "flux_fractions" : flux_fractions,
        "total_flux"     : total_flux,
    }
