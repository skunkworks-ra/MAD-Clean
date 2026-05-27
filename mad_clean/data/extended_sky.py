"""
mad_clean.data.extended_sky
============================
Extended-source sky generators for the SBI-Asp training pipeline.

Each generator renders a morphological component to a 2D float32 array
and also returns a 6D target tuple representing the anisotropic-Gaussian
fit to that component, matching the Aspen parameter convention in
flow_plan.md:

    target = (x, y, log_flux, log_sig_maj, log_sig_min, PA)

where PA is the position angle in radians (not the (sin2θ, cos2θ) encoded
form — that encoding lives in the MDN head, not here), and (x, y) are
column, row centre coordinates (pixel units, 0-indexed).

Coordinate convention: (row, col) for array indexing; (x=col, y=row) for
target storage.  Follows the same convention as point_sky.py's catalog
tuples (row, col, flux).

Beam reference: G55 D-config L-band synthesised beam sigma W ≈ 1.4 px.
The plan's ROI cap is σ ≤ 8W ≈ 11.2 px.
"""

from __future__ import annotations

import math
from typing import NamedTuple

import numpy as np
from scipy.special import erf as _erf

__all__ = [
    "Target6D",
    "render_gaussian_blob",
    "render_shell",
    "render_filament",
    "assemble_mixed_field",
]

# G55 D-config L-band beam sigma (px).  Used for the lower bound on
# component scale; upper bound driven by 8 * BEAM_SIGMA_PX.
BEAM_SIGMA_PX: float = 1.4
SIGMA_MAX_PX: float = 8.0 * BEAM_SIGMA_PX  # ≈ 11.2 px


class Target6D(NamedTuple):
    """Anisotropic-Gaussian Aspen target in native units."""
    x: float          # column centre (px)
    y: float          # row centre (px)
    log_flux: float   # log of integrated flux (Jy)
    log_sig_maj: float  # log of major-axis sigma (px)
    log_sig_min: float  # log of minor-axis sigma (px)
    pa: float           # position angle, radians (E of N, i.e. CCW from +col)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_xy_grids(size: int) -> tuple[np.ndarray, np.ndarray]:
    """Return (X, Y) meshgrids where X=col index, Y=row index."""
    cols = np.arange(size, dtype=np.float64)
    rows = np.arange(size, dtype=np.float64)
    X, Y = np.meshgrid(cols, rows)
    return X, Y


def _aniso_gaussian(
    size: int,
    cx: float, cy: float,
    sig_maj: float, sig_min: float,
    pa: float,
) -> np.ndarray:
    """
    Evaluate an anisotropic Gaussian on a (size × size) grid.

    Returns a float64 array normalised so its sum equals 1 (analytic
    integral of continuous Gaussian ≈ pixel sum for sigma > ~1 px).

    Parameters
    ----------
    cx, cy    : centre (col, row)
    sig_maj   : sigma along the major axis (px)
    sig_min   : sigma along the minor axis (px)
    pa        : position angle of major axis in radians, CCW from +col.
    """
    X, Y = _make_xy_grids(size)
    dx = X - cx
    dy = Y - cy
    cos_pa = math.cos(pa)
    sin_pa = math.sin(pa)
    # Rotate into principal axes
    u = dx * cos_pa + dy * sin_pa   # along major axis
    v = -dx * sin_pa + dy * cos_pa  # along minor axis
    exponent = 0.5 * ((u / sig_maj) ** 2 + (v / sig_min) ** 2)
    g = np.exp(-exponent)
    # Normalise to unit sum (pixel-integral approximation)
    g_sum = g.sum()
    if g_sum > 0:
        g /= g_sum
    return g


# ---------------------------------------------------------------------------
# Generator 1: Anisotropic Gaussian blob
# ---------------------------------------------------------------------------

def render_gaussian_blob(
    size: int = 128,
    cx: float | None = None,
    cy: float | None = None,
    flux_jy: float = 1.0,
    sig_maj: float | None = None,
    sig_min: float | None = None,
    pa: float | None = None,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, Target6D]:
    """
    Render an anisotropic Gaussian blob.

    All free parameters (position, scale, angle, flux) are randomised if
    not supplied.  Scale is bounded to [BEAM_SIGMA_PX, SIGMA_MAX_PX].

    Returns
    -------
    image  : (size, size) float32, Jy/pixel
    target : Target6D
    """
    if rng is None:
        rng = np.random.default_rng()

    margin = int(math.ceil(SIGMA_MAX_PX * 3))
    lo = margin
    hi = size - margin

    if cx is None:
        cx = rng.uniform(lo, hi)
    if cy is None:
        cy = rng.uniform(lo, hi)
    if sig_maj is None:
        sig_maj = rng.uniform(BEAM_SIGMA_PX, SIGMA_MAX_PX)
    if sig_min is None:
        # Minor axis ≤ major axis
        sig_min = rng.uniform(BEAM_SIGMA_PX, sig_maj)
    if pa is None:
        pa = rng.uniform(0.0, math.pi)

    g = _aniso_gaussian(size, cx, cy, sig_maj, sig_min, pa)
    image = (flux_jy * g).astype(np.float32)

    target = Target6D(
        x=float(cx),
        y=float(cy),
        log_flux=float(math.log(flux_jy)),
        log_sig_maj=float(math.log(sig_maj)),
        log_sig_min=float(math.log(sig_min)),
        pa=float(pa),
    )
    return image, target


# ---------------------------------------------------------------------------
# Generator 2: Limb-brightened shell (circular v1)
# ---------------------------------------------------------------------------

def render_shell(
    size: int = 128,
    cx: float | None = None,
    cy: float | None = None,
    flux_jy: float = 1.0,
    radius: float | None = None,
    thickness: float | None = None,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, Target6D]:
    """
    Render a circular limb-brightened shell.

    The brightness profile is a thin annular ring modelled as a Gaussian
    in the radial direction with mean = radius and sigma = thickness / 2.

    Target encoding: σ_maj = σ_min = radius (per flow_plan.md "Target
    preparation" note: 'Shells fit to circular blobs with σ ≈ radius').
    PA is set to 0 (degenerate for circular morphology).

    Scale bound: radius bounded to [BEAM_SIGMA_PX, SIGMA_MAX_PX].
    """
    if rng is None:
        rng = np.random.default_rng()

    margin = int(math.ceil(SIGMA_MAX_PX * 2 + 3))
    lo = margin
    hi = size - margin

    if cx is None:
        cx = rng.uniform(lo, hi)
    if cy is None:
        cy = rng.uniform(lo, hi)
    if radius is None:
        radius = rng.uniform(BEAM_SIGMA_PX, SIGMA_MAX_PX)
    if thickness is None:
        # Shell thickness: between 0.5 beam and half the radius
        thickness = rng.uniform(BEAM_SIGMA_PX * 0.5, max(BEAM_SIGMA_PX, radius * 0.5))

    X, Y = _make_xy_grids(size)
    r = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2)
    shell_sigma = thickness / 2.0
    # Gaussian radial profile centred at the shell radius
    profile = np.exp(-0.5 * ((r - radius) / shell_sigma) ** 2)
    total = profile.sum()
    if total > 0:
        profile /= total
    image = (flux_jy * profile).astype(np.float32)

    target = Target6D(
        x=float(cx),
        y=float(cy),
        log_flux=float(math.log(flux_jy)),
        log_sig_maj=float(math.log(radius)),
        log_sig_min=float(math.log(radius)),
        pa=0.0,
    )
    return image, target


# ---------------------------------------------------------------------------
# Generator 3: Filament (line segment with Gaussian cross-section)
# ---------------------------------------------------------------------------

def render_filament(
    size: int = 128,
    cx: float | None = None,
    cy: float | None = None,
    flux_jy: float = 1.0,
    length: float | None = None,
    width: float | None = None,
    pa: float | None = None,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, Target6D]:
    """
    Render a filament: uniform along a line segment, Gaussian cross-section.

    Target encoding: σ_maj = length / 2 (half the segment length),
    σ_min = width (Gaussian cross-section sigma), PA = filament PA.
    Per flow_plan.md: 'filaments fit to elongated Gaussians'.

    Scale bounds: length/2 ≤ SIGMA_MAX_PX; width ≥ BEAM_SIGMA_PX.
    """
    if rng is None:
        rng = np.random.default_rng()

    margin = int(math.ceil(SIGMA_MAX_PX * 3))
    lo = margin
    hi = size - margin

    if cx is None:
        cx = rng.uniform(lo, hi)
    if cy is None:
        cy = rng.uniform(lo, hi)
    if pa is None:
        pa = rng.uniform(0.0, math.pi)
    if length is None:
        # Length such that half-length ≤ SIGMA_MAX_PX
        length = rng.uniform(2 * BEAM_SIGMA_PX, 2 * SIGMA_MAX_PX)
    if width is None:
        width = rng.uniform(BEAM_SIGMA_PX, SIGMA_MAX_PX)

    half_len = length / 2.0

    X, Y = _make_xy_grids(size)
    dx = X - cx
    dy = Y - cy
    cos_pa = math.cos(pa)
    sin_pa = math.sin(pa)
    # Project onto filament axes
    along = dx * cos_pa + dy * sin_pa    # along the filament
    across = -dx * sin_pa + dy * cos_pa  # perpendicular

    # Along axis: uniform within [-half_len, +half_len], tapered as Gaussian
    # at the ends (half-sigma = width for smooth taper).
    # Implementation: sigmoid-based box function
    taper_sigma = max(width, BEAM_SIGMA_PX)
    sqrt2_sigma = math.sqrt(2) * taper_sigma
    along_profile = (
        0.5 * (1.0 + _erf((along + half_len) / sqrt2_sigma))
        - 0.5 * (1.0 + _erf((along - half_len) / sqrt2_sigma))
    )
    # Across axis: Gaussian
    across_profile = np.exp(-0.5 * (across / width) ** 2)

    profile = along_profile * across_profile
    total = profile.sum()
    if total > 0:
        profile /= total
    image = (flux_jy * profile).astype(np.float32)

    # Target: elongated Gaussian fit
    sig_maj = half_len  # clamp to at least beam sigma
    sig_maj = max(sig_maj, BEAM_SIGMA_PX)
    sig_min = width

    target = Target6D(
        x=float(cx),
        y=float(cy),
        log_flux=float(math.log(flux_jy)),
        log_sig_maj=float(math.log(sig_maj)),
        log_sig_min=float(math.log(sig_min)),
        pa=float(pa),
    )
    return image, target


# ---------------------------------------------------------------------------
# Mixed-field assembler
# ---------------------------------------------------------------------------

_EXTENDED_GENERATORS = [render_gaussian_blob, render_shell, render_filament]


def assemble_mixed_field(
    size: int = 128,
    n_sources: int | tuple[int, int] = (5, 30),
    flux_range_jy: tuple[float, float] = (1e-4, 1e-1),
    dn_ds_slope: float = -1.6,
    edge_margin: int = 16,
    extended_rate: float = 0.05,
    rng: np.random.Generator | None = None,
    return_per_source: bool = False,
) -> tuple:
    """
    Assemble a mixed point-source + extended-source field.

    For each source drawn from the log N-log S distribution there is a
    5% (default) per-source chance of replacing it with an extended source
    drawn uniformly from the three templates.

    Parameters
    ----------
    size           : int         Field side length (pixels).
    n_sources      : int | (a,b) Source count (or range) as for point_sky.
    flux_range_jy  : (S_min, S_max)
    dn_ds_slope    : float       Differential power-law exponent.
    edge_margin    : int         Minimum distance from field edge.
    extended_rate  : float       Per-source probability of being extended
                                 (default 0.05 = 5 %).
    rng            : np.random.Generator | None

    Returns
    -------
    sky      : (size, size) float32, Jy/pixel.
    targets  : list of Target6D (one per source, point sources included).
               Point sources use log_sig_maj = log_sig_min = log(BEAM_SIGMA_PX),
               PA = 0 by convention.
    per_source : (only if return_per_source=True) list of (size, size) float32
               arrays, one per target, each holding that source's rendered
               contribution to ``sky`` exactly. ``sum(per_source) == sky``.
    """
    # Inline import to avoid circular dependency issues at module level.
    from mad_clean.data.point_sky import (
        _sample_truncated_power_law,
    )

    if rng is None:
        rng = np.random.default_rng()

    # Determine source count
    if isinstance(n_sources, tuple):
        a, b = n_sources
        n = int(rng.integers(a, b + 1))
    else:
        n = int(n_sources)

    s_min, s_max = flux_range_jy
    fluxes = _sample_truncated_power_law(n, s_min, s_max, dn_ds_slope, rng)

    lo = edge_margin
    hi = size - edge_margin

    sky = np.zeros((size, size), dtype=np.float32)
    targets: list[Target6D] = []
    per_source: list[np.ndarray] = []

    for flux in fluxes:
        is_extended = rng.random() < extended_rate
        cx = rng.uniform(lo, hi)
        cy = rng.uniform(lo, hi)

        if is_extended:
            gen_fn = _EXTENDED_GENERATORS[int(rng.integers(0, 3))]
            patch, target = gen_fn(
                size=size, cx=cx, cy=cy, flux_jy=float(flux), rng=rng
            )
            sky += patch
            targets.append(target)
            if return_per_source:
                per_source.append(patch)
        else:
            # Point source: single-pixel delta
            r = int(round(cy))
            c = int(round(cx))
            r = max(lo, min(hi - 1, r))
            c = max(lo, min(hi - 1, c))
            sky[r, c] += np.float32(flux)
            targets.append(Target6D(
                x=float(c),
                y=float(r),
                log_flux=float(math.log(float(flux))),
                log_sig_maj=float(math.log(BEAM_SIGMA_PX)),
                log_sig_min=float(math.log(BEAM_SIGMA_PX)),
                pa=0.0,
            ))
            if return_per_source:
                delta = np.zeros((size, size), dtype=np.float32)
                delta[r, c] = np.float32(flux)
                per_source.append(delta)

    if return_per_source:
        return sky, targets, per_source
    return sky, targets
