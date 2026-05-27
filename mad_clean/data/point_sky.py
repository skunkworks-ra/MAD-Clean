"""
mad_clean.data.point_sky
========================
Procedural point-source sky generator for the step 1.0 conditional flow
falsifier (see ``flow_plan.md``).

Each call returns a fresh truth field — a (size, size) Jy/pixel image
populated with single-pixel deltas — together with the source catalog. No
PSF convolution and no noise; the forward model lives elsewhere.

Conventions
-----------
- Sky image: ``float32``, Jy/pixel, single-pixel deltas at integer positions.
- Catalog : list of ``(row, col, flux_jy)`` tuples in insertion order.
- Fluxes  : drawn from a truncated power law ``p(S) ∝ S^slope`` on
  ``[S_min, S_max]`` via inverse-CDF sampling.
"""

from __future__ import annotations

import numpy as np

__all__ = ["generate_point_source_field"]


def _sample_truncated_power_law(
    n: int,
    s_min: float,
    s_max: float,
    slope: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Inverse-CDF draw from p(S) ∝ S^slope on [s_min, s_max].

    Slope is the differential exponent (e.g. -1.6 for 1.4 GHz dN/dS).
    Handles the slope = -1 limit (log-uniform) separately.
    """
    u = rng.uniform(size=n)
    if np.isclose(slope, -1.0):
        # p(S) ∝ 1/S → CDF ∝ log S
        log_min = np.log(s_min)
        log_max = np.log(s_max)
        return np.exp(log_min + u * (log_max - log_min))

    a = slope + 1.0
    # CDF(S) = (S^a - s_min^a) / (s_max^a - s_min^a)
    s_min_a = s_min ** a
    s_max_a = s_max ** a
    return (s_min_a + u * (s_max_a - s_min_a)) ** (1.0 / a)


def generate_point_source_field(
    size: int = 512,
    n_sources: int | tuple[int, int] = (5, 30),
    flux_range_jy: tuple[float, float] = (1e-4, 1e-1),
    dn_ds_slope: float = -1.6,
    edge_margin: int = 16,
    min_radius_from_center: int = 0,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, list[tuple[int, int, float]]]:
    """
    Generate a point-source truth field for step 1.0 training.

    Parameters
    ----------
    size          : int        Field side length (pixels). Output is (size, size).
    n_sources     : int | (a, b)
                              Fixed count if int; uniform[a, b] inclusive if tuple.
    flux_range_jy : (S_min, S_max)
                              Truncation range for the dN/dS power law (Jy).
    dn_ds_slope   : float     Differential power-law exponent (default -1.6,
                              1.4 GHz extragalactic).
    edge_margin   : int       Minimum distance (px) of any source from the field
                              edge. Positions sampled on
                              ``[edge_margin, size - edge_margin)``.
    rng           : np.random.Generator | None
                              Source of randomness. ``None`` → default_rng().

    Returns
    -------
    sky     : (size, size) float32, Jy/pixel.
    catalog : list of (row, col, flux_jy).

    Notes
    -----
    Positions are integer pixels and de-duplicated within 1 px (i.e. exact
    pixel collisions are rejected and resampled). Sub-pixel positioning is
    out of scope for step 1.0.
    """
    if rng is None:
        rng = np.random.default_rng()

    if edge_margin < 0 or 2 * edge_margin >= size:
        raise ValueError(
            f"edge_margin={edge_margin} incompatible with size={size}"
        )

    s_min, s_max = flux_range_jy
    if not (s_min > 0 and s_max > s_min):
        raise ValueError(
            f"flux_range_jy must satisfy 0 < s_min < s_max; got {flux_range_jy}"
        )

    if isinstance(n_sources, tuple):
        a, b = n_sources
        if a < 0 or b < a:
            raise ValueError(f"n_sources tuple must satisfy 0 <= a <= b; got {n_sources}")
        # Uniform inclusive on [a, b].
        n = int(rng.integers(a, b + 1))
    else:
        n = int(n_sources)
        if n < 0:
            raise ValueError(f"n_sources must be non-negative; got {n}")

    lo = edge_margin
    hi = size - edge_margin  # exclusive upper bound

    # Reject duplicate integer pixels. With n <= ~30 and (hi-lo)^2 cells
    # available, collision probability is negligible but we guard anyway.
    occupied: set[tuple[int, int]] = set()
    positions: list[tuple[int, int]] = []
    max_attempts = max(1000, 50 * n)
    attempts = 0
    cy, cx = size / 2 - 0.5, size / 2 - 0.5
    min_r_sq = float(min_radius_from_center) ** 2
    while len(positions) < n and attempts < max_attempts:
        rows = rng.integers(lo, hi, size=n - len(positions))
        cols = rng.integers(lo, hi, size=n - len(positions))
        for r, c in zip(rows, cols):
            key = (int(r), int(c))
            if key in occupied:
                continue
            if min_r_sq > 0 and (r - cy) ** 2 + (c - cx) ** 2 < min_r_sq:
                continue
            occupied.add(key)
            positions.append(key)
        attempts += 1
    if len(positions) < n:
        raise RuntimeError(
            f"Could not place {n} unique sources after {attempts} attempts "
            f"on a {size}x{size} grid with edge_margin={edge_margin}"
        )

    fluxes = _sample_truncated_power_law(n, s_min, s_max, dn_ds_slope, rng)

    sky = np.zeros((size, size), dtype=np.float32)
    catalog: list[tuple[int, int, float]] = []
    for (r, c), f in zip(positions, fluxes):
        sky[r, c] += np.float32(f)
        catalog.append((r, c, float(f)))

    return sky, catalog
