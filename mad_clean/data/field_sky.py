"""
mad_clean.data.field_sky
========================
Statistics-corpus sky generator for the field-posterior program (Fork A).

This is the §4 corpus of ``field_posterior_design.md``.  Unlike
``extended_sky.py`` (which renders parametric *templates* carrying 6D
Gaussian-fit targets for the now-retired MDN-Asp head), this module encodes
generic radio-sky *statistics*, not source shapes.  The aim is a prior over
the patch transformation, shape-agnostic, spanning the statistical range a
target like Cyg A lives in without containing it.

Every field produced here is strictly non-negative: Stokes I total intensity
is ``>= 0`` (absorption against a bright background is deferred with Q/U).
Negatives are *not* sky — they are produced later by convolving this positive
sky with the real (zero-DC) interferometric PSF, which lives in the forward
model, not here.

Components (§4.2)
-----------------
- **Diffuse:** a log-normal field ``s = exp(g)`` with ``g`` a zero-mean
  Gaussian random field whose power spectrum is a power law
  ``P(k) ∝ k^{-alpha}``.  This is RESOLVE's prior form (a raw GRF is half
  negative; the exponential makes it positive and lets flux rearrange in
  log-space).
- **Compact:** a Poisson point process with fluxes drawn from a measured
  ``dN/dS`` (reuses ``point_sky._sample_truncated_power_law``).
- **Sharp non-Gaussian:** rectified limb-brightened arcs (shock / lobe-edge
  fronts), all positive, to enrich the higher-order statistics a pure GRF
  cannot reproduce.

Statistics constants (``alpha``, ``sigma_log``, ``dn_ds_slope``) are
PLACEHOLDERS pending the §4.6 calibration against measured survey statistics
(which ``P(k)`` / ``dN/dS`` / one-point PDF, from which surveys).  They are
flagged at each definition.

Conventions
-----------
- Sky image: ``float32``, Jy/pixel, strictly ``>= 0``.
- Coordinates: ``(row, col)`` for array indexing, ``(x=col, y=row)`` elsewhere,
  matching ``point_sky`` / ``extended_sky``.
- No PSF, no noise.  The forward model is a separate piece.
"""

from __future__ import annotations

import numpy as np

from mad_clean.data.point_sky import (
    _sample_truncated_power_law,
    generate_point_source_field,
)

__all__ = [
    "DIFFUSE_ALPHA",
    "DIFFUSE_SIGMA_LOG",
    "DN_DS_SLOPE",
    "gaussian_random_field",
    "generate_diffuse_lognormal",
    "render_ridge",
    "assemble_corpus_field",
]

# ── statistics constants (PLACEHOLDERS — §4.6 open) ──────────────────────────
# Power-law index of the diffuse power spectrum P(k) ∝ k^{-alpha}.  Radio
# diffuse / Galactic-foreground spectra sit roughly in alpha ≈ 2.4–3.0; 2.7 is
# a mid-range placeholder until calibrated to a measured survey power spectrum.
DIFFUSE_ALPHA: float = 2.7
# Standard deviation of the log-field g (sets the diffuse dynamic range).
DIFFUSE_SIGMA_LOG: float = 1.0
# Differential dN/dS exponent (1.4 GHz extragalactic), matching point_sky.
DN_DS_SLOPE: float = -1.6


# ---------------------------------------------------------------------------
# Diffuse: log-normal Gaussian random field
# ---------------------------------------------------------------------------

def gaussian_random_field(
    size: int,
    alpha: float = DIFFUSE_ALPHA,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """
    Draw a real, zero-mean, unit-variance Gaussian random field whose power
    spectrum is a power law ``P(k) ∝ k^{-alpha}``.

    The field is synthesised in Fourier space: white complex noise is shaped by
    an amplitude ``|F(k)| ∝ k^{-alpha/2}`` (so ``P = |F|^2 ∝ k^{-alpha}``), the
    DC mode is zeroed (enforcing zero mean), and the real part of the inverse
    transform is returned.  It is normalised to exactly zero mean and unit
    variance, so callers control the amplitude downstream.

    Parameters
    ----------
    size  : int   Field side length (pixels). Output is ``(size, size)``.
    alpha : float Power-law index of P(k).
    rng   : np.random.Generator | None

    Returns
    -------
    g : (size, size) float64, zero mean, unit variance.
    """
    if rng is None:
        rng = np.random.default_rng()

    kx = np.fft.fftfreq(size)
    ky = np.fft.fftfreq(size)
    KX, KY = np.meshgrid(kx, ky)
    k = np.sqrt(KX**2 + KY**2)

    with np.errstate(divide="ignore"):
        amp = np.where(k > 0, k ** (-alpha / 2.0), 0.0)

    noise = rng.standard_normal((size, size)) + 1j * rng.standard_normal((size, size))
    field_k = noise * amp
    g = np.fft.ifft2(field_k).real

    g -= g.mean()
    std = g.std()
    if std > 0:
        g /= std
    return g


def generate_diffuse_lognormal(
    size: int = 512,
    alpha: float = DIFFUSE_ALPHA,
    sigma_log: float = DIFFUSE_SIGMA_LOG,
    total_flux_jy: float | None = None,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """
    Render a strictly-positive diffuse log-normal field ``s = exp(sigma_log * g)``
    where ``g`` is a unit-variance power-law GRF (:func:`gaussian_random_field`).

    Parameters
    ----------
    size          : int   Field side length (pixels).
    alpha         : float Power-law index of the GRF power spectrum.
    sigma_log     : float Std of the log-field (diffuse dynamic range).
    total_flux_jy : float | None
                          If given, the field is rescaled so its pixel sum equals
                          this value (a multiplicative constant, which only shifts
                          the log-field mean — the field stays log-normal). If
                          None, the field is left at its natural ``exp`` scale.
    rng           : np.random.Generator | None

    Returns
    -------
    s : (size, size) float32, Jy/pixel, strictly > 0.
    """
    if rng is None:
        rng = np.random.default_rng()

    g = gaussian_random_field(size=size, alpha=alpha, rng=rng)
    s = np.exp(sigma_log * g)

    if total_flux_jy is not None:
        total = float(s.sum())
        if total > 0:
            s = s * (total_flux_jy / total)

    return s.astype(np.float32)


# ---------------------------------------------------------------------------
# Sharp non-Gaussian feature: rectified limb-brightened arc (shock front)
# ---------------------------------------------------------------------------

def render_ridge(
    size: int = 512,
    cx: float | None = None,
    cy: float | None = None,
    flux_jy: float = 1.0,
    radius: float | None = None,
    width: float | None = None,
    theta0: float | None = None,
    span: float | None = None,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """
    Render a sharp, positive, limb-brightened arc: an angular segment of a
    circle with a narrow radial Gaussian cross-section.  This is the canonical
    shock / lobe-edge front named in §4.2 — a brightness ridge the diffuse GRF
    cannot reproduce because it is locally non-Gaussian.

    The arc is feathered at its angular endpoints (half-power taper over ~0.15
    of the span) so it does not introduce hard pixel discontinuities along the
    field grid; the *radial* profile is deliberately narrow (sharp).

    Parameters
    ----------
    size    : int   Field side length (pixels).
    cx, cy  : arc centre (col, row); randomised if None.
    flux_jy : float Integrated flux (pixel sum) of the arc.
    radius  : float Arc radius (px); randomised if None.
    width   : float Radial Gaussian sigma (px); narrow → sharp. Randomised if None.
    theta0  : float Start angle (rad); randomised if None.
    span    : float Angular extent (rad); randomised if None.
    rng     : np.random.Generator | None

    Returns
    -------
    image : (size, size) float32, Jy/pixel, >= 0, pixel sum ≈ flux_jy.
    """
    if rng is None:
        rng = np.random.default_rng()

    margin = int(0.15 * size)
    lo, hi = margin, size - margin
    if cx is None:
        cx = rng.uniform(lo, hi)
    if cy is None:
        cy = rng.uniform(lo, hi)
    if radius is None:
        radius = rng.uniform(0.1 * size, 0.35 * size)
    if width is None:
        width = rng.uniform(0.5, 2.0)  # sharp cross-section
    if theta0 is None:
        theta0 = rng.uniform(0.0, 2.0 * np.pi)
    if span is None:
        span = rng.uniform(np.pi / 4.0, 1.5 * np.pi)

    cols = np.arange(size, dtype=np.float64)
    rows = np.arange(size, dtype=np.float64)
    X, Y = np.meshgrid(cols, rows)
    dx = X - cx
    dy = Y - cy
    r = np.sqrt(dx**2 + dy**2)

    # Radial profile: narrow Gaussian about the arc radius (the sharp edge).
    radial = np.exp(-0.5 * ((r - radius) / width) ** 2)

    # Angular mask within [theta0, theta0 + span], wrapped to [0, 2π), with a
    # cosine taper at the two ends so the arc fades smoothly.
    phi = np.mod(np.arctan2(dy, dx) - theta0, 2.0 * np.pi)
    taper = max(1e-3, 0.15 * span)
    inside = phi <= span
    angular = np.zeros_like(phi)
    angular[inside] = 1.0
    # Feather both ends.
    lead = inside & (phi < taper)
    trail = inside & (phi > span - taper)
    angular[lead] = 0.5 * (1.0 - np.cos(np.pi * phi[lead] / taper))
    angular[trail] = 0.5 * (1.0 - np.cos(np.pi * (span - phi[trail]) / taper))

    profile = radial * angular
    total = profile.sum()
    # Guard against a degenerate off-field arc: when the arc's in-field pixels
    # all sit far from the ring, ``total`` can be subnormal (~1e-38) and
    # ``flux_jy / total`` overflows to inf/NaN.  Such an arc contributes nothing,
    # so return zeros below the threshold rather than a poisoned field.
    if total > 1e-6:
        profile = profile * (flux_jy / total)
    else:
        profile = np.zeros_like(profile)
    return profile.astype(np.float32)


# ---------------------------------------------------------------------------
# Corpus assembler
# ---------------------------------------------------------------------------

def assemble_corpus_field(
    size: int = 512,
    *,
    include_diffuse: bool = True,
    diffuse_alpha: float = DIFFUSE_ALPHA,
    diffuse_sigma_log: float = DIFFUSE_SIGMA_LOG,
    diffuse_flux_jy: float = 1.0,
    n_points: int | tuple[int, int] = (20, 100),
    point_flux_range_jy: tuple[float, float] = (1e-4, 1e-1),
    dn_ds_slope: float = DN_DS_SLOPE,
    n_ridges: int | tuple[int, int] = (0, 3),
    ridge_flux_range_jy: tuple[float, float] = (1e-3, 1e-1),
    edge_margin: int = 16,
    rng: np.random.Generator | None = None,
    return_components: bool = False,
) -> np.ndarray | tuple[np.ndarray, dict[str, np.ndarray]]:
    """
    Assemble one strictly-positive corpus field from the three §4.2 components.

    The output is the *clean positive sky* only — the teaching target for a pure
    score prior (Fork A core), which trains on the positive side and lets the
    likelihood attribute all negatives to the PSF response (§4.4).  The forward
    model (PSF + noise) is applied elsewhere.

    Parameters
    ----------
    size              : int   Field side length (pixels).
    include_diffuse   : bool  Add the log-normal diffuse component.
    diffuse_alpha     : float GRF power-law index.
    diffuse_sigma_log : float GRF log-amplitude std.
    diffuse_flux_jy   : float Total flux of the diffuse component (pixel sum).
    n_points          : int | (a,b) Poisson point count (or inclusive range).
    point_flux_range_jy : (S_min, S_max) for the dN/dS draw.
    dn_ds_slope       : float Differential dN/dS exponent.
    n_ridges          : int | (a,b) Number of sharp arcs (or inclusive range).
    ridge_flux_range_jy : (S_min, S_max) for arc integrated flux.
    edge_margin       : int   Keep point sources this far from the edge.
    rng               : np.random.Generator | None
    return_components : bool  Also return a dict {"diffuse","points","ridges"}.

    Returns
    -------
    sky : (size, size) float32, Jy/pixel, strictly >= 0.
    components : (only if return_components) dict of the three layers; their sum
                 equals ``sky``.
    """
    if rng is None:
        rng = np.random.default_rng()

    sky = np.zeros((size, size), dtype=np.float32)
    components: dict[str, np.ndarray] = {}

    # Diffuse log-normal field.
    if include_diffuse:
        diffuse = generate_diffuse_lognormal(
            size=size,
            alpha=diffuse_alpha,
            sigma_log=diffuse_sigma_log,
            total_flux_jy=diffuse_flux_jy,
            rng=rng,
        )
        sky += diffuse
        if return_components:
            components["diffuse"] = diffuse
    elif return_components:
        components["diffuse"] = np.zeros((size, size), dtype=np.float32)

    # Compact point process (reuse the dN/dS generator).
    points, _catalog = generate_point_source_field(
        size=size,
        n_sources=n_points,
        flux_range_jy=point_flux_range_jy,
        dn_ds_slope=dn_ds_slope,
        edge_margin=edge_margin,
        rng=rng,
    )
    sky += points
    if return_components:
        components["points"] = points

    # Sharp non-Gaussian arcs.
    if isinstance(n_ridges, tuple):
        a, b = n_ridges
        n_arc = int(rng.integers(a, b + 1))
    else:
        n_arc = int(n_ridges)
    ridge_layer = np.zeros((size, size), dtype=np.float32)
    if n_arc > 0:
        r_min, r_max = ridge_flux_range_jy
        ridge_fluxes = _sample_truncated_power_law(
            n_arc, r_min, r_max, dn_ds_slope, rng
        )
        for f in ridge_fluxes:
            ridge_layer += render_ridge(size=size, flux_jy=float(f), rng=rng)
    sky += ridge_layer
    if return_components:
        components["ridges"] = ridge_layer

    if return_components:
        return sky, components
    return sky
