"""
tests/test_field_sky.py
=======================
Unit tests for mad_clean.data.field_sky — the §4 statistics-corpus generator
for the field-posterior program (Fork A).  All tests run on CPU in seconds;
no GPU, no file I/O.

The load-bearing properties under test:
  * strict non-negativity (Stokes I sky is positive; negatives are instrumental),
  * the diffuse field is log-normal with a power-law power spectrum P(k) ∝ k^{-α},
  * flux bookkeeping and determinism.
"""

from __future__ import annotations

import numpy as np
import pytest

from mad_clean.data.field_sky import (
    DIFFUSE_ALPHA,
    assemble_corpus_field,
    gaussian_random_field,
    generate_diffuse_lognormal,
    render_ridge,
)

SIZE = 128


def fresh_rng(seed: int = 42) -> np.random.Generator:
    return np.random.default_rng(seed)


# ── helpers ───────────────────────────────────────────────────────────────────

def _radial_power_slope(g: np.ndarray) -> float:
    """Fit the slope of log P(k) vs log k for a 2D field's radially-averaged
    power spectrum.  Returns the slope (≈ -alpha for a P(k) ∝ k^{-alpha} field)."""
    size = g.shape[0]
    P = np.abs(np.fft.fft2(g)) ** 2
    freq = np.fft.fftfreq(size) * size  # integer cycles per field
    KX, KY = np.meshgrid(freq, freq)
    k = np.sqrt(KX**2 + KY**2)
    kbin = np.rint(k).astype(int)

    kvals = np.arange(1, size // 2)
    Pk = np.array([P[kbin == kk].mean() for kk in kvals])
    # Fit over a mid range that avoids the smallest k (few samples) and the
    # Nyquist edge.
    mask = (kvals >= 3) & (kvals <= size // 4) & np.isfinite(Pk) & (Pk > 0)
    slope, _ = np.polyfit(np.log(kvals[mask]), np.log(Pk[mask]), 1)
    return float(slope)


# ── gaussian random field ───────────────────────────────────────────────────

def test_grf_shape_and_moments():
    g = gaussian_random_field(SIZE, alpha=DIFFUSE_ALPHA, rng=fresh_rng())
    assert g.shape == (SIZE, SIZE)
    assert np.isfinite(g).all()
    # Normalised to zero mean, unit variance.
    assert abs(float(g.mean())) < 1e-10
    assert abs(float(g.std()) - 1.0) < 1e-10


def test_grf_deterministic():
    g1 = gaussian_random_field(SIZE, rng=np.random.default_rng(7))
    g2 = gaussian_random_field(SIZE, rng=np.random.default_rng(7))
    np.testing.assert_array_equal(g1, g2)


@pytest.mark.parametrize("alpha", [2.0, 2.7, 3.5])
def test_grf_power_law_slope(alpha):
    """The radially-averaged power spectrum slope should recover -alpha.

    Uses a larger field for spectral stability; tolerance ±0.5 covers the
    finite-grid binning and the unit-variance renormalisation."""
    g = gaussian_random_field(256, alpha=alpha, rng=fresh_rng(123))
    slope = _radial_power_slope(g)
    assert abs(slope - (-alpha)) < 0.5, f"alpha={alpha}: measured slope {slope:.3f}"


# ── diffuse log-normal ──────────────────────────────────────────────────────

def test_diffuse_positive_and_finite():
    s = generate_diffuse_lognormal(SIZE, rng=fresh_rng())
    assert s.shape == (SIZE, SIZE)
    assert s.dtype == np.float32
    assert (s > 0).all(), "log-normal field must be strictly positive"
    assert np.isfinite(s).all()


def test_diffuse_flux_normalisation():
    target = 3.7
    s = generate_diffuse_lognormal(SIZE, total_flux_jy=target, rng=fresh_rng())
    assert abs(float(s.sum()) - target) / target < 1e-4


# ── ridge ────────────────────────────────────────────────────────────────────

def test_ridge_shape_positive_flux():
    img = render_ridge(SIZE, flux_jy=0.5, rng=fresh_rng())
    assert img.shape == (SIZE, SIZE)
    assert img.dtype == np.float32
    assert (img >= 0).all()
    assert abs(float(img.sum()) - 0.5) / 0.5 < 0.01


def test_ridge_degenerate_offfield_is_finite():
    """A far off-field arc (ring well outside the frame) must return finite zeros,
    not inf/NaN from dividing by a subnormal sum."""
    img = render_ridge(
        SIZE, cx=0.0, cy=0.0, flux_jy=0.1,
        radius=10_000.0, width=0.5, theta0=3.0, span=0.5, rng=fresh_rng(),
    )
    assert np.isfinite(img).all()
    assert float(np.abs(img).max()) == 0.0


def test_ridge_fuzz_all_finite():
    """Many random arcs are all finite (no subnormal-sum blowups)."""
    rng = fresh_rng(2024)
    for _ in range(200):
        img = render_ridge(SIZE, rng=rng)
        assert np.isfinite(img).all()


def test_ridge_is_localised():
    """A sharp arc concentrates its flux: a small fraction of pixels carry most
    of the brightness (unlike a diffuse field)."""
    img = render_ridge(SIZE, flux_jy=1.0, radius=30.0, width=1.0, rng=fresh_rng())
    flat = np.sort(img.ravel())[::-1]
    n_top = int(0.05 * flat.size)
    frac_in_top = float(flat[:n_top].sum()) / float(flat.sum())
    assert frac_in_top > 0.8, f"arc not concentrated: top-5% holds {frac_in_top:.2f}"


# ── corpus assembler ─────────────────────────────────────────────────────────

def test_corpus_shape_dtype_positive():
    sky = assemble_corpus_field(size=SIZE, rng=fresh_rng())
    assert sky.shape == (SIZE, SIZE)
    assert sky.dtype == np.float32
    assert (sky >= 0).all(), "corpus field must be non-negative (Stokes I)"
    assert np.isfinite(sky).all()


def test_corpus_deterministic():
    s1 = assemble_corpus_field(size=SIZE, rng=np.random.default_rng(1))
    s2 = assemble_corpus_field(size=SIZE, rng=np.random.default_rng(1))
    np.testing.assert_array_equal(s1, s2)


def test_corpus_components_sum_to_sky():
    sky, comps = assemble_corpus_field(
        size=SIZE, n_ridges=2, rng=fresh_rng(5), return_components=True
    )
    assert set(comps) == {"diffuse", "points", "ridges"}
    recon = comps["diffuse"] + comps["points"] + comps["ridges"]
    np.testing.assert_allclose(recon, sky, rtol=0, atol=1e-5)


def test_corpus_diffuse_can_be_disabled():
    sky, comps = assemble_corpus_field(
        size=SIZE, include_diffuse=False, n_ridges=0, rng=fresh_rng(),
        return_components=True,
    )
    assert float(comps["diffuse"].sum()) == 0.0
    # With no diffuse and no ridges, the field is the (sparse) point process.
    assert (sky >= 0).all()


def test_corpus_diffuse_flux_present():
    """The diffuse component carries its requested total flux inside the sum."""
    flux = 2.0
    _, comps = assemble_corpus_field(
        size=SIZE, diffuse_flux_jy=flux, n_points=0, n_ridges=0,
        rng=fresh_rng(3), return_components=True,
    )
    assert abs(float(comps["diffuse"].sum()) - flux) / flux < 1e-4
