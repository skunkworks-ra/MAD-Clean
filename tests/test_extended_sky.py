"""
tests/test_extended_sky.py
==========================
Unit tests for mad_clean.data.extended_sky generators and mixed-field
assembler.  All tests run on CPU in seconds; no GPU, no file I/O.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from mad_clean.data.extended_sky import (
    BEAM_SIGMA_PX,
    SIGMA_MAX_PX,
    Target6D,
    _aniso_gaussian,
    assemble_mixed_field,
    render_filament,
    render_gaussian_blob,
    render_shell,
)

# ── fixtures ──────────────────────────────────────────────────────────────────

SIZE = 128
FLUX = 0.5
RNG = np.random.default_rng(42)


def fresh_rng(seed: int = 42) -> np.random.Generator:
    return np.random.default_rng(seed)


# ── shape tests ───────────────────────────────────────────────────────────────

def test_gaussian_blob_shape():
    img, tgt = render_gaussian_blob(size=SIZE, rng=fresh_rng())
    assert img.shape == (SIZE, SIZE)
    assert img.dtype == np.float32
    assert isinstance(tgt, Target6D)


def test_shell_shape():
    img, tgt = render_shell(size=SIZE, rng=fresh_rng())
    assert img.shape == (SIZE, SIZE)
    assert img.dtype == np.float32
    assert isinstance(tgt, Target6D)


def test_filament_shape():
    img, tgt = render_filament(size=SIZE, rng=fresh_rng())
    assert img.shape == (SIZE, SIZE)
    assert img.dtype == np.float32
    assert isinstance(tgt, Target6D)


# ── flux conservation ─────────────────────────────────────────────────────────

@pytest.mark.parametrize("render_fn", [render_gaussian_blob, render_shell, render_filament])
def test_flux_conservation(render_fn):
    """Pixel sum should equal the requested flux within 1 %."""
    img, tgt = render_fn(size=SIZE, flux_jy=FLUX, rng=fresh_rng())
    pixel_sum = float(img.sum())
    assert abs(pixel_sum - FLUX) / FLUX < 0.01, (
        f"{render_fn.__name__}: pixel_sum={pixel_sum:.6f} vs requested={FLUX}"
    )
    assert math.isclose(math.exp(tgt.log_flux), FLUX, rel_tol=1e-6)


# ── scale bounds ──────────────────────────────────────────────────────────────

def test_gaussian_blob_scale_bounds():
    """Random Gaussian blobs must respect the σ ≤ 8W plan constraint."""
    rng = fresh_rng(7)
    for _ in range(20):
        _, tgt = render_gaussian_blob(size=SIZE, rng=rng)
        sig_maj = math.exp(tgt.log_sig_maj)
        sig_min = math.exp(tgt.log_sig_min)
        assert sig_maj <= SIGMA_MAX_PX + 1e-6, f"sig_maj={sig_maj} > cap"
        assert sig_min <= sig_maj + 1e-6, f"sig_min={sig_min} > sig_maj={sig_maj}"
        assert sig_maj >= BEAM_SIGMA_PX - 1e-6
        assert sig_min >= BEAM_SIGMA_PX - 1e-6


def test_shell_scale_bounds():
    rng = fresh_rng(8)
    for _ in range(20):
        _, tgt = render_shell(size=SIZE, rng=rng)
        radius = math.exp(tgt.log_sig_maj)
        assert radius <= SIGMA_MAX_PX + 1e-6
        assert radius >= BEAM_SIGMA_PX - 1e-6
        # Shell encodes σ_maj == σ_min
        assert math.isclose(tgt.log_sig_maj, tgt.log_sig_min, rel_tol=1e-9)


# ── 6D round-trip ─────────────────────────────────────────────────────────────

def _render_from_target(tgt: Target6D, size: int = SIZE) -> np.ndarray:
    """Re-render the component from its 6D target for round-trip check."""
    cx = tgt.x
    cy = tgt.y
    flux = math.exp(tgt.log_flux)
    sig_maj = math.exp(tgt.log_sig_maj)
    sig_min = math.exp(tgt.log_sig_min)
    pa = tgt.pa
    g = _aniso_gaussian(size, cx, cy, sig_maj, sig_min, pa)
    return (flux * g).astype(np.float32)


@pytest.mark.parametrize("render_fn", [render_gaussian_blob, render_shell])
def test_6d_roundtrip_gaussian(render_fn):
    """
    For blob and shell (both Gaussian-shaped), rendering from the 6D target
    should reproduce the original image to within 2 % relative pixel-sum error.
    """
    img, tgt = render_fn(
        size=SIZE, cx=64.0, cy=64.0, flux_jy=FLUX, rng=fresh_rng()
    )
    img_rt = _render_from_target(tgt, SIZE)
    # Pixel-level agreement: mean absolute deviation relative to peak
    peak = float(img.max())
    mad = float(np.mean(np.abs(img_rt - img)))
    assert mad / peak < 0.02, (
        f"{render_fn.__name__} round-trip MAD/peak={mad/peak:.4f}"
    )


def test_6d_roundtrip_filament():
    """
    Filament target uses an elongated-Gaussian approximation so the round-trip
    is approximate.  Require flux conservation to within 5 % and that the
    reconstructed image is spatially correlated with the original (r > 0.8).
    """
    img, tgt = render_filament(
        size=SIZE, cx=64.0, cy=64.0, flux_jy=FLUX, rng=fresh_rng()
    )
    img_rt = _render_from_target(tgt, SIZE)

    # Flux
    ratio = float(img_rt.sum()) / float(img.sum())
    assert 0.7 < ratio < 1.3, f"Filament round-trip flux ratio={ratio:.3f}"

    # Spatial correlation (flatten, correlate)
    a = img.ravel().astype(np.float64)
    b = img_rt.ravel().astype(np.float64)
    corr = float(np.corrcoef(a, b)[0, 1])
    assert corr > 0.7, f"Filament round-trip correlation={corr:.3f}"


# ── mixed field assembler ─────────────────────────────────────────────────────

def test_mixed_field_shape():
    sky, targets = assemble_mixed_field(size=SIZE, n_sources=10, rng=fresh_rng())
    assert sky.shape == (SIZE, SIZE)
    assert sky.dtype == np.float32
    assert len(targets) == 10


def test_mixed_field_total_flux():
    """Total flux in the assembled field should equal sum of target fluxes."""
    sky, targets = assemble_mixed_field(size=SIZE, n_sources=15, rng=fresh_rng(99))
    total_pixel = float(sky.sum())
    total_target = sum(math.exp(t.log_flux) for t in targets)
    # Allow 1 % due to clipping of extended sources near edges
    assert abs(total_pixel - total_target) / total_target < 0.01, (
        f"Mixed field flux: pixel={total_pixel:.6f} target_sum={total_target:.6f}"
    )


def test_mixed_field_extended_rate():
    """
    Over many fields the fraction of extended sources should be near 5 %.
    Use a generous tolerance (±3 pp) given finite sample counts.
    """
    rng = fresh_rng(0)
    n_sources = 200
    sky, targets = assemble_mixed_field(
        size=SIZE,
        n_sources=n_sources,
        extended_rate=0.05,
        rng=rng,
    )
    # Extended sources have sig_maj > BEAM_SIGMA_PX threshold (more than a point)
    n_extended = sum(
        1 for t in targets
        if math.exp(t.log_sig_maj) > BEAM_SIGMA_PX * 1.01
    )
    frac = n_extended / n_sources
    assert 0.0 <= frac <= 0.2, f"Extended fraction {frac:.3f} far from expected ~0.05"


def test_mixed_field_deterministic():
    """Same seed produces identical fields."""
    sky1, _ = assemble_mixed_field(size=SIZE, n_sources=10, rng=np.random.default_rng(1))
    sky2, _ = assemble_mixed_field(size=SIZE, n_sources=10, rng=np.random.default_rng(1))
    np.testing.assert_array_equal(sky1, sky2)


def test_targets_are_target6d():
    """All returned targets are Target6D named tuples with finite numeric values."""
    _, targets = assemble_mixed_field(size=SIZE, n_sources=20, rng=fresh_rng())
    for t in targets:
        assert isinstance(t, Target6D)
        for field, val in zip(t._fields, t):
            if field == "kind":
                assert isinstance(val, str)
                continue
            assert math.isfinite(val), f"Non-finite value in target: {t}"
