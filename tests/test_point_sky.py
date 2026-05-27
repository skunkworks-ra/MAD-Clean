"""Unit tests for mad_clean.data.point_sky.

point_sky was restored from git history at branch creation; pin its
contract on this branch so regressions surface here.
"""
from __future__ import annotations

import numpy as np
import pytest

from mad_clean.data.point_sky import (
    _sample_truncated_power_law,
    generate_point_source_field,
)


# ---------------------------------------------------------------------------
# _sample_truncated_power_law
# ---------------------------------------------------------------------------

def test_power_law_range_respected():
    rng = np.random.default_rng(0)
    s = _sample_truncated_power_law(2000, 1e-4, 1e-1, slope=-1.6, rng=rng)
    assert s.min() >= 1e-4
    assert s.max() <= 1e-1
    assert s.shape == (2000,)


def test_power_law_slope_neg_one_branch():
    """slope=-1 takes a log-uniform branch; verify it produces valid draws."""
    rng = np.random.default_rng(0)
    s = _sample_truncated_power_law(500, 1e-4, 1e-1, slope=-1.0, rng=rng)
    assert (s >= 1e-4).all()
    assert (s <= 1e-1).all()


def test_power_law_dn_ds_shape():
    """Steeper-than-Euclidean: many faint, few bright (median below midpoint)."""
    rng = np.random.default_rng(0)
    s_min, s_max = 1e-4, 1e-1
    s = _sample_truncated_power_law(5000, s_min, s_max, slope=-1.6, rng=rng)
    log_mid = 0.5 * (np.log(s_min) + np.log(s_max))
    median_log = np.median(np.log(s))
    assert median_log < log_mid, (
        f"For slope=-1.6 the log-median should sit below the log-midpoint; "
        f"got median_log={median_log:.3f}, log_mid={log_mid:.3f}"
    )


# ---------------------------------------------------------------------------
# generate_point_source_field
# ---------------------------------------------------------------------------

def test_field_shape_and_dtype():
    sky, cat = generate_point_source_field(
        size=128, n_sources=5,
        rng=np.random.default_rng(0),
    )
    assert sky.shape == (128, 128)
    assert sky.dtype == np.float32
    assert len(cat) == 5


def test_field_deterministic_under_seed():
    a, _ = generate_point_source_field(size=64, n_sources=8, rng=np.random.default_rng(42))
    b, _ = generate_point_source_field(size=64, n_sources=8, rng=np.random.default_rng(42))
    np.testing.assert_array_equal(a, b)


def test_total_flux_matches_catalog():
    sky, cat = generate_point_source_field(
        size=64, n_sources=10, rng=np.random.default_rng(0)
    )
    cat_flux = sum(f for (_r, _c, f) in cat)
    sky_flux = float(sky.sum())
    assert abs(cat_flux - sky_flux) < 1e-5


def test_edge_margin_respected():
    margin = 8
    _, cat = generate_point_source_field(
        size=64, n_sources=20, edge_margin=margin,
        rng=np.random.default_rng(0),
    )
    for r, c, _ in cat:
        assert margin <= r < 64 - margin
        assert margin <= c < 64 - margin


def test_n_sources_tuple_range():
    rng = np.random.default_rng(0)
    counts = set()
    for _ in range(30):
        _, cat = generate_point_source_field(
            size=64, n_sources=(3, 7), rng=rng
        )
        counts.add(len(cat))
    assert counts.issubset(set(range(3, 8))), counts
    assert len(counts) > 1, "tuple should produce variable counts across draws"


def test_invalid_flux_range_raises():
    with pytest.raises(ValueError):
        generate_point_source_field(
            size=32, n_sources=1, flux_range_jy=(1e-1, 1e-4),
            rng=np.random.default_rng(0),
        )


def test_invalid_edge_margin_raises():
    with pytest.raises(ValueError):
        generate_point_source_field(
            size=16, n_sources=1, edge_margin=10,
            rng=np.random.default_rng(0),
        )


def test_no_duplicate_positions():
    _, cat = generate_point_source_field(
        size=64, n_sources=25, rng=np.random.default_rng(0)
    )
    positions = {(r, c) for r, c, _ in cat}
    assert len(positions) == len(cat), "all integer positions must be unique"
