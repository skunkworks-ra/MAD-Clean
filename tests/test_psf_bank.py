"""
tests/test_psf_bank.py — unit tests for mad_clean.data.psf_bank

Requires data/g55/chunk_*/psf.fits to be present (~1.7 GB, not in git).
Tests are skipped automatically when the PSF files are absent so the
standard CPU test suite (pixi run test) does not fail on clean checkouts.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

# Locate repo root relative to this file
_REPO_ROOT = Path(__file__).parent.parent
_G55_DIR = _REPO_ROOT / "data" / "g55"
_PSF_DATA_PRESENT = any(_G55_DIR.glob("chunk_*/psf.fits"))

skip_no_data = pytest.mark.skipif(
    not _PSF_DATA_PRESENT,
    reason="G55 PSF data not present (rsync data/g55/ separately)",
)


@skip_no_data
def test_load_g55_psf_bank_count_no_augment():
    from mad_clean.data.psf_bank import load_g55_psf_bank

    bank = load_g55_psf_bank(_REPO_ROOT, target_size=512, rotation_augment=False)
    # 10 chunks, no rotation → 10 PSFs
    assert len(bank) == 10, f"Expected 10, got {len(bank)}"


@skip_no_data
def test_load_g55_psf_bank_count_with_augment():
    from mad_clean.data.psf_bank import load_g55_psf_bank

    bank = load_g55_psf_bank(_REPO_ROOT, target_size=512, rotation_augment=True)
    # 10 chunks × 4 rotations = 40 PSFs
    assert len(bank) == 40, f"Expected 40, got {len(bank)}"


@skip_no_data
def test_psf_shape_and_peak_norm():
    from mad_clean.data.psf_bank import load_g55_psf_bank

    bank = load_g55_psf_bank(_REPO_ROOT, target_size=512, rotation_augment=False)
    for i in range(len(bank)):
        psf, meta = bank[i]
        assert psf.shape == (512, 512), f"PSF {i} shape {psf.shape} != (512, 512)"
        assert psf.dtype == np.float32, f"PSF {i} dtype {psf.dtype} != float32"
        assert abs(psf.max() - 1.0) < 1e-5, f"PSF {i} peak {psf.max()} not peak-normalised"
        assert "source_path" in meta
        assert meta["rotation_deg"] == 0


@skip_no_data
def test_rotation_augment_deterministic():
    """Same index must return the same array on repeated calls."""
    from mad_clean.data.psf_bank import load_g55_psf_bank

    bank = load_g55_psf_bank(_REPO_ROOT, target_size=512, rotation_augment=True)
    psf_a, _ = bank[5]
    psf_b, _ = bank[5]
    np.testing.assert_array_equal(psf_a, psf_b)


@skip_no_data
def test_rotation_augment_meta_degrees():
    """Check that the four rotation variants have the expected degree labels."""
    from mad_clean.data.psf_bank import load_g55_psf_bank

    bank = load_g55_psf_bank(_REPO_ROOT, target_size=512, rotation_augment=True)
    # First chunk → indices 0..3
    degrees = [bank[i][1]["rotation_deg"] for i in range(4)]
    assert degrees == [0, 90, 180, 270], f"Unexpected rotation degrees: {degrees}"


@skip_no_data
def test_exclude_reduces_count():
    from mad_clean.data.psf_bank import load_g55_psf_bank

    bank_all = load_g55_psf_bank(_REPO_ROOT, rotation_augment=False)
    bank_ex = load_g55_psf_bank(_REPO_ROOT, rotation_augment=False, exclude=["scan"])
    assert len(bank_ex) == len(bank_all) - 1


@skip_no_data
def test_sample_returns_valid_psf():
    from mad_clean.data.psf_bank import load_g55_psf_bank

    bank = load_g55_psf_bank(_REPO_ROOT, rotation_augment=True)
    rng = np.random.default_rng(42)
    psf, meta = bank.sample(rng)
    assert psf.shape == (512, 512)
    assert psf.max() == pytest.approx(1.0, abs=1e-5)


# ---------------------------------------------------------------------------
# Smoke tests that run without any data (test module imports and helpers)
# ---------------------------------------------------------------------------

def test_psf_bank_empty_paths_raises():
    from mad_clean.data.psf_bank import PSFBank

    with pytest.raises(ValueError, match="psf_paths is empty"):
        PSFBank(psf_paths=[])


def test_peak_centred_resize_shape():
    from mad_clean.data.psf_bank import _peak_centred_resize

    psf = np.zeros((200, 200), dtype=np.float32)
    psf[100, 100] = 1.0
    out = _peak_centred_resize(psf, target_size=64)
    assert out.shape == (64, 64)
    assert out[32, 32] == pytest.approx(1.0)


def test_peak_centred_resize_pad():
    from mad_clean.data.psf_bank import _peak_centred_resize

    # Small PSF padded up to larger target
    psf = np.zeros((32, 32), dtype=np.float32)
    psf[16, 16] = 1.0
    out = _peak_centred_resize(psf, target_size=64)
    assert out.shape == (64, 64)
    assert out[32, 32] == pytest.approx(1.0)
