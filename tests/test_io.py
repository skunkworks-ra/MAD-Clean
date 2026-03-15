"""
tests/test_io.py — tests for mad_clean.io
"""

import tempfile
from pathlib import Path

import numpy as np
import pytest

from mad_clean.io import load_image, load_image_data, save_fits


# ── helpers ───────────────────────────────────────────────────────────────────

def _make_image(h: int = 32, w: int = 32) -> np.ndarray:
    rng = np.random.default_rng(0)
    return rng.uniform(0, 1, (h, w)).astype(np.float32)


def _write_fits(array: np.ndarray, path: Path, shape_override=None) -> None:
    """Write array (or a reshaped version) to a FITS file."""
    from astropy.io import fits
    data = array if shape_override is None else array.reshape(shape_override)
    fits.PrimaryHDU(data.astype(np.float32)).writeto(path, overwrite=True)


# ── tests ─────────────────────────────────────────────────────────────────────

def test_load_numpy_passthrough():
    """Passing a numpy array returns float32 with the same values."""
    img = _make_image().astype(np.float64)
    result = load_image_data(img)
    assert result.dtype == np.float32
    np.testing.assert_allclose(result, img.astype(np.float32))


def test_load_numpy_already_float32():
    """float32 input is returned without unnecessary copy (dtype preserved)."""
    img = _make_image()
    result = load_image_data(img)
    assert result.dtype == np.float32


def test_load_fits_round_trip():
    """save_fits then load_image_data recovers pixel values within float32 tolerance."""
    img = _make_image()
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "test.fits"
        save_fits(img, path)
        loaded = load_image_data(str(path))
    assert loaded.shape == img.shape
    np.testing.assert_allclose(loaded, img, rtol=1e-6)


def test_load_fits_squeeze_casa_axes():
    """FITS with CASA degenerate shape (1, 1, H, W) is squeezed to (H, W)."""
    img = _make_image(20, 30)
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "casa.fits"
        _write_fits(img, path, shape_override=(1, 1, 20, 30))
        loaded = load_image_data(str(path))
    assert loaded.shape == (20, 30)
    np.testing.assert_allclose(loaded, img, rtol=1e-6)


def test_load_fits_missing_file():
    """FileNotFoundError raised for a non-existent path."""
    with pytest.raises(FileNotFoundError):
        load_image_data("/tmp/definitely_does_not_exist_mad_clean.fits")


def test_save_fits_creates_parent_dirs():
    """save_fits creates intermediate directories if they don't exist."""
    img = _make_image()
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "subdir" / "deep" / "out.fits"
        save_fits(img, path)
        assert path.exists()


def test_load_image_returns_header_for_fits():
    """load_image (not load_image_data) returns (array, header) tuple for FITS."""
    img = _make_image()
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "hdr.fits"
        save_fits(img, path)
        result = load_image(str(path))
    assert isinstance(result, tuple)
    assert len(result) == 2
    assert result[0].shape == img.shape
