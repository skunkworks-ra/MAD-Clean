"""
tests/test_deconvolver.py — tests for mad_clean.deconvolver.MADClean
"""

import tempfile
from pathlib import Path

import numpy as np
import torch

from mad_clean.filters    import FilterBank
from mad_clean.detection  import IslandDetector
from mad_clean.solvers    import PatchSolver, ConvSolver
from mad_clean.deconvolver import MADClean


# ── helpers ───────────────────────────────────────────────────────────────────

def _make_fb(k: int = 8, f: int = 5, seed: int = 0) -> FilterBank:
    rng   = np.random.default_rng(seed)
    atoms = rng.standard_normal((k, f, f)).astype(np.float32)
    return FilterBank(atoms, device="cpu")


def _delta_psf(h: int = 32, w: int = 32) -> np.ndarray:
    """PSF that is a delta function at the image centre — convolution = identity."""
    psf = np.zeros((h, w), dtype=np.float32)
    psf[h // 2, w // 2] = 1.0
    return psf


def _gaussian_psf(h: int = 32, w: int = 32, sigma: float = 2.0) -> np.ndarray:
    """Gaussian PSF centred at image centre."""
    y  = np.arange(h) - h // 2
    x  = np.arange(w) - w // 2
    yy, xx = np.meshgrid(y, x, indexing="ij")
    psf = np.exp(-(yy ** 2 + xx ** 2) / (2 * sigma ** 2)).astype(np.float32)
    psf /= psf.sum()
    return psf


def _point_source_image(h: int = 32, w: int = 32, value: float = 1.0) -> np.ndarray:
    img = np.zeros((h, w), dtype=np.float32)
    img[h // 2, w // 2] = value
    return img


def _build_mc(fb: FilterBank, variant: str = "A", **kwargs) -> MADClean:
    detector = IslandDetector(sigma_thresh=2.0, min_island=1, device="cpu")
    if variant == "A":
        solver = PatchSolver(fb, n_nonzero=2, stride=4)
    else:
        solver = ConvSolver(fb, lmbda=0.1, n_iter=10)
    return MADClean(fb, solver, detector, verbose=False, **kwargs)


# ── tests ─────────────────────────────────────────────────────────────────────

def test_numpy_input_accepted():
    """deconvolve() accepts raw numpy arrays for dirty and psf."""
    fb    = _make_fb()
    mc    = _build_mc(fb, n_max=2)
    dirty = _point_source_image(32, 32, value=10.0)
    psf   = _delta_psf(32, 32)
    result = mc.deconvolve(dirty, psf)
    assert "model" in result
    assert "residual" in result
    assert result["model"].shape == dirty.shape
    assert result["residual"].shape == dirty.shape


def test_result_dtype():
    """Returned arrays are float32."""
    fb    = _make_fb()
    mc    = _build_mc(fb, n_max=2)
    result = mc.deconvolve(_point_source_image(), _delta_psf())
    assert result["model"].dtype    == np.float32
    assert result["residual"].dtype == np.float32
    assert result["rms_curve"].dtype == np.float32


def test_n_iter_in_result():
    """n_iter in result matches rms_curve length minus 1."""
    fb    = _make_fb()
    mc    = _build_mc(fb, n_max=3)
    result = mc.deconvolve(_point_source_image(), _delta_psf())
    assert result["n_iter"] == len(result["rms_curve"]) - 1


def test_psf_convention_delta():
    """With a delta PSF, PSF convolution should be identity.
    Verify by checking that _convolve_psf(image, delta_psf_fft) ≈ image.
    """
    fb      = _make_fb()
    mc      = _build_mc(fb, n_max=1)
    H, W    = 32, 32
    image   = torch.from_numpy(_point_source_image(H, W, value=5.0))
    psf_t   = torch.from_numpy(_delta_psf(H, W))
    psf_fft = mc._prepare_psf(psf_t)
    result  = mc._convolve_psf(image, psf_fft)
    np.testing.assert_allclose(result.numpy(), image.numpy(), atol=1e-4)


def test_residual_shape():
    """Residual has the same shape as the dirty image."""
    fb    = _make_fb()
    mc    = _build_mc(fb, n_max=2)
    dirty = np.random.default_rng(0).uniform(0, 0.01, (32, 32)).astype(np.float32)
    psf   = _delta_psf()
    result = mc.deconvolve(dirty, psf)
    assert result["residual"].shape == dirty.shape


def test_no_islands_stops_early():
    """If residual has no emission (all zeros), loop stops at iteration 0."""
    fb    = _make_fb()
    mc    = _build_mc(fb, n_max=100)
    # All-zero dirty image → RMS = 0 → IslandDetector returns [] immediately
    dirty = np.zeros((32, 32), dtype=np.float32)
    result = mc.deconvolve(dirty, _delta_psf())
    assert result["n_iter"] < 100


def test_fits_output_written(tmp_path):
    """If out_dir is given, output FITS files are written to disk."""
    fb    = _make_fb()
    mc    = _build_mc(fb, n_max=2)
    dirty = _point_source_image(32, 32, value=10.0)
    result = mc.deconvolve(dirty, _delta_psf(), out_dir=tmp_path)
    variant = "A"
    assert (tmp_path / f"mad_clean_{variant}_model.fits").exists()
    assert (tmp_path / f"mad_clean_{variant}_residual.fits").exists()
    assert (tmp_path / f"mad_clean_{variant}_rms_curve.npy").exists()


def test_conv_variant_runs():
    """MADClean with ConvSolver (Variant B) runs without error."""
    fb    = _make_fb()
    mc    = _build_mc(fb, variant="B", n_max=2)
    result = mc.deconvolve(_point_source_image(), _delta_psf())
    assert result["model"].shape == (32, 32)


def test_dirty_psf_shape_mismatch_raises():
    """ValueError is raised when dirty and PSF shapes differ."""
    import pytest
    fb    = _make_fb()
    mc    = _build_mc(fb, n_max=1)
    dirty = np.zeros((32, 32), dtype=np.float32)
    psf   = np.zeros((64, 64), dtype=np.float32)
    with pytest.raises(ValueError, match="same shape"):
        mc.deconvolve(dirty, psf)
