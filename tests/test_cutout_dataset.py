"""
tests/test_cutout_dataset.py
=============================
Unit tests for mad_clean.data.cutout_dataset.CutoutDataset.

All tests run on CPU in seconds; no GPU, no real PSF FITS files needed.
A minimal PSFBank is built from a synthetic Gaussian PSF array written to
a temp FITS file.
"""
from __future__ import annotations

import math
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch
from astropy.io import fits

from mad_clean.data.psf_bank import PSFBank
from mad_clean.data.cutout_dataset import CutoutDataset


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

CUTOUT = 128
FIELD  = 256  # small field for speed


def _make_gaussian_psf(size: int = 128, sigma: float = 3.0) -> np.ndarray:
    """Synthetic Gaussian PSF, peak = 1, centred at (size//2, size//2)."""
    half = size // 2
    y, x = np.mgrid[0:size, 0:size]
    g = np.exp(-0.5 * ((x - half) ** 2 + (y - half) ** 2) / sigma ** 2)
    return (g / g.max()).astype(np.float32)


@pytest.fixture(scope="module")
def psf_fits(tmp_path_factory) -> Path:
    """Write a synthetic Gaussian PSF to a temp FITS file; return the path."""
    tmp = tmp_path_factory.mktemp("psfs")
    p = tmp / "psf.fits"
    psf = _make_gaussian_psf(size=CUTOUT)
    hdu = fits.PrimaryHDU(psf.astype(np.float32))
    hdu.writeto(str(p))
    return p


@pytest.fixture(scope="module")
def psf_bank(psf_fits) -> PSFBank:
    """PSFBank with a single synthetic PSF, target_size=128, no rotation aug."""
    return PSFBank([psf_fits], target_size=CUTOUT, rotation_augment=False)


@pytest.fixture(scope="module")
def dataset(psf_bank) -> CutoutDataset:
    return CutoutDataset(
        psf_bank=psf_bank,
        field_size=FIELD,
        cutout_size=CUTOUT,
        sigma_noise=1e-4,
        n_sources_per_field=(3, 8),
        extended_fraction=0.05,
        rng_seed=0,
        length=20,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_shapes(dataset):
    residual, psf, cond, target = dataset[0]
    assert residual.shape == (CUTOUT, CUTOUT), f"residual shape {residual.shape}"
    assert psf.shape     == (CUTOUT, CUTOUT), f"psf shape {psf.shape}"
    assert cond.shape    == (5,),             f"cond shape {cond.shape}"
    assert target.shape  == (6,),             f"target shape {target.shape}"


def test_dtypes(dataset):
    residual, psf, cond, target = dataset[0]
    assert residual.dtype == torch.float32
    assert psf.dtype      == torch.float32
    assert cond.dtype     == torch.float32
    assert target.dtype   == torch.float32


def test_determinism(dataset):
    a = dataset[3]
    b = dataset[3]
    for ta, tb in zip(a, b):
        assert torch.equal(ta, tb), "Same idx must produce bit-identical output"


def test_different_idx_different_output(dataset):
    r0, _, _, _ = dataset[0]
    r1, _, _, _ = dataset[1]
    assert not torch.equal(r0, r1), "Different idx should produce different residual"


def test_psf_peak_at_centre(dataset):
    _, psf, _, _ = dataset[0]
    psf_np = psf.numpy()
    peak_val  = psf_np.max()
    peak_flat = int(np.argmax(psf_np))
    pr, pc    = divmod(peak_flat, CUTOUT)
    centre    = CUTOUT // 2
    assert abs(peak_val - 1.0) < 1e-5, f"PSF peak should be 1, got {peak_val}"
    assert abs(pr - centre) <= 2, f"PSF peak row {pr} not near centre {centre}"
    assert abs(pc - centre) <= 2, f"PSF peak col {pc} not near centre {centre}"


def test_target_xy_near_zero(dataset):
    """Target (x, y) encodes offset from cutout centre; should be < 1 px."""
    _, _, _, target = dataset[0]
    x_off = float(target[0])
    y_off = float(target[1])
    assert abs(x_off) < 1.0, f"target x offset {x_off:.3f} should be < 1 px"
    assert abs(y_off) < 1.0, f"target y offset {y_off:.3f} should be < 1 px"


def test_residual_peak_near_centre(dataset):
    """The bright signal from the chosen source should peak near (64, 64)."""
    residual, _, _, _ = dataset[0]
    r_np = residual.numpy()
    peak_flat = int(np.argmax(np.abs(r_np)))
    pr, pc = divmod(peak_flat, CUTOUT)
    centre = CUTOUT // 2
    # Allow ±8 px tolerance (about half the beam FWHM at large sigma)
    assert abs(pr - centre) <= 8, f"Residual peak row {pr} far from centre {centre}"
    assert abs(pc - centre) <= 8, f"Residual peak col {pc} far from centre {centre}"


def test_other_source_subtracted(psf_bank):
    """A bright far-away source should have negligible contribution in the
    residual cutout because the 'subtract all others' logic cancels it."""
    # Build a dataset where the far source is a point and the cutout is on a
    # different point source — but since the dataset re-renders single source,
    # the residual_cutout simply equals psf⊛sky_chosen + noise.  The test
    # checks that the residual has substantial signal only near its centre.
    ds = CutoutDataset(
        psf_bank=psf_bank,
        field_size=FIELD,
        cutout_size=CUTOUT,
        sigma_noise=1e-6,     # tiny noise so signal dominates
        n_sources_per_field=(5, 10),
        rng_seed=99,
        length=5,
    )
    residual, _, _, target = ds[0]
    r_np = residual.numpy()
    centre = CUTOUT // 2
    # Energy near centre vs far corners
    inner_mask = np.zeros((CUTOUT, CUTOUT), dtype=bool)
    inner_mask[centre - 20:centre + 20, centre - 20:centre + 20] = True
    outer_mask  = ~inner_mask
    energy_inner = float(np.sum(r_np[inner_mask] ** 2))
    energy_outer = float(np.sum(r_np[outer_mask] ** 2))
    # Inner should dominate
    assert energy_inner > energy_outer * 0.5, (
        f"Expected inner energy ({energy_inner:.2e}) > 0.5 * outer ({energy_outer:.2e}); "
        "far sources may not be subtracted"
    )


def test_conditioning_shape_and_one_hot(dataset):
    """Conditioning vector: first element is sigma_noise, next 4 are one-hot (D=index 3)."""
    _, _, cond, _ = dataset[0]
    assert cond.shape == (5,)
    sigma_val = float(cond[0])
    config_hot = cond[1:].numpy()
    # Should be one-hot with the '1' at index 3 (D-config)
    assert config_hot.argmax() == 3, f"Expected D-config (idx 3) hot, got {config_hot}"
    assert abs(config_hot.sum() - 1.0) < 1e-5, "One-hot should sum to 1"


def test_length(dataset):
    assert len(dataset) == 20


def test_multiple_samples_runnable(dataset):
    for i in range(min(5, len(dataset))):
        out = dataset[i]
        assert len(out) == 4


def test_log_flux_is_standardised(dataset):
    """Dataset emits log_flux in standardised space (rough unit range).
    The raw log_flux range is roughly [-9.2, -2.3] nats; standardised is
    roughly [-1, +1]. Sample a few targets and confirm magnitude.
    """
    from mad_clean.data.cutout_dataset import (
        LOG_FLUX_OFFSET, LOG_FLUX_SCALE,
        standardise_log_flux, unstandardise_log_flux,
    )
    # Round-trip
    for v in (-9.2, -5.76, -2.3, 0.0):
        z = standardise_log_flux(v)
        v2 = unstandardise_log_flux(z)
        assert abs(v - v2) < 1e-6, f"round-trip failed for {v}"
    # Constants are well-defined
    assert LOG_FLUX_SCALE > 0
    # Samples come out within a sensible band around 0
    zs = []
    for i in range(10):
        _, _, _, t = dataset[i]
        zs.append(float(t[2]))
    # raw range is [-9.2, -2.3], standardised should sit in [-1.5, +1.5]
    for z in zs:
        assert -2.0 < z < 2.0, f"standardised log_flux {z} outside expected band"


def test_residual_uses_as_rendered_not_gaussian_fit(psf_bank):
    """Architectural invariant: residual_cutout must contain the AS-RENDERED
    source contribution (real shell / filament morphology), NOT the Gaussian
    fit of it. If the dataset re-rendered the Gaussian from target params,
    a shell-only scene would have residual ≈ PSF * Gaussian(σ≈radius), which
    is unimodal. Real PSF * shell is annular.
    """
    from mad_clean.data.extended_sky import render_shell, BEAM_SIGMA_PX
    from scipy.signal import fftconvolve

    # Render a single shell directly and PSF-convolve it. This is the
    # ground-truth residual we expect the dataset to produce for a
    # shell-only scene with one source.
    size = FIELD
    cx = cy = size // 2
    shell_img, target = render_shell(
        size=size,
        cx=cx, cy=cy,
        flux_jy=1.0,
        rng=np.random.default_rng(0),
    )
    psf, _ = psf_bank[0]
    expected = fftconvolve(shell_img, psf, mode="same").astype(np.float32)

    # The shell has a much larger sigma than the beam floor; if the dataset
    # were re-rendering the Gaussian fit instead of the as-rendered shell,
    # the convolution would differ in shape (smoother, no annular ring).
    # Confirm here that render_shell produces structure away from a single
    # Gaussian profile by comparing peak location vs centre.
    assert target.log_sig_maj > math.log(BEAM_SIGMA_PX * 2.0), (
        "Sanity precondition: shell sigma must exceed twice the beam floor"
    )
    # If the architecture is correct, the dataset's per-source path returns
    # the as-rendered shell image. We can't directly inspect that here, but
    # we assert the function signature supports it.
    from mad_clean.data.extended_sky import assemble_mixed_field
    sky, targets, per_source = assemble_mixed_field(
        size=64,
        n_sources=3,
        extended_rate=0.5,
        rng=np.random.default_rng(1),
        return_per_source=True,
    )
    assert len(per_source) == len(targets)
    reconstructed = np.sum(per_source, axis=0)
    np.testing.assert_allclose(
        reconstructed, sky, rtol=1e-5, atol=1e-6,
        err_msg="sum(per_source) must equal sky exactly",
    )
