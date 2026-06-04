"""Tests for PatchFlow GPU data generation pipeline."""
import math
import tempfile
from pathlib import Path

import pytest
import torch

from mad_clean.data.extended_sky_torch import (
    BEAM_SIGMA_PX,
    SIGMA_MAX_PX,
    fft_convolve_batch,
    render_gaussian_blob_batch,
    render_shell_batch,
    render_filament_batch,
)

DEVICE = torch.device("cpu")
SIZE   = 64   # small for fast tests
B      = 4


def _rand(shape, lo, hi):
    return torch.empty(shape).uniform_(lo, hi)


# ---------------------------------------------------------------------------
# extended_sky_torch
# ---------------------------------------------------------------------------

def test_blob_batch_shape():
    cx  = _rand(B, 16, SIZE - 16)
    cy  = _rand(B, 16, SIZE - 16)
    flux = torch.ones(B)
    sig_maj = _rand(B, BEAM_SIGMA_PX, 20.0)
    sig_min = torch.stack([_rand(1, BEAM_SIGMA_PX, float(s))[0] for s in sig_maj])
    pa   = _rand(B, 0, math.pi)
    out  = render_gaussian_blob_batch(SIZE, cx, cy, flux, sig_maj, sig_min, pa, DEVICE)
    assert out.shape == (B, SIZE, SIZE)


def test_blob_batch_flux_conservation():
    cx  = torch.full((B,), SIZE / 2)
    cy  = torch.full((B,), SIZE / 2)
    flux = _rand(B, 0.1, 1.0)
    sig_maj = torch.full((B,), 5.0)
    sig_min = torch.full((B,), 5.0)
    pa   = torch.zeros(B)
    out  = render_gaussian_blob_batch(SIZE, cx, cy, flux, sig_maj, sig_min, pa, DEVICE)
    for b in range(B):
        assert abs(float(out[b].sum()) - float(flux[b])) < 1e-3


def test_shell_batch_shape():
    cx     = _rand(B, 16, SIZE - 16)
    cy     = _rand(B, 16, SIZE - 16)
    flux   = torch.ones(B)
    radius = _rand(B, BEAM_SIGMA_PX, 15.0)
    thick  = _rand(B, 1.0, 5.0)
    out    = render_shell_batch(SIZE, cx, cy, flux, radius, thick, DEVICE)
    assert out.shape == (B, SIZE, SIZE)


def test_shell_batch_flux_conservation():
    cx     = torch.full((B,), SIZE / 2)
    cy     = torch.full((B,), SIZE / 2)
    flux   = _rand(B, 0.1, 1.0)
    radius = torch.full((B,), 10.0)
    thick  = torch.full((B,), 2.0)
    out    = render_shell_batch(SIZE, cx, cy, flux, radius, thick, DEVICE)
    for b in range(B):
        assert abs(float(out[b].sum()) - float(flux[b])) < 1e-3


def test_filament_batch_shape():
    cx     = _rand(B, 16, SIZE - 16)
    cy     = _rand(B, 16, SIZE - 16)
    flux   = torch.ones(B)
    length = _rand(B, 4.0, 20.0)
    width  = _rand(B, BEAM_SIGMA_PX, 5.0)
    pa     = _rand(B, 0, math.pi)
    out    = render_filament_batch(SIZE, cx, cy, flux, length, width, pa, DEVICE)
    assert out.shape == (B, SIZE, SIZE)


def test_filament_batch_flux_conservation():
    cx     = torch.full((B,), SIZE / 2)
    cy     = torch.full((B,), SIZE / 2)
    flux   = _rand(B, 0.1, 1.0)
    length = torch.full((B,), 10.0)
    width  = torch.full((B,), 2.0)
    pa     = torch.zeros(B)
    out    = render_filament_batch(SIZE, cx, cy, flux, length, width, pa, DEVICE)
    for b in range(B):
        assert abs(float(out[b].sum()) - float(flux[b])) < 1e-3


def test_fft_convolve_batch_shape():
    images = torch.randn(B, SIZE, SIZE)
    psf    = torch.zeros(SIZE, SIZE)
    psf[SIZE // 2, SIZE // 2] = 1.0   # delta PSF
    out    = fft_convolve_batch(images, psf)
    assert out.shape == (B, SIZE, SIZE)


def test_fft_convolve_batch_delta_psf():
    """Convolution with delta PSF should return the image unchanged."""
    images = torch.randn(B, SIZE, SIZE)
    psf    = torch.zeros(SIZE, SIZE)
    psf[SIZE // 2, SIZE // 2] = 1.0
    out    = fft_convolve_batch(images, psf)
    assert torch.allclose(out, images, atol=1e-4)


# ---------------------------------------------------------------------------
# Disk dataset
# ---------------------------------------------------------------------------

def test_disk_dataset_roundtrip():
    from mad_clean.data.patch_flow_dataset_disk import PatchFlowDatasetDisk

    S = 128
    N = 8
    shard = {
        "dirty": torch.randn(N, 1, S, S),
        "clean": torch.randn(N, 1, S, S),
        "psf":   torch.rand(N, 1, S, S),
        "sigma": torch.rand(N),
    }
    with tempfile.TemporaryDirectory() as tmp:
        torch.save(shard, Path(tmp) / "shard_0000.pt")
        ds = PatchFlowDatasetDisk(tmp)
        assert len(ds) == N
        dirty, psf, sigma, clean = ds[0]
        assert dirty.shape == (1, S, S)
        assert clean.shape == (1, S, S)
        assert psf.shape   == (1, S, S)
        assert sigma.shape == ()
