"""End-to-end integration: CutoutDataset → MDNAsp → nll_loss → backward.

Goes deeper than the per-module smoke tests by:
  - running multiple training steps,
  - asserting loss stays finite (no NaN / +inf) throughout,
  - asserting at least one of the spike-recovery training dynamics works
    (loss either decreases overall or stays bounded; never NaNs out).

Runs on CPU in <5 s with a tiny MDN.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest
import torch
from astropy.io import fits

from mad_clean.data.cutout_dataset import (
    CutoutDataset, LOG_FLUX_SCALE, unstandardise_log_flux,
)
from mad_clean.data.psf_bank import PSFBank
from mad_clean.models.mdn_asp import MDNAsp


CUTOUT = 128
FIELD  = 256


@pytest.fixture(scope="module")
def psf_bank(tmp_path_factory) -> PSFBank:
    tmp = tmp_path_factory.mktemp("psfs_int")
    p = tmp / "psf.fits"
    half = CUTOUT // 2
    y, x = np.mgrid[0:CUTOUT, 0:CUTOUT]
    g = np.exp(-0.5 * ((x - half) ** 2 + (y - half) ** 2) / 3.0 ** 2)
    fits.PrimaryHDU((g / g.max()).astype(np.float32)).writeto(str(p))
    return PSFBank([p], target_size=CUTOUT, rotation_augment=False)


@pytest.fixture(scope="module")
def small_dataset(psf_bank) -> CutoutDataset:
    return CutoutDataset(
        psf_bank=psf_bank,
        field_size=FIELD,
        cutout_size=CUTOUT,
        sigma_noise=1e-4,
        n_sources_per_field=(3, 6),
        extended_fraction=0.5,
        rng_seed=0,
        length=4,
    )


def _stack_batch(ds: CutoutDataset, n: int):
    res, psf, cond, target = [], [], [], []
    for i in range(n):
        r, p, c, t = ds[i]
        res.append(r); psf.append(p); cond.append(c); target.append(t)
    image = torch.stack([torch.stack(res), torch.stack(psf)], dim=1)
    return image, torch.stack(cond), torch.stack(target)


def test_full_stack_step_runs_no_nan(small_dataset):
    """Single training step on a tiny model — no NaN, finite loss, grads flow."""
    image, cond, target = _stack_batch(small_dataset, 4)
    model = MDNAsp(base_channels=4, hidden=32, n_components=3, cond_dim=5)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    params = model(image, cond)
    loss = model.nll_loss(params, target)
    assert torch.isfinite(loss), f"loss not finite: {loss.item()}"
    opt.zero_grad()
    loss.backward()
    # at least one parameter received a non-zero grad
    nonzero = [
        p.grad.abs().sum().item() > 0
        for p in model.parameters()
        if p.grad is not None
    ]
    assert any(nonzero), "no parameter received a non-zero gradient"
    opt.step()


def test_short_training_loop_bounded_loss(small_dataset):
    """20 steps on a tiny model; loss must stay finite throughout."""
    image, cond, target = _stack_batch(small_dataset, 4)
    torch.manual_seed(0)
    model = MDNAsp(base_channels=4, hidden=32, n_components=3, cond_dim=5)
    opt = torch.optim.Adam(model.parameters(), lr=5e-4)
    losses = []
    for _step in range(20):
        opt.zero_grad()
        params = model(image, cond)
        loss = model.nll_loss(params, target)
        assert torch.isfinite(loss), f"loss diverged: {loss.item()}"
        loss.backward()
        opt.step()
        losses.append(float(loss.item()))
    # Loss should have moved (network learned something), but we do NOT
    # assert a specific direction or magnitude — this is a contract test,
    # not a performance gate.
    assert not all(abs(losses[0] - l) < 1e-9 for l in losses[1:]), (
        "loss did not change across 20 steps — gradient probably not flowing"
    )


def test_mode_decode_log_flux_roundtrip(small_dataset):
    """The target's log_flux (standardised) can be unstandardised back to a
    physically reasonable Jy/beam range."""
    _, _, target = _stack_batch(small_dataset, 4)
    for i in range(4):
        z = float(target[i, 2])
        log_f = unstandardise_log_flux(z)
        # Default flux range is (1e-4, 1e-1) Jy → log_f in [-9.21, -2.30].
        # Allow 1-nat slack for floating-point and standardisation accuracy.
        assert -10.5 < log_f < -1.5, (
            f"unstandardised log_flux {log_f:.2f} outside plausible range"
        )


def test_pa_error_metric_period_pi():
    """Pin the script's PA error convention: distance under the (sin 2θ, cos 2θ)
    encoding is min(|Δ|, π − |Δ|). θ=+π/2 and θ=−π/2 are the same point
    (error ≈ 0), NOT distance π.
    """
    def pa_err(a, b):
        d = abs(a - b)
        return min(d, math.pi - d)

    assert pa_err(math.pi / 2, -math.pi / 2) < 1e-9
    assert abs(pa_err(0.3, 0.3 + math.pi) - 0.0) < 1e-9
    assert abs(pa_err(0.0, math.pi / 4) - math.pi / 4) < 1e-9
    assert abs(pa_err(0.0, math.pi / 2) - math.pi / 2) < 1e-9
