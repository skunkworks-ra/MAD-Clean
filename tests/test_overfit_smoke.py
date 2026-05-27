"""Smoke test for scripts/overfit_one_batch.py core loop.

Validates:
  - Dataset + model build without error.
  - 5 gradient steps run on CPU.
  - Loss is finite and decreases (or at least not diverges).

No GPU, no real PSF data needed.  Uses the same synthetic Gaussian PSF
fixture pattern as tests/test_cutout_dataset.py.
"""
from __future__ import annotations

import math
import tempfile
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
import torch.optim as optim  # noqa: E402 — after importorskip

from astropy.io import fits

from mad_clean.data.psf_bank import PSFBank
from mad_clean.data.cutout_dataset import CutoutDataset
from mad_clean.models.mdn_asp import MDNAsp


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------

CUTOUT = 128
N_SAMPLES = 4
STEPS = 5


def _make_gaussian_psf(size: int = 128, sigma: float = 3.0) -> np.ndarray:
    half = size // 2
    y, x = np.mgrid[0:size, 0:size]
    g = np.exp(-0.5 * ((x - half) ** 2 + (y - half) ** 2) / sigma ** 2)
    return (g / g.max()).astype(np.float32)


@pytest.fixture(scope="module")
def psf_fits(tmp_path_factory) -> Path:
    tmp = tmp_path_factory.mktemp("smoke_psfs")
    p = tmp / "psf.fits"
    psf = _make_gaussian_psf(size=CUTOUT)
    hdu = fits.PrimaryHDU(psf.astype(np.float32))
    hdu.writeto(str(p))
    return p


@pytest.fixture(scope="module")
def psf_bank(psf_fits) -> PSFBank:
    return PSFBank([psf_fits], target_size=CUTOUT, rotation_augment=False)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_overfit_smoke(psf_bank):
    """Build dataset, pull frozen batch, run 5 gradient steps; loss stays finite."""
    device = torch.device("cpu")

    dataset = CutoutDataset(
        psf_bank=psf_bank,
        field_size=256,
        cutout_size=CUTOUT,
        sigma_noise=1e-4,
        n_sources_per_field=(3, 8),
        extended_fraction=0.0,   # points only for speed
        rng_seed=7,
        length=N_SAMPLES,
    )

    residuals, psfs, conds, targets = [], [], [], []
    for i in range(N_SAMPLES):
        r, p, c, t = dataset[i]
        residuals.append(r)
        psfs.append(p)
        conds.append(c)
        targets.append(t)

    res_t  = torch.stack(residuals)
    psf_t  = torch.stack(psfs)
    image  = torch.stack([res_t, psf_t], dim=1).to(device)  # (B, 2, H, W)
    cond   = torch.stack(conds).to(device)                   # (B, 5)
    target = torch.stack(targets).to(device)                 # (B, 6)

    # Small network for CPU speed
    model = MDNAsp(base_channels=4, hidden=32, n_components=5, cond_dim=5).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    losses = []
    for _ in range(STEPS):
        model.train()
        optimizer.zero_grad()
        params = model(image, cond)
        loss = model.nll_loss(params, target)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.item()))

    assert all(math.isfinite(l) for l in losses), (
        f"Loss became non-finite during overfit smoke: {losses}"
    )
    # Weak check: loss must not have diverged (last step ≤ 10× first step)
    assert losses[-1] <= losses[0] * 10.0, (
        f"Loss diverged: first={losses[0]:.4f}, last={losses[-1]:.4f}"
    )


def test_mode_shape(psf_bank):
    """model.mode() returns (B, 6) on a forward pass."""
    device = torch.device("cpu")

    dataset = CutoutDataset(
        psf_bank=psf_bank,
        field_size=256,
        cutout_size=CUTOUT,
        sigma_noise=1e-4,
        n_sources_per_field=(2, 5),
        rng_seed=13,
        length=2,
    )
    r0, p0, c0, _ = dataset[0]
    r1, p1, c1, _ = dataset[1]

    image  = torch.stack([torch.stack([r0, p0]), torch.stack([r1, p1])])  # (2,2,H,W)
    cond   = torch.stack([c0, c1])                                         # (2,5)

    model = MDNAsp(base_channels=4, hidden=32).to(device)
    model.eval()
    with torch.no_grad():
        params = model(image, cond)
        mode   = model.mode(params)

    assert mode.shape == (2, 6), f"Expected (2, 6), got {tuple(mode.shape)}"
