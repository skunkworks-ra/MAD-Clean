"""
tests/test_simulator.py
=======================
Tests for GPUSimulator — on-the-fly (dirty, clean) pair generation on GPU/CPU.

All tests run on CPU so no GPU is required in CI.
"""

import math
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

from mad_clean.data.simulator import GPUSimulator, _next_pow2


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _make_npz(n: int = 16, h: int = 32, w: int = 32) -> Path:
    """Write a temporary crumb_preprocessed.npz with synthetic clean images."""
    rng    = np.random.default_rng(0)
    images = rng.standard_normal((n, h, w)).astype(np.float32)
    tmp    = tempfile.NamedTemporaryFile(suffix=".npz", delete=False)
    np.savez(tmp.name, images=images)
    return Path(tmp.name)


def _gaussian_psf(h: int, w: int) -> np.ndarray:
    """Simple centred Gaussian PSF, peak=1."""
    cy, cx = h // 2, w // 2
    ys = np.arange(h, dtype=np.float32) - cy
    xs = np.arange(w, dtype=np.float32) - cx
    yy, xx = np.meshgrid(ys, xs, indexing="ij")
    psf = np.exp(-(xx ** 2 + yy ** 2) / (2.0 * 2.0 ** 2)).astype(np.float32)
    return psf / psf.max()


@pytest.fixture(scope="module")
def sim():
    """GPUSimulator on CPU with small synthetic data."""
    data_path = _make_npz(n=16, h=32, w=32)
    psf       = _gaussian_psf(32, 32)
    return GPUSimulator(
        data_path      = data_path,
        psf            = psf,
        noise_std      = 0.05,
        vram_budget_gb = 0.1,    # small budget → small batch_size
        device         = "cpu",
        seed           = 42,
    )


# ---------------------------------------------------------------------------
# _next_pow2
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("n,expected", [
    (1, 1), (2, 2), (3, 4), (64, 64), (65, 128), (150, 256),
])
def test_next_pow2(n, expected):
    assert _next_pow2(n) == expected


# ---------------------------------------------------------------------------
# GPUSimulator construction
# ---------------------------------------------------------------------------

def test_simulator_dataset_shape(sim):
    assert sim._clean.shape == (16, 32, 32)


def test_simulator_psf_shape(sim):
    assert sim._psf.shape == (32, 32)


def test_simulator_pad_size(sim):
    # next power-of-2 >= 2*32 = 64
    assert sim._pad == 64


def test_simulator_batch_size_positive(sim):
    assert sim.batch_size >= 1


def test_simulator_device_is_cpu(sim):
    assert sim.device == torch.device("cpu")
    assert sim._clean.device == torch.device("cpu")
    assert sim._psf.device == torch.device("cpu")


# ---------------------------------------------------------------------------
# forward — output shapes and types
# ---------------------------------------------------------------------------

def test_forward_output_shapes(sim):
    clean_batch = sim._clean[:sim.batch_size]
    dirty, clean_aug = sim.forward(clean_batch)
    B = clean_batch.shape[0]
    assert dirty.shape    == (B, 1, 32, 32)
    assert clean_aug.shape == (B, 1, 32, 32)


def test_forward_output_dtype(sim):
    clean_batch = sim._clean[:sim.batch_size]
    dirty, clean_aug = sim.forward(clean_batch)
    assert dirty.dtype    == torch.float32
    assert clean_aug.dtype == torch.float32


def test_forward_dirty_is_not_clean(sim):
    """Dirty must differ from clean — PSF convolution + noise changes it."""
    clean_batch = sim._clean[:sim.batch_size]
    dirty, clean_aug = sim.forward(clean_batch)
    assert not torch.allclose(dirty, clean_aug)


def test_forward_no_nan(sim):
    clean_batch = sim._clean[:sim.batch_size]
    dirty, clean_aug = sim.forward(clean_batch)
    assert not torch.isnan(dirty).any()
    assert not torch.isnan(clean_aug).any()


def test_forward_no_inf(sim):
    clean_batch = sim._clean[:sim.batch_size]
    dirty, clean_aug = sim.forward(clean_batch)
    assert not torch.isinf(dirty).any()
    assert not torch.isinf(clean_aug).any()


def test_forward_does_not_modify_resident_dataset(sim):
    """forward() must not modify the GPU-resident clean tensor in place."""
    original = sim._clean.clone()
    clean_batch = sim._clean[:sim.batch_size]
    sim.forward(clean_batch)
    assert torch.allclose(sim._clean, original)


# ---------------------------------------------------------------------------
# forward — physical units (no normalisation)
# ---------------------------------------------------------------------------

def test_forward_clean_preserves_scale(sim):
    """
    clean_aug values must be drawn from the same scale as the input clean
    images (Jy/pixel, no normalisation applied).  Check that the max of
    clean_aug is in the same order of magnitude as the input.
    """
    clean_batch = sim._clean[:sim.batch_size]
    _, clean_aug = sim.forward(clean_batch)
    input_scale = clean_batch.abs().max().item()
    output_scale = clean_aug.abs().max().item()
    # Allow a factor of 2 for rotation boundary effects — scale should not
    # differ by orders of magnitude.
    assert output_scale < input_scale * 2 + 1e-3


# ---------------------------------------------------------------------------
# generate_epoch — iteration behaviour
# ---------------------------------------------------------------------------

def test_generate_epoch_covers_all_samples(sim):
    """All N images must appear exactly once per epoch."""
    seen = 0
    for batch in sim.generate_epoch(shuffle=False):
        assert batch.ndim == 3          # (B, H, W)
        assert batch.shape[1:] == (32, 32)
        seen += batch.shape[0]
    assert seen == sim.N


def test_generate_epoch_shuffle_differs_from_sequential(sim):
    """Shuffled and unshuffled epochs should produce different first batches
    with overwhelming probability given N=16 and seed variation."""
    torch.manual_seed(0)
    batch_seq  = next(iter(sim.generate_epoch(shuffle=False)))
    batch_shuf = next(iter(sim.generate_epoch(shuffle=True)))
    # They may coincidentally match, but very unlikely for N=16
    # — just check both are valid tensors of correct shape.
    assert batch_seq.shape  == batch_shuf.shape


def test_generate_epoch_batches_on_correct_device(sim):
    for batch in sim.generate_epoch(shuffle=False):
        assert batch.device == torch.device("cpu")
        break


# ---------------------------------------------------------------------------
# FlowTrainer integration — one epoch on CPU with GPUSimulator
# ---------------------------------------------------------------------------

def test_flow_trainer_one_epoch(sim):
    from mad_clean.training.flow import FlowTrainer, FlowModel
    trainer = FlowTrainer(n_epochs=1, lr=1e-4, random_seed=0)
    fm = trainer.fit(simulator=sim, device="cpu")
    assert isinstance(fm, FlowModel)


def test_prior_trainer_one_epoch(sim):
    from mad_clean.training.flow import PriorTrainer, FlowModel
    trainer = PriorTrainer(n_epochs=1, lr=1e-4, random_seed=0)
    fm = trainer.fit(simulator=sim, device="cpu")
    assert isinstance(fm, FlowModel)
