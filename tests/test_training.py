"""
tests/test_training.py — smoke tests for PatchDictTrainer and ConvDictTrainer.

All tests use synthetic data — no CRUMB dataset required.
All tests run on CPU only.
"""

import numpy as np
import pytest

from mad_clean.filters    import FilterBank
from mad_clean.patch_dict import PatchDictTrainer
from mad_clean.conv_dict  import ConvDictTrainer


# ── helpers ───────────────────────────────────────────────────────────────────

def _synthetic_images(
    n: int = 12,
    h: int = 40,
    w: int = 40,
    seed: int = 0,
) -> np.ndarray:
    """
    Small synthetic radio-like images: sparse Gaussian blobs on a zero background.
    Using h=w=40 keeps unit-test runtime fast while still being larger than the
    atom size (F=5) so the convolutional model is non-degenerate.
    """
    rng    = np.random.default_rng(seed)
    images = np.zeros((n, h, w), dtype=np.float32)
    for i in range(n):
        n_blobs = rng.integers(1, 4)
        for _ in range(n_blobs):
            cy     = rng.integers(5, h - 5)
            cx     = rng.integers(5, w - 5)
            sigma  = rng.uniform(1.5, 4.0)
            amp    = rng.uniform(0.5, 1.0)
            y      = np.arange(h) - cy
            x      = np.arange(w) - cx
            yy, xx = np.meshgrid(y, x, indexing="ij")
            images[i] += amp * np.exp(-(yy ** 2 + xx ** 2) / (2 * sigma ** 2))
    return images.clip(0, 1)


# ── PatchDictTrainer ──────────────────────────────────────────────────────────

def test_patch_trainer_returns_filterbank():
    """PatchDictTrainer.fit() returns a FilterBank instance."""
    imgs    = _synthetic_images(n=10)
    trainer = PatchDictTrainer(k=4, atom_size=5, n_iter=5, patches_per_img=5)
    fb      = trainer.fit(imgs, device="cpu")
    assert isinstance(fb, FilterBank)


def test_patch_trainer_filterbank_shape():
    """Returned FilterBank has correct K and F."""
    k, f    = 6, 5
    imgs    = _synthetic_images(n=10)
    trainer = PatchDictTrainer(k=k, atom_size=f, n_iter=5, patches_per_img=5)
    fb      = trainer.fit(imgs, device="cpu")
    assert fb.K == k
    assert fb.F == f


def test_patch_trainer_atoms_normalised():
    """Atoms returned by PatchDictTrainer are unit L2 norm."""
    imgs    = _synthetic_images(n=10)
    trainer = PatchDictTrainer(k=4, atom_size=5, n_iter=5, patches_per_img=5)
    fb      = trainer.fit(imgs, device="cpu")
    norms   = fb.atoms.reshape(fb.K, -1).norm(dim=1)
    np.testing.assert_allclose(norms.numpy(), np.ones(fb.K), atol=1e-5)


def test_patch_trainer_save(tmp_path):
    """PatchDictTrainer.save() writes a .npy file that can be loaded."""
    imgs    = _synthetic_images(n=8)
    trainer = PatchDictTrainer(k=4, atom_size=5, n_iter=3, patches_per_img=3)
    fb      = trainer.fit(imgs, device="cpu")
    path    = tmp_path / "patch_atoms.npy"
    trainer.save(path, fb)
    assert path.exists()
    fb2 = FilterBank.load(path)
    assert fb2.K == fb.K


# ── ConvDictTrainer ───────────────────────────────────────────────────────────

def test_conv_trainer_returns_filterbank():
    """ConvDictTrainer.fit() returns a FilterBank instance."""
    imgs    = _synthetic_images(n=10)
    trainer = ConvDictTrainer(k=4, atom_size=5, batch_size=4, n_epochs=2,
                               fista_iter_train=5)
    fb      = trainer.fit(imgs, device="cpu")
    assert isinstance(fb, FilterBank)


def test_conv_trainer_filterbank_shape():
    """Returned FilterBank has the requested K and F."""
    k, f    = 6, 5
    imgs    = _synthetic_images(n=10)
    trainer = ConvDictTrainer(k=k, atom_size=f, batch_size=4, n_epochs=2,
                               fista_iter_train=5)
    fb      = trainer.fit(imgs, device="cpu")
    assert fb.K == k
    assert fb.F == f


def test_conv_trainer_atoms_normalised():
    """Atoms returned by ConvDictTrainer are unit L2 norm (unit-ball projection)."""
    imgs    = _synthetic_images(n=8)
    trainer = ConvDictTrainer(k=4, atom_size=5, batch_size=4, n_epochs=2,
                               fista_iter_train=5)
    fb      = trainer.fit(imgs, device="cpu")
    norms   = fb.atoms.reshape(fb.K, -1).norm(dim=1)
    np.testing.assert_allclose(norms.numpy(), np.ones(fb.K), atol=1e-4)


def test_conv_trainer_loss_decreases():
    """Reconstruction loss at epoch 3 is lower than at epoch 1 (on synthetic data)."""
    # Patch ConvDictTrainer to capture per-epoch loss.
    # We run two short trainers and compare their final losses as a proxy.
    imgs     = _synthetic_images(n=12, seed=7)
    # Few epochs — loss should still trend downward over 4 epochs on this data.
    trainer1 = ConvDictTrainer(k=4, atom_size=5, batch_size=4, n_epochs=1,
                                fista_iter_train=10, random_seed=0)
    trainer4 = ConvDictTrainer(k=4, atom_size=5, batch_size=4, n_epochs=4,
                                fista_iter_train=10, random_seed=0)
    fb1 = trainer1.fit(imgs, device="cpu")
    fb4 = trainer4.fit(imgs, device="cpu")

    # Measure reconstruction error on training images with each filter bank.
    from mad_clean.solvers import ConvSolver
    import torch

    def _mean_recon_error(fb: FilterBank, images: np.ndarray) -> float:
        solver = ConvSolver(fb, lmbda=0.1, n_iter=20)
        errors = []
        for img_np in images[:6]:
            img   = torch.from_numpy(img_np)
            recon = solver.decode_island(img)
            errors.append(float((img - recon).pow(2).mean()))
        return float(np.mean(errors))

    err1 = _mean_recon_error(fb1, imgs)
    err4 = _mean_recon_error(fb4, imgs)
    # 4 epochs of training should not be worse than 1 epoch (5% slack for noise)
    assert err4 <= err1 * 1.05, f"err(4 epochs)={err4:.4f} > err(1 epoch)={err1:.4f}"


def test_conv_trainer_save(tmp_path):
    """ConvDictTrainer.save() writes a loadable .npy file."""
    imgs    = _synthetic_images(n=8)
    trainer = ConvDictTrainer(k=4, atom_size=5, batch_size=4, n_epochs=1,
                               fista_iter_train=5)
    fb      = trainer.fit(imgs, device="cpu")
    path    = tmp_path / "conv_atoms.npy"
    trainer.save(path, fb)
    assert path.exists()
    fb2 = FilterBank.load(path)
    assert fb2.K == fb.K
