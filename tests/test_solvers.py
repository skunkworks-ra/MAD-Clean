"""
tests/test_solvers.py — tests for mad_clean.solvers.PatchSolver and ConvSolver
"""

import numpy as np
import torch

from mad_clean.filters import FilterBank
from mad_clean.solvers import PatchSolver, ConvSolver


# ── helpers ───────────────────────────────────────────────────────────────────

def _make_fb(k: int = 8, f: int = 5, seed: int = 0) -> FilterBank:
    rng   = np.random.default_rng(seed)
    atoms = rng.standard_normal((k, f, f)).astype(np.float32)
    return FilterBank(atoms, device="cpu")


def _gaussian_blob(size: int = 32, sigma: float = 4.0) -> torch.Tensor:
    """2D Gaussian blob on a zero background — representative source island."""
    y = torch.arange(size, dtype=torch.float32)
    x = torch.arange(size, dtype=torch.float32)
    cy, cx = size / 2, size / 2
    blob   = torch.exp(-((y[:, None] - cy) ** 2 + (x[None, :] - cx) ** 2) / (2 * sigma ** 2))
    return blob


# ── PatchSolver ───────────────────────────────────────────────────────────────

def test_patch_solver_output_shape():
    """PatchSolver.decode_island returns a tensor with the same shape as the input."""
    fb     = _make_fb(k=8, f=5)
    solver = PatchSolver(fb, n_nonzero=3, stride=4)
    island = _gaussian_blob(size=20)
    recon  = solver.decode_island(island)
    assert recon.shape == island.shape


def test_patch_solver_output_dtype():
    """Reconstruction is float32."""
    fb     = _make_fb(k=8, f=5)
    solver = PatchSolver(fb, n_nonzero=3, stride=4)
    recon  = solver.decode_island(_gaussian_blob(20))
    assert recon.dtype == torch.float32


def test_patch_solver_non_zero_reconstruction():
    """Reconstruction of a non-trivial island is not all zeros."""
    fb     = _make_fb(k=8, f=5)
    solver = PatchSolver(fb, n_nonzero=3, stride=4)
    island = _gaussian_blob(20)
    recon  = solver.decode_island(island)
    assert recon.abs().sum().item() > 0.0


def test_patch_solver_smaller_residual():
    """Reconstruction reduces energy relative to zero baseline (||recon|| > 0)."""
    fb     = _make_fb(k=16, f=5)
    solver = PatchSolver(fb, n_nonzero=3, stride=3)
    island = _gaussian_blob(24)
    recon  = solver.decode_island(island)
    # OMP will at least activate some atoms; residual ||s - recon|| < ||s||
    residual_norm = (island - recon).norm()
    island_norm   = island.norm()
    assert residual_norm < island_norm


# ── ConvSolver ────────────────────────────────────────────────────────────────

def test_conv_solver_output_shape():
    """ConvSolver.decode_island returns the same spatial shape as the input."""
    fb     = _make_fb(k=8, f=5)
    solver = ConvSolver(fb, lmbda=0.1, n_iter=10)
    island = _gaussian_blob(24)
    recon  = solver.decode_island(island)
    assert recon.shape == island.shape


def test_conv_solver_output_dtype():
    """Reconstruction is float32."""
    fb     = _make_fb(k=8, f=5)
    solver = ConvSolver(fb, lmbda=0.1, n_iter=10)
    recon  = solver.decode_island(_gaussian_blob(24))
    assert recon.dtype == torch.float32


def test_conv_solver_residual_decreases():
    """FISTA reduces ||island - recon|| relative to the zero baseline."""
    fb     = _make_fb(k=16, f=5)
    solver = ConvSolver(fb, lmbda=0.05, n_iter=50)
    island = _gaussian_blob(30)
    recon  = solver.decode_island(island)
    residual_norm = (island - recon).norm()
    island_norm   = island.norm()
    assert residual_norm < island_norm


def test_conv_solver_island_smaller_than_atom():
    """ConvSolver handles islands smaller than the atom size (pads island)."""
    fb     = _make_fb(k=4, f=9)  # atom size 9
    solver = ConvSolver(fb, lmbda=0.1, n_iter=5)
    island = torch.ones(5, 5)    # smaller than atom
    recon  = solver.decode_island(island)
    assert recon.shape == island.shape  # cropped back to original size


def test_encode_island_shape():
    """encode_island returns Z with shape (K, H, W) where H, W >= island dims."""
    k, f   = 8, 5
    fb     = _make_fb(k=k, f=f)
    solver = ConvSolver(fb, lmbda=0.1, n_iter=10)
    H, W   = 20, 20
    island = _gaussian_blob(H)
    Z      = solver.encode_island(island)
    assert Z.shape[0] == k
    assert Z.shape[1] >= H
    assert Z.shape[2] >= W


def test_encode_island_sparsity():
    """With a reasonable lambda, Z should be sparse (many zero activations)."""
    fb     = _make_fb(k=16, f=5)
    solver = ConvSolver(fb, lmbda=0.1, n_iter=30)
    island = _gaussian_blob(20)
    Z      = solver.encode_island(island)
    sparsity = (Z.abs() < 1e-6).float().mean().item()
    # Expect > 50% zeros for a smooth blob with L1 regularisation
    assert sparsity > 0.5


def test_conv_solver_fista_convergence():
    """Running more FISTA iterations reduces the data fidelity term."""
    fb   = _make_fb(k=8, f=5)
    isl  = _gaussian_blob(20)

    solver_few  = ConvSolver(fb, lmbda=0.05, n_iter=5,   tol=0.0)
    solver_many = ConvSolver(fb, lmbda=0.05, n_iter=100, tol=0.0)

    recon_few  = solver_few.decode_island(isl)
    recon_many = solver_many.decode_island(isl)

    err_few  = (isl - recon_few).pow(2).sum()
    err_many = (isl - recon_many).pow(2).sum()

    assert err_many <= err_few * 1.05   # many iters should not be worse (5% slack)
