"""
tests/test_mgvi.py
=================
Unit tests for mad_clean.imaging.mgvi — vanilla MGVI, the RESOLVE-style
variational correctness reference (Fork B).  CPU, small grids, seconds.

The load-bearing test is closed-form: with a linear link the posterior over ξ is
exactly Gaussian N(ξ*, M⁻¹) with constant metric, and MGVI's natural-gradient
step is exact — so ξ̄ must equal the analytic ξ* = M⁻¹ b in one outer iteration.
Supporting tests: CG correctness, the C^{1/2} / J / Jᵀ adjoint structure, the
Fisher metric's symmetry and positive-definiteness, prior positivity, and that
the curvature sampler's covariance matches M⁻¹.
"""

from __future__ import annotations

import math

import torch

from mad_clean.imaging.forward import ImageDomainForward
from mad_clean.imaging.mgvi import (
    LogNormalFieldPrior,
    conjugate_gradient,
    draw_curvature_sample,
    fisher_matvec,
    mgvi_inference,
)

DT = torch.float64


def _gaussian_psf(size: int, fwhm: float = 2.5) -> torch.Tensor:
    sigma = fwhm / (2.0 * math.sqrt(2.0 * math.log(2.0)))
    c = size // 2
    ax = torch.arange(size, dtype=DT) - c
    yy, xx = torch.meshgrid(ax, ax, indexing="ij")
    psf = torch.exp(-(xx**2 + yy**2) / (2.0 * sigma**2))
    return psf / psf.max()


def _densify(matvec, shape) -> torch.Tensor:
    """Materialise the dense matrix of a linear operator on (H, W) images."""
    H, W = shape
    n = H * W
    M = torch.zeros(n, n, dtype=DT)
    for i in range(n):
        e = torch.zeros(n, dtype=DT)
        e[i] = 1.0
        M[:, i] = matvec(e.reshape(H, W)).reshape(-1)
    return M


# ── conjugate gradient ──────────────────────────────────────────────────────

def test_cg_solves_spd_system():
    torch.manual_seed(0)
    n = 30
    A = torch.randn(n, n, dtype=DT)
    A = A @ A.t() + n * torch.eye(n, dtype=DT)  # SPD, well-conditioned
    x_true = torch.randn(n, dtype=DT)
    b = A @ x_true
    x = conjugate_gradient(lambda v: A @ v, b, max_iter=n, tol=1e-12)
    assert float((x - x_true).norm()) < 1e-6


# ── prior operator structure ─────────────────────────────────────────────────

def test_c_half_self_adjoint():
    prior = LogNormalFieldPrior((8, 8), alpha=2.7, dtype=DT)
    torch.manual_seed(1)
    a = torch.randn(8, 8, dtype=DT)
    b = torch.randn(8, 8, dtype=DT)
    lhs = float((prior.apply_C_half(a) * b).sum())
    rhs = float((a * prior.apply_C_half(b)).sum())
    assert abs(lhs - rhs) < 1e-10


def test_jacobian_adjoint_pair():
    """⟨J v, y⟩ == ⟨v, Jᵀ y⟩ at a random linearization point (exp link)."""
    prior = LogNormalFieldPrior((8, 8), alpha=2.5, sigma_log=0.5, link="exp", dtype=DT)
    torch.manual_seed(2)
    xi_bar = 0.3 * torch.randn(8, 8, dtype=DT)
    v = torch.randn(8, 8, dtype=DT)
    y = torch.randn(8, 8, dtype=DT)
    lhs = float((prior.apply_J(xi_bar, v) * y).sum())
    rhs = float((v * prior.apply_Jt(xi_bar, y)).sum())
    assert abs(lhs - rhs) < 1e-10


def test_prior_positive_exp_link():
    prior = LogNormalFieldPrior((16, 16), sigma_log=1.0, mean_log=-2.0, link="exp", dtype=DT)
    torch.manual_seed(3)
    s = prior.forward(torch.randn(16, 16, dtype=DT))
    assert (s > 0).all()


# ── Fisher metric ─────────────────────────────────────────────────────────────

def test_fisher_metric_symmetric_and_pd():
    prior = LogNormalFieldPrior((6, 6), alpha=2.5, sigma_log=0.5, link="exp", dtype=DT)
    op = ImageDomainForward(_gaussian_psf(6))
    torch.manual_seed(4)
    xi_bar = 0.2 * torch.randn(6, 6, dtype=DT)
    matvec = fisher_matvec(prior, op, xi_bar, noise_std=0.3)
    M = _densify(matvec, (6, 6))
    # Symmetric.
    assert float((M - M.t()).abs().max()) < 1e-9
    # Positive-definite (M = I + PSD ⇒ eigenvalues ≥ 1).
    eig = torch.linalg.eigvalsh(M)
    assert float(eig.min()) > 0.99


# ── closed-form correctness reference (linear link) ──────────────────────────

def test_linear_gaussian_recovers_analytic_mean():
    """Linear link ⇒ posterior over ξ is exactly N(ξ*, M⁻¹); MGVI's Newton step
    is exact, so one outer iteration must land on ξ* = M⁻¹ b."""
    size = 8
    prior = LogNormalFieldPrior((size, size), alpha=2.5, sigma_log=1.0,
                                mean_log=0.0, link="identity", dtype=DT)
    op = ImageDomainForward(_gaussian_psf(size))
    noise_std = 0.2

    torch.manual_seed(5)
    xi_true = torch.randn(size, size, dtype=DT)
    s_true = prior.forward(xi_true)
    d = op.make_dirty(s_true, noise_std=noise_std,
                      generator=torch.Generator().manual_seed(6))

    # Analytic: M constant (J = C^{1/2}), b = C^{1/2} Aᵀ d / σ², ξ* = M⁻¹ b.
    matvec = fisher_matvec(prior, op, torch.zeros(size, size, dtype=DT), noise_std)
    M = _densify(matvec, (size, size))
    b = prior.apply_C_half(op.adjoint(d) / noise_std**2).reshape(-1)
    xi_star = torch.linalg.solve(M, b).reshape(size, size)

    xi_bar, _, sky = mgvi_inference(
        prior, op, d, noise_std, n_outer=1, n_samples=2, cg_iters=size * size,
        generator=torch.Generator().manual_seed(7),
    )
    rel = float((xi_bar - xi_star).norm() / xi_star.norm())
    assert rel < 1e-6, f"MGVI mean off analytic ξ* by {rel:.2e}"


def test_curvature_sample_covariance_matches_Minv():
    """Empirical covariance of the curvature samples ≈ M⁻¹ (statistical check)."""
    size = 6
    prior = LogNormalFieldPrior((size, size), alpha=2.0, sigma_log=0.7,
                                link="identity", dtype=DT)
    op = ImageDomainForward(_gaussian_psf(size))
    noise_std = 0.4
    xi_bar = torch.zeros(size, size, dtype=DT)

    matvec = fisher_matvec(prior, op, xi_bar, noise_std)
    M = _densify(matvec, (size, size))
    M_inv = torch.linalg.inv(M)

    gen = torch.Generator().manual_seed(11)
    n = 6000
    cols = []
    for _ in range(n):
        cols.append(
            draw_curvature_sample(prior, op, xi_bar, noise_std,
                                  generator=gen, cg_iters=size * size).reshape(-1)
        )
    X = torch.stack(cols, dim=1)  # (d, n)
    emp_cov = (X @ X.t()) / n

    # Trace ratio is a robust, low-variance summary of overall scale.
    ratio = float(emp_cov.trace() / M_inv.trace())
    assert abs(ratio - 1.0) < 0.12, f"sample-cov trace ratio {ratio:.3f}"
