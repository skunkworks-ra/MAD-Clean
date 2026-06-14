"""
tests/test_langevin.py
=====================
Unit tests for mad_clean.imaging.langevin — the annealed ULA field sampler.
CPU, seconds, no trained network.

The load-bearing test is closed-form: the generic ULA engine, run on an
analytic Gaussian target, must recover that Gaussian's mean and variance (an
SBC-lite calibration check).  The remaining tests check the log-sky wrapper:
strict positivity of samples, shapes, determinism, and that an uninformative
likelihood leaves the sampler at the (Gaussian) prior.
"""

from __future__ import annotations

import math

import torch

from mad_clean.imaging.forward import ImageDomainForward
from mad_clean.imaging.langevin import (
    annealed_langevin,
    geometric_sigma_schedule,
    sample_field_posterior,
)
from mad_clean.imaging.score import EDMDenoiser, UNet


def _gaussian_psf(size: int, fwhm: float = 3.0) -> torch.Tensor:
    sigma = fwhm / (2.0 * math.sqrt(2.0 * math.log(2.0)))
    c = size // 2
    ax = torch.arange(size, dtype=torch.float64) - c
    yy, xx = torch.meshgrid(ax, ax, indexing="ij")
    psf = torch.exp(-(xx**2 + yy**2) / (2.0 * sigma**2))
    return psf / psf.max()


# ── closed-form engine calibration ──────────────────────────────────────────

def test_engine_recovers_gaussian_moments():
    """ULA on a fixed 1-D Gaussian target N(m, v) recovers its moments.

    A single noise level (T=1) makes η_t = η_0; the score is sigma-independent.
    Many parallel chains let us read the stationary mean/variance.  ULA has an
    O(η) bias, so η is small and the tolerance generous."""
    torch.manual_seed(0)
    m, v = 2.0, 0.5
    n_chains = 20000

    def posterior_score(x, sigma):
        return -(x - m) / v  # ∇ log N(x; m, v)

    init = torch.zeros(n_chains, 1)
    sigmas = torch.tensor([1.0])  # single level
    gen = torch.Generator().manual_seed(0)
    x = annealed_langevin(
        posterior_score, init, sigmas,
        n_steps_per_level=2000, step_size=2e-3, generator=gen,
    )
    # Discard the first samples implicitly via long burn-in already done.
    mean = float(x.mean())
    var = float(x.var())
    assert abs(mean - m) < 0.05, f"mean {mean:.3f} != {m}"
    assert abs(var - v) < 0.08, f"var {var:.3f} != {v}"


def test_schedule_descending():
    s = geometric_sigma_schedule(10.0, 0.01, 15)
    assert s.shape == (15,)
    assert float(s[0]) > float(s[-1])
    assert torch.all(s[:-1] >= s[1:])  # monotone non-increasing


# ── log-sky wrapper ─────────────────────────────────────────────────────────

def _toy_setup(size=16, sigma_data=1.0):
    """Forward op + a zero-init EDM denoiser whose score is the exact Gaussian
    prior score in standardised log-sky space."""
    op = ImageDomainForward(_gaussian_psf(size).to(torch.float32))
    model = EDMDenoiser(UNet(base=8, mults=(1, 2), emb_dim=16), sigma_data=sigma_data)
    model.eval()
    return op, model


def test_samples_strictly_positive():
    op, model = _toy_setup()
    d = torch.zeros(16, 16)
    gen = torch.Generator().manual_seed(0)
    # Physical-ish standardisation (mu<0, tau<1) keeps s = exp(mu+tau·f') bounded.
    s = sample_field_posterior(
        op, model, d, mu=-3.0, tau=0.5, noise_std=0.1,
        n_samples=4, sigma_max=2.0, sigma_min=0.05,
        n_levels=8, n_steps_per_level=5, step_size=1e-5, generator=gen,
    )
    assert s.shape == (4, 16, 16)
    assert (s > 0).all(), "sky samples must be strictly positive (s = exp f)"
    assert torch.isfinite(s).all()


def test_deterministic_with_seed():
    op, model = _toy_setup()
    d = torch.randn(16, 16) * 0.1
    kw = dict(mu=-3.0, tau=0.5, noise_std=0.1, n_samples=2,
              sigma_max=2.0, sigma_min=0.05, n_levels=6, n_steps_per_level=4,
              step_size=1e-5)
    s1 = sample_field_posterior(op, model, d, generator=torch.Generator().manual_seed(7), **kw)
    s2 = sample_field_posterior(op, model, d, generator=torch.Generator().manual_seed(7), **kw)
    torch.testing.assert_close(s1, s2)


def test_uninformative_likelihood_returns_to_prior():
    """With a huge noise_std the data term vanishes and the sampler explores the
    Gaussian prior (zero-init denoiser ⇒ exact N(0,1) prior score): standardised
    log-sky f' = log(s) should be ~zero mean (symmetry, robust to mixing)."""
    op, model = _toy_setup()
    d = torch.zeros(16, 16)
    gen = torch.Generator().manual_seed(3)
    s = sample_field_posterior(
        op, model, d, mu=0.0, tau=1.0, noise_std=1e6,
        n_samples=64, sigma_max=2.0, sigma_min=0.05,
        n_levels=12, n_steps_per_level=20, step_size=1e-4, generator=gen,
    )
    f = torch.log(s)
    assert torch.isfinite(f).all()
    assert abs(float(f.mean())) < 0.5, f"log-sky mean {float(f.mean()):.3f} not ~0"
