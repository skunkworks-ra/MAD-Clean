"""
tests/test_score.py
===================
Unit tests for mad_clean.imaging.score — the EDM-preconditioned denoiser /
prior-score model for the field-posterior loop.  CPU, tiny nets, seconds.

Load-bearing properties:
  * EDM preconditioning has the correct σ→0 / σ→∞ limits,
  * at initialisation (F_θ ≡ 0) the denoiser is the exact Bayes denoiser for a
    unit-variance Gaussian prior, giving a closed-form score check,
  * the DSM loss is finite and a single field can be overfit (loss drops),
  * shapes and the Tweedie score relation.
"""

from __future__ import annotations

import torch

from mad_clean.imaging.score import EDMDenoiser, UNet, edm_loss


def _tiny_model(sigma_data: float = 1.0) -> EDMDenoiser:
    net = UNet(in_ch=1, base=8, mults=(1, 2), emb_dim=16)
    return EDMDenoiser(net, sigma_data=sigma_data)


def test_unet_output_shape():
    torch.manual_seed(0)
    model = _tiny_model()
    x = torch.randn(2, 1, 32, 32)
    sigma = torch.tensor([0.5, 2.0])
    d = model(x, sigma)
    assert d.shape == x.shape


def test_precond_limits():
    """c_skip → 1, c_out → 0 as σ→0; c_skip → 0 as σ→∞."""
    model = _tiny_model(sigma_data=1.0)
    s_small = torch.tensor(1e-4)
    s_large = torch.tensor(1e4)
    cs0, co0, _, _ = model._coeffs(s_small)
    csL, _, _, _ = model._coeffs(s_large)
    assert abs(float(cs0) - 1.0) < 1e-6
    assert float(co0) < 1e-3
    assert float(csL) < 1e-6


def test_init_is_gaussian_bayes_denoiser():
    """With F_θ ≡ 0 (zero-init output conv), D = c_skip·x exactly, so the score
    equals the analytic score of a unit-variance Gaussian prior:
        s_θ(x,σ) = (D - x)/σ² = (c_skip - 1) x / σ² = -x / (σ² + σ_data²)."""
    model = _tiny_model(sigma_data=1.0)
    model.eval()
    x = torch.randn(4, 1, 16, 16)
    for sigma_val in (0.1, 1.0, 5.0):
        sigma = torch.tensor(sigma_val)
        score = model.score(x, sigma)
        analytic = -x / (sigma_val**2 + 1.0)
        torch.testing.assert_close(score, analytic, rtol=1e-4, atol=1e-5)


def test_score_matches_tweedie():
    """score(x,σ) == (forward(x,σ) - x)/σ² by definition."""
    torch.manual_seed(1)
    model = _tiny_model()
    x = torch.randn(3, 1, 16, 16)
    sigma = torch.tensor([0.3, 1.0, 3.0])
    s = sigma.view(-1, 1, 1, 1)
    expected = (model(x, sigma) - x) / s**2
    torch.testing.assert_close(model.score(x, sigma), expected)


def test_loss_finite_and_deterministic():
    model = _tiny_model()
    f0 = torch.randn(4, 1, 32, 32)
    g1 = torch.Generator().manual_seed(5)
    g2 = torch.Generator().manual_seed(5)
    l1 = edm_loss(model, f0, generator=g1)
    l2 = edm_loss(model, f0, generator=g2)
    assert torch.isfinite(l1)
    torch.testing.assert_close(l1, l2)


def test_overfit_single_field():
    """With a fixed σ / noise draw each step (a deterministic DSM objective), the
    net should drive the loss down sharply on one field — confirming the
    learning path (loss → grads → backbone → preconditioned output) is wired."""
    torch.manual_seed(0)
    model = _tiny_model()
    # A smoothly structured field is denoisable (unlike pure white noise).
    yy, xx = torch.meshgrid(torch.linspace(-2, 2, 32), torch.linspace(-2, 2, 32), indexing="ij")
    f0 = torch.exp(-(xx**2 + yy**2))[None, None]
    opt = torch.optim.Adam(model.parameters(), lr=2e-3)
    losses = []
    for _ in range(120):
        opt.zero_grad()
        # Reseed each step → identical (σ, noise) → fixed objective in the weights.
        loss = edm_loss(model, f0, generator=torch.Generator().manual_seed(0))
        loss.backward()
        opt.step()
        losses.append(float(loss))
    early = sum(losses[:5]) / 5
    late = sum(losses[-5:]) / 5
    assert late < 0.3 * early, f"loss did not drop: early={early:.4f} late={late:.4f}"
