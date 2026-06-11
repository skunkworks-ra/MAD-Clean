"""Tests for the conditional coefficient flow (wavelet NPE head)."""
from __future__ import annotations

import torch

from mad_clean.models.coeff_flow import CoeffFlow

# Small everything for CPU tests
_THETA_DIM = 24
_B = 3


def _tiny_flow() -> CoeffFlow:
    torch.manual_seed(0)
    return CoeffFlow(
        theta_dim=_THETA_DIM,
        base_channels=4,
        context_dim=32,
        hidden=48,
        n_layers=4,
    )


def _inputs():
    torch.manual_seed(1)
    image = torch.randn(_B, 2, 128, 128)
    cond = torch.randn(_B, 5)
    theta = torch.randn(_B, _THETA_DIM)
    return theta, image, cond


def test_log_prob_shape_and_finite():
    flow = _tiny_flow()
    theta, image, cond = _inputs()
    lp = flow.log_prob(theta, image, cond)
    assert lp.shape == (_B,)
    assert torch.isfinite(lp).all()


def test_identity_init_log_prob_is_standard_normal():
    """Couplings are zero-initialised, so at init log q == N(0, I) exactly."""
    flow = _tiny_flow()
    theta, image, cond = _inputs()
    lp = flow.log_prob(theta, image, cond)
    expected = torch.distributions.Normal(0.0, 1.0).log_prob(theta).sum(-1)
    assert torch.allclose(lp, expected, atol=1e-4)


def test_forward_inverse_consistency():
    flow = _tiny_flow()
    # Perturb weights so the flow is not the identity
    with torch.no_grad():
        for p in flow.parameters():
            p.add_(0.01 * torch.randn_like(p))
    theta, image, cond = _inputs()
    ctx = flow._context(image, cond)
    z = theta
    for layer in flow.layers:
        z, _ = layer(z, ctx)
    x = z
    for layer in reversed(flow.layers):
        x = layer.inverse(x, ctx)
    assert torch.allclose(x, theta, atol=1e-4)


def test_sample_shape():
    flow = _tiny_flow()
    _, image, cond = _inputs()
    s = flow.sample(image, cond, n=7)
    assert s.shape == (_B, 7, _THETA_DIM)
    assert torch.isfinite(s).all()


def test_nll_decreases_on_overfit():
    """A few gradient steps on a frozen batch must reduce the NLL."""
    flow = _tiny_flow()
    theta, image, cond = _inputs()
    opt = torch.optim.Adam(flow.parameters(), lr=1e-3)
    flow.train()
    first = None
    for _ in range(30):
        opt.zero_grad()
        loss = flow.nll_loss(theta, image, cond)
        loss.backward()
        opt.step()
        if first is None:
            first = loss.item()
    assert loss.item() < first


def test_context_dependence():
    """log_prob must actually depend on the image (posterior, not prior)."""
    flow = _tiny_flow()
    with torch.no_grad():
        for p in flow.parameters():
            p.add_(0.05 * torch.randn_like(p))
    theta, image, cond = _inputs()
    lp1 = flow.log_prob(theta, image, cond)
    lp2 = flow.log_prob(theta, torch.randn_like(image), cond)
    assert not torch.allclose(lp1, lp2, atol=1e-6)
