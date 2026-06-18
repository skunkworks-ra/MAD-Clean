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


def test_context_discriminates_at_physical_scale():
    """The context encoder must produce distinct vectors for distinct scenes
    at *physical* input amplitudes (residual ~1e-4 Jy, PSF peak ~1).

    Regression test for the 2026-06-11 conditioning failure: with raw
    inputs the GroupNorm statistics are dominated by the O(1) PSF channel
    and the scene signal is a 1e-4 perturbation, so contexts collapse to
    near-identical vectors and the flow trains unconditionally.
    """
    torch.manual_seed(0)
    flow = _tiny_flow()
    B = 4
    # Distinct scenes at physical scale: sparse positive sources ~1e-4
    residual = torch.zeros(B, 128, 128)
    for b in range(B):
        residual[b, 30 + 20 * b: 40 + 20 * b, 40: 60] = 1e-4
    psf = torch.zeros(B, 128, 128)
    psf[:, 64, 64] = 1.0
    image = torch.stack([residual, psf], dim=1)
    cond = torch.zeros(B, 5); cond[:, 0] = 1e-4; cond[:, 4] = 1.0
    ctx = flow._context(image, cond)
    # Pairwise distances between contexts must be a meaningful fraction of
    # the context magnitude, not numerical dust.
    d = torch.cdist(ctx, ctx)
    off = d[~torch.eye(B, dtype=torch.bool)]
    rel = float(off.min() / (ctx.norm(dim=1).mean() + 1e-12))
    # The 2026-06-11 collapse produced rel ~1e-6 (numerical dust). A healthy
    # encoder at random init lands ~1e-2..1e-1 depending on seed and layer
    # widths, so the bar separates collapse from health without being
    # init-sensitive.
    assert rel > 0.01, (
        f"contexts nearly identical across distinct scenes (rel min dist "
        f"{rel:.2e}); encoder cannot discriminate at physical input scale"
    )


def test_weighted_log_prob_matches_unweighted_at_ones():
    flow = _tiny_flow()
    theta, image, cond = _inputs()
    lp = flow.log_prob(theta, image, cond)
    lp_w = flow.log_prob(theta, image, cond,
                         dim_weights=torch.ones(_B, _THETA_DIM))
    assert torch.allclose(lp, lp_w, atol=1e-5)


def test_weighted_log_prob_downweights_dims():
    """Zero weight on every dim must give zero log-prob; intermediate
    weights must change the value."""
    flow = _tiny_flow()
    theta, image, cond = _inputs()
    lp0 = flow.log_prob(theta, image, cond,
                        dim_weights=torch.zeros(_B, _THETA_DIM))
    assert torch.allclose(lp0, torch.zeros(_B), atol=1e-6)
    w = torch.full((_B, _THETA_DIM), 0.5)
    lp_half = flow.log_prob(theta, image, cond, dim_weights=w)
    lp_full = flow.log_prob(theta, image, cond)
    assert torch.allclose(lp_half, 0.5 * lp_full, atol=1e-5)


def test_context_carries_absolute_scale():
    """Two residuals identical up to a flux scale must produce different
    contexts: the per-sample normalisation divides the scale out of the
    image, so it has to re-enter through cond (log10 scale append)."""
    flow = _tiny_flow()
    torch.manual_seed(3)
    res = torch.randn(1, 1, 128, 128) * 1e-4
    psf = torch.randn(1, 1, 128, 128)
    cond = torch.zeros(1, 5)
    img_a = torch.cat([res, psf], dim=1)
    img_b = torch.cat([res * 100.0, psf], dim=1)
    ctx_a = flow._context(img_a, cond)
    ctx_b = flow._context(img_b, cond)
    rel = (ctx_a - ctx_b).norm() / ctx_a.norm().clamp_min(1e-12)
    assert float(rel) > 1e-3, (
        f"contexts identical across a 100x flux rescale (rel {rel:.2e}); "
        "absolute scale is not reaching the network"
    )
