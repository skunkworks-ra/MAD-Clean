"""Conditional normalising flow over starlet coefficients (wavelet NPE).

Replaces the MDN head of mdn_asp.py: instead of K Gaussian components over
a 6D source parameterisation, the posterior target is theta = decimated
starlet coefficients of the true sky cutout (see mad_clean.wavelet.starlet).

Architecture
------------
- Context encoder: same CNN + FiLM pattern as MDNAsp — 2-channel
  (residual, PSF) 128x128 input, (sigma_local, config_one_hot) FiLM
  conditioning — producing a context vector.
- Flow: stack of affine coupling layers (RealNVP style) with fixed
  alternating binary masks and the context vector concatenated into every
  coupling MLP.  Affine coupling is the simplest adequate choice; escalate
  to splines only if coverage tests show miscalibration.

Loss is exact NLL: -log q(theta | context).  Sampling is exact and cheap
(one MLP pass per coupling layer).
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from mad_clean.models.mdn_asp import COND_DIM, FiLMBlock

__all__ = ["CoeffFlow"]

# Clamp on the coupling log-scale: keeps the Jacobian bounded and training
# stable on heavy-tailed coefficient targets.
_LOG_SCALE_CLAMP = 4.0


class _ContextEncoder(nn.Module):
    """CNN + FiLM context encoder; mirrors MDNAsp.enc/proj/film."""

    def __init__(self, base_channels: int, context_dim: int, cond_dim: int):
        super().__init__()
        c = base_channels

        def block(ic: int, oc: int) -> nn.Sequential:
            return nn.Sequential(
                nn.Conv2d(ic, oc, 3, stride=2, padding=1),
                nn.GroupNorm(1, oc),
                nn.GELU(),
                nn.Conv2d(oc, oc, 3, stride=1, padding=1),
                nn.GroupNorm(1, oc),
                nn.GELU(),
            )

        # 2-channel input: (residual, PSF); 128 -> 64 -> 32 -> 16 -> 8 -> 4
        self.enc = nn.Sequential(
            block(2, c),
            block(c, 2 * c),
            block(2 * c, 4 * c),
            block(4 * c, 4 * c),
            block(4 * c, 4 * c),
        )
        feat_dim = 4 * c * 4 * 4
        self.proj = nn.Linear(feat_dim, context_dim)
        self.film1 = FiLMBlock(context_dim, cond_dim)
        self.film2 = FiLMBlock(context_dim, cond_dim)

    def forward(self, image: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        h = self.enc(image).flatten(1)
        h = F.gelu(self.proj(h))
        h = self.film1(h, cond)
        h = self.film2(h, cond)
        return h


class _Coupling(nn.Module):
    """One affine coupling layer with a fixed binary mask."""

    def __init__(self, dim: int, context_dim: int, hidden: int, mask: torch.Tensor):
        super().__init__()
        self.register_buffer("mask", mask.float())
        self.net = nn.Sequential(
            nn.Linear(dim + context_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, 2 * dim),
        )
        # Identity initialisation: zero the last layer so the flow starts
        # as the base distribution.
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def _st(self, x_masked: torch.Tensor, ctx: torch.Tensor):
        h = self.net(torch.cat([x_masked, ctx], dim=-1))
        s, t = h.chunk(2, dim=-1)
        s = _LOG_SCALE_CLAMP * torch.tanh(s / _LOG_SCALE_CLAMP)
        return s, t

    def forward(self, x: torch.Tensor, ctx: torch.Tensor):
        """x -> z. Returns (z, log_det)."""
        xm = x * self.mask
        s, t = self._st(xm, ctx)
        free = 1.0 - self.mask
        z = xm + free * (x * torch.exp(s) + t)
        log_det = (free * s).sum(dim=-1)
        return z, log_det

    def inverse(self, z: torch.Tensor, ctx: torch.Tensor):
        zm = z * self.mask
        s, t = self._st(zm, ctx)
        free = 1.0 - self.mask
        x = zm + free * ((z - t) * torch.exp(-s))
        return x


def _alternating_masks(dim: int, n_layers: int) -> list[torch.Tensor]:
    """Checkerboard-in-index masks, flipped layer to layer, with a random
    permutation of indices every two layers so all dims mix."""
    g = torch.Generator().manual_seed(0)
    masks = []
    perm = torch.arange(dim)
    base = (torch.arange(dim) % 2).float()
    for i in range(n_layers):
        if i % 2 == 0 and i > 0:
            perm = torch.randperm(dim, generator=g)
        m = torch.zeros(dim)
        m[perm] = base if i % 2 == 0 else 1.0 - base
        masks.append(m)
    return masks


class CoeffFlow(nn.Module):
    """Conditional affine-coupling flow over the theta vector.

    Parameters
    ----------
    theta_dim : dimension of the starlet coefficient vector (StarletCodec.theta_dim).
    base_channels : CNN encoder width (32 default, 8 for CPU tests).
    context_dim : context vector width.
    hidden : coupling MLP hidden width.
    n_layers : number of coupling layers.
    cond_dim : FiLM conditioning length (5: sigma_local + config one-hot).
    """

    def __init__(
        self,
        theta_dim: int,
        base_channels: int = 32,
        context_dim: int = 256,
        hidden: int = 512,
        n_layers: int = 8,
        cond_dim: int = COND_DIM,
    ):
        super().__init__()
        self.theta_dim = theta_dim
        self.encoder = _ContextEncoder(base_channels, context_dim, cond_dim)
        masks = _alternating_masks(theta_dim, n_layers)
        self.layers = nn.ModuleList(
            _Coupling(theta_dim, context_dim, hidden, m) for m in masks
        )

    def _context(self, image: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        return self.encoder(image, cond)

    def log_prob(
        self,
        theta: torch.Tensor,   # (B, theta_dim)
        image: torch.Tensor,   # (B, 2, 128, 128)
        cond:  torch.Tensor,   # (B, cond_dim)
    ) -> torch.Tensor:
        """Exact log q(theta | image, cond). Returns (B,)."""
        ctx = self._context(image, cond)
        z = theta
        log_det = torch.zeros(theta.shape[0], device=theta.device)
        for layer in self.layers:
            z, ld = layer(z, ctx)
            log_det = log_det + ld
        log_base = -0.5 * (z ** 2 + math.log(2.0 * math.pi)).sum(dim=-1)
        return log_base + log_det

    def nll_loss(self, theta, image, cond) -> torch.Tensor:
        return -self.log_prob(theta, image, cond).mean()

    @torch.no_grad()
    def sample(
        self,
        image: torch.Tensor,
        cond:  torch.Tensor,
        n: int = 1,
    ) -> torch.Tensor:
        """Draw posterior samples. Returns (B, n, theta_dim)."""
        B = image.shape[0]
        ctx = self._context(image, cond)
        ctx_rep = ctx.repeat_interleave(n, dim=0)
        z = torch.randn(B * n, self.theta_dim, device=image.device,
                        dtype=ctx.dtype)
        for layer in reversed(self.layers):
            z = layer.inverse(z, ctx_rep)
        return z.view(B, n, self.theta_dim)
