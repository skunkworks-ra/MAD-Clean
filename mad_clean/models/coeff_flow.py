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
  alternating binary masks.  The context FiLM-modulates every coupling
  MLP's hidden layers.  The first version concatenated the context into
  the MLP input instead; trained to step 3000 it ignored the context
  completely (cross-assignment diagonal nll == off-diagonal to 4
  decimals, eval_wavelet_npe.py 2026-06-11) — a 256-dim concat against
  5472 theta dims is too easy to ignore.  FiLM forces every hidden unit
  through a context-dependent affine map, the same mechanism that
  demonstrably conditions the MDN.  Affine coupling is the simplest
  adequate choice; escalate to splines only if coverage tests show
  miscalibration.

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
        # +1: the per-sample residual normalisation scale (log10) is
        # appended to cond.  Without it absolute flux is unrecoverable —
        # theta is in physical units but the image channel is divided by
        # its own std, and sigma_local alone does not determine that std.
        self.film1 = FiLMBlock(context_dim, cond_dim + 1)
        self.film2 = FiLMBlock(context_dim, cond_dim + 1)

    def forward(self, image: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        # Per-sample normalisation of the residual channel.  Physical
        # residuals are ~1e-4 Jy while the PSF channel peaks at 1; without
        # this the GroupNorm statistics are PSF-dominated and the contexts
        # collapse to near-identical vectors (2026-06-11 conditioning
        # failure, pinned by test_context_discriminates_at_physical_scale).
        # The scale itself is appended to cond as log10 so the network
        # can map the normalised image back to absolute flux (sigma_local
        # alone is the noise level, not this factor).
        res, psf = image[:, 0:1], image[:, 1:2]
        scale = res.flatten(1).std(dim=1).clamp_min(1e-12).view(-1, 1, 1, 1)
        image = torch.cat([res / scale, psf], dim=1)
        cond = torch.cat(
            [cond, torch.log10(scale.flatten(1))], dim=-1)
        h = self.enc(image).flatten(1)
        h = F.gelu(self.proj(h))
        h = self.film1(h, cond)
        h = self.film2(h, cond)
        return h


class _Coupling(nn.Module):
    """One affine coupling layer with a fixed binary mask.

    The context FiLM-modulates both hidden layers — see module docstring
    for why concat conditioning is not used.
    """

    def __init__(self, dim: int, context_dim: int, hidden: int, mask: torch.Tensor):
        super().__init__()
        self.register_buffer("mask", mask.float())
        self.fc_in = nn.Linear(dim, hidden)
        self.film1 = FiLMBlock(hidden, context_dim)
        self.film2 = FiLMBlock(hidden, context_dim)
        self.fc_out = nn.Linear(hidden, 2 * dim)
        # Identity initialisation: zero the last layer so the flow starts
        # as the base distribution.
        nn.init.zeros_(self.fc_out.weight)
        nn.init.zeros_(self.fc_out.bias)

    def _st(self, x_masked: torch.Tensor, ctx: torch.Tensor):
        h = F.gelu(self.fc_in(x_masked))
        h = self.film1(h, ctx)
        h = self.film2(h, ctx)
        h = self.fc_out(h)
        s, t = h.chunk(2, dim=-1)
        s = _LOG_SCALE_CLAMP * torch.tanh(s / _LOG_SCALE_CLAMP)
        return s, t

    def forward(self, x: torch.Tensor, ctx: torch.Tensor):
        """x -> z. Returns (z, log_det_dims) with per-dim log-det (B, D).

        Affine couplings have a diagonal Jacobian on the free dims, so the
        log-det decomposes per dimension — which is what makes the
        support-weighted NLL exact dimension-wise."""
        xm = x * self.mask
        s, t = self._st(xm, ctx)
        free = 1.0 - self.mask
        z = xm + free * (x * torch.exp(s) + t)
        return z, free * s

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
        hidden: int = 128,
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

    def _log_prob_given_ctx(self, theta, ctx):
        """Per-dim log-density for theta under a precomputed context.

        theta (M, D), ctx (M, C) -> (M, D).  Factored out so the conv
        encoder can be run once and its context reused across many theta
        (cross-assignment matrix, in-support split)."""
        z = theta
        log_det = torch.zeros_like(theta)
        for layer in self.layers:
            z, ld = layer(z, ctx)
            log_det = log_det + ld
        return -0.5 * (z ** 2 + math.log(2.0 * math.pi)) + log_det

    def log_prob_per_dim(self, theta, image, cond) -> torch.Tensor:
        """Unweighted per-dim log-density, (B, D).  Lets the caller split
        the NLL by support (source vs background coefficients)."""
        return self._log_prob_given_ctx(theta, self._context(image, cond))

    def log_prob(
        self,
        theta: torch.Tensor,   # (B, theta_dim)
        image: torch.Tensor,   # (B, 2, 128, 128)
        cond:  torch.Tensor,   # (B, cond_dim)
        dim_weights: torch.Tensor | None = None,  # (B, theta_dim)
    ) -> torch.Tensor:
        """log q(theta | image, cond). Returns (B,).

        With ``dim_weights`` the per-dim base term and per-dim coupling
        log-det are weighted before summing (a tempered likelihood for
        training only — see StarletCodec.support_weights).  Without it
        this is the exact NLL; evaluation must always use the unweighted
        form."""
        per_dim = self.log_prob_per_dim(theta, image, cond)
        if dim_weights is not None:
            per_dim = per_dim * dim_weights
        return per_dim.sum(dim=-1)

    def log_prob_matrix(self, theta, image, cond) -> torch.Tensor:
        """Cross-assignment matrix L[i, j] = log q(theta_j | image_i).

        Context is computed once per image (B conv passes) and reused
        across all theta, so cost scales as B conv + B**2 coupling MLP
        rather than B**2 conv.  Used for the in-batch InfoNCE term."""
        B = theta.shape[0]
        ctx = self._context(image, cond)                  # (B, C)
        ctx_rep = ctx.repeat_interleave(B, dim=0)          # row i, B times
        theta_rep = theta.repeat(B, 1)                     # theta tiled per row
        per_dim = self._log_prob_given_ctx(theta_rep, ctx_rep)
        return per_dim.sum(dim=-1).view(B, B)

    def nll_loss(self, theta, image, cond, dim_weights=None) -> torch.Tensor:
        return -self.log_prob(theta, image, cond, dim_weights).mean()

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

    def sample_with_grad(
        self,
        image: torch.Tensor,
        cond:  torch.Tensor,
        n: int = 1,
    ) -> torch.Tensor:
        """Reparameterized sample with gradients. Returns (B, n, theta_dim).

        Used for residual-loss training: gradients flow back through the
        inverse coupling layers into the flow parameters.
        """
        B = image.shape[0]
        ctx = self._context(image, cond)
        ctx_rep = ctx.repeat_interleave(n, dim=0)
        z = torch.randn(B * n, self.theta_dim, device=image.device,
                        dtype=ctx.dtype)
        for layer in reversed(self.layers):
            z = layer.inverse(z, ctx_rep)
        return z.view(B, n, self.theta_dim)
