"""Convolutional pixel-space flow for deconvolution.

Replaces the MLP affine coupling layers in CoeffFlow with CNN coupling
layers using 2D checkerboard masks.  The CNN sees local pixel
neighborhoods so a blob at any position shares learned filters with
blobs at other positions -- the spatial structure problem that flat
alternating-index masks cannot solve.

Interface is identical to CoeffFlow (nll_loss, log_prob, log_prob_per_dim,
sample, sample_with_grad) so the training script needs only a flag switch.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from mad_clean.models.coeff_flow import _ContextEncoder, _LOG_SCALE_CLAMP
from mad_clean.models.mdn_asp import COND_DIM

__all__ = ["ConvPixelFlow"]


class _SpatialFiLM(nn.Module):
    """FiLM conditioning for 2-D feature maps.

    Context vector (B, ctx_dim) -> per-channel affine on (B, C, H, W).
    """

    def __init__(self, channels: int, context_dim: int):
        super().__init__()
        self.proj = nn.Linear(context_dim, 2 * channels)

    def forward(self, x: torch.Tensor, ctx: torch.Tensor) -> torch.Tensor:
        gb = self.proj(ctx)                          # (B, 2C)
        gamma, beta = gb.chunk(2, dim=-1)
        gamma = gamma.view(-1, x.shape[1], 1, 1)
        beta  = beta.view(-1, x.shape[1], 1, 1)
        return x * (1.0 + gamma) + beta


class _ConvCoupling(nn.Module):
    """One affine coupling layer with a 2-D checkerboard mask.

    The coupling network is a 3-layer CNN with spatial FiLM so it
    sees pixel neighbourhoods and produces spatially coherent (s, t).
    """

    def __init__(self, channels: int, context_dim: int,
                 mask_2d: torch.Tensor):
        super().__init__()
        # mask_2d: (H, W), 1 = masked (pass-through), 0 = free (transform)
        self.register_buffer("mask", mask_2d.unsqueeze(0).unsqueeze(0).float())

        c = channels
        self.conv1 = nn.Conv2d(1, c, 3, padding=1)
        self.norm1 = nn.GroupNorm(1, c)
        self.film1 = _SpatialFiLM(c, context_dim)

        self.conv2 = nn.Conv2d(c, c, 3, padding=1)
        self.norm2 = nn.GroupNorm(1, c)
        self.film2 = _SpatialFiLM(c, context_dim)

        self.conv3 = nn.Conv2d(c, c, 3, padding=1)
        self.norm3 = nn.GroupNorm(1, c)
        self.film3 = _SpatialFiLM(c, context_dim)

        self.out = nn.Conv2d(c, 2, 1)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def _st(self, x_2d: torch.Tensor, ctx: torch.Tensor):
        h = F.gelu(self.film1(self.norm1(self.conv1(x_2d)), ctx))
        h = F.gelu(self.film2(self.norm2(self.conv2(h)),   ctx))
        h = F.gelu(self.film3(self.norm3(self.conv3(h)),   ctx))
        st = self.out(h)                             # (B, 2, H, W)
        s, t = st[:, 0:1], st[:, 1:2]
        s = _LOG_SCALE_CLAMP * torch.tanh(s / _LOG_SCALE_CLAMP)
        return s, t

    def forward(self, x: torch.Tensor, ctx: torch.Tensor):
        """x (B, D) -> z (B, D), log_det (B, D)."""
        B, D = x.shape
        H = W = int(D ** 0.5)
        x2 = x.view(B, 1, H, W)
        mask = self.mask                             # (1, 1, H, W)
        s, t = self._st(x2 * mask, ctx)
        free = 1.0 - mask
        z2 = x2 * mask + free * (x2 * torch.exp(s) + t)
        return z2.flatten(1), (free * s).flatten(1)

    def inverse(self, z: torch.Tensor, ctx: torch.Tensor):
        """z (B, D) -> x (B, D)."""
        B, D = z.shape
        H = W = int(D ** 0.5)
        z2 = z.view(B, 1, H, W)
        mask = self.mask
        s, t = self._st(z2 * mask, ctx)
        free = 1.0 - mask
        x2 = z2 * mask + free * ((z2 - t) * torch.exp(-s))
        return x2.flatten(1)


def _checkerboard_masks(H: int, W: int, n_layers: int) -> list[torch.Tensor]:
    """Alternating 2-D checkerboard masks, one per coupling layer."""
    r, c = torch.meshgrid(torch.arange(H), torch.arange(W), indexing="ij")
    even = ((r + c) % 2 == 0).float()
    odd  = 1.0 - even
    return [even if i % 2 == 0 else odd for i in range(n_layers)]


class ConvPixelFlow(nn.Module):
    """Conditional flow over pixel-space sky images.

    Same interface as CoeffFlow; drop-in replacement.

    Parameters
    ----------
    image_size : side length of the square sky image (default 128).
    base_channels : CNN encoder width.
    context_dim : context vector width (fed to encoder and coupling FiLM).
    coupling_channels : channels inside each coupling CNN.
    n_layers : number of coupling layers.
    cond_dim : FiLM conditioning length (5: sigma_local + config one-hot).
    """

    def __init__(
        self,
        image_size: int = 128,
        base_channels: int = 32,
        context_dim: int = 256,
        coupling_channels: int = 32,
        n_layers: int = 8,
        cond_dim: int = COND_DIM,
    ):
        super().__init__()
        self.theta_dim = image_size * image_size
        self.image_size = image_size
        self.encoder = _ContextEncoder(base_channels, context_dim, cond_dim)
        masks = _checkerboard_masks(image_size, image_size, n_layers)
        self.layers = nn.ModuleList(
            _ConvCoupling(coupling_channels, context_dim, m) for m in masks
        )

    def _context(self, image: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        return self.encoder(image, cond)

    def _log_prob_given_ctx(self, theta: torch.Tensor,
                            ctx: torch.Tensor) -> torch.Tensor:
        """Per-dim log-density. theta (B, D), ctx (B, C) -> (B, D)."""
        z = theta
        log_det = torch.zeros_like(theta)
        for layer in self.layers:
            z, ld = layer(z, ctx)
            log_det = log_det + ld
        return -0.5 * (z ** 2 + math.log(2.0 * math.pi)) + log_det

    def log_prob_per_dim(self, theta, image, cond) -> torch.Tensor:
        return self._log_prob_given_ctx(theta, self._context(image, cond))

    def log_prob(
        self,
        theta: torch.Tensor,
        image: torch.Tensor,
        cond:  torch.Tensor,
        dim_weights: torch.Tensor | None = None,
    ) -> torch.Tensor:
        per_dim = self.log_prob_per_dim(theta, image, cond)
        if dim_weights is not None:
            per_dim = per_dim * dim_weights
        return per_dim.sum(dim=-1)

    def log_prob_matrix(self, theta, image, cond) -> torch.Tensor:
        B = theta.shape[0]
        ctx = self._context(image, cond)
        ctx_rep   = ctx.repeat_interleave(B, dim=0)
        theta_rep = theta.repeat(B, 1)
        per_dim = self._log_prob_given_ctx(theta_rep, ctx_rep)
        return per_dim.sum(dim=-1).view(B, B)

    def nll_loss(self, theta, image, cond, dim_weights=None) -> torch.Tensor:
        return -self.log_prob(theta, image, cond, dim_weights).mean()

    @torch.no_grad()
    def sample(self, image: torch.Tensor, cond: torch.Tensor,
               n: int = 1) -> torch.Tensor:
        B = image.shape[0]
        ctx = self._context(image, cond)
        ctx_rep = ctx.repeat_interleave(n, dim=0)
        z = torch.randn(B * n, self.theta_dim,
                        device=image.device, dtype=ctx.dtype)
        for layer in reversed(self.layers):
            z = layer.inverse(z, ctx_rep)
        return z.view(B, n, self.theta_dim)

    def sample_with_grad(self, image: torch.Tensor, cond: torch.Tensor,
                         n: int = 1) -> torch.Tensor:
        B = image.shape[0]
        ctx = self._context(image, cond)
        ctx_rep = ctx.repeat_interleave(n, dim=0)
        z = torch.randn(B * n, self.theta_dim,
                        device=image.device, dtype=ctx.dtype)
        for layer in reversed(self.layers):
            z = layer.inverse(z, ctx_rep)
        return z.view(B, n, self.theta_dim)
