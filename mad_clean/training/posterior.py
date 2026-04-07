"""
mad_clean.training.posterior
=============================
Phase 2: Amortised posterior via conditional flow matching.

Replaces DPS's expensive N×50-step-per-island sampling with a single conditional
flow q_φ(x_clean | dirty) trained on DPS posterior samples (posterior distillation).
At inference: N samples cost N×20 forward passes instead of N×50 — and with
fewer steps because the dirty image is available as direct conditioning.

Training data: crumb_data/dps_samples.npz (produced by scripts/collect_dps_samples.py).

Classes
-------
ConditionalUNetVelocityField
    UNetVelocityField with 2-channel input (x_t concatenated with dirty_conditioning).
    Same architecture otherwise — ~2.5M params.

ConditionalFlowModel
    Thin wrapper: save/load/velocity interface for the conditional network.
    velocity(x_t, dirty, t) → v
    sample(dirty, n_samples) → (mean, std)

AmortisedPosteriorTrainer
    fit(dps_samples_path, device) → ConditionalFlowModel
    Trains on (dirty, x_1_sample) pairs from DPS output.
    x_1_sample is one draw from the DPS posterior for a given dirty image.
    Loss: conditional CFM heteroscedastic NLL.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F_

from .flow import (
    _sinusoidal_embedding,
    _ConvBlock,
    _Downsample,
    _Upsample,
    _augment_pair,
)


__all__ = ["ConditionalUNetVelocityField", "ConditionalFlowModel", "AmortisedPosteriorTrainer"]


# ---------------------------------------------------------------------------
# Conditional UNet velocity field (2-channel input: x_t + dirty)
# ---------------------------------------------------------------------------

class ConditionalUNetVelocityField(nn.Module):
    """
    2-channel UNet velocity field for the amortised posterior.

    Input  : (x_t, dirty) concatenated → (B, 2, 150, 150)
    Output : (velocity, log_var) — both (B, 1, 150, 150)

    Architecture is identical to UNetVelocityField except enc1 takes 2 channels.
    All other layers (encoder depth, bottleneck, decoder, heads) are unchanged.
    """

    TIME_DIM = 128

    def __init__(self):
        super().__init__()
        td = self.TIME_DIM

        self.time_mlp = nn.Sequential(
            nn.Linear(td, td * 4),
            nn.SiLU(),
            nn.Linear(td * 4, td),
        )

        # Encoder — enc1 takes 2 channels (x_t + dirty conditioning)
        self.enc1  = _ConvBlock(2,   32,  td)
        self.down1 = _Downsample(32)
        self.enc2  = _ConvBlock(32,  64,  td)
        self.down2 = _Downsample(64)

        # Bottleneck
        self.bot1  = _ConvBlock(64,  128, td)
        self.bot2  = _ConvBlock(128, 128, td)

        # Decoder
        self.up2   = _Upsample(128)
        self.dec2  = _ConvBlock(128 + 64, 64, td)
        self.up1   = _Upsample(64)
        self.dec1  = _ConvBlock(64 + 32, 32, td)

        # Output heads
        self.head     = nn.Conv2d(32, 1, 1)
        self.var_head = nn.Conv2d(32, 1, 1)

    def forward(
        self, x: torch.Tensor, t: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        x : (B, 2, H, W) — x_t concatenated with dirty conditioning
        t : (B,) — time values in [0, 1]

        Returns
        -------
        v       : (B, 1, H, W) — mean velocity
        log_var : (B, 1, H, W) — log-variance of velocity
        """
        temb = _sinusoidal_embedding(t, self.TIME_DIM)
        temb = self.time_mlp(temb)

        e1 = self.enc1(x,            temb)
        e2 = self.enc2(self.down1(e1), temb)
        b  = self.bot1(self.down2(e2), temb)
        b  = self.bot2(b,            temb)

        d2 = self.up2(b)
        d2 = F_.interpolate(d2, size=e2.shape[2:], mode="bilinear", align_corners=False)
        d2 = self.dec2(torch.cat([d2, e2], dim=1), temb)

        d1 = self.up1(d2)
        d1 = F_.interpolate(d1, size=e1.shape[2:], mode="bilinear", align_corners=False)
        d1 = self.dec1(torch.cat([d1, e1], dim=1), temb)

        return self.head(d1), self.var_head(d1)


# ---------------------------------------------------------------------------
# ConditionalFlowModel — wrapper
# ---------------------------------------------------------------------------

class ConditionalFlowModel:
    """
    Wrapper around ConditionalUNetVelocityField providing save/load/velocity.

    Usage
    -----
        cfm = ConditionalFlowModel(device="cuda")
        # ... train via AmortisedPosteriorTrainer.fit() ...
        cfm.save("models/amortised_posterior.pt")

        cfm2 = ConditionalFlowModel.load("models/amortised_posterior.pt")
        v    = cfm2.velocity(x_t, dirty_cond, t)   # (B, 1, H, W)
    """

    def __init__(self, device: str = "cpu"):
        self.device = torch.device(device)
        self._net   = ConditionalUNetVelocityField().to(self.device)

    def velocity(
        self,
        x_t   : torch.Tensor,   # (B, 1, H, W)
        dirty  : torch.Tensor,   # (B, 1, H, W) — conditioning
        t      : torch.Tensor,   # (B,)
    ) -> torch.Tensor:
        """Mean velocity v_φ(x_t, dirty, t) — (B, 1, H, W)."""
        self._net.eval()
        x_t   = x_t.to(self.device)
        dirty = dirty.to(self.device)
        t     = t.to(self.device)
        inp   = torch.cat([x_t, dirty], dim=1)   # (B, 2, H, W)
        with torch.no_grad():
            v, _ = self._net(inp, t)
        return v

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {"state_dict": self._net.state_dict(), "device": str(self.device)},
            path,
        )
        print(f"ConditionalFlowModel saved → {path}")

    @classmethod
    def load(cls, path: str | Path, device: Optional[str] = None) -> "ConditionalFlowModel":
        path = Path(path)
        ckpt = torch.load(path, map_location="cpu", weights_only=True)
        dev  = device if device is not None else ckpt.get("device", "cpu")
        cfm  = cls(device=dev)
        missing, _ = cfm._net.load_state_dict(ckpt["state_dict"], strict=False)
        if missing:
            print(f"ConditionalFlowModel loaded ← {path}  "
                  f"[{len(missing)} keys missing — warm start]")
        else:
            print(f"ConditionalFlowModel loaded ← {path}  (device={dev})")
        cfm._net.eval()
        return cfm

    def __repr__(self) -> str:
        n = sum(p.numel() for p in self._net.parameters())
        return f"ConditionalFlowModel(device={self.device}, params={n:,})"


# ---------------------------------------------------------------------------
# AmortisedPosteriorTrainer
# ---------------------------------------------------------------------------

class AmortisedPosteriorTrainer:
    """
    Train a conditional flow q_φ(x_clean | dirty) via posterior distillation.

    Training data
    -------------
    Expects crumb_data/dps_samples.npz (from scripts/collect_dps_samples.py)
    with keys:
        dirty   : (N, H, W) float32 — conditioning dirty images
        samples : (N, M, H, W) float32 — M DPS posterior draws per image

    Algorithm (per minibatch)
    -------------------------
    1. Pick one sample x_1 ~ Uniform({samples[i, 0..M-1]}) per image
    2. x_0 = N(0,1)  (noise starting point)
    3. t ~ U[0,1]
    4. x_t = (1-t)*x_0 + t*x_1
    5. u  = x_1 - x_0  (CFM target velocity)
    6. inp = cat([x_t, dirty_conditioning], dim=1)   (2-channel input)
    7. v, log_var = net(inp, t)
    8. Loss = heteroscedastic NLL:  0.5 * [(v-u)² / exp(log_var) + log_var]

    D4 augmentation is applied identically to (dirty, x_1) pair to preserve
    spatial correspondence.

    Parameters
    ----------
    n_epochs   : int   (default 300 — fewer than PriorTrainer since distillation
                         converges faster with direct conditioning)
    batch_size : int   (default 16)
    lr         : float (default 1e-4)
    random_seed: int   (default 42)
    """

    def __init__(
        self,
        n_epochs   : int   = 300,
        batch_size : int   = 16,
        lr         : float = 1e-4,
        random_seed: int   = 42,
    ):
        self.n_epochs    = n_epochs
        self.batch_size  = batch_size
        self.lr          = lr
        self.random_seed = random_seed

    def fit(
        self,
        dps_samples_path : str | Path,
        device           : str = "cpu",
        resume_from      : Optional[str | Path] = None,
    ) -> ConditionalFlowModel:
        """
        Train on DPS posterior samples from dps_samples.npz.

        Parameters
        ----------
        dps_samples_path : path to dps_samples.npz
        device           : torch device string
        resume_from      : path to an existing .pt checkpoint to warm-start from
        """
        data    = np.load(dps_samples_path)
        dirty   = data["dirty"].astype(np.float32)     # (N, H, W)
        samples = data["samples"].astype(np.float32)   # (N, M, H, W)
        N, M, H, W = samples.shape

        dev = torch.device(device)
        rng = np.random.default_rng(self.random_seed)

        if resume_from is not None:
            cfm = ConditionalFlowModel.load(resume_from, device=device)
            print(f"Resuming from {resume_from}")
        else:
            cfm = ConditionalFlowModel(device=device)
        optimizer = torch.optim.Adam(cfm._net.parameters(), lr=self.lr)

        print(f"AmortisedPosteriorTrainer: N={N}  M={M}  {H}×{W}  "
              f"batch={self.batch_size}  epochs={self.n_epochs}  device={dev}")

        for epoch in range(self.n_epochs):
            idx        = rng.permutation(N)
            epoch_loss = 0.0
            n_batches  = 0

            cfm._net.train()
            for b_start in range(0, N, self.batch_size):
                batch_idx  = idx[b_start : b_start + self.batch_size]

                # Pick one posterior sample per image (random draw from M)
                sample_idx = rng.integers(0, M, size=len(batch_idx))
                pairs = [
                    _augment_pair(dirty[i], samples[i, sample_idx[k]], rng)
                    for k, i in enumerate(batch_idx)
                ]
                dirty_np  = np.stack([p[0] for p in pairs])    # (B, H, W)
                x1_np     = np.stack([p[1] for p in pairs])    # (B, H, W)

                dirty_t = torch.from_numpy(dirty_np[:, None]).float().to(dev)  # (B,1,H,W)
                x1      = torch.from_numpy(x1_np[:, None]).float().to(dev)     # (B,1,H,W)
                x0      = torch.randn_like(x1)
                B       = x1.shape[0]

                t  = torch.rand(B, device=dev)
                t4 = t[:, None, None, None]

                x_t      = (1.0 - t4) * x0 + t4 * x1
                u_target = x1 - x0

                inp = torch.cat([x_t, dirty_t], dim=1)     # (B, 2, H, W)

                optimizer.zero_grad()
                v_pred, log_var = cfm._net(inp, t)
                log_var = log_var.clamp(-10, 10)
                loss = 0.5 * (
                    (v_pred - u_target) ** 2 / log_var.exp() + log_var
                ).mean()
                loss.backward()
                optimizer.step()

                epoch_loss += float(loss)
                n_batches  += 1

            mean_sigma = log_var.detach().mul(0.5).exp().mean().item()
            print(f"  Epoch {epoch + 1:3d}/{self.n_epochs}  "
                  f"loss={epoch_loss / n_batches:.3e}  "
                  f"mean_sigma={mean_sigma:.3f}", flush=True)

        cfm._net.eval()
        return cfm

    def __repr__(self) -> str:
        return (f"AmortisedPosteriorTrainer(n_epochs={self.n_epochs}, "
                f"batch_size={self.batch_size}, lr={self.lr})")
