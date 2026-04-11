"""
mad_clean.training.psf_flow
============================
PSF-conditioned flow matching for radio deconvolution.

The network takes a 2-channel input (dirty, PSF) and learns to predict
the velocity field toward the clean sky image. The PSF is an explicit
input — the model is PSF-agnostic at training time and generalises to
unseen PSFs at inference.

Classes
-------
PSFUNetVelocityField
    Same U-Net as UNetVelocityField but with a 2-channel input (dirty + PSF).

PSFFlowModel
    Thin wrapper: save / load / velocity interface.

PSFFlowTrainer
    Trains on (dirty, psf, clean) triples from psf_pairs.npz.
    Uses conditional flow matching: x_0=dirty, x_1=clean, straight-line path.
    Loss: heteroscedastic NLL (same as FlowTrainer).
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F_
from torch.utils.data import DataLoader

__all__ = ["PSFFlowModel", "PSFFlowTrainer"]


# ---------------------------------------------------------------------------
# Shared building blocks (copied from flow.py — kept local to avoid coupling)
# ---------------------------------------------------------------------------

def _sinusoidal_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    assert dim % 2 == 0
    half  = dim // 2
    freqs = torch.exp(
        -math.log(10000) * torch.arange(half, dtype=torch.float32, device=t.device) / half
    )
    angles = t[:, None].float() * freqs[None, :]
    return torch.cat([angles.sin(), angles.cos()], dim=-1)


class _ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, time_dim: int):
        super().__init__()
        self.conv1     = nn.Conv2d(in_ch,  out_ch, 3, padding=1)
        self.conv2     = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.norm1     = nn.GroupNorm(min(8, out_ch), out_ch)
        self.norm2     = nn.GroupNorm(min(8, out_ch), out_ch)
        self.time_proj = nn.Linear(time_dim, out_ch)
        self.act       = nn.SiLU()

    def forward(self, x: torch.Tensor, temb: torch.Tensor) -> torch.Tensor:
        h = self.act(self.norm1(self.conv1(x)))
        h = h + self.time_proj(temb)[:, :, None, None]
        h = self.act(self.norm2(self.conv2(h)))
        return h


class _Downsample(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.conv = nn.Conv2d(ch, ch, 3, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class _Upsample(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.conv = nn.ConvTranspose2d(ch, ch, kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


# ---------------------------------------------------------------------------
# PSF-conditioned U-Net velocity field
# ---------------------------------------------------------------------------

class PSFUNetVelocityField(nn.Module):
    """
    U-Net velocity field conditioned on the PSF via a 2-channel input.

    Input  : (B, 2, 150, 150) — channel 0: dirty image, channel 1: PSF
    Output : (B, 1, 150, 150) — predicted velocity (dirty → clean direction)

    Architecture identical to UNetVelocityField except the first conv
    takes 2 input channels instead of 1.
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

        # Encoder — 2 input channels
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
        self.dec1  = _ConvBlock(64 + 32,  32, td)

        # Output heads
        self.head     = nn.Conv2d(32, 1, 1)
        self.var_head = nn.Conv2d(32, 1, 1)

    def forward(
        self, x: torch.Tensor, t: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        x : (B, 2, H, W)  — [dirty, PSF] stacked along channel dim
        t : (B,)           — time values in [0, 1]

        Returns
        -------
        v       : (B, 1, H, W)
        log_var : (B, 1, H, W)
        """
        temb = _sinusoidal_embedding(t, self.TIME_DIM)
        temb = self.time_mlp(temb)

        e1 = self.enc1(x,              temb)
        e2 = self.enc2(self.down1(e1), temb)
        b  = self.bot1(self.down2(e2), temb)
        b  = self.bot2(b,              temb)

        d2 = self.up2(b)
        d2 = F_.interpolate(d2, size=e2.shape[2:], mode="bilinear", align_corners=False)
        d2 = self.dec2(torch.cat([d2, e2], dim=1), temb)

        d1 = self.up1(d2)
        d1 = F_.interpolate(d1, size=e1.shape[2:], mode="bilinear", align_corners=False)
        d1 = self.dec1(torch.cat([d1, e1], dim=1), temb)

        return self.head(d1), self.var_head(d1)


# ---------------------------------------------------------------------------
# PSFFlowModel — save / load / inference wrapper
# ---------------------------------------------------------------------------

class PSFFlowModel:
    """
    Wrapper around PSFUNetVelocityField.

    Usage
    -----
        fm = PSFFlowModel.load("models/psf_flow_model.pt", device="cuda")
        # dirty: (1, 1, H, W), psf: (1, 1, H, W)
        x_in = torch.cat([dirty, psf], dim=1)   # (1, 2, H, W)
        v    = fm.velocity(x_in, t)              # (1, 1, H, W)
    """

    def __init__(self, device: str = "cpu"):
        self.device = torch.device(device)
        self._net   = PSFUNetVelocityField().to(self.device)

    def velocity(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """x: (B, 2, H, W), t: (B,) → v: (B, 1, H, W)."""
        self._net.eval()
        with torch.no_grad():
            v, _ = self._net(x.to(self.device), t.to(self.device))
        return v

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self._net.state_dict(), path)
        print(f"PSFFlowModel saved → {path}")

    @classmethod
    def load(cls, path: str | Path, device: str = "cpu") -> "PSFFlowModel":
        path  = Path(path)
        model = cls(device=device)
        model._net.load_state_dict(
            torch.load(path, map_location=model.device, weights_only=True)
        )
        model._net.eval()
        print(f"PSFFlowModel loaded ← {path}  (device={device})")
        return model

    def decode_island(
        self,
        dirty: torch.Tensor,
        psf:   torch.Tensor,
        n_steps: int = 16,
    ) -> torch.Tensor:
        """
        Euler ODE from dirty → clean conditioned on psf.

        The PSF channel is fixed throughout. Only the image channel evolves.

        Parameters
        ----------
        dirty   : (H, W) float32 Tensor
        psf     : (H, W) float32 Tensor  peak=1
        n_steps : Euler integration steps

        Returns
        -------
        clean_hat : (H, W) float32 Tensor
        """
        dev = self.device
        net = self._net.eval()
        dt  = 1.0 / n_steps

        # Image channel evolves; PSF channel is fixed
        x_img = dirty.unsqueeze(0).unsqueeze(0).to(dev)   # (1, 1, H, W)
        p     = psf.unsqueeze(0).unsqueeze(0).to(dev)     # (1, 1, H, W)

        with torch.no_grad():
            for step in range(n_steps):
                t_batch = torch.full((1,), step * dt, device=dev)
                x_in    = torch.cat([x_img, p], dim=1)    # (1, 2, H, W)
                v, _    = net(x_in, t_batch)               # (1, 1, H, W)
                x_img   = x_img + dt * v

        return x_img[0, 0]   # (H, W)


# ---------------------------------------------------------------------------
# PSFFlowTrainer
# ---------------------------------------------------------------------------

class PSFFlowTrainer:
    """
    Train PSFFlowModel on (dirty, psf, clean) triples.

    Conditional flow matching: straight-line path from dirty to clean.
        x_0 = dirty
        x_1 = clean
        x_t = (1-t)*x_0 + t*x_1
        u_t = x_1 - x_0  (target velocity)

    The 2-channel input to the network is [x_t, psf].

    Loss: heteroscedastic NLL
        L = 0.5 * (u - v)² * exp(-log_var) + 0.5 * log_var

    Parameters
    ----------
    n_epochs   : int
    batch_size : int
    lr         : float
    """

    def __init__(
        self,
        n_epochs   : int   = 100,
        batch_size : int   = 16,
        lr         : float = 1e-4,
    ):
        self.n_epochs   = n_epochs
        self.batch_size = batch_size
        self.lr         = lr

    def fit(
        self,
        data_root   : str | Path,
        device      : str        = "cpu",
        resume_from : str | None = None,
        out_path    : str | Path | None = None,
        num_workers : int        = 4,
    ) -> "PSFFlowModel":
        """
        Train and return a PSFFlowModel.

        Parameters
        ----------
        data_root   : path to psf_pairs/ directory (from generate_psf_data.py)
        device      : 'cpu' or 'cuda'
        resume_from : optional checkpoint path to warm-start from
        out_path    : save best checkpoint here during training
        num_workers : DataLoader worker processes
        """
        from mad_clean.data.psf_dataset import PSFPairsDataset

        dev   = torch.device(device)
        model = PSFFlowModel(device=device)

        if resume_from is not None:
            model._net.load_state_dict(
                torch.load(resume_from, map_location=dev, weights_only=True)
            )
            print(f"Resumed from {resume_from}")

        dataset = PSFPairsDataset(data_root, train=True)
        loader  = DataLoader(
            dataset,
            batch_size  = self.batch_size,
            shuffle     = True,
            num_workers = num_workers,
            pin_memory  = (device == "cuda"),
            persistent_workers = (num_workers > 0),
        )

        # Multi-GPU via DataParallel if available
        n_gpus = torch.cuda.device_count() if device == "cuda" else 0
        net    = model._net
        if n_gpus > 1:
            net = torch.nn.DataParallel(model._net)
            print(f"Using {n_gpus} GPUs via DataParallel")

        print(f"PSFFlowTrainer: {len(dataset):,} training pairs  "
              f"batch={self.batch_size}  epochs={self.n_epochs}  "
              f"workers={num_workers}  device={device}")

        opt       = torch.optim.Adam(model._net.parameters(), lr=self.lr)
        best_loss = float("inf")

        for epoch in range(1, self.n_epochs + 1):
            model._net.train()
            epoch_loss = 0.0
            n_batches  = 0

            for dirty, psf, clean in loader:
                x0 = dirty.to(dev)   # (B, H, W)
                x1 = clean.to(dev)
                p  = psf.to(dev)

                t  = torch.rand(x0.shape[0], device=dev)
                t4 = t[:, None, None]
                xt = (1 - t4) * x0 + t4 * x1
                u  = x1 - x0

                x_in = torch.stack([xt, p], dim=1)        # (B, 2, H, W)

                v, log_var = net(x_in, t)
                v       = v.squeeze(1)
                log_var = log_var.squeeze(1).clamp(-10, 10)
                loss    = (0.5 * (u - v) ** 2 * torch.exp(-log_var)
                           + 0.5 * log_var).mean()

                opt.zero_grad()
                loss.backward()
                opt.step()

                epoch_loss += loss.item()
                n_batches  += 1

            avg_loss = epoch_loss / max(n_batches, 1)
            print(f"  Epoch {epoch:4d}/{self.n_epochs}  loss={avg_loss:.4e}", flush=True)

            if avg_loss < best_loss and out_path is not None:
                best_loss = avg_loss
                model.save(Path(str(out_path).replace(".pt", ".best.pt")))

        if out_path is not None:
            model.save(out_path)

        return model
