"""
mad_clean.training.latent_flow
==============================
Flow prior p_φ(z) for the VAE latent space (Phase 2).

Trains a Conditional Flow Matching (CFM) model on VAE z codes collected
from clean images (scripts/collect_z_codes.py).  At DPS inference, the
trained prior generates latent samples by integrating its velocity field,
which are then decoded to pixel space via the VAE decoder.

Why an MLP (not UNet)
---------------------
The latent codes z ∈ ℝ^d are vectors, not 2D spatial fields.  The UNet
in flow.py is designed for 2D inputs with spatial inductive biases that
do not apply here.  A 3-layer MLP with sinusoidal time embedding suffices
for d=128 and is ~8× cheaper per forward pass.

Classes
-------
LatentFlowModel
    MLP velocity field v_φ(z_t, t) for latent space flow matching.
    velocity(z_t, t) → v ∈ ℝ^d
    sample(n, device, n_steps) → z ∈ ℝ^(n, d)  [Euler integration]
    save(path), load(path, device)

LatentPriorTrainer
    fit(z_codes, device) → LatentFlowModel
    CFM on z codes: z_0~N(0,I), z_1=z_code, interpolate, learn velocity.
    Loss: heteroscedastic NLL (Kendall & Gal 2017), same as PriorTrainer.
    Gradient clipping at max_norm=1.0 for var_head stability.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn


__all__ = ["LatentFlowModel", "LatentPriorTrainer"]

TIME_DIM = 128   # sinusoidal time embedding dimension


# ---------------------------------------------------------------------------
# Sinusoidal time embedding (same schedule as flow.py)
# ---------------------------------------------------------------------------

def _sinusoidal_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    """t: (B,) → (B, dim) sinusoidal embedding."""
    assert dim % 2 == 0
    half  = dim // 2
    freqs = torch.exp(
        -math.log(10000) * torch.arange(half, dtype=torch.float32, device=t.device) / half
    )
    angles = t[:, None].float() * freqs[None, :]
    return torch.cat([angles.sin(), angles.cos()], dim=-1)


# ---------------------------------------------------------------------------
# MLP velocity network
# ---------------------------------------------------------------------------

class _LatentVelocityNet(nn.Module):
    """
    MLP velocity field for z ∈ ℝ^d.

    Input:  [z_t (d,); sin_emb(t) (TIME_DIM,)] → (d + TIME_DIM,)
    Hidden: Linear → SiLU → Linear → SiLU
    Output: velocity head (d,) + log_var head (d,)
    """

    def __init__(self, latent_dim: int):
        super().__init__()
        d      = latent_dim
        in_dim = d + TIME_DIM

        self.shared = nn.Sequential(
            nn.Linear(in_dim, 512), nn.SiLU(),
            nn.Linear(512,    512), nn.SiLU(),
        )
        self.vel_head     = nn.Linear(512, d)
        self.log_var_head = nn.Linear(512, d)

    def forward(
        self, z: torch.Tensor, t: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        z : (B, d)  — latent code at time t
        t : (B,)    — time values in [0, 1]

        Returns
        -------
        v       : (B, d)  — mean velocity
        log_var : (B, d)  — log-variance of velocity
        """
        temb = _sinusoidal_embedding(t, TIME_DIM)   # (B, TIME_DIM)
        h    = self.shared(torch.cat([z, temb], dim=-1))
        return self.vel_head(h), self.log_var_head(h)


# ---------------------------------------------------------------------------
# LatentFlowModel
# ---------------------------------------------------------------------------

class LatentFlowModel:
    """
    Wrapper around _LatentVelocityNet providing velocity/sample/save/load.

    Usage
    -----
        lf  = LatentFlowModel(latent_dim=128, device="cuda")
        lf2 = LatentFlowModel.load("models/latent_prior_d128.pt", device="cpu")
        v   = lf2.velocity(z_t, t)        # (B, d)
        z   = lf2.sample(8, device="cuda") # (8, d) posterior samples
    """

    def __init__(self, latent_dim: int = 128, device: str = "cpu"):
        self.latent_dim = latent_dim
        self.device     = torch.device(device)
        self._net       = _LatentVelocityNet(latent_dim).to(self.device)

    # ------------------------------------------------------------------
    def velocity(self, z: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Mean velocity v_φ(z_t, t) — no gradient, eval mode.

        Parameters
        ----------
        z : (B, d) float32
        t : (B,)   float32 in [0, 1]

        Returns
        -------
        v : (B, d) float32
        """
        self._net.eval()
        z = z.to(self.device)
        t = t.to(self.device)
        with torch.no_grad():
            v, _ = self._net(z, t)
        return v

    # ------------------------------------------------------------------
    def sample(
        self,
        n       : int,
        device  : str | torch.device = "cpu",
        n_steps : int = 50,
    ) -> torch.Tensor:
        """
        Generate n latent samples by Euler integration of the velocity field.

        Integrates from z_0 ~ N(0, I) at t=0 to z_1 at t=1.

        Parameters
        ----------
        n       : number of samples
        device  : target device
        n_steps : Euler discretisation steps (default 50)

        Returns
        -------
        z : (n, d) float32  — samples from the learned prior p_φ(z)
        """
        dev = torch.device(device)
        self._net.eval()
        z  = torch.randn(n, self.latent_dim, device=dev)
        dt = 1.0 / n_steps

        with torch.no_grad():
            for i in range(n_steps):
                t    = torch.full((n,), i * dt, device=dev)
                v, _ = self._net(z, t)
                z    = z + dt * v

        return z

    # ------------------------------------------------------------------
    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {"state_dict": self._net.state_dict(),
             "latent_dim": self.latent_dim,
             "device":     str(self.device)},
            path,
        )
        print(f"LatentFlowModel saved → {path}")

    @classmethod
    def load(cls, path: str | Path, device: Optional[str] = None) -> "LatentFlowModel":
        path = Path(path)
        ckpt = torch.load(path, map_location="cpu", weights_only=True)
        dev  = device if device is not None else ckpt.get("device", "cpu")
        d    = ckpt["latent_dim"]
        lf   = cls(latent_dim=d, device=dev)
        lf._net.load_state_dict(ckpt["state_dict"])
        lf._net.eval()
        print(f"LatentFlowModel loaded ← {path}  (device={dev}, d={d})")
        return lf

    def __repr__(self) -> str:
        n = sum(p.numel() for p in self._net.parameters())
        return f"LatentFlowModel(latent_dim={self.latent_dim}, device={self.device}, params={n:,})"


# ---------------------------------------------------------------------------
# LatentPriorTrainer
# ---------------------------------------------------------------------------

class LatentPriorTrainer:
    """
    Train a LatentFlowModel on VAE z codes using Conditional Flow Matching.

    Algorithm (per minibatch)
    -------------------------
    1. z_0 ~ N(0, I)          — Gaussian noise in ℝ^d
       z_1 = z_code           — VAE encoding of a clean source image
    2. t  ~ U[0, 1]
    3. z_t = (1 - t) * z_0 + t * z_1    [straight-line interpolant]
    4. u   = z_1 - z_0                  [target velocity]
    5. Loss: heteroscedastic NLL (Kendall & Gal 2017)
         L = 0.5 * [(v - u)² / exp(log_var) + log_var]
    6. Adam step + gradient clipping (max_norm=1.0)

    Parameters
    ----------
    n_epochs   : int    (default 300)
    batch_size : int    (default 64)
    lr         : float  (default 1e-3)
    random_seed: int    (default 42)
    """

    def __init__(
        self,
        n_epochs   : int   = 300,
        batch_size : int   = 64,
        lr         : float = 1e-3,
        random_seed: int   = 42,
    ):
        self.n_epochs    = n_epochs
        self.batch_size  = batch_size
        self.lr          = lr
        self.random_seed = random_seed

    def fit(
        self,
        z_codes     : np.ndarray,
        device      : str = "cpu",
        resume_from : Optional[str | Path] = None,
        out_path    : Optional[str | Path] = None,
    ) -> LatentFlowModel:
        """
        Train on (N, d) float32 z codes from collect_z_codes.py.

        Parameters
        ----------
        z_codes     : np.ndarray (N, d) float32
        device      : torch device string
        resume_from : path to existing .pt checkpoint to resume from
        out_path    : if provided, best model (lowest loss) is saved to
                      out_path.with_suffix('.best.pt') after every improvement.
        """
        assert z_codes.ndim == 2, f"Expected (N, d), got {z_codes.shape}"
        N, d = z_codes.shape

        dev = torch.device(device)
        rng = np.random.default_rng(self.random_seed)

        if resume_from is not None:
            lf = LatentFlowModel.load(resume_from, device=device)
            print(f"Resuming from {resume_from} — running {self.n_epochs} additional epochs")
        else:
            lf = LatentFlowModel(latent_dim=d, device=device)

        optimizer = torch.optim.Adam(lf._net.parameters(), lr=self.lr)

        best_path = Path(out_path).with_suffix(".best.pt") if out_path is not None else None
        best_loss = float("inf")

        print(f"LatentPriorTrainer: N={N}  d={d}  "
              f"batch={self.batch_size}  epochs={self.n_epochs}  device={dev}")
        if best_path is not None:
            print(f"  Best checkpoint → {best_path}")

        for epoch in range(self.n_epochs):
            idx          = rng.permutation(N)
            epoch_loss   = 0.0
            epoch_sigma  = 0.0
            n_batches    = 0

            lf._net.train()
            for b_start in range(0, N, self.batch_size):
                batch_idx = idx[b_start : b_start + self.batch_size]
                z1 = torch.from_numpy(z_codes[batch_idx]).float().to(dev)  # (B, d)
                z0 = torch.randn_like(z1)
                B  = z1.shape[0]

                t  = torch.rand(B, device=dev)
                t2 = t[:, None]                          # (B, 1) for broadcasting

                z_t      = (1.0 - t2) * z0 + t2 * z1   # interpolant
                u_target = z1 - z0                       # target velocity

                optimizer.zero_grad()
                v_pred, log_var = lf._net(z_t, t)
                log_var = log_var.clamp(-10, 10)
                loss = 0.5 * (
                    (v_pred - u_target) ** 2 / log_var.exp() + log_var
                ).mean()
                loss.backward()
                nn.utils.clip_grad_norm_(lf._net.parameters(), max_norm=1.0)
                optimizer.step()

                epoch_loss  += float(loss)
                epoch_sigma += log_var.detach().mul(0.5).exp().mean().item()
                n_batches   += 1

            avg_loss = epoch_loss / n_batches
            print(f"  Epoch {epoch + 1:3d}/{self.n_epochs}  "
                  f"loss={avg_loss:.3e}  "
                  f"mean_sigma={epoch_sigma / n_batches:.3f}", flush=True)

            if best_path is not None and avg_loss < best_loss:
                best_loss = avg_loss
                lf.save(best_path)
                print(f"    ↳ best checkpoint saved (loss={best_loss:.3e})", flush=True)

        lf._net.eval()
        return lf

    def __repr__(self) -> str:
        return (f"LatentPriorTrainer(n_epochs={self.n_epochs}, "
                f"batch_size={self.batch_size}, lr={self.lr})")
