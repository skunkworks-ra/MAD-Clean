"""
mad_clean.training.vae
======================
Convolutional Variational Autoencoder for radio source morphology.

Encodes 150×150 clean source images to a d-dimensional latent space z ∈ ℝ^d
and reconstructs them via a non-linear decoder.  The decoder supports autograd
so that ∇_z = (∂Dec/∂ẑ₁)^T · ∇_x can be computed during LatentDPSSolver
inference (Phase 2 DPS).

Design decisions
----------------
- No final activation on the decoder.  Radio Stokes I is non-negative, but
  Stokes Q/U are signed.  Clipping to non-negative is deferred to the catalogue
  step so the model can generalise across polarisations.
- Encoder uses GroupNorm + SiLU, consistent with flow.py, for stable gradient
  flow through the deep encoder.
- Spatial handling: encoder 150→75→37→18 (three stride-2 convs, floor division).
  Decoder starts from 19×19 via a linear projection, producing 152×152 after
  three stride-2 transposed convs, then cropped to 150×150.  This avoids
  bilinear interpolation artefacts in the output.
- PSF-agnostic: the VAE never sees dirty images or the PSF during training.
- Per-image peak normalisation: x_norm = x / x.max(), mapping each image to [0, 1].
  This compresses dynamic range so bright and faint sources contribute equally to
  the MSE loss.  The VAE learns morphology only; flux is recovered at inference.

Classes
-------
VAEModel
    Encoder E: x → (μ, logσ) ∈ ℝ^d
    Decoder Dec: z → x̂ ∈ ℝ^(150×150)
    encode(x), decode(z), forward(x) → (x̂, μ, logσ)
    save(path), load(path, device)

VAETrainer
    fit(clean, device) → VAEModel
    Loss: MSE(x̂, x) + β * KL(N(μ,σ²) || N(0,I))
    Parameters: n_epochs=200, batch_size=16, lr=1e-3, beta=1.0
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F_


__all__ = ["VAEModel", "VAETrainer"]


# ---------------------------------------------------------------------------
# Encoder building block: Conv + GroupNorm + SiLU
# ---------------------------------------------------------------------------

class _EncoderBlock(nn.Module):
    """Strided conv (2×) + GroupNorm + SiLU — one resolution level."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1)
        self.norm = nn.GroupNorm(min(8, out_ch), out_ch)
        self.act  = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.norm(self.conv(x)))


# ---------------------------------------------------------------------------
# VAEModel
# ---------------------------------------------------------------------------

_ENC_SPATIAL = 18
_DEC_SPATIAL = 19
_ENC_CH      = 128
_ENC_FLAT    = _ENC_CH * _ENC_SPATIAL * _ENC_SPATIAL   # 41 472
_DEC_FLAT    = _ENC_CH * _DEC_SPATIAL * _DEC_SPATIAL   # 43 264
_OUT_SIZE    = 150


class VAEModel:
    """
    Convolutional VAE for 150×150 radio source images.

    Usage
    -----
        vae = VAEModel(latent_dim=128, device="cuda")
        vae.save("models/vae_d128.pt")
        vae2 = VAEModel.load("models/vae_d128.pt", device="cpu")
        z    = vae2.encode(x)[0]   # use mean; (B, d)
        xhat = vae2.decode(z)      # (B, 1, 150, 150)
    """

    def __init__(self, latent_dim: int = 128, device: str = "cpu"):
        self.latent_dim = latent_dim
        self.device     = torch.device(device)
        self._net       = _VAENet(latent_dim).to(self.device)

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """(B,1,150,150) → (mu (B,d), logsd (B,d))"""
        x = x.to(self.device)
        return self._net.encode(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """(B,d) → (B,1,150,150)"""
        z = z.to(self.device)
        return self._net.decode(z)

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """(B,1,150,150) → (x_hat, mu, logsd)"""
        x = x.to(self.device)
        return self._net(x)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {"state_dict": self._net.state_dict(),
             "latent_dim": self.latent_dim,
             "device":     str(self.device)},
            path,
        )
        print(f"VAEModel saved → {path}")

    @classmethod
    def load(cls, path: str | Path, device: Optional[str] = None) -> "VAEModel":
        path = Path(path)
        ckpt = torch.load(path, map_location="cpu", weights_only=True)
        dev  = device if device is not None else ckpt.get("device", "cpu")
        d    = ckpt["latent_dim"]
        vae  = cls(latent_dim=d, device=dev)
        vae._net.load_state_dict(ckpt["state_dict"])
        vae._net.eval()
        print(f"VAEModel loaded ← {path}  (device={dev}, d={d})")
        return vae

    def __repr__(self) -> str:
        n = sum(p.numel() for p in self._net.parameters())
        return f"VAEModel(latent_dim={self.latent_dim}, device={self.device}, params={n:,})"


# ---------------------------------------------------------------------------
# Internal nn.Module
# ---------------------------------------------------------------------------

class _VAENet(nn.Module):
    def __init__(self, latent_dim: int):
        super().__init__()
        d = latent_dim

        self.enc1 = _EncoderBlock(1,   32)
        self.enc2 = _EncoderBlock(32,  64)
        self.enc3 = _EncoderBlock(64, 128)
        self.fc_mu    = nn.Linear(_ENC_FLAT, d)
        self.fc_logsd = nn.Linear(_ENC_FLAT, d)

        self.fc_dec = nn.Linear(d, _DEC_FLAT)
        self.dec1 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(8, 64), nn.SiLU(),
        )
        self.dec2 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(8, 32), nn.SiLU(),
        )
        self.dec3 = nn.ConvTranspose2d(32, 1, kernel_size=4, stride=2, padding=1)

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.enc3(self.enc2(self.enc1(x))).flatten(1)
        return self.fc_mu(h), self.fc_logsd(h)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        h = self.fc_dec(z).view(z.shape[0], _ENC_CH, _DEC_SPATIAL, _DEC_SPATIAL)
        h = self.dec3(self.dec2(self.dec1(h)))
        return h[:, :, 1:-1, 1:-1]   # crop 152→150

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logsd = self.encode(x)
        z   = mu + logsd.exp() * torch.randn_like(mu)   # reparameterisation
        return self.decode(z), mu, logsd


# ---------------------------------------------------------------------------
# VAETrainer
# ---------------------------------------------------------------------------

def _augment_single(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    k    = int(rng.integers(0, 4))
    flip = bool(rng.integers(0, 2))
    img  = np.rot90(img, k=k)
    if flip:
        img = np.fliplr(img)
    return np.ascontiguousarray(img)


class VAETrainer:
    """
    Train a VAEModel on clean source images (no dirty, no PSF).

    Loss: MSE(x̂, x) + β * KL[N(μ,σ²) || N(0,I)]

    Parameters
    ----------
    n_epochs   : int    (default 200)
    batch_size : int    (default 16)
    lr         : float  (default 1e-3)
    beta       : float  KL weight (default 1.0)
    latent_dim : int    (default 128)
    random_seed: int    (default 42)
    """

    def __init__(
        self,
        n_epochs   : int   = 200,
        batch_size : int   = 16,
        lr         : float = 1e-3,
        beta       : float = 1.0,
        latent_dim : int   = 128,
        random_seed: int   = 42,
    ):
        self.n_epochs    = n_epochs
        self.batch_size  = batch_size
        self.lr          = lr
        self.beta        = beta
        self.latent_dim  = latent_dim
        self.random_seed = random_seed

    def fit(
        self,
        clean       : np.ndarray,
        device      : str = "cpu",
        resume_from : Optional[str | Path] = None,
    ) -> VAEModel:
        """Train on (N, H, W) clean float32 images."""
        assert clean.ndim == 3
        N, H, W = clean.shape
        assert H == W == _OUT_SIZE, f"Expected 150×150, got {H}×{W}"

        dev = torch.device(device)
        rng = np.random.default_rng(self.random_seed)

        if resume_from is not None:
            vae = VAEModel.load(resume_from, device=device)
            print(f"Resuming from {resume_from} — running {self.n_epochs} additional epochs")
        else:
            vae = VAEModel(latent_dim=self.latent_dim, device=device)

        optimizer = torch.optim.Adam(vae._net.parameters(), lr=self.lr)

        print(f"VAETrainer: N={N}  {H}×{W}  d={self.latent_dim}  "
              f"beta={self.beta}  batch={self.batch_size}  "
              f"epochs={self.n_epochs}  device={dev}")

        for epoch in range(self.n_epochs):
            idx       = rng.permutation(N)
            epoch_mse = 0.0
            epoch_kl  = 0.0
            n_batches = 0

            vae._net.train()
            for b_start in range(0, N, self.batch_size):
                batch_idx = idx[b_start : b_start + self.batch_size]
                clean_aug = [_augment_single(clean[i], rng) for i in batch_idx]
                x = torch.from_numpy(
                    np.stack(clean_aug)[:, None]
                ).float().to(dev)

                # Peak normalisation: map each image to [0, 1].
                peak = x.amax(dim=(2, 3), keepdim=True).clamp(min=1e-8)
                x    = x / peak

                optimizer.zero_grad()

                if self.beta == 0.0:
                    # Deterministic AE: use μ directly, no reparameterisation noise.
                    mu, logsd = vae._net.encode(x)
                    x_hat = vae._net.decode(mu)
                    kl_loss = torch.tensor(0.0, device=dev)
                else:
                    x_hat, mu, logsd = vae._net(x)
                    kl_loss = -0.5 * (
                        1.0 + 2.0 * logsd - mu.pow(2) - (2.0 * logsd).exp()
                    ).mean()

                mse_loss = F_.mse_loss(x_hat, x)
                loss = mse_loss + self.beta * kl_loss
                loss.backward()
                nn.utils.clip_grad_norm_(vae._net.parameters(), max_norm=1.0)
                optimizer.step()

                epoch_mse += float(mse_loss)
                epoch_kl  += float(kl_loss)
                n_batches += 1

            print(f"  Epoch {epoch + 1:3d}/{self.n_epochs}  "
                  f"mse={epoch_mse / n_batches:.3e}  "
                  f"kl={epoch_kl / n_batches:.3e}", flush=True)

        vae._net.eval()
        return vae

    def __repr__(self) -> str:
        return (f"VAETrainer(n_epochs={self.n_epochs}, batch_size={self.batch_size}, "
                f"lr={self.lr}, beta={self.beta}, latent_dim={self.latent_dim})")
