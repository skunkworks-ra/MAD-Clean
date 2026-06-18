"""Starlet (isotropic undecimated wavelet) transform and codec.

The inference target for the wavelet-NPE head is theta = the
per-scale-normalised starlet coefficients of the true sky cutout,
kept at FULL RESOLUTION (no decimation).  This module provides:

- ``starlet_transform`` / ``starlet_reconstruct``: the standard a-trous
  B3-spline starlet.  Reconstruction is exact (sum of detail planes plus
  the smooth plane), which is pinned by a unit test.
- ``StarletCodec``: encode an image to a flat theta vector (optionally
  drop sub-beam planes, asinh-compress, standardise) and decode back.
  All kept planes are full-resolution (H×W); the codec is lossless up
  to the asinh round-trip on calibrated coefficients.

Scale conventions
-----------------
Detail plane ``w_j`` (j = 1..J) carries structure at roughly ``2**(j-1)``
to ``2**j`` px.  The beam FWHM is ~2.8 px (BEAM_SIGMA_PX = 1.4), so plane
w_1 (~1-2 px) is sub-beam.  ``drop_scales=(1,)`` omits it; ``drop_scales=()``
keeps it (Option A — required for point-source peak localisation).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F

__all__ = ["starlet_transform", "starlet_reconstruct", "StarletCodec"]

# B3-spline scaling kernel (1D); the 2D kernel is the separable product.
_B3 = torch.tensor([1.0, 4.0, 6.0, 4.0, 1.0]) / 16.0


def _smooth(c: torch.Tensor, step: int) -> torch.Tensor:
    """One a-trous smoothing pass with the B3 kernel dilated by ``step``.

    c : (B, 1, H, W) -> (B, 1, H, W), reflect-padded so support is preserved.
    """
    k1 = _B3.to(c.dtype).to(c.device)
    pad = 2 * step
    # Separable: rows then columns.
    kh = k1.view(1, 1, 1, 5)
    kv = k1.view(1, 1, 5, 1)
    c = F.pad(c, (pad, pad, 0, 0), mode="reflect")
    c = F.conv2d(c, kh, dilation=(1, step))
    c = F.pad(c, (0, 0, pad, pad), mode="reflect")
    c = F.conv2d(c, kv, dilation=(step, 1))
    return c


def starlet_transform(img: torch.Tensor, n_scales: int) -> list[torch.Tensor]:
    """A-trous starlet transform.

    Parameters
    ----------
    img : (B, H, W) or (H, W) float tensor.
    n_scales : number of detail planes J.

    Returns
    -------
    [w_1, ..., w_J, c_J] — J detail planes plus the smooth plane, each the
    same shape as ``img``.  ``sum(planes) == img`` exactly (float precision).
    """
    squeeze = img.dim() == 2
    if squeeze:
        img = img.unsqueeze(0)
    c = img.unsqueeze(1)  # (B, 1, H, W)
    planes: list[torch.Tensor] = []
    for j in range(n_scales):
        c_next = _smooth(c, step=2 ** j)
        planes.append((c - c_next).squeeze(1))
        c = c_next
    planes.append(c.squeeze(1))
    if squeeze:
        planes = [p.squeeze(0) for p in planes]
    return planes


def starlet_reconstruct(planes: list[torch.Tensor]) -> torch.Tensor:
    """Inverse of :func:`starlet_transform`: plain sum of all planes."""
    out = planes[0]
    for p in planes[1:]:
        out = out + p
    return out


@dataclass
class StarletCodec:
    """Image <-> flat theta vector via decimated, normalised starlet planes.

    Parameters
    ----------
    image_size : side length of the (square) input image.
    n_scales : number of detail planes J. Default 6 for 128 px cutouts
        (finest kept structure ~2 px, coarsest ~64 px, plus smooth plane).
    drop_scales : indices (1-based, matching w_j) of detail planes to drop.
        Default (1,) — the sub-beam plane.
    asinh_softening : per-plane softening b_j for asinh compression is
        ``asinh_softening * mad_j`` where mad_j comes from ``calibrate``.
    z_clamp : decode-side clamp on standardised coefficients, in units of
        the calibration std.  |z| beyond this is outside the calibrated
        range and would be exponentially amplified by sinh; clamping
        bounds the damage from posterior tail samples.  Default 6.0.
    """

    image_size: int = 128
    n_scales: int = 6
    drop_scales: tuple[int, ...] = (1,)
    asinh_softening: float = 1.0
    z_clamp: float = 6.0

    # Per-kept-plane calibration: softening b_j and post-asinh std s_j.
    _b: list[float] = field(default_factory=list)
    _s: list[float] = field(default_factory=list)

    def __post_init__(self) -> None:
        pass

    # ------------------------------------------------------------------
    # Layout
    # ------------------------------------------------------------------

    def kept_planes(self) -> list[int]:
        """Plane indices kept in theta: detail j in 1..J minus drops, then
        J+1 for the smooth plane."""
        kept = [j for j in range(1, self.n_scales + 1)
                if j not in self.drop_scales]
        kept.append(self.n_scales + 1)  # smooth plane sentinel
        return kept

    def pool_factor(self, j: int) -> int:
        """No decimation — all planes kept at full resolution."""
        return 1

    def plane_dims(self) -> list[tuple[int, int]]:
        """(side, n_elements) per kept plane, in theta order."""
        n = self.image_size * self.image_size
        return [(self.image_size, n) for _ in self.kept_planes()]

    @property
    def theta_dim(self) -> int:
        return sum(n for _, n in self.plane_dims())

    # ------------------------------------------------------------------
    # Calibration
    # ------------------------------------------------------------------

    def calibrate(self, images: torch.Tensor) -> None:
        """Estimate per-plane (b_j, s_j) from a batch of representative
        true-sky cutouts (B, H, W).  Must be called before encode/decode;
        the values are part of the model contract and must be checkpointed.
        """
        raw = self._raw_planes(images)  # list of (B, h, h)
        self._b, self._s = [], []
        for p in raw:
            flat = p.reshape(p.shape[0], -1)
            # Statistics over ACTIVE coefficients only.  Fine planes are
            # almost all zeros (empty sky); a MAD over the full plane is
            # dominated by them, making b tiny, z huge, and the decode
            # z-clamp then destroys compact-source flux (pilot v2,
            # 2026-06-11: point round-trip rel L2 ~0.99).
            peak = flat.abs().max(dim=1).values.clamp_min(1e-12)
            active = flat[flat.abs() > 1e-3 * peak.unsqueeze(1)]
            if active.numel() < 16:
                active = flat.reshape(-1)
            mad = 1.4826 * active.abs().median().item()
            b = self.asinh_softening * max(mad, 1e-8)
            z = torch.asinh(active / b)
            s = max(z.std().item(), 1e-8)
            self._b.append(b)
            self._s.append(s)

    @property
    def is_calibrated(self) -> bool:
        return len(self._b) > 0

    def state_dict(self) -> dict:
        return {
            "image_size": self.image_size,
            "n_scales": self.n_scales,
            "drop_scales": list(self.drop_scales),
            "asinh_softening": self.asinh_softening,
            "z_clamp": self.z_clamp,
            "b": list(self._b),
            "s": list(self._s),
        }

    @classmethod
    def from_state_dict(cls, d: dict) -> "StarletCodec":
        codec = cls(
            image_size=d["image_size"],
            n_scales=d["n_scales"],
            drop_scales=tuple(d["drop_scales"]),
            asinh_softening=d["asinh_softening"],
            z_clamp=d.get("z_clamp", 6.0),
        )
        codec._b = list(d["b"])
        codec._s = list(d["s"])
        return codec

    # ------------------------------------------------------------------
    # True-sky support weights (training-loss shaping only)
    # ------------------------------------------------------------------

    def support_weights(
        self,
        images: torch.Tensor,
        outside_weight: float = 0.05,
        thresh_frac: float = 1e-3,
    ) -> torch.Tensor:
        """Per-dimension loss weights from the TRUE sky support.

        Oracle information is safe in the loss but poison in the input:
        these weights shape the training gradient only and are never fed
        to the network, so nothing changes at inference.  A coefficient
        at scale j is "inside" if any sky pixel above ``thresh_frac`` of
        the image peak lies within the plane-j filter footprint (mask
        dilated by radius 2**j, then max-pooled onto the decimated grid).
        Inside dims get weight 1, outside dims ``outside_weight`` (small
        but nonzero, so the flow still learns that empty sky is quiet).

        images : (B, H, W) true skies -> (B, theta_dim) weights.
        """
        B = images.shape[0]
        peak = images.abs().flatten(1).max(dim=1).values.clamp_min(1e-12)
        mask = (images.abs() > thresh_frac * peak.view(-1, 1, 1)).float()
        mask = mask.unsqueeze(1)  # (B, 1, H, W)
        parts = []
        for j in self.kept_planes():
            scale = min(j, self.n_scales)
            r = 2 ** scale
            m = F.max_pool2d(mask, kernel_size=2 * r + 1, stride=1, padding=r)
            f = self.pool_factor(j)
            if f > 1:
                m = F.max_pool2d(m, f)
            parts.append(m.flatten(1))
        w = torch.cat(parts, dim=1)  # (B, theta_dim), values in {0, 1}
        return outside_weight + (1.0 - outside_weight) * w

    # ------------------------------------------------------------------
    # Encode / decode
    # ------------------------------------------------------------------

    def _raw_planes(self, images: torch.Tensor) -> list[torch.Tensor]:
        """Starlet planes at full resolution, no normalisation. images: (B, H, W)."""
        planes = starlet_transform(images, self.n_scales)
        return [
            planes[j - 1] if j <= self.n_scales else planes[-1]
            for j in self.kept_planes()
        ]

    def encode(self, images: torch.Tensor) -> torch.Tensor:
        """(B, H, W) true-sky images -> (B, theta_dim) theta vectors."""
        if not self.is_calibrated:
            raise RuntimeError("StarletCodec.calibrate() must run first")
        raw = self._raw_planes(images)
        zs = []
        for p, b, s in zip(raw, self._b, self._s):
            z = torch.asinh(p / b) / s
            zs.append(z.flatten(1))
        return torch.cat(zs, dim=1)

    def decode(self, theta: torch.Tensor) -> torch.Tensor:
        """(B, theta_dim) -> (B, H, W) images.

        All planes are full-resolution so no upsampling is needed.
        Dropped planes contribute zero.
        """
        if not self.is_calibrated:
            raise RuntimeError("StarletCodec.calibrate() must run first")
        B = theta.shape[0]
        out = torch.zeros(
            B, self.image_size, self.image_size,
            dtype=theta.dtype, device=theta.device,
        )
        i = 0
        for (side, n), b, s in zip(self.plane_dims(), self._b, self._s):
            z = theta[:, i : i + n].reshape(B, self.image_size, self.image_size)
            i += n
            z = z.clamp(-self.z_clamp, self.z_clamp)
            out = out + torch.sinh(z * s) * b
        return out
