"""Starlet (isotropic undecimated wavelet) transform and the decimated codec.

The inference target for the wavelet-NPE head is theta = the decimated,
per-scale-normalised starlet coefficients of the true sky cutout.  This
module provides:

- ``starlet_transform`` / ``starlet_reconstruct``: the standard a-trous
  B3-spline starlet.  Reconstruction is exact (sum of detail planes plus
  the smooth plane), which is pinned by a unit test.
- ``StarletCodec``: encode an image to a flat theta vector (drop sub-beam
  planes, decimate each kept plane by a per-scale factor, asinh-compress,
  standardise) and decode back (approximately — decimation is lossy; the
  reconstruction error at typical source scales is pinned by tests).

Scale conventions
-----------------
Detail plane ``w_j`` (j = 1..J) carries structure at roughly ``2**(j-1)``
to ``2**j`` px.  The beam FWHM is ~2.8 px (BEAM_SIGMA_PX = 1.4), so plane
w_1 (~1-2 px) is sub-beam and is dropped by default per the project's
physical constraint: sub-beam structure is PSF artefact, not morphology.

Default decimation pools plane w_j by ``2**(j-1)`` (half its
characteristic scale, i.e. roughly critical sampling) and the smooth
plane by ``2**(J-1)``.
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
        for j in self.kept_planes():
            f = self.pool_factor(j)
            if self.image_size % f:
                raise ValueError(
                    f"image_size={self.image_size} not divisible by pool "
                    f"factor {f} of plane {j}"
                )

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
        """Decimation factor for plane j (smooth plane uses J-1)."""
        if j == self.n_scales + 1:
            return 2 ** (self.n_scales - 1)
        return max(1, 2 ** (j - 1))

    def plane_dims(self) -> list[tuple[int, int]]:
        """(side, n_elements) per kept plane, in theta order."""
        out = []
        for j in self.kept_planes():
            side = self.image_size // self.pool_factor(j)
            out.append((side, side * side))
        return out

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
            flat = p.reshape(-1)
            mad = 1.4826 * flat.abs().median().item()
            b = self.asinh_softening * max(mad, 1e-8)
            z = torch.asinh(flat / b)
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
    # Encode / decode
    # ------------------------------------------------------------------

    def _raw_planes(self, images: torch.Tensor) -> list[torch.Tensor]:
        """Starlet + decimation, no normalisation. images: (B, H, W)."""
        planes = starlet_transform(images, self.n_scales)
        out = []
        for j in self.kept_planes():
            p = planes[j - 1] if j <= self.n_scales else planes[-1]
            f = self.pool_factor(j)
            if f > 1:
                p = F.avg_pool2d(p.unsqueeze(1), f).squeeze(1)
            out.append(p)
        return out

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
        """(B, theta_dim) -> (B, H, W) approximate images.

        Upsamples each plane bilinearly back to image_size and sums.
        Dropped planes contribute zero.  Lossy by construction; the loss
        is quantified in tests/test_starlet.py.
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
            z = theta[:, i : i + n].reshape(B, 1, side, side)
            i += n
            z = z.clamp(-self.z_clamp, self.z_clamp)
            p = torch.sinh(z * s) * b
            if side != self.image_size:
                p = F.interpolate(
                    p, size=(self.image_size, self.image_size),
                    mode="bilinear", align_corners=False,
                )
            out = out + p.squeeze(1)
        return out
