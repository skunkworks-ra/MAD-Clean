"""
mad_clean.imaging.forward
=========================
Image-domain interferometric forward operator for the field-posterior loop
(Fork A).  Provides the measurement operator ``A`` (PSF / dirty-beam
convolution), its true adjoint ``A^T``, and the noise-weighted residual that
the design calls the negative log-likelihood gradient.

The single load-bearing identity (field_posterior_design.md §1.2):

    ∇_s L(s) = -A^T N^{-1} (d - A s) = -A^T N^{-1} r,

so the term consumed by the sampler is ``A^T N^{-1} r`` — supplied here by
:meth:`ImageDomainForward.likelihood_score`.

Conventions
-----------
- The PSF is the **dirty beam**, peak-centred and **peak-normalised** (peak = 1),
  exactly as :class:`mad_clean.data.psf_bank.PSFBank` produces it.  Peak (not
  sum) normalisation is required: a real interferometric dirty beam integrates
  to ≈ 0 (missing DC / short spacings), so sum-normalisation would divide by ≈ 0.
  Convolving a strictly-positive sky with this zero-DC beam reproduces both the
  sidelobe ringing and the negative bowl — the instrumental negatives of §4.3.
- Circular convolution via FFT, ``ifftshift`` on the PSF so its peak maps to the
  array origin (matching ``simulate_observations._convolve_psf`` and MADClean).
- Image-domain white noise ``N = σ² I`` is the §2.7-A0 approximation (true
  image noise is PSF-correlated; the vis-domain operator is the faithful path
  and is deferred).  This is flagged, not hidden.
- All operators act on the trailing two dims, so inputs may be ``(H, W)`` or
  batched ``(..., H, W)``.
"""

from __future__ import annotations

import torch

__all__ = ["ImageDomainForward"]


class ImageDomainForward:
    """Image-domain forward operator ``A`` and adjoint ``A^T`` for one PSF.

    Parameters
    ----------
    psf : (H, W) real tensor
        Dirty beam, peak-centred, peak-normalised.  Stored as its real FFT.
    """

    def __init__(self, psf: torch.Tensor):
        if psf.ndim != 2:
            raise ValueError(f"psf must be 2D (H, W); got shape {tuple(psf.shape)}")
        self.shape = (int(psf.shape[0]), int(psf.shape[1]))
        self.device = psf.device
        self.dtype = psf.dtype
        # H(k) for circular convolution: peak shifted to the origin.
        psf_shift = torch.fft.ifftshift(psf)
        self._H = torch.fft.rfft2(psf_shift, s=self.shape)

    # ── core operators ──────────────────────────────────────────────────────

    def forward(self, s: torch.Tensor) -> torch.Tensor:
        """Apply ``A``: convolve sky ``s`` with the dirty beam.  ``A s = h * s``."""
        Sk = torch.fft.rfft2(s, s=self.shape)
        return torch.fft.irfft2(Sk * self._H, s=self.shape)

    def adjoint(self, y: torch.Tensor) -> torch.Tensor:
        """Apply ``A^T``: correlation with the dirty beam (multiply by conj(H)).

        This is the exact adjoint of :meth:`forward` under the Euclidean inner
        product, so ``⟨A x, y⟩ = ⟨x, A^T y⟩`` to machine precision."""
        Yk = torch.fft.rfft2(y, s=self.shape)
        return torch.fft.irfft2(Yk * self._H.conj(), s=self.shape)

    # ── data products ───────────────────────────────────────────────────────

    def residual(self, s: torch.Tensor, d: torch.Tensor) -> torch.Tensor:
        """The residual ``r = d - A s``."""
        return d - self.forward(s)

    def likelihood_score(
        self, s: torch.Tensor, d: torch.Tensor, noise_std: float
    ) -> torch.Tensor:
        """The data term of the posterior score: ``A^T N^{-1} r = A^T (d - A s) / σ²``.

        This equals ``-∇_s L`` (the major cycle's gridded residual), the quantity
        the Langevin update adds to the prior score."""
        r = self.residual(s, d)
        return self.adjoint(r) / (noise_std**2)

    def make_dirty(
        self,
        s: torch.Tensor,
        noise_std: float = 0.0,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        """Forward-model a dirty image ``d = A s + n``, ``n ~ N(0, σ² I)``.

        Set ``noise_std = 0`` for the noiseless dirty image.  The instrumental
        negatives appear here, from the zero-DC beam acting on positive sky."""
        d = self.forward(s)
        if noise_std > 0.0:
            noise = torch.randn(
                d.shape, generator=generator, device=d.device, dtype=d.dtype
            )
            d = d + noise_std * noise
        return d
