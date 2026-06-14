"""
mad_clean.imaging.mgvi
=====================
Vanilla Metric Gaussian Variational Inference (MGVI) for the field posterior
(Fork B, design §3.2–3.3) — the RESOLVE-style variational reference.

This is the *correctness reference* of the §5 build order, not the fast path: it
stands up MGVI in standardized coordinates with the matrix-free Fisher metric and
conjugate-gradient solves, before any of the §3.5 speed ladder (S1–S6).  Fork A's
annealed sampler is the calibration oracle this is checked against.

Standardized coordinates (§3.2)
-------------------------------
Write the prior generatively, ``s = G(ξ)`` with ``ξ ~ N(0, I)``.  For a
log-normal field with power-spectrum covariance ``C`` and log-mean ``m``,

    s = exp(C^{1/2} ξ + m),     p(ξ) = N(0, I),

so the prior term collapses to ``½‖ξ‖²`` and all structure is carried by ``G``.
``C^{1/2}`` is a power-law Fourier multiplier (the same amplitude that synthesises
the diffuse field in ``data/field_sky.py``), normalised so the marginal log-sky
variance equals ``sigma_log²``.  ``C^{1/2}`` is real and self-adjoint.

The posterior over ξ (§3.2):

    -log p(ξ|d) = ½ (d - A G(ξ))ᵀ N⁻¹ (d - A G(ξ)) + ½‖ξ‖² + const.

MGVI (§3.3) approximates it by ``q = N(ξ̄, M⁻¹)`` with the Gauss–Newton/Fisher
metric, evaluated at the current mean and applied matrix-free:

    M(ξ̄) = I + Jᵀ Aᵀ N⁻¹ A J,    J = ∂s/∂ξ|_{ξ̄}.

Two matrix-free uses: (a) draw posterior samples without inverting M, by drawing
from the curvature then CG-solving; (b) a natural-gradient mean update, again via
CG.  Alternate to convergence.

Honesty (§3.6): reverse-KL / Gaussian-q is mode-seeking and underestimates
variance; deconvolution posteriors are generically multimodal.  This reference is
the *fast-but-overconfident* fork by construction; Fork A is the honest baseline.
"""

from __future__ import annotations

from typing import Callable

import torch

__all__ = [
    "LogNormalFieldPrior",
    "conjugate_gradient",
    "fisher_matvec",
    "draw_curvature_sample",
    "mean_update",
    "mgvi_inference",
]


# ---------------------------------------------------------------------------
# Generative prior  s = G(ξ) = link(C^{1/2} ξ + m)
# ---------------------------------------------------------------------------

class LogNormalFieldPrior:
    """Standardized-coordinate generative prior with a power-law covariance.

    Parameters
    ----------
    shape     : (H, W).
    alpha     : power-law index of the covariance power spectrum P(k) ∝ k^{-alpha}.
    sigma_log : marginal std of the log-sky field C^{1/2} ξ.
    mean_log  : log-sky mean m (scalar).
    link      : "exp" (log-normal, production) or "identity" (linear-Gaussian,
                for the closed-form correctness test).
    """

    def __init__(
        self,
        shape: tuple[int, int],
        alpha: float = 2.7,
        sigma_log: float = 1.0,
        mean_log: float = 0.0,
        link: str = "exp",
        device=None,
        dtype: torch.dtype = torch.float64,
    ):
        if link not in ("exp", "identity"):
            raise ValueError(f"link must be 'exp' or 'identity'; got {link}")
        self.shape = (int(shape[0]), int(shape[1]))
        self.mean_log = float(mean_log)
        self.link = link
        self.dtype = dtype

        H, W = self.shape
        ky = torch.fft.fftfreq(H, dtype=dtype, device=device)
        kx = torch.fft.fftfreq(W, dtype=dtype, device=device)
        KX, KY = torch.meshgrid(kx, ky, indexing="xy")
        k = torch.sqrt(KX**2 + KY**2)
        amp = torch.where(k > 0, k ** (-alpha / 2.0), torch.zeros_like(k))
        # Normalise so the marginal variance of C^{1/2}ξ is sigma_log²: with
        # orthonormal FFTs and white ξ, per-pixel variance = mean(amp²).
        rms = torch.sqrt(torch.mean(amp**2))
        if float(rms) > 0:
            amp = amp * (sigma_log / rms)
        self.amp = amp  # (H, W) real Fourier multiplier

    # ── covariance square root (self-adjoint) ─────────────────────────────────

    def apply_C_half(self, v: torch.Tensor) -> torch.Tensor:
        """C^{1/2} v — circular convolution by the power-law kernel (self-adjoint)."""
        V = torch.fft.fft2(v.to(self.dtype), norm="ortho")
        return torch.fft.ifft2(V * self.amp, norm="ortho").real

    # ── generative map and its Jacobian ───────────────────────────────────────

    def _u(self, xi: torch.Tensor) -> torch.Tensor:
        return self.apply_C_half(xi) + self.mean_log

    def forward(self, xi: torch.Tensor) -> torch.Tensor:
        """s = G(ξ)."""
        u = self._u(xi)
        return torch.exp(u) if self.link == "exp" else u

    def _link_deriv(self, xi_bar: torch.Tensor) -> torch.Tensor:
        """link'(u) at the linearization point ξ̄."""
        if self.link == "exp":
            return self.forward(xi_bar)  # = s
        return torch.ones_like(xi_bar)

    def apply_J(self, xi_bar: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """J v = link'(u) ⊙ (C^{1/2} v), J = ∂s/∂ξ at ξ̄."""
        return self._link_deriv(xi_bar) * self.apply_C_half(v)

    def apply_Jt(self, xi_bar: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Jᵀ y = C^{1/2} (link'(u) ⊙ y)  (true adjoint of apply_J)."""
        return self.apply_C_half(self._link_deriv(xi_bar) * y)


# ---------------------------------------------------------------------------
# Conjugate gradient (matrix-free SPD solve)
# ---------------------------------------------------------------------------

def conjugate_gradient(
    matvec: Callable[[torch.Tensor], torch.Tensor],
    b: torch.Tensor,
    x0: torch.Tensor | None = None,
    max_iter: int = 100,
    tol: float = 1e-8,
) -> torch.Tensor:
    """Solve ``matvec(x) = b`` for SPD ``matvec`` by conjugate gradient.

    Inner products sum over all elements, so ``b`` may be any shape the matvec
    preserves."""
    x = torch.zeros_like(b) if x0 is None else x0.clone()
    r = b - matvec(x)
    p = r.clone()
    rs = float((r * r).sum())
    b_norm = float((b * b).sum()) ** 0.5 + 1e-30
    for _ in range(max_iter):
        Ap = matvec(p)
        denom = float((p * Ap).sum())
        if denom == 0.0:
            break
        alpha = rs / denom
        x = x + alpha * p
        r = r - alpha * Ap
        rs_new = float((r * r).sum())
        if rs_new**0.5 / b_norm < tol:
            break
        p = r + (rs_new / rs) * p
        rs = rs_new
    return x


# ---------------------------------------------------------------------------
# Fisher metric  M(ξ̄) = I + Jᵀ Aᵀ N⁻¹ A J
# ---------------------------------------------------------------------------

def fisher_matvec(
    prior: LogNormalFieldPrior,
    forward_op,
    xi_bar: torch.Tensor,
    noise_std: float,
) -> Callable[[torch.Tensor], torch.Tensor]:
    """Return the matrix-free metric operator ``v ↦ M(ξ̄) v``."""
    inv_n2 = 1.0 / (noise_std**2)

    def matvec(v: torch.Tensor) -> torch.Tensor:
        Jv = prior.apply_J(xi_bar, v)
        AtAJv = forward_op.adjoint(forward_op.forward(Jv))
        return v + inv_n2 * prior.apply_Jt(xi_bar, AtAJv)

    return matvec


# ---------------------------------------------------------------------------
# Posterior sample from the curvature (§3.3a)
# ---------------------------------------------------------------------------

def draw_curvature_sample(
    prior: LogNormalFieldPrior,
    forward_op,
    xi_bar: torch.Tensor,
    noise_std: float,
    generator: torch.Generator | None = None,
    cg_iters: int = 50,
) -> torch.Tensor:
    """Draw ξ_s ~ N(0, M⁻¹) without inverting M.

    First draw from the curvature, η = ξ₀ + Jᵀ Aᵀ N^{-1/2} ε with
    ξ₀, ε ~ N(0, I), so Cov(η) = I + Jᵀ Aᵀ N⁻¹ A J = M; then CG-solve M ξ_s = η,
    giving Cov(ξ_s) = M⁻¹."""
    shape = (prior.shape[0], prior.shape[1])
    dt = prior.dtype
    dev = xi_bar.device
    xi0 = torch.randn(shape, generator=generator, device=dev, dtype=dt)
    eps = torch.randn(shape, generator=generator, device=dev, dtype=dt)
    eta = xi0 + prior.apply_Jt(xi_bar, forward_op.adjoint(eps) / noise_std)
    matvec = fisher_matvec(prior, forward_op, xi_bar, noise_std)
    return conjugate_gradient(matvec, eta, max_iter=cg_iters)


# ---------------------------------------------------------------------------
# Natural-gradient mean update (§3.3b)
# ---------------------------------------------------------------------------

def _energy_gradient(prior, forward_op, d, xi_prime, noise_std):
    """∇_{ξ'} [½‖d - A G(ξ')‖²_{N⁻¹} + ½‖ξ'‖²] = ξ' - Jᵀ Aᵀ N⁻¹ (d - A G(ξ'))."""
    s = prior.forward(xi_prime)
    r = d - forward_op.forward(s)
    data = prior.apply_Jt(xi_prime, forward_op.adjoint(r) / (noise_std**2))
    return xi_prime - data


def mean_update(
    prior: LogNormalFieldPrior,
    forward_op,
    d: torch.Tensor,
    xi_bar: torch.Tensor,
    samples: list[torch.Tensor],
    noise_std: float,
    cg_iters: int = 50,
) -> torch.Tensor:
    """One natural-gradient step ξ̄ ← ξ̄ - M⁻¹ ∇H, ∇H sample-averaged over the
    antithetic pairs ξ̄ ± ξ_s."""
    grads = []
    for xi_s in samples:
        grads.append(_energy_gradient(prior, forward_op, d, xi_bar + xi_s, noise_std))
        grads.append(_energy_gradient(prior, forward_op, d, xi_bar - xi_s, noise_std))
    gbar = torch.stack(grads, dim=0).mean(dim=0)
    matvec = fisher_matvec(prior, forward_op, xi_bar, noise_std)
    delta = conjugate_gradient(matvec, gbar, max_iter=cg_iters)
    return xi_bar - delta


# ---------------------------------------------------------------------------
# MGVI inference loop
# ---------------------------------------------------------------------------

def mgvi_inference(
    prior: LogNormalFieldPrior,
    forward_op,
    d: torch.Tensor,
    noise_std: float,
    n_outer: int = 10,
    n_samples: int = 4,
    cg_iters: int = 50,
    xi_init: torch.Tensor | None = None,
    generator: torch.Generator | None = None,
):
    """Alternate (a) curvature sampling and (b) the natural-gradient mean update.

    Returns
    -------
    xi_bar  : latent posterior mean (H, W).
    samples : the last batch of antithetic curvature samples ξ_s.
    sky_samples : list of posterior sky samples G(ξ̄ ± ξ_s) (strictly positive
                  for the exp link).
    """
    if xi_init is None:
        xi_bar = torch.zeros(prior.shape, device=d.device, dtype=prior.dtype)
    else:
        xi_bar = xi_init.clone()

    samples: list[torch.Tensor] = []
    for _ in range(n_outer):
        samples = [
            draw_curvature_sample(prior, forward_op, xi_bar, noise_std,
                                  generator=generator, cg_iters=cg_iters)
            for _ in range(n_samples)
        ]
        xi_bar = mean_update(prior, forward_op, d, xi_bar, samples, noise_std,
                             cg_iters=cg_iters)

    sky_samples = []
    for xi_s in samples:
        sky_samples.append(prior.forward(xi_bar + xi_s))
        sky_samples.append(prior.forward(xi_bar - xi_s))
    return xi_bar, samples, sky_samples
