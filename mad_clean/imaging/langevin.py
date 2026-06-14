"""
mad_clean.imaging.langevin
==========================
Annealed unadjusted Langevin sampler for the field posterior (Fork A, design
step 3, §2.2–2.6).  One full annealing sweep returns one posterior sample; an
ensemble of sweeps gives per-pixel and morphological credible intervals.

Three spaces are in play and the change of variables between them is the crux:

    s   = linear sky (Jy/pixel), s > 0           — where the likelihood lives
    f   = log s                                  — log-sky (positivity, flux moves)
    f'  = (f - mu) / tau                          — standardised log-sky
                                                   — where the score model lives

The sampler runs in standardised log-sky ``f'`` (the prior score's native
space).  The posterior score there is (design §2.1, §2.6 + the standardisation
chain rule):

    ∇_{f'} log p(f' | d) = s_θ(f', σ_t)                         (prior score)
                         + tau · s ⊙ (A^T N^{-1} r),            (data score)
        with  s = exp(mu + tau·f'),  r = d - A s.

The ``tau · s`` factor is the Jacobian of ``s = exp(mu + tau·f')``; it is the
RESOLVE positivity trick (∂s/∂f = s) with the extra ``tau`` from standardisation.

Annealed update (Euler–Maruyama, §2.3), levels σ_1 > … > σ_T, K steps each:

    f' ← f' + η_t · score + sqrt(2 η_t) · ξ,   η_t = η_0 (σ_t / σ_T)²,  ξ ~ N(0,I).

Honesty flag (design §2.3 / §2.7): the data score is the *clean-data* gradient,
added at full strength at every annealing level.  This is exact only in the
slow-annealing limit; the principled fix is the DPS Tweedie likelihood (§2.4).
It is the main calibration risk on real fields, alongside prior coverage.
"""

from __future__ import annotations

from typing import Callable

import torch

from mad_clean.imaging.forward import ImageDomainForward
from mad_clean.imaging.score import EDMDenoiser

__all__ = [
    "annealed_langevin",
    "geometric_sigma_schedule",
    "make_field_posterior_score",
    "sample_field_posterior",
]

# Cap on the log-sky argument to exp() (float64 exp overflows near 709); this is
# a numerical guard, well above any physical standardised log-sky value.
_EXP_CLAMP_MAX: float = 30.0


def geometric_sigma_schedule(
    sigma_max: float, sigma_min: float, n_levels: int, device=None
) -> torch.Tensor:
    """Geometrically spaced noise levels, descending σ_max → σ_min, shape (T,)."""
    return torch.logspace(
        torch.log10(torch.tensor(float(sigma_max))),
        torch.log10(torch.tensor(float(sigma_min))),
        n_levels,
        device=device,
    )


def annealed_langevin(
    posterior_score: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    init: torch.Tensor,
    sigmas: torch.Tensor,
    n_steps_per_level: int = 10,
    step_size: float = 1e-5,
    generator: torch.Generator | None = None,
    project: Callable[[torch.Tensor], torch.Tensor] | None = None,
    return_trajectory: bool = False,
):
    """Generic annealed ULA engine (Song & Ermon convention).

    The per-level step is ``η_t = η_0 (σ_t / σ_min)²``.  This σ² growth is
    deliberate: a level-matched prior score scales as ~1/σ², so the product
    ``η_t · score`` stays roughly uniform across levels — BUT only when ``η_0``
    is the small Song-Ermon ε (~1e-5).  A large ``η_0`` makes the top-level step
    diverge; in log-space that overflows the ``exp`` in the data term.  Hence the
    small default and the optional ``project`` guard.

    Parameters
    ----------
    posterior_score : callable(x, σ) → tensor
        The full posterior score ∇_x log p(x | d) at noise level σ (a 0-d tensor).
    init   : initial state, any shape; the leading dim is the sample/batch dim.
    sigmas : (T,) descending noise levels.
    n_steps_per_level : K Langevin steps at each level.
    step_size : η_0 (Song-Ermon ε); per-level step η_t = η_0 (σ_t / σ_min)².
    project : optional callable applied to the state after each step (e.g. a
        clamp), used by the log-sky wrapper to keep ``exp`` finite.
    return_trajectory : if True, also return the list of per-level states.

    Returns
    -------
    x : final state (one posterior sample per leading-dim entry).
    trajectory : (only if return_trajectory) list of states, one per level.
    """
    x = init.clone()
    sigma_min = sigmas[-1]
    traj = []
    for sigma in sigmas:
        eta = step_size * float((sigma / sigma_min) ** 2)
        sqrt_2eta = (2.0 * eta) ** 0.5
        for _ in range(n_steps_per_level):
            score = posterior_score(x, sigma)
            noise = torch.randn(
                x.shape, generator=generator, device=x.device, dtype=x.dtype
            )
            x = x + eta * score + sqrt_2eta * noise
            if project is not None:
                x = project(x)
        if return_trajectory:
            traj.append(x.clone())
    if return_trajectory:
        return x, traj
    return x


def make_field_posterior_score(
    forward_op: ImageDomainForward,
    score_model: EDMDenoiser,
    d: torch.Tensor,
    mu: float,
    tau: float,
    noise_std: float,
) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
    """Build the log-sky posterior-score closure for one observation ``d``.

    Input/output states are standardised log-sky ``f'`` of shape ``(B, H, W)``.
    """

    def posterior_score(f_std: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        # Prior score in standardised log-sky space (add/remove channel dim).
        prior = score_model.score(f_std.unsqueeze(1), sigma).squeeze(1)
        # Data score via the change of variables s = exp(mu + tau·f').  The clamp
        # is a defensive guard against transient excursions overflowing exp; the
        # sampler's ``project`` keeps f' in the same band, so it rarely binds.
        s = torch.exp(torch.clamp(mu + tau * f_std, max=_EXP_CLAMP_MAX))
        r = d - forward_op.forward(s)
        lik = forward_op.adjoint(r) / (noise_std**2)  # A^T N^{-1} r
        data = tau * s * lik
        return prior + data

    return posterior_score


def sample_field_posterior(
    forward_op: ImageDomainForward,
    score_model: EDMDenoiser,
    d: torch.Tensor,
    mu: float,
    tau: float,
    noise_std: float,
    n_samples: int = 1,
    sigma_max: float = 5.0,
    sigma_min: float = 1e-2,
    n_levels: int = 20,
    n_steps_per_level: int = 20,
    step_size: float = 1e-5,
    init_std: float = 1.0,
    f_clip: tuple[float, float] = (-20.0, 20.0),
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Draw ``n_samples`` field posterior samples (in linear sky space ``s``).

    The sampler runs in standardised log-sky; the returned samples are
    ``s = exp(mu + tau·f')`` — strictly positive by construction.  ``step_size``
    is the Song-Ermon ε and must stay small (the per-level step grows as
    ``(σ_t/σ_min)²``).  ``f_clip`` bounds the standardised state each step to keep
    ``exp`` finite (a numerical guard, not a physical prior).

    Returns
    -------
    s : (n_samples, H, W) tensor of posterior sky samples.
    """
    H, W = forward_op.shape
    device = d.device
    sigmas = geometric_sigma_schedule(sigma_max, sigma_min, n_levels, device=device)

    init = init_std * torch.randn(
        n_samples, H, W, generator=generator, device=device, dtype=d.dtype
    )
    pscore = make_field_posterior_score(
        forward_op, score_model, d, mu, tau, noise_std
    )
    lo, hi = f_clip
    f_std = annealed_langevin(
        pscore, init, sigmas,
        n_steps_per_level=n_steps_per_level,
        step_size=step_size,
        generator=generator,
        project=lambda x: x.clamp(lo, hi),
    )
    return torch.exp(mu + tau * f_std)
