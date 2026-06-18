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
    "measure_data_scale",
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


def _state_to_sky(
    x: torch.Tensor, mu: float, tau: float, space: str
) -> tuple[torch.Tensor, torch.Tensor]:
    """Change of variables from the standardised state ``x`` to linear sky ``s``,
    returning ``(s, ds/dx)`` for the data-term Jacobian.

    - ``"log"``    : ``s = exp(mu + tau·x)`` (RESOLVE positivity, flux rearranges
                     in log-space), ``ds/dx = tau·s``.  The ``exp`` is the source
                     of the log-sky densification — a sparse sky has no zero in
                     log-space, every pixel carries a finite floor.
    - ``"linear"`` : ``s = relu(mu + tau·x)`` (positivity by clamp, empty sky is
                     genuinely zero), ``ds/dx = tau`` where ``s > 0`` else 0.  This
                     keeps a sparse sky sparse so the prior loss is not dominated
                     by a dense floor (the overfit_field_inloop gate isolated the
                     log-sky floor as the source-destroying mechanism).
    """
    raw = mu + tau * x
    if space == "log":
        s = torch.exp(torch.clamp(raw, max=_EXP_CLAMP_MAX))
        jac = tau * s
    elif space == "linear":
        s = torch.clamp(raw, min=0.0)
        jac = tau * (raw > 0).to(s.dtype)
        # Constant Jacobian (no s factor): the data gradient on a bright pixel
        # does not vanish the way tau·s does for a faint one — sources are not
        # down-weighted by their own (small, in log-space) value.
    else:
        raise ValueError(f"space must be 'log' or 'linear'; got {space!r}")
    return s, jac


def make_field_posterior_score(
    forward_op: ImageDomainForward,
    score_model: EDMDenoiser,
    d: torch.Tensor,
    mu: float,
    tau: float,
    noise_std: float,
    data_scale: float = 1.0,
    space: str = "log",
) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
    """Build the posterior-score closure for one observation ``d``.

    Input/output states are the standardised field ``x`` of shape ``(B, H, W)``
    in the space named by ``space`` (``"log"`` ⇒ standardised log-sky, the
    original Fork-A formulation; ``"linear"`` ⇒ standardised linear sky).  The
    prior ``score_model`` must have been trained in the SAME space — the model is
    space-agnostic, the standardisation ``(mu, tau)`` and this closure carry the
    space.

    ``data_scale`` multiplies the data term before it is added to the prior
    score.  The two terms live on different scales — the prior score is O(1) per
    pixel while the data term carries the ``1/noise_std²`` likelihood weight — so
    combining them at unit weight lets the data term overshoot the annealing step
    and rail.  ``data_scale`` is the single measured scalar that puts the data
    term on the prior's scale, mirroring the flow encoder's per-sample residual
    normalisation.  ``1.0`` reproduces the raw additive score.
    """

    def posterior_score(x: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        # Prior score in the standardised space (add/remove channel dim).
        prior = score_model.score(x.unsqueeze(1), sigma).squeeze(1)
        # Data score via the change of variables s(x); jac = ds/dx.
        s, jac = _state_to_sky(x, mu, tau, space)
        r = d - forward_op.forward(s)
        lik = forward_op.adjoint(r) / (noise_std**2)  # A^T N^{-1} r
        data = jac * lik
        return prior + data_scale * data

    return posterior_score


def measure_data_scale(
    forward_op: ImageDomainForward,
    score_model: EDMDenoiser,
    d: torch.Tensor,
    mu: float,
    tau: float,
    noise_std: float,
    sigma_max: float = 5.0,
    space: str = "log",
) -> float:
    """λ = |prior score| / |data score| at cold init (state = 0) and σ = σ_max.

    The single measured scalar that brings the likelihood term onto the prior
    score's scale — the flow encoder's "measure the residual scale and divide"
    move, here matched at the top annealing level where the raw run overshoots.
    Space-aware via :func:`_state_to_sky`, so it works for both 'log' and
    'linear' priors.  Returns a positive float; multiply by a small factor if a
    stronger/weaker data term is wanted (the in-loop gate's coupling knob).
    """
    H, W = forward_op.shape
    f0 = torch.zeros(1, H, W, device=d.device, dtype=d.dtype)
    sig = torch.tensor(float(sigma_max), device=d.device, dtype=d.dtype)
    prior = score_model.score(f0.unsqueeze(1), sig).squeeze(1)
    s, jac = _state_to_sky(f0, mu, tau, space)
    data = jac * (forward_op.adjoint(d - forward_op.forward(s)) / noise_std**2)
    return float(prior.norm() / (data.norm() + 1e-30))


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
    data_scale: float | str = 1.0,
    space: str = "log",
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Draw ``n_samples`` field posterior samples (in linear sky space ``s``).

    The sampler runs in the standardised space named by ``space`` (``"log"`` ⇒
    standardised log-sky, ``"linear"`` ⇒ standardised linear sky); the returned
    samples are mapped back to linear sky ``s`` and are non-negative by
    construction (``exp`` for log, ``relu`` for linear).  ``step_size`` is the
    Song-Ermon ε and must stay small (the per-level step grows as
    ``(σ_t/σ_min)²``).  ``f_clip`` bounds the standardised state each step (a
    numerical guard, not a physical prior).

    Returns
    -------
    s : (n_samples, H, W) tensor of posterior sky samples.
    """
    H, W = forward_op.shape
    device = d.device
    sigmas = geometric_sigma_schedule(sigma_max, sigma_min, n_levels, device=device)

    if isinstance(data_scale, str):
        if data_scale != "auto":
            raise ValueError(f"data_scale string must be 'auto'; got {data_scale!r}")
        data_scale = measure_data_scale(
            forward_op, score_model, d, mu, tau, noise_std, sigma_max, space)

    init = init_std * torch.randn(
        n_samples, H, W, generator=generator, device=device, dtype=d.dtype
    )
    pscore = make_field_posterior_score(
        forward_op, score_model, d, mu, tau, noise_std,
        data_scale=data_scale, space=space,
    )
    lo, hi = f_clip
    f_std = annealed_langevin(
        pscore, init, sigmas,
        n_steps_per_level=n_steps_per_level,
        step_size=step_size,
        generator=generator,
        project=lambda x: x.clamp(lo, hi),
    )
    s, _ = _state_to_sky(f_std, mu, tau, space)
    return s
