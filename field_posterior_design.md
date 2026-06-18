# Looped Field-Posterior Imaging — Design

**Date:** 2026-06-13
**Status:** design, pre-implementation
**Relation to other docs:** `bayesian_imaging.md` gives the literature map (RESOLVE,
IRIS, DDRM, AIRI). This doc specifies how to obtain a *pixel-level posterior*
inside the existing MAD-CLEAN explicit-PSF major/minor loop, in two forms:
Fork A (sampling, lowest effort) and Fork B (variational / RESOLVE, with an
explicit focus on making RESOLVE faster).

The thesis: we are not abandoning the loop. Pixel-level posterior deconvolution
*is* a loop. The residual the major cycle already returns is the likelihood
gradient, so the loop is the natural host. What changes is only what runs inside
the minor cycle and how the iteration is interpreted.

---

## 0. Notation and conventions

| Symbol | Meaning |
|---|---|
| $s \in \mathbb{R}^{N}$ | sky brightness on an island, $N$ pixels (Jy/pixel) |
| $f \in \mathbb{R}^{N}$ | log-sky field, $s = e^{f}$ elementwise |
| $A$ | measurement operator (image-domain PSF convolution, or gridding/NUFFT) |
| $A^{\top}$ | its adjoint (the major cycle applies both) |
| $d$ | data (dirty image, or visibilities) |
| $n$ | noise, $n \sim \mathcal{N}(0, N)$, $N$ the noise covariance |
| $r$ | residual, $r := d - A s$ |
| $D_\sigma$ | MMSE denoiser at noise level $\sigma$ (a diffusion model) |
| $s_\theta$ | learned score, $s_\theta(x,\sigma) \approx \nabla_x \log p_\sigma(x)$ |
| $G$ | generative prior map, $s = G(\xi)$, $\xi \sim \mathcal{N}(0,I)$ |
| $J$ | Jacobian $\partial s / \partial \xi$ |

Everything below is written per island. The outer CLEAN structure (island
detection, major-cycle residual update, gain) is preserved unless stated.

---

## 1. Shared substrate

### 1.1 Forward model

$$
d = A s + n, \qquad n \sim \mathcal{N}(0, N).
$$

In the visibility domain $A = S\,F\,C$ (sampling $S$, Fourier $F$,
primary-beam/gridding $C$) and $N = \sigma_n^2 I$ is white. In the image-domain
approximation $A s = h * s$ (PSF convolution); note that there the noise is
*PSF-correlated*, $N = \sigma_n^2\,(h \star h)$-like, which matters for honest
error bars and is one reason the vis-domain major cycle is preferable for the
likelihood. (Assumption A0, revisited in §2.7.)

### 1.2 Likelihood and the residual identity

Gaussian likelihood:

$$
p(d \mid s) = \mathcal{N}(d;\,A s,\,N),
\qquad
\mathcal{L}(s) \equiv -\log p(d\mid s) = \tfrac{1}{2}(d - As)^{\top} N^{-1} (d - As) + \text{const}.
$$

Gradient with respect to the sky:

$$
\boxed{\;\nabla_s \mathcal{L}(s) = -A^{\top} N^{-1} (d - A s) = -A^{\top} N^{-1} r\;}
$$

This is the single most important line in the document. $A^{\top} N^{-1} r$ is
the noise-weighted residual gridded back through the adjoint, which is exactly
what `tclean niter=0` (a major cycle) hands you. **CLEAN's residual is the
negative log-likelihood gradient.** Every method below consumes this quantity;
none of them has to recompute it.

The data Fisher information (used heavily in Fork B) is the Hessian of
$\mathcal{L}$:

$$
\nabla_s^2 \mathcal{L} = A^{\top} N^{-1} A \;=:\; M_d,
$$

constant in $s$ because the likelihood is quadratic.

### 1.3 Log-space and positivity

Put the field in log-space, $s = e^{f}$, so $s>0$ by construction and flux can
rearrange freely (a gradient step on $f$ moves flux without a clip). Chain rule,
with $\partial s_i/\partial f_i = e^{f_i} = s_i$:

$$
\nabla_f \mathcal{L} = \mathrm{diag}(e^{f})\,\nabla_s \mathcal{L}
= -\,e^{f} \odot \big(A^{\top} N^{-1} r\big).
$$

This is the RESOLVE positivity trick and it costs one elementwise multiply on top
of the residual you already have. The prior (below) is then defined over $f$.

### 1.4 Posterior

$$
p(s \mid d) \propto p(d \mid s)\,p(s),
\qquad
-\log p(s\mid d) = \mathcal{L}(s) - \log p(s) + \text{const}.
$$

All of the modelling freedom is in the prior $p(s)$ (or $p(f)$). The forks differ
only in how they turn $\mathcal{L}$ and $\log p$ into a posterior.

### 1.5 The prior as a learned score (Tweedie)

Define the noised prior $x_\sigma = x + \sigma \varepsilon$,
$\varepsilon \sim \mathcal{N}(0,I)$, with marginal $p_\sigma$. Tweedie's identity:

$$
\nabla_x \log p_\sigma(x) = \frac{\mathbb{E}[x_0 \mid x_\sigma = x] - x}{\sigma^2}
= \frac{D_\sigma(x) - x}{\sigma^2},
$$

where $D_\sigma$ is the MMSE denoiser, i.e. precisely what a diffusion model
learns. So a trained denoiser *is* the prior score. The score network is trained
by denoising score matching:

$$
\mathcal{J}(\theta) = \mathbb{E}_{\sigma}\,
\mathbb{E}_{x_0 \sim p_{\text{data}}}\,
\mathbb{E}_{\varepsilon}\Big[\,
\lambda(\sigma)\,\big\| s_\theta(x_0 + \sigma\varepsilon,\ \sigma) + \varepsilon/\sigma \big\|^2
\Big],
$$

equivalently in denoiser form $\lambda(\sigma)\|D_\theta(x_0+\sigma\varepsilon,\sigma) - x_0\|^2$.
$p_{\text{data}}$ is the corpus of clean sky patches (§4 is about what that corpus
must contain).

### 1.6 What is reused from the current loop

| Component | Current role | Role here |
|---|---|---|
| `IslandDetector` | isolate a source | isolate the field patch |
| major cycle (`tclean niter=0`) | residual update | supplies $A^\top N^{-1} r$ |
| `sigma_local` | conditioning | supplies $N$ / $\sigma_n$ |
| `loop_gain` | regularisation | becomes step size / schedule |
| minor cycle | greedy pick | **replaced** (score step or VI update) |

---

## 2. Fork A — Sampling in the loop (lowest effort)

### 2.1 Posterior score decomposition

The gradient of the log-posterior splits:

$$
\nabla_x \log p(x \mid d)
= \underbrace{\nabla_x \log p(d \mid x)}_{=\,-\nabla_x \mathcal{L} \,=\, A^\top N^{-1} r}
\;+\;
\underbrace{\nabla_x \log p(x)}_{\text{prior score } s_\theta}.
$$

Both terms are in hand: the first from the major cycle, the second from the
denoiser. Sampling means following this stochastic gradient.

### 2.2 Unadjusted Langevin (ULA) and the MALA correction

The overdamped Langevin diffusion $dx = \nabla \log p(x\mid d)\,dt + \sqrt{2}\,dw$
has $p(x\mid d)$ as its stationary distribution. Euler–Maruyama discretisation
with step $\eta$:

$$
\boxed{\;x_{k+1} = x_k + \eta\,\big[A^\top N^{-1} r(x_k) + s_\theta(x_k)\big] + \sqrt{2\eta}\,\xi_k,\quad \xi_k \sim \mathcal{N}(0,I)\;}
$$

This is ULA. The discretisation introduces an $O(\eta)$ bias; if you need it
removed, wrap each step in a Metropolis accept/reject (MALA) with acceptance

$$
\alpha = \min\!\Big(1,\ \frac{p(x_{k+1}\mid d)\,q(x_k\mid x_{k+1})}{p(x_k\mid d)\,q(x_{k+1}\mid x_k)}\Big),
\qquad
q(x'\mid x) = \mathcal{N}\!\big(x';\,x+\eta\nabla\log p(x\mid d),\,2\eta I\big).
$$

For a first build ULA is enough; MALA is the "make it exact" upgrade and needs a
likelihood evaluation per step.

### 2.3 Annealed Langevin (the practical algorithm)

Raw ULA mixes badly in high dimension because $\nabla \log p$ is uninformative far
from the data manifold. The fix is to anneal across noise levels
$\sigma_1 > \sigma_2 > \dots > \sigma_T$, using the level-matched score
$s_\theta(\cdot,\sigma_t)$ and a step size scaled to the level,
$\eta_t = \eta_0\,(\sigma_t/\sigma_T)^2$:

```
x ← sample from broad init
for t = 1..T:                       # noise levels, high → low
    for k = 1..K:                   # a few Langevin steps per level
        r   = d - A x               # major cycle (or cached A,Aᵀ)
        g   = Aᵀ N⁻¹ r + s_θ(x, σ_t)
        x   = x + η_t g + sqrt(2 η_t) ξ
return x                            # one posterior sample
```

One full sweep returns one posterior sample. Repeat for the ensemble, or read
the trajectory statistics.

### 2.4 Reverse-SDE / Diffusion Posterior Sampling (DPS)

The continuous-time version. The prior is a forward SDE (VE form)

$$
dx = \sqrt{\tfrac{d\sigma^2(t)}{dt}}\;dw, \qquad x(0)\sim p_{\text{data}},
$$

with the reverse-time SDE (Anderson)

$$
dx = -\,\tfrac{d\sigma^2}{dt}\,\nabla_x \log p_t(x\mid d)\;dt + \sqrt{\tfrac{d\sigma^2}{dt}}\;d\bar w.
$$

The conditional score again splits, $\nabla_x\log p_t(x\mid d) = \nabla_x\log p_t(x) + \nabla_x\log p_t(d\mid x)$.
The first term is $s_\theta(x,\sigma_t)$. The second is intractable at intermediate
noise levels; DPS approximates it by pushing the likelihood through the Tweedie
estimate of the clean image,

$$
\hat{x}_0(x_t) = x_t + \sigma_t^2\,s_\theta(x_t,\sigma_t),
\qquad
\nabla_{x_t}\log p_t(d\mid x_t) \approx -\,\zeta_t\,\nabla_{x_t}\,\tfrac{1}{2}\big\| d - A\,\hat{x}_0(x_t)\big\|^2_{N^{-1}}.
$$

The gradient on the right requires backprop through the denoiser (a
Jacobian-vector product), and $\zeta_t$ is a per-step weight. This is the IRIS /
DDRM family written in our notation. DPS is faster per sample than annealed ULA
(no inner $K$ loop) but the likelihood approximation is its main source of
miscalibration.

### 2.5 Mapping to the major/minor loop

The annealed sweep maps onto the existing structure with almost no change to the
outer code:

- **major cycle** = compute $r = d - Ax$ and grid it to $A^\top N^{-1} r$ (this is
  already `tclean niter=0`). Run it at the current iterate $x$ (or at
  $\hat{x}_0$ for DPS).
- **minor cycle** = the inner Langevin steps: evaluate $s_\theta$, combine with the
  cached residual gradient, take the noisy step. No peak-finding, no commit.
- **gain** = the step size $\eta_t$ and the cadence of major cycles. The
  gain-as-regularisation argument survives: small steps are now required for
  *sampling correctness*, not only stability.
- **output** = one full sweep is one posterior sample image; the ensemble gives
  per-pixel and morphological credible intervals.

### 2.6 Log-space variant

Run everything in $f$, $s = e^{f}$. The only changes:

$$
\nabla_f \log p(d\mid f) = -\,e^{f} \odot \big(A^\top N^{-1} r\big),
\qquad
s_\theta \text{ trained on log-sky},
$$

and positivity is automatic, flux rearranges. Recommended default.

### 2.7 Assumptions and failure modes

1. **Prior coverage (load-bearing).** $s_\theta$ only knows morphologies in
   $p_{\text{data}}$. Cyg A is plausibly out of distribution for the current
   parametric sims; the sampler will then drift toward the prior. This dominates
   everything and is *fork-independent* (§4). Risk: high.
2. **Operator consistency (A0).** $A,A^\top$ must be a true adjoint pair and the
   noise model honest. The vis-domain major cycle is faithful; image-domain PSF
   convolution has edge effects and PSF-correlated noise. Confidence the
   vis-domain path is fine: high.
3. **Mixing / step size.** ULA bias is $O(\eta)$; too large $\eta$ biases the
   posterior (not just unstable). MALA removes it at one likelihood eval per step.
4. **DPS approximation.** The Tweedie likelihood term is approximate; expect mild
   miscalibration unless corrected.
5. **Cost.** $O(T\cdot K)$ score+residual evaluations per sample,
   times $n_{\text{samples}}$ per island. Slow per image; embarrassingly
   parallel across samples and islands.

### 2.8 Calibration test

Reuse the existing coverage machinery (`eval_coverage.py`) lifted to fields:
draw $(s, d)$ from the simulator, run the sampler, check that the true $s$ falls
in the $1{-}\alpha$ posterior credible set at rate $1-\alpha$ (simulation-based
calibration / posterior-predictive coverage), per-pixel and on integrated-flux
and morphology summaries. Honest sampling should pass if the prior is right and
the chain mixes.

---

## 3. Fork B — RESOLVE / MGVI, and making it faster

Fork A gives honest samples slowly. Fork B trades some honesty for speed by
*optimising* an approximate posterior instead of sampling the exact one. This is
what RESOLVE does. The section ends with the speed menu, which is the part you
flagged as the real subject.

### 3.1 The variational objective

Pick a tractable family $q_\phi$ and minimise the reverse KL:

$$
\mathrm{KL}\!\big(q_\phi \,\|\, p(\cdot\mid d)\big)
= \mathbb{E}_{q_\phi}\big[\mathcal{L}(s) - \log p(s)\big] + \mathbb{E}_{q_\phi}\big[\log q_\phi\big] + \log p(d).
$$

Dropping the constant $\log p(d)$, minimising KL is maximising the ELBO

$$
\mathrm{ELBO}(\phi) = \mathbb{E}_{q_\phi}\big[\log p(d\mid s) + \log p(s) - \log q_\phi(s)\big].
$$

The reverse-KL direction is mode-seeking: it *underestimates* variance and will
ignore secondary modes. That is the structural source of RESOLVE-style
overconfidence (§3.6).

### 3.2 Standardized coordinates

Write the prior generatively, $s = G(\xi)$ with $\xi \sim \mathcal{N}(0,I)$. For a
log-normal field with covariance $C$ and mean $m$,

$$
s = \exp\!\big(C^{1/2}\xi + m\big), \qquad p(\xi) = \mathcal{N}(0,I).
$$

Now the prior term collapses to $-\log p(\xi) = \tfrac12\|\xi\|^2$ and *all* the
structure is carried by $G$. The posterior we approximate is over $\xi$:

$$
-\log p(\xi\mid d) = \tfrac12 (d - A\,G(\xi))^\top N^{-1} (d - A\,G(\xi)) + \tfrac12\|\xi\|^2 + \text{const}.
$$

### 3.3 The Fisher metric and MGVI

MGVI takes $q = \mathcal{N}(\bar\xi,\,M^{-1})$ with $M$ the Gauss–Newton / Fisher
metric of the standardized problem, evaluated at the current mean:

$$
\boxed{\;M(\bar\xi) = \underbrace{I}_{\text{prior}} + \underbrace{J^\top A^\top N^{-1} A\,J}_{\text{data Fisher},\ J = \partial s/\partial\xi|_{\bar\xi}}\;}
$$

The covariance is $M^{-1}$. MGVI never forms $M$; it uses it matrix-free in two
places.

**(a) Drawing posterior samples without inverting $M$.** To get
$\xi_s \sim \mathcal{N}(0, M^{-1})$, first draw a sample from the *curvature*,
$\eta \sim \mathcal{N}(0, M)$, by

$$
\eta = \xi_0 + J^\top A^\top N^{-1/2}\,\epsilon,
\qquad \xi_0 \sim \mathcal{N}(0,I),\ \epsilon \sim \mathcal{N}(0,I),
$$

since $\mathrm{Cov}(\eta) = I + J^\top A^\top N^{-1} A J = M$. Then solve

$$
M\,\xi_s = \eta \quad\text{by conjugate gradient},
\qquad \mathrm{Cov}(\xi_s) = M^{-1} M M^{-1} = M^{-1}.
$$

Each CG iteration is one application of $M$, i.e. one $J,\,A,\,A^\top,\,J^\top$
chain, i.e. one major-cycle-equivalent plus two backprops through $G$. Use
antithetic pairs $\pm\xi_s$ to halve variance.

**(b) Updating the mean.** With $K$ samples $\{\xi_s^{(i)}\}$, minimise the
sample-averaged energy

$$
H(\bar\xi) = \frac{1}{K}\sum_i \Big[\,\tfrac12 (d - A G(\bar\xi + \xi_s^{(i)}))^\top N^{-1}(\cdot) + \tfrac12\|\bar\xi + \xi_s^{(i)}\|^2\,\Big]
$$

by a Newton step preconditioned with the same metric $M$ (a natural gradient):

$$
\bar\xi \leftarrow \bar\xi - M^{-1}\nabla_{\bar\xi} H,
$$

the $M^{-1}$ again applied by CG. Alternate (a) and (b) to convergence. geoVI
generalises this with a nonlinear coordinate map to capture non-Gaussian
curvature, at higher cost.

### 3.4 Why vanilla RESOLVE is slow

Count the $A/A^\top$ applications (the expensive operation):

$$
\#\,(A,A^\top) \;\approx\; \underbrace{n_{\text{outer}}}_{\sim 10\text{--}30}
\times \Big(\underbrace{n_{\text{samp}}}_{\sim 4\text{--}20}
+ \underbrace{n_{\text{Newton}}}_{\sim 1\text{--}5}\Big)
\times \underbrace{n_{\text{CG}}}_{\sim 10\text{--}100}.
$$

That is $10^3$–$10^5$ operator applications per field. Speeding RESOLVE means
attacking one of these three factors or replacing the inner solve entirely.

### 3.5 Speed menu (what to learn / how to sample / shortcuts)

A ladder from least to most learned. Each rung removes one inner loop; the cost is
paid in calibration guarantees, noted per rung.

**S1. Learn the prior $G$.** Replace per-observation power-spectrum inference with
a fixed trained generative prior (score-decoder, VAE decoder, or flow). Removes
the prior-hyperparameter inference loop and makes $G$ and $J = \partial G/\partial\xi$
cheap, differentiable closed forms. Calibration: unchanged in principle; now
bounded by the learned prior's fidelity. Effort: moderate.

**S2. Learn / analytic preconditioner for CG.** The $n_{\text{CG}}$ factor is the
bottleneck. Use the prior square root $C^{1/2}$ as an analytic preconditioner
(standard in RESOLVE), or train $P_\omega(\cdot, d) \approx M^{-1}$ as a network
applied inside CG. Cuts $n_{\text{CG}}$ by an order of magnitude. Calibration:
unaffected (preconditioning changes speed, not the fixed point). Effort: low to
moderate.

**S3. Amortise the variational parameters.** Train a network
$h_\psi: d \mapsto (\bar\xi,\,\Sigma_\phi)$ predicting the posterior in one forward
pass, with $\Sigma_\phi$ a structured (diagonal + low-rank) covariance:

$$
\psi^\star = \arg\max_\psi\;
\mathbb{E}_{(s,d)\sim\text{sim}}\;\mathbb{E}_{\xi\sim q_\psi(\cdot\mid d)}
\big[\log p(d\mid G(\xi)) + \log p(\xi) - \log q_\psi(\xi\mid d)\big],
$$

reparameterised gradients. Inference becomes a single pass; $n_{\text{outer}}$ and
$n_{\text{CG}}$ vanish. Calibration: pays the *amortisation gap* (the network is
not optimal per image) on top of the Gaussian-$q$ gap. Effort: high (training),
trivial (inference).

**S4. Learn the sampler directly.** Skip $q$ entirely; train a conditional
generator $S_\omega(\epsilon; d)$ whose outputs are posterior samples, either by
distilling MGVI/Langevin outputs (sample matching) or by an amortised variational
/ score objective. Inference = forward passes, no CG, no SDE. This is the fast
end. Calibration: only as good as the distillation target; least guaranteed.
Effort: high.

**S5. Reduced-dimension latent.** Do VI in a learned low-dim $z \in \mathbb{R}^m$,
$m \ll N$, $s = G(z)$. The metric becomes $m \times m$,

$$
M_z = I_m + J_z^\top A^\top N^{-1} A\,J_z, \qquad J_z = \partial s/\partial z \in \mathbb{R}^{N\times m},
$$

dense and directly invertible, no CG. Calibration: now limited by what $G$'s
low-dim latent can express, which reintroduces the basis-arbitrariness concern we
spent this design avoiding. Use only if $m$-dim morphology coverage is acceptable.
Effort: moderate.

**S6. Hybrid MGVI + short Langevin tail.** Run cheap MGVI for the mean and a
Gaussian first guess, then a few annealed-Langevin steps (Fork A) around it to
correct non-Gaussianity and recover lost modes. Buys back calibration at a
fraction of full sampling cost. Effort: low once both forks exist.

A reasonable fast-RESOLVE target is **S1 + S2 + S3** (learned prior, learned
preconditioner, amortised $q$) for production throughput, with **S6** as the
calibration patch when error bars must be trusted.

### 3.6 Assumptions and calibration

1. **Prior coverage.** Same load-bearing assumption as Fork A (§4).
2. **Gaussian / unimodal posterior in $\xi$.** MGVI's $q$ is Gaussian; deconvolution
   posteriors are generically multimodal (position ambiguity, the
   unmeasured-spatial-frequency null space). Modes collapse, intervals run narrow.
   ~70% this bites for the general-morphology cases that motivated the rewrite.
3. **Reverse-KL underestimates variance.** Documented overconfidence; geoVI or the
   S6 Langevin tail mitigate.
4. **CG conditioning.** $M$ must be solvable matrix-free; preconditioning (S2) is
   usually required at scale.

### 3.7 Cost comparison

| | Per-image inference | Calibration | Rewrite effort | Multimodal |
|---|---|---|---|---|
| Fork A (annealed ULA) | slow ($T\cdot K \cdot n_{\text{samp}}$ evals) | honest (asymptotic) | low | yes |
| Fork A (DPS) | medium | approx | low–med | yes |
| Vanilla RESOLVE/MGVI | slow ($10^3$–$10^5$ ops) | overconfident | high | no |
| Fast RESOLVE (S1+S2+S3) | fast (≈1 pass) | overconfident + amortisation gap | very high | no |
| Fast RESOLVE + S6 | medium | improved | very high | partially |

---

## 4. The prior corpus and forward model

Settled 2026-06-13. Neither fork invents structure the prior never saw, so the
corpus is the dominant, fork-independent risk. The resolution: make the prior
encode generic radio-sky *statistics*, not source *templates*, and let the
explicit forward model produce all instrumental negatives.

### 4.1 Prior strength — statistics, not templates

Spectrum: weakest is a Gaussian random field plus inferred power spectrum
(RESOLVE proper, never OOD because it claims nothing about morphology); strongest
is a morphology-specific generative prior (e.g. a model fit to CRUMB radio
galaxies, which learns FR-type shapes and makes anything off-catalogue OOD).
**CRUMB is deliberately set aside here**: it is a radio-galaxy catalogue, so a
prior over it learns shapes, the strong/OOD-fragile end. The target is the
middle: a learned prior over generic radio-sky statistics. The aim is to learn
the *transformation* of a patch (a general restoration operator), not a shape
distribution, so the corpus must span the statistical range Cyg A lives in, not
contain Cyg A.

### 4.2 True-sky generator (strictly non-negative)

Stokes I total intensity is $\ge 0$ (exception: absorption against a bright
background, deferred with Q/U). The corpus is positive-only; negatives are not
sky. Components:

- **Diffuse:** a log-normal field $s_{\text{diff}} = \exp(g)$, $g$ a zero-mean GRF
  with power-law spatial power spectrum $P(k)\propto k^{-\alpha}$. **Not** a raw
  GRF (half-negative). This is RESOLVE's prior form.
- **Compact:** a point process, Poisson positions, fluxes drawn from measured
  source counts $dN/dS$.
- **Sharp non-Gaussian features:** rectified ridges / brightness discontinuities
  (shocks, lobe edges), all positive.

Calibrate the generator's second-order statistics (power spectrum, one-point flux
PDF, source counts) to *measured* radio-sky statistics, which are robust to CLEAN
artifacts in a way the images are not. The corpus is then statistically real
without containing a real object. Extend `extended_sky.py` (already a generator),
do not ingest a catalogue.

### 4.3 Forward model — the source of all negatives

$\text{dirty} = A s + n$, using the real VLA PSF bank (`load_g55_psf_bank`),
which carries true sidelobe structure. Convolving positive sky with the real PSF
reproduces **both** the sidelobe ringing **and** the negative bowl, because the
interferometric PSF integrates to $\approx 0$ (missing DC / short spacings), so
extended positive emission is necessarily surrounded by negatives. The negatives
are produced by the instrument, not fabricated: positive prior + zero-DC PSF
$\Rightarrow$ physical negatives.

Two rigor caveats on PSF manipulation:

- **Support / crop.** The bowl's spatial scale is set by the shortest measured
  baseline and can exceed an island. A PSF cropped to island size truncates the
  bowl for extended sources. Keep the PSF support large enough to contain the bowl
  scale of interest.
- **Rotation vs scaling.** Rotation is physically real (parallactic-angle rotation
  of the dirty beam over a track). Scaling a PSF up/down is acceptable as *data
  augmentation* to teach a coverage-agnostic operator, but it is *not* a faithful
  model of a specific different array/frequency/configuration: real uv-coverage is
  not self-similar under scaling (hole size, density, sidelobe pattern all change
  non-trivially). For a specific instrument use its real PSF; for the general
  operator, scale-and-rotate the bank as augmentation and own that it is
  augmentation.

Noise: add in the domain matching deployment (vis-domain gives honest
PSF-correlated image noise; image-domain is an approximation).

### 4.4 What is learned, what is penalized

The $(\text{dirty-with-negatives},\ \text{clean-positive})$ pair is the teaching
signal: input carries the negatives, target is positive truth. Penalization is
structural, not a tuned loss term: log-space $s=e^{f}$ makes a negative output
impossible (infinite penalty) and the likelihood attributes input negatives to
the PSF response of positive sky. Architecture dependence: a pure score prior
(Fork A core) trains *only* on the positive clean side and handles every negative
analytically via the likelihood, so it needs no dirty examples; an
amortized/learned operator (S3/S4, or the "learn the transformation" goal) needs
the forward-modeled dirty pairs to learn negative attribution.

### 4.5 Validation asymmetry

- **Sims:** ground truth available $\Rightarrow$ full image-space coverage / SBC.
- **Real (3C391, Cyg A):** no ground truth $\Rightarrow$ posterior-predictive only
  (push the posterior through $A$, check the data residual is noise). The explicit
  likelihood is a free OOD alarm: a too-narrow prior cannot fit data it cannot
  represent and surfaces as structured residual, not a confident wrong answer.

### 4.6 Open

- Which uv-coverage(s) to forward-model: single deployment array (narrow,
  rigorous) vs a sampled range (general, slower to converge). PJ leans general.
- How much non-Gaussian texture: GRF + points (fast, Gaussian texture) vs physical
  MHD/ISM sims for sharp shocks.
- Source of the measured target statistics ($P(k)$, $dN/dS$): which surveys.

---

## 5. Build order

1. **Fork A, log-space annealed ULA, lowest effort first.** Reuses the loop almost
   unchanged; the residual is already the gradient. Deliverable: honest per-pixel
   and morphological posteriors on simulated islands, coverage-tested.
2. **Branch to RESOLVE.** Stand up vanilla MGVI in standardized coordinates
   (correctness reference), then climb the speed ladder S1→S2→S3, using Fork A as
   the calibration oracle (S6) and as the honest baseline the fast variational
   posterior is checked against.

Recommendation, ~65%: sampling first, because it reuses the loop, gives honest
multimodal uncertainty (the actual deliverable), and needs only a score model the
simulator can train. RESOLVE-fast is the throughput play once the sampler has
established what "correct" looks like.
