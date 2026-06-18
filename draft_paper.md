# MAD-CLEAN: Amortised Posterior Deconvolution with an Explicit Measurement Operator

*Working draft — method and mathematics. Notation is fixed in §1 and reused throughout.*

---

## 1. The deconvolution problem

A radio interferometer measures samples of the sky's complex visibility function.
Stacking the measured visibilities into a vector $V \in \mathbb{C}^M$, the linear
measurement model is

$$
V = A\,x + n ,
$$

where $x \in \mathbb{R}^N$ is the sky brightness distribution discretised on an
image grid, $A : \mathbb{R}^N \to \mathbb{C}^M$ is the **measurement operator**
(the gridded, weighted Fourier sampling implied by the $uv$-coverage, optionally
including $w$-projection and A-projection kernels), and $n$ is additive noise,
taken zero-mean Gaussian with covariance $\Sigma_n = \sigma^2 I$ after
whitening.

The **dirty image** is the adjoint applied to the data,

$$
x_D \;=\; A^{\dagger} V \;=\; A^{\dagger} A\, x + A^{\dagger} n
\;=\; B\, x + A^{\dagger} n ,
\qquad B \equiv A^{\dagger} A ,
$$

where $B$ is the **point spread function (PSF) operator** — convolution by the
dirty beam under a uniform-coverage approximation. Deconvolution is the inverse
problem of recovering $x$ from $x_D$ given $B$ (or, more fundamentally, $x$ from
$V$ given $A$). It is ill-posed: $A$ has a large null space (unsampled $uv$
cells), so infinitely many sky models reproduce the data.

---

## 2. A Bayesian outlook on deconvolution

We treat the sky as a random field and seek the **posterior**

$$
p(x \mid V) \;\propto\; p(V \mid x)\, p(x) ,
\qquad
p(V \mid x) \;\propto\; \exp\!\Big(-\tfrac{1}{2\sigma^2}\,\lVert V - A x\rVert_2^2\Big) .
$$

The likelihood is fixed by the instrument and the noise; all of the
deconvolution difficulty lives in the **prior** $p(x)$, which must encode what
real radio sky emission looks like (compact sources, smooth extended structure,
non-negativity).

Two observations motivate the entire method:

1. **Point estimates discard the most scientifically useful information.**
   Classical CLEAN returns a single model image $\hat{x}$, implicitly a
   maximum-a-posteriori (MAP) estimate under a sparse prior solved by greedy
   matching pursuit. It says nothing about *which* features are well constrained
   by the data and which are reconstructions the $uv$-coverage cannot support.
   The posterior $p(x\mid V)$ contains exactly that information.

2. **The posterior is intractable, but it is amortisable.** $p(x\mid V)$ is an
   $N$-dimensional distribution with a data-defined, non-conjugate prior; it
   admits no closed form. However, the *mapping* from a local data feature to the
   posterior over the source that produced it is **stable across the sky and
   across observations**. We can learn that mapping once, offline, from
   simulations, and then apply it everywhere — *amortised inference* in the
   simulation-based-inference (SBI) sense.

This is the philosophical centre of MAD-CLEAN: we do not learn a denoiser or an
end-to-end image-to-image map. We learn an **amortised posterior over the
parameters of a sky model component, conditioned on the local residual**, and we
keep the measurement operator $A$ explicit and exact in the outer loop.

---

## 3. The major/minor cycle as an inference scaffold

We deliberately retain the **Cotton–Schwab major/minor cycle** structure rather
than replace it with a one-shot network or a plug-and-play (PnP) prior. The
reasons are both numerical and statistical.

**Minor cycle (image domain, approximate operator).** Operating on the current
residual image $r^{(j)}$, we identify emission, propose model components, and
subtract their PSF response using the image-domain beam $B$. This is cheap and
runs many iterations.

**Major cycle (data domain, exact operator).** Periodically we recompute the
residual through the *exact* measurement operator,

$$
r^{(j)} \;=\; A^{\dagger}\big(V - A\, \hat{x}^{(j)}\big)
        \;=\; x_D - B\,\hat{x}^{(j)}\big|_{\text{exact } A},
$$

which corrects the errors incurred by the approximate image-domain $B$ (gridding,
$w$-term, beam variation). In practice the major cycle is a `tclean` call with
`niter=0` that degrids the accumulated model and reimages the residual.

**Design invariant.** The learned network *never sees and never touches* $A$. It
only ever receives a residual island and returns a sky-model proposal; the PSF is
applied to that proposal by the explicit operator in the loop. This separates
MAD-CLEAN from PnP/AIRI-style methods, where a learned prior is fused into the
data-fidelity gradient and the network's behaviour is entangled with the
operator. Keeping $A$ explicit means:

- the data-fidelity term is always exact and the algorithm cannot hallucinate
  flux that is inconsistent with the measured visibilities;
- a small loop gain $\gamma \ll 1$ provides the usual CLEAN regularisation
  against PSF sidelobes and model error;
- the network's job is reduced to the *well-posed, local* task of describing the
  source under a residual peak, which is exactly the task an amortised posterior
  can be trained to solve.

The loop is therefore a **posterior-guided matching pursuit**: at each minor-cycle
step the network proposes the (distribution over the) component most consistent
with the local residual, and the explicit operator enforces global data fidelity.

---

## 4. Method A — MDN-Asp: amortised parametric posterior

### 4.1 Source parameterisation (the "Aspen")

We model the sky as a sum of parametric components, each an elliptical Gaussian
("Aspen") with parameter vector

$$
\theta \;=\; \big(x_c,\; y_c,\; \log F,\; \log\sigma_{\mathrm{maj}},\;
\log\sigma_{\mathrm{min}},\; \varphi\big) \;\in\; \mathbb{R}^6 ,
$$

i.e. centroid $(x_c, y_c)$ in pixels, total integrated flux $F$ (log-coded), major
and minor axis widths $\sigma_{\mathrm{maj}} \ge \sigma_{\mathrm{min}}$ (log-coded,
floored at one beam), and position angle $\varphi$. The rendering operator that
turns $\theta$ into an image stamp is

$$
g_\theta(p) \;=\; F \cdot \frac{1}{Z(\theta)}
\exp\!\Big(-\tfrac{1}{2}\big[(u/\sigma_{\mathrm{maj}})^2 + (v/\sigma_{\mathrm{min}})^2\big]\Big),
$$

where $(u,v)$ are the pixel coordinates of $p$ rotated into the source frame by
$\varphi$, and $Z(\theta)$ normalises the stamp so that $\sum_p g_\theta(p) = F$.

### 4.2 Position-angle encoding

A symmetric ellipse is invariant under $\varphi \mapsto \varphi + \pi$. To make
the regression target single-valued we encode

$$
\xi(\varphi) \;=\; \big(\sin 2\varphi,\; \cos 2\varphi\big) \in \mathbb{R}^2,
\qquad
\varphi \;=\; \tfrac{1}{2}\,\operatorname{atan2}(\sin 2\varphi, \cos 2\varphi)
\;\in\; (-\tfrac{\pi}{2}, \tfrac{\pi}{2}] .
$$

The network operates in the **7-dimensional emitted space**
$(x_c, y_c, \log F, \log\sigma_{\mathrm{maj}}, \log\sigma_{\mathrm{min}},
\sin 2\varphi, \cos 2\varphi)$ and decodes back to $\mathbb{R}^6$ at output.

### 4.3 The amortised posterior

Let $r$ be a $128\times128$ residual island and $c$ a conditioning vector
(below). The network defines a **mixture-density posterior**

$$
q_\psi(\theta \mid r, c) \;=\; \sum_{k=1}^{K} \pi_k(r,c)\;
\mathcal{N}\!\big(\tilde\theta;\, \mu_k(r,c),\, \operatorname{diag} s_k^2(r,c)\big),
\qquad K = 5,
$$

evaluated in the 7-dim emitted space $\tilde\theta = (\theta_{1:5}, \xi(\varphi))$.
The mixture weights $\pi_k = \mathrm{softmax}(\ell_k)$ come from logits $\ell_k$;
the per-component means $\mu_k$ and log-standard-deviations $\log s_k$ are emitted
by the head. Per-dimension floors are imposed on $\log s_k$ (tighter on the scale
dimensions, to stop components collapsing to near-delta peaks; looser on the
$\xi$ dimensions, whose natural range is $[-1,1]$).

**Why a mixture.** Under source confusion and PSF sidelobes the posterior over the
source beneath a residual peak is genuinely multi-modal (e.g. two nearly-equal
flux/position explanations). A single Gaussian cannot represent this; a $K$-mode
mixture can. Note that $K=5$ is the number of **posterior modes for one source**,
not a claim of up to five sources per island.

### 4.4 Conditioning

$$
c \;=\; \big(\sigma_{\text{local}},\; \mathbf{e}_{\text{config}}\big) \in \mathbb{R}^5,
\qquad
\sigma_{\text{local}} = 1.4826 \cdot \operatorname{median}_p |r_p| ,
$$

a robust (MAD) noise estimate on the island, concatenated with a one-hot encoding
$\mathbf{e}_{\text{config}}$ of the array configuration (A/B/C/D). The
conditioning enters the network through FiLM (feature-wise linear modulation)
layers, $h \mapsto (1+\gamma(c))\odot h + \beta(c)$.

> **Known limitation (to be addressed in a later model).** In the current
> training set the thermal noise is held fixed, so $\sigma_{\text{local}}$ carries
> no learnable signal and the network learns source-scale uncertainty from the
> image alone rather than from this channel. A noise-adaptive posterior requires
> (i) varying the simulated noise level across scenes and (ii) standardising
> $\sigma_{\text{local}}$ before the FiLM layer. See the audit (§2.1).

### 4.5 Architecture

A strided convolutional encoder maps the 2-channel input (residual island, PSF
island) $128\to64\to32\to16\to8\to4$ with GroupNorm+GELU blocks, then **flattens**
(rather than global-average-pools) to preserve sub-pixel localisation
information, projects to a hidden vector, applies two FiLM blocks conditioned on
$c$, and emits $K(1 + 2\cdot 7)$ numbers: $K$ logits, $K\times7$ means,
$K\times7$ log-stds. (~$1.6\times10^6$ parameters at the production width.)

### 4.6 Training objective (neural posterior estimation)

We draw simulated pairs $(r, \theta^\star)$ from the generative model of §4.7 and
minimise the **negative log-likelihood of the true parameters under the
predicted mixture**:

$$
\mathcal{L}(\psi) \;=\; -\,\mathbb{E}_{(r,\theta^\star,c)\sim \text{sim}}
\Big[\log q_\psi\big(\theta^\star \mid r, c\big)\Big] .
$$

For a single example, with $\tilde\theta^\star$ the emitted-space target,

$$
\log q_\psi(\theta^\star\mid r,c)
= \operatorname{logsumexp}_{k}\Big[
\log \pi_k - \tfrac{1}{2}\sum_{d=1}^{7}\Big(
\frac{(\tilde\theta^\star_d - \mu_{k,d})^2}{s_{k,d}^2}
+ 2\log s_{k,d} + \log 2\pi \Big)\Big].
$$

This is exactly the SBI / neural-posterior-estimation objective: minimising
$\mathcal{L}$ is equivalent to minimising the expected forward KL divergence

$$
\mathbb{E}_{r}\big[\, \mathrm{KL}\big(p(\theta\mid r)\,\Vert\, q_\psi(\theta\mid r)\big)\big]
+ \text{const},
$$

so the trained $q_\psi$ is a calibrated amortised approximation to the true
posterior over the source parameters — **provided the test data lies in the
simulated distribution.**

### 4.7 Simulation (the implicit prior)

Each training scene is generated on the fly: a field of distractor sources
(drawn from a morphology mixture) is convolved with a PSF sampled from a bank of
realistic $uv$-coverage realisations and corrupted with noise; one centred
"target" source of controlled morphology is placed with sufficient margin and
its true $\theta^\star$ recorded. The residual island is *not* cleaned of the
other sources — it carries realistic PSF contamination, exactly as it would
during a real minor cycle. A minimum-SNR floor rescales the target so that
undetectable sources are excluded, matching the operational regime of the loop.
The morphology distribution and PSF bank together *are* the prior $p(x)$; the
network never sees an explicit prior density.

### 4.8 Inference inside the minor cycle

At minor-cycle step $j$:

1. locate the peak of the masked residual $r^{(j)}$;
2. extract the $128\times128$ island centred there and build $c$;
3. evaluate $q_\psi(\theta\mid r,c)$ and take the **mode** (argmax-weight
   component mean) $\hat\theta$, clamping flux to the residual peak and widths to
   the physical $[\,1\,\text{beam}, \sigma_{\max}]$ range to guard against
   out-of-distribution extrapolation;
4. render $g_{\hat\theta}$ and commit a gain-weighted increment to the model,
   subtracting its PSF response from the residual:
   $$
   \hat{x} \mathrel{+}= \gamma\, g_{\hat\theta},
   \qquad
   r \mathrel{-}= \gamma\, (B \circledast g_{\hat\theta}),
   \qquad \gamma \approx 0.1 .
   $$

Major cycles periodically reset $r$ through the exact operator (§3).

### 4.9 Uncertainty image — posterior pushforward (Layers 1–3)

The native object is the **parameter posterior** (Layer 1) $q_\psi(\theta\mid r,c)$
for each committed component $i$. The deployable scientific product is its
**image-domain pushforward** (Layer 2). Let $\theta^{(s)}_i \sim q_\psi$ be
posterior draws ($s=1,\dots,S$) for committed component $i$. The per-pixel image
posterior induced by the model is

$$
I^{(s)}(p) \;=\; \sum_i g_{\theta_i^{(s)}}(p),
\qquad
\sigma_I(p) \;=\; \operatorname{std}_s\, I^{(s)}(p) .
$$

Because the per-component posteriors are independent given the residual, the
variance separates,

$$
\operatorname{Var}\big[I(p)\big] \;=\; \sum_i
\operatorname{Var}_{\theta\sim q_{\psi,i}}\!\big[g_\theta(p)\big],
\qquad
\sigma_I(p) = \sqrt{\textstyle\sum_i \operatorname{Var}_i[g(p)]} ,
$$

which is the statistically correct combination and **replaces the earlier
pixel-wise-maximum heuristic**. Crucially the draws vary the *full* 6D
$\theta$ — position, flux, and shape — so positional and morphological
uncertainty propagate into the per-pixel flux uncertainty, not flux alone.

*Implementation note (planned, "S3").* Naïvely this is $S\times$ more full-frame
renders than the model image; it is made tractable by rendering each draw into a
local bounding box ($\sim\!\pm5\sigma_{\mathrm{maj}}$) and accumulating an online
(Welford) per-pixel variance into the full map, so no $S$ full images are held in
memory. The existing full-frame `render_aspen` is retained unchanged for the
model image.

**Layer 3 — what this uncertainty is *not*.** $\sigma_I$ expresses aleatoric
(thermal-noise) plus parametric epistemic uncertainty **conditional on the source
belonging to the trained morphology family**. It does **not** include model
misspecification. Presented with emission outside the training distribution, the
network can return a confident, narrow posterior around the wrong morphology — the
uncertainty map will look most trustworthy exactly where the model is most wrong.
Claims must therefore be restricted to in-distribution calibration, which is
verified directly by held-out coverage tests (empirical $68\%/95\%$ interval
coverage per parameter dimension).

---

## 5. Method B — Conditional Flow Matching for island deconvolution

### 5.1 Probability path

Method B learns a **non-parametric image-domain map** from dirty to clean,
rather than a parametric component posterior. Let $x_0$ be a dirty island and
$x_1$ the corresponding clean (true) island. Define the straight-line conditional
probability path

$$
x_t \;=\; (1-t)\,x_0 + t\,x_1, \qquad t \in [0,1],
$$

interpolating from the dirty image ($t=0$) to the clean image ($t=1$). The
**conditional velocity** of this path is constant,

$$
u_t(x_t \mid x_0, x_1) \;=\; \frac{\mathrm{d}x_t}{\mathrm{d}t} \;=\; x_1 - x_0 .
$$

### 5.2 Flow-matching objective

A time-dependent velocity field $v_\theta(x,t)$ (a U-Net with sinusoidal time
embedding, $\sim\!2.5\times10^6$ parameters) is trained to regress the conditional
velocity:

$$
\mathcal{L}_{\mathrm{CFM}}(\theta)
= \mathbb{E}_{t\sim\mathcal{U}[0,1]}\;
  \mathbb{E}_{(x_0,x_1)}\;
  \tfrac{1}{2}\big\lVert v_\theta(x_t, t) - (x_1 - x_0)\big\rVert_2^2 .
$$

By the flow-matching identity, the minimiser is the **marginal** velocity
$v^\star_t(x) = \mathbb{E}[\,x_1 - x_0 \mid x_t = x\,]$, whose flow transports the
dirty distribution onto the clean distribution. Training therefore needs only
paired samples and an MSE loss — no simulation of the reverse-time SDE, no score
matching, no PSF at inference (the beam is implicit in the $x_0\!\to\!x_1$
pairing).

### 5.3 Inference

Deconvolution integrates the learned ODE forward from the dirty island:

$$
\frac{\mathrm{d}x}{\mathrm{d}t} = v_\theta(x, t),
\qquad x(0) = x_{\text{dirty}},
\qquad \hat{x}_{\text{clean}} = x(1),
$$

discretised by explicit Euler over $n$ steps, $x \mathrel{+}= \tfrac{1}{n}\,
v_\theta(x, t)$.

### 5.4 Uncertainty via trajectory ensembles

An uncertainty estimate is obtained by perturbing the start point and integrating
an ensemble,

$$
x^{(s)}(0) = x_{\text{dirty}} + \varepsilon^{(s)},\quad
\varepsilon^{(s)}\sim\mathcal{N}(0,\tau^2 I),\quad s = 1,\dots,S,
$$

$$
\mu(p) = \operatorname{mean}_s x^{(s)}(1, p),
\qquad
\sigma(p) = \operatorname{std}_s x^{(s)}(1, p).
$$

> **Caveat (flag, ~80% confidence this needs care).** This ensemble spread is a
> *perturbation-sensitivity* estimate, not a calibrated Bayesian posterior. Its
> width is set by the injected $\tau$ and the local Lipschitz behaviour of
> $v_\theta$, and it should be calibration-checked (coverage vs. truth) before any
> quantitative use, exactly as for Method A's Layer 2.

---

## 6. Relationship between the two methods

| | **Method A — MDN-Asp** | **Method B — CFM flow** |
|---|---|---|
| Posterior over | source **parameters** $\theta\in\mathbb{R}^6$ | **image** pixels (island) |
| Representation | $K$-mode Gaussian mixture | implicit, via ODE samples |
| Prior | morphology mixture + PSF bank (simulation) | dirty$\to$clean pairs (simulation) |
| Operator in loop | explicit $A$ (major/minor cycle) | implicit in training pairs |
| Native uncertainty | parametric, calibratable by coverage | trajectory-ensemble spread |
| Strength | interpretable, sub-beam localisation, sparse | extended/diffuse morphology, non-parametric |

Both are amortised, simulation-trained posteriors that keep the learned component
*proposing* a sky model while the explicit measurement operator enforces data
fidelity. Method A excels where the sky is well described by compact parametric
components and where calibrated per-source error bars are the deliverable; Method
B is the natural choice for extended emission that resists parametric
description. They share the major/minor scaffold of §3 and differ only in how the
minor-cycle proposal is represented.

---

## 7. Open items feeding the final draft

- In-distribution coverage table for the released MDN-Asp model (Layer-1
  calibration evidence).
- Noise-adaptive conditioning experiment (§4.4 limitation).
- Morphology-complete retrain (point+blob+shell+filament) for the extended G55
  science case.
- S3 pushforward uncertainty map (§4.9) and its calibration check.
- Quantitative head-to-head of Methods A and B on matched fields.
