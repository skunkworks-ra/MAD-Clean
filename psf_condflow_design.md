# PSF-CondFlow: Conditional Flow Matching for Single-Source Island Deconvolution

**Date:** 2026-06-15
**Status:** design, pre-implementation
**Supersedes:** the field-posterior score prior approach (field_posterior_design.md Fork A)

---

## Why This, Why Now

The score-prior approach (field_posterior_design.md) failed on two compounding problems:

1. **Representation.** Log-sky densified a sparse field into a full-support floor. Every pixel
   carried a finite value, so the DSM loss was dominated by the flat background and
   the ~0.1%-of-pixels source spikes carried essentially no gradient.

2. **Prior generalization.** Even with the linear-sky fix, the held-out eval at 25k steps showed
   median post/true = 0.07 on points and extended corr = 0.19. The loss plateaued at 0.005
   (floor-dominated, non-diagnostic). A separate prior that must generalize to all possible
   sky configurations is the wrong abstraction for the deconvolution problem.

MDN-Asp is the one method that actually deconvolves real data (3C391: residual 0.141→0.0089 Jy
over 5 cycles, G55: 0.0067→0.0004 Jy). It works because it sidesteps the prior-generalization
problem entirely: it learns a direct conditional model p(source | dirty, PSF), not a prior
p(sky) combined with a separate sampler. Its limitation is that the output lives in a 6D
Gaussian parameter space (amplitude, x, y, sigma_maj, sigma_min, PA). That Gaussian restriction
is the only thing preventing it from representing shells, filaments, and other non-Gaussian
morphology.

**The insight:** replace the MDN head with a conditional flow over the pixel-level source image.
Keep everything else — the proven 2-channel encoder, FiLM conditioning, the single-source-per-
island structure, the outer loop. Change only what is producing the Gaussian bottleneck.

---

## 0. Problem Statement

Given a dirty image cutout `d` (H × W) containing a single isolated source and the PSF `B`
(H × W, peak-normalized), draw samples from the posterior over clean source images:

    p(s | d, B)   where   d = B * s + n,   n ~ N(0, σ² I),   s >= 0.

`s` is a sparse, non-negative image. For a point source `s` is a single nonzero pixel;
for an extended source `s` is an arc, lobe, or shell — any morphology that the sky can have.
The flow is the posterior sampler. No separate prior needed: the model learns `p(s | d, B)`
directly from simulated training pairs.

---

## 1. Architecture

Three components, two of which already exist:

### 1.1 Context Encoder  (MDN-Asp encoder, unchanged)

```
Input:   (B, 2, 128, 128)  —  channel 0: dirty residual cutout
                            —  channel 1: PSF cutout (same 128×128)
Cond:    (B, 5)            —  (sigma_local, config_one_hot[4])

Per-sample residual normalization:
  scale = std(residual channel)
  image = [residual / scale, PSF]
  cond  = [cond, log10(scale)]        ← absolute flux path (same fix as CoeffFlow)

CNN: 5 × (stride-2 Conv2d + GroupNorm + GELU + Conv2d + GroupNorm + GELU)
  128 → 64 → 32 → 16 → 8 → 4, channels: c, 2c, 4c, 4c, 4c  (c=32)

Flatten → Linear(4*c*16, 256) → FiLM(cond) → FiLM(cond) → ctx ∈ R^256
```

This is the exact encoder from `mad_clean/models/mdn_asp.py` and `coeff_flow.py`.
Nothing changes here — this component demonstrably works.

### 1.2 Velocity U-Net  (new, minimal)

The flow backbone maps `(x_t, t, ctx)` → velocity field `v` of shape `(B, 1, H, W)`.

```
Input:
  x_t  (B, 1, H, W)  — noisy interpolant at time t
  t    (B,)           — flow time ∈ [0, 1]
  ctx  (B, 256)       — context from encoder

Architecture: same UNet from mad_clean/imaging/score.py
  emb_dim=128, base=32, mults=(1, 2, 2, 4), in_ch=1
  BUT: emb is computed from t (not log(sigma)) AND appended ctx

Time conditioning:
  t_emb = sinusoidal_embedding(t, 128)     ← same helper as score.py
  full_emb = MLP([t_emb || ctx]) ∈ R^128  ← t and ctx fused here

ResBlock embedding injection: full_emb → same add-to-channel FiLM as score.py
```

The U-Net already exists in `score.py`. The only change is how the embedding
is computed: `t` replaces `log(sigma)`, and `ctx` from the encoder is fused in
before injection. This is a minimal surgery — the backbone parameters are identical.

### 1.3 Output: clean source image

At inference, ODE integration from `x_0 ~ N(0, I)` gives `x_1 ≈ s` (the clean source).
Non-negativity is enforced by `relu` on the final output — not in the flow trajectory,
only on the returned sample.

---

## 2. Training

### 2.1 Conditional Flow Matching objective

Linear interpolant between noise and data (Lipman et al. 2022, Liu et al. 2022):

    x_t = t * s + (1 - t) * ε,     ε ~ N(0, I),     t ~ Uniform(0, 1)

The target velocity (the conditional vector field) is:

    u_t = s - ε

The training loss is:

    L(θ) = E_{(d,B,s), t, ε} [ || v_θ(x_t, t, encoder(d, B)) - u_t ||² ]

This is a pixel-wise MSE on the velocity field. No log-det, no score matching,
no noise level schedule. The encoder sees `(d, B)` for every step; the U-Net
sees `x_t` and `t`. Gradients flow through both.

### 2.2 Training corpus

Single-source per island. Three morphology classes:

| Class      | Definition                                     | Fraction |
|------------|------------------------------------------------|----------|
| point      | delta function, flux 10⁻³–10⁰ Jy               | 50%      |
| compact    | 2D Gaussian, sigma 0–3 px                       | 30%      |
| extended   | ridge / arc (from `assemble_corpus_field`)       | 20%      |

All strictly positive, no diffuse background floor. Generate from the existing
`assemble_corpus_field` generator with `n_points=(1,1)` or `n_ridges=(1,1)` and
`diffuse_flux=0` (no diffuse component; the PSF itself produces the sidelobes).

Forward model: `ImageDomainForward` with PSFs from `load_g55_psf_bank`, SNR range 10–200
(peak flux / noise std). Noise is added to the dirty image. The training pair is
`(dirty_cutout, PSF_cutout, clean_source)` for a 128×128 patch.

### 2.3 Why no diffuse background in training

The field-posterior experiment showed that diffuse background (`diffuse_flux=1.0`) was the
source of the log-sky collapse and the loss floor. Here the model never sees diffuse background
in its training target — it is only ever shown a single source. Background diffuse emission is
handled by the outer loop: the major cycle gridded residual already subtracts the current sky
model, so by the time a patch arrives at the minor cycle, the local residual is dominated by
the residual of whatever isolated source triggered the island detector.

### 2.4 Pixel weighting

The dominant-pixel problem is avoided by construction: each training target `s` is a single
source, so the source pixels are a large fraction of the nonzero content. No special weighting
needed in the MSE loss. If point sources prove hard (single bright pixel vs. 128² zeros), add
a support-weighted loss:

    L_w = E [ w(s) * || v_θ - u_t ||² ],   w(s) = 1 + λ * (s / s_mean)

with λ ∈ {3, 10} set by the overfit gate, same as `edm_loss pixel_weight`.

---

## 3. Inference in the Loop

The outer loop is unchanged (island detector → crop → minor cycle → PSF subtract → major
cycle). Only the minor cycle solver changes:

```
Input:  residual cutout d (128×128), PSF cutout B (128×128), sigma_local, config_idx

1.  ctx = encoder(d, B, sigma_local, config_idx)          ← MDN-Asp encoder
2.  x   = randn(n_samples, 1, 128, 128)                    ← n_samples ~ 8–16
3.  ODE integration (Euler, n_steps=20):
        for t in linspace(0, 1, n_steps):
            x = x + (1/n_steps) * v_θ(x, t, ctx)
4.  s_samples = relu(x)                                    ← non-negativity
5.  s_model   = mean(s_samples)                            ← point estimate for subtraction
6.  s_std     = std(s_samples)                             ← per-pixel uncertainty

Subtract: residual -= loop_gain * B * s_model
```

The loop_gain here plays the same regularization role as in MDN-Asp and CLEAN.
The uncertainty `s_std` is the first honest per-pixel uncertainty image for free.

---

## 4. Relationship to Existing Components

| Component             | Role here                          | Change from current use  |
|-----------------------|------------------------------------|--------------------------|
| `MDNAsp.enc`          | context encoder                    | REUSE, unchanged         |
| `FiLMBlock`           | context injection in velocity net  | REUSE, same pattern      |
| `UNet` (score.py)     | velocity field backbone            | REUSE, rewire embedding  |
| `ImageDomainForward`  | forward model in training          | REUSE, unchanged         |
| `load_g55_psf_bank`   | PSF family for training            | REUSE, unchanged         |
| `assemble_corpus_field`| source generation                 | REUSE, single-source mode|
| `MDNAsp.head` (6D MDN) | →                                 | REPLACE with flow        |

New files needed:
- `mad_clean/models/psf_condflow.py` — `PSFCondFlow` model (encoder + velocity U-Net)
- `scripts/train_psf_condflow.py` — training loop
- `scripts/overfit_psf_condflow.py` — overfit gate (mirrors `overfit_field_inloop.py`)

The training script mirrors `train_wavelet_npe.py` in structure: infinite corpus stream,
per-step MSE loss, EMA, checkpoint with config stored.

---

## 5. Overfit Gate (before full training)

The failure mode of PatchFlow was discovered only after training. Run this first:

```
overfit N=8 single-source patches (mix of points and one arc), 2000 steps.
Reconstruct patch 0 with n_samples=8.
Check: mean post/true windowed-flux >= 0.5 for all sources.
       peak argmax within 2 px of true source.
```

This is faster than `overfit_field_inloop.py` because the training target is explicit
(single source, no background). If this fails, the flow backbone or encoder wiring is wrong.
If it passes, proceed to full training.

---

## 6. What This Buys Over MDN-Asp

| Property                        | MDN-Asp      | PSF-CondFlow              |
|---------------------------------|--------------|---------------------------|
| Source morphology               | Gaussian only| arbitrary pixel image      |
| Posterior geometry              | K Gaussians  | general (non-Gaussian OK)  |
| Point-source deconvolution      | yes          | yes (simpler target)       |
| Shell / filament morphology     | no           | yes (arc in training set)  |
| Per-pixel uncertainty           | no           | yes (sample variance)      |
| Encoder architecture            | CNN + FiLM   | same CNN + FiLM            |
| Real-data track record          | yes (3C391)  | unproven (same encoder)    |

MDN-Asp stays as the fallback: it is the only method with proven convergence on real data.
PSF-CondFlow is an extension, not a replacement, until it passes the same real-data test.

---

## 7. Risks

1. **Point-source sparsity in pixel space.** A 1-pixel source in 128² = 0.006% support.
   The flow has to push flux to a single pixel from Gaussian noise. This is the same
   dimensionality problem as before, but now the model sees the dirty image (the PSF response
   at the right location) as conditioning. The likelihood in the dirty image is strong, so the
   encoder should localize well. Risk: medium. Mitigation: the pixel-weight loss lever; start
   with compact Gaussians (width >= 1 px) in the training corpus before pure delta functions.

2. **ODE integration artifacts.** Few-step Euler integration of the velocity field introduces
   discretization error. This may produce blurry posteriors. Mitigation: use 50-100 steps at
   inference first, then reduce once the model is validated.

3. **Encoder sees PSF once but velocity net needs to "know" the PSF at every integration step.**
   The PSF information lives only in `ctx`, computed once at the start of the ODE. At later
   steps, when `x_t` is close to `s`, the velocity should respect the PSF-convolved appearance.
   The context injection via FiLM should carry this — but if the integration drifts, the ODE
   trajectory may produce sources inconsistent with the dirty image. Mitigation: the gate
   tests reconstruction quality directly; a unit test that checks the posterior mean is
   PSF-consistent (convolve `s_model` with `B`, compare to `d`, measure chi-squared) catches
   this early.

4. **Overconfident uncertainty.** CFM posterior samples do not come with calibration guarantees.
   The sample variance is a rough uncertainty estimate. For honest uncertainty, a coverage test
   (simulation-based calibration) is needed. Treat this as a v2 deliverable.

---

## 8. Build Order

1. **Kill the current training run** (`train_field_score.py` at step ~30k, loss plateaued).

2. **`mad_clean/models/psf_condflow.py`** — define `PSFCondFlow`:
   - `_ContextEncoder` — copy directly from `coeff_flow.py` (it already has per-sample
     normalization and log-scale appended to cond)
   - `_VelocityUNet` — thin wrapper around `score.UNet` that fuses `t` embedding and `ctx`
     into the U-Net embedding before the first ResBlock
   - `PSFCondFlow.forward(x_t, t, image, cond)` → velocity field
   - `PSFCondFlow.sample(image, cond, n_samples, n_steps)` → clean source samples

3. **`scripts/overfit_psf_condflow.py`** — overfit gate on 8 fixed patches. Gate must
   pass before committing GPU time.

4. **`scripts/train_psf_condflow.py`** — full training:
   - Corpus stream: single-source patches, mixed morphology, `diffuse_flux=0`
   - Batch size 16, image size 128×128
   - Adam lr=1e-4, 100k steps, checkpoint every 5k
   - EMA decay 0.9999

5. **Held-out eval**: same `overfit_psf_condflow.py --ckpt` pattern as `overfit_field_inloop.py`.

6. **Loop integration**: wire `PSFCondFlow` into `mad_clean/minor_cycle.py` as a new solver
   option alongside `MDNAsp`, running under `run_imaging_3c391.py`.

---

## 9. Stopping Criteria

**Gate fails:** if the overfit gate (step 3 above) shows median post/true < 0.5 on points
with n_steps=100 and 2000 overfit steps, the velocity U-Net embedding is wrong. Debug the
t + ctx fusion before training.

**Training loss:** CFM loss on single-source patches is not floor-dominated (source pixels
are a non-negligible fraction of H×W for a Gaussian). A decreasing training loss is meaningful.
If loss flatlines above ~0.1 by step 20k, the encoder context is not flowing into the U-Net.

**Held-out eval success criterion:** at step 50k, held-out patch reconstruction:
- Points: median post/true windowed-flux >= 0.5, argmax within 3 px.
- Extended (arc): flux_ratio 0.5–2.0, corr >= 0.6 on the dilated mask.

If held-out passes: proceed to `run_imaging_3c391.py` test. MDN-Asp is the baseline.
