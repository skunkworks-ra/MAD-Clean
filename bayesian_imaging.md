# Bayesian Radio Imaging — State of the Art

**Last updated:** 2026-03-30
**Purpose:** Literature context for MAD-CLEAN Variant C (flow-based posterior solver)

---

## 1. The Arras / Enßlin / NIFTy Group (MPA Garching)

**Core framework: RESOLVE + MGVI**

The foundational line. All work uses Information Field Theory (IFT) with Metric Gaussian
Variational Inference (MGVI) as the posterior approximation. PSF / measurement operator is
explicit in all cases. All work operates at full-image scale.

| Paper | Year | arXiv | Key contribution |
|---|---|---|---|
| Radio Imaging with IFT (Arras, Knollmüller, Enßlin et al.) | 2018 | 1803.02174 | Foundational RESOLVE; log-normal field prior; calibrated noise; uncertainty maps without MCMC |
| Comparison of classical and Bayesian imaging (Arras et al.) | 2021 | 2008.11435 | RESOLVE vs CLEAN on Cygnus A (VLA); uncertainty estimates; no negative flux artefacts |
| Unified calibration + imaging with joint UQ (Arras, Enßlin et al.) | 2019 | — | Joint MGVI over calibration and image parameters simultaneously |
| Bayesian RI imaging with direction-dependent calibration (Arras et al.) | 2023 | 2305.05489 | Extends RESOLVE to direction-dependent gains |
| fast-resolve (Roth, Arras, Enßlin et al.) | 2024 | 2406.09144 | ~144× faster via JAX/NIFTy.re + FFT convolutions; enables MeerKAT-scale datasets |
| aim-resolve (Fuchs et al.) | 2024 | 2512.04840 | Adds U-Net semantic segmentation on top of RESOLVE backbone |
| Bayesian self-cal and imaging in VLBI (Kim, Roth, Enßlin, Arras et al.) | 2024 | 2407.14873 | Applied to M87 @ 43 GHz; better resolved than CLEAN |
| Bayesian polarization calibration VLBI | 2025 | 2511.16556 | Extension to Stokes polarization; same variational framework |
| NIFTy.re (Edenhofer, Frank, Roth, Enßlin et al.) | 2024 | 2402.16683 | JAX rewrite of NIFTy; MGVI + geoVI; backbone for all recent work above |

**Characterisation for MAD-CLEAN comparison:**
- Prior: hand-crafted log-normal field prior (IFT). Not learned from data.
- Posterior approximation: MGVI (Gaussian in a transformed space). Not a flow.
- Scale: global full-image inference. No source decomposition.
- No flow matching. No island-level factorisation.

---

## 2. Score-Based / Diffusion Prior Methods

These replace the hand-crafted IFT prior with a neural prior learned from galaxy image datasets.
PSF / measurement operator remains explicit in the likelihood. All operate at full-image scale.

| Paper | Year | arXiv | Method | Notes |
|---|---|---|---|---|
| Bayesian Imaging for RI with Score-Based Priors (Dia, Scaife, Bowles et al.) | 2023 | 2311.18012 | DDPM-based score prior + annealed Langevin | Prior trained on optical galaxies; tested on DSHARP protoplanetary disk |
| IRIS (Dia et al.) | 2025 | 2501.02473 | Score-SDE prior + posterior reverse-SDE | Full posterior sampling from visibilities; ALMA DSHARP survey |
| Radio-RI Reconstruction with DDRM (Potevineau et al.) | 2026 | 2601.15844 | DDPM prior trained on VLA FIRST + DDRM sampling | Unsupervised posterior sampling; no retraining per observation; PSNR > 60; tested on VLA/EHT/ALMA |
| VIC-DDPM | 2023 | 2305.09121 | Conditional DDPM conditioned on dirty map + visibilities | PSF implicit (conditioned on dirty map) — generalisation concern |

**Characterisation for MAD-CLEAN comparison:**
- DDRM (Potevineau et al.) is the closest existing approach to Variant C: unsupervised prior,
  explicit likelihood, radio-specific training data (VLA FIRST). Full image, not island-level.
- IRIS uses score-SDE (Song et al. 2020 SDE framework), not flow matching (Lipman et al.).

---

## 3. VLBI / EHT Diffusion-Based Methods

| Paper | Year | arXiv | Method | Notes |
|---|---|---|---|---|
| GenDIReCT | 2025 | 2510.12093 | Latent diffusion + closure invariants | Gain-free; no explicit PSF model; PASA |
| Closure diffusion EHT (Cen A, 3C 279) | 2026 | 2602.21507 | Same as GenDIReCT | Applied to real EHT data |
| HIBI (Tiede et al.) | 2025 | 2511.17706 | Hierarchical Bayes + MRF + MCMC | Constrains M87* ring width at 9.3 ± 1.3 µas |

---

## 4. GAN / Generative Posterior Sampling (McEwen Group, UCL)

| Paper | Year | arXiv | Method | Notes |
|---|---|---|---|---|
| RI-GAN (Mars, Liaudat, Betcke, McEwen et al.) | 2025 | 2507.21270 | rcGAN + GU-Net with explicit measurement operator | Fast approximate posterior; robust to varying uv-coverage |
| Learned RI imaging for varying visibility (Mars, Betcke, McEwen) | 2024 | 2405.08958 | Unrolled + learned post-processing | Generalises across uv-coverages; PSF explicit; RASTI 2025 |
| QuantifAI (Liaudat, Mars, Price, Pereyra, Betcke, McEwen) | 2023 | 2312.00125 | Convex NN prior + MAP UQ | Scalable to SKA resolutions; calibrated credible intervals; RASTI 2024 |

---

## 5. AIRI / PnP Group (Wiaux Group, Heriot-Watt / Edinburgh)

| Paper | Year | arXiv | Method | Notes |
|---|---|---|---|---|
| AIRI on ASKAP (Terris, Tang, Jackson, Wiaux) | 2023 | 2302.14149 | PnP + DNN denoiser in proximal loop | 4× faster than uSARA; improved spectral index; MNRAS |
| AIRI variations and robustness (Terris et al.) | 2025 | 2312.07137 | Constrained variant (cAIRI) | Robustness to denoiser choice; Bayesian MAP connection; MNRAS |
| PnP-SARA → AIRI survey | 2022 | 2202.12959 | Comparison: handcrafted → learned denoisers | Establishes the lineage; MNRAS |

**Characterisation:**
AIRI is the most directly comparable published approach to a learned-prior solver slot in an
iterative deconvolution loop. Difference from MAD-CLEAN Variant C: AIRI uses a denoiser
(one forward pass, point estimate) rather than a generative flow prior (posterior distribution).
AIRI provides no per-pixel uncertainty estimates.

---

## 6. MAP-Based UQ Foundations (Cai / Pereyra / McEwen)

| Paper | Year | Notes |
|---|---|---|
| UQ for RI: I. Proximal MCMC; II. MAP estimation (Cai, Pereyra, McEwen) | 2018 | MNRAS 480. Theoretical backbone for QuantifAI. MAP-UQ ~10^5× faster than proximal MCMC. |

---

## 7. Flow Matching — Gap in the Literature

**As of March 2026, no published method applies Lipman-style flow matching (conditional flow
matching, CFM) specifically to radio interferometric imaging.**

The score-based / SDE works (IRIS, Dia et al. 2023) use the Song et al. SDE framework, which
is the diffusion-model side of the score-flow duality. The FlowDPS framework (Kim et al.,
ICCV 2025) addresses flow matching for general inverse problems but has no radio application.

This is an open gap.

---

## 8. MAD-CLEAN Variant C — Positioning

**What every existing method does:**
- Operates at **full-image scale** — global inference over the entire dirty image simultaneously
- No source decomposition prior to the generative step
- The learned component (denoiser, score model, flow) must handle the full 150×150 (or larger) image

**What MAD-CLEAN Variant C would do differently:**
- **Island-level decomposition first** — the IslandDetector isolates individual sources before
  any learned component is invoked
- **Per-source posterior** — the flow prior operates on an isolated source patch, not the full image
- **Explicit CLEAN outer loop** — global residual update and convergence remain in the
  explicit physics loop; the learned component handles only the per-source model estimation
- **Flow matching (Lipman et al.)** — not score-SDE; CFM-trained flow as the prior

This factorisation:
```
global problem  =  outer CLEAN loop (explicit physics)
                +  per-source posterior (flow prior + explicit Gaussian likelihood)
```
is not represented in the published literature as of March 2026. The closest is AIRI (which
uses a denoiser, not a posterior) and DDRM-radio (which uses the full image, not islands).

**Key claim this enables:** per-source uncertainty estimates (flux error, morphology confidence)
rather than per-pixel credible intervals over the full image. This is scientifically the right
granularity for a source catalogue.

---

## 9. Summary Table

| Method | Year | Neural prior type | PSF explicit | Scale | Posterior / point est. |
|---|---|---|---|---|---|
| RESOLVE / fast-resolve | 2018–2024 | IFT log-normal (hand-crafted) | Yes | Full image | Posterior (MGVI) |
| IRIS | 2025 | Score-SDE (learned) | Yes | Full image | Posterior (samples) |
| DDRM-radio | 2026 | DDPM (VLA FIRST trained) | Yes | Full image | Posterior (samples) |
| VIC-DDPM | 2023 | Conditional DDPM | Implicit | Full image | Point est. |
| GenDIReCT | 2025 | Latent diffusion | No (closure-based) | Full image | Samples |
| RI-GAN | 2025 | rcGAN | Yes | Full image | Approx. posterior |
| QuantifAI | 2023 | Convex NN prior | Yes | Full image | MAP + UQ |
| AIRI | 2023–2025 | DNN denoiser | Yes | Full image | Point est. (MAP) |
| **MAD-CLEAN Var. C** | **—** | **Flow matching (CFM, Lipman)** | **Yes** | **Island level** | **Per-source posterior** |
