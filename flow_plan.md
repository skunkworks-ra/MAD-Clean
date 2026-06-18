# MAD-Clean plan — Aspen-posterior SBI minor cycle

**Status:** design replacing prior plans, 2026-05-27.

## Progress log (sbi-asp branch)

### 2026-05-27 — scaffolding + diagnostic 1

Built on branch `sbi-asp` off `main`, pushed to
`skunkworks-ra/MAD-Clean:sbi-asp` in three commits:

- **Data pipeline.** `mad_clean/data/{psf_bank, point_sky, extended_sky,
  cutout_dataset}.py`. PSF bank loads 10 G55 D-config L-band chunks +
  rotation augmentation → 40 PSFs. `assemble_mixed_field` returns
  optional per-source rendered images so cutouts use as-rendered
  morphology (real shell / filament shapes) and not the Gaussian fit.
  `CutoutDataset` emits `(residual, psf, conditioning, target)` quadruples
  at 128² with log_flux standardised to ~unit range.
- **MDN.** `mad_clean/models/mdn_asp.py` ported from radiosharp. K=5
  components, 7 emitted dims per component, FiLM conditioning on
  `(sigma_local, config_one_hot)`, PSF as input channel (not in
  conditioning, per Lesson 2).
- **Diagnostic 1.** `scripts/overfit_one_batch.py`. 8 frozen examples,
  2000 Adam steps, three PNGs (loss curve, target-vs-mode scatter,
  residual cutout grid with target/mode markers).

What diagnostic 1 showed:

- **Default 5%-extended batch:** all six dims at floor for the seven
  point sources; the one extended source had `log_sig` stuck at the
  beam floor — network collapsed to "always point source" because 7/8
  examples were points.
- **`--extended_fraction 1.0` batch:** all six dims at floor across the
  full batch. Scale recovered cleanly over a factor-of-5 dynamic range,
  flux over factor-of-250, position to sub-pixel. PA wraps cleanly at
  the `θ ↔ θ + π` degeneracy as the `(sin 2θ, cos 2θ)` encoding
  predicts.

What this tells us:

- Architecture, loss, and target encoding wire correctly. Gradient flow
  on the full 6D space is real. Diagnostic 1's first job (rule out
  architectural breakage) is satisfied.
- Per-sample loss balance matters — class imbalance silently produces
  collapse-to-dominant-class. Full training will need an explicit
  balancing knob.
- MDN loss has periodic spike behaviour (component variance hits floor,
  recovers). Not blocking at this scale; worth watching.
- A small bug found in the script's PA error metric (gated form left
  some near-π distances unwrapped); fixed and pinned in
  `tests/test_integration_dataset_mdn.py`. The training loss itself was
  correct — only the reporting was off.

Test surface: 69 tests across psf_bank, point_sky, extended_sky,
cutout_dataset, mdn_asp, overfit smoke, integration. All pass on CPU in
under 2 s.

### 2026-05-30 — ground-truth status check (no new runs)

Re-derived the actual state of the branch from `results/` and the code,
because the wiring state had been forgotten over a few days. Recorded
here so the next session starts from fact, not memory.

**Epistemic ledger — what is and is not proven:**

- **Single-cutout regression capacity: proven.**
  `results/overfit_extended_only/summary.json` — 8 all-extended frozen
  samples, every dim (x, y, log_flux, log_sig_maj/min, PA) at ~1e-3.
  The K=5 mixture head can represent an extended source's posterior.
  `overfit_one_batch` shows the 1-in-8 extended sample pinned at the
  ±2.0 clamp: that is class-imbalance collapse (already diagnosed),
  not a capacity wall.
- **Single-cutout generalization (held-out): NOT run.** No held-out
  evaluation exists anywhere in `results/`. Every MDN run on disk is an
  overfit diagnostic (8 frozen samples, 2000 steps).
- **Iterative loop convergence: NOT run, and NOT wired.** `mad_clean/`
  on this branch is only `data/` + `models/mdn_asp.py`. There is no
  `minor_cycle()`, no `render_aspen()`, no deconvolver. The loop is
  still only the signature in this doc. The old `MADClean`/`solvers.py`
  are not on this branch.
- **One-shot set prediction: failing.** The untracked
  `scripts/overfit_set_prediction.py` cannot even overfit 8 samples:
  position errors 17–42 px, `final_loss` diverged back to +2.86 after
  touching −6.19, and matched-component weights pinned flat at ~1/K
  (0.15–0.26) — components never break symmetry. This is a DETR-style
  set-matching collapse, a property of the v2 set-prediction loss, NOT
  a verdict on the K=5 mixture (the iterative head overfits extended
  fine right next to it). Set prediction is the deferred v2 ambition;
  its failure is not evidence against the v1 iterative thesis.

**Decision framing.** The ditch question is not "did K=5 fail" — it
failed at set prediction (v2), the wrong proxy. The real question is
whether iterative convergence holds, and that has never been tested.
Two unrun gates are stacked: (1) cutout-MDN generalization, then
(2) loop convergence on top of it. Gate (2) cannot start until
`render_aspen` + the `minor_cycle` loop are built and one real
(non-overfit) training run exists. That is the renderer + the loop +
a training run — not a one-script afternoon. It is also the first
artifact on this branch that tests the actual thesis instead of a proxy.

**The convergence test, once the pieces exist.** One controlled
synthetic field (a few points + one shell), load the trained iterative
MDN, run the real `minor_cycle` loop, plot residual RMS per iteration.
Watch whether the shell assembles from cumulative Gaussian commits and
whether the residual descends monotonically to the sidelobe floor.
Most likely disappointment mode is per-call calibration drifting across
the chain (an open question in this doc) — diagnosable, not fatal.

### Next

Move from the frozen-batch diagnostic to a full training run. Sampling
strategy (extended oversampling vs class-balanced loss), epoch length,
checkpoint cadence, and where to call diagnostics 2 and 3 from are
open and to be designed before implementation. Note the dependency: the
loop-convergence gate above needs `render_aspen` + `minor_cycle` built
and a generalizing (not overfit) checkpoint before it can run.

## One-line deliverable

A Python library that exposes `minor_cycle(residual, psf, conditioning) →
(model_update, aspen_catalogue)`, where each Aspen carries a calibrated
posterior over its 6D shape parameters. Any major-cycle host (LibRA,
tclean, a notebook) can call it. The destination is a calibrated-
uncertainty replacement for the WAsp minor cycle that feeds VROOM-SBI
downstream for spectral modelling.

## Why this shape

Three constraints decide the architecture together:

- **Per-source posteriors with calibrated uncertainty.** The deliverable
  is `(flux ± σ, morphology ± σ)` per Aspen, validated by SBC. Not a
  per-pixel image with uncertainty maps glued on.
- **Image-domain.** VROOM-SBI does spectral SBI on per-source flux series
  downstream; MAD-Clean has to produce Aspen-with-posteriors that
  VROOM-SBI can consume. Visibility-domain SBI is out of scope.
- **Independent of the major cycle.** LibRA, CASA, anyone's gridder does
  the vis-domain residual recompute. We do not own that boundary. Our
  call signature is image-domain in and image-domain out.

These three together rule out per-pixel image-regression flows (the
prior C path: not per-source, posterior-collapse pathology) and rule out
visibility-domain methods (radiosharp's approach: not VROOM-SBI-shaped,
not minor-cycle-shaped).

## Boundary — what we own, what we don't

**We own.** The minor cycle. Given a residual image and a PSF, decide
where Aspen go, fit their parameters with posteriors, accumulate them
into a model image and a per-Aspen catalogue.

**We do not own.** Vis-domain residual recompute, gridding, w-projection,
primary beam correction, joint spatio-frequency normal-equation solves
(MS-MFS), restoration. The host (LibRA / CASA) does these. They hand us a
new residual when the previous minor cycle is done; we hand them back a
model update.

**Interface:**

```python
def minor_cycle(
    residual: NDArray,            # (H, W) float32, Jy/beam
    psf:      NDArray,            # (H, W) float32, peak=1
    sigma:    float,              # noise estimate, Jy/beam
    config:   str,                # 'A' | 'B' | 'C' | 'D' (VLA configuration)
    loop_gain: float = 0.1,
    stop:      StopRule = ...,    # N-sigma threshold + sidelobe-floor cap
    max_aspen: int = 1000,
) -> tuple[NDArray, list[Aspen]]:
    """Returns (model_update_image, list of Aspen with posteriors)."""
```

Each `Aspen` carries: location `(x, y)`, scale `(σ_maj, σ_minor)`,
position angle `PA`, integrated flux, plus per-parameter mean and
covariance from the MDN posterior.

## Component representation — 6D framework, 4D specialisation

Each Aspen is parameterised by 6 numbers:

`(x, y, log_flux, log_σ_maj, log_σ_minor, PA)`

PA is encoded inside the network as `(sin 2θ, cos 2θ)` and decoded back
to a single PA at sampling. The factor of 2 collapses the
`PA ↔ PA + π` ambiguity.

The 4D WAsp-compatible Aspen is the constraint `σ_maj = σ_minor` (PA
becomes degenerate; the posterior naturally widens on PA when this holds).
This is *not* a separate code path — it is a runtime constraint on the
MDN output. We can run apples-to-apples against WAsp on G55 by sampling
under the constraint without forking the implementation.

## Inner loop structure (iterative MDN replacing ALGLIB)

WAsp's expensive step is the per-Aspen non-linear optimisation by ALGLIB
(Appendix A.3 of Hsieh et al. 2026). We replace that with a single MDN
forward pass per Aspen. The rest of the inner loop is our own design,
not inherited from WAsp.

```
minor_cycle(residual, psf, σ, config, ...):
    model_update = zeros_like(residual)
    aspen_list = []
    repeat:
        # 1. Locate next candidate
        peak_pos = argmax(|residual|)
        peak_val = residual[peak_pos]

        # 2. Stopping check  (any-of)
        if |peak_val| < N · σ:                      break  # noise floor
        if |peak_val| < sidelobe_level · model_peak: break  # PSF floor
        if len(aspen_list) >= max_aspen:             break

        # 3. Extract ROI cutout (fixed size; see ROI section)
        residual_cutout = crop(residual, peak_pos, ROI)
        psf_cutout      = crop(psf, peak_centred, ROI)
        sigma_local     = estimate from cutout

        # 4. MDN forward pass — posterior over 6D Aspen parameters
        posterior = MDN(residual_cutout, psf_cutout, [sigma_local, config])
        aspen     = posterior.mode()

        # 5. Commit
        commit  = aspen × loop_gain
        psf_resp = render_aspen(commit, psf)        # B · P(p)
        residual     -= psf_resp                    # eq. (7) of WAsp
        model_update += render_aspen(commit, delta) # eq. (6) of WAsp
        aspen_list.append((commit, posterior))      # carry posterior, not just mode

    return model_update, aspen_list
```

The `model_update` returned is the image-domain delta to be added to the
host's running model. The posteriors carry per-parameter covariance for
downstream propagation.

### One-shot variant (v2)

Once the iterative path is calibrated, the v2 ambition is replacing the
loop with a single network forward pass over the whole residual+PSF that
emits a variable-length Aspen list with posteriors. DETR-style set
prediction. Cheaper at inference. Deferred — iterative is the falsifier.

## ROI size

Fixed cutout, not scale-adaptive. WAsp itself does not vary the cutout
size — it optimises `σ` continuously over the full residual; we are not
losing anything by fixing the network's input window.

**128 × 128 px** with the cap `σ ≤ 8W` (≈ 11 px for D-config G55 L-band).
This matches the full WAsp default scale envelope and gives ±4σ
containment at the largest expected Aspen size. Picked from the start
rather than starting smaller and widening, so v1 does not need a
ROI-size revision after the first training run.

## Stopping rule

Stop on first-of:

- `|residual peak| < N · σ̂` (noise floor; N from the host)
- `|residual peak| < sidelobe_level · max_committed_flux` (PSF-error
  floor; `sidelobe_level` is the worst negative sidelobe of the dirty
  beam, ~0.1–0.3 for VLA D-config L-band)
- `len(aspen_list) >= max_aspen`

The second criterion is the load-bearing one for the major/minor
boundary: when residual signal is at the level of sidelobes from
committed components, the remaining error is by definition PSF-attributable
and the host's next vis-domain recompute should resolve it.

## Training

**Sample format.** `(residual_cutout, psf_cutout, conditioning, target)`
quadruples. Replaces the whole-field `(dirty, clean, psf)` triples used
by the prior flow trainer.

**Cutouts.** 128 × 128. The residual cutout is the dirty image minus the
PSF response of all other in-field sources (so the cutout looks like
what a real residual looks like late in a minor cycle, not like a clean
dirty image).

**Sky generator.** Each training scene has:

- Point sources from a log N–log S power law (existing
  `mad_clean/data/point_sky.py`).
- A 5 % per-source rate of extended sources drawn uniformly from:
  - **Anisotropic Gaussian blob** — `(σ_maj, σ_minor, PA)` free.
  - **Limb-brightened shell** — radius and shell thickness free,
    circular for v1.
  - **Filament** — line segment with Gaussian cross-section,
    `(length, width, PA)` free.

Disks and core+lobe templates deferred to v1.1.

**Conditioning vector.** `(sigma_local, config_one_hot)`. FiLM into the
MLP head, identical to radiosharp's pattern. Band / elevation / frequency
deferred — covered v1.1 if cross-band evaluation shows residual structure.

**PSF.** v1 trains on the G55 D-config L-band PSF bank (the nine cached
chunks + rotation augmentation). v1.1 widens to a cross-configuration
bank.

**Target preparation.** Each extended cutout is fit at simulator time
with an anisotropic Gaussian to produce `(σ_maj, σ_minor, PA)` targets.
For points, `σ_maj = σ_minor = floor`, PA arbitrary (the network is
trained with randomised PA on symmetric truths so the PA posterior
correctly widens). Shells fit to circular blobs with `σ ≈ radius`;
filaments fit to elongated Gaussians.

## Sanity checks (diagnostics, not pass/fail)

Three diagnostics we run at each step to know whether we are looking at
the right kind of evidence. None of them is a pass/fail gate — they are
ways of disambiguating "the architecture is broken" from "the data are
not informative enough" from "the network is undertrained." Calibration
in particular is a continuum, not a checkbox; SBC results need looking
at in context, not against a single threshold.

1. **Overfit one batch.** Freeze 8 (cutout, true) examples; train MDN to
   NLL floor. Predicted mode hitting each truth within sub-pixel says the
   loss and architecture are wired right. Overfit *not* converging is
   informative; overfit converging tells us nothing about the full data
   distribution (lesson 5 — `psf_flow_sanity1` overfit converged, full
   training collapsed).
2. **Coverage on held-out cutouts.** What fraction of truths sit inside
   the 68 % and 95 % credible regions? Approximate calibration in the
   neighborhood of those numbers, on the marginals and on the joint,
   is what we are looking for. Persistent under-coverage means
   miscalibrated posteriors; persistent over-coverage means the network
   has hedged its variance head into the floor.
3. **Posterior shape spot-check.** Look at the marginals on the four
   cases that *should* differ: isolated bright point (tight, scale at
   floor); isolated extended shell (scale marginal away from floor);
   faint near detection (broader); two-equal-flux blend (`(x, y)`
   bimodal). If the marginals all look the same we have learned a
   regression network with a variance head, not a posterior.

These three together are not sufficient for a research conclusion; they
are sufficient to know we are looking at a working artifact and can
start asking the actual scientific questions.

## Variants A, B, C — the diagnostic chain

Not baselines. Each variant taught us something the next inherits.

| Rung | What it showed | What the next rung inherits |
|---|---|---|
| **A** PatchSolver (OMP + sklearn dictionary, 15-px atoms) | A sparse learned basis represents the sky cleanly on patches when the PSF is absent. Establishes that *learnability* is real. | Continuous-parameter generalisation: the MDN's 6D posterior subsumes A's discrete dictionary. |
| **B** MCASolver (FISTA, m_c + D⊛z, explicit PSF) | A convex learned prior reaches the noise floor on synthetic 512² but paints sidelobes on real G55 (2026-05-17). Tells us that L1-with-explicit-PSF on a learned basis is *not* enough on real data; the prior over-explains sidelobes. | The need for a scale-sensitive component representation. This is exactly what WAsp Section 3.1.2 solves classically; the MDN-Asp solves it with SBI calibration on top. |
| **C** PSFFlowModel (conditional flow matching, per-pixel NLL) | Per-pixel heteroscedastic NLL on image regression collapses on sparse + structured targets (the `psf_flow_sanity1` / `nll_tightvar` runs, 2026-05-26). Tells us that whole-image SBI as regression is wrong-shaped. | The right shape: per-Aspen posteriors emitted iteratively, not per-pixel velocity fields. |
| **MDN-Asp** (this plan) | TBD | — |

The original variants stay in tree. A and B are not deprecated — they
are the rungs above which the MDN-Asp result must demonstrate gain. C is
parked code: kept as reference for what does not work, not retrained.

## WAsp as the existing classical instance

Hsieh, Bhatnagar, Rau 2026 (arXiv:2604.22691). WAsp is the existing
classical algorithm in the niche this project enters: a scale-sensitive
minor cycle for wide-band imaging, runnable on G55 D-config L-band.

We are not trying to beat WAsp on its own metrics. We are a different
point on the trade curve, with these claims:

- **Calibrated per-Aspen uncertainty.** WAsp emits a deterministic best-fit
  Aspen per iteration; we emit a posterior. This is the load-bearing
  difference and it is what makes the output composable with VROOM-SBI.
- **Amortised inference.** WAsp's per-Aspen ALGLIB non-linear optimisation
  (their Appendix A.3) is replaced by a single MDN forward pass. The
  variability they describe across ALGLIB builds and operating systems
  vanishes by construction.
- **Richer per-component representation.** Their 3-parameter symmetric
  Aspen vs our 6-parameter anisotropic Aspen. Their cumulative-components
  argument still applies to us, but on top of a more expressive primitive.
- **Fewer hand-tuned heuristics.** No initial scale set, no `largestscale`
  override (their Section 3.1.1), no fused-Hogbom switch (their Section
  3.1.3), no Aspen-amplitude normalisation method (their Appendix A.1).
  The trained MDN absorbs these into one model.

Acceptable approximations vs WAsp:

- Lower fidelity per iteration is fine if the posterior is calibrated.
- More iterations to convergence is fine if each iteration is cheaper.
- Worse residual depth is fine if the residual is sidelobe-dominated at
  termination (the host's next major cycle resolves it).

We do not inherit their loop structure. Their fused-Hogbom switch, their
initial-scale-set peak-finding, and their permanent-vs-active Aspen
bookkeeping are their solutions to their constraints, not ours.

## Lessons learned

These are paid-for and need not be re-learned.

1. **Per-pixel heteroscedastic NLL collapses on sparse + structured
   targets.** Variance head provides a free escape via `log_var → floor`.
   Confirmed `psf_flow_sanity1` (loss → −4.66) and `nll_tightvar` (loss
   → −0.9996 against `[-2, 2]` clamp).
2. **PSF as a conditioning input fails OOD silently.** Put the PSF in
   the cutout channel.
3. **Image-domain SBI as regression conflates detection, deconvolution,
   and morphology in one loss.** The Asp framing separates them
   cleanly: peak-find → posterior over one component → commit.
4. **Sparse-dictionary solvers paint sidelobes under L1.** The WAsp
   scale-sensitive single-Aspen-per-iteration argument addresses this
   classically; SBI-on-Aspen addresses it with calibration.
5. **Overfit on one sample does not predict full-dataset performance.**
   Three-gate sanity (overfit + coverage + bimodality) is the standard.
6. **Don't duplicate radiosharp.** Their MDN design, training infra,
   and sanity protocol are settled. Port, swap the conditioning vector
   for our regime, do not re-explore their option space.

## Deferred

- **One-shot Aspen-list emission.** v2 after iterative is calibrated.
- **VROOM-SBI handoff schema.** Match the interface when the spectral
  side becomes the focus.
- **Cross-VLA-configuration training.** v1.1 once the single-config v1
  passes the three-gate.
- **Scale-adaptive ROI.** v2 if v1.1 fixed ROI clips real structure.
- **Active-set Aspen refit.** v2. Justification: cumulative iterations
  build morphology, single-shot does not need to refit old components
  (radiosharp v1 argument).
- **LibRA integration.** Once the minor-cycle library passes the
  three-gate, expose it as a callable from LibRA. Not before.
- **Vis-domain residual recompute, gridding, primary-beam correction.**
  Out of scope by design; the host owns these.

## Out of scope

- Visibility-domain SBI (radiosharp's regime).
- Score-based diffusion priors, latent DPS, VAE priors.
- One-shot full-image deconvolution.
- PSF-agnostic priors trained on clean sky only with explicit `H` at
  inference (the Adler-Öktem path).

These are reachable from the current architecture if the iterative MDN-Asp
falls short of the gates; they are not the v1 plan.

## Exploration steps (rough order, not a Gantt)

A research direction, not a deliverable schedule. Each step is "what we
do next if the previous step looks like it is working." Each is expected
to surface its own surprises that may rearrange the steps after it.

- **Sky generator + 128² cutout pipeline + MDN port from radiosharp**,
  conditioning vector swapped, on points only. Goal at this step is to
  *see the gradient flow*, not declare anything.
- **Three extended templates added** (Gaussian blob, shell, filament).
  Look at how the network handles the scale/PA marginals on extended
  cases. The interesting failure modes likely live here.
- **Full training run on G55 D-config L-band PSF bank.** Run the three
  sanity diagnostics and *look* at the results — calibration plots,
  marginal shapes, where the network is and isn't confident.
- **Wire MDN into the `minor_cycle()` callable.** Run on G55 chunks.
  What we are likely to discover: how Aspen accumulation interacts with
  posterior calibration after several iterations (the posteriors are
  calibrated per call, not per chain).
- **Cross-VLA-configuration training** (mixed PSF bank, config
  conditioning live). The honest test of whether PSF-as-input-channel
  generalises. May reveal that the conditioning vector needs
  enrichment.
- **One-shot Aspen-list variant.** Long-term ambition. Will require a
  set-prediction loss and may benefit from the iterative version as a
  pre-training signal.

The point of writing the order down is to keep us from drifting
mid-step; the order is rearrangeable when a step's results disagree
with the assumption that put it there.

## Open questions

We will likely learn the right answer to most of these only by doing the
work. Listing them so they are not forgotten, not so they are decided
in advance.

- **Shell representation.** Filling a ring with cumulative symmetric or
  anisotropic Gaussians may or may not produce a posterior that says
  "this is a ring" in a useful way. A template-id channel (Gaussian /
  shell / filament, 7D output) is one alternative; learning a shell as
  a coherent draw of an annular component is another. Decide after
  looking at real failure modes, not in advance.
- **PA encoding and degeneracy.** `(sin 2θ, cos 2θ)` collapses the
  `PA ↔ PA + π` ambiguity but leaves the σ_maj = σ_minor degeneracy
  (symmetric truth → arbitrary PA) for the variance head to express.
  Whether the K=5 mixture handles this gracefully is empirical.
- **Loop gain.** WAsp runs at 0.4–0.6 stably; CLEAN tradition is 0.1.
  Where our posteriors sit on this trade-off is unknown until we run
  through several iterations and look at how the residual evolves.
- **Per-call vs per-chain calibration.** SBC validates each MDN call
  individually. After many committed Aspen, the chain's joint posterior
  may drift from per-call calibration. How to characterise and fix that
  is open.
- **Conditioning vector sufficiency.** `(sigma_local, config_one_hot)`
  may or may not be enough for PSF-channel generalisation across
  configurations. Likely enrichment paths if not: band, elevation,
  sidelobe-level summary statistics.
- **Where the iterative-to-one-shot tradeoff actually breaks.** One-shot
  is faster in principle but may sacrifice calibration in the source-
  count direction; iterative carries detection bias through the chain.
  Neither limit is theoretically derivable.

Many of these have provisional answers in the doc above. None of those
should bind us if the work disagrees.
