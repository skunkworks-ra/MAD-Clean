# Code Audit — MDN-Asp Training Stack

Scope: the `sbi-asp` training and inference path as it stands at branch `sbi-asp`.
Prelude to the method paper. Findings are evidence-based with `file:line`
references; confidence percentages are given where a claim is a judgement call
rather than a fact in the code.

Files in scope:
- `scripts/train_mdn_asp.py` — training entry point
- `mad_clean/models/mdn_asp.py` — MDN model + losses
- `mad_clean/data/cutout_dataset.py` — on-the-fly training data
- `mad_clean/data/psf_bank.py` — PSF family loader
- `mad_clean/minor_cycle.py` — inference-time minor cycle (consistency anchor)
- `scripts/run_imaging.py` — inference driver

---

## 1. What is solid

- **Train/inference conditioning contract matches.** `sigma_local =
  1.4826·median(|residual|)` is computed identically at training
  (`cutout_dataset.py:262`) and inference (`minor_cycle.py:170`); `config_idx`
  flows through `make_cond` the same way in both. No silent train/serve skew on
  the conditioning vector.
- **Flux standardisation is symmetric.** `standardise_log_flux` /
  `unstandardise_log_flux` (`cutout_dataset.py:67-72`) are exact inverses and
  the constants (`LOG_FLUX_OFFSET=-5.76`, `LOG_FLUX_SCALE=3.45`) are used on both
  sides (`run_imaging.py:210`, `minor_cycle.py:199`).
- **PA ambiguity handled correctly.** Encoding to `(sin2θ, cos2θ)` and decoding
  via `atan2/2` (`mdn_asp.py:63-81`) collapses the θ ↔ θ+π degeneracy, which is
  the physically correct symmetry for an unsigned ellipse orientation.
- **SNR floor removes undetectable training examples** (`cutout_dataset.py:197-211`),
  matching the operational regime of the minor cycle (never fits below the stop
  threshold). This is a defensible, well-motivated design choice.
- **Checkpoint discipline is reasonable**: best-on-val tracked separately from
  periodic checkpoints and a final save (`train_mdn_asp.py:186-204`).

---

## 2. Findings

### 2.1 Conditioning — `sigma_local` is inert AND untrainable as-is (confidence ~85%)
Two compounding problems:

1. **Unnormalised scale.** `make_cond` feeds `sigma_local` **raw** (`mdn_asp.py:386-401`,
   docstring admits "caller is responsible"). Its magnitude is ~1e-4 (Jy/beam-scale
   MAD), while the config one-hot entries are order 1. Into the FiLM `Linear`
   (`mdn_asp.py:92`), a 1e-4 scalar contributes negligible gradient relative to the
   one-hot, so the network learns to ignore it.
2. **No training signal to learn from.** Training noise is **fixed** at
   `sigma_noise = 1e-4` (`cutout_dataset.py:146`). SNR varies in training only
   because flux spans 1e-4–1e-1 Jy (three decades), not because noise moves. So
   even a *well-scaled* `sigma_local` channel would have nothing to learn — the
   noise level never changes across scenes.

**What "posterior width adapts to noise" means physically.** The mixture spread in
each dim is the model's stated uncertainty. Two pieces have known scalings:
flux width is radiometric, `σ_flux ≈ σ_noise / √N_beam`; position width is the
centroiding bound, `σ_pos ≈ θ_beam / SNR`. A calibrated model should widen these as
local noise rises.

**Consequence.** The model *can* learn width-vs-SNR — but it learns it **from the
image** (the CNN sees a faint vs bright source directly), not from `sigma_local`.
It never learns width-vs-*noise-level*, because the noise level was constant. This
is harmless on the synthetic val set (same 1e-4 floor) but **bites at deployment**:
G55 real data sits at a different, spatially varying noise floor, the channel meant
to carry "what's the noise here" is dead, and absolute posterior widths were
calibrated at one specific floor. **Real-data uncertainty maps may be miscalibrated
in absolute Jy even if their relative shape is right.**

**Recommendation (two parts, order matters):** (a) vary `sigma_noise` across
training scenes so there is something to learn; (b) standardise `sigma_local`
(`log10` then z-score) before `make_cond`. (b) without (a) does nothing. Treat as a
pre-paper experiment if "noise-adaptive calibrated uncertainty" is a claim;
otherwise document as a known limitation and restrict uncertainty claims to the
relative/morphological sense. Directly load-bearing for the stage-3 uncertainty map.

### 2.2 Single-source target vs. multi-component model (confidence ~90% this needs stating)
Training uses `nll_loss` on a **single centred source** per cutout
(`train_mdn_asp.py:167`, `cutout_dataset.py` always emits one `centred_target`),
while the field contains uncontrolled distractors whose PSF sidelobes land in the
cutout (`cutout_dataset.py:215-218`). The K=5 mixture is therefore trained as a
**multi-modal posterior over one true source**, not as a set predictor.
**Impact:** this is a legitimate and arguably elegant design (mixture captures
positional/flux ambiguity of the one source under confusion), but the paper must
state it explicitly — a reader will otherwise assume K=5 means "up to 5 sources."

### 2.3 Dead code on the published path (confidence ~95%)
- `set_nll_loss` + Hungarian matching (`mdn_asp.py:284-347`) and `all_modes`
  (`mdn_asp.py:353-367`) are **not reached** by `train_mdn_asp.py` (which calls
  only `nll_loss`). They implement a set-prediction variant that the `sbi-asp`
  result does not use.
- The refit/shape-swap scaffolding (commit `1dee3f2`, parked) lives in
  `minor_cycle.py` second pass (`minor_cycle.py:270+`).
- `scripts/overfit_set_prediction.py` exercises the unused set path.
**RESOLVED (excised).** `set_nll_loss` + `all_modes` removed from `mdn_asp.py`;
`scripts/overfit_set_prediction.py` deleted; `refit_pass` removed from
`minor_cycle.py` along with the `--refit` flag and call block in `run_imaging.py`.
The shared crop helpers `_safe_crop`/`_crop_psf_centred` were retained (used by the
live `minor_cycle`). Published code now matches the method actually used.

### 2.4 Architecture args hard-coded at inference — RESOLVED
A single safe loader, `load_mdn_checkpoint` (`mdn_asp.py`), now serves both
`run_imaging.py` and `eval_coverage.py`: safetensors dirs load via
`from_pretrained` (arch from `config.json`); legacy `.pt` loads with
`weights_only=True` and reads arch from a sibling `run_config.json` when present,
falling back to constructor defaults rather than blind hard-coding. The
`weights_only=False` pickle loads are gone from both scripts.

### 2.5 Released model trained on point+blob only — OOD on the science target (confidence ~95%, CONFIRMED)
**Confirmed:** the released `best.pt` (50k steps) was trained with
`morphologies=point,blob`. Evidence — the captured training banner reads
`[train] Training for 50000 steps, batch=64, device=cuda, morphologies=point,blob`
(recovered from a Claude paste-cache of the training session; not from any
persisted run log — see 2.6). The dataset supports `shell` and `filament`
(`cutout_dataset.py:107`) but they were **not** in the released run.
**Impact (significant):** the SNR G55 target is an extended **shell** supernova
remnant. The released model has never seen a shell or filament centred source, so
on the paper's headline science case it is **extrapolating off-distribution**.
**Recommendations:**
- State this plainly in the paper; do not let "handles extended emission" stand
  unqualified.
- Strongly consider a retrain with `morphologies=point,blob,shell,filament` before
  final G55 results, and compare — this is the single highest-leverage experiment
  surfaced by the audit.
- Persist the morphology set in the checkpoint metadata going forward (2.6).

### 2.6 Reproducibility gaps (confidence high)
- Seeds are set (`train_mdn_asp.py:103-104`) but `DataLoader` workers are not
  seeded per-worker; with `num_workers>0` the per-sample RNG is `rng_seed+idx`
  inside `__getitem__` (`cutout_dataset.py:168`), which *is* deterministic in idx,
  so this is actually fine — worth a one-line note confirming it.
- The training **command and hyperparameters are not persisted** alongside the
  checkpoint. `log.json` stores losses only (`train_mdn_asp.py:199-206`).
  **Recommendation:** dump `vars(args)` to `out_dir/config.json` at startup so
  every checkpoint is self-describing. (The HF `save_pretrained` config covers
  architecture but not data/optim hyperparameters.)

### 2.7 Test coverage of the trained path
The MDN unit tests (`tests/test_mdn_asp.py`) and overfit smoke
(`tests/test_overfit_smoke.py`) pass (15/15). However the legacy Variant A/B
suite is **broken on this branch** (10 failures): `test_training.py` calls
`PatchDictTrainer(n_iter=...)` but the constructor takes `n_epochs`
(`patch_dict.py:59`) — a pre-existing test/source contract mismatch dating to the
initial commit, independent of the Python 3.12 env bump; and `test_deconvolver`
fails on `F.unfold` over a 1×1 island. These are in the classical sparse-coding
path the paper does not use, but a release advertising "50 passed" must either fix
or quarantine them. **Recommendation:** mark the Variant A/B tests `xfail` with a
reason, or repair the `n_iter`→`n_epochs` rename, before tagging.

---

## 3. Summary table

| # | Finding | Severity | Confidence | Action |
|---|---------|----------|-----------|--------|
| 2.1 | `sigma_local` inert + no noise variation to learn from | Medium | 85% | Vary noise + standardise + ablate (V31) |
| 2.2 | Single-source target, K-mixture posterior | Doc | 90% | State explicitly in paper |
| 2.3 | Dead set-prediction / refit code | Low | 95% | ✅ RESOLVED — excised (H4) |
| 2.4 | Inference arch hard-coded (legacy .pt) | Low | high | ✅ RESOLVED — `load_mdn_checkpoint` (H2) |
| 2.5 | **Released model point+blob only — OOD on G55 shell** | **High** | 95% (confirmed) | Retrain w/ shell+filament (v3); qualify claims |
| 2.6 | Training hyperparams not persisted | Medium | high | ✅ RESOLVED — `run_config.json` (H1) |
| 2.7 | Legacy Variant A/B tests broken | Low | high | ✅ RESOLVED — xfail quarantine (H3) |

---

## 4. Open questions for Preshanth

1. Keep or excise the set-prediction path for the released code? (2.3)
2. ~~Which `--morphologies` did the released `best.pt` train on?~~ **Resolved:
   point+blob only (2.5).** **Decision: qualify the extended-emission claims for
   the current paper; add shell+filament in a later retrain.**
3. Is adaptive-to-noise posterior width a claim the paper wants to make? If yes,
   2.1 becomes load-bearing — requires *both* varying training noise and
   standardising `sigma_local` before final runs.

---

## 5. What the uncertainty means — three layers

The method must be explicit about *which* uncertainty it claims. It natively gives
Layer 1, derives Layer 2, and cannot give Layer 3.

**Layer 1 — parameter posterior (emitted).** `p(θ | residual, σ_local, config)`
over the 6D Aspen `(x, y, log_flux, log_sig_maj, log_sig_min, PA)` per committed
source. Primitive, per-component, per-parameter ("flux 5±1 mJy, position ±0.3 px,
PA ±20°"). *Calibrated* = across cases where the model states ±1σ, truth lands
inside 68% of the time. Directly testable; this is what `scripts/eval_coverage.py`
measures (68/95% empirical coverage per dim). **This is the layer where a
"calibrated uncertainty" claim is clean — on data matching training.**

**Layer 2 — image pushforward (the uncertainty map; stage 3).** Derived: push the
Layer-1 posterior through the rendering operator into Jy/pixel. Per-pixel σ is a
*function* of the parameter posterior, not a network output. Calibration here is
strictly weaker and messier; current code distorts it two ways (flux-only collapse,
`max`-combine across components — see stage-3 plan). Stage 3 fixes the propagation;
it cannot add information absent from Layer 1.

**Layer 3 — true-sky brightness uncertainty (NOT provided).** Layer-1 width is
*conditional on the source being drawn from the trained morphology family*. It
captures aleatoric (thermal noise) + parametric epistemic (which point/blob fits)
uncertainty *inside that family*. It does NOT capture **model misspecification**.
Feed an OOD shell (exactly G55, finding 2.5) and it returns a confident, wrong
posterior — narrow bars around the wrong morphology. **The uncertainty map looks
most trustworthy exactly where the model is most wrong.** This is the referee-bait
failure mode and must be disclaimed.

### Paper framing (recommended)
Claim Layer-1 calibration (provable via `eval_coverage.py`); present the Layer-2
map as a *propagated* uncertainty with stated caveats; explicitly disclaim Layer 3.
Do not let the uncertainty image imply Layer 3.

Three qualifiers ride along:
1. Calibrated **in-distribution** only (Layer 1).
2. **Not** a true-sky uncertainty — omits model misspecification (OOD-shell blind spot).
3. **Noise-adaptive** is currently aspirational, not delivered — training noise was
   fixed at 1e-4, so absolute widths are calibrated at one floor only (finding 2.1).

### Evidence to collect
Run `eval_coverage.py` on `results/mdn_asp_v2/best.pt` (point+blob, in-distribution)
and record the 68/95% coverage table — that is the Layer-1 calibration evidence.
