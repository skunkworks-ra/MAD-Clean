# MAD-CLEAN Implementation Plan

**Last updated:** 2026-03-30
**Context note:** Read `design.md` first, then `bayesian_imaging.md` for literature context, then this document.

---

## Variant C — Flow Matching Solver (approved 2026-03-30)

See `bayesian_imaging.md` for literature positioning. Key novelty: island-level flow matching prior — no published method does this. No published method applies Lipman et al. CFM to radio imaging.

### New files

**`train/flow_dict.py`** — three classes:

- **`UNetVelocityField(nn.Module)`**: lightweight U-Net (~2–4M params), input `(B,1,150,150)` + time embedding, output `(B,1,150,150)` velocity field. Pure PyTorch, no new dependencies.
- **`FlowModel`**: thin wrapper. `save(path)` → `.pt` state dict. `load(path, device)`. `velocity(x, t) → Tensor`.
- **`FlowTrainer`**: `fit(images: np.ndarray, device) → FlowModel`. Per-image normalisation (zero mean, unit std, consistent with ConvDictTrainer). 8-fold on-the-fly augmentation (4 rotations + flip). CFM loss with OT paths: `loss = ||v_θ(xt,t) - (x1-x0)||²`. Adam lr=1e-4.

### Modified files

**`solvers.py`** — add `FlowSolver`:
- `__init__(flow_model, psf, device, n_samples=8, n_steps=16, guidance_scale=1.0)`
- PSF passed at construction; `psf_fft` precomputed internally
- `decode_island(island) → Tensor` — posterior mean, backward compatible with MADClean
- `decode_island_with_uncertainty(island) → (mean, std)` — both `(H_i, W_i)`
- Island zero-padded to 150×150, cropped back per sample
- Guided Euler ODE (t: 1→0): `x1hat = xt + (1-t)*v`; guidance = `PSF^T(y - PSF⊛x1hat)/σ²`; σ from `island.std()`

**`deconvolver.py`** — two minimal changes:
1. Duck-typed uncertainty accumulation: `hasattr(solver, 'decode_island_with_uncertainty')`; adds `"uncertainty": ndarray|None` to return dict
2. Variant label: `getattr(solver, '_variant_label', cls.__name__)`; FlowSolver sets `_variant_label = "C/Flow"`

**`scripts/run_train.py`** — `--variant C` branch

**`__init__.py`** — export `FlowSolver`, `FlowTrainer`, `FlowModel`

### What does NOT change
`MADClean.__init__` signature, `IslandDetector`, `FilterBank`, `PatchSolver`, `ConvSolver`, outer CLEAN loop.

### Build sequence
1. `UNetVelocityField` → `FlowModel` → `FlowTrainer` (in `train/flow_dict.py`)
2. `FlowSolver` (in `solvers.py`)
3. `deconvolver.py` patch
4. `scripts/run_train.py` variant C branch
5. `__init__.py` exports

### Verification
```bash
# Smoke-test training
python scripts/run_train.py --variant C \
    --data crumb_data/crumb_preprocessed.npz \
    --out models/flow_model.pt --n_epochs 2 --batch_size 4

# Velocity field shape check
python -c "
from train.flow_dict import FlowModel; import torch
fm = FlowModel.load('models/flow_model.pt', device='cpu')
v = fm.velocity(torch.randn(1,1,150,150), torch.tensor([0.5]))
print(v.shape)  # torch.Size([1, 1, 150, 150])
"

# FlowSolver uncertainty check
python -c "
import torch, numpy as np
from train.flow_dict import FlowModel; from solvers import FlowSolver
fm = FlowModel.load('models/flow_model.pt', device='cpu')
psf = np.zeros((150,150), dtype=np.float32); psf[75,75] = 1.0
s = FlowSolver(fm, psf, 'cpu', n_samples=2, n_steps=4)
mean, std = s.decode_island_with_uncertainty(torch.randn(40,35))
assert mean.shape == (40,35) and std.shape == (40,35)
print('OK')
"
```

---

## TODO — Variant C

- [x] **C-1** `UNetVelocityField` — encoder/bottleneck/decoder with skip connections, sinusoidal time embedding
- [x] **C-2** `FlowModel` — wrapper with `save`/`load`/`velocity`
- [x] **C-3** `FlowTrainer.fit()` — CFM training loop with 8-fold augmentation
- [x] **C-4** `FlowSolver.__init__` — PSF FFT precomputation, hyperparams
- [x] **C-5** `FlowSolver.decode_island_with_uncertainty` — guided Euler ODE, pad/crop, n_samples
- [x] **C-6** `FlowSolver.decode_island` — calls `decode_island_with_uncertainty`, returns mean
- [x] **C-7** `deconvolver.py` — uncertainty accumulation + variant label
- [x] **C-8** `scripts/run_train.py` — `--variant C` branch
- [x] **C-9** `__init__.py` — exports
- [x] **C-10** Smoke test training (2 epochs, GPU) — loss decreased 0.127 → 0.074
- [x] **C-11** Smoke test FlowSolver — shape assertions pass
- [x] **C-12** Full deconvolution run — `result['uncertainty']` is not None

---

## Variant C — Refactor: Conditional Flow Matching (approved 2026-03-30)

**Problem with initial implementation:** FlowTrainer trained on noise→clean (x_0 ~ N(0,I)).
FlowSolver required a PSF internally for data-consistency guidance. This is architecturally
inconsistent with Variants A/B where the PSF lives entirely in the outer CLEAN loop.

**Solution:** Conditional flow matching — train dirty→clean directly.
- x_0 = dirty island, x_1 = clean island (naturally paired, no OT needed)
- Velocity field v_θ learns the mapping dirty→clean
- FlowSolver: Euler ODE starts from dirty island, no PSF required
- Uncertainty: perturb x_0 = dirty + ε·N(0,I) per sample

### Unified data pipeline

```
crumb_preprocessed.npz  +  PSF (FITS/npy or --psf_fwhm Gaussian)
         ↓
scripts/simulate_observations.py
         ↓
flow_pairs.npz  { clean: (N,H,W), dirty: (N,H,W), psf: (H,W), noise_std: float }
         ↓
scripts/run_train.py --variant [A|B|C] --data flow_pairs.npz
```

- **A**: loads `data["clean"]` → PatchDictTrainer (unchanged)
- **B**: loads `data["clean"]` → ConvDictTrainer (unchanged)
- **C**: loads `data["dirty"]` + `data["clean"]` → FlowTrainer

### What changes

| File | Change |
|---|---|
| `scripts/simulate_observations.py` | **new** — PSF conv + noise → saves clean/dirty/psf/noise_std |
| `flow_dict.py` | `FlowTrainer.fit(dirty, clean, device)` — remove OT pairing; normalise by clean stats |
| `solvers.py` | `FlowSolver.__init__` — remove PSF; `decode_island` — Euler ODE from dirty island |
| `scripts/run_train.py` | load correct key(s) per variant; `--variant C` takes `--pairs` npz |
| `scripts/smoke_test_flow.py` | update for new interface |
| `pyproject.toml` | update `train-flow-gpu` task |

### What does NOT change
`UNetVelocityField`, `FlowModel`, `deconvolver.py`, `__init__.py`, Variants A/B trainers.

### TODO — Variant C refactor

- [ ] **CR-1** `scripts/simulate_observations.py` — PSF convolution + noise, output npz
- [ ] **CR-2** `FlowTrainer.fit(dirty, clean, device)` — dirty→clean CFM, remove OT pairing
- [ ] **CR-3** `FlowSolver` — remove PSF, Euler ODE from dirty island, perturb for uncertainty
- [ ] **CR-4** `scripts/run_train.py` — load clean/dirty keys per variant
- [ ] **CR-5** `pyproject.toml` — update `train-flow-gpu` task
- [ ] **CR-6** `scripts/smoke_test_flow.py` — update for new interface
- [ ] **CR-7** Smoke test: `simulate_observations.py` produces valid npz
- [ ] **CR-8** Smoke test: `run_train.py --variant C` loss decreases on dirty→clean
- [ ] **CR-9** Smoke test: `FlowSolver.decode_island` returns (H_i, W_i) without PSF

---

## Current Status Summary (Variants A and B)

| Component | Status | Notes |
|---|---|---|
| Package structure | ✅ Done | flat layout, conftest.py shim for tests |
| `io.py` | ✅ Done | FITS + numpy, no torch |
| `filters.py` FilterBank | ✅ Done | atoms, D, D_fft, save/load |
| `detection.py` IslandDetector | ✅ Done | GPU iterative dilation |
| `solvers.py` PatchSolver | ✅ Done | OMP + PyTorch fold/unfold |
| `solvers.py` ConvSolver | ✅ Done | FISTA refactored — _run_fista + encode_island |
| `deconvolver.py` MADClean | ✅ Done | Outer loop, PSF conv, FITS out |
| `patch_dict.py` | ✅ Done | sklearn MiniBatchDL wrapper |
| `conv_dict.py` | ✅ Done | True CDL — minibatch alternating minimisation |
| `pyproject.toml` | ✅ Done | pixi, cpu + gpu envs, tasks point to scripts/ |
| `scripts/run_train.py` | ✅ Done | CLI wrapper for Variant A and B training |
| `scripts/run_deconvolve.py` | ✅ Done | CLI wrapper for deconvolution |
| `tests/conftest.py` | ✅ Done | sys.path shim for flat package layout |
| `tests/test_io.py` | ✅ Done | 7 tests |
| `tests/test_filters.py` | ✅ Done | 6 tests |
| `tests/test_detection.py` | ✅ Done | 7 tests |
| `tests/test_solvers.py` | ✅ Done | 10 tests (PatchSolver + ConvSolver) |
| `tests/test_deconvolver.py` | ✅ Done | 9 tests |
| `tests/test_training.py` | ✅ Done | 9 tests (patch + conv trainer smoke tests) |
| Real data validation | ❌ Not started | Need dirty beam FITS from simobserve |
| CASA/LibRA integration | ❌ Not started | Interface contract defined in §10 |

**Legacy files removed:** `train_conv_dict.py` (SPORCO), `mad_clean.py` (scipy/SPORCO standalone)

---

## Immediate Next Steps (in order)

### ~~Step 1 — Implement `ConvDictTrainer` properly~~ ✅ DONE

Full minibatch CDL implemented in `conv_dict.py`. Z-step via `ConvSolver.encode_island()`,
D-step via PyTorch autograd + Adam + unit-ball projection. See DESIGN.md §13.2.

### ~~Step 2 — Write test suite~~ ✅ DONE

50 tests across 5 files, all passing. Run with `pixi run test`.
See `TESTING.md` for full test plan and manual validation protocol.

### ~~Step 3 — Write CLI scripts~~ ✅ DONE

`scripts/run_train.py` and `scripts/run_deconvolve.py` written.
All pixi tasks (`train-patch`, `train-conv`, `deconvolve-A`, `deconvolve-B`) point to them.

---

### Step 1 (was Step 4) — Real data validation

**File:** `mad_clean/train/conv_dict.py`  
**Replace:** the current stub that calls `PatchDictTrainer`  
**Implement:** minibatch alternating minimisation CDL in pure PyTorch

The full algorithm is specified in DESIGN.md §7. Key points:

- Train on full 150×150 images, not patches (convolutional model is degenerate
  if signal size == filter size)
- Z-step: call `ConvSolver.decode_island()` per image in minibatch
  (reuse existing FISTA — do not reimplement)
- D-step: Fourier gradient + Adam + unit ball projection
- Minibatch size B=8 default (yields ~230MB activation maps on GPU)
- Log per-epoch loss and sparsity to stdout

**Hyperparameters to use (defaults, all tunable via constructor):**

```python
ConvDictTrainer(
    k            = 32,      # TBD after Variant A results
    atom_size    = 15,      # TBD after Variant A results
    batch_size   = 8,
    n_epochs     = 20,
    lr_D         = 1e-3,
    lmbda        = 0.1,     # Z-step FISTA λ
    fista_iter_train = 50,  # fewer than inference
    random_seed  = 42,
)
```

**Important:** `ConvDictTrainer.fit()` must return a `FilterBank` — same as
`PatchDictTrainer.fit()`. The interface is identical. The outer code
(`MADClean`, `ConvSolver`) must not need to change.

**After implementation:** smoke test with 10 synthetic 150×150 images,
5 epochs, check that loss decreases monotonically.

---

### Step 2 — Write test suite

**Directory:** `tests/`  
**Framework:** pytest (already in pixi tasks)

Minimum required tests:

```
tests/
├── test_io.py
│     - test_load_numpy_passthrough
│     - test_load_fits_round_trip
│     - test_load_fits_squeeze_casa_axes   [shape (1,1,H,W) → (H,W)]
│
├── test_filters.py
│     - test_filterbank_normalisation      [all atom norms == 1.0]
│     - test_filterbank_save_load_roundtrip
│     - test_filterbank_to_device
│     - test_dead_atom_report
│
├── test_detection.py
│     - test_two_point_sources_detected    [known input → 2 islands]
│     - test_empty_image_no_islands
│     - test_bbox_padding                  [bbox padded by atom_size//2]
│     - test_min_island_filter             [tiny blob filtered out]
│
├── test_solvers.py
│     - test_patch_solver_flux_conservation  [sum(recon) ≈ sum(island)]
│     - test_conv_solver_residual_decreases  [||s - recon|| < ||s||]
│     - test_conv_solver_fista_convergence   [loss monotone non-increasing]
│
├── test_deconvolver.py
│     - test_psf_convention               [delta ⊛ PSF peak at centre]
│     - test_residual_decreases           [RMS(r) monotone non-increasing]
│     - test_fits_output                  [files written if out_dir given]
│     - test_numpy_input                  [accepts numpy arrays directly]
│
└── test_training.py
      - test_patch_trainer_smoke          [5 epochs, 10 images, loss decreases]
      - test_conv_trainer_smoke           [5 epochs, 10 images, loss decreases]
      - test_trainer_returns_filterbank   [return type is FilterBank]
```

All tests must run without real CRUMB data — use synthetic numpy arrays.
No GPU required for tests — all tests run on CPU.

---

### Step 3 — Write CLI scripts

**Files:** `scripts/run_train.py`, `scripts/run_deconvolve.py`

These are thin argparse wrappers around the package classes. No logic belongs
here — all logic is in the package.

**`scripts/run_train.py`:**
```
args: --variant {A,B} --data <npz> --out <npy>
      --k --atom_size --alpha --n_iter   [Variant A]
      --k --atom_size --batch_size --n_epochs --lr_D --lmbda  [Variant B]
      --device {cpu,cuda}
```

**`scripts/run_deconvolve.py`:**
```
args: --variant {A,B}
      --dirty <fits or npy> --psf <fits or npy>
      --atoms <npy>
      --out_dir <dir>
      --gamma --n_max --epsilon_frac --detect_sigma
      --n_nonzero --stride   [Variant A]
      --lmbda --admm_iter    [Variant B]
      --device {cpu,cuda}
```

---

### Step 2 (was Step 5) — CASA/LibRA integration

**What is needed:**
1. A real CRUMB source (clean image) — available from
   `crumb_data/crumb_preprocessed.npz` once `fetch_crumb.py` is run
2. A real dirty beam FITS — must come from CASA `simobserve` or `tclean`
   on a VLA/FIRST-configuration observation, or extracted from any existing
   FIRST FITS image via `tclean`'s PSF output

**Validation protocol:**
```
1. Take one CRUMB source per class (FR-I, FR-II, Hybrid)
2. Convolve with real dirty beam → synthetic dirty image (ground truth known)
3. Run MADClean Variant A → model_A, residual_A
4. Run MADClean Variant B → model_B, residual_B
5. Compare: MSE(model_A, truth), MSE(model_B, truth), RMS curves
6. Visual inspection: model images, residual images, atom activation maps
```

**Success criterion:** RMS(residual) decreases monotonically and
MSE(model, truth) < MSE(dirty, truth) for both variants.

---

### Step 3 (deferred) — CASA/LibRA integration

**Approach:** minor cycle callback hook.

CASA `tclean` supports a `cycleniter` callback mechanism. LibRA has an
equivalent plugin interface. MAD-CLEAN registers as the minor cycle handler:

```python
# CASA pseudocode
from mad_clean import FilterBank, PatchSolver, IslandDetector, MADClean

fb  = FilterBank.load("models/cdl_filters_patch.npy", device="cuda")
mc  = MADClean(fb, PatchSolver(fb), IslandDetector(device="cuda"), device="cuda")

def mad_clean_minor_cycle(dirty, psf):
    result = mc.deconvolve(dirty, psf)
    return result["model"], result["residual"]

tclean(..., minor_cycle_hook=mad_clean_minor_cycle)
```

**What CASA/LibRA provides:** dirty image and PSF as numpy arrays or FITS
at each major cycle iteration.  
**What we return:** model and residual as numpy arrays.  
**What we never touch:** visibilities, gridding, FFT, measurement operator.

---

## Deferred Items (do not implement until explicitly reactivated)

| Item | Why deferred | What triggers it |
|---|---|---|
| Full batched OMP | Not a bottleneck at 150×150 | Profiling shows OMP is slow |
| Alpha sweep (Variant A) | Need K results first | After K sweep complete |
| K + atom_size sweep (Variant B) | Need Variant A results | After Variant A validated |
| Algorithm 2 (CGSR) | Long-term goal | After Variant B validated |
| Sub-beam filter constraint | Verified empirically | If atoms show PSF structure |
| LISTA unrolled solver | Faster than FISTA | After FISTA validated |
| uv-plane residuals | Image-plane first | After image-plane validated |

---

## Key File Locations

```
mad_clean/                  installable package
pyproject.toml              pixi manifest, cpu + gpu envs
DESIGN.md                   full architecture and math spec  ← read first
PLAN.md                     this file
crumb_data/
    crumb_preprocessed.npz  (N, 150, 150) float32 images + labels
models/
    cdl_filters_patch.npy   (K, 15, 15) Variant A atoms
    cdl_filters_conv.npy    (K, 15, 15) Variant B atoms  [after CDL training]
results/
    mad_clean_A_model.fits
    mad_clean_A_residual.fits
    mad_clean_A_rms_curve.npy
    mad_clean_B_model.fits
    mad_clean_B_residual.fits
    mad_clean_B_rms_curve.npy
scripts/
    fetch_crumb.py          download CRUMB data
    run_train.py            CLI training wrapper  [not yet written]
    run_deconvolve.py       CLI inference wrapper [not yet written]
tests/                      pytest suite          [not yet written]
```

---

## How to Resume a Session

1. Read DESIGN.md §1-4 (motivation, constraints, architecture)
2. Read PLAN.md Current Status table
3. Identify the first ❌ or ⚠️ item
4. Read the relevant DESIGN.md section for that item
5. Ask Preshanth for one-sentence deliverable goal before writing code
6. Follow design-first / implement-after protocol — no code until spec agreed

---

## Parameters Reference (all defaults)

| Parameter | Value | Where set |
|---|---|---|
| Atom size F | 15 px | FilterBank, both trainers |
| K (Variant A) | 32 | PatchDictTrainer |
| K (Variant B) | TBD | ConvDictTrainer |
| Alpha (Variant A) | 0.1 | PatchDictTrainer |
| OMP sparsity S | 5 | PatchSolver |
| Tiling stride | 8 px | PatchSolver |
| FISTA λ (inference) | 0.1 | ConvSolver |
| FISTA n_iter (inference) | 100 | ConvSolver |
| FISTA tol | 1e-4 | ConvSolver |
| CDL λ (training) | 0.1 | ConvDictTrainer |
| CDL FISTA iters (training) | 50 | ConvDictTrainer |
| CDL batch size | 8 | ConvDictTrainer |
| CDL n_epochs | 20 | ConvDictTrainer |
| CDL lr_D | 1e-3 | ConvDictTrainer |
| Loop gain γ | 0.1 | MADClean |
| Convergence ε | 1% initial RMS | MADClean |
| Max iterations | 500 | MADClean |
| Detection threshold | 3σ | IslandDetector |
| Min island size | 9 px | IslandDetector |
| Dilation max iter | 150 | IslandDetector |
| PSF FWHM (FIRST) | 2.8 px | Physical constant |