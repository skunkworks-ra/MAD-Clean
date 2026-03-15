# MAD-CLEAN Implementation Plan

**Last updated:** 2026-03-14  
**Context note:** This document exists to restore full working context in a new
session. Read DESIGN.md first, then this document. Together they are sufficient
to resume implementation without re-deriving anything.

---

## Current Status Summary

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