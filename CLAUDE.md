# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Install (CPU — tests, inference)
pixi install

# Install (GPU — training)
pixi install --environment gpu

# Run all tests
pixi run test

# Run a single test file
pixi run python -m pytest tests/test_solvers.py -v

# Run a single test by name
pixi run python -m pytest tests/test_solvers.py::test_patch_solver_shape -v

# Simulate training data (one-time; all variants)
pixi run simulate

# Train
pixi run -e gpu train-patch-gpu   # Variant A
pixi run -e gpu train-conv-gpu    # Variant B
pixi run -e gpu train-flow-gpu    # Variant C

# Deconvolve
pixi run deconvolve-A
pixi run deconvolve-B
pixi run deconvolve-hogbom
```

Expected test result: **50 passed** in ~4–6 seconds on CPU. No GPU, no CRUMB data needed.

## Architecture

MAD-CLEAN replaces the standard CLEAN minor cycle delta-function picker with a learned sparse coding step. The PSF (measurement operator) is always kept explicit in the outer loop — the learned component only generates the source model from a residual island, never touches the PSF. This is the key design invariant that separates it from PnP/AIRI-style methods.

Three solver variants share the same outer `MADClean` deconvolution loop:

| Variant | Solver | Training | Notes |
|---|---|---|---|
| A | `PatchSolver` (OMP) | sklearn `MiniBatchDictionaryLearning` | Patch-based; fast CPU inference |
| B | `ConvSolver` (FISTA) | PyTorch minibatch alternating min | Full-image convolutional atoms |
| C | `FlowSolver` (Euler ODE) | Conditional flow matching U-Net | Dirty→clean; produces uncertainty maps |

### Module map

- `filters.py` — `FilterBank`: stores and normalises atom arrays; shared by A and B
- `detection.py` — `IslandDetector`: sigma-threshold source finding; returns bounding boxes
- `solvers.py` — `PatchSolver`, `ConvSolver`, `FlowSolver`: each takes an island cutout and returns a model image
- `deconvolver.py` — `MADClean`: outer CLEAN loop; calls detector → solver → PSF subtract per major cycle
- `patch_dict.py`, `conv_dict.py`, `flow_dict.py` — trainers and model classes for each variant
- `psf_utils.py` — PSF FFT helpers; `ifftshift` convention is critical (tested explicitly)
- `io.py` — FITS + numpy I/O; no torch dependency; squeezes degenerate CASA axes
- `hogbom.py` — reference Hogbom CLEAN for comparison
- `train_patch_dict.py` — standalone training entry point (also called via `scripts/run_train.py`)

### Data flow

```
crumb_preprocessed.npz + PSF
        ↓ simulate_observations.py
flow_pairs.npz { clean: (N,150,150), dirty: (N,150,150), psf: (150,150) }
        ↓ run_train.py
models/  (cdl_filters_patch.npy | cdl_filters_conv.npy | flow_model.pt)
        ↓ run_deconvolve.py / MADClean.deconvolve()
results/ (model.fits, residual.fits, rms_curve.npy [, uncertainty.fits])
```

Variants A and B train only on the `clean` key. Variant C trains on both `dirty` and `clean`.

### Physical constraint

Atom size is 15×15 px (27 arcsec at 1.8 arcsec/px). The minimum physically meaningful scale is 1 full beam FWHM (~2.8 px for VLA FIRST). Do not reduce atom size below this without a physics justification — sub-beam structure is PSF artefact, not source morphology.

## Key design references

- `design.md` — full algorithm spec, class dependency graph, PSF convention, physical constraints
- `TESTING.md` — end-to-end validation protocols and expected failure modes per variant
- `plan.md` — current implementation status
