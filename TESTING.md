# MAD-CLEAN Testing Guide

**Last updated:** 2026-03-14

This document covers the automated test suite and the manual end-to-end
validation protocol for both Variant A (patch OMP) and Variant B (convolutional
CDL). Read DESIGN.md §5–7 for the algorithm specification before running
end-to-end tests.

---

## 1. Running the Automated Test Suite

### 1.1 Setup

```bash
# Install the CPU pixi environment (one-time)
pixi install --environment default

# Run all tests
pixi run test
```

Expected output: **50 passed** in ~4–6 seconds on CPU.

### 1.2 Test file overview

| File | Tests | What it covers |
|---|---|---|
| `tests/test_io.py` | 7 | FITS round-trip, numpy passthrough, CASA axis squeeze, header preservation |
| `tests/test_filters.py` | 7 | Unit-norm normalisation, save/load, dead atom report |
| `tests/test_detection.py` | 7 | Single/dual source detection, bbox padding, min-island filter, RMS return |
| `tests/test_solvers.py` | 11 | PatchSolver shape/flux; ConvSolver shape/residual/FISTA convergence; `encode_island` |
| `tests/test_deconvolver.py` | 9 | PSF convention (delta = identity), shape/dtype, FITS output, mismatch error |
| `tests/test_training.py` | 9 | PatchDictTrainer and ConvDictTrainer smoke tests (synthetic data, CPU) |

All tests use **synthetic data only** — no CRUMB dataset required. All tests
run on **CPU** — no GPU required.

### 1.3 Notable test design choices

- `test_psf_convention_delta`: verifies that `_convolve_psf(image, psf_fft)` is
  identity when PSF is a delta at the image centre. This directly tests the
  `ifftshift` convention that is easy to get wrong.
- `test_conv_solver_fista_convergence`: runs FISTA with 5 vs 100 iterations on
  the same input and asserts the 100-iteration result is not worse (5% slack).
  Tests that more iterations monotonically improve the solution.
- `test_conv_trainer_loss_decreases`: trains two ConvDictTrainer instances for
  1 and 4 epochs from the same seed, then measures reconstruction error on the
  same images. Four epochs must not be worse than one (5% slack).
- Detection tests expose and cover the sentinel-masking fix in
  `IslandDetector._label_components` — the original implementation incorrectly
  erased labels for isolated foreground pixels.

---

## 2. Variant A — Patch OMP (PatchDictTrainer + PatchSolver)

### 2.1 What it does

- **Training:** sklearn `MiniBatchDictionaryLearning` on 15×15 patches extracted
  from CRUMB images with random full-image rotation. Returns K=32 atoms of
  shape (15, 15).
- **Inference:** For each detected island, tiles with overlapping 15×15 patches
  (stride 8px), runs OMP (up to 5 nonzero atoms per patch) on CPU, reconstructs
  via `torch.nn.functional.fold` with overlap-averaging.

### 2.2 Training — step by step

```bash
# 1. Fetch CRUMB data (requires scripts/fetch_crumb.py — not yet written)
pixi run fetch

# 2. Train Variant A filter bank
pixi run train-patch
# equivalent:
# python scripts/run_train.py --variant A \
#     --data crumb_data/crumb_preprocessed.npz \
#     --out models/cdl_filters_patch.npy \
#     --k 32 --atom_size 15 --device cpu
```

Expected stdout (abridged):
```
Loaded N images  shape=150×150
Extracting patches  (n_images=..., patches/img=20) …
  Total patches: ...  shape: 15×15
Training MiniBatchDictionaryLearning  K=32  alpha=0.1  n_iter=1000 …
  Active atoms: 32/32  norm mean=1.000
Saved FilterBank → models/cdl_filters_patch.npy
```

**What to check:**
- Active atoms = 32/32 (no dead atoms with default settings)
- Training completes without NaN/Inf

### 2.3 Inspection — atom visualisation

```python
import numpy as np
import matplotlib.pyplot as plt

atoms = np.load("models/cdl_filters_patch.npy")  # (32, 15, 15)

fig, axes = plt.subplots(4, 8, figsize=(16, 8))
for ax, atom in zip(axes.flat, atoms):
    ax.imshow(atom, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.axis("off")
plt.tight_layout()
plt.savefig("results/atoms_A.png", dpi=150)
```

**What to look for:**
- Atoms should show morphologically distinct radio galaxy structures: extended
  jets (FR-I), double-lobe (FR-II), compact point-like, diffuse emission.
- No atom should be all-zero or nearly constant (indicates dead atom).
- Atoms should not look like Gabor filters or DCT basis functions — these would
  indicate the dictionary collapsed to a generic basis rather than learning
  radio morphology.

### 2.4 Inference

```bash
pixi run deconvolve-A
# equivalent:
# python scripts/run_deconvolve.py --variant A \
#     --dirty dirty.fits --psf psf.fits \
#     --atoms models/cdl_filters_patch.npy \
#     --out_dir results/
```

Outputs: `results/mad_clean_A_model.fits`, `results/mad_clean_A_residual.fits`,
`results/mad_clean_A_rms_curve.npy`.

### 2.5 Validation protocol (Variant A)

```python
import numpy as np
import matplotlib.pyplot as plt

# 1. Check RMS curve is monotonically non-increasing
rms = np.load("results/mad_clean_A_rms_curve.npy")
assert np.all(np.diff(rms) <= 0.01 * rms[:-1]), "RMS not decreasing — check PSF convention"
print(f"Converged in {len(rms)-1} iterations. Final RMS / Initial RMS = {rms[-1]/rms[0]:.3f}")

# 2. Check model is not all-zero
from astropy.io import fits
model    = fits.getdata("results/mad_clean_A_model.fits")
residual = fits.getdata("results/mad_clean_A_residual.fits")
dirty    = fits.getdata("dirty.fits")  # squeezed to (H, W)

print(f"Model peak: {model.max():.4e}  (should be ~dirty peak)")
print(f"Residual RMS: {residual.std():.4e}  (should be < dirty RMS)")
print(f"Dirty RMS: {dirty.std():.4e}")
assert residual.std() < dirty.std(), "Deconvolution made residual worse"

# 3. Visual inspection
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
for ax, img, title in zip(axes, [dirty, model, residual],
                           ["Dirty", "Model A", "Residual A"]):
    ax.imshow(np.squeeze(img), origin="lower", cmap="inferno")
    ax.set_title(title)
plt.tight_layout()
plt.savefig("results/validation_A.png", dpi=150)
```

**Success criteria:**
- RMS(residual) < RMS(dirty)
- RMS curve non-increasing
- Model visually resembles the source structure

---

## 3. Variant B — Convolutional CDL (ConvDictTrainer + ConvSolver)

### 3.1 What it does

- **Training:** Minibatch alternating minimisation on full 150×150 images.
  Z-step: FISTA sparse coding via `ConvSolver.encode_island()` (50 iterations
  per image, detached from graph). D-step: PyTorch autograd on the
  convolutional forward model + Adam + unit-ball projection per atom.
- **Inference:** For each detected island, runs FISTA (100 iterations) on the
  full island as a convolutional sparse coding problem. No tiling — the island
  is decoded at native resolution.

### 3.2 Training — step by step

```bash
pixi run train-conv
# equivalent (GPU recommended for full training):
# python scripts/run_train.py --variant B \
#     --data crumb_data/crumb_preprocessed.npz \
#     --out models/cdl_filters_conv.npy \
#     --k 32 --atom_size 15 \
#     --batch_size 8 --n_epochs 20 --lr_d 1e-3 --lmbda 0.1 \
#     --device cuda   # or cpu for testing
```

Expected stdout per epoch:
```
ConvDictTrainer: K=32  F=15  images=N×150×150  batch=8  epochs=20  device=cuda
  Epoch   1/20  loss=0.123456  sparsity=0.812
  Epoch   2/20  loss=0.098123  sparsity=0.841
  ...
  Epoch  20/20  loss=0.041234  sparsity=0.893
ConvDictTrainer complete  active atoms: 32/32  norm mean=1.000
```

**What to check:**
- Loss decreases epoch over epoch (not necessarily monotone per batch, but
  the epoch average should trend downward)
- Sparsity increases (approaches 0.9+ means most activations are zero — good)
- No dead atoms in the final FilterBank

**Expected runtime (approximate):**
- CPU, 9000 images, 20 epochs: several hours — use GPU for real training
- GPU (A100), 9000 images, 20 epochs: 30–60 minutes
- Quick sanity check (100 images, 3 epochs, cpu): ~2–3 minutes

### 3.3 Inspection — atom visualisation and comparison

```python
import numpy as np
import matplotlib.pyplot as plt

atoms_A = np.load("models/cdl_filters_patch.npy")   # (32, 15, 15)
atoms_B = np.load("models/cdl_filters_conv.npy")    # (32, 15, 15)

fig, axes = plt.subplots(2, 8, figsize=(16, 5))
for ax, atom in zip(axes[0], atoms_A[:8]):
    ax.imshow(atom, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.axis("off")
axes[0, 0].set_ylabel("Variant A")
for ax, atom in zip(axes[1], atoms_B[:8]):
    ax.imshow(atom, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.axis("off")
axes[1, 0].set_ylabel("Variant B")
plt.savefig("results/atoms_AB_comparison.png", dpi=150)
```

**What to look for (Variant B specific):**
- Atoms should look smoother and more spatially coherent than Variant A —
  convolutional training on full images should capture more global structure.
- Ideally, Variant B atoms should look less patch-like and more like extended
  emission templates (jets, lobes).
- If atoms look identical to Variant A, the CDL training may not have run
  enough epochs or the lmbda may be too high.

### 3.4 Inference

```bash
pixi run deconvolve-B
# equivalent:
# python scripts/run_deconvolve.py --variant B \
#     --dirty dirty.fits --psf psf.fits \
#     --atoms models/cdl_filters_conv.npy \
#     --out_dir results/ \
#     --lmbda 0.1 --fista_iter 100
```

### 3.5 Validation protocol (Variant B)

Same as Variant A (§2.5) but with Variant B outputs. Additionally compare both:

```python
import numpy as np
from astropy.io import fits

rms_A = np.load("results/mad_clean_A_rms_curve.npy")
rms_B = np.load("results/mad_clean_B_rms_curve.npy")
dirty = fits.getdata("dirty.fits")

print(f"Variant A: {len(rms_A)-1} iters  final RMS = {rms_A[-1]:.4e}")
print(f"Variant B: {len(rms_B)-1} iters  final RMS = {rms_B[-1]:.4e}")
print(f"Dirty  RMS = {dirty.std():.4e}")

# Success criterion: both variants reduce residual RMS
assert rms_A[-1] < dirty.std(), "Variant A failed to reduce RMS"
assert rms_B[-1] < dirty.std(), "Variant B failed to reduce RMS"
```

---

## 4. Head-to-Head Comparison (Variants A vs B)

Once both variants have been run on the same dirty image:

```python
import numpy as np
from astropy.io import fits

truth    = fits.getdata("crumb_data/source_truth.fits")   # clean image (ground truth)
model_A  = fits.getdata("results/mad_clean_A_model.fits")
model_B  = fits.getdata("results/mad_clean_B_model.fits")
dirty    = fits.getdata("dirty.fits")

def mse(a, b):
    return float(np.mean((a - b) ** 2))

print(f"MSE(dirty,  truth) = {mse(dirty,   truth):.4e}  [baseline]")
print(f"MSE(modelA, truth) = {mse(model_A, truth):.4e}  [Variant A]")
print(f"MSE(modelB, truth) = {mse(model_B, truth):.4e}  [Variant B]")
```

**Success criteria (from DESIGN.md §4, validation protocol):**
- `MSE(model_X, truth) < MSE(dirty, truth)` for both variants
- RMS(residual) decreases monotonically
- Visual inspection: model resembles the true morphology, residual is noise-like

**Expected failure modes to watch for:**
- Variant A: over-smooth reconstruction if K is too small; flux errors if OMP
  sparsity is too tight
- Variant B: slow convergence if lmbda is too high; atoms don't differ from A
  if n_epochs is insufficient
- Both: PSF convention error causes diverging RMS — check `ifftshift` is applied

---

## 5. GPU Training (Variant B)

For full-scale CDL training on CRUMB:

```bash
pixi run --environment gpu train-conv
# or explicitly:
pixi run --environment gpu python scripts/run_train.py \
    --variant B \
    --data crumb_data/crumb_preprocessed.npz \
    --out models/cdl_filters_conv.npy \
    --k 32 --atom_size 15 \
    --batch_size 8 --n_epochs 20 \
    --device cuda
```

Memory budget at B=8, K=32, 150×150:
- Activation maps Z: ~230 MB
- Atom FFTs: ~28 MB
- Total: ~260 MB — fits on any modern GPU with ≥ 8GB VRAM

---

## 6. Resuming After a Session Break

```
1. Read DESIGN.md §1–4 (motivation, constraints, architecture)
2. Read PLAN.md Current Status table
3. Read TESTING.md §2 or §3 for the relevant variant
4. Run `pixi run test` — confirm 50/50 still passing
5. State the one-sentence deliverable goal before writing code
```
