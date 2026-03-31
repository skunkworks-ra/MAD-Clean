# MAD-CLEAN

**Morphological Atom Decomposition CLEAN** — a learned sparse-coding minor cycle for radio interferometric image reconstruction.

MAD-CLEAN replaces the standard CLEAN minor cycle with a learned prior over radio galaxy morphology. Three solver variants are provided, each progressively more expressive:

| Variant | Method | Prior | Uncertainty |
|---|---|---|---|
| A | Patch OMP | Patch dictionary (sklearn) | No |
| B | Convolutional CDL | Convolutional dictionary (PyTorch) | No |
| C | Conditional Flow Matching | Dirty→clean neural flow (PyTorch) | Yes |

---

## Quick start

```bash
# 1. Install pixi (https://pixi.sh) if needed, then:
pixi install                          # CPU environment (tests, inference)
pixi install --environment gpu        # GPU environment (training)

# 2. Fetch CRUMB training data
pixi run fetch

# 3. Simulate dirty/clean training pairs (all variants use this)
pixi run simulate

# 4. Train your chosen variant
pixi run -e gpu train-patch-gpu       # Variant A
pixi run -e gpu train-conv-gpu        # Variant B
pixi run -e gpu train-flow-gpu        # Variant C

# 5. Deconvolve
pixi run deconvolve-A                 # Variant A
pixi run deconvolve-B                 # Variant B
```

---

## Data pipeline

All training data is generated from a single simulation step:

```
crumb_preprocessed.npz  +  PSF
         ↓
pixi run simulate
         ↓
crumb_data/flow_pairs.npz
  { clean: (N, 150, 150)   — ground truth sky
    dirty: (N, 150, 150)   — PSF-convolved + noise
    psf:   (150, 150)      — PSF used for simulation
    noise_std: float }
```

- **Variants A/B** consume the `clean` key (dictionary learning on sky morphology)
- **Variant C** consumes both `dirty` and `clean` (conditional flow training)

To use a real dirty beam instead of the default synthetic Gaussian (FWHM=3px):

```bash
python scripts/simulate_observations.py \
    --data crumb_data/crumb_preprocessed.npz \
    --psf  path/to/psf.fits \
    --noise_std 0.05 \
    --out  crumb_data/flow_pairs.npz
```

---

## Variants

### Variant A — Patch OMP

Learns a patch dictionary from clean sky images using sklearn `MiniBatchDictionaryLearning`. At inference, each detected island is tiled into overlapping 15×15 patches, decoded via OMP, and reconstructed with overlap-averaging.

```bash
pixi run -e gpu train-patch-gpu
pixi run deconvolve-A
```

### Variant B — Convolutional CDL

Learns a convolutional filter bank via minibatch alternating minimisation (Z-step: FISTA, D-step: Adam + unit-ball projection). At inference, each island is decoded via FISTA convolutional sparse coding.

```bash
pixi run -e gpu train-conv-gpu
pixi run deconvolve-B
```

### Variant C — Conditional Flow Matching

Trains a U-Net velocity field to map dirty islands to clean sky estimates via conditional flow matching (Lipman et al. 2022). The flow is trained on dirty→clean pairs so no PSF is needed at inference. Provides per-pixel uncertainty estimates via ensemble trajectories.

```bash
pixi run simulate                     # generate dirty/clean pairs first
pixi run -e gpu train-flow-gpu
# deconvolution via FlowSolver — result["uncertainty"] is populated
```

Output dict from `MADClean.deconvolve()` for Variant C includes:

```python
result = mc.deconvolve(dirty, psf)
result["model"]        # np.ndarray (H, W) — sky model
result["residual"]     # np.ndarray (H, W) — dirty - PSF⊛model
result["uncertainty"]  # np.ndarray (H, W) — per-pixel std (None for A/B)
result["rms_curve"]    # np.ndarray (n_iter,) — RMS per major cycle
```

---

## Python API

```python
import numpy as np
from mad_clean import FilterBank, PatchSolver, ConvSolver, IslandDetector, MADClean
from mad_clean import FlowModel, FlowSolver

# --- Variant A/B inference ---
fb       = FilterBank.load("models/cdl_filters_patch.npy", device="cuda")
solver   = PatchSolver(fb, n_nonzero=5, stride=8)
detector = IslandDetector(sigma_thresh=3.0, device="cuda")
mc       = MADClean(fb, solver, detector, gamma=0.1, device="cuda")
result   = mc.deconvolve("dirty.fits", "psf.fits", out_dir="results/")

# --- Variant C inference ---
fm       = FlowModel.load("models/flow_model.pt", device="cuda")
solver   = FlowSolver(fm, device="cuda", n_samples=8, n_steps=16)
detector = IslandDetector(sigma_thresh=3.0, device="cuda")
mc       = MADClean(None, solver, detector, gamma=0.1, device="cuda")
result   = mc.deconvolve("dirty.fits", "psf.fits", out_dir="results/")
print(result["uncertainty"].shape)   # (H, W)
```

---

## Tests

```bash
pixi run test
```

Expected: **50 passed** on CPU in ~4–6 seconds. No GPU required. No CRUMB dataset required.

See `TESTING.md` for the full validation protocol including end-to-end checks and head-to-head variant comparison.

---

## Project structure

```
MAD-clean/
├── flow_dict.py              # Variant C: UNetVelocityField, FlowModel, FlowTrainer
├── conv_dict.py              # Variant B: ConvDictTrainer
├── patch_dict.py             # Variant A: PatchDictTrainer
├── solvers.py                # PatchSolver, ConvSolver, FlowSolver
├── deconvolver.py            # MADClean outer CLEAN loop
├── detection.py              # IslandDetector
├── filters.py                # FilterBank
├── io.py                     # FITS + numpy I/O
├── scripts/
│   ├── simulate_observations.py   # Generate dirty/clean training pairs
│   ├── run_train.py               # Training CLI (all variants)
│   ├── run_deconvolve.py          # Deconvolution CLI
│   └── smoke_test_flow.py         # Variant C smoke tests
├── tests/                    # Automated test suite
├── crumb_data/               # Training data (not committed)
├── models/                   # Trained models (not committed)
└── design.md                 # Algorithm specification
```

---

## Reference

- Lipman et al. (2022) — Flow Matching for Generative Modeling
- design.md §14 — scientific motivation and literature positioning
- bayesian_imaging.md — literature review
