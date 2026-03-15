# MAD-CLEAN Design Document

**Project:** Morphological Atom Decomposition CLEAN  
**Last updated:** 2026-03-14  
**Status:** Package implemented, CDL training in progress, real-data validation pending  

---

## 1. Scientific Motivation

Standard CLEAN deconvolution uses delta functions as the source basis. This is
physically wrong for extended radio sources (FR-I jets, FR-II lobes, diffuse
cluster emission) where the morphology has structure at multiple scales. The
central question is:

> Can a morphologically-informed generative prior — learned from real labelled
> radio galaxy images — improve deconvolution over delta-function CLEAN, and
> under what conditions (uv-coverage sparsity, SNR, source complexity)?

MAD-CLEAN answers this by replacing the CLEAN minor cycle's delta-function
component finder with a sparse coding step over a learned morphological
dictionary. The measurement operator (PSF convolution) remains explicit and
outside the learned component at all times. This avoids the generalisation
failure mode identified by Mars, Betcke & McEwen (2024/2025) in learned
post-processing methods.

---

## 2. Positioning in the Literature

| Method | Learned component | PSF handling | Generalisation |
|---|---|---|---|
| Standard CLEAN | None | Explicit | Full |
| LISTA-style unrolled | Minor cycle + PSF approx | Baked into weights | Poor across uv-coverages |
| PnP/AIRI (Wiaux group) | Denoiser on residuals | Explicit proximal loop | Good, but black box |
| **MAD-CLEAN (ours)** | **Source model generator** | **Explicit outer loop** | **Good — PSF never enters learned component** |

MAD-CLEAN is not PnP. The learned component generates the source model
directly from a residual patch — it does not denoise the residual.

---

## 3. Physical Constraints

All scale, size, and resolution choices are grounded in instrument parameters.

| Parameter | Value | Source |
|---|---|---|
| Survey | VLA FIRST, 1.4 GHz | CRUMB dataset |
| Pixel scale | 1.8 arcsec/pixel | CRUMB/FIRST |
| Image size | 150×150 px = 270×270 arcsec | CRUMB |
| Beam FWHM | 5 arcsec = 2.8 px | FIRST beam |
| Minimum meaningful atom scale | 2.8 px (1 full beam FWHM) | Physics rule |
| **Atom size (validated)** | **15×15 px = 27 arcsec** | **Empirical: 75px too large** |
| Training images | ~2100 (CRUMB), target 9000 | CRUMB v2.0 + augmentation |

**Rule:** No atom may encode structure below 1 full beam FWHM (2.8 px).
Sub-beam structure is PSF artefact, not source morphology. The 15px atom size
(5.4 beams) satisfies this constraint with margin.

---

## 4. Package Architecture

```
mad_clean/
├── __init__.py          public API
├── io.py                FITS + numpy I/O, no torch dependency
├── filters.py           FilterBank class
├── detection.py         IslandDetector class
├── solvers.py           PatchSolver, ConvSolver
├── deconvolver.py       MADClean outer loop
└── train/
    ├── __init__.py
    ├── patch_dict.py    PatchDictTrainer (sklearn)
    └── conv_dict.py     ConvDictTrainer (PyTorch CDL)
```

### 4.1 Class dependency graph

```
FilterBank  ←─────────────────────────────────────────────┐
    │                                                       │
    ├──► PatchSolver.fb      (uses .D, .atoms)             │
    │       │                                              │
    └──► ConvSolver.fb       (uses .D_fft, .atoms)         │
            │                                              │
    both passed to                                         │
            │                                              │
        MADClean                                           │
            │ also uses                                    │
        IslandDetector                                     │
                                                           │
PatchDictTrainer.fit()  ──────────────────────────► FilterBank
ConvDictTrainer.fit()   ──────────────────────────► FilterBank
```

### 4.2 Module responsibilities

**`io.py`** — pure functions, no torch.
- `load_image(source)` → `(np.ndarray, header)` — accepts numpy array or FITS
  path. Squeezes CASA degenerate axes (1,1,H,W) → (H,W).
- `load_image_data(source)` → `np.ndarray` — drops header, convenience wrapper.
- `save_fits(array, path, header)` — writes float32 FITS, preserves WCS.

**`filters.py` — `FilterBank`**
- Holds (K, F, F) atom array on a torch device.
- Normalises atoms to unit L2 norm on construction.
- Exposes:
  - `.atoms` : Tensor (K, F, F) — spatial atoms
  - `.D`     : Tensor (F², K)   — flattened, column-per-atom, for PatchSolver
  - `.D_fft` : Tensor (K, F, F//2+1) complex — precomputed rfft2 for ConvSolver
  - `.K`, `.F`, `.device`
- `save(path)` / `FilterBank.load(path, device)` — .npy serialisation.
- `to(device)` — returns new FilterBank on different device.
- `dead_atom_report()` — diagnostics dict.

**`detection.py` — `IslandDetector`**
- Thresholds residual at `sigma_thresh × RMS` on GPU.
- Labels connected components via iterative binary dilation (PyTorch, no scipy).
- Returns list of `(r0, r1, c0, c1)` bounding boxes padded by `atom_size // 2`.
- Filters components smaller than `min_island` pixels.
- `to(device)` — returns new detector on different device.

**`solvers.py` — `PatchSolver`, `ConvSolver`**
- Both accept an island `Tensor (H_i, W_i)` and return a reconstruction
  `Tensor (H_i, W_i)`.
- See §5 and §6 for mathematical specification.

**`deconvolver.py` — `MADClean`**
- Owns the outer loop, PSF convolution, model accumulation, convergence check.
- PSF convolution: Fourier-plane via `torch.fft.rfft2`. PSF `ifftshift` applied
  once per `deconvolve()` call, result cached as `psf_fft`.
- Returns `dict(model, residual, rms_curve, n_iter)` as numpy arrays.
- Optionally writes FITS outputs.
- Works with either `PatchSolver` or `ConvSolver` — no internal branching.

**`train/patch_dict.py` — `PatchDictTrainer`**
- Wraps sklearn `MiniBatchDictionaryLearning`.
- Extracts random 15×15 patches from full images with random full-image
  rotation before extraction (avoids border artefacts).
- Returns `FilterBank`.

**`train/conv_dict.py` — `ConvDictTrainer`**
- Pure PyTorch minibatch CDL.
- Trains on full 150×150 images (not patches) — required for convolutional
  model to be non-degenerate (signal must be larger than filter).
- Alternating FISTA Z-step + Fourier gradient D-step.
- Returns `FilterBank`.
- See §7 for mathematical specification.

---

## 5. Algorithm 1 — Variant A: Patch Dictionary MAD-CLEAN

### 5.1 Filter bank

Trained by `PatchDictTrainer` using sklearn `MiniBatchDictionaryLearning`.
Atoms D ∈ ℝ^{F²×K} are columns of the dictionary matrix (F=15, K=32 default).
Each atom has unit L2 norm.

### 5.2 Outer loop

```
INPUTS:
    dirty image     x_d  ∈ ℝ^{H×W}
    PSF             Φ    ∈ ℝ^{H×W}   (peak at image centre, CASA convention)
    filter bank     D    ∈ ℝ^{F²×K}
    loop gain       γ    = 0.1
    convergence     ε    = 0.01 × ||r_0||_RMS
    max iterations  N    = 500

INITIALISE:
    r ← x_d
    m ← 0

repeat:
    1. DETECT
       σ = RMS(r)
       binary = (r > 3σ)
       islands {B_i} = connected_components(binary)   [GPU, iterative dilation]
       filter islands with area < 9px

    2. DECODE  (per island)
       for each island B_i = r[r0:r1, c0:c1]:
           tile B_i with 15×15 patches, stride 8px
           for each patch p_j:
               z_j = OMP(D, p_j, n_nonzero=5)
               m_j = reshape(D · z_j, 15×15)
           m_i = fold_average({m_j})    [overlap-average via torch.nn.functional.fold]

    3. ACCUMULATE
       m ← m + γ · Σ_i m_i

    4. RESIDUAL UPDATE
       r ← x_d - IFFT(FFT(m) · FFT(ifftshift(Φ)))

    5. CONVERGENCE
       if RMS(r) < ε  or  no islands:  break

OUTPUT:  m, r
```

### 5.3 Patch normalisation

Each patch is normalised to zero mean, unit variance before OMP and
denormalised after reconstruction. This is required because sklearn atoms are
trained on normalised patches. Failure to denormalise produces flux-incorrect
reconstructions.

### 5.4 Overlap averaging

Tiling with stride 8px on 15px atoms gives 47% overlap. Each pixel accumulates
contributions from multiple patch reconstructions. Final value is the mean
across contributing patches, computed via `torch.nn.functional.fold` and a
matching `ones` fold for the divisor. This conserves flux.

---

## 6. Algorithm 1 — Variant B: Convolutional Dictionary MAD-CLEAN

### 6.1 Filter bank

Trained by `ConvDictTrainer` using pure PyTorch minibatch CDL on full
150×150 images. Atoms {d_k} ∈ ℝ^{F×F}, k=1..K, each unit L2 norm.

### 6.2 Forward model

For an island of size H_i × W_i:

```
s_i ≈ Σ_k  d_k ⊛ z_k
```

where `z_k ∈ ℝ^{H_i × W_i}` is the activation map for filter k, and ⊛
denotes 2D convolution. The activation maps are jointly sparse.

### 6.3 Outer loop

Identical to Variant A except step 2 is replaced:

```
    2. DECODE  (per island, FISTA)
       for each island B_i  (H_i × W_i, extracted at bounding box):
           {z_k} = FISTA(D, B_i, λ=0.1, n_iter=100)
           m_i = Σ_k d_k ⊛ z_k
```

No tiling required. Island handed to FISTA at native size.

### 6.4 FISTA specification

Solves:

```
min_{Z}  (1/2) ||A Z - s||²  +  λ ||Z||_1

where A: Z → Σ_k d_k ⊛ z_k   (convolutional forward operator)
```

**Lipschitz constant:**

```
L = max_ω  Σ_k |D_k(ω)|²
```

computed from precomputed atom FFTs zero-padded to island size.
This is exact — no approximation.

**Step size:** `η = 1/L`

**Update rule:**

```
INITIALISE:  Z = 0,  Y = 0,  t = 1

for iter in 1..n_iter:
    grad = A^T (A Y - s)
         = IFFT( conj(D_fft) · (Σ_k D_fft[k] · Y_fft[k]  -  s_fft) )

    Z_new = soft_threshold(Y - η · grad,  λ·η)

    t_new = (1 + sqrt(1 + 4t²)) / 2
    Y     = Z_new + ((t-1)/t_new) · (Z_new - Z)
    Z     = Z_new
    t     = t_new

    if relative_obj_change < tol=1e-4:  break
```

**Soft threshold:** `sign(x) · max(|x| - threshold, 0)`  applied elementwise.

**Atom padding:** atoms zero-padded from F×F to H_i×W_i before FFT so
convolution is computed at island resolution. If island is smaller than
atom in any dimension, island is zero-padded to F×F instead.

---

## 7. CDL Training — ConvDictTrainer

### 7.1 Problem

```
min_{D, {Z_i}}  Σ_i  (1/2)||Σ_k d_k ⊛ z_k^i - s_i||²  +  λ Σ_i Σ_k ||z_k^i||_1

subject to:  ||d_k||_2 ≤ 1  for all k
```

where {s_i} are training images (full 150×150), {z_k^i} are activation maps
(150×150 per filter per image), D = {d_k} are the atoms (15×15 each).

### 7.2 Algorithm — Minibatch Alternating Minimisation

```
INITIALISE:
    D_0 = random (K, F, F), each atom normalised to unit L2 norm
    Adam optimiser on D, lr=1e-3

for epoch in 1..n_epochs:
    shuffle training images

    for each minibatch {s_1 ... s_B}  (B=8 default):

        Z-STEP  (per image, in parallel across batch):
            for i = 1..B:
                {z_k^i} = FISTA(D, s_i, λ=0.1, n_iter=50)
                           [approximate — fewer iters than inference]

        D-STEP  (gradient across batch, Adam update):
            for each k:
                recon_i    = Σ_k' d_k' ⊛ z_k'^i
                residual_i = recon_i - s_i
                grad_k     = Σ_i  IFFT( conj(Z_k^i_fft) · residual_i_fft )
            D ← Adam(D, grad)
            project each d_k onto unit L2 ball:
                d_k ← d_k / max(||d_k||_2, 1)

    log: per-epoch mean reconstruction loss, mean activation sparsity
```

### 7.3 Memory budget

At B=8, K=32, image size 150×150:
- Activation maps Z: 8 × 32 × 150 × 150 × 4 bytes ≈ 230 MB
- Atom FFTs: 32 × 150 × 76 × 8 bytes (complex64) ≈ 28 MB
- Total working set per minibatch: ~260 MB — fits comfortably on modern GPU

At 9000 images, n_epochs=20: 9000/8 × 20 = 22,500 D-steps. With 50 FISTA
iters per Z-step: ~56M FISTA iterations total. Feasible overnight on GPU.

### 7.4 Default hyperparameters

| Parameter | Default | Rationale |
|---|---|---|
| K | TBD after Variant A | Sweep pending |
| F (atom size) | TBD after Variant A | Sweep pending |
| batch_size B | 8 | Conservative GPU memory |
| n_epochs | 20 | Standard for this problem scale |
| lr_D | 1e-3 | Adam default |
| λ (Z-step) | 0.1 | Matches ConvSolver inference default |
| FISTA iters (Z-step) | 50 | Approximate during training |
| FISTA iters (inference) | 100 | More precise at inference |
| tol (early stop) | 1e-4 | Relative objective change |

---

## 8. PSF Convolution Convention

CASA outputs PSFs with peak at image centre (H//2, W//2). To use this as a
convolution kernel via FFT multiplication, the peak must be at (0,0).

**Implementation:**
```python
psf_shifted = torch.fft.ifftshift(psf)   # peak to (0,0)
psf_fft     = torch.fft.rfft2(psf_shifted)
result      = torch.fft.irfft2(rfft2(image) * psf_fft, s=image.shape)
```

`ifftshift` is applied once per `deconvolve()` call and the result cached.
This correctly handles sidelobes — no wrap-around artefacts.

**Requirement:** PSF and dirty image must have identical shape (H, W). If the
CASA PSF FITS is a different size, crop or pad before calling `deconvolve()`.

---

## 9. GPU Architecture

### 9.1 Device boundary

```
GPU tensors throughout major cycle:
    dirty, psf_fft, model, residual, atoms, D, D_fft

CPU only:
    bounding box integers (4 scalars per island — negligible transfer)

GPU for each island:
    island extraction (tensor slice — zero copy)
    sparse coding (FISTA or OMP matmul + fold)
    model accumulation (tensor scatter-add)
```

### 9.2 Connected components

GPU iterative binary dilation (PyTorch `max_pool2d`). No scipy at inference.
Algorithm: assign each foreground pixel a unique label equal to its flat index.
Propagate minimum label within 3×3 neighbourhood via max-pool on negated labels.
Converges in O(island_diameter) iterations — typically < 50 for CRUMB sources.

### 9.3 OMP in Variant A

OMP (sklearn) runs on CPU. The dictionary matrix multiply `D @ z → recon` and
the fold/unfold tiling are in PyTorch on GPU. Islands are decoded sequentially
in Python but the per-island operations are GPU-accelerated.

Full batched OMP (all islands in one GPU call) is a deferred optimisation.

---

## 10. Interface Contract with CASA / LibRA

MAD-CLEAN is designed as a drop-in minor cycle for any radio imaging framework.
The framework handles the major cycle (gridding, FFT, measurement operator).
MAD-CLEAN handles only the minor cycle.

**Input from framework:**
```python
dirty  : np.ndarray (H, W) float32   # or FITS path
psf    : np.ndarray (H, W) float32   # or FITS path, peak at image centre
```

**Output to framework:**
```python
model    : np.ndarray (H, W) float32
residual : np.ndarray (H, W) float32
```

**Example CASA hook (pseudocode):**
```python
from mad_clean import FilterBank, PatchSolver, IslandDetector, MADClean

fb  = FilterBank.load("models/cdl_filters_patch.npy", device="cuda")
mc  = MADClean(fb, PatchSolver(fb), IslandDetector(device="cuda"), device="cuda")

# Inside tclean minor cycle callback:
def minor_cycle(dirty_array, psf_array):
    result = mc.deconvolve(dirty_array, psf_array)
    return result["model"], result["residual"]
```

---

## 11. Dependency Policy

| Package | Channel | Purpose | Required |
|---|---|---|---|
| numpy | conda-forge | array I/O, scipy bridge | Yes |
| scipy | conda-forge | patch extraction in training | Yes |
| scikit-learn | conda-forge | OMP, MiniBatchDictionaryLearning | Yes |
| astropy | conda-forge | FITS I/O | Yes |
| torch ≥2.2 | pytorch | all GPU operations, FISTA, CDL | Yes |
| pytorch-cuda ≥12.1 | pytorch | CUDA backend | gpu env only |

**No SPORCO. No CuPy. No RAPIDS.**

SPORCO was removed because: (1) not on conda-forge, (2) unmaintained since
2022, (3) only `ConvBPDN` was needed — now replaced by `ConvSolver` (FISTA,
pure PyTorch). CuPy was considered for GPU connected components but rejected
in favour of iterative dilation in PyTorch to keep the dependency list clean.

---

## 12. Open Decisions

| Decision | Status | Note |
|---|---|---|
| K sweep for Variant B | Pending Variant A results | Use same K=32 initially |
| Atom size sweep for Variant B | Pending Variant A results | Use same F=15 initially |
| Alpha sweep for Variant A | Deferred | After K results reviewed |
| Full batched OMP | Deferred | Current: per-island loop, GPU matmul |
| CASA/LibRA integration | Not started | Interface contract defined in §10 |
| Real data validation | Not started | Need dirty beam FITS from simobserve |

---

## 13. Implementation Notes (post-session)

### 13.1 ConvSolver refactor

`ConvSolver.decode_island()` was refactored into three components:

- `_prepare_atoms_fft(island)` — handles atom/island size mismatch, pads atoms to island size, returns atoms_fft and a working copy of the island
- `_run_fista(atoms_fft, island, H, W)` — pure FISTA kernel, returns activation maps Z (K, H, W)
- `decode_island(island)` — calls the above and reconstructs Σ_k d_k ⊛ z_k
- `encode_island(island)` — calls `_prepare_atoms_fft` + `_run_fista`, returns Z directly (used by ConvDictTrainer Z-step)

### 13.2 ConvDictTrainer — true CDL implementation

`conv_dict.py` now implements full minibatch alternating minimisation per §7:

**Z-step:** Creates a temporary `FilterBank` from current D, instantiates `ConvSolver`, calls `encode_island()` per image in the minibatch. Z is detached from the graph (no gradient through the Z-step).

**D-step:** D is a `torch.nn.Parameter`. Forward model computes `recon = IFFT(D_fft * Z_fft)` using PyTorch autograd. `loss.backward()` + `Adam.step()` update D. After each step, D is projected onto the unit L2 ball per atom: `d_k ← d_k / max(||d_k||, 1)`.

Key design choices:
- D-step uses autograd rather than manually coded Fourier-domain gradient — cleaner and correct by construction
- Z is detached — the CDL alternating minimisation treats Z as fixed during the D-step
- A temporary FilterBank is created each minibatch for the Z-step. Since D is projected onto the unit ball after each step, FilterBank normalisation is a near-no-op (norms ≈ 1)

### 13.3 Connected-component labelling bug fix

The original `_label_components` in `detection.py` was broken for any foreground pixel adjacent to background. Background pixels (label 0) negate to 0, which is the maximum after negation, contaminating every border foreground pixel.

**Fix:** Replace background with a sentinel value S = H×W+2 before negating. In negated space, -S is smaller than any negated foreground label so background never wins the max-pool. Image boundary padding is also set to -S. After dilation, foreground pixels take `min(dilated, current_label)` — isolated pixels see S from the pool and min(S, label) = label, so they keep their label unchanged.

### 13.4 Package layout and test shim

All source files live flat in the project root (`MAD-clean/`). Imports throughout the codebase use `from mad_clean.xxx import yyy` which assumes the package is installed. `tests/conftest.py` builds a `mad_clean` proxy package at test-collection time using `importlib.util.spec_from_file_location`, registering each module under `mad_clean.<name>` only — not under the bare name — to avoid shadowing stdlib modules (particularly `io`).