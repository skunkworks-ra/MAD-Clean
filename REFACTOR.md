# MAD-CLEAN Refactor Plan

## Goal
Installable package (`pip install -e .`) with clean subpackage layout.
Kills: import shim in every script, scattered normalisation, duplicated utilities.

## Target Layout

```
mad_clean/
├── __init__.py              clean public API (re-exports unchanged)
├── _utils.py         NEW    soft_threshold, clip_box, guide  (3 dupes → 1)
├── normalise.py      NEW    ImageNormaliser class (canonical dirty/clean normalisation)
├── filters.py               unchanged
├── detection.py             unchanged
├── psf_utils.py             unchanged
├── hogbom.py                update: use _utils.clip_box, _utils.guide
├── io.py                    unchanged
├── deconvolver.py           update: use _utils.clip_box, _utils.guide
├── solvers.py               update: use _utils.soft_threshold
├── training/
│   ├── __init__.py          exports PatchDictTrainer, ConvDictTrainer, FlowModel, FlowTrainer
│   ├── patch.py        ←    patch_dict.py
│   ├── conv.py         ←    conv_dict.py
│   └── flow.py         ←    flow_dict.py  (preserve resume_from param)
└── data/
    ├── __init__.py          exports CRUMBDataset, SimulateObservations
    ├── crumb.py        ←    CRUMB.py  (factory pattern, kills ~500 dup lines)
    └── simulate.py     ←    simulate_observations.py logic as a class
```

```
scripts/
├── train.py          ←    run_train.py       (no shim — clean imports only)
├── deconvolve.py     ←    run_deconvolve.py
├── simulate.py       ←    simulate_observations.py
├── eval.py           ←    eval_deconvolution.py
├── extract_psf.py         unchanged
├── fetch_crumb.py         unchanged
└── visualize_atoms.py     unchanged
```

**Delete:** `patch_dict.py`, `conv_dict.py`, `flow_dict.py`, `train_patch_dict.py`, root-level `simulate_observations.py`

---

## pyproject.toml additions

```toml
[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.backends.legacy:build"

[tool.setuptools.packages.find]
where = ["."]
include = ["mad_clean*"]

[project.scripts]
mad-train      = "scripts.train:main"
mad-deconvolve = "scripts.deconvolve:main"
mad-simulate   = "scripts.simulate:main"
mad-eval       = "scripts.eval:main"
```

---

## Key new files

### mad_clean/_utils.py
```python
"""Shared low-level utilities — eliminates 3 copies of soft_threshold, 2 of clip_box."""
import torch

def soft_threshold(x: torch.Tensor, threshold: float) -> torch.Tensor:
    return x.sign() * (x.abs() - threshold).clamp(min=0.0)

def clip_box(py, px, psf_cy, psf_cx, psf_h, psf_w, H, W):
    # exact logic from hogbom.py and deconvolver.py
    ...

def guide(residual: torch.Tensor) -> torch.Tensor:
    # exact logic from hogbom.py and deconvolver.py
    ...
```

### mad_clean/normalise.py
```python
"""Single canonical normalisation for dirty/clean pairs.

Fixes the dirty std=101 bug: previously simulate_observations.py
normalised before saving, then FlowTrainer normalised again, and
eval scripts had to add a third pass. Now one class owns it.
"""
import numpy as np

class ImageNormaliser:
    def fit(self, clean: np.ndarray) -> "ImageNormaliser":
        self._mean = clean.mean(axis=(1,2), keepdims=True)
        self._std  = clean.std(axis=(1,2),  keepdims=True) + 1e-8
        return self

    def transform(self, images: np.ndarray) -> np.ndarray:
        return (images - self._mean) / self._std

    def inverse_transform(self, images: np.ndarray) -> np.ndarray:
        return images * self._std + self._mean

    def fit_transform(self, clean: np.ndarray, dirty: np.ndarray = None):
        self.fit(clean)
        clean_n = self.transform(clean)
        if dirty is not None:
            return clean_n, self.transform(dirty)
        return clean_n
```

`FlowTrainer.fit()` and `SimulateObservations` both use `ImageNormaliser` — no more inline normalisation in either.

---

## Public API (must remain unchanged in __init__.py)

```python
from mad_clean.filters       import FilterBank
from mad_clean.detection     import IslandDetector
from mad_clean.solvers       import PatchSolver, ConvSolver, FlowSolver
from mad_clean.deconvolver   import MADClean
from mad_clean.io            import load_image, save_fits
from mad_clean.training      import FlowModel, FlowTrainer, PatchDictTrainer, ConvDictTrainer
from mad_clean.hogbom        import hogbom_clean
from mad_clean.psf_utils     import compute_psf_patch
from mad_clean.normalise     import ImageNormaliser
```

---

## Implementation order (to avoid broken intermediate states)

1. Create `mad_clean/_utils.py` and `mad_clean/normalise.py`
2. Create `mad_clean/training/` — move patch/conv/flow, update imports
3. Create `mad_clean/data/` — move crumb + simulate, refactor CRUMB factory
4. Update `hogbom.py`, `deconvolver.py`, `solvers.py` to use `_utils`
5. Update `mad_clean/__init__.py` exports
6. Rewrite scripts as clean CLIs (no shim)
7. Update `pyproject.toml`
8. Delete legacy files
9. Run `pip install -e .` and smoke test

---

## Do on a branch

```bash
git checkout -b refactor/package-structure
# implement
pip install -e .
pixi run -e gpu python -c "import mad_clean; print(mad_clean.__version__)"
python scripts/eval.py --n 3 --device cuda --central --out models/eval_post_refactor.png
```
