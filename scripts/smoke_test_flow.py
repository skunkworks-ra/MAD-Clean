"""
smoke_test_flow.py — C-11 and C-12 verification for Variant C.

Run from project root:
    pixi run -e gpu python scripts/smoke_test_flow.py
"""

import sys
import importlib
import importlib.util
import types
from pathlib import Path

import numpy as np
import torch

# ── mad_clean import shim ─────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

if "mad_clean" not in sys.modules:
    pkg = types.ModuleType("mad_clean")
    pkg.__path__    = [str(ROOT)]
    pkg.__package__ = "mad_clean"
    sys.modules["mad_clean"] = pkg

    def _load_local(key: str, filename: str):
        full_key = f"mad_clean.{key}"
        spec = importlib.util.spec_from_file_location(full_key, ROOT / filename)
        mod  = importlib.util.module_from_spec(spec)
        sys.modules[full_key] = mod
        spec.loader.exec_module(mod)
        return mod

    for _name, _file in {
        "filters"    : "filters.py",
        "detection"  : "detection.py",
        "io"         : "io.py",
        "patch_dict" : "patch_dict.py",
        "conv_dict"  : "conv_dict.py",
        "flow_dict"  : "flow_dict.py",
        "solvers"    : "solvers.py",
        "deconvolver": "deconvolver.py",
    }.items():
        setattr(pkg, _name, _load_local(_name, _file))
# ─────────────────────────────────────────────────────────────────────────────

from flow_dict import FlowModel
from solvers   import FlowSolver


DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_PATH = ROOT / "models" / "flow_model.pt"


def test_cr7_simulate_observations():
    """CR-7: simulate_observations.py produces a valid npz."""
    import tempfile, subprocess
    print("─── CR-7: simulate_observations ───")
    with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as f:
        out_path = f.name
    result = subprocess.run(
        ["python", str(ROOT / "scripts" / "simulate_observations.py"),
         "--data",      str(ROOT / "crumb_data" / "crumb_preprocessed.npz"),
         "--psf_fwhm",  "3.0",
         "--noise_std", "0.05",
         "--out",       out_path],
        capture_output=True, text=True
    )
    assert result.returncode == 0, f"simulate failed:\n{result.stderr}"
    data = np.load(out_path)
    assert "clean"     in data, "missing 'clean' key"
    assert "dirty"     in data, "missing 'dirty' key"
    assert "psf"       in data, "missing 'psf' key"
    assert "noise_std" in data, "missing 'noise_std' key"
    assert data["clean"].shape == data["dirty"].shape
    print(f"  clean {data['clean'].shape}  dirty {data['dirty'].shape}  OK")


def test_cr8_velocity_shape():
    """CR-8a: velocity field shape check."""
    print("─── CR-8a: velocity field shape ───")
    fm = FlowModel.load(MODEL_PATH, device=DEVICE)
    x  = torch.randn(1, 1, 150, 150, device=DEVICE)
    t  = torch.tensor([0.5], device=DEVICE)
    v  = fm.velocity(x, t)
    assert v.shape == torch.Size([1, 1, 150, 150]), f"unexpected shape {v.shape}"
    print(f"  velocity shape: {v.shape}  OK")


def test_cr8_flowsolver_shapes():
    """CR-8b: FlowSolver shape assertions — no PSF required."""
    print("─── CR-8b: FlowSolver shape assertions ───")
    fm = FlowModel.load(MODEL_PATH, device=DEVICE)
    s  = FlowSolver(fm, DEVICE, n_samples=2, n_steps=4)

    island = torch.randn(40, 35, device=DEVICE)
    mean, std = s.decode_island_with_uncertainty(island)

    assert mean.shape == (40, 35), f"mean shape {mean.shape}"
    assert std.shape  == (40, 35), f"std shape {std.shape}"
    print(f"  mean {mean.shape}  std {std.shape}  OK")


def test_cr9_uncertainty_not_none():
    """CR-9: deconvolver uncertainty key is populated."""
    print("─── CR-9: deconvolver uncertainty key ───")
    from detection   import IslandDetector
    from deconvolver import MADClean

    fm       = FlowModel.load(MODEL_PATH, device=DEVICE)
    solver   = FlowSolver(fm, DEVICE, n_samples=2, n_steps=4)
    detector = IslandDetector(sigma_thresh=1.0, device=DEVICE)

    psf   = np.zeros((64, 64), dtype=np.float32); psf[32, 32] = 1.0
    dirty = np.random.randn(64, 64).astype(np.float32) * 0.1
    dirty[30:34, 30:34] += 1.0

    mc     = MADClean(None, solver, detector, gamma=0.1, n_max=2, device=DEVICE, verbose=True)
    result = mc.deconvolve(dirty, psf)

    assert "uncertainty" in result
    assert result["uncertainty"] is not None
    print(f"  uncertainty shape: {result['uncertainty'].shape}  OK")


if __name__ == "__main__":
    test_cr7_simulate_observations()
    test_cr8_velocity_shape()
    test_cr8_flowsolver_shapes()
    test_cr9_uncertainty_not_none()
    print("\nAll smoke tests passed.")
