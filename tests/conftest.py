"""
conftest.py — pytest bootstrap for MAD-CLEAN tests.

When running `pytest tests/` outside the installed pixi environment,
this file adds the project root to sys.path and registers a `mad_clean`
pseudo-package so that `from mad_clean.xxx import yyy` imports resolve
correctly from the flat source layout.

No production code is changed; this shim is test-environment-only.
"""

import importlib
import importlib.util
import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent  # /path/to/MAD-clean

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _load_local(mad_clean_key: str, filename: str):
    """
    Load a source file from ROOT by explicit path, bypassing sys.path name
    resolution. Registers the module ONLY under `mad_clean.<mad_clean_key>`,
    never under the bare name, to avoid shadowing stdlib modules (e.g. 'io').
    """
    path     = ROOT / filename
    full_key = f"mad_clean.{mad_clean_key}"
    spec     = importlib.util.spec_from_file_location(full_key, path)
    mod      = importlib.util.module_from_spec(spec)
    sys.modules[full_key] = mod          # register as mad_clean.xxx only
    spec.loader.exec_module(mod)
    return mod


# If mad_clean isn't already importable (i.e. not installed via pixi),
# build a proxy package from the flat root modules.
if "mad_clean" not in sys.modules:
    pkg = types.ModuleType("mad_clean")
    pkg.__path__    = [str(ROOT)]
    pkg.__package__ = "mad_clean"
    sys.modules["mad_clean"] = pkg

    # Load each flat module by explicit path.
    # Key: the name used in `mad_clean.<key>` imports.
    # Value: the filename on disk.
    # 'io' must NOT be registered under bare "io" — that would shadow stdlib io.
    _local_mods = {
        "filters"    : "filters.py",
        "detection"  : "detection.py",
        "io"         : "io.py",
        "patch_dict" : "patch_dict.py",
        "conv_dict"  : "conv_dict.py",
        "solvers"    : "solvers.py",
        "deconvolver": "deconvolver.py",
    }
    for _name, _file in _local_mods.items():
        _mod = _load_local(_name, _file)
        setattr(pkg, _name, _mod)

    # mad_clean.train sub-package (patch_dict and conv_dict)
    _train = types.ModuleType("mad_clean.train")
    _train.__package__ = "mad_clean.train"
    sys.modules["mad_clean.train"] = _train
    setattr(pkg, "train", _train)
    for _name in ("patch_dict", "conv_dict"):
        _mod = sys.modules[f"mad_clean.{_name}"]
        setattr(_train, _name, _mod)
        sys.modules[f"mad_clean.train.{_name}"] = _mod
