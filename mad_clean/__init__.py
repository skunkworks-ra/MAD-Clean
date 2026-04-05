"""
mad_clean
=========
Morphological Atom Decomposition CLEAN — a learned sparse-coding minor cycle
for radio interferometric image reconstruction.

Quick start
-----------
    import numpy as np
    from mad_clean import FilterBank, PatchSolver, ConvSolver
    from mad_clean import IslandDetector, MADClean
    from mad_clean.training import PatchDictTrainer, ConvDictTrainer

    # --- Training ---
    images  = np.load("crumb_data/crumb_preprocessed.npz")["images"]
    trainer = PatchDictTrainer(k=32, atom_size=15)
    fb      = trainer.fit(images, device="cuda")
    fb.save("models/cdl_filters_patch")

    # --- Inference ---
    fb       = FilterBank.load("models/cdl_filters_patch.npz", device="cuda")
    solver   = PatchSolver(fb, n_nonzero=5, stride=8)
    detector = IslandDetector(sigma_thresh=3.0, device="cuda")
    mc       = MADClean(fb, solver, detector, gamma=0.1, device="cuda")

    result   = mc.deconvolve("dirty.fits", "psf.fits", out_dir="results/")
    model    = result["model"]      # np.ndarray (H, W) float32
    residual = result["residual"]   # np.ndarray (H, W) float32
"""

from .filters     import FilterBank
from .detection   import IslandDetector
from .solvers     import PatchSolver, ConvSolver, FlowSolver
from .deconvolver import MADClean
from .io          import load_image, load_image_data, save_fits
from .training    import FlowModel, FlowTrainer, PatchDictTrainer, ConvDictTrainer
from .hogbom      import hogbom_clean
from .psf_utils   import compute_psf_patch
from .normalise   import ImageNormaliser

__all__ = [
    "FilterBank",
    "IslandDetector",
    "PatchSolver",
    "ConvSolver",
    "FlowSolver",
    "MADClean",
    "load_image",
    "load_image_data",
    "save_fits",
    "FlowModel",
    "FlowTrainer",
    "PatchDictTrainer",
    "ConvDictTrainer",
    "compute_psf_patch",
    "hogbom_clean",
    "ImageNormaliser",
]

__version__ = "0.1.0"
