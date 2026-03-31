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
    from mad_clean.train import PatchDictTrainer, ConvDictTrainer

    # --- Training ---
    images  = np.load("crumb_data/crumb_preprocessed.npz")["images"]
    trainer = PatchDictTrainer(k=32, atom_size=15)
    fb      = trainer.fit(images, device="cuda")
    fb.save("models/cdl_filters_patch.npy")

    # --- Inference ---
    fb       = FilterBank.load("models/cdl_filters_patch.npy", device="cuda")
    solver   = PatchSolver(fb, n_nonzero=5, stride=8)
    detector = IslandDetector(sigma_thresh=3.0, device="cuda")
    mc       = MADClean(fb, solver, detector, gamma=0.1, device="cuda")

    result   = mc.deconvolve("dirty.fits", "psf.fits", out_dir="results/")
    model    = result["model"]      # np.ndarray (H, W) float32
    residual = result["residual"]   # np.ndarray (H, W) float32
"""

from mad_clean.filters     import FilterBank
from mad_clean.detection   import IslandDetector
from mad_clean.solvers     import PatchSolver, ConvSolver, FlowSolver
from mad_clean.deconvolver import MADClean
from mad_clean.io          import load_image, load_image_data, save_fits
from mad_clean.flow_dict   import FlowModel, FlowTrainer
from mad_clean.psf_utils   import compute_psf_patch
from mad_clean.hogbom      import hogbom_clean

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
    "compute_psf_patch",
    "hogbom_clean",
]

__version__ = "0.1.0"
