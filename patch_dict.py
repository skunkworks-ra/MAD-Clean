"""
mad_clean.train.patch_dict
==========================
PatchDictTrainer — trains a patch dictionary via sklearn
MiniBatchDictionaryLearning and returns a FilterBank.

Classes
-------
PatchDictTrainer
    .fit(images) -> FilterBank
    .save(path)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.ndimage import rotate
from sklearn.decomposition import MiniBatchDictionaryLearning
from sklearn.feature_extraction.image import extract_patches_2d

from mad_clean.filters import FilterBank


__all__ = ["PatchDictTrainer"]


class PatchDictTrainer:
    """
    Train a patch dictionary on CRUMB-style radio galaxy images.

    Parameters
    ----------
    k               : int    number of atoms (default 32)
    atom_size       : int    patch / atom size in pixels (default 15)
    alpha           : float  sklearn sparsity regularisation (default 0.1)
    n_iter          : int    mini-batch iterations (default 1000)
    patches_per_img : int    random patches extracted per image (default 20)
    batch_size      : int    mini-batch size (default 512)
    random_seed     : int    (default 42)
    """

    def __init__(
        self,
        k               : int   = 32,
        atom_size       : int   = 15,
        alpha           : float = 0.1,
        n_iter          : int   = 1000,
        patches_per_img : int   = 20,
        batch_size      : int   = 512,
        random_seed     : int   = 42,
    ):
        self.k               = k
        self.atom_size       = atom_size
        self.alpha           = alpha
        self.n_iter          = n_iter
        self.patches_per_img = patches_per_img
        self.batch_size      = batch_size
        self.random_seed     = random_seed
        self._dl             = None   # sklearn model, set after fit()

    def fit(
        self,
        images : np.ndarray,
        device : str = "cpu",
    ) -> FilterBank:
        """
        Train on (N, H, W) float32 images. Returns a FilterBank.

        Parameters
        ----------
        images : np.ndarray (N, H, W) float32, values in [0, 1]
        device : torch device for the returned FilterBank (default "cpu")
        """
        rng = np.random.default_rng(self.random_seed)

        # ── 80/20 train split ──────────────────────────────────────────────
        n     = len(images)
        idx   = np.arange(n)
        rng.shuffle(idx)
        train = images[idx[:int(0.8 * n)]]

        # ── patch extraction with random rotation ─────────────────────────
        print(f"Extracting patches  "
              f"(n_images={len(train)}, patches/img={self.patches_per_img}) …")
        patches = self._extract_patches(train, rng)
        print(f"  Total patches: {len(patches)}  "
              f"shape: {self.atom_size}×{self.atom_size}")

        # Per-patch normalise: zero mean, unit variance
        p_mean = patches.mean(axis=1, keepdims=True)
        p_std  = patches.std (axis=1, keepdims=True) + 1e-8
        patches_n = (patches - p_mean) / p_std

        # ── train dictionary ───────────────────────────────────────────────
        print(f"Training MiniBatchDictionaryLearning  "
              f"K={self.k}  alpha={self.alpha}  n_iter={self.n_iter} …")
        self._dl = MiniBatchDictionaryLearning(
            n_components=self.k,
            alpha=self.alpha,
            max_iter=self.n_iter,
            batch_size=self.batch_size,
            fit_algorithm="lars",
            transform_algorithm="omp",
            transform_n_nonzero_coefs=5,
            n_jobs=-1,
            random_state=self.random_seed,
            verbose=1,
        )
        self._dl.fit(patches_n)

        atoms = (self._dl.components_
                 .reshape(self.k, self.atom_size, self.atom_size)
                 .astype(np.float32))

        fb = FilterBank(atoms, device=device)
        report = fb.dead_atom_report()
        print(f"  Active atoms: {report['n_active']}/{self.k}  "
              f"norm mean={report['norm_mean']:.3f}")
        if report["n_dead"] > 0:
            print(f"  WARNING: {report['n_dead']} dead atoms — "
                  f"consider increasing alpha or n_iter")
        return fb

    def save(self, path: str | Path, filter_bank: FilterBank) -> None:
        """Save the FilterBank atoms to a .npy file."""
        filter_bank.save(path)
        print(f"Saved FilterBank → {path}")

    # ── internals ─────────────────────────────────────────────────────────────

    def _extract_patches(
        self,
        images : np.ndarray,
        rng    : np.random.Generator,
    ) -> np.ndarray:
        """
        Extract random patches from images with random full-image rotation.
        Rotation is applied to the full image before patch extraction to avoid
        border interpolation artefacts at patch boundaries.
        """
        A   = self.atom_size
        all_patches = []

        for img in images:
            angle   = rng.uniform(0, 360)
            img_rot = rotate(img, angle, reshape=False,
                             order=1, mode="reflect")
            patches = extract_patches_2d(
                img_rot,
                patch_size=(A, A),
                max_patches=self.patches_per_img,
                random_state=int(rng.integers(0, 2**31)),
            )
            all_patches.append(patches.reshape(len(patches), -1))

        return np.vstack(all_patches).astype(np.float32)

    def __repr__(self) -> str:
        return (f"PatchDictTrainer(k={self.k}, atom_size={self.atom_size}, "
                f"alpha={self.alpha}, n_iter={self.n_iter})")
