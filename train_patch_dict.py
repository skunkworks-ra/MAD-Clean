"""
train_patch_dict.py — Variant A filter bank training
Retrain sklearn MiniBatchDictionaryLearning on CRUMB patches at 15×15px.
Saves: models/cdl_filters_patch.npy  shape (K, 15, 15)

Usage:
    python train_patch_dict.py --data crumb_data/crumb_preprocessed.npz
"""

import argparse
import numpy as np
from pathlib import Path
from sklearn.decomposition import MiniBatchDictionaryLearning
from sklearn.feature_extraction.image import extract_patches_2d


# ── defaults ──────────────────────────────────────────────────────────────────
ATOM_SIZE   = 15       # px — empirically validated
K           = 32       # number of atoms
ALPHA       = 0.1      # sparsity regularisation
N_ITER      = 1000     # mini-batch iterations
BATCH_SIZE  = 512
RANDOM_SEED = 42


def extract_patches(images: np.ndarray, atom_size: int, patches_per_image: int = 20,
                    rng: np.random.Generator = None) -> np.ndarray:
    """
    Extract random patches from a stack of images.
    images: (N, H, W) float32
    Returns: (N * patches_per_image, atom_size * atom_size) float32
    """
    if rng is None:
        rng = np.random.default_rng(RANDOM_SEED)

    all_patches = []
    for img in images:
        # Random rotation of full image before patch extraction
        # (avoids border interpolation artefacts on sub-patches)
        angle_deg = rng.uniform(0, 360)
        img_rot = _rotate(img, angle_deg)
        patches = extract_patches_2d(
            img_rot,
            patch_size=(atom_size, atom_size),
            max_patches=patches_per_image,
            random_state=int(rng.integers(0, 2**31))
        )
        all_patches.append(patches.reshape(len(patches), -1))

    patches_2d = np.vstack(all_patches).astype(np.float32)
    print(f"  Extracted {len(patches_2d)} patches of size {atom_size}×{atom_size}")
    return patches_2d


def _rotate(image: np.ndarray, angle_deg: float) -> np.ndarray:
    """Rotate a 2D image by angle_deg using scipy."""
    from scipy.ndimage import rotate
    return rotate(image, angle_deg, reshape=False, order=1, mode='reflect')


def train(data_path: str, out_dir: str, k: int, atom_size: int,
          alpha: float, n_iter: int) -> None:

    data_path = Path(data_path)
    out_dir   = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── load data ─────────────────────────────────────────────────────────────
    print(f"Loading {data_path}")
    npz    = np.load(data_path)
    images = npz["images"].astype(np.float32)   # (N, 150, 150)
    labels = npz["labels"]
    print(f"  Images: {images.shape}  Labels: {labels.shape}")

    # ── train / test split (80/20, stratified by label) ──────────────────────
    rng = np.random.default_rng(RANDOM_SEED)
    n   = len(images)
    idx = np.arange(n)
    rng.shuffle(idx)
    split      = int(0.8 * n)
    train_idx  = idx[:split]
    train_imgs = images[train_idx]

    # ── extract patches ───────────────────────────────────────────────────────
    print("Extracting patches …")
    patches = extract_patches(train_imgs, atom_size, patches_per_image=20, rng=rng)

    # Per-patch normalise to zero mean, unit variance (standard for DL)
    patch_mean = patches.mean(axis=1, keepdims=True)
    patch_std  = patches.std(axis=1, keepdims=True) + 1e-8
    patches_n  = (patches - patch_mean) / patch_std

    # ── train dictionary ──────────────────────────────────────────────────────
    print(f"Training MiniBatchDictionaryLearning  K={k}  alpha={alpha}  "
          f"n_iter={n_iter} …")
    dl = MiniBatchDictionaryLearning(
        n_components=k,
        alpha=alpha,
        max_iter=n_iter,
        batch_size=BATCH_SIZE,
        fit_algorithm="lars",
        transform_algorithm="omp",
        transform_n_nonzero_coefs=5,   # default sparsity S=5
        n_jobs=-1,
        random_state=RANDOM_SEED,
        verbose=1,
    )
    dl.fit(patches_n)

    # ── save atoms ────────────────────────────────────────────────────────────
    atoms = dl.components_.reshape(k, atom_size, atom_size).astype(np.float32)
    out_path = out_dir / "cdl_filters_patch.npy"
    np.save(out_path, atoms)
    print(f"Saved {atoms.shape} atoms → {out_path}")

    # ── quick sanity: atom norm distribution ─────────────────────────────────
    norms = np.linalg.norm(atoms.reshape(k, -1), axis=1)
    print(f"Atom L2 norms  min={norms.min():.3f}  "
          f"mean={norms.mean():.3f}  max={norms.max():.3f}")
    dead = (norms < 0.01).sum()
    if dead > 0:
        print(f"  WARNING: {dead}/{k} dead atoms (norm < 0.01) — consider "
              f"increasing alpha or n_iter")
    else:
        print(f"  All {k} atoms active.")


def main():
    p = argparse.ArgumentParser(description="Train Variant A patch dictionary")
    p.add_argument("--data",      default="crumb_data/crumb_preprocessed.npz")
    p.add_argument("--out_dir",   default="models")
    p.add_argument("--k",         type=int,   default=K)
    p.add_argument("--atom_size", type=int,   default=ATOM_SIZE)
    p.add_argument("--alpha",     type=float, default=ALPHA)
    p.add_argument("--n_iter",    type=int,   default=N_ITER)
    args = p.parse_args()

    train(args.data, args.out_dir, args.k, args.atom_size, args.alpha, args.n_iter)


if __name__ == "__main__":
    main()
