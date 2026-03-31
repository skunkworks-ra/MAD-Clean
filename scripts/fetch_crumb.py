#!/usr/bin/env python3
"""
fetch_crumb.py — Download CRUMB and produce crumb_data/crumb_preprocessed.npz.

Downloads CRUMB_batches.tar.gz from the Jodrell Bank server (if not already
present and verified), unpickles all train + test batches, and saves:

    images : (N, 150, 150) float32, normalised to [0, 1]
    labels : (N,) int64, basic CRUMB label per image

Usage:
    python scripts/fetch_crumb.py [--out crumb_data/crumb_preprocessed.npz]

Note: train and test splits are pooled (~2100 images total). For dictionary
learning there is no concept of a held-out set, so all images are used.

Data source: http://www.jb.man.ac.uk/research/MiraBest/CRUMB/
Format: CIFAR-style pickle batches, 150×150 greyscale uint8 images.
"""

import argparse
import hashlib
import pickle
import sys
import tarfile
import urllib.request
from pathlib import Path

import numpy as np

# ── CRUMB dataset constants (from CRUMB.py) ───────────────────────────────────

URL      = "http://www.jb.man.ac.uk/research/MiraBest/CRUMB/CRUMB_batches.tar.gz"
FILENAME = "CRUMB_batches.tar.gz"
TGZ_MD5  = "a33c0564b99d66fb825e224a0392bc78"

TRAIN_BATCHES = [
    ("data_batch_1", "004e97220b29da803cf67e762ade4b52"),
    ("data_batch_2", "a05122141382c3ccec5d5c717a582b16"),
    ("data_batch_3", "aada5e8eab52732b3d171b158081bfa7"),
    ("data_batch_4", "ebc353fb9059dbeb44da28a50e6092bc"),
    ("data_batch_5", "5d9459f61a710b27b3a790d3686fb14d"),
    ("data_batch_6", "965c62bfff96acf83245e68ca42e0c10"),
]
TEST_BATCHES = [
    ("test_batch", "0cd9c3869700b720f4adcadba79d793c"),
]
ALL_BATCHES = TRAIN_BATCHES + TEST_BATCHES


# ── helpers ───────────────────────────────────────────────────────────────────

def _md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _check_integrity(path: Path, expected_md5: str) -> bool:
    return path.exists() and _md5(path) == expected_md5


def _download(root: Path) -> None:
    tgz_path = root / FILENAME
    if _check_integrity(tgz_path, TGZ_MD5):
        print("Archive already downloaded and verified.")
        return

    print(f"Downloading {URL} ...")
    def _progress(block_count, block_size, total):
        downloaded = block_count * block_size
        if total > 0:
            pct = min(100, downloaded * 100 // total)
            print(f"\r  {pct:3d}%  ({downloaded // 2**20} / {total // 2**20} MB)", end="", flush=True)

    urllib.request.urlretrieve(URL, tgz_path, reporthook=_progress)
    print()

    actual = _md5(tgz_path)
    if actual != TGZ_MD5:
        raise RuntimeError(f"MD5 mismatch: expected {TGZ_MD5}, got {actual}")

    print("Extracting...")
    with tarfile.open(tgz_path, "r:gz") as tar:
        tar.extractall(path=root)
    print("Extraction complete.")


def _load_batch(path: Path) -> tuple[np.ndarray, list]:
    with open(path, "rb") as f:
        entry = pickle.load(f, encoding="latin1")
    images  = entry["data"]                         # (300, 22500) uint8
    labels  = entry.get("labels", entry.get("fine_labels"))
    return images, labels


# ── main ──────────────────────────────────────────────────────────────────────

def fetch(out_path: Path) -> None:
    root = out_path.parent
    root.mkdir(parents=True, exist_ok=True)

    batch_dir = root / "CRUMB_batches"

    # Check if batches already extracted; if not, download + extract
    all_present = all(
        _check_integrity(batch_dir / name, md5)
        for name, md5 in ALL_BATCHES
    )
    if not all_present:
        _download(root)
    else:
        print("All batch files already present and verified.")

    # Load and stack all batches
    all_images: list[np.ndarray] = []
    all_labels: list[int] = []

    for name, md5 in ALL_BATCHES:
        batch_path = batch_dir / name
        if not _check_integrity(batch_path, md5):
            raise RuntimeError(f"Corrupt batch file: {batch_path}")
        images, labels = _load_batch(batch_path)
        all_images.append(images)
        all_labels.extend(labels)
        print(f"  Loaded {name}: {len(images)} images")

    # images are (N, 22500) uint8 — reshape to (N, 150, 150)
    images_np = np.vstack(all_images).reshape(-1, 150, 150).astype(np.float32) / 255.0
    labels_np = np.array(all_labels, dtype=np.int64)

    print(f"\nTotal images : {len(images_np)}")
    print(f"Shape        : {images_np.shape}  dtype={images_np.dtype}")
    print(f"Label range  : {labels_np.min()} – {labels_np.max()}")
    print(f"Pixel range  : [{images_np.min():.4f}, {images_np.max():.4f}]")

    np.savez(out_path, images=images_np, labels=labels_np)
    print(f"\nSaved → {out_path}")


def main() -> None:
    p = argparse.ArgumentParser(description="Download CRUMB and export to .npz")
    p.add_argument(
        "--out",
        default="crumb_data/crumb_preprocessed.npz",
        help="Output .npz path (default: crumb_data/crumb_preprocessed.npz)",
    )
    args = p.parse_args()
    fetch(Path(args.out))


if __name__ == "__main__":
    main()
