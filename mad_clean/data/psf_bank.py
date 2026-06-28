"""PSF bank loader for SBI-Asp training.

Loads a family of PSFs from disk, normalises shape via peak-centred
crop/pad, peak-normalises amplitude, and (optionally) augments with
exact 90-degree rotations. Used to marginalise the MDN over a realistic
uv-coverage family during training.

v1 scope: G55 D-config L-band only (10 chunks → 40 with rotation aug).
Cross-config generalisation is deferred to v1.1; do not add casa_sim
cross-config work here.

PSF files are NOT in git (data/g55/chunk_*/psf.fits, ~1.7 GB total).
rsync separately from the data store.
"""
from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
from astropy.io import fits


def _load_fits_psf(path: Path) -> np.ndarray:
    with fits.open(path, memmap=False) as hdul:
        data = hdul[0].data
    if data is None:
        raise ValueError(f"No data in primary HDU of {path}")
    arr = np.squeeze(np.asarray(data)).astype(np.float32)
    if arr.ndim != 2:
        raise ValueError(
            f"Expected 2D PSF after squeeze, got shape {arr.shape} from {path}"
        )
    return arr


def _peak_centred_resize(psf: np.ndarray, target_size: int) -> np.ndarray:
    """Crop or zero-pad ``psf`` to ``(target_size, target_size)``
    centred on the PSF peak."""
    py, px = np.unravel_index(int(np.argmax(psf)), psf.shape)
    H, W = psf.shape
    T = target_size
    half = T // 2  # target peak index

    out = np.zeros((T, T), dtype=np.float32)

    # Source slice: clamp [py - half, py - half + T) into [0, H)
    src_y0 = py - half
    src_y1 = src_y0 + T
    src_x0 = px - half
    src_x1 = src_x0 + T

    # Clip to source bounds
    s_y0 = max(0, src_y0)
    s_y1 = min(H, src_y1)
    s_x0 = max(0, src_x0)
    s_x1 = min(W, src_x1)

    # Corresponding destination region
    d_y0 = s_y0 - src_y0
    d_y1 = d_y0 + (s_y1 - s_y0)
    d_x0 = s_x0 - src_x0
    d_x1 = d_x0 + (s_x1 - s_x0)

    out[d_y0:d_y1, d_x0:d_x1] = psf[s_y0:s_y1, s_x0:s_x1]
    return out


class PSFBank:
    """Bank of PSFs sampled during conditional flow training."""

    def __init__(
        self,
        psf_paths: Sequence[str | Path],
        target_size: int = 512,
        rotation_augment: bool = True,
    ):
        if len(psf_paths) == 0:
            raise ValueError("psf_paths is empty")
        self.target_size = int(target_size)
        self.rotation_augment = bool(rotation_augment)

        rotations = (0, 90, 180, 270) if self.rotation_augment else (0,)

        self._psfs: list[np.ndarray] = []
        self._meta: list[dict] = []
        for p in psf_paths:
            path = Path(p)
            raw = _load_fits_psf(path)
            base = _peak_centred_resize(raw, self.target_size)
            peak = float(base.max())
            if not np.isfinite(peak) or peak <= 0.0:
                raise ValueError(f"Non-positive/non-finite PSF peak in {path}")
            base = (base / peak).astype(np.float32)
            for k, deg in enumerate(rotations):
                rot = np.rot90(base, k=k) if k else base
                self._psfs.append(np.ascontiguousarray(rot, dtype=np.float32))
                self._meta.append({"source_path": str(path), "rotation_deg": int(deg)})

    def __len__(self) -> int:
        return len(self._psfs)

    def __getitem__(self, i: int) -> tuple[np.ndarray, dict]:
        return self._psfs[i].copy(), dict(self._meta[i])

    def sample(self, rng: np.random.Generator) -> tuple[np.ndarray, dict]:
        idx = int(rng.integers(0, len(self._psfs)))
        return self[idx]


def load_corpus_psf_bank(
    corpus_fits_dir: str | Path,
    target_size: int = 128,
    rotation_augment: bool = True,
) -> PSFBank:
    """Load all ``corpus_field_XXXX_psf.fits`` PSFs from a corpus directory.

    These span multiple VLA configurations and frequencies, giving realistic
    PSF diversity for SBI training.
    """
    d = Path(corpus_fits_dir)
    paths = sorted(d.glob("corpus_field_*_psf.fits"))
    if not paths:
        raise FileNotFoundError(f"No corpus_field_*_psf.fits found in {d}")
    return PSFBank(psf_paths=paths, target_size=target_size,
                   rotation_augment=rotation_augment)


def load_g55_psf_bank(
    repo_root: str | Path,
    target_size: int = 512,
    rotation_augment: bool = True,
    exclude: list[str] | None = None,
) -> PSFBank:
    """Load all ``data/g55/chunk_*/psf.fits`` PSFs into a :class:`PSFBank`.

    ``exclude`` matches chunk labels (the suffix after ``chunk_``), e.g.
    ``["scan"]`` skips ``data/g55/chunk_scan/psf.fits``.
    """
    root = Path(repo_root)
    g55 = root / "data" / "g55"
    if not g55.is_dir():
        raise FileNotFoundError(f"No such directory: {g55}")

    excluded = set(exclude or [])
    paths: list[Path] = []
    for d in sorted(g55.glob("chunk_*")):
        if not d.is_dir():
            continue
        label = d.name[len("chunk_"):]
        if label in excluded:
            continue
        psf_path = d / "psf.fits"
        if psf_path.is_file():
            paths.append(psf_path)

    if not paths:
        raise FileNotFoundError(
            f"No chunk_*/psf.fits found under {g55} (after exclude={sorted(excluded)})"
        )

    return PSFBank(
        psf_paths=paths,
        target_size=target_size,
        rotation_augment=rotation_augment,
    )


def load_psf_bank_from_npy(
    npy_path: str | Path,
    target_size: int = 128,
    rotation_augment: bool = True,
) -> "PSFBankNpy":
    """Load PSFs directly from a stacks psf.npy array.

    The stacks psf.npy has shape (N_fields, H, W) where each entry is a
    peak-normalised PSF from the corpus.  No FITS I/O required -- works
    on remote machines that have the stacks but not the raw FITS files.
    """
    return PSFBankNpy(
        npy_path=Path(npy_path),
        target_size=target_size,
        rotation_augment=rotation_augment,
    )


class PSFBankNpy:
    """PSF bank backed by a stacks psf.npy array instead of FITS files."""

    def __init__(self, npy_path: Path, target_size: int, rotation_augment: bool):
        raw = np.load(npy_path, mmap_mode="r")   # (N_fields, H, W)
        rotations = (0, 90, 180, 270) if rotation_augment else (0,)
        self._psfs: list[np.ndarray] = []
        for i in range(len(raw)):
            base = _peak_centred_resize(
                np.asarray(raw[i], dtype=np.float32), target_size)
            peak = float(base.max())
            if not np.isfinite(peak) or peak <= 0.0:
                continue
            base = (base / peak).astype(np.float32)
            for k in range(len(rotations)):
                rot = np.rot90(base, k=k) if k else base
                self._psfs.append(np.ascontiguousarray(rot, dtype=np.float32))
        if not self._psfs:
            raise ValueError(f"No valid PSFs loaded from {npy_path}")

    def __len__(self) -> int:
        return len(self._psfs)

    def __getitem__(self, i: int) -> tuple[np.ndarray, dict]:
        return self._psfs[i].copy(), {}

    def sample(self, rng: np.random.Generator) -> tuple[np.ndarray, dict]:
        idx = int(rng.integers(0, len(self._psfs)))
        return self[idx]
