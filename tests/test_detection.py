"""
tests/test_detection.py — tests for mad_clean.detection.IslandDetector
"""

import numpy as np
import torch

from mad_clean.detection import IslandDetector


# ── helpers ───────────────────────────────────────────────────────────────────

def _blank(h: int = 64, w: int = 64) -> torch.Tensor:
    return torch.zeros(h, w, dtype=torch.float32)


def _add_blob(t: torch.Tensor, cy: int, cx: int, radius: int, value: float) -> torch.Tensor:
    """Add a filled circular blob to a 2D tensor."""
    t = t.clone()
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            if dy ** 2 + dx ** 2 <= radius ** 2:
                ry, rx = cy + dy, cx + dx
                if 0 <= ry < t.shape[0] and 0 <= rx < t.shape[1]:
                    t[ry, rx] = value
    return t


# ── tests ─────────────────────────────────────────────────────────────────────

def test_empty_image_no_islands():
    """All-zeros residual returns empty island list."""
    det   = IslandDetector(sigma_thresh=3.0, device="cpu")
    boxes, rms = det.detect(_blank())
    assert boxes == []


def test_single_point_source_detected():
    """A bright point above 3σ is detected as one island."""
    img  = _blank()
    img[32, 32] = 100.0
    det  = IslandDetector(sigma_thresh=3.0, min_island=1, device="cpu")
    boxes, _ = det.detect(img)
    assert len(boxes) >= 1


def test_two_point_sources_detected():
    """Two well-separated blobs are returned as two distinct islands."""
    img  = _blank(64, 64)
    img  = _add_blob(img, 10, 10, radius=3, value=50.0)
    img  = _add_blob(img, 50, 50, radius=3, value=50.0)
    det  = IslandDetector(sigma_thresh=3.0, min_island=1, atom_size=5, device="cpu")
    boxes, _ = det.detect(img)
    assert len(boxes) == 2


def test_bbox_padding():
    """Bounding boxes are padded by atom_size // 2 on each side."""
    img         = _blank(64, 64)
    img[30, 30] = 100.0           # single foreground pixel
    atom_size   = 11
    pad         = atom_size // 2  # 5
    det  = IslandDetector(sigma_thresh=1.0, min_island=1, atom_size=atom_size, device="cpu")
    boxes, _ = det.detect(img)
    assert len(boxes) >= 1
    r0, r1, c0, c1 = boxes[0]
    # The original foreground pixel is at (30, 30); after padding, bbox must
    # extend at least pad pixels in each direction (subject to image boundary).
    assert r0 <= 30 - pad or r0 == 0
    assert r1 >= 30 + pad + 1 or r1 == 64
    assert c0 <= 30 - pad or c0 == 0
    assert c1 >= 30 + pad + 1 or c1 == 64


def test_min_island_filter():
    """Islands below min_island pixels are removed."""
    img         = _blank(64, 64)
    img[32, 32] = 100.0   # single-pixel island → area = 1
    det  = IslandDetector(sigma_thresh=1.0, min_island=9, device="cpu")
    boxes, _ = det.detect(img)
    assert boxes == []


def test_bbox_clipped_to_image_bounds():
    """Bounding boxes never exceed image dimensions."""
    img  = _blank(32, 32)
    img  = _add_blob(img, 1, 1, radius=2, value=50.0)   # blob near top-left corner
    det  = IslandDetector(sigma_thresh=1.0, min_island=1, atom_size=15, device="cpu")
    boxes, _ = det.detect(img)
    for r0, r1, c0, c1 in boxes:
        assert r0 >= 0
        assert r1 <= 32
        assert c0 >= 0
        assert c1 <= 32


def test_rms_returned():
    """detect() returns a float RMS value as second element."""
    img       = _blank()
    img[0, 0] = 1.0
    det       = IslandDetector(device="cpu")
    _, rms    = det.detect(img)
    assert isinstance(rms, float)
    assert rms > 0
