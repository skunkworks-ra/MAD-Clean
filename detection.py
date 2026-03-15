"""
mad_clean.detection
===================
IslandDetector — threshold a residual image and return source island bounding
boxes. Thresholding and connected-component labelling run on GPU via PyTorch
iterative binary dilation. No scipy dependency at inference time.

Classes
-------
IslandDetector
    .detect(residual: Tensor) -> List[Tuple[int,int,int,int]]
        Returns a list of (r0, r1, c0, c1) bounding boxes, one per island.
"""

from __future__ import annotations

from typing import List, Tuple

import torch
import torch.nn.functional as F


__all__ = ["IslandDetector"]


class IslandDetector:
    """
    GPU source island detector.

    Algorithm
    ---------
    1. Compute residual RMS on GPU.
    2. Threshold at sigma_thresh × RMS → binary mask (GPU tensor).
    3. Label connected components via iterative binary dilation (GPU).
    4. Extract bounding boxes, padded by atom_size // 2.

    Parameters
    ----------
    sigma_thresh : float  detection threshold in units of residual RMS (default 3.0)
    min_island   : int    minimum island area in pixels (default 9)
    atom_size    : int    atom footprint in pixels; bounding boxes are padded
                          by atom_size // 2 on each side (default 15)
    device       : str | torch.device
    max_dilation_iter : int
                   Maximum iterations for the connected-component dilation loop.
                   For 150×150 images with compact islands, 150 is conservative
                   upper bound. (default 150)
    """

    def __init__(
        self,
        sigma_thresh    : float = 3.0,
        min_island      : int   = 9,
        atom_size       : int   = 15,
        device          : str | torch.device = "cpu",
        max_dilation_iter: int  = 150,
    ):
        self.sigma_thresh     = sigma_thresh
        self.min_island       = min_island
        self.atom_size        = atom_size
        self.device           = torch.device(device)
        self.max_dilation_iter = max_dilation_iter

        # 3×3 connectivity kernel for 8-connected dilation
        self._kernel = torch.ones(
            1, 1, 3, 3, dtype=torch.float32, device=self.device
        )

    def detect(
        self,
        residual: torch.Tensor,
    ) -> Tuple[List[Tuple[int, int, int, int]], float]:
        """
        Detect source islands in a residual image.

        Parameters
        ----------
        residual : torch.Tensor (H, W) float32, on self.device

        Returns
        -------
        bboxes : List of (r0, r1, c0, c1) int tuples
                 Bounding boxes padded by atom_size // 2.
                 Empty list if no islands found above threshold.
        rms    : float  residual RMS used for thresholding
        """
        if residual.device != self.device:
            residual = residual.to(self.device)

        rms = float(residual.std())
        if rms == 0.0:
            return [], rms

        # ── 1. threshold ──────────────────────────────────────────────────────
        binary = (residual > self.sigma_thresh * rms).float()   # (H, W)

        if binary.sum() == 0:
            return [], rms

        # ── 2. connected components via iterative dilation ────────────────────
        labels = self._label_components(binary)   # (H, W) int32, 0 = background

        # ── 3. bounding boxes ─────────────────────────────────────────────────
        bboxes = self._extract_bboxes(labels, residual.shape)

        return bboxes, rms

    def _label_components(self, binary: torch.Tensor) -> torch.Tensor:
        """
        Label connected components of a binary (H, W) float tensor.

        Strategy: assign each foreground pixel a unique initial label equal to
        its flat index. Iteratively propagate the minimum label within each
        connected neighbourhood until convergence (no label changes).

        This is equivalent to a parallel union-find via min-pooling dilation
        and converges in O(diameter) iterations — for compact radio sources on
        150×150 images this is fast.

        Returns integer label tensor (H, W), 0 = background.

        Implementation note on sentinel masking
        ----------------------------------------
        Naively negating labels and max-pooling fails because background pixels
        (label 0) become 0 after negation, which is the maximum — they
        contaminate every foreground pixel that has a background neighbour.

        Fix: replace background labels with a sentinel value S = H*W+2 (larger
        than any foreground label). In negated space, -S is smaller than any
        negated foreground label, so background never wins the max. Image
        boundary padding is also set to -S for the same reason.

        After dilation, foreground pixels take the min of their current label
        and the propagated label — isolated pixels (only background neighbours)
        return S from the pool and min(S, label) = label, so they keep their
        own label unchanged.
        """
        H, W     = binary.shape
        flat_idx = torch.arange(1, H * W + 1, dtype=torch.float32,
                                device=self.device).reshape(H, W)
        labels   = binary * flat_idx   # background = 0, foreground = unique index

        # Sentinel: larger than any valid foreground label so it sorts last.
        sentinel     = float(H * W + 2)
        neg_sentinel = -sentinel       # used for pad fill — sorts first in max-pool

        for _ in range(self.max_dilation_iter):
            # Replace background with sentinel so it cannot win the max-pool.
            labels_fg  = torch.where(
                binary > 0, labels,
                torch.tensor(sentinel, dtype=torch.float32, device=self.device),
            )                                                      # (H, W)

            # Negate: minimum label becomes maximum; sentinel becomes most negative.
            neg_lab    = -labels_fg.unsqueeze(0).unsqueeze(0)     # (1, 1, H, W)

            # Manual pad with neg_sentinel so image boundary never contaminates.
            neg_padded = F.pad(neg_lab, (1, 1, 1, 1), value=neg_sentinel)
            neg_dilated = F.max_pool2d(neg_padded, kernel_size=3,
                                       stride=1, padding=0)        # (1, 1, H, W)
            dilated    = -neg_dilated.squeeze()                    # (H, W) — min foreground label in 3×3

            # For each foreground pixel: take the smaller of its current label
            # and the neighbourhood minimum. Isolated pixels see dilated=sentinel;
            # min(sentinel, label) = label — no change.
            new_labels = torch.where(
                binary > 0,
                torch.minimum(dilated, labels),
                labels,
            )
            if torch.equal(new_labels, labels):
                break
            labels = new_labels

        # Re-index labels to consecutive integers 1..n_components
        return self._reindex_labels(labels, binary)

    @staticmethod
    def _reindex_labels(
        labels: torch.Tensor,
        binary: torch.Tensor,
    ) -> torch.Tensor:
        """Map arbitrary float label values to consecutive integers 1..N."""
        unique = torch.unique(labels)
        unique = unique[unique > 0]   # exclude background
        out    = torch.zeros_like(labels, dtype=torch.int32)
        for new_id, val in enumerate(unique, start=1):
            out[labels == val] = new_id
        return out

    def _extract_bboxes(
        self,
        labels  : torch.Tensor,   # (H, W) int32
        shape   : Tuple[int, int],
    ) -> List[Tuple[int, int, int, int]]:
        """
        Extract padded bounding boxes for each labelled component.
        Filters out components smaller than self.min_island pixels.
        Bounding boxes are padded by atom_size // 2 and clipped to image bounds.
        """
        H, W  = shape
        pad   = self.atom_size // 2
        n_lab = int(labels.max().item())
        bboxes: List[Tuple[int, int, int, int]] = []

        for i in range(1, n_lab + 1):
            mask = (labels == i)
            if mask.sum().item() < self.min_island:
                continue
            # Move to CPU for index arithmetic (cheap — just 4 scalars)
            rows, cols = torch.where(mask)
            r0 = int(max(0,     rows.min().item() - pad))
            r1 = int(min(H,     rows.max().item() + pad + 1))
            c0 = int(max(0,     cols.min().item() - pad))
            c1 = int(min(W,     cols.max().item() + pad + 1))
            bboxes.append((r0, r1, c0, c1))

        return bboxes

    def to(self, device: str | torch.device) -> "IslandDetector":
        """Return a new IslandDetector on a different device."""
        new = IslandDetector(
            sigma_thresh=self.sigma_thresh,
            min_island=self.min_island,
            atom_size=self.atom_size,
            device=device,
            max_dilation_iter=self.max_dilation_iter,
        )
        return new

    def __repr__(self) -> str:
        return (f"IslandDetector(sigma={self.sigma_thresh}, "
                f"min_island={self.min_island}, "
                f"atom_size={self.atom_size}, "
                f"device={self.device})")
