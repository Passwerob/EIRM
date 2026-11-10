"""E2VID voxel grid (5-bin bilinear-in-time) and nonzero normalization."""

from __future__ import annotations

import numpy as np
import torch


def events_to_voxel_grid(
    t: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    p: np.ndarray,
    *,
    num_bins: int,
    width: int,
    height: int,
) -> np.ndarray:
    """Build a [num_bins, H, W] voxel. Polarity is +1/-1. Empty → zeros."""
    voxel = np.zeros((num_bins, height, width), dtype=np.float32)
    if t.size == 0:
        return voxel
    ts = np.asarray(t, dtype=np.float64)
    xs = np.asarray(x, dtype=np.int32)
    ys = np.asarray(y, dtype=np.int32)
    pols = np.asarray(p, dtype=np.float32)
    pols = np.where(pols >= 0, 1.0, -1.0)
    valid = (xs >= 0) & (xs < width) & (ys >= 0) & (ys < height)
    ts, xs, ys, pols = ts[valid], xs[valid], ys[valid], pols[valid]
    if ts.size == 0:
        return voxel
    delta = float(ts[-1] - ts[0])
    if delta <= 0:
        delta = 1.0
    scaled = (num_bins - 1) * (ts - ts[0]) / delta
    tis = scaled.astype(np.int32)
    dts = scaled - tis
    flat = voxel.ravel()
    left = pols * (1.0 - dts)
    right = pols * dts
    ok = (tis >= 0) & (tis < num_bins)
    np.add.at(flat, xs[ok] + ys[ok] * width + tis[ok] * width * height, left[ok])
    ok = (tis + 1 >= 0) & (tis + 1 < num_bins)
    np.add.at(flat, xs[ok] + ys[ok] * width + (tis[ok] + 1) * width * height, right[ok])
    return voxel


def normalize_nonzero(voxel: np.ndarray | torch.Tensor) -> np.ndarray | torch.Tensor:
    """Match rpg_e2vid EventPreprocessor: nonzero mean=0, std=1."""
    if isinstance(voxel, torch.Tensor):
        mask = voxel != 0
        n = mask.sum()
        if int(n.item()) == 0:
            return voxel
        mean = voxel.sum() / n
        std = torch.sqrt((voxel ** 2).sum() / n - mean ** 2).clamp_min(1e-6)
        return torch.where(mask, (voxel - mean) / std, voxel)
    mask = voxel != 0
    n = int(mask.sum())
    if n == 0:
        return voxel
    mean = float(voxel.sum() / n)
    std = float(np.sqrt(max((voxel ** 2).sum() / n - mean ** 2, 0.0)))
    if std < 1e-6:
        return voxel
    out = voxel.copy()
    out[mask] = (voxel[mask] - mean) / std
    return out
