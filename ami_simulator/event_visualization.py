"""Deterministic event visualization for paper figures.

The renderer changes display colors only. It never changes event coordinates,
timestamps, polarity, or event counts.
"""
from __future__ import annotations

import numpy as np


BG = np.asarray([3.0, 7.0, 18.0], dtype=np.float32)
POS = np.asarray([55.0, 220.0, 255.0], dtype=np.float32)
NEG = np.asarray([225.0, 72.0, 255.0], dtype=np.float32)


def eirm_peak(counts: np.ndarray, *, lam: float = 3.0) -> float:
    """Peak of the log-compressed counts, for sharing one scale across panels."""
    return float(np.log1p(float(lam) * np.maximum(np.asarray(counts, dtype=np.float32), 0.0)).max())


def eirm_from_counts(
    positive: np.ndarray,
    negative: np.ndarray,
    *,
    lam: float = 3.0,
    gamma: float = 0.65,
    peak: tuple[float, float] | None = None,
) -> np.ndarray:
    """Map positive/negative per-pixel counts to the EIRM palette.

    ``peak`` overrides the per-panel log-compressed maxima with a shared
    (positive, negative) pair, so panels in one figure row stay comparable.
    """
    pos = np.asarray(positive, dtype=np.float32)
    neg = np.asarray(negative, dtype=np.float32)
    if pos.shape != neg.shape or pos.ndim != 2:
        raise ValueError("positive and negative must be same-shaped HxW arrays")

    def normalize(counts: np.ndarray, reference: float | None) -> np.ndarray:
        compressed = np.log1p(float(lam) * np.maximum(counts, 0.0))
        scale = float(compressed.max()) if reference is None else float(reference)
        if scale <= 1e-8:
            return np.zeros_like(compressed)
        return np.power(np.clip(compressed / scale, 0.0, 1.0), float(gamma))

    pos_level = normalize(pos, None if peak is None else peak[0])
    neg_level = normalize(neg, None if peak is None else peak[1])
    image = BG + pos_level[..., None] * (POS - BG) + neg_level[..., None] * (NEG - BG)
    return np.clip(image, 0.0, 255.0).astype(np.uint8)


def eirm_counts_from_events(events, height: int, width: int) -> tuple[np.ndarray, np.ndarray]:
    """Accumulate an EventStream into per-pixel positive and negative counts."""
    positive = np.zeros((height, width), dtype=np.float32)
    negative = np.zeros((height, width), dtype=np.float32)
    if len(events):
        x = np.rint(np.asarray(events.x)).astype(np.int64)
        y = np.rint(np.asarray(events.y)).astype(np.int64)
        p = np.asarray(events.p)
        valid = (x >= 0) & (x < width) & (y >= 0) & (y < height)
        np.add.at(positive, (y[valid & (p > 0)], x[valid & (p > 0)]), 1.0)
        np.add.at(negative, (y[valid & (p < 0)], x[valid & (p < 0)]), 1.0)
    return positive, negative


def eirm_from_events(
    events,
    height: int,
    width: int,
    *,
    lam: float = 3.0,
    gamma: float = 0.65,
    peak: tuple[float, float] | None = None,
) -> np.ndarray:
    """Accumulate an EventStream and render it using EIRM."""
    positive, negative = eirm_counts_from_events(events, height, width)
    return eirm_from_counts(positive, negative, lam=lam, gamma=gamma, peak=peak)
