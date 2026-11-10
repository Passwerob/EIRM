"""Small deterministic Contrast Maximization (CM) event representation."""

from __future__ import annotations

from typing import Iterable

import numpy as np

from .types import EventStream


def _warped_image(events: EventStream, shape: tuple[int, int], vx: float, vy: float, reference_time: float) -> np.ndarray:
    height, width = map(int, shape)
    image = np.zeros((2, height, width), dtype=np.float64)
    if not len(events):
        return image
    x = np.rint(events.x.astype(np.float64) - vx * (events.t - reference_time)).astype(np.int64)
    y = np.rint(events.y.astype(np.float64) - vy * (events.t - reference_time)).astype(np.int64)
    valid = (x >= 0) & (x < width) & (y >= 0) & (y < height)
    polarity = (events.p > 0).astype(np.int64)
    np.add.at(image, (polarity[valid], y[valid], x[valid]), 1.0)
    return image


def contrast_maximization(
    events: EventStream,
    image_shape: tuple[int, int],
    velocity_grid: Iterable[tuple[float, float]],
    *,
    reference_time: float | None = None,
) -> dict[str, object]:
    """Select the candidate event warp with maximum polarity-separated contrast.

    The grid, timestamp window, and image shape must be identical for every
    event source. This transparent translation-only baseline is intended for a
    controlled downstream comparison, not as a learned motion estimator.
    """
    tref = float(np.median(events.t) if len(events) and reference_time is None else (0.0 if reference_time is None else reference_time))
    candidates = list(velocity_grid)
    if not candidates:
        raise ValueError("velocity_grid must contain at least one candidate.")
    scores: list[float] = []
    images: list[np.ndarray] = []
    for velocity in candidates:
        if len(velocity) != 2 or not np.all(np.isfinite(np.asarray(velocity, dtype=np.float64))):
            raise ValueError("Each velocity candidate must contain two finite values.")
        image = _warped_image(events, image_shape, float(velocity[0]), float(velocity[1]), tref)
        images.append(image)
        scores.append(float(np.var(image[0]) + np.var(image[1])))
    best = int(np.argmax(np.asarray(scores, dtype=np.float64)))
    return {"velocity": (float(candidates[best][0]), float(candidates[best][1])), "contrast": scores[best], "scores": np.asarray(scores, dtype=np.float64), "image": images[best], "reference_time": tref}
