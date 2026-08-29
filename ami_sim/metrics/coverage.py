"""Event/structure metrics that explicitly honor the all-orientation valid mask."""

from __future__ import annotations

import cv2
import numpy as np

from ami_sim.types import EventStream


def edge_mask(intensity: np.ndarray, *, low: int = 50, high: int = 150) -> np.ndarray:
    image = np.clip(np.asarray(intensity) * 255.0, 0, 255).astype(np.uint8)
    return cv2.Canny(image, low, high) > 0


def _event_pixels(events: EventStream, valid_mask: np.ndarray | None = None, *, compensated: bool = False) -> np.ndarray:
    if valid_mask is None and len(events) == 0:
        return np.zeros((0, 0), dtype=bool)
    height, width = valid_mask.shape if valid_mask is not None else (
        int(events.y.max(initial=0)) + 1, int(events.x.max(initial=0)) + 1
    )
    pixels = np.zeros((height, width), dtype=bool)
    if compensated and "x_float" in events.extras and "y_float" in events.extras:
        x = np.rint(events.extras["x_float"]).astype(np.int64)
        y = np.rint(events.extras["y_float"]).astype(np.int64)
    else:
        x = events.x
        y = events.y
    valid = (x >= 0) & (y >= 0) & (x < width) & (y < height)
    if valid_mask is not None:
        valid &= valid_mask[y.clip(0, height - 1), x.clip(0, width - 1)]
    pixels[y[valid], x[valid]] = True
    return pixels


def edge_coverage(intensity: np.ndarray, events: EventStream, valid_mask: np.ndarray | None = None, *, compensated: bool = False) -> dict[str, float]:
    edges = edge_mask(intensity)
    if valid_mask is not None:
        edges &= valid_mask
    event_pixels = _event_pixels(events, valid_mask, compensated=compensated)
    denominator = max(int(edges.sum()), 1)
    return {
        "edge_pixels": float(edges.sum()),
        "event_pixels": float(event_pixels.sum()),
        "covered_edge_pixels": float((edges & event_pixels).sum()),
        "coverage": float((edges & event_pixels).sum() / denominator),
        "event_count": float(len(events)),
    }


def newly_recovered_structure(
    intensity: np.ndarray,
    vanilla: EventStream,
    ami: EventStream,
    valid_mask: np.ndarray | None = None,
    *,
    ami_compensated: bool = False,
) -> dict[str, float]:
    edges = edge_mask(intensity)
    if valid_mask is not None:
        edges &= valid_mask
    vanilla_pixels = _event_pixels(vanilla, valid_mask)
    ami_pixels = _event_pixels(ami, valid_mask, compensated=ami_compensated)
    new_pixels = ami_pixels & edges & ~vanilla_pixels
    return {
        "edge_pixels": float(edges.sum()),
        "newly_recovered_edge_pixels": float(new_pixels.sum()),
        "new_coverage": float(new_pixels.sum() / max(1, int(edges.sum()))),
        "additional_event_count": float(max(0, len(ami) - len(vanilla))),
        "efficiency": float(new_pixels.sum() / max(1, len(ami) - len(vanilla))),
    }
