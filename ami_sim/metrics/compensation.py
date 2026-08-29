"""Warp-back sharpness diagnostics."""

from __future__ import annotations

import cv2
import numpy as np

from ami_sim.types import EventStream


def _coordinates(events: EventStream, compensated: bool) -> tuple[np.ndarray, np.ndarray]:
    if compensated and "x_float" in events.extras and "y_float" in events.extras:
        return events.extras["x_float"].astype(np.float64), events.extras["y_float"].astype(np.float64)
    return events.x.astype(np.float64), events.y.astype(np.float64)


def event_spatial_variance(events: EventStream, *, compensated: bool = False) -> dict[str, float]:
    x, y = _coordinates(events, compensated)
    if len(events) == 0:
        return {"x_variance": 0.0, "y_variance": 0.0, "radial_variance": 0.0}
    return {
        "x_variance": float(np.var(x)),
        "y_variance": float(np.var(y)),
        "radial_variance": float(np.var(np.stack((x, y), axis=1), axis=0).sum()),
    }


def edge_distance(intensity: np.ndarray, events: EventStream, *, compensated: bool = False, valid_mask: np.ndarray | None = None) -> dict[str, float]:
    image = np.clip(np.asarray(intensity) * 255.0, 0, 255).astype(np.uint8)
    edge = cv2.Canny(image, 50, 150) > 0
    if valid_mask is not None:
        edge &= valid_mask
    distance = cv2.distanceTransform((~edge).astype(np.uint8), cv2.DIST_L2, 3)
    x, y = _coordinates(events, compensated)
    x_i = np.rint(x).astype(np.int64)
    y_i = np.rint(y).astype(np.int64)
    valid = (x_i >= 0) & (y_i >= 0) & (x_i < distance.shape[1]) & (y_i < distance.shape[0])
    values = distance[y_i[valid], x_i[valid]] if np.any(valid) else np.empty(0)
    return {"mean_edge_distance": float(values.mean()) if len(values) else float("nan"), "event_count": float(len(values))}
