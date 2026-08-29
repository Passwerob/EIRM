"""Natural-motion and prism-motion diagnostics; never used to synthesize frames."""

from __future__ import annotations

import cv2
import numpy as np

from ami_sim.types import EventStream


def relative_motion_diagnostics(
    intensity0: np.ndarray,
    intensity1: np.ndarray,
    prism_flow: np.ndarray,
    *,
    gradient_threshold: float = 0.02,
    cancellation_threshold: float = 0.1,
) -> dict[str, float]:
    first = np.asarray(intensity0, dtype=np.float32)
    second = np.asarray(intensity1, dtype=np.float32)
    first_u8 = np.clip(first * 255.0, 0, 255).astype(np.uint8)
    second_u8 = np.clip(second * 255.0, 0, 255).astype(np.uint8)
    flow = cv2.calcOpticalFlowFarneback(first_u8, second_u8, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    gx = cv2.Sobel(first, cv2.CV_32F, 1, 0, ksize=3) / 8.0
    gy = cv2.Sobel(first, cv2.CV_32F, 0, 1, ksize=3) / 8.0
    gradient = np.stack((gx, gy), axis=-1)
    natural = flow.astype(np.float32)
    effective = natural + np.asarray(prism_flow, dtype=np.float32)
    natural_magnitude = np.linalg.norm(natural, axis=-1)
    prism_magnitude = np.linalg.norm(prism_flow, axis=-1)
    effective_magnitude = np.linalg.norm(effective, axis=-1)
    gradient_magnitude = np.linalg.norm(gradient, axis=-1)
    projected_natural = np.sum(gradient * natural, axis=-1)
    projected_effective = np.sum(gradient * effective, axis=-1)
    denominator = np.maximum(gradient_magnitude * natural_magnitude, 1e-8)
    angle = np.arccos(np.clip(projected_natural / denominator, -1.0, 1.0))
    valid_gradient = gradient_magnitude > gradient_threshold
    return {
        "mean_natural_motion_px": float(natural_magnitude.mean()),
        "mean_prism_motion_px": float(prism_magnitude.mean()),
        "mean_effective_motion_px": float(effective_magnitude.mean()),
        "mean_abs_gradient_dot_natural_motion": float(np.mean(np.abs(projected_natural))),
        "mean_abs_gradient_dot_effective_motion": float(np.mean(np.abs(projected_effective))),
        "low_observability_fraction": float(np.mean(np.abs(projected_natural[valid_gradient]) < cancellation_threshold)) if np.any(valid_gradient) else 0.0,
        "near_cancellation_fraction": float(np.mean(effective_magnitude < cancellation_threshold)),
        "mean_gradient_motion_angle_rad": float(np.mean(angle[valid_gradient])) if np.any(valid_gradient) else 0.0,
    }


def event_observability_stats(
    intensity: np.ndarray,
    events: EventStream,
    prism_flow: np.ndarray,
    *,
    valid_mask: np.ndarray | None = None,
    threshold: float = 0.02,
) -> dict[str, object]:
    """Measure gradient alignment at raw events without changing their formation."""
    image = np.asarray(intensity, dtype=np.float32)
    gx = cv2.Sobel(image, cv2.CV_32F, 1, 0, ksize=3) / 8.0
    gy = cv2.Sobel(image, cv2.CV_32F, 0, 1, ksize=3) / 8.0
    gradient = np.stack((gx, gy), axis=-1)
    flow = np.asarray(prism_flow, dtype=np.float32)
    x = events.x
    y = events.y
    valid = (x >= 0) & (y >= 0) & (x < image.shape[1]) & (y < image.shape[0])
    if valid_mask is not None:
        valid &= valid_mask[y.clip(0, image.shape[0] - 1), x.clip(0, image.shape[1] - 1)]
    if not np.any(valid):
        return {"event_count": 0, "mean_abs_gradient_dot_prism_motion": 0.0, "low_observability_fraction": 0.0, "gradient_motion_angle_histogram": [0] * 12}
    g = gradient[y[valid], x[valid]]
    u = flow[y[valid], x[valid]]
    dot = np.sum(g * u, axis=-1)
    g_norm = np.linalg.norm(g, axis=-1)
    u_norm = np.linalg.norm(u, axis=-1)
    angle = np.arccos(np.clip(dot / np.maximum(g_norm * u_norm, 1e-8), -1.0, 1.0))
    histogram, _ = np.histogram(angle[np.isfinite(angle)], bins=12, range=(0.0, np.pi))
    return {
        "event_count": int(np.sum(valid)),
        "mean_abs_gradient_dot_prism_motion": float(np.mean(np.abs(dot))),
        "fraction_gradient_motion_below_threshold": float(np.mean(np.abs(dot) < threshold)),
        "mean_prism_motion_px": float(np.mean(u_norm)),
        "gradient_motion_angle_histogram": histogram.astype(int).tolist(),
    }
