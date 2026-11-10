"""Trajectory evaluation using the existing prism and event simulator."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Sequence

import numpy as np
import torch

from ami_simulator.event_quality import active_pixel_ids, event_quality_metrics
from ami_simulator.events import ThresholdEventSimulator
from ami_simulator.geometry import RotatingWedgePrism
from ami_simulator.pipeline import rgb_to_linear_luminance
from ami_simulator.types import CameraIntrinsics, EventStream
from ami_simulator.warp import PrismWarper, compensate_events

from ..control.trajectory import OpticalTrajectory
from .perception_metrics import FeatureMatchingConfig, event_time_surface, feature_matching_metrics


@dataclass(frozen=True)
class SweepConfig:
    prism_angle_deg: float = 0.5
    contrast_threshold: float = 0.2
    event_window_seconds: float = 0.1
    time_surface_decay_seconds: float = 0.03
    temporal_bins: int = 5
    edge_quantile: float = 0.75
    padding_mode: str = "border"
    device: str = "cpu"

    def __post_init__(self) -> None:
        positive = (self.prism_angle_deg, self.contrast_threshold, self.event_window_seconds, self.time_surface_decay_seconds)
        if not np.all(np.isfinite(positive)) or min(positive) <= 0.0 or self.temporal_bins < 1:
            raise ValueError("Sweep angles, thresholds, windows, decay, and temporal bins must be positive.")
        if not 0.0 < self.edge_quantile < 1.0:
            raise ValueError("edge_quantile must lie in (0,1).")


def _slice_events(events: EventStream, start: float, end: float) -> EventStream:
    mask = (events.t > start) & (events.t <= end)
    return EventStream(events.x[mask], events.y[mask], events.t[mask], events.p[mask])


def _reference_events(compensated: EventStream, start: float, end: float, image_shape: tuple[int, int]) -> EventStream:
    height, width = image_shape
    mask = (compensated.t > start) & (compensated.t <= end)
    x = np.rint(compensated.extras["x_reference"][mask]).astype(np.int32)
    y = np.rint(compensated.extras["y_reference"][mask]).astype(np.int32)
    valid = (x >= 0) & (x < width) & (y >= 0) & (y < height)
    return EventStream(x[valid], y[valid], compensated.t[mask][valid], compensated.p[mask][valid])


def temporal_diversity(events: EventStream, *, time_range: tuple[float, float], bins: int) -> float:
    start, end = time_range
    if bins < 1 or end <= start:
        raise ValueError("Temporal-diversity bins and time range must be valid.")
    valid = (events.t > start) & (events.t <= end)
    if not np.any(valid):
        return 0.0
    occupied = np.unique(np.minimum(((events.t[valid] - start) / (end - start) * bins).astype(np.int64), bins - 1))
    return float(len(occupied) / bins)


def evaluate_trajectory(
    name: str,
    rgb_frames: np.ndarray,
    timestamps: np.ndarray,
    intrinsics: CameraIntrinsics,
    trajectory: OpticalTrajectory,
    window_edges: Sequence[float],
    *,
    config: SweepConfig | None = None,
    matching_config: FeatureMatchingConfig | None = None,
    homography_reference_to_candidate: np.ndarray | None = None,
    homographies_reference_to_candidate: np.ndarray | None = None,
) -> dict[str, object]:
    """Evaluate one complete trajectory over fixed, synchronized windows."""
    cfg = config or SweepConfig()
    frames = np.asarray(rgb_frames)
    times = np.asarray(timestamps, dtype=np.float64).reshape(-1)
    edges = np.asarray(window_edges, dtype=np.float64).reshape(-1)
    if frames.ndim != 4 or frames.shape[-1] < 3 or len(frames) != len(times):
        raise ValueError("rgb_frames must have shape [T,H,W,3+] with one timestamp per frame.")
    if len(times) < 2 or not np.all(np.diff(times) > 0.0):
        raise ValueError("timestamps must be strictly increasing.")
    tolerance = 1e-12 * max(1.0, abs(float(times[0])), abs(float(times[-1])))
    if len(edges) < 2 or not np.all(np.diff(edges) > 0.0) or edges[0] < times[0] - tolerance or edges[-1] > times[-1] + tolerance:
        raise ValueError("window_edges must be increasing and lie inside the frame timeline.")
    height, width = frames.shape[1:3]
    if (intrinsics.width, intrinsics.height) != (width, height):
        raise ValueError("Intrinsics must match the exact frame resolution.")
    luminance = rgb_to_linear_luminance(frames)
    prism = RotatingWedgePrism(intrinsics, cfg.prism_angle_deg)
    warper = PrismWarper(prism, trajectory, height=height, width=width, reference_time=float(times[0]), padding_mode=cfg.padding_mode)
    device = torch.device(cfg.device)
    with torch.inference_mode():
        sensor, valid = warper.warp(torch.from_numpy(luminance).to(device).unsqueeze(1), times)
    simulator = ThresholdEventSimulator(cfg.contrast_threshold)
    raw = simulator.simulate(sensor[:, 0].cpu().numpy(), times)
    compensated = compensate_events(raw, prism, trajectory, theta0=0.0, reference_time=float(times[0]), device=cfg.device)

    if homography_reference_to_candidate is not None and homographies_reference_to_candidate is not None:
        raise ValueError("Provide either one shared homography or per-window homographies, not both.")
    matching_pair_count = len(edges) - 2
    if homographies_reference_to_candidate is not None:
        homographies = np.asarray(homographies_reference_to_candidate, dtype=np.float64)
        if homographies.shape != (matching_pair_count, 3, 3) or not np.all(np.isfinite(homographies)):
            raise ValueError(f"Per-window homographies must have shape [{matching_pair_count},3,3].")
    else:
        shared = np.eye(3, dtype=np.float64) if homography_reference_to_candidate is None else np.asarray(homography_reference_to_candidate, dtype=np.float64)
        if shared.shape != (3, 3) or not np.all(np.isfinite(shared)):
            raise ValueError("Shared homography must have shape [3,3] and finite values.")
        homographies = np.repeat(shared[None], matching_pair_count, axis=0)
    rows: list[dict[str, object]] = []
    previous_active: np.ndarray | None = None
    previous_surface: np.ndarray | None = None
    for window_index, (start, end) in enumerate(zip(edges[:-1], edges[1:])):
        reference = _reference_events(compensated, float(start), float(end), (height, width))
        sensor_window = _slice_events(raw, float(start), float(end))
        frame_index = int(np.searchsorted(times, start, side="right") - 1)
        metrics = event_quality_metrics(
            reference,
            reference_intensity=luminance[frame_index],
            previous_active_pixels=previous_active,
            edge_quantile=cfg.edge_quantile,
            image_shape=(height, width),
        )
        metrics["sensor_event_count"] = len(sensor_window)
        metrics["reference_valid_fraction"] = float(len(reference) / len(sensor_window)) if len(sensor_window) else 1.0
        metrics["invalid_border_fraction"] = 1.0 - metrics["reference_valid_fraction"]
        metrics["temporal_diversity"] = temporal_diversity(reference, time_range=(float(start), float(end)), bins=cfg.temporal_bins)
        surface = event_time_surface(
            reference,
            image_shape=(height, width),
            time_range=(float(start), float(end)),
            decay_seconds=cfg.time_surface_decay_seconds,
        )
        if previous_surface is None:
            matching = {
                "repeatability": 0.0,
                "matching_precision": 0.0,
                "matching_recall": 0.0,
                "ransac_inlier_ratio": 0.0,
                "median_localization_error_px": None,
                "matching_failed": 1,
                "accepted_matches": 0,
            }
        else:
            matching = feature_matching_metrics(previous_surface, surface, homographies[window_index - 1], config=matching_config)
        rows.append({
            "window_start_s": float(start),
            "window_end_s": float(end),
            **{key: value for key, value in metrics.items()},
            **{key: matching[key] for key in ("repeatability", "matching_precision", "matching_recall", "ransac_inlier_ratio", "median_localization_error_px", "matching_failed", "accepted_matches")},
        })
        previous_active = active_pixel_ids(reference, image_shape=(height, width))
        previous_surface = surface

    matching_rows = rows[1:]
    aggregate_keys = (
        "event_density",
        "unique_pixel_ratio",
        "edge_coverage",
        "edge_precision",
        "edge_f1",
        "event_efficiency",
        "temporal_overlap",
        "temporal_diversity",
        "reference_valid_fraction",
    )
    aggregate = {f"mean_{key}": float(np.mean([float(row[key]) for row in rows])) for key in aggregate_keys}
    for key in ("repeatability", "matching_precision", "matching_recall", "ransac_inlier_ratio"):
        aggregate[f"mean_{key}"] = float(np.mean([float(row[key]) for row in matching_rows])) if matching_rows else 0.0
    aggregate["matching_failure_rate"] = float(np.mean([int(row["matching_failed"]) for row in matching_rows])) if matching_rows else 1.0
    aggregate["event_count"] = int(sum(int(row["event_count"]) for row in rows))
    return {
        "policy": name,
        "trajectory": trajectory.metadata(),
        "sweep_config": asdict(cfg),
        "matching_config": (matching_config or FeatureMatchingConfig()).metadata(),
        "resolution": [width, height],
        "frame_count": len(frames),
        "window_count": len(rows),
        "valid_image_fraction": float(valid.all(dim=0).float().mean().item()),
        "raw_event_count": len(raw),
        "compensated_event_count": len(compensated),
        "aggregate": aggregate,
        "windows": rows,
    }
