"""Causal streaming rollout for the analytical trajectory controller."""

from __future__ import annotations

from dataclasses import asdict, replace
from typing import Sequence

import numpy as np
import torch

from ami_simulator.event_quality import active_pixel_ids, event_quality_metrics
from ami_simulator.geometry import RotatingWedgePrism
from ami_simulator.pipeline import rgb_to_linear_luminance
from ami_simulator.types import CameraIntrinsics, EventStream, ImuStream
from ami_simulator.warp import PrismWarper, compensate_events

from ..control.adaptive_controller import RuleBasedAdaptiveController
from ..control.trajectory import ActuatorConstrainedTrajectory, PiecewiseTrajectory, TrajectoryLimits, TrajectoryParameters
from ..motion.motion_features import build_state_features, extract_event_features, extract_flow_features, extract_imu_features
from ..motion.optical_flow import FarnebackFlowBackend, OpticalFlowResult
from .perception_metrics import FeatureMatchingConfig, event_time_surface, feature_matching_metrics
from .trajectory_sweep import SweepConfig, temporal_diversity


class StreamingThresholdState:
    """Stateful equivalent of the root CPU threshold backend.

    The log reference survives calls, so event thresholds are not reset at a
    control-window boundary. This class intentionally keeps the root backend's
    deterministic timing and polarity convention.
    """

    def __init__(self, contrast_threshold: float = 0.2, epsilon: float = 1e-6) -> None:
        if not np.isfinite(contrast_threshold) or contrast_threshold <= 0.0 or epsilon <= 0.0:
            raise ValueError("contrast_threshold and epsilon must be positive and finite.")
        self.contrast_threshold = float(contrast_threshold)
        self.epsilon = float(epsilon)
        self.reference: np.ndarray | None = None

    def reset(self) -> None:
        self.reference = None

    def simulate_interval(self, intensity: np.ndarray, timestamps: np.ndarray) -> EventStream:
        frames = np.asarray(intensity, dtype=np.float64)
        times = np.asarray(timestamps, dtype=np.float64).reshape(-1)
        if frames.ndim != 3 or len(frames) != len(times) or len(times) < 2 or not np.all(np.diff(times) > 0.0):
            raise ValueError("Streaming intensity must have shape [T,H,W] with increasing timestamps and T >= 2.")
        log_frames = np.log(np.clip(frames, self.epsilon, 1.0))
        if self.reference is None:
            self.reference = log_frames[0].copy()
        if self.reference.shape != log_frames.shape[1:]:
            raise ValueError("Streaming intensity resolution changed between windows.")
        xs: list[np.ndarray] = []
        ys: list[np.ndarray] = []
        event_times: list[np.ndarray] = []
        polarities: list[np.ndarray] = []
        for index in range(1, len(times)):
            before, after = log_frames[index - 1], log_frames[index]
            delta = after - self.reference
            positive = np.floor(np.maximum(delta, 0.0) / self.contrast_threshold + self.epsilon).astype(np.int32)
            negative = np.floor(np.maximum(-delta, 0.0) / self.contrast_threshold + self.epsilon).astype(np.int32)
            for sign, counts in ((1, positive), (-1, negative)):
                v, u = np.nonzero(counts > 0)
                if not len(u):
                    continue
                repeated = np.repeat(np.arange(len(u)), counts[v, u])
                event_u, event_v = u[repeated], v[repeated]
                levels = np.concatenate([
                    self.reference[row, col] + sign * self.contrast_threshold * np.arange(1, counts[row, col] + 1)
                    for row, col in zip(v, u)
                ])
                slope = after[event_v, event_u] - before[event_v, event_u]
                alpha = np.divide(levels - before[event_v, event_u], slope, out=np.full_like(levels, 0.5), where=np.abs(slope) > self.epsilon)
                xs.append(event_u.astype(np.int32))
                ys.append(event_v.astype(np.int32))
                event_times.append(times[index - 1] + np.clip(alpha, 0.0, 1.0) * (times[index] - times[index - 1]))
                polarities.append(np.full(len(event_u), sign, dtype=np.int8))
            self.reference += (positive - negative) * self.contrast_threshold
        if not xs:
            return EventStream.empty()
        return EventStream(np.concatenate(xs), np.concatenate(ys), np.concatenate(event_times), np.concatenate(polarities)).sorted_by_time()


def _window_trajectory(selected: Sequence[TrajectoryParameters], edges: np.ndarray) -> PiecewiseTrajectory:
    segments = [
        (float(edges[index]), replace(parameters, duration_s=float(edges[index + 1] - edges[index])))
        for index, parameters in enumerate(selected)
    ]
    return PiecewiseTrajectory(segments)


def _aggregate(rows: list[dict[str, object]]) -> dict[str, float | int]:
    if not rows:
        raise ValueError("At least one rollout window is required.")
    keys = ("event_density", "unique_pixel_ratio", "edge_coverage", "edge_precision", "edge_f1", "event_efficiency", "temporal_overlap", "temporal_diversity", "reference_valid_fraction")
    aggregate: dict[str, float | int] = {f"mean_{key}": float(np.mean([float(row[key]) for row in rows])) for key in keys}
    matching_rows = rows[1:]
    for key in ("repeatability", "matching_precision", "matching_recall", "ransac_inlier_ratio"):
        aggregate[f"mean_{key}"] = float(np.mean([float(row[key]) for row in matching_rows])) if matching_rows else 0.0
    aggregate["matching_failure_rate"] = float(np.mean([int(row["matching_failed"]) for row in matching_rows])) if matching_rows else 1.0
    aggregate["event_count"] = int(sum(int(row["event_count"]) for row in rows))
    return aggregate


def _flow_result(flow_backend: FarnebackFlowBackend, previous: np.ndarray, current: np.ndarray, dt_seconds: float) -> OpticalFlowResult:
    return flow_backend.estimate(previous, current, dt_seconds=dt_seconds)


def rollout_adaptive_controller(
    rgb_frames: np.ndarray,
    timestamps: np.ndarray,
    intrinsics: CameraIntrinsics,
    window_edges: Sequence[float],
    candidates: Sequence[TrajectoryParameters],
    limits: TrajectoryLimits,
    *,
    config: SweepConfig | None = None,
    matching_config: FeatureMatchingConfig | None = None,
    homographies_reference_to_candidate: np.ndarray | None = None,
    initial_index: int = 0,
    objective: str = "useful_event_quality",
    use_flow: bool = True,
    imu_stream: ImuStream | None = None,
    useful_speed_cap_rad_s: float | None = None,
) -> dict[str, object]:
    """Run a causal controller with persistent event and actuator state."""
    cfg = config or SweepConfig()
    frames = np.asarray(rgb_frames)
    times = np.asarray(timestamps, dtype=np.float64).reshape(-1)
    edges = np.asarray(window_edges, dtype=np.float64).reshape(-1)
    if frames.ndim != 4 or frames.shape[-1] < 3 or len(frames) != len(times) or len(times) < 2 or not np.all(np.diff(times) > 0.0):
        raise ValueError("rgb_frames/timestamps must be synchronized [T,H,W,3+] with increasing timestamps.")
    if len(edges) < 2 or not np.all(np.diff(edges) > 0.0):
        raise ValueError("window_edges must be increasing with at least two boundaries.")
    if imu_stream is not None and (imu_stream.t[0] > edges[0] or imu_stream.t[-1] < edges[-1]):
        raise ValueError("imu_stream must cover the complete adaptive rollout interval.")
    tolerance = 1e-12 * max(1.0, abs(float(times[0])), abs(float(times[-1])))
    if edges[0] < times[0] - tolerance or edges[-1] > times[-1] + tolerance:
        raise ValueError("window_edges must lie inside the frame timeline.")
    height, width = frames.shape[1:3]
    if (intrinsics.width, intrinsics.height) != (width, height):
        raise ValueError("intrinsics must match RGB frame resolution.")
    pair_count = len(edges) - 2
    if homographies_reference_to_candidate is None:
        homographies = np.repeat(np.eye(3, dtype=np.float64)[None], pair_count, axis=0)
    else:
        homographies = np.asarray(homographies_reference_to_candidate, dtype=np.float64)
        if homographies.shape != (pair_count, 3, 3) or not np.all(np.isfinite(homographies)):
            raise ValueError(f"homographies_reference_to_candidate must have shape [{pair_count},3,3].")

    luminance = rgb_to_linear_luminance(frames)
    prism = RotatingWedgePrism(intrinsics, cfg.prism_angle_deg)
    flow_backend = FarnebackFlowBackend()
    controller = RuleBasedAdaptiveController(
        candidates,
        limits,
        initial_index=initial_index,
        objective=objective,  # type: ignore[arg-type]
        minimum_dwell_s=float(edges[1] - edges[0]),
        switch_margin=0.01,
        useful_speed_cap_rad_s=useful_speed_cap_rad_s,
    )
    controller.reset(timestamp=float(edges[0]))
    selected: list[TrajectoryParameters] = [controller.candidates[controller.current_index]]
    decisions: list[dict[str, object]] = [{"timestamp": float(edges[0]), "selected_index": controller.current_index, "reason": "initial_action_before_observation"}]
    states: list[dict[str, float]] = []
    rows: list[dict[str, object]] = []
    event_state = StreamingThresholdState(cfg.contrast_threshold)
    previous_active: np.ndarray | None = None
    previous_surface: np.ndarray | None = None

    for window_index, (start, end) in enumerate(zip(edges[:-1], edges[1:])):
        start_index = int(np.argmin(np.abs(times - start)))
        end_index = int(np.argmin(np.abs(times - end)))
        if not np.isclose(times[start_index], start, rtol=0.0, atol=1e-12) or not np.isclose(times[end_index], end, rtol=0.0, atol=1e-12):
            raise ValueError("Every adaptive window boundary must align with an RGB timestamp.")
        partial_times = times[: end_index + 1]
        desired = _window_trajectory(selected, edges[: len(selected) + 1])
        constrained = ActuatorConstrainedTrajectory(desired, partial_times, limits)
        warper = PrismWarper(prism, constrained, height=height, width=width, reference_time=float(times[0]), padding_mode=cfg.padding_mode)
        window_times = times[start_index : end_index + 1]
        with torch.inference_mode():
            sensor, valid = warper.warp(torch.from_numpy(luminance[start_index : end_index + 1]).unsqueeze(1), window_times)
        raw = event_state.simulate_interval(sensor[:, 0].cpu().numpy(), window_times)
        compensated = compensate_events(raw, prism, constrained, theta0=0.0, reference_time=float(times[0]), device=cfg.device)
        reference_mask = (compensated.t > start) & (compensated.t <= end)
        x = np.rint(compensated.extras["x_reference"][reference_mask]).astype(np.int32)
        y = np.rint(compensated.extras["y_reference"][reference_mask]).astype(np.int32)
        valid_reference = (x >= 0) & (x < width) & (y >= 0) & (y < height)
        reference_events = EventStream(x[valid_reference], y[valid_reference], compensated.t[reference_mask][valid_reference], compensated.p[reference_mask][valid_reference])
        metrics = event_quality_metrics(reference_events, reference_intensity=luminance[start_index], previous_active_pixels=previous_active, edge_quantile=cfg.edge_quantile, image_shape=(height, width))
        metrics["sensor_event_count"] = len(raw)
        metrics["reference_valid_fraction"] = float(len(reference_events) / len(raw)) if len(raw) else 1.0
        metrics["invalid_border_fraction"] = 1.0 - metrics["reference_valid_fraction"]
        metrics["temporal_diversity"] = temporal_diversity(reference_events, time_range=(float(start), float(end)), bins=cfg.temporal_bins)
        surface = event_time_surface(reference_events, image_shape=(height, width), time_range=(float(start), float(end)), decay_seconds=cfg.time_surface_decay_seconds)
        if previous_surface is None:
            matching = {"repeatability": 0.0, "matching_precision": 0.0, "matching_recall": 0.0, "ransac_inlier_ratio": 0.0, "median_localization_error_px": None, "matching_failed": 1, "accepted_matches": 0}
        else:
            matching = feature_matching_metrics(previous_surface, surface, homographies[window_index - 1], config=matching_config)
        rows.append({"window_start_s": float(start), "window_end_s": float(end), "selected_index": controller.current_index, "selected_parameters": selected[-1].metadata(), "valid_image_fraction": float(valid.all(dim=0).float().mean().item()), **metrics, **{key: matching[key] for key in ("repeatability", "matching_precision", "matching_recall", "ransac_inlier_ratio", "median_localization_error_px", "matching_failed", "accepted_matches")}})
        previous_active = active_pixel_ids(reference_events, image_shape=(height, width))
        previous_surface = surface

        if window_index + 1 < len(edges) - 1:
            event_features = extract_event_features(metrics)
            flow = _flow_result(flow_backend, frames[start_index], frames[end_index], float(end - start))
            flow_features = extract_flow_features(flow, low_motion_threshold_px_s=1.0, magnitude_histogram_max_px_s=100.0, reference_intensity=frames[start_index])
            imu_features = None
            if imu_stream is not None:
                imu_mask = (imu_stream.t >= float(start)) & (imu_stream.t <= float(end))
                if np.count_nonzero(imu_mask) < 2:
                    raise ValueError("Each adaptive window must contain at least two IMU samples.")
                imu_features = extract_imu_features(ImuStream(imu_stream.t[imu_mask], imu_stream.accel[imu_mask], imu_stream.gyro[imu_mask]))
            state = build_state_features(flow_features, event_features, selected[-1], imu_features=imu_features, use_flow=use_flow)
            states.append(state)
            next_parameters = controller.select(state, timestamp=float(end))
            selected.append(next_parameters)
            assert controller.last_decision is not None
            decisions.append({**asdict(controller.last_decision), "parameters": next_parameters.metadata()})

    return {
        "policy": "adaptive_trajectory",
        "controller": controller.metadata(),
        "trajectory_segments": [parameters.metadata() for parameters in selected],
        "decisions": decisions,
        "state_count": len(states),
        "states": states,
        "sweep_config": asdict(cfg),
        "matching_config": (matching_config or FeatureMatchingConfig()).metadata(),
        "resolution": [width, height],
        "frame_count": len(frames),
        "window_count": len(rows),
        "aggregate": _aggregate(rows),
        "windows": rows,
    }
