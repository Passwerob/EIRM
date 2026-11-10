"""Named, reproducible features for the adaptive trajectory controller."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import cv2
import numpy as np

from ami_simulator.types import ImuStream

from ..control.trajectory import TrajectoryParameters
from .optical_flow import OpticalFlowResult


FLOW_FEATURE_NAMES = (
    "flow_mean_px_s",
    "flow_median_px_s",
    "flow_p90_px_s",
    "flow_max_px_s",
    "flow_variance_px2_s2",
    "flow_iqr_px_s",
    "flow_magnitude_entropy",
    "flow_direction_entropy",
    "flow_spatial_consistency",
    "flow_low_motion_fraction",
    "flow_valid_fraction",
    "flow_edge_weighted_px_s",
)

IMU_FEATURE_NAMES = (
    "imu_available",
    "imu_gyro_mean_rad_s",
    "imu_gyro_p90_rad_s",
    "imu_gyro_max_rad_s",
    "imu_rotational_energy_rad2_s",
    "imu_accel_mean_m_s2",
    "imu_accel_std_m_s2",
    "imu_dominant_axis_x",
    "imu_dominant_axis_y",
    "imu_dominant_axis_z",
)

EVENT_FEATURE_NAMES = (
    "event_density",
    "active_pixel_ratio",
    "edge_coverage",
    "edge_precision",
    "edge_f1",
    "structural_ratio",
    "event_efficiency",
    "repeat_event_fraction",
    "temporal_overlap",
    "polarity_imbalance",
    "reference_valid_fraction",
    "invalid_border_fraction",
    "temporal_diversity",
)

TRAJECTORY_FEATURE_NAMES = (
    "previous_amplitude_rad",
    "previous_frequency_hz",
    "previous_direction",
    "previous_duration_s",
    "previous_angular_speed_rad_s",
    "previous_waveform_constant",
    "previous_waveform_sinusoid",
    "previous_waveform_triangle",
    "previous_waveform_chirp",
)


@dataclass(frozen=True)
class FeatureNormalization:
    """Per-feature bounds fitted on training scenes and frozen afterwards."""

    bounds: Mapping[str, tuple[float, float]]

    def normalize(self, features: Mapping[str, float]) -> dict[str, float]:
        output: dict[str, float] = {}
        for name, value in features.items():
            if name not in self.bounds:
                output[name] = float(value)
                continue
            low, high = self.bounds[name]
            if not np.isfinite([low, high]).all() or high <= low:
                raise ValueError(f"Invalid normalization bounds for {name!r}.")
            output[name] = float(np.clip((float(value) - low) / (high - low), 0.0, 1.0))
        return output


def _normalized_entropy(values: np.ndarray, bins: int, value_range: tuple[float, float]) -> float:
    if len(values) == 0:
        return 0.0
    counts, _ = np.histogram(values, bins=bins, range=value_range)
    probabilities = counts[counts > 0].astype(np.float64)
    probabilities /= probabilities.sum()
    entropy = -float(np.sum(probabilities * np.log(probabilities)))
    return entropy / np.log(bins) if bins > 1 else 0.0


def _gray_float(image: np.ndarray) -> np.ndarray:
    values = np.asarray(image)
    if values.ndim == 3 and values.shape[-1] in (3, 4):
        values = cv2.cvtColor(values[..., :3].astype(np.float32), cv2.COLOR_RGB2GRAY)
    if values.ndim != 2:
        raise ValueError("reference_intensity must have shape [H,W] or [H,W,3/4].")
    return values.astype(np.float64)


def extract_flow_features(
    result: OpticalFlowResult,
    *,
    low_motion_threshold_px_s: float,
    magnitude_histogram_max_px_s: float,
    reference_intensity: np.ndarray | None = None,
    normalization: FeatureNormalization | None = None,
) -> dict[str, float]:
    """Summarize dense flow displacement as rate features in pixels/second."""
    if low_motion_threshold_px_s < 0.0 or magnitude_histogram_max_px_s <= 0.0:
        raise ValueError("Flow thresholds must be non-negative with a positive histogram maximum.")
    flow = np.asarray(result.flow_xy_px, dtype=np.float64) / result.dt_seconds
    valid = np.asarray(result.valid_mask, dtype=bool) & np.isfinite(flow).all(axis=-1)
    vectors = flow[valid]
    magnitude = np.linalg.norm(vectors, axis=-1)
    if len(magnitude):
        mean_vector = np.mean(vectors, axis=0)
        mean_norm = float(np.linalg.norm(mean_vector))
        moving = magnitude > max(low_motion_threshold_px_s, 1e-12)
        if mean_norm > 1e-12 and np.any(moving):
            cosine = (vectors[moving] @ mean_vector) / (magnitude[moving] * mean_norm)
            consistency = float(np.mean(cosine >= np.cos(np.deg2rad(30.0))))
        else:
            consistency = 1.0 if not np.any(moving) else 0.0
        directions = np.mod(np.arctan2(vectors[moving, 1], vectors[moving, 0]), 2.0 * np.pi)
        direction_entropy = _normalized_entropy(directions, 12, (0.0, 2.0 * np.pi))
    else:
        consistency = direction_entropy = 0.0

    edge_weighted = 0.0
    if reference_intensity is not None:
        gray = _gray_float(reference_intensity)
        if gray.shape != valid.shape:
            raise ValueError("reference_intensity shape must match the flow field.")
        gy, gx = np.gradient(gray)
        weights = np.hypot(gx, gy)[valid]
        if len(weights) and float(weights.sum()) > 1e-12:
            edge_weighted = float(np.sum(weights * magnitude) / np.sum(weights))

    q25, q75 = np.quantile(magnitude, [0.25, 0.75]) if len(magnitude) else (0.0, 0.0)
    features = {
        "flow_mean_px_s": float(np.mean(magnitude)) if len(magnitude) else 0.0,
        "flow_median_px_s": float(np.median(magnitude)) if len(magnitude) else 0.0,
        "flow_p90_px_s": float(np.quantile(magnitude, 0.9)) if len(magnitude) else 0.0,
        "flow_max_px_s": float(np.max(magnitude)) if len(magnitude) else 0.0,
        "flow_variance_px2_s2": float(np.var(magnitude)) if len(magnitude) else 0.0,
        "flow_iqr_px_s": float(q75 - q25),
        "flow_magnitude_entropy": _normalized_entropy(np.clip(magnitude, 0.0, magnitude_histogram_max_px_s), 12, (0.0, magnitude_histogram_max_px_s)),
        "flow_direction_entropy": direction_entropy,
        "flow_spatial_consistency": consistency,
        "flow_low_motion_fraction": float(np.mean(magnitude <= low_motion_threshold_px_s)) if len(magnitude) else 1.0,
        "flow_valid_fraction": float(np.mean(valid)),
        "flow_edge_weighted_px_s": edge_weighted,
    }
    return features if normalization is None else normalization.normalize(features)


def extract_imu_features(imu_window: ImuStream | None) -> dict[str, float]:
    """Return an optional motion prior; no field is a prism command."""
    if imu_window is None:
        return {name: 0.0 for name in IMU_FEATURE_NAMES}
    gyro_norm = np.linalg.norm(imu_window.gyro, axis=1)
    accel_norm = np.linalg.norm(imu_window.accel, axis=1)
    duration = float(imu_window.t[-1] - imu_window.t[0])
    energy = float(np.trapz(gyro_norm**2, imu_window.t) / duration)
    axis = int(np.argmax(np.mean(np.abs(imu_window.gyro), axis=0)))
    return {
        "imu_available": 1.0,
        "imu_gyro_mean_rad_s": float(np.mean(gyro_norm)),
        "imu_gyro_p90_rad_s": float(np.quantile(gyro_norm, 0.9)),
        "imu_gyro_max_rad_s": float(np.max(gyro_norm)),
        "imu_rotational_energy_rad2_s": energy,
        "imu_accel_mean_m_s2": float(np.mean(accel_norm)),
        "imu_accel_std_m_s2": float(np.std(accel_norm)),
        "imu_dominant_axis_x": float(axis == 0),
        "imu_dominant_axis_y": float(axis == 1),
        "imu_dominant_axis_z": float(axis == 2),
    }


def extract_event_features(metrics: Mapping[str, float | int]) -> dict[str, float]:
    """Map root event-quality output to a stable controller feature schema."""
    aliases = {
        "active_pixel_ratio": "unique_pixel_ratio",
        "polarity_imbalance": "polarity_imbalance",
        "reference_valid_fraction": "reference_valid_fraction",
        "invalid_border_fraction": "invalid_border_fraction",
        "temporal_diversity": "temporal_diversity",
    }
    output: dict[str, float] = {}
    for name in EVENT_FEATURE_NAMES:
        source = aliases.get(name, name)
        if source in metrics:
            value = float(metrics[source])
        elif name == "invalid_border_fraction" and "reference_valid_fraction" in metrics:
            value = 1.0 - float(metrics["reference_valid_fraction"])
        elif name == "temporal_diversity":
            value = 1.0 - float(metrics.get("temporal_overlap", 0.0))
        else:
            value = 0.0
        if not np.isfinite(value):
            raise ValueError(f"Event metric {source!r} must be finite.")
        output[name] = value
    return output


def _trajectory_features(parameters: TrajectoryParameters) -> dict[str, float]:
    return {
        "previous_amplitude_rad": parameters.amplitude_rad,
        "previous_frequency_hz": parameters.frequency_hz,
        "previous_direction": float(parameters.direction),
        "previous_duration_s": parameters.duration_s,
        "previous_angular_speed_rad_s": float(parameters.angular_speed_rad_s or 0.0),
        **{f"previous_waveform_{name}": float(parameters.waveform == name) for name in ("constant", "sinusoid", "triangle", "chirp")},
    }


def build_state_features(
    flow_features: Mapping[str, float],
    event_features: Mapping[str, float],
    previous_trajectory: TrajectoryParameters,
    *,
    imu_features: Mapping[str, float] | None = None,
    use_flow: bool = True,
) -> dict[str, float]:
    """Build a fixed-schema causal state; ablations mask values, not dimensions."""
    missing_flow = set(FLOW_FEATURE_NAMES) - set(flow_features)
    missing_event = set(EVENT_FEATURE_NAMES) - set(event_features)
    if missing_flow or missing_event:
        raise ValueError(f"Missing state features: flow={sorted(missing_flow)}, event={sorted(missing_event)}")
    imu = {name: 0.0 for name in IMU_FEATURE_NAMES}
    if imu_features is not None:
        imu.update({name: float(imu_features[name]) for name in IMU_FEATURE_NAMES})
    state = {name: float(flow_features[name]) if use_flow else 0.0 for name in FLOW_FEATURE_NAMES}
    if not use_flow:
        # Missing flow is not evidence of high motion. Use a conservative
        # low-motion prior so the ablation can still test event-only control.
        state["flow_low_motion_fraction"] = 1.0
    state.update({name: float(event_features[name]) for name in EVENT_FEATURE_NAMES})
    state.update(imu)
    state.update(_trajectory_features(previous_trajectory))
    if not np.all(np.isfinite(list(state.values()))):
        raise ValueError("Controller state must contain only finite values.")
    return state


def state_vector(state: Mapping[str, float]) -> tuple[np.ndarray, tuple[str, ...]]:
    """Serialize state deterministically with names alongside values."""
    names = FLOW_FEATURE_NAMES + EVENT_FEATURE_NAMES + IMU_FEATURE_NAMES + TRAJECTORY_FEATURE_NAMES
    missing = set(names) - set(state)
    if missing:
        raise ValueError(f"State is missing features: {sorted(missing)}")
    return np.asarray([state[name] for name in names], dtype=np.float32), names
