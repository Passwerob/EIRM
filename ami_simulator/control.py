"""Causal, redundancy-aware optical modulation control."""

from __future__ import annotations

import numpy as np


class CausalGearSelector:
    """Apply dwell time and hysteresis to a causal per-window cost table."""

    def __init__(self, rpm_values: np.ndarray, *, initial_rpm: float, initial_time: float, minimum_dwell_seconds: float = 0.05, switch_margin: float = 0.02) -> None:
        values = np.asarray(rpm_values, dtype=np.float64).reshape(-1)
        if not len(values) or not np.all(np.isfinite(values)) or np.any(values < 0.0):
            raise ValueError("rpm_values must be a non-empty finite non-negative array.")
        if not np.isfinite(initial_rpm) or not np.isfinite(initial_time) or minimum_dwell_seconds < 0.0 or switch_margin < 0.0:
            raise ValueError("Initial state, dwell time, and switch margin must be finite and valid.")
        self.rpm_values = values
        self.current_gear = int(np.argmin(np.abs(values - initial_rpm)))
        self.minimum_dwell_seconds = float(minimum_dwell_seconds)
        self.switch_margin = float(switch_margin)
        self.last_switch_time = float(initial_time) - self.minimum_dwell_seconds
        self.last_timestamp = float(initial_time)

    @property
    def rpm(self) -> float:
        return float(self.rpm_values[self.current_gear])

    def select(self, costs: np.ndarray, *, timestamp: float) -> tuple[int, int]:
        """Return actuator-feasible and unconstrained gears for this window."""
        values = np.asarray(costs, dtype=np.float64).reshape(-1)
        if values.shape != self.rpm_values.shape or not np.all(np.isfinite(values)):
            raise ValueError("costs must be finite and match rpm_values.")
        if not np.isfinite(timestamp) or timestamp < self.last_timestamp:
            raise ValueError("Gear timestamps must be finite and non-decreasing.")
        best = int(np.argmin(values))
        dwell_ready = float(timestamp) - self.last_switch_time >= self.minimum_dwell_seconds
        improvement = float(values[self.current_gear] - values[best])
        if best != self.current_gear and dwell_ready and improvement >= self.switch_margin:
            self.current_gear = best
            self.last_switch_time = float(timestamp)
        self.last_timestamp = float(timestamp)
        return self.current_gear, best


class OpticalModulationController:
    """Select prism RPM from natural-motion prior and event-quality feedback.

    ``natural_motion_ratio`` is a bounded prior in [0, 1], where one means
    strong natural image motion. It never overrides event feedback: if natural
    motion is high but edge coverage is poor (for example camera and object
    move together), the controller increases prism speed.
    """

    name = "causal_quality_feedback"

    def __init__(
        self,
        *,
        min_rpm: float = 180.0,
        max_rpm: float = 1440.0,
        initial_rpm: float = 720.0,
        step_rpm: float = 120.0,
        max_accel_rpm_s: float = 2400.0,
        target_edge_coverage: float = 0.65,
        max_temporal_overlap: float = 0.65,
        min_structural_ratio: float = 0.25,
        prior_weight: float = 0.25,
    ) -> None:
        values = (min_rpm, max_rpm, initial_rpm, step_rpm, max_accel_rpm_s)
        if not np.all(np.isfinite(values)) or min_rpm < 0.0 or max_rpm < min_rpm or not min_rpm <= initial_rpm <= max_rpm or step_rpm <= 0.0 or max_accel_rpm_s <= 0.0:
            raise ValueError("RPM bounds, initial speed, step, and acceleration must be finite and ordered.")
        if not 0.0 <= target_edge_coverage <= 1.0 or not 0.0 <= max_temporal_overlap <= 1.0 or not 0.0 <= min_structural_ratio <= 1.0:
            raise ValueError("Quality targets must lie in [0, 1].")
        if not 0.0 <= prior_weight <= 1.0:
            raise ValueError("prior_weight must lie in [0, 1].")
        self.min_rpm = float(min_rpm)
        self.max_rpm = float(max_rpm)
        self.rpm = float(initial_rpm)
        self.step_rpm = float(step_rpm)
        self.max_accel_rpm_s = float(max_accel_rpm_s)
        self.target_edge_coverage = float(target_edge_coverage)
        self.max_temporal_overlap = float(max_temporal_overlap)
        self.min_structural_ratio = float(min_structural_ratio)
        self.prior_weight = float(prior_weight)

    def reset(self, rpm: float | None = None) -> float:
        value = self.rpm if rpm is None else float(rpm)
        if not np.isfinite(value) or not self.min_rpm <= value <= self.max_rpm:
            raise ValueError("reset RPM must lie within the configured bounds.")
        self.rpm = value
        return self.rpm

    def update(self, metrics: dict[str, float | int], *, dt_seconds: float, natural_motion_ratio: float | None = None) -> float:
        """Update RPM causally from the previous window's quality metrics."""
        if not np.isfinite(dt_seconds) or dt_seconds <= 0.0:
            raise ValueError("dt_seconds must be positive and finite.")
        required = ("edge_coverage", "temporal_overlap", "structural_ratio")
        if any(key not in metrics for key in required):
            raise ValueError(f"metrics must contain {required}.")
        if natural_motion_ratio is not None and (not np.isfinite(natural_motion_ratio) or not 0.0 <= natural_motion_ratio <= 1.0):
            raise ValueError("natural_motion_ratio must lie in [0, 1].")

        target = self.rpm
        coverage = float(metrics["edge_coverage"])
        overlap = float(metrics["temporal_overlap"])
        structure = float(metrics["structural_ratio"])
        if coverage < self.target_edge_coverage or structure < self.min_structural_ratio:
            target += self.step_rpm
        elif overlap > self.max_temporal_overlap:
            target -= self.step_rpm
        elif natural_motion_ratio is not None and natural_motion_ratio > 0.5:
            # Natural motion is only a gentle feed-forward reduction. The
            # quality conditions above always win, including co-motion cases.
            prior = self.max_rpm - natural_motion_ratio * (self.max_rpm - self.min_rpm)
            target = (1.0 - self.prior_weight) * target + self.prior_weight * prior

        target = float(np.clip(target, self.min_rpm, self.max_rpm))
        max_delta = self.max_accel_rpm_s * float(dt_seconds)
        self.rpm += float(np.clip(target - self.rpm, -max_delta, max_delta))
        self.rpm = float(np.clip(self.rpm, self.min_rpm, self.max_rpm))
        return self.rpm

    def metadata(self) -> dict[str, object]:
        return {
            "policy": self.name,
            "rpm_bounds": [self.min_rpm, self.max_rpm],
            "current_rpm": self.rpm,
            "step_rpm": self.step_rpm,
            "target_edge_coverage": self.target_edge_coverage,
            "max_temporal_overlap": self.max_temporal_overlap,
            "min_structural_ratio": self.min_structural_ratio,
            "prior_weight": self.prior_weight,
        }


class NeuralOpticalModulationController:
    """Causal inference wrapper for :class:`CausalRPMPolicy`.

    The wrapper keeps actuator constraints outside the network. This prevents
    a learned policy from emitting an instantaneous, physically implausible RPM
    jump even when it sees an out-of-distribution feature window.
    """

    name = "neural_causal_quality_feedback"

    def __init__(self, policy: object, *, min_rpm: float = 0.0, max_rpm: float = 1440.0, initial_rpm: float = 720.0, max_accel_rpm_s: float = 2400.0) -> None:
        if not min_rpm <= initial_rpm <= max_rpm or min_rpm < 0.0 or max_rpm <= min_rpm or max_accel_rpm_s <= 0.0:
            raise ValueError("RPM bounds and acceleration must be valid.")
        self.policy = policy
        self.min_rpm = float(min_rpm)
        self.max_rpm = float(max_rpm)
        self.rpm = float(initial_rpm)
        self.max_accel_rpm_s = float(max_accel_rpm_s)

    def update(self, features: "object", *, dt_seconds: float) -> float:
        """Predict and rate-limit one RPM command from a causal feature tensor."""
        import torch

        if not np.isfinite(dt_seconds) or dt_seconds <= 0.0:
            raise ValueError("dt_seconds must be positive and finite.")
        values = features if isinstance(features, torch.Tensor) else torch.as_tensor(features, dtype=torch.float32)
        if values.ndim == 2:
            values = values.unsqueeze(0)
        if values.ndim != 3:
            raise ValueError("features must have shape [time, feature_dim] or [batch, time, feature_dim].")
        with torch.inference_mode():
            prediction = self.policy(values).reshape(-1)
        if len(prediction) != 1 or not bool(torch.isfinite(prediction).item()):
            raise ValueError("policy must return one finite RPM prediction.")
        target = float(np.clip(float(prediction.item()), self.min_rpm, self.max_rpm))
        max_delta = self.max_accel_rpm_s * float(dt_seconds)
        self.rpm = float(np.clip(self.rpm + np.clip(target - self.rpm, -max_delta, max_delta), self.min_rpm, self.max_rpm))
        return self.rpm

    def metadata(self) -> dict[str, object]:
        return {"policy": self.name, "rpm_bounds": [self.min_rpm, self.max_rpm], "current_rpm": self.rpm, "max_accel_rpm_s": self.max_accel_rpm_s}
