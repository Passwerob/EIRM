"""Transparent rule-based trajectory controller for the Stage-1 experiment."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Mapping, Sequence

import numpy as np

from .trajectory import TrajectoryLimits, TrajectoryParameters, make_trajectory, sample_candidate_trajectories


Objective = Literal["useful_event_quality", "event_count"]


@dataclass(frozen=True)
class TrajectoryDecision:
    timestamp: float
    information_deficit: float
    unconstrained_index: int
    selected_index: int
    switched: bool
    objective: Objective
    candidate_costs: tuple[float, ...]


class RuleBasedAdaptiveController:
    """Select trajectory parameters from causal motion and event feedback."""

    name = "rule_based_adaptive_trajectory"

    def __init__(
        self,
        candidates: Sequence[TrajectoryParameters],
        limits: TrajectoryLimits,
        *,
        initial_index: int = 0,
        objective: Objective = "useful_event_quality",
        minimum_dwell_s: float = 0.05,
        switch_margin: float = 0.02,
        effort_weight: float = 0.08,
        switch_weight: float = 0.01,
        imu_prior_weight: float = 0.20,
        imu_motion_scale_rad_s: float = 0.005,
        useful_speed_cap_rad_s: float | None = None,
    ) -> None:
        self.candidates = tuple(sample_candidate_trajectories(candidates, limits))
        if not 0 <= initial_index < len(self.candidates):
            raise ValueError("initial_index falls outside the candidate set.")
        if objective not in ("useful_event_quality", "event_count"):
            raise ValueError("Unsupported controller objective.")
        values = (minimum_dwell_s, switch_margin, effort_weight, switch_weight, imu_prior_weight, imu_motion_scale_rad_s)
        if not np.all(np.isfinite(values)) or min(values) < 0.0:
            raise ValueError("Controller penalties and dwell values must be finite and non-negative.")
        if imu_motion_scale_rad_s <= 0.0:
            raise ValueError("imu_motion_scale_rad_s must be positive.")
        if useful_speed_cap_rad_s is not None and (not np.isfinite(useful_speed_cap_rad_s) or useful_speed_cap_rad_s <= 0.0):
            raise ValueError("useful_speed_cap_rad_s must be positive and finite when provided.")
        self.limits = limits
        self.objective = objective
        self.minimum_dwell_s = float(minimum_dwell_s)
        self.switch_margin = float(switch_margin)
        self.effort_weight = float(effort_weight)
        self.switch_weight = float(switch_weight)
        self.imu_prior_weight = float(imu_prior_weight)
        self.imu_motion_scale_rad_s = float(imu_motion_scale_rad_s)
        self.useful_speed_cap_rad_s = None if useful_speed_cap_rad_s is None else float(useful_speed_cap_rad_s)
        self.initial_index = int(initial_index)
        self.current_index = int(initial_index)
        self.last_timestamp: float | None = None
        self.last_switch_time: float | None = None
        self.last_decision: TrajectoryDecision | None = None

    def reset(self, *, timestamp: float, index: int | None = None) -> None:
        if not np.isfinite(timestamp):
            raise ValueError("timestamp must be finite.")
        selected = self.initial_index if index is None else int(index)
        if not 0 <= selected < len(self.candidates):
            raise ValueError("reset index falls outside the candidate set.")
        self.current_index = selected
        self.last_timestamp = float(timestamp)
        self.last_switch_time = float(timestamp)
        self.last_decision = None

    def _peak_speed(self, candidate: TrajectoryParameters) -> float:
        trajectory = make_trajectory(candidate)
        times = np.linspace(0.0, candidate.duration_s, 257, dtype=np.float64)
        return float(np.max(np.abs(trajectory.omega_values(times)), initial=0.0))

    def _information_deficit(self, state: Mapping[str, float]) -> float:
        if self.objective == "event_count":
            # Count maximization deliberately drives toward the most intense
            # feasible candidate, independent of structural usefulness.
            return 1.0
        required = ("flow_low_motion_fraction", "edge_coverage", "event_efficiency", "temporal_overlap", "invalid_border_fraction")
        if any(name not in state for name in required):
            raise ValueError(f"State must contain {required}.")
        low_motion = np.clip(float(state["flow_low_motion_fraction"]), 0.0, 1.0)
        # IMU is a weak natural-motion prior. It only reduces the deficit and
        # never maps directly to a prism angle or speed command.
        if float(state.get("imu_available", 0.0)) > 0.0:
            imu_activity = np.clip(float(state.get("imu_gyro_p90_rad_s", 0.0)) / self.imu_motion_scale_rad_s, 0.0, 1.0)
            low_motion *= 1.0 - self.imu_prior_weight * imu_activity
        missing_edges = 1.0 - np.clip(float(state["edge_coverage"]), 0.0, 1.0)
        inefficient = 1.0 - np.clip(float(state["event_efficiency"]), 0.0, 1.0)
        overlap = np.clip(float(state["temporal_overlap"]), 0.0, 1.0)
        invalid = np.clip(float(state["invalid_border_fraction"]), 0.0, 1.0)
        # Event quality is evidence of a deficit only where natural motion is
        # itself weak. This prevents ordinary moving scenes from being driven
        # toward prism excitation merely because their event stream is sparse.
        event_deficit = 0.55 * missing_edges + 0.30 * inefficient + 0.15 * overlap
        raw = low_motion * (0.55 + 0.45 * event_deficit) - 0.20 * invalid
        return float(np.clip(raw, 0.0, 1.0))

    def _costs(self, deficit: float) -> np.ndarray:
        peaks = np.asarray([self._peak_speed(candidate) for candidate in self.candidates], dtype=np.float64)
        normalized = np.clip(peaks / self.limits.max_speed_rad_s, 0.0, 1.0)
        if self.objective == "event_count":
            fit = -normalized
        else:
            fit = np.square(normalized - deficit)
            diversity = np.asarray([candidate.waveform != "constant" for candidate in self.candidates], dtype=np.float64)
            fit -= 0.03 * diversity * deficit * (1.0 - deficit)
            if self.useful_speed_cap_rad_s is not None:
                over_cap = peaks > self.useful_speed_cap_rad_s + 1e-9
                if np.all(over_cap):
                    raise ValueError("useful_speed_cap_rad_s excludes every candidate trajectory.")
                fit = np.where(over_cap, np.inf, fit)
        current = self.candidates[self.current_index]
        switching = np.asarray([candidate != current for candidate in self.candidates], dtype=np.float64)
        return fit + self.effort_weight * normalized**2 + self.switch_weight * switching

    def select(self, state: Mapping[str, float], *, timestamp: float) -> TrajectoryParameters:
        if not np.isfinite(timestamp):
            raise ValueError("timestamp must be finite.")
        if self.last_timestamp is None:
            self.reset(timestamp=float(timestamp))
        elif timestamp < self.last_timestamp:
            raise ValueError("Controller timestamps must be non-decreasing.")
        assert self.last_switch_time is not None
        deficit = self._information_deficit(state)
        costs = self._costs(deficit)
        unconstrained = int(np.argmin(costs))
        improvement = float(costs[self.current_index] - costs[unconstrained])
        dwell_ready = float(timestamp) - self.last_switch_time >= self.minimum_dwell_s
        switched = unconstrained != self.current_index and dwell_ready and improvement >= self.switch_margin
        if switched:
            self.current_index = unconstrained
            self.last_switch_time = float(timestamp)
        self.last_timestamp = float(timestamp)
        self.last_decision = TrajectoryDecision(
            timestamp=float(timestamp),
            information_deficit=deficit,
            unconstrained_index=unconstrained,
            selected_index=self.current_index,
            switched=switched,
            objective=self.objective,
            candidate_costs=tuple(float(value) for value in costs),
        )
        return self.candidates[self.current_index]

    def metadata(self) -> dict[str, object]:
        return {
            "controller": self.name,
            "objective": self.objective,
            "candidate_count": len(self.candidates),
            "limits": {
                "max_speed_rad_s": self.limits.max_speed_rad_s,
                "max_acceleration_rad_s2": self.limits.max_acceleration_rad_s2,
                "min_duration_s": self.limits.min_duration_s,
                "max_duration_s": self.limits.max_duration_s,
            },
            "minimum_dwell_s": self.minimum_dwell_s,
            "switch_margin": self.switch_margin,
            "effort_weight": self.effort_weight,
            "switch_weight": self.switch_weight,
            "imu_prior_weight": self.imu_prior_weight,
            "imu_motion_scale_rad_s": self.imu_motion_scale_rad_s,
            "useful_speed_cap_rad_s": self.useful_speed_cap_rad_s,
        }
