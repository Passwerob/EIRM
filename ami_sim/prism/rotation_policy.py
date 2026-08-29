"""Continuous, physically realizable prism rotation policies."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class RotationState:
    timestamp: float
    theta: float
    omega: float
    policy_state: str = "constant"


class RotationPolicy(ABC):
    """Policy owns integration of theta; warp code never integrates rotation itself."""

    name = "base"

    @abstractmethod
    def omega(self, timestamp: float, state: RotationState | None = None) -> float:
        """Return angular velocity in radians per second."""

    def initial_state(self, timestamp: float, theta0: float = 0.0) -> RotationState:
        return RotationState(float(timestamp), float(theta0), float(self.omega(timestamp)), self.name)

    def advance(self, state: RotationState, timestamp: float) -> RotationState:
        timestamp = float(timestamp)
        delta = timestamp - state.timestamp
        if delta < -1e-12:
            raise ValueError("Rotation states must be advanced in non-decreasing timestamp order.")
        # Explicit Euler is exact for the constant baseline and is a documented
        # first-order integrator for future adaptive policies.
        current_omega = float(self.omega(state.timestamp, state))
        theta = state.theta + current_omega * delta
        return RotationState(timestamp, theta, float(self.omega(timestamp, state)), self.name)

    def states_at(self, timestamps: np.ndarray, *, theta0: float = 0.0, reference_time: float | None = None) -> list[RotationState]:
        values = np.asarray(timestamps, dtype=np.float64).reshape(-1)
        if len(values) == 0:
            return []
        if len(values) > 1 and not np.all(np.diff(values) >= 0):
            raise ValueError("Rotation policy timestamps must be sorted for continuous integration.")
        start_time = float(values[0] if reference_time is None else reference_time)
        state = self.initial_state(start_time, theta0)
        states: list[RotationState] = []
        for timestamp in values:
            state = self.advance(state, float(timestamp))
            states.append(state)
        return states

    def theta_values(
        self,
        timestamps: np.ndarray,
        *,
        theta0: float = 0.0,
        reference_time: float | None = None,
    ) -> np.ndarray:
        """Return integrated angles without exposing per-sample state objects.

        The default preserves the policy's existing integration semantics. Policies
        with a closed-form trajectory can override this to avoid Python work for
        very large event streams.
        """
        states = self.states_at(timestamps, theta0=theta0, reference_time=reference_time)
        return np.asarray([state.theta for state in states], dtype=np.float64)

    def theta_at(self, timestamp: float, *, theta0: float = 0.0, reference_time: float = 0.0) -> float:
        return self.states_at(np.asarray([reference_time, timestamp], dtype=np.float64), theta0=theta0, reference_time=reference_time)[-1].theta

    def metadata(self) -> dict[str, object]:
        return {"rotation_policy": self.name}


class ConstantRotationPolicy(RotationPolicy):
    name = "constant"

    def __init__(self, rpm: float = 720.0) -> None:
        self.rpm = float(rpm)
        self._omega = self.rpm * 2.0 * np.pi / 60.0

    def omega(self, timestamp: float, state: RotationState | None = None) -> float:
        del timestamp, state
        return self._omega

    def theta_values(
        self,
        timestamps: np.ndarray,
        *,
        theta0: float = 0.0,
        reference_time: float | None = None,
    ) -> np.ndarray:
        values = np.asarray(timestamps, dtype=np.float64).reshape(-1)
        if len(values) == 0:
            return values
        if len(values) > 1 and not np.all(np.diff(values) >= 0):
            raise ValueError("Rotation policy timestamps must be sorted for continuous integration.")
        reference = float(values[0] if reference_time is None else reference_time)
        return float(theta0) + self._omega * (values - reference)

    def metadata(self) -> dict[str, object]:
        return {**super().metadata(), "rpm": self.rpm, "omega_rad_s": self._omega}


class MotionMagnitudeAdaptiveRotation(RotationPolicy):
    """Naive adaptive baseline retained for characterization, not a learned policy."""

    name = "naive_motion_magnitude_adaptive"

    def __init__(self, min_rpm: float = 360.0, max_rpm: float = 960.0, gain: float = 1.0) -> None:
        if min_rpm < 0 or max_rpm < min_rpm:
            raise ValueError("Adaptive RPM limits are invalid.")
        self.min_rpm = float(min_rpm)
        self.max_rpm = float(max_rpm)
        self.gain = float(gain)
        self.motion_magnitude = 0.0

    def set_motion_magnitude(self, magnitude: float) -> None:
        self.motion_magnitude = max(0.0, float(magnitude))

    def omega(self, timestamp: float, state: RotationState | None = None) -> float:
        del timestamp, state
        rpm = np.clip(self.max_rpm - self.gain * self.motion_magnitude, self.min_rpm, self.max_rpm)
        return float(rpm) * 2.0 * np.pi / 60.0

    def metadata(self) -> dict[str, object]:
        return {**super().metadata(), "min_rpm": self.min_rpm, "max_rpm": self.max_rpm, "gain": self.gain}
