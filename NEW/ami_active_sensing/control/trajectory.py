"""Single-DoF optical excitation trajectories compatible with PrismWarper."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal, Protocol, Sequence, runtime_checkable

import numpy as np


Waveform = Literal["constant", "sinusoid", "triangle", "chirp"]


@dataclass(frozen=True)
class TrajectoryParameters:
    """Parameters for one phase-continuous control segment.

    Angles describe rotation about the existing prism's single actuator axis.
    They do not change the wedge apex angle or add another optical axis.
    """

    waveform: Waveform
    amplitude_rad: float = 0.0
    frequency_hz: float = 0.0
    phase_rad: float = 0.0
    direction: int = 1
    duration_s: float = 0.1
    angular_speed_rad_s: float | None = None
    end_frequency_hz: float | None = None

    def __post_init__(self) -> None:
        numeric = (
            self.amplitude_rad,
            self.frequency_hz,
            self.phase_rad,
            self.duration_s,
        )
        if not np.all(np.isfinite(numeric)):
            raise ValueError("Trajectory parameters must be finite.")
        if self.direction not in (-1, 1):
            raise ValueError("direction must be -1 or +1.")
        if self.duration_s <= 0.0:
            raise ValueError("duration_s must be positive.")
        if self.amplitude_rad < 0.0 or self.frequency_hz < 0.0:
            raise ValueError("amplitude_rad and frequency_hz must be non-negative.")
        if self.waveform == "constant":
            if self.angular_speed_rad_s is None or not np.isfinite(self.angular_speed_rad_s):
                raise ValueError("constant trajectories require finite angular_speed_rad_s.")
            if self.angular_speed_rad_s < 0.0:
                raise ValueError("angular_speed_rad_s is a magnitude and must be non-negative.")
        elif self.angular_speed_rad_s is not None:
            raise ValueError("angular_speed_rad_s is only valid for constant trajectories.")
        if self.waveform in ("sinusoid", "triangle", "chirp") and (
            self.amplitude_rad <= 0.0 or self.frequency_hz <= 0.0
        ):
            raise ValueError(f"{self.waveform} requires positive amplitude_rad and frequency_hz.")
        if self.waveform == "chirp":
            if self.end_frequency_hz is None or not np.isfinite(self.end_frequency_hz) or self.end_frequency_hz <= 0.0:
                raise ValueError("chirp requires positive finite end_frequency_hz.")
        elif self.end_frequency_hz is not None:
            raise ValueError("end_frequency_hz is only valid for chirp trajectories.")

    def metadata(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class TrajectoryLimits:
    """Physical limits applied uniformly to every candidate trajectory."""

    max_speed_rad_s: float
    max_acceleration_rad_s2: float
    min_duration_s: float = 0.0
    max_duration_s: float = float("inf")

    def __post_init__(self) -> None:
        values = (self.max_speed_rad_s, self.max_acceleration_rad_s2, self.min_duration_s, self.max_duration_s)
        if np.any(np.isnan(values)) or self.max_speed_rad_s <= 0.0 or self.max_acceleration_rad_s2 <= 0.0:
            raise ValueError("Speed and acceleration limits must be positive.")
        if self.min_duration_s < 0.0 or self.max_duration_s < self.min_duration_s:
            raise ValueError("Duration limits must be non-negative and ordered.")


@runtime_checkable
class OpticalTrajectory(Protocol):
    """Interface consumed by the existing root ``PrismWarper``."""

    def theta_values(
        self,
        timestamps: np.ndarray,
        *,
        theta0: float = 0.0,
        reference_time: float | None = None,
    ) -> np.ndarray: ...

    def omega_values(self, timestamps: np.ndarray) -> np.ndarray: ...

    def metadata(self) -> dict[str, object]: ...


def _validate_times(timestamps: np.ndarray) -> np.ndarray:
    times = np.asarray(timestamps, dtype=np.float64).reshape(-1)
    if len(times) and (not np.all(np.isfinite(times)) or not np.all(np.diff(times) >= 0.0)):
        raise ValueError("Trajectory timestamps must be finite and sorted.")
    return times


class AnalyticTrajectory:
    """One analytic trajectory segment with angle zeroed at ``start_time``."""

    def __init__(self, parameters: TrajectoryParameters, *, start_time: float = 0.0) -> None:
        if not np.isfinite(start_time):
            raise ValueError("start_time must be finite.")
        self.parameters = parameters
        self.start_time = float(start_time)
        self.end_time = self.start_time + parameters.duration_s

    def _relative_phase(self, relative_time: np.ndarray) -> np.ndarray:
        p = self.parameters
        if p.waveform == "chirp":
            rate = (float(p.end_frequency_hz) - p.frequency_hz) / p.duration_s
            return 2.0 * np.pi * (p.frequency_hz * relative_time + 0.5 * rate * relative_time**2) + p.phase_rad
        return 2.0 * np.pi * p.frequency_hz * relative_time + p.phase_rad

    def _angle_from_start(self, timestamps: np.ndarray) -> np.ndarray:
        relative = timestamps - self.start_time
        p = self.parameters
        if p.waveform == "constant":
            return p.direction * float(p.angular_speed_rad_s) * relative
        phase = self._relative_phase(relative)
        phase_at_start = p.phase_rad
        if p.waveform in ("sinusoid", "chirp"):
            wave = np.sin(phase) - np.sin(phase_at_start)
        elif p.waveform == "triangle":
            wave = (2.0 / np.pi) * (np.arcsin(np.sin(phase)) - np.arcsin(np.sin(phase_at_start)))
        else:  # pragma: no cover - protected by the Literal/dataclass validation
            raise ValueError(f"Unsupported waveform {p.waveform!r}.")
        return p.direction * p.amplitude_rad * wave

    def theta_values(
        self,
        timestamps: np.ndarray,
        *,
        theta0: float = 0.0,
        reference_time: float | None = None,
    ) -> np.ndarray:
        times = _validate_times(timestamps)
        if not len(times):
            return times
        origin = self.start_time if reference_time is None else float(reference_time)
        if not np.isfinite(origin):
            raise ValueError("reference_time must be finite.")
        return float(theta0) + self._angle_from_start(times) - self._angle_from_start(np.asarray([origin]))[0]

    def omega_values(self, timestamps: np.ndarray) -> np.ndarray:
        times = _validate_times(timestamps)
        relative = times - self.start_time
        p = self.parameters
        if p.waveform == "constant":
            return np.full(len(times), p.direction * float(p.angular_speed_rad_s), dtype=np.float64)
        phase = self._relative_phase(relative)
        if p.waveform == "chirp":
            rate = (float(p.end_frequency_hz) - p.frequency_hz) / p.duration_s
            frequency = p.frequency_hz + rate * relative
        else:
            frequency = np.full(len(times), p.frequency_hz, dtype=np.float64)
        if p.waveform in ("sinusoid", "chirp"):
            derivative = 2.0 * np.pi * frequency * np.cos(phase)
        elif p.waveform == "triangle":
            derivative = 4.0 * frequency * np.sign(np.cos(phase))
        else:  # pragma: no cover
            raise ValueError(f"Unsupported waveform {p.waveform!r}.")
        return p.direction * p.amplitude_rad * derivative

    def metadata(self) -> dict[str, object]:
        return {
            "policy": "analytic_optical_excitation",
            "single_dof": True,
            "start_time_seconds": self.start_time,
            "parameters": self.parameters.metadata(),
        }


class PiecewiseTrajectory:
    """A phase-continuous sequence of analytic trajectory segments."""

    def __init__(self, segments: Sequence[tuple[float, TrajectoryParameters]]) -> None:
        if not segments:
            raise ValueError("PiecewiseTrajectory requires at least one segment.")
        starts = np.asarray([item[0] for item in segments], dtype=np.float64)
        if not np.all(np.isfinite(starts)) or (len(starts) > 1 and not np.all(np.diff(starts) > 0.0)):
            raise ValueError("Segment start times must be finite and strictly increasing.")
        self.segments = tuple(AnalyticTrajectory(params, start_time=float(start)) for start, params in segments)
        for left, right in zip(self.segments[:-1], self.segments[1:]):
            if not np.isclose(left.end_time, right.start_time, rtol=0.0, atol=1e-9):
                raise ValueError("Piecewise segments must be contiguous and durations must match their start times.")
        self.start_time = self.segments[0].start_time
        self.end_time = self.segments[-1].end_time
        offsets = [0.0]
        for segment in self.segments[:-1]:
            offsets.append(offsets[-1] + float(segment._angle_from_start(np.asarray([segment.end_time]))[0]))
        self._offsets = np.asarray(offsets, dtype=np.float64)
        self._starts = starts

    def _validate_range(self, times: np.ndarray) -> None:
        if len(times) and (times[0] < self.start_time - 1e-12 or times[-1] > self.end_time + 1e-12):
            raise ValueError("Requested timestamps fall outside the piecewise trajectory range.")

    def _absolute_angle(self, times: np.ndarray) -> np.ndarray:
        indices = np.searchsorted(self._starts, times, side="right") - 1
        indices = np.clip(indices, 0, len(self.segments) - 1)
        values = np.empty(len(times), dtype=np.float64)
        for index, segment in enumerate(self.segments):
            mask = indices == index
            if np.any(mask):
                values[mask] = self._offsets[index] + segment._angle_from_start(times[mask])
        return values

    def theta_values(
        self,
        timestamps: np.ndarray,
        *,
        theta0: float = 0.0,
        reference_time: float | None = None,
    ) -> np.ndarray:
        times = _validate_times(timestamps)
        self._validate_range(times)
        if not len(times):
            return times
        origin = self.start_time if reference_time is None else float(reference_time)
        self._validate_range(np.asarray([origin]))
        return float(theta0) + self._absolute_angle(times) - self._absolute_angle(np.asarray([origin]))[0]

    def omega_values(self, timestamps: np.ndarray) -> np.ndarray:
        times = _validate_times(timestamps)
        self._validate_range(times)
        indices = np.searchsorted(self._starts, times, side="right") - 1
        indices = np.clip(indices, 0, len(self.segments) - 1)
        values = np.empty(len(times), dtype=np.float64)
        for index, segment in enumerate(self.segments):
            mask = indices == index
            if np.any(mask):
                values[mask] = segment.omega_values(times[mask])
        return values

    def metadata(self) -> dict[str, object]:
        return {
            "policy": "piecewise_optical_excitation",
            "single_dof": True,
            "time_range_seconds": [self.start_time, self.end_time],
            "segments": [segment.metadata() for segment in self.segments],
        }


class ActuatorConstrainedTrajectory:
    """Sample, rate-limit, and integrate a desired single-DoF trajectory."""

    def __init__(
        self,
        desired: OpticalTrajectory,
        sample_times: np.ndarray,
        limits: TrajectoryLimits,
        *,
        initial_omega_rad_s: float = 0.0,
    ) -> None:
        times = _validate_times(sample_times)
        if len(times) < 2 or not np.all(np.diff(times) > 0.0):
            raise ValueError("Actuator constraint samples must be strictly increasing.")
        if not np.isfinite(initial_omega_rad_s) or abs(initial_omega_rad_s) > limits.max_speed_rad_s:
            raise ValueError("initial_omega_rad_s must satisfy the speed limit.")
        desired_omega = np.asarray(desired.omega_values(times), dtype=np.float64)
        if desired_omega.shape != times.shape or not np.all(np.isfinite(desired_omega)):
            raise ValueError("Desired trajectory returned invalid angular velocity.")
        omega = np.empty_like(desired_omega)
        omega[0] = float(initial_omega_rad_s)
        for index, dt in enumerate(np.diff(times), start=1):
            target = float(np.clip(desired_omega[index], -limits.max_speed_rad_s, limits.max_speed_rad_s))
            max_delta = limits.max_acceleration_rad_s2 * float(dt)
            omega[index] = omega[index - 1] + np.clip(target - omega[index - 1], -max_delta, max_delta)
        increments = np.diff(times) * (omega[:-1] + omega[1:]) * 0.5
        self.desired = desired
        self.t = times
        self.omega = omega
        self.angle = np.concatenate(([0.0], np.cumsum(increments, dtype=np.float64)))
        self.limits = limits

    def _validate_range(self, times: np.ndarray) -> None:
        if len(times) and (times[0] < self.t[0] - 1e-12 or times[-1] > self.t[-1] + 1e-12):
            raise ValueError("Requested timestamps fall outside the actuator trajectory range.")

    def theta_values(
        self,
        timestamps: np.ndarray,
        *,
        theta0: float = 0.0,
        reference_time: float | None = None,
    ) -> np.ndarray:
        times = _validate_times(timestamps)
        self._validate_range(times)
        if not len(times):
            return times
        origin = self.t[0] if reference_time is None else float(reference_time)
        self._validate_range(np.asarray([origin]))
        return float(theta0) + np.interp(times, self.t, self.angle) - float(np.interp(origin, self.t, self.angle))

    def omega_values(self, timestamps: np.ndarray) -> np.ndarray:
        times = _validate_times(timestamps)
        self._validate_range(times)
        return np.interp(times, self.t, self.omega)

    def metadata(self) -> dict[str, object]:
        return {
            "policy": "actuator_constrained_optical_excitation",
            "single_dof": True,
            "sample_count": len(self.t),
            "time_range_seconds": [float(self.t[0]), float(self.t[-1])],
            "speed_range_rad_s": [float(np.min(self.omega)), float(np.max(self.omega))],
            "limits": {
                "max_speed_rad_s": self.limits.max_speed_rad_s,
                "max_acceleration_rad_s2": self.limits.max_acceleration_rad_s2,
            },
            "desired": self.desired.metadata(),
        }


def make_trajectory(parameters: TrajectoryParameters, *, start_time: float = 0.0) -> AnalyticTrajectory:
    """Construct one analytic trajectory segment."""
    return AnalyticTrajectory(parameters, start_time=start_time)


def validate_trajectory_limits(
    parameters: TrajectoryParameters,
    limits: TrajectoryLimits,
    *,
    sample_count: int = 1025,
) -> None:
    """Reject a candidate that violates sampled actuator limits.

    Triangle-wave acceleration is undefined at its corners. Such candidates are
    accepted only by the speed and duration checks; a real actuator adapter must
    replace the ideal corner with a ramp before hardware execution.
    """
    if sample_count < 3:
        raise ValueError("sample_count must be at least three.")
    if not limits.min_duration_s <= parameters.duration_s <= limits.max_duration_s:
        raise ValueError("Trajectory duration violates configured limits.")
    trajectory = make_trajectory(parameters)
    times = np.linspace(0.0, parameters.duration_s, sample_count, dtype=np.float64)
    omega = trajectory.omega_values(times)
    if np.max(np.abs(omega), initial=0.0) > limits.max_speed_rad_s + 1e-9:
        raise ValueError("Trajectory exceeds max_speed_rad_s.")
    if parameters.waveform != "triangle":
        acceleration = np.gradient(omega, times)
        if np.max(np.abs(acceleration), initial=0.0) > limits.max_acceleration_rad_s2 * 1.002:
            raise ValueError("Trajectory exceeds max_acceleration_rad_s2.")


def sample_candidate_trajectories(
    candidates: Sequence[TrajectoryParameters],
    limits: TrajectoryLimits,
) -> list[TrajectoryParameters]:
    """Return valid candidates in stable order, removing exact duplicates."""
    valid: list[TrajectoryParameters] = []
    seen: set[TrajectoryParameters] = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        validate_trajectory_limits(candidate, limits)
        seen.add(candidate)
        valid.append(candidate)
    if not valid:
        raise ValueError("No trajectory candidate satisfies the configured limits.")
    return valid
