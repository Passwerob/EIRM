"""Time-continuous rotation policies."""

from __future__ import annotations

import numpy as np

from .types import ImuStream


def motion_quantile_thresholds(
    imu: ImuStream,
    quantiles: tuple[float, ...] = (0.2, 0.4, 0.6, 0.8),
) -> tuple[float, ...]:
    """Return scene-specific gear thresholds from the IMU motion distribution."""
    if len(quantiles) == 0 or any(not 0.0 < value < 1.0 for value in quantiles):
        raise ValueError("quantiles must be non-empty and strictly inside (0, 1).")
    if any(left >= right for left, right in zip(quantiles[:-1], quantiles[1:])):
        raise ValueError("quantiles must be strictly increasing.")
    motion = np.linalg.norm(np.asarray(imu.gyro, dtype=np.float64), axis=1)
    thresholds = tuple(float(value) for value in np.quantile(motion, quantiles))
    if any(left >= right for left, right in zip(thresholds[:-1], thresholds[1:])):
        raise ValueError("IMU motion quantiles are not distinct; use fixed thresholds or more data.")
    return thresholds


def motion_adaptive_hysteresis(
    imu: ImuStream,
    *,
    fraction: float = 0.1,
    minimum: float = 1e-3,
    maximum: float = 1e-2,
) -> float:
    """Choose a hysteresis margin from the robust scene motion range."""
    if not 0.0 < fraction or minimum <= 0.0 or maximum < minimum:
        raise ValueError("fraction must be positive and maximum must be >= minimum > 0.")
    motion = np.linalg.norm(np.asarray(imu.gyro, dtype=np.float64), axis=1)
    robust_range = float(np.quantile(motion, 0.99) - np.quantile(motion, 0.01))
    return float(np.clip(fraction * robust_range, minimum, maximum))


class ConstantRotation:
    """Analytic constant-speed rotation with theta(t) = theta0 + omega * dt."""

    name = "constant"

    def __init__(self, rpm: float) -> None:
        self.rpm = float(rpm)
        self.omega_rad_s = self.rpm * 2.0 * np.pi / 60.0

    def theta_values(
        self,
        timestamps: np.ndarray,
        *,
        theta0: float = 0.0,
        reference_time: float | None = None,
    ) -> np.ndarray:
        times = np.asarray(timestamps, dtype=np.float64).reshape(-1)
        if len(times) > 1 and not np.all(np.diff(times) >= 0.0):
            raise ValueError("Rotation timestamps must be sorted.")
        if len(times) == 0:
            return times
        origin = float(times[0] if reference_time is None else reference_time)
        return float(theta0) + self.omega_rad_s * (times - origin)

    def omega_values(self, timestamps: np.ndarray) -> np.ndarray:
        """Return constant angular velocity at arbitrary timestamps."""
        times = np.asarray(timestamps, dtype=np.float64).reshape(-1)
        if len(times) and (not np.all(np.isfinite(times)) or not np.all(np.diff(times) >= 0.0)):
            raise ValueError("Rotation timestamps must be finite and sorted.")
        return np.full(len(times), self.omega_rad_s, dtype=np.float64)

    def metadata(self) -> dict[str, float | str]:
        return {"policy": self.name, "rpm": self.rpm, "omega_rad_s": self.omega_rad_s}


class ScheduledRotation:
    """Piecewise-linear prism speed schedule with globally continuous phase."""

    name = "scheduled_optical_modulation"

    def __init__(self, timestamps: np.ndarray, rpm: np.ndarray, *, initial_angle: float = 0.0) -> None:
        self.t = np.asarray(timestamps, dtype=np.float64).reshape(-1)
        self.rpm = np.asarray(rpm, dtype=np.float64).reshape(-1)
        if len(self.t) < 2 or len(self.t) != len(self.rpm) or not np.all(np.isfinite(self.t)) or not np.all(np.diff(self.t) > 0.0):
            raise ValueError("Scheduled rotation needs strictly increasing timestamps and one speed per sample.")
        if not np.all(np.isfinite(self.rpm)) or np.any(self.rpm < 0.0):
            raise ValueError("Scheduled RPM values must be finite and non-negative.")
        self.initial_angle = float(initial_angle)
        omega = self.rpm * (2.0 * np.pi / 60.0)
        increments = np.diff(self.t) * (omega[:-1] + omega[1:]) * 0.5
        self._omega = omega
        self._angles = self.initial_angle + np.concatenate(([0.0], np.cumsum(increments, dtype=np.float64)))

    def _validate(self, timestamps: np.ndarray) -> np.ndarray:
        values = np.asarray(timestamps, dtype=np.float64).reshape(-1)
        if len(values) and (not np.all(np.isfinite(values)) or not np.all(np.diff(values) >= 0.0)):
            raise ValueError("Rotation timestamps must be finite and sorted.")
        if len(values) and (values[0] < self.t[0] or values[-1] > self.t[-1]):
            raise ValueError("Requested rotation timestamps fall outside the schedule range.")
        return values

    def theta_values(self, timestamps: np.ndarray, *, theta0: float = 0.0, reference_time: float | None = None) -> np.ndarray:
        values = self._validate(timestamps)
        origin = float(self.t[0] if reference_time is None else reference_time)
        if origin < self.t[0] or origin > self.t[-1]:
            raise ValueError("reference_time falls outside the schedule range.")
        return float(theta0) + np.interp(values, self.t, self._angles) - float(np.interp(origin, self.t, self._angles))

    def omega_values(self, timestamps: np.ndarray) -> np.ndarray:
        return np.interp(self._validate(timestamps), self.t, self._omega)

    def metadata(self) -> dict[str, object]:
        return {"policy": self.name, "sample_count": len(self.t), "rpm_range": [float(self.rpm.min()), float(self.rpm.max())]}


class ImuRotation:
    """Piecewise-linear prism rotation driven by one IMU gyro axis.

    The selected angular velocity is integrated with the trapezoidal rule and
    interpolated at arbitrary event/frame timestamps. ``initial_omega_rad_s``
    is useful when the first IMU sample is just outside the RGB interval.
    """

    name = "imu_gyro_axis"

    def __init__(
        self,
        imu: ImuStream,
        *,
        axis: int = 2,
        sign: float = 1.0,
        initial_angle: float = 0.0,
        initial_omega_rad_s: float | None = None,
    ) -> None:
        if axis not in (0, 1, 2):
            raise ValueError("IMU gyro axis must be 0, 1, or 2.")
        if not np.isfinite(sign) or sign == 0.0:
            raise ValueError("IMU gyro sign must be finite and non-zero.")
        self.imu = imu
        self.axis = int(axis)
        self.sign = float(sign)
        self.initial_angle = float(initial_angle)
        omega = self.sign * imu.gyro[:, self.axis]
        if initial_omega_rad_s is not None:
            omega = omega.copy()
            omega[0] = float(initial_omega_rad_s)
        self._omega = omega
        self._angles = self.initial_angle + np.concatenate(([0.0], np.cumsum(np.diff(imu.t) * (omega[:-1] + omega[1:]) * 0.5)))

    def theta_values(self, timestamps: np.ndarray, *, theta0: float = 0.0, reference_time: float | None = None) -> np.ndarray:
        times = np.asarray(timestamps, dtype=np.float64).reshape(-1)
        if len(times) and (not np.all(np.isfinite(times)) or not np.all(np.diff(times) >= 0.0)):
            raise ValueError("Rotation timestamps must be finite and sorted.")
        if len(times) and (times[0] < self.imu.t[0] or times[-1] > self.imu.t[-1]):
            raise ValueError("Requested rotation timestamps fall outside the IMU time range.")
        angles = np.interp(times, self.imu.t, self._angles)
        origin = float(self.imu.t[0] if reference_time is None else reference_time)
        origin_angle = float(np.interp(origin, self.imu.t, self._angles))
        return float(theta0) + angles - origin_angle

    def metadata(self) -> dict[str, object]:
        return {
            "policy": self.name,
            "gyro_axis": self.axis,
            "gyro_sign": self.sign,
            "gyro_unit": "rad/s",
            "sample_count": len(self.imu.t),
            "time_range_seconds": [float(self.imu.t[0]), float(self.imu.t[-1])],
        }


class ImuGearRotation:
    """AMI rotation with IMU-conditioned discrete speed gears.

    Gears are ordered from low to high measured motion.  The first gear has
    the largest AMI speed, while the last gear has the smallest speed.  The
    outer policy changes gears with hysteresis; the inner command follows the
    target speed with a finite acceleration.  Phase is integrated from
    relative timestamp differences in float64, so large absolute dataset
    timestamps do not enter the accumulator.
    """

    name = "imu_gear_modulated"

    def __init__(
        self,
        imu: ImuStream,
        *,
        base_rpm: float = 720.0,
        motion_axis: int | None = None,
        speed_thresholds_rad_s: tuple[float, ...] = (0.05, 0.15, 0.30, 0.60),
        gear_scales: tuple[float, ...] = (1.5, 1.0, 0.5, 0.25, 0.0),
        hysteresis_rad_s: float = 0.01,
        smoothing_seconds: float = 0.05,
        max_accel_rad_s2: float = 100.0,
        initial_angle: float = 0.0,
    ) -> None:
        if not np.isfinite(base_rpm) or base_rpm < 0.0:
            raise ValueError("base_rpm must be finite and non-negative.")
        if motion_axis is not None and motion_axis not in (0, 1, 2):
            raise ValueError("motion_axis must be None, 0, 1, or 2.")
        if len(gear_scales) < 2 or not np.all(np.isfinite(gear_scales)) or any(scale < 0.0 for scale in gear_scales):
            raise ValueError("gear_scales must contain at least two non-negative finite values.")
        if len(speed_thresholds_rad_s) != len(gear_scales) - 1:
            raise ValueError("There must be one fewer speed threshold than gear scales.")
        if any(left >= right for left, right in zip(speed_thresholds_rad_s[:-1], speed_thresholds_rad_s[1:])):
            raise ValueError("speed_thresholds_rad_s must be strictly increasing.")
        if not np.all(np.isfinite(speed_thresholds_rad_s)) or any(value <= 0.0 for value in speed_thresholds_rad_s):
            raise ValueError("speed_thresholds_rad_s must be positive and finite.")
        if not np.isfinite(hysteresis_rad_s) or hysteresis_rad_s < 0.0:
            raise ValueError("hysteresis_rad_s must be finite and non-negative.")
        if not np.isfinite(smoothing_seconds) or smoothing_seconds < 0.0:
            raise ValueError("smoothing_seconds must be finite and non-negative.")
        if not np.isfinite(max_accel_rad_s2) or max_accel_rad_s2 <= 0.0:
            raise ValueError("max_accel_rad_s2 must be finite and positive.")

        self.imu = imu
        self.base_rpm = float(base_rpm)
        self.motion_axis = motion_axis
        self.speed_thresholds_rad_s = tuple(float(value) for value in speed_thresholds_rad_s)
        self.gear_scales = tuple(float(value) for value in gear_scales)
        self.hysteresis_rad_s = float(hysteresis_rad_s)
        self.smoothing_seconds = float(smoothing_seconds)
        self.max_accel_rad_s2 = float(max_accel_rad_s2)
        self.initial_angle = float(initial_angle)

        gyro = np.asarray(imu.gyro, dtype=np.float64)
        motion = np.abs(gyro[:, motion_axis]) if motion_axis is not None else np.linalg.norm(gyro, axis=1)
        self._motion = motion
        self._motion_ema = self._smooth_motion(motion)
        self._gears = self._select_gears(self._motion_ema)
        target = self.omega_base * np.asarray([self.gear_scales[index] for index in self._gears], dtype=np.float64)
        self._omega = self._ramp(target)
        increments = np.diff(imu.t) * (self._omega[:-1] + self._omega[1:]) * 0.5
        self._angles = self.initial_angle + np.concatenate(([0.0], np.cumsum(increments, dtype=np.float64)))

    @property
    def omega_base(self) -> float:
        return self.base_rpm * 2.0 * np.pi / 60.0

    @property
    def gear_indices(self) -> np.ndarray:
        """Selected gear per IMU sample, exposed for offline label generation."""
        return self._gears.copy()

    @property
    def smoothed_motion_rad_s(self) -> np.ndarray:
        """Smoothed motion magnitude used by the gear selector."""
        return self._motion_ema.copy()

    def _smooth_motion(self, motion: np.ndarray) -> np.ndarray:
        if self.smoothing_seconds == 0.0:
            return motion.copy()
        result = np.empty_like(motion, dtype=np.float64)
        result[0] = motion[0]
        for index, dt in enumerate(np.diff(self.imu.t), start=1):
            alpha = 1.0 - np.exp(-float(dt) / self.smoothing_seconds)
            result[index] = result[index - 1] + alpha * (motion[index] - result[index - 1])
        return result

    def _select_gears(self, motion: np.ndarray) -> np.ndarray:
        gears = np.empty(len(motion), dtype=np.int32)
        gear = 0
        for index, value in enumerate(motion):
            while gear < len(self.gear_scales) - 1 and value > self.speed_thresholds_rad_s[gear] + self.hysteresis_rad_s:
                gear += 1
            while gear > 0 and value < self.speed_thresholds_rad_s[gear - 1] - self.hysteresis_rad_s:
                gear -= 1
            gears[index] = gear
        return gears

    def _ramp(self, target: np.ndarray) -> np.ndarray:
        omega = np.empty_like(target, dtype=np.float64)
        omega[0] = target[0]
        for index, dt in enumerate(np.diff(self.imu.t), start=1):
            limit = self.max_accel_rad_s2 * float(dt)
            omega[index] = omega[index - 1] + np.clip(target[index] - omega[index - 1], -limit, limit)
        return omega

    def theta_values(self, timestamps: np.ndarray, *, theta0: float = 0.0, reference_time: float | None = None) -> np.ndarray:
        times = np.asarray(timestamps, dtype=np.float64).reshape(-1)
        if len(times) and (not np.all(np.isfinite(times)) or not np.all(np.diff(times) >= 0.0)):
            raise ValueError("Rotation timestamps must be finite and sorted.")
        if len(times) and (times[0] < self.imu.t[0] or times[-1] > self.imu.t[-1]):
            raise ValueError("Requested rotation timestamps fall outside the IMU time range.")
        angles = np.interp(times, self.imu.t, self._angles)
        origin = float(self.imu.t[0] if reference_time is None else reference_time)
        if origin < self.imu.t[0] or origin > self.imu.t[-1]:
            raise ValueError("reference_time falls outside the IMU time range.")
        origin_angle = float(np.interp(origin, self.imu.t, self._angles))
        return float(theta0) + angles - origin_angle

    def omega_values(self, timestamps: np.ndarray) -> np.ndarray:
        """Interpolate the acceleration-limited prism speed in rad/s."""
        times = np.asarray(timestamps, dtype=np.float64).reshape(-1)
        if len(times) and (not np.all(np.isfinite(times)) or not np.all(np.diff(times) >= 0.0)):
            raise ValueError("Rotation timestamps must be finite and sorted.")
        if len(times) and (times[0] < self.imu.t[0] or times[-1] > self.imu.t[-1]):
            raise ValueError("Requested rotation timestamps fall outside the IMU time range.")
        return np.interp(times, self.imu.t, self._omega)

    def metadata(self) -> dict[str, object]:
        return {
            "policy": self.name,
            "base_rpm": self.base_rpm,
            "motion_axis": self.motion_axis,
            "speed_thresholds_rad_s": list(self.speed_thresholds_rad_s),
            "gear_scales": list(self.gear_scales),
            "hysteresis_rad_s": self.hysteresis_rad_s,
            "smoothing_seconds": self.smoothing_seconds,
            "max_accel_rad_s2": self.max_accel_rad_s2,
            "gear_counts": [int(np.count_nonzero(self._gears == index)) for index in range(len(self.gear_scales))],
            "motion_range_rad_s": [float(np.min(self._motion)), float(np.max(self._motion))],
        }
