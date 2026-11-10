"""Shared data types and timestamp utilities."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass(frozen=True)
class CameraIntrinsics:
    """OpenCV-style intrinsics at the exact image resolution in use."""

    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float

    def __post_init__(self) -> None:
        if self.width <= 0 or self.height <= 0:
            raise ValueError("Image dimensions must be positive.")
        if self.fx <= 0 or self.fy <= 0:
            raise ValueError("Focal lengths must be positive pixel values.")

    def resized(self, width: int, height: int) -> "CameraIntrinsics":
        """Return intrinsics transformed with pixel-center resize semantics."""
        scale_x = width / self.width
        scale_y = height / self.height
        return CameraIntrinsics(
            width=width,
            height=height,
            fx=self.fx * scale_x,
            fy=self.fy * scale_y,
            cx=(self.cx + 0.5) * scale_x - 0.5,
            cy=(self.cy + 0.5) * scale_y - 0.5,
        )


@dataclass
class EventStream:
    """Canonical event stream: seconds, integer sensor coordinates, polarity +/-1."""

    x: np.ndarray
    y: np.ndarray
    t: np.ndarray
    p: np.ndarray
    extras: dict[str, np.ndarray] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.x = np.asarray(self.x, dtype=np.int32).reshape(-1)
        self.y = np.asarray(self.y, dtype=np.int32).reshape(-1)
        self.t = np.asarray(self.t, dtype=np.float64).reshape(-1)
        self.p = np.asarray(self.p, dtype=np.int8).reshape(-1)
        count = len(self.x)
        if any(len(values) != count for values in (self.y, self.t, self.p)):
            raise ValueError("Event x, y, t, and p arrays must have the same length.")
        if not np.all(np.isin(self.p, (-1, 1))):
            raise ValueError("Event polarity must be -1 or +1.")
        self.extras = {key: np.asarray(value).reshape(-1) for key, value in self.extras.items()}
        if any(len(values) != count for values in self.extras.values()):
            raise ValueError("Every event extra field must match the event count.")

    @classmethod
    def empty(cls) -> "EventStream":
        return cls(np.empty(0, np.int32), np.empty(0, np.int32), np.empty(0, np.float64), np.empty(0, np.int8))

    def __len__(self) -> int:
        return len(self.t)

    def sorted_by_time(self) -> "EventStream":
        order = np.argsort(self.t, kind="stable")
        return EventStream(self.x[order], self.y[order], self.t[order], self.p[order], {key: value[order] for key, value in self.extras.items()})

    def with_extras(self, **extras: np.ndarray) -> "EventStream":
        return EventStream(self.x, self.y, self.t, self.p, {**self.extras, **extras})

    def metadata(self) -> dict[str, Any]:
        return {"event_count": len(self), "timestamp_unit": "seconds", "polarity_values": [-1, 1]}


@dataclass
class ImuStream:
    """Timestamped IMU samples in SI units.

    ``gyro`` is angular velocity in rad/s and ``accel`` is specific force in
    m/s^2. Timestamps are always seconds in the canonical API.
    """

    t: np.ndarray
    accel: np.ndarray
    gyro: np.ndarray

    def __post_init__(self) -> None:
        self.t = np.asarray(self.t, dtype=np.float64).reshape(-1)
        self.accel = np.asarray(self.accel, dtype=np.float64)
        self.gyro = np.asarray(self.gyro, dtype=np.float64)
        if self.accel.shape != (len(self.t), 3) or self.gyro.shape != (len(self.t), 3):
            raise ValueError("IMU accel and gyro must have shape [N, 3].")
        if len(self.t) < 2 or not np.all(np.isfinite(self.t)) or not np.all(np.diff(self.t) > 0.0):
            raise ValueError("IMU timestamps must contain two or more finite, strictly increasing values.")
        if not np.all(np.isfinite(self.accel)) or not np.all(np.isfinite(self.gyro)):
            raise ValueError("IMU samples must be finite.")

    def slice(self, start: float | None = None, end: float | None = None) -> "ImuStream":
        """Return samples in the inclusive time interval ``[start, end]``."""
        mask = np.ones(len(self.t), dtype=bool)
        if start is not None:
            mask &= self.t >= float(start)
        if end is not None:
            mask &= self.t <= float(end)
        if np.count_nonzero(mask) < 2:
            raise ValueError("The selected IMU interval must contain at least two samples.")
        return ImuStream(self.t[mask], self.accel[mask], self.gyro[mask])
