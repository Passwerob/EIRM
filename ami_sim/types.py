"""Small immutable-ish data containers shared across simulator layers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass(frozen=True)
class CameraIntrinsics:
    """Pixel-coordinate pinhole intrinsics for the RGB camera being simulated."""

    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float

    def resized(self, width: int, height: int) -> "CameraIntrinsics":
        """Scale intrinsics with an image resize, preserving pixel-center geometry."""
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
    """Canonical event representation used by every backend and downstream stage."""

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
        size = len(self.x)
        if any(len(value) != size for value in (self.y, self.t, self.p)):
            raise ValueError("Event x, y, t, and p arrays must have equal lengths.")
        if not np.all(np.isin(self.p, (-1, 1))):
            raise ValueError("Event polarity must use the documented {-1, +1} representation.")
        self.extras = {
            key: np.asarray(value).reshape(-1) for key, value in self.extras.items()
        }
        if any(len(value) != size for value in self.extras.values()):
            raise ValueError("Every event extra field must have one value per event.")

    @classmethod
    def empty(cls) -> "EventStream":
        return cls(
            x=np.empty(0, dtype=np.int32),
            y=np.empty(0, dtype=np.int32),
            t=np.empty(0, dtype=np.float64),
            p=np.empty(0, dtype=np.int8),
        )

    def __len__(self) -> int:
        return len(self.t)

    def masked(self, mask: np.ndarray) -> "EventStream":
        mask = np.asarray(mask, dtype=bool).reshape(-1)
        if len(mask) != len(self):
            raise ValueError("Event mask length does not match the event stream.")
        return EventStream(
            self.x[mask], self.y[mask], self.t[mask], self.p[mask],
            {key: value[mask] for key, value in self.extras.items()},
        )

    def sorted_by_time(self) -> "EventStream":
        order = np.argsort(self.t, kind="stable")
        return EventStream(
            self.x[order], self.y[order], self.t[order], self.p[order],
            {key: value[order] for key, value in self.extras.items()},
        )

    def with_extras(self, **extras: np.ndarray) -> "EventStream":
        return EventStream(self.x, self.y, self.t, self.p, {**self.extras, **extras})

    def metadata(self) -> dict[str, Any]:
        return {"count": len(self), "time_unit": "seconds", "polarity_values": [-1, 1]}
