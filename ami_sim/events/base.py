"""Event backend interface and canonical HDF5 serialization."""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from ami_sim.types import EventStream


class EventSimulatorBackend(ABC):
    name = "base"
    supports_torch = False

    @abstractmethod
    def simulate(self, frames: np.ndarray, timestamps: np.ndarray, config: dict[str, Any] | None = None) -> EventStream:
        """Simulate canonical events from [T,H,W] float32 intensity frames."""

    def metadata(self) -> dict[str, object]:
        return {"event_backend": self.name}


def _h5_node(handle: h5py.File) -> h5py.Group | h5py.File:
    return handle["events"] if "events" in handle else handle


def _normalise_polarity(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values).reshape(-1)
    if np.all(np.isin(values, (-1, 1))):
        return values.astype(np.int8)
    if np.all(np.isin(values, (0, 1))):
        return np.where(values > 0, 1, -1).astype(np.int8)
    raise ValueError("External event polarity must use {-1,+1} or documented {0,1} values.")


def normalise_events(
    x: np.ndarray,
    y: np.ndarray,
    t: np.ndarray,
    p: np.ndarray,
    *,
    time_unit: str = "s",
) -> EventStream:
    scale = {"s": 1.0, "ms": 1e-3, "us": 1e-6, "ns": 1e-9}.get(time_unit)
    if scale is None:
        raise ValueError(f"Unsupported event timestamp unit {time_unit!r}.")
    result = EventStream(np.asarray(x), np.asarray(y), np.asarray(t, dtype=np.float64) * scale, _normalise_polarity(p))
    return result.sorted_by_time()


def write_events_h5(path: str | Path, events: EventStream, metadata: dict[str, Any] | None = None) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as handle:
        group = handle.create_group("events")
        group.create_dataset("x", data=events.x, compression="gzip")
        group.create_dataset("y", data=events.y, compression="gzip")
        group.create_dataset("t", data=events.t, compression="gzip")
        group.create_dataset("p", data=events.p, compression="gzip")
        for key, value in events.extras.items():
            group.create_dataset(key, data=value, compression="gzip")
        attrs = {**events.metadata(), **(metadata or {})}
        for key, value in attrs.items():
            if isinstance(value, (dict, list, tuple)):
                value = json.dumps(value, sort_keys=True)
            group.attrs[key] = value


def read_events_h5(path: str | Path, *, time_unit: str = "s") -> EventStream:
    with h5py.File(path, "r") as handle:
        group = _h5_node(handle)
        required = ["x", "y", "t", "p"]
        if any(name not in group for name in required):
            raise ValueError(f"HDF5 event file {path} must contain x, y, t, and p.")
        raw_t = np.asarray(group["t"], dtype=np.float64)
        order = np.argsort(raw_t, kind="stable")
        base = normalise_events(group["x"], group["y"], raw_t, group["p"], time_unit=time_unit)
        extras = {key: np.asarray(group[key])[order] for key in group.keys() if key not in required}
        return base.with_extras(**extras)
