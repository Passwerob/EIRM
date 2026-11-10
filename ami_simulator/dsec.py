"""Small, dependency-light adapters for DSEC event and IMU subsets.

DSEC event timestamps are stored as integer microseconds plus ``t_offset``;
this module converts them to the simulator's seconds convention. IMU CSV
loading is intentionally generic so a ROS bag can be exported once with any
ROS tool and then consumed without a ROS runtime.
"""

from __future__ import annotations

import csv
from pathlib import Path

import h5py
import numpy as np

# DSEC uses Blosc/Zstd-compressed HDF5 datasets. Importing hdf5plugin registers
# those filters with h5py before any event dataset is opened.
try:
    import hdf5plugin  # noqa: F401
except ImportError:  # Keep synthetic/non-DSEC use importable with base h5py.
    hdf5plugin = None

from .types import EventStream, ImuStream


def _column_index(names: list[str], candidates: tuple[str, ...]) -> int | None:
    normalized = {name.strip().lower().replace(".", "_"): index for index, name in enumerate(names)}
    for candidate in candidates:
        if candidate in normalized:
            return normalized[candidate]
    return None


def read_dsec_events_h5(path: str | Path, *, start_time: float | None = None, end_time: float | None = None) -> EventStream:
    """Read a DSEC ``events.h5`` file, optionally selecting a time interval."""
    with h5py.File(path, "r") as handle:
        group = handle["events"]
        offset_us = int(np.asarray(handle["t_offset"])) if "t_offset" in handle else int(group.attrs.get("t_offset", 0))
        event_count = len(group["t"])
        start_index, end_index = 0, event_count
        if start_time is not None or end_time is not None:
            # DSEC's millisecond index avoids loading the complete timestamp
            # array when extracting a short training window.
            ms_to_idx = np.asarray(handle["ms_to_idx"], dtype=np.int64) if "ms_to_idx" in handle else None
            if ms_to_idx is not None and len(ms_to_idx):
                relative_start_us = max(0.0, (float(start_time if start_time is not None else -np.inf) * 1e6) - offset_us)
                relative_end_us = max(0.0, (float(end_time if end_time is not None else np.inf) * 1e6) - offset_us)
                if np.isfinite(relative_start_us):
                    start_ms = min(len(ms_to_idx) - 1, int(relative_start_us // 1000.0))
                    start_index = int(ms_to_idx[start_ms])
                if np.isfinite(relative_end_us):
                    end_ms = min(len(ms_to_idx) - 1, int(np.ceil(relative_end_us / 1000.0)) + 1)
                    end_index = int(ms_to_idx[end_ms]) if end_ms < len(ms_to_idx) else event_count
        x = np.asarray(group["x"][start_index:end_index], dtype=np.int32)
        y = np.asarray(group["y"][start_index:end_index], dtype=np.int32)
        t_us = np.asarray(group["t"][start_index:end_index], dtype=np.int64)
        polarity = np.asarray(group["p"][start_index:end_index])
    if polarity.size and np.all(np.isin(polarity, (0, 1))):
        polarity = polarity * 2 - 1
    polarity = polarity.astype(np.int8, copy=False)
    times = (t_us.astype(np.float64) + offset_us) * 1e-6
    mask = np.ones(len(times), dtype=bool)
    if start_time is not None:
        mask &= times >= float(start_time)
    if end_time is not None:
        mask &= times <= float(end_time)
    return EventStream(x[mask], y[mask], times[mask], polarity[mask])


def read_imu_csv(
    path: str | Path,
    *,
    timestamp_unit: str = "s",
    gyro_unit: str = "rad/s",
    accel_unit: str = "m/s^2",
) -> ImuStream:
    """Load a CSV with timestamp and 3-axis accel/gyro columns.

    Header names may be ``t``/``timestamp``, ``ax ay az`` and ``gx gy gz``
    (common DSEC exports use the same short names). Headerless seven-column
    files are accepted as ``t ax ay az gx gy gz``.
    """
    scales_time = {"s": 1.0, "ms": 1e-3, "us": 1e-6, "ns": 1e-9}
    scales_gyro = {"rad/s": 1.0, "deg/s": np.pi / 180.0, "dps": np.pi / 180.0}
    scales_accel = {"m/s^2": 1.0, "mps2": 1.0, "g": 9.80665}
    if timestamp_unit not in scales_time or gyro_unit not in scales_gyro or accel_unit not in scales_accel:
        raise ValueError("Unsupported IMU unit; use timestamp s/ms/us/ns, gyro rad/s or deg/s, accel m/s^2 or g.")
    rows = list(csv.reader(Path(path).read_text(encoding="utf-8").splitlines()))
    rows = [row for row in rows if any(field.strip() for field in row)]
    if not rows:
        raise ValueError(f"No IMU samples found in {path}.")
    try:
        numeric = np.asarray([[float(value) for value in row] for row in rows], dtype=np.float64)
        names: list[str] | None = None
    except ValueError:
        names = [value.strip() for value in rows.pop(0)]
        numeric = np.asarray([[float(value) for value in row] for row in rows], dtype=np.float64)
    if names is None:
        if numeric.shape[1] >= 7:
            t_index, accel_indices, gyro_indices = 0, (1, 2, 3), (4, 5, 6)
        else:
            raise ValueError("Headerless IMU CSV must contain t ax ay az gx gy gz.")
    else:
        t_index = _column_index(names, ("t", "time", "timestamp", "timestamp_s", "timestamp_us", "imu_timestamp", "time_us"))
        accel_indices = tuple(_column_index(names, (axis, f"a{axis}", f"accel_{axis}", f"accel_{axis}_m_s2", f"linear_acceleration_{axis}", f"linear_acceleration_{axis}_m_s2")) for axis in "xyz")
        gyro_indices = tuple(_column_index(names, (axis, f"g{axis}", f"gyro_{axis}", f"gyro_{axis}_rad_s", f"angular_velocity_{axis}", f"angular_velocity_{axis}_rad_s")) for axis in "xyz")
        if t_index is None or any(index is None for index in (*accel_indices, *gyro_indices)):
            raise ValueError("IMU CSV needs timestamp, ax/ay/az and gx/gy/gz columns.")
        accel_indices = tuple(int(index) for index in accel_indices)
        gyro_indices = tuple(int(index) for index in gyro_indices)
    return ImuStream(
        numeric[:, t_index] * scales_time[timestamp_unit],
        numeric[:, accel_indices] * scales_accel[accel_unit],
        numeric[:, gyro_indices] * scales_gyro[gyro_unit],
    )
