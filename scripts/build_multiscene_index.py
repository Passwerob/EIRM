#!/usr/bin/env python3
"""Build a compact JSONL index of DSEC event windows and optional IMU stats."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import h5py

from ami_simulator.dsec import read_dsec_events_h5, read_imu_csv
from ami_simulator.events import read_events_h5


def summarize_events(path: Path, start: float, end: float, width: int, height: int) -> dict[str, float | int | None]:
    # Prepared windows use the simulator's seconds convention; raw DSEC
    # events.h5 uses integer microseconds plus t_offset.
    with h5py.File(path, "r") as handle:
        is_prepared = "t_offset" not in handle and "events" in handle and "t" in handle["events"] and handle["events"]["t"].dtype.kind == "f"
    events = (read_events_h5(path) if is_prepared else read_dsec_events_h5(path, start_time=start, end_time=end))
    count = len(events)
    if count == 0:
        return {"event_count": 0, "event_rate_hz": 0.0, "positive_fraction": None, "x_mean": None, "y_mean": None, "x_std": None, "y_std": None, "xy_cov": None, "time_span_ms": 0.0}
    x = events.x.astype(np.float64)
    y = events.y.astype(np.float64)
    centered = np.column_stack((x - x.mean(), y - y.mean()))
    covariance = centered.T @ centered / count
    return {
        "event_count": count,
        "event_rate_hz": float(count / max(end - start, 1e-9)),
        "positive_fraction": float(np.mean(events.p > 0)),
        "x_mean": float(x.mean() / max(width - 1, 1)),
        "y_mean": float(y.mean() / max(height - 1, 1)),
        "x_std": float(x.std() / max(width - 1, 1)),
        "y_std": float(y.std() / max(height - 1, 1)),
        "xy_cov": float(covariance[0, 1] / max((width - 1) * (height - 1), 1)),
        "time_span_ms": float((events.t[-1] - events.t[0]) * 1e3),
    }


def imu_summary(path: Path | None, start: float, end: float) -> dict[str, float | int | bool]:
    if path is None or not path.exists():
        return {"available": False, "sample_count": 0, "gyro_norm_mean": 0.0, "gyro_norm_max": 0.0}
    imu = read_imu_csv(path, timestamp_unit="us", gyro_unit="rad/s").slice(start, end)
    norms = np.linalg.norm(imu.gyro, axis=1)
    return {"available": True, "sample_count": len(imu.t), "gyro_norm_mean": float(norms.mean()), "gyro_norm_max": float(norms.max())}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, action="append", required=True, help="Prepared DSEC manifest; repeat for multiple scenes.")
    parser.add_argument("--imu-csv", type=Path, action="append", default=[], help="Optional IMU CSV in the same order as --manifest.")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if len(args.imu_csv) > len(args.manifest):
        parser.error("Provide at most one --imu-csv per --manifest.")
    rows: list[dict[str, object]] = []
    for scene_index, manifest_path in enumerate(args.manifest):
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        width, height = int(manifest["resolution"]["width"]), int(manifest["resolution"]["height"])
        imu_path = args.imu_csv[scene_index] if scene_index < len(args.imu_csv) else None
        for frame in manifest["frames"]:
            event_path = manifest_path.parent / frame["event_file"]
            start = float(frame["event_window_start_seconds"])
            end = float(frame["event_window_end_seconds"])
            rows.append({"scene": manifest["scene"], "scene_index": scene_index, "frame_index": int(frame["index"]), "timestamp_seconds": float(frame["timestamp_seconds"]), "event_file": str(event_path), "imu": imu_summary(imu_path, start, end), "events": summarize_events(event_path, start, end, width, height)})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    print(f"wrote {len(rows)} window records to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
