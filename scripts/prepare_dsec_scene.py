#!/usr/bin/env python3
"""Prepare a compact, indexed DSEC scene package for experiments.

The output keeps only selected left RGB frames and event windows. It never
fabricates IMU values: the manifest records that IMU must be extracted from
DSEC's official lidar_imu ROS archive separately.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import imageio.v3 as imageio
import numpy as np

from ami_simulator.dsec import read_dsec_events_h5
from ami_simulator.events import write_events_h5


def main() -> int:
    parser = argparse.ArgumentParser(description="Create a compact DSEC scene package.")
    parser.add_argument("--scene-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-frames", type=int, default=32)
    parser.add_argument("--event-window-ms", type=float, default=50.0)
    args = parser.parse_args()
    if args.max_frames < 2 or args.event_window_ms <= 0:
        parser.error("max-frames must be >= 2 and event-window-ms must be positive")
    root, output = args.scene_root, args.output
    timestamps_path = root / next(path.name for path in root.glob("*_image_timestamps.txt"))
    timestamps_us = np.loadtxt(timestamps_path, dtype=np.float64).reshape(-1)
    image_paths = sorted(root.glob("*.png"))
    count = min(args.max_frames, len(image_paths), len(timestamps_us))
    if count < 2:
        raise ValueError("Scene must contain at least two timestamped images.")
    selected_images = image_paths[:count]
    selected_times = timestamps_us[:count] * 1e-6
    output.mkdir(parents=True, exist_ok=True)
    image_dir = output / "images"
    image_dir.mkdir(exist_ok=True)
    for index, source in enumerate(selected_images):
        imageio.imwrite(image_dir / f"{index:06d}.png", imageio.imread(source))
    event_path = root / "events.h5"
    event_dir = output / "events"
    event_dir.mkdir(exist_ok=True)
    frame_records = []
    for index, timestamp in enumerate(selected_times):
        start = float(timestamp - args.event_window_ms * 1e-3)
        end = float(timestamp)
        events = read_dsec_events_h5(event_path, start_time=start, end_time=end)
        event_file = event_dir / f"{index:06d}.h5"
        write_events_h5(event_file, events, {"source": "DSEC", "window_start_seconds": start, "window_end_seconds": end})
        frame_records.append({"index": index, "image": f"images/{index:06d}.png", "timestamp_seconds": float(timestamp), "event_file": f"events/{index:06d}.h5", "event_window_start_seconds": start, "event_window_end_seconds": end, "event_count": len(events)})
    for pattern in ("*_calibration.zip", "cam_to_cam.yaml", "cam_to_lidar.yaml", "*_image_timestamps.txt", "*_image_exposure_timestamps_left.txt"):
        for source in root.glob(pattern):
            destination = output / source.name
            if source.is_file() and not destination.exists():
                shutil.copy2(source, destination)
    manifest = {"dataset": "DSEC", "scene": root.name, "resolution": {"width": 640, "height": 480}, "frame_count": count, "event_window_ms": args.event_window_ms, "imu": {"status": "not_extracted", "source": "official lidar_imu.zip ROS archive", "expected_csv_columns": ["timestamp_us", "ax", "ay", "az", "gx", "gy", "gz"]}, "frames": frame_records}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"wrote {count} frames and {sum(record['event_count'] for record in frame_records)} windowed events to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
