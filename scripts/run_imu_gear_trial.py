#!/usr/bin/env python3
"""Run fixed and IMU-conditioned prism trials on a generic RGB/IMU scene."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ami_simulator.dsec import read_imu_csv
from ami_simulator.pipeline import SimulationConfig, load_rgb_sequence, run_simulation
from ami_simulator.rotation import ConstantRotation, ImuGearRotation
from ami_simulator.types import CameraIntrinsics


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-frames", type=int, default=32)
    parser.add_argument("--timestamp-file", type=Path, default=None)
    parser.add_argument("--timestamp-unit", choices=["s", "ms", "us", "ns"], default="us", help="Unit of the selected timestamp column; Eveyes timestamps.csv ends with timestamp_us.")
    parser.add_argument("--imu-csv", type=Path, default=None)
    parser.add_argument("--imu-timestamp-unit", choices=["s", "ms", "us", "ns"], default="s")
    parser.add_argument("--base-rpm", type=float, default=720.0)
    parser.add_argument("--contrast-threshold", type=float, default=0.2)
    args = parser.parse_args()
    scene = args.scene_root
    timestamp_file = args.timestamp_file or scene / "timestamps.csv"
    imu_path = args.imu_csv or scene / "imu.csv"
    frames, timestamps = load_rgb_sequence(scene / "rgb", timestamp_file, timestamp_unit=args.timestamp_unit, max_frames=args.max_frames)
    intrinsics_data = json.loads((scene / "camera_intrinsics.json").read_text(encoding="utf-8"))
    height, width = frames.shape[1:3]
    intrinsics = CameraIntrinsics(width, height, float(intrinsics_data["fx"]), float(intrinsics_data["fy"]), float(intrinsics_data["cx"]), float(intrinsics_data["cy"]))
    imu = read_imu_csv(imu_path, timestamp_unit=args.imu_timestamp_unit, gyro_unit="rad/s")
    rotation = ImuGearRotation(imu, base_rpm=args.base_rpm)
    config = SimulationConfig(rpm=args.base_rpm, contrast_threshold=args.contrast_threshold)
    fixed = run_simulation(frames, timestamps, intrinsics, args.output / "fixed_ami", config=config, rotation_policy=ConstantRotation(args.base_rpm), source_metadata={"dataset": "Eveyes", "scene": scene.name, "controller": "fixed_ami"})
    gear = run_simulation(frames, timestamps, intrinsics, args.output / "imu_gear", config=config, rotation_policy=rotation, source_metadata={"dataset": "Eveyes", "scene": scene.name, "controller": rotation.metadata()})
    print(json.dumps({"output": str(args.output), "fixed_raw_events": fixed["raw_event_count"], "gear_raw_events": gear["raw_event_count"], "gear_metadata": rotation.metadata()}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
