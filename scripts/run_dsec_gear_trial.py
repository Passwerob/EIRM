#!/usr/bin/env python3
"""Run fixed-AMI and IMU gear-modulated trials on a compact DSEC scene."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ami_simulator.dsec import read_imu_csv
from ami_simulator.pipeline import SimulationConfig, load_rgb_sequence, run_simulation
from ami_simulator.rotation import ConstantRotation, ImuGearRotation, motion_adaptive_hysteresis, motion_quantile_thresholds
from ami_simulator.types import CameraIntrinsics


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-frames", type=int, default=8)
    parser.add_argument("--base-rpm", type=float, default=720.0)
    parser.add_argument("--motion-axis", type=int, choices=[0, 1, 2], default=None)
    parser.add_argument("--speed-thresholds", type=float, nargs=4, default=(0.025, 0.032, 0.040, 0.048), metavar=("T0", "T1", "T2", "T3"), help="Four DSEC-calibrated motion thresholds in rad/s.")
    parser.add_argument("--adaptive-thresholds", action="store_true", help="Use the IMU 20/40/60/80%% motion quantiles for this scene.")
    parser.add_argument("--event-backend", choices=["threshold_cpu", "threshold_cuda", "threshold_cuda_compiled"], default="threshold_cpu")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--contrast-threshold", type=float, default=0.2)
    args = parser.parse_args()

    scene = args.scene_root
    timestamp_path = next(scene.glob("*_image_timestamps.txt"))
    frames, timestamps = load_rgb_sequence(
        scene / "images",
        timestamp_path,
        timestamp_unit="us",
        max_frames=args.max_frames,
    )
    imu = read_imu_csv(scene / "imu.csv", timestamp_unit="us", gyro_unit="rad/s")
    speed_thresholds = motion_quantile_thresholds(imu) if args.adaptive_thresholds else tuple(args.speed_thresholds)
    hysteresis = motion_adaptive_hysteresis(imu) if args.adaptive_thresholds else 0.01
    rotation = ImuGearRotation(
        imu,
        base_rpm=args.base_rpm,
        motion_axis=args.motion_axis,
        speed_thresholds_rad_s=speed_thresholds,
        hysteresis_rad_s=hysteresis,
    )
    height, width = frames.shape[1:3]
    # DSEC rectified frame-camera intrinsics, matching the RGB images copied by
    # prepare_dsec_scene.py.  The prism simulator operates on this RGB source.
    intrinsics = CameraIntrinsics(width, height, 1164.6238115833075, 1164.6238115833075, 713.5791168212891, 570.9349365234375)
    config = SimulationConfig(
        rpm=args.base_rpm,
        contrast_threshold=args.contrast_threshold,
        event_backend=args.event_backend,
        device=args.device,
    )
    fixed_output = args.output / "fixed_ami"
    gear_output = args.output / "imu_gear"
    fixed_summary = run_simulation(
        frames,
        timestamps,
        intrinsics,
        fixed_output,
        config=config,
        rotation_policy=ConstantRotation(args.base_rpm),
        source_metadata={"dataset": "DSEC", "scene": scene.name, "controller": "fixed_ami"},
    )
    gear_summary = run_simulation(
        frames,
        timestamps,
        intrinsics,
        gear_output,
        config=config,
        rotation_policy=rotation,
        source_metadata={"dataset": "DSEC", "scene": scene.name, "controller": rotation.metadata()},
    )
    print(json.dumps({
        "output": str(args.output),
        "fixed_ami": {"raw_event_count": fixed_summary["raw_event_count"], "compensated_event_count": fixed_summary["compensated_event_count"]},
        "imu_gear": {"raw_event_count": gear_summary["raw_event_count"], "compensated_event_count": gear_summary["compensated_event_count"], "rotation": gear_summary["rotation"]},
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
