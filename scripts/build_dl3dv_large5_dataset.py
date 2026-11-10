#!/usr/bin/env python3
"""Build five scene-level DL3DV packages for GS2E/AMI experiments.

The local DL3DV export contains RGB, COLMAP/NeRF camera poses, and per-frame
DVS-Voltmeter tensors, but no measured IMU. This builder preserves those
artifacts, normalizes the camera calibration to the 960x540 images, creates an
explicit nominal timeline, and derives a pose-based IMU for controller smoke
tests. The derived IMU is labelled synthetic and must not be treated as sensor
ground truth.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shutil
from pathlib import Path

import numpy as np


DEFAULT_SCENES = ("screen-000083", "screen-000087", "screen-000089", "screen-000095", "screen-000108")


def _rotation_log(matrix: np.ndarray) -> np.ndarray:
    trace = float(np.trace(matrix))
    cosine = np.clip((trace - 1.0) * 0.5, -1.0, 1.0)
    angle = math.acos(cosine)
    if angle < 1e-8:
        return np.zeros(3, dtype=np.float64)
    scale = angle / (2.0 * math.sin(angle))
    return scale * np.asarray([matrix[2, 1] - matrix[1, 2], matrix[0, 2] - matrix[2, 0], matrix[1, 0] - matrix[0, 1]], dtype=np.float64)


def _link_or_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def _build_scene(source: Path, destination: Path, fps: float, include_reference_events: bool = False) -> dict[str, object]:
    meta = json.loads((source / "meta.json").read_text(encoding="utf-8"))
    source_transforms = json.loads((source / "transforms.json").read_text(encoding="utf-8"))
    image_paths = sorted((source / "images").glob("*.png"))
    event_paths = sorted((source / "events").glob("*.pt"))
    if len(image_paths) != len(source_transforms["frames"]):
        raise ValueError(f"{source}: RGB and transforms counts must match")
    if len(event_paths) != len(image_paths):
        raise ValueError(f"{source}: official reference event tensor count must match RGB count")
    destination.mkdir(parents=True, exist_ok=True)
    image_dir, event_dir = destination / "images", destination / "events"
    for index, image in enumerate(image_paths, start=1):
        _link_or_copy(image, image_dir / f"frame_{index:05d}.png")
    if include_reference_events:
        for index, event in enumerate(event_paths):
            _link_or_copy(event, event_dir / f"{index:05d}.pt")

    width, height = 960, 540
    sx, sy = width / float(source_transforms["w"]), height / float(source_transforms["h"])
    transforms = dict(source_transforms)
    transforms.update({"w": width, "h": height, "fl_x": source_transforms["fl_x"] * sx, "fl_y": source_transforms["fl_y"] * sy, "cx": (source_transforms["cx"] + 0.5) * sx - 0.5, "cy": (source_transforms["cy"] + 0.5) * sy - 0.5})
    transforms["frames"] = [{"file_path": f"images/frame_{index:05d}.png", "transform_matrix": frame["transform_matrix"], "colmap_im_id": frame.get("colmap_im_id")} for index, frame in enumerate(source_transforms["frames"], start=1)]
    (destination / "transforms.json").write_text(json.dumps(transforms, indent=2), encoding="utf-8")
    shutil.copy2(source / "transforms.json", destination / "transforms_source_4k.json")

    timestamps = np.arange(len(image_paths), dtype=np.float64) / fps
    poses = np.asarray([frame["transform_matrix"] for frame in source_transforms["frames"]], dtype=np.float64)
    imu_rows = []
    for index, timestamp in enumerate(timestamps):
        if index == 0:
            velocity = np.zeros(3, dtype=np.float64)
            omega = np.zeros(3, dtype=np.float64)
            acceleration = np.zeros(3, dtype=np.float64)
        else:
            dt = timestamps[index] - timestamps[index - 1]
            velocity = (poses[index, :3, 3] - poses[index - 1, :3, 3]) / dt
            omega = _rotation_log(poses[index - 1, :3, :3].T @ poses[index, :3, :3]) / dt
            if index == 1:
                acceleration = np.zeros(3, dtype=np.float64)
            else:
                previous_velocity = (poses[index - 1, :3, 3] - poses[index - 2, :3, 3]) / (timestamps[index - 1] - timestamps[index - 2])
                acceleration = (velocity - previous_velocity) / dt
        imu_rows.append([timestamp * 1e6, *acceleration, *omega])
    with (destination / "timestamps.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["index", "timestamp_seconds", "timeline"])
        writer.writerows((index, f"{timestamp:.17g}", "nominal_from_explicit_fps") for index, timestamp in enumerate(timestamps))
    with (destination / "imu.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["timestamp_us", "ax", "ay", "az", "gx", "gy", "gz"])
        writer.writerows(([f"{value:.17g}" for value in row] for row in imu_rows))
    with (destination / "camera_pose.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["index", "timestamp_seconds", *[f"T_{row}{column}" for row in range(4) for column in range(4)]])
        writer.writerows(([index, f"{timestamps[index]:.17g}", *poses[index].reshape(-1).tolist()] for index in range(len(poses))))

    scene_metadata = {
        "scene_id": destination.name,
        "source_export": str(source),
        "source_scene": meta.get("scene"),
        "source_clip": meta.get("clip"),
        "frame_count": len(image_paths),
        "resolution": {"width": width, "height": height},
        "fps_assumption": fps,
        "timeline": "nominal_from_explicit_fps",
        "camera_calibration": {"source_resolution": [source_transforms["w"], source_transforms["h"]], "normalized_resolution": [width, height], "fl_x": transforms["fl_x"], "fl_y": transforms["fl_y"], "cx": transforms["cx"], "cy": transforms["cy"]},
        "imu": {"type": "pose_derived_synthetic", "source": "transforms.json", "timestamp_unit": "us", "gyro_unit": "rad/s", "accel_unit": "m/s^2", "not_sensor_ground_truth": True},
        "official_reference_events": {"format": "DVS-Voltmeter per-frame tensor", "source_directory": str(source / "events"), "tensor_shape": [8, 540, 960], "dtype": "torch.float32", "included_in_package": include_reference_events, "owned_by_this_builder": False},
        "gs2e_contract": "RGB + calibrated poses + nominal timeline + pose-derived IMU; official preprocessed event tensors remain a separate reference source; no Blender scene is included",
    }
    (destination / "scene.json").write_text(json.dumps(scene_metadata, indent=2), encoding="utf-8")
    return scene_metadata


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=Path("/data/fcr/data/eval_data/DL3DV"))
    parser.add_argument("--output", type=Path, default=Path("data/dl3dv_large5"))
    parser.add_argument("--scenes", nargs=5, default=DEFAULT_SCENES)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--include-reference-events", action="store_true", help="Also hard-link the official DVS-Voltmeter tensors under events/; disabled by default to keep them separate from AMI/GS2E outputs.")
    args = parser.parse_args()
    if args.fps <= 0:
        raise SystemExit("fps must be positive")
    args.output.mkdir(parents=True, exist_ok=True)
    records = []
    for scene_id in args.scenes:
        source = args.source_root / scene_id
        if not source.is_dir():
            raise SystemExit(f"Missing DL3DV scene directory: {source}")
        records.append(_build_scene(source, args.output / scene_id, args.fps, args.include_reference_events))
    manifest = {"dataset": "DL3DV-large5", "scene_count": len(records), "fps_assumption": args.fps, "timeline_policy": "nominal_from_explicit_fps", "event_policy": "official DVS-Voltmeter tensors are external reference data, not AMI/GS2E outputs", "scenes": records}
    (args.output / "dataset_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(args.output), "scene_count": len(records), "frames_per_scene": [record["frame_count"] for record in records]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
