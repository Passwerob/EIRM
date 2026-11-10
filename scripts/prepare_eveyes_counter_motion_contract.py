#!/usr/bin/env python3
"""Freeze Scene_005 counter-motion poses from the existing co-motion dump.

RGB still has to be Blender-rendered. This writes the numeric contract the
renderer must match: camera/ego copy co-motion, neighbor keeps the co-motion
lock plus the object_only along-track sine, direction stays +1.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
from pathlib import Path


AMPLITUDE_M = 2.5
RGB_COUNT = 480


def _read(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _shift_along(row: dict[str, str], axis: str, delta: float) -> dict[str, str]:
    out = dict(row)
    key = "position_x" if axis == "x" else "position_y"
    out[key] = f"{float(row[key]) + delta:.17g}"
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=Path("data/eveyes_co_motion_v2"))
    parser.add_argument("--split-dir", default="test")
    parser.add_argument("--scene", default="Scene_005")
    args = parser.parse_args()
    source = args.source_root / args.split_dir / args.scene / "co_motion"
    object_only = args.source_root / args.split_dir / args.scene / "object_only"
    destination = args.source_root / args.split_dir / args.scene / "counter_motion"
    if not (source / "neighbor_pose.csv").is_file():
        raise SystemExit(f"missing co-motion dump {source}")

    axis = json.loads((source / "motion_config.json").read_text(encoding="utf-8"))["road_coordinate_system"]["axis"]
    axis = "y" if axis == "world_y" else "x"
    neighbor = _read(source / "neighbor_pose.csv")
    if len(neighbor) != RGB_COUNT:
        raise SystemExit(f"expected {RGB_COUNT} neighbor poses, got {len(neighbor)}")
    shifted = []
    for index, row in enumerate(neighbor):
        sine = AMPLITUDE_M * math.sin(2.0 * math.pi * index / RGB_COUNT)
        shifted.append(_shift_along(row, axis, sine))

    destination.mkdir(parents=True, exist_ok=True)
    for name in ("camera_pose.csv", "ego_pose.csv", "timestamps.csv", "motion_state.csv", "imu.csv"):
        shutil.copy2(source / name, destination / name)
    _write(destination / "neighbor_pose.csv", shifted)
    for name in ("camera_intrinsics.json", "camera_extrinsics.json", "render_config.json"):
        shutil.copy2(source / name, destination / name)

    if (object_only / "neighbor_pose.csv").is_file():
        object_rows = _read(object_only / "neighbor_pose.csv")
        object_key = "position_x" if axis == "x" else "position_y"
        object_center = float(object_rows[0][object_key])
        max_sine_error = max(
            abs((float(row[object_key]) - object_center) - AMPLITUDE_M * math.sin(2.0 * math.pi * index / RGB_COUNT))
            for index, row in enumerate(object_rows)
        )
    else:
        max_sine_error = None

    rel = []
    ego = _read(destination / "ego_pose.csv")
    along_key = "position_x" if axis == "x" else "position_y"
    for neighbor_row, ego_row in zip(shifted, ego):
        rel.append(float(neighbor_row[along_key]) - float(ego_row[along_key]))

    config = json.loads((source / "motion_config.json").read_text(encoding="utf-8"))
    config.update(
        {
            "sequence_id": f"{args.scene}_COUNTER_MOTION_V2",
            "motion_regime": "COUNTER_MOTION",
            "co_motion_invariant": "not applicable",
            "freeze_rule": "camera follows path; neighbor keeps co-motion lock plus object_only along-track sine; direction stays +1",
            "object_only_amplitude_m": AMPLITUDE_M,
            "neighbor_direction": 1,
            "rgb_status": "pending_blender",
            "relative_pose_validation": {
                "longitudinal_offset_m": 5.5,
                "max_longitudinal_offset_error_m": max(abs(value - 5.5) for value in rel),
                "relative_pose_constant": False,
            },
        }
    )
    (destination / "motion_config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    contract = {
        "status": "POSES_FROZEN_RGB_PENDING",
        "scene": f"{args.split_dir}/{args.scene}/counter_motion",
        "rule": "neighbor_along = co_motion_neighbor_along + 2.5*sin(2π i/480); camera/ego/IMU copied from co_motion; direction=+1",
        "not": [
            "oncoming traffic / neighbor direction=-1",
            "eveyes_full opposing cars",
            "64x64 procedural counter_motion",
            "drive_stop_drive relabeled as counter_motion",
        ],
        "relative_ahead_m": {"min": min(rel), "max": max(rel), "mean": sum(rel) / len(rel)},
        "object_only_sine_match_m": max_sine_error,
        "rgb": None,
        "object_masks": None,
    }
    (destination / "CONTRACT.json").write_text(json.dumps(contract, indent=2) + "\n", encoding="utf-8")
    (destination / "README.txt").write_text(
        "Counter-motion pose contract for native EVEYES v2.\n"
        "RGB and object_masks are NOT here yet. Render with\n"
        "  render_comotion_v2.py --motion-regime counter_motion\n"
        "Neighbor poses must match neighbor_pose.csv to 1e-4 m.\n"
        "Do not flip neighbor direction to -1.\n",
        encoding="utf-8",
    )
    print(json.dumps(contract, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
