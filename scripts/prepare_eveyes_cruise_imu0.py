#!/usr/bin/env python3
"""Stage eveyes_full CRUISE clips (gyro identically 0) into a ready layout.

These are 50 mm / fx=960 sequences, not the v2 16 mm pack. They exist so the
gear policy can see high visual motion with IMU=0. Do not treat them as the
three-regime factorial.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

from scripts.prepare_eveyes_v2_comotion import _link

PAIRS = (
    ("train", "Scene_001", "cam_moving_obj_static", "camera_only"),
    ("train", "Scene_001", "cam_static_obj_moving", "object_only"),
    ("valid", "Scene_004", "cam_moving_obj_static", "camera_only"),
    ("valid", "Scene_004", "cam_static_obj_moving", "object_only"),
)


def _prepare(source: Path, destination: Path) -> dict[str, object]:
    if not (source / "rgb").is_dir():
        raise SystemExit(f"missing RGB {source / 'rgb'}")
    destination.mkdir(parents=True, exist_ok=True)
    _link(source / "rgb", destination / "images")
    for name in ("imu.csv", "camera_pose.csv", "motion_config.json", "render_config.json", "object_masks"):
        if (source / name).exists():
            _link(source / name, destination / name)
    intrinsics = json.loads((source / "camera_intrinsics.json").read_text(encoding="utf-8"))
    keep = {key: intrinsics[key] for key in ("width", "height", "fx", "fy", "cx", "cy")}
    (destination / "camera_intrinsics.json").write_text(json.dumps(keep, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with (source / "timestamps.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    with (destination / "timestamps.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("index", "timestamp_seconds"))
        writer.writerows((int(row["frame_id"]), float(row["timestamp_s"])) for row in rows)
    config = json.loads((source / "motion_config.json").read_text(encoding="utf-8"))
    return {
        "source": str(source),
        "ready": str(destination),
        "speed_profile": config.get("speed_profile"),
        "motion_regime": config.get("motion_regime"),
        "frames": len(rows),
        "fx": keep["fx"],
    }


def main() -> int:
    source_root = Path("data/eveyes_full")
    output = Path("data/eveyes_cruise_imu0_ready")
    records = []
    for split, scene, folder, regime in PAIRS:
        packed_split = "train" if scene != "Scene_005" else "test"
        source = source_root / packed_split / scene / folder
        records.append({"split": split, "scene": f"{split}/{scene}/{regime}", **_prepare(source, output / split / scene / regime)})
    manifest = {
        "dataset": "EVEYES-full-CRUISE-imu0",
        "not_the_v2_16mm_factorial": True,
        "gyro": "identically zero; constant-velocity analytic IMU",
        "records": records,
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
