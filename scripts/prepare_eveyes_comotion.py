#!/usr/bin/env python3
"""Prepare the EVEYES co-motion archive for the existing AMI sensing pipeline.

This adapter does not create masks, depth, or events.  It only normalizes the
timestamp column, exposes the RGB directory under the historical ``images``
name, and carries the calibrated camera intrinsics used by the EVEYES package.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
from pathlib import Path


def _read_timestamps(path: Path) -> list[tuple[int, float]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows or "timestamp_s" not in rows[0]:
        raise ValueError(f"{path} must contain a timestamp_s column.")
    return [(int(row.get("frame_id", index)), float(row["timestamp_s"])) for index, row in enumerate(rows)]


def _link_or_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() or destination.is_symlink():
        return
    try:
        destination.symlink_to(source.resolve(), target_is_directory=source.is_dir())
    except OSError:
        if source.is_dir():
            shutil.copytree(source, destination)
        else:
            shutil.copy2(source, destination)


def _prepare_scene(source: Path, destination: Path, split: str, intrinsics: dict[str, float]) -> dict[str, object]:
    destination.mkdir(parents=True, exist_ok=True)
    _link_or_copy(source / "rgb", destination / "images")
    for name in ("imu.csv", "camera_pose.csv", "ego_pose.csv", "neighbor_pose.csv", "motion_config.json", "render_config.json", "validation_report.md", "DONE.json"):
        if (source / name).exists():
            _link_or_copy(source / name, destination / name)
    timestamps = _read_timestamps(source / "timestamps.csv")
    with (destination / "timestamps.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("index", "timestamp_seconds"))
        writer.writerows(timestamps)
    (destination / "camera_intrinsics.json").write_text(json.dumps(intrinsics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    metadata = json.loads((source / "motion_config.json").read_text(encoding="utf-8"))
    return {
        "scene": f"{split}/{source.parent.name}",
        "source": str(source),
        "split": split,
        "scene_name": source.parent.name,
        "condition": "co_motion",
        "frame_count": len(timestamps),
        "resolution": {"width": int(intrinsics["width"]), "height": int(intrinsics["height"])},
        "motion_regime": metadata.get("motion_regime"),
        "speed_profile": metadata.get("speed_profile"),
        "timestamp_source": "timestamp_s",
        "intrinsics_source": "existing EVEYES calibrated 960x540 camera model",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=Path("data/eveyes_co_motion_final"))
    parser.add_argument("--output", type=Path, default=Path("data/eveyes_comotion_ready"))
    parser.add_argument("--intrinsics-source", type=Path, default=Path("data/eveyes_full/train/Scene_001/cam_moving_obj_moving/camera_intrinsics.json"))
    args = parser.parse_args()
    intrinsics = json.loads(args.intrinsics_source.read_text(encoding="utf-8"))
    # Keep only the fields consumed by CameraIntrinsics and record the source.
    intrinsics = {key: intrinsics[key] for key in ("width", "height", "fx", "fy", "cx", "cy")}
    mapping = (("train", "Scene_001"), ("valid", "Scene_002"), ("test", "Scene_003"))
    records = []
    for split, scene_name in mapping:
        source = args.source_root / ("test" if scene_name == "Scene_003" else "train") / scene_name / "co_motion"
        if not (source / "rgb").is_dir():
            raise FileNotFoundError(f"Missing EVEYES RGB directory: {source / 'rgb'}")
        records.append(_prepare_scene(source, args.output / split / scene_name, split, intrinsics))
    manifest = {
        "dataset": "EVEYES-co-motion-final-adapted",
        "source_archive_root": str(args.source_root),
        "purpose": "active event sensing and optical modulation control; not reconstruction",
        "intrinsics": intrinsics,
        "splits": {"train": ["Scene_001"], "valid": ["Scene_002"], "test": ["Scene_003"]},
        "records": records,
        "no_masks_or_depth_required": True,
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "records": records}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
