#!/usr/bin/env python3
"""Normalize EVEYES co-motion v2 for the AMI sensing/controller scripts.

Canonical ready path is split/scene/regime. split/scene remains a co_motion
alias so older Table 1 commands keep working.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path


def _link(source: Path, destination: Path) -> None:
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


REGIMES = ("co_motion", "camera_only", "object_only", "counter_motion", "drive_stop_drive")
SPECS = (("train", "Scene_002"), ("valid", "Scene_004"), ("test", "Scene_005"))


def _prepare(source: Path, destination: Path, split: str, scene: str, regime: str) -> dict[str, object]:
    destination.mkdir(parents=True, exist_ok=True)
    _link(source / "rgb", destination / "images")
    for name in ("imu.csv", "camera_pose.csv", "ego_pose.csv", "neighbor_pose.csv", "motion_state.csv", "camera_extrinsics.json", "motion_config.json", "render_config.json", "validation_report.md", "DONE.json", "object_masks"):
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
        "scene": f"{split}/{scene}/{regime}",
        "split": split,
        "condition": regime,
        "seed": int(scene.split("_")[-1]),
        "output": f"results/eveyes_comotion_v2_oracles/{split}_{scene}_{regime}.npz",
        "frame_count": len(rows),
        "speed_profile": config.get("speed_profile"),
        "has_object_masks": (source / "object_masks").is_dir(),
        "motion_regime": config.get("motion_regime", regime.upper()),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=Path("data/eveyes_co_motion_v2"))
    parser.add_argument("--output", type=Path, default=Path("data/eveyes_comotion_v2_ready"))
    args = parser.parse_args()
    records = []
    for split, scene in SPECS:
        packed = args.source_root / ("test" if split == "test" else "train") / scene
        for regime in REGIMES:
            source = packed / regime
            if not (source / "rgb").is_dir():
                continue
            records.append(_prepare(source, args.output / split / scene / regime, split, scene, regime))
            if regime == "co_motion":
                _prepare(source, args.output / split / scene, split, scene, regime)
    manifest = {
        "dataset": "EVEYES-co-motion-v2",
        "purpose": "active event sensing and optical modulation control",
        "source_root": str(args.source_root),
        "splits": {"train": ["Scene_002"], "valid": ["Scene_004"], "test": ["Scene_005"]},
        "regimes": [regime for regime in REGIMES if any(record["condition"] == regime for record in records)],
        "records": records,
        "imu": "nonzero gyro and variable acceleration; timestamp_s selected explicitly",
        "layout": "split/scene/regime is canonical; split/scene remains the co_motion alias for older scripts",
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
