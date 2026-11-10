#!/usr/bin/env python3
"""Extract a compact RGB/IMU/pose subset from the Eveyes dataset archive."""

from __future__ import annotations

import argparse
import zipfile
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--sequence", default="train/Scene_001/cam_moving_obj_moving")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-frames", type=int, default=32)
    args = parser.parse_args()
    if args.max_frames < 2:
        parser.error("max-frames must be at least 2")
    prefix = args.sequence.rstrip("/") + "/"
    metadata_names = {
        "camera_intrinsics.json",
        "camera_pose.csv",
        "DONE.json",
        "imu.csv",
        "motion_config.json",
        "render_config.json",
        "timestamps.csv",
        "validation_report.md",
    }
    with zipfile.ZipFile(args.archive) as archive:
        names = set(archive.namelist())
        metadata = [prefix + name for name in metadata_names if prefix + name in names]
        images = sorted(
            (name for name in names if name.startswith(prefix + "rgb/") and name.endswith(".png")),
            key=lambda name: int(Path(name).stem),
        )[: args.max_frames]
        if len(images) < 2:
            raise ValueError(f"Sequence {args.sequence!r} contains fewer than two RGB frames.")
        selected = metadata + images
        for name in selected:
            relative = Path(name).relative_to(prefix)
            destination = args.output / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(name) as source, destination.open("wb") as target:
                target.write(source.read())
    print(f"extracted {len(images)} RGB frames and {len(metadata)} metadata files to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
