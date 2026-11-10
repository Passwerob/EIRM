#!/usr/bin/env python3
"""Download a small DSEC scene subset from the official per-sequence files.

The official DSEC ``lidar_imu.zip`` is one 26 GB archive covering train and
test. This script intentionally does not download it by default. Use
``--include-lidar-imu`` only when that archive is wanted; afterwards extract
the target scene's IMU messages with your ROS bag tooling.
"""

from __future__ import annotations

import argparse
import urllib.request
from pathlib import Path


BASE = "https://download.ifi.uzh.ch/rpg/DSEC"


def download(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and destination.stat().st_size > 0:
        print(f"exists {destination}")
        return
    print(f"download {url} -> {destination}")
    temporary = destination.with_suffix(destination.suffix + ".part")
    try:
        urllib.request.urlretrieve(url, temporary)
        temporary.replace(destination)
    finally:
        if temporary.exists():
            temporary.unlink()


def main() -> int:
    parser = argparse.ArgumentParser(description="Download one compact DSEC training scene subset.")
    parser.add_argument("--scene", default="thun_00_a", help="DSEC training scene, e.g. thun_00_a")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--include-images", action="store_true", help="Also download the large rectified-left image archive.")
    parser.add_argument("--include-lidar-imu", action="store_true", help="Download the official 26 GB train+test lidar/IMU archive.")
    args = parser.parse_args()
    scene = args.scene.strip()
    if not scene or "/" in scene or "\\" in scene:
        parser.error("scene must be a simple DSEC sequence name")
    prefix = f"{BASE}/train/{scene}/{scene}"
    files = [
        (f"{prefix}_events_left.zip", f"{scene}_events_left.zip"),
        (f"{prefix}_calibration.zip", f"{scene}_calibration.zip"),
        (f"{prefix}_image_timestamps.txt", f"{scene}_image_timestamps.txt"),
        (f"{prefix}_image_exposure_timestamps_left.txt", f"{scene}_image_exposure_timestamps_left.txt"),
    ]
    if args.include_images:
        files.append((f"{prefix}_images_rectified_left.zip", f"{scene}_images_rectified_left.zip"))
    if args.include_lidar_imu:
        files.append((f"{BASE}/lidar_imu.zip", "lidar_imu.zip"))
        print("warning: lidar_imu.zip is an official 26 GB archive containing all train and test scenes")
    for url, name in files:
        download(url, args.output / name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
