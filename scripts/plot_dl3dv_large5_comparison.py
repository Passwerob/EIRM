#!/usr/bin/env python3
"""Create a compact visual summary from evaluate_dl3dv_large5.py output."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("results/dl3dv_large5_comparison.json"))
    parser.add_argument("--output", type=Path, default=Path("results/dl3dv_large5_comparison.png"))
    parser.add_argument("--csv", type=Path, default=Path("results/dl3dv_large5_comparison.csv"))
    args = parser.parse_args()
    data = json.loads(args.input.read_text(encoding="utf-8"))
    scenes = data["scenes"]
    labels = [s["scene_id"].replace("screen-", "") for s in scenes]
    methods = [("passive", "Passive", "#667085"), ("fixed_ami", "Fixed AMI", "#d97706"), ("imu_gear", "IMU Gear", "#087f5b")]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["scene_id", "passive_events", "fixed_ami_events", "imu_gear_events", "gyro_mean_rad_s", "imu_gear_mean_rpm"])
        for scene in scenes:
            writer.writerow([scene["scene_id"], scene["passive"]["events"], scene["fixed_ami"]["events"], scene["imu_gear"]["events"], scene["gyro_norm_rad_s"]["mean"], scene["imu_gear"]["mean_rpm"]])
    x = np.arange(len(labels))
    width = 0.25
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), dpi=160)
    for offset, (key, label, color) in enumerate(methods):
        values = np.asarray([s[key]["events"] for s in scenes]) / 1e6
        axes[0].bar(x + (offset - 1) * width, values, width, label=label, color=color)
    axes[0].set_ylabel("Events (million)")
    axes[0].set_xlabel("DL3DV scene")
    axes[0].set_xticks(x, labels)
    axes[0].grid(axis="y", alpha=0.25)
    axes[0].legend(frameon=False)
    gyro = [s["gyro_norm_rad_s"]["mean"] for s in scenes]
    rpm = [s["imu_gear"]["mean_rpm"] for s in scenes]
    ax = axes[1]
    ax.plot(x, gyro, "o-", color="#1d4ed8", label="IMU gyro mean (rad/s)")
    ax.set_ylabel("Gyro magnitude (rad/s)", color="#1d4ed8")
    ax.tick_params(axis="y", labelcolor="#1d4ed8")
    ax.set_xticks(x, labels)
    ax.set_xlabel("DL3DV scene")
    ax.grid(alpha=0.25)
    ax2 = ax.twinx()
    ax2.plot(x, rpm, "s--", color="#b42318", label="IMU Gear mean RPM")
    ax2.set_ylabel("Prism speed (RPM)", color="#b42318")
    ax2.tick_params(axis="y", labelcolor="#b42318")
    lines = ax.lines + ax2.lines
    ax.legend(lines, [line.get_label() for line in lines], frameon=False, loc="upper left")
    fig.suptitle("DL3DV Large-5: event volume and IMU-conditioned prism speed")
    fig.tight_layout()
    fig.savefig(args.output, bbox_inches="tight")
    print(json.dumps({"png": str(args.output), "csv": str(args.csv)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
