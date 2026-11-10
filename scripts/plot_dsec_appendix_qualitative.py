#!/usr/bin/env python3
"""Extract appendix DSEC qualitative rows from an existing rendered video.

This script only reads the stored DSEC video/JSON. It does not regenerate
events or run a simulator. The video already contains RGB, native DSEC,
passive, Fixed AMI, and IMU Gear compensated panels with shared rendering.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _representative_frames(rows: list[dict], count: int = 3) -> list[int]:
    motion = np.asarray([float(row["gyro_norm_rad_s"]) for row in rows], dtype=np.float64)
    selected: list[int] = []
    for quantile in np.linspace(0.1, 0.9, count):
        index = int(np.argmin(np.abs(motion - np.quantile(motion, quantile))))
        if index not in selected:
            selected.append(index)
    if len(selected) < count:
        for index in np.argsort(motion):
            if int(index) not in selected:
                selected.append(int(index))
            if len(selected) == count:
                break
    return selected[:count]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=Path, default=Path("results/dsec_consistent_ami_compensated_120f_4s.mp4"))
    parser.add_argument("--metadata", type=Path, default=Path("results/dsec_consistent_ami_compensated_120f_4s.json"))
    parser.add_argument("--output", type=Path, default=Path("results/ami_next/figures/dsec_appendix_qualitative"))
    parser.add_argument("--panel-width", type=int, default=512)
    parser.add_argument("--top-height", type=int, default=300)
    args = parser.parse_args()

    metadata = json.loads(args.metadata.read_text(encoding="utf-8"))
    detail = metadata["frames_detail"]
    frame_indices = _representative_frames(detail, 3)
    capture = cv2.VideoCapture(str(args.video))
    if not capture.isOpened():
        raise SystemExit(f"failed to open {args.video}")

    rows: list[dict[str, object]] = []
    for frame_index in frame_indices:
        capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, frame = capture.read()
        if not ok:
            raise SystemExit(f"failed to read video frame {frame_index}")
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        panels = [
            frame[: args.top_height, column * args.panel_width : (column + 1) * args.panel_width]
            for column in range(5)
        ]
        row_meta = detail[frame_index]
        rows.append({
            "video_frame": int(frame_index),
            "time_seconds": float(row_meta["time_seconds"]),
            "absolute_time_seconds": float(row_meta["absolute_time_seconds"]),
            "gyro_norm_rad_s": float(row_meta["gyro_norm_rad_s"]),
            "smoothed_motion_rad_s": float(row_meta["smoothed_motion_rad_s"]),
            "gear": int(row_meta["gear"]),
            "actual_rpm": float(row_meta["actual_rpm"]),
            "dsec_reference_events_50ms": int(row_meta["dsec_reference_events_50ms"]),
            "passive_events_50ms": int(row_meta["passive_events_50ms"]),
            "fixed_events_50ms": int(row_meta["fixed_events_50ms"]),
            "imu_gear_events_50ms": int(row_meta["imu_gear_events_50ms"]),
            "passive_coverage": float(row_meta["passive_coverage"]),
            "fixed_coverage": float(row_meta["fixed_coverage"]),
            "imu_gear_coverage": float(row_meta["imu_gear_coverage"]),
            "fixed_raw_flicker": None if row_meta.get("fixed_raw_flicker") is None else float(row_meta["fixed_raw_flicker"]),
            "imu_gear_raw_flicker": None if row_meta.get("imu_gear_raw_flicker") is None else float(row_meta["imu_gear_raw_flicker"]),
            "fixed_comp_flicker": None if row_meta.get("fixed_comp_flicker") is None else float(row_meta["fixed_comp_flicker"]),
            "imu_gear_comp_flicker": None if row_meta.get("imu_gear_comp_flicker") is None else float(row_meta["imu_gear_comp_flicker"]),
            "panels": panels,
        })
    capture.release()

    labels = ["RGB source", "Native DSEC\n(un-aligned)", "Passive\n(same source)", "Fixed AMI\n(stored video label)", "IMU Gear / adaptive\n(prism-comp.)"]
    figure, axes = plt.subplots(3, 5, figsize=(15.5, 7.6), dpi=220, squeeze=False)
    figure.patch.set_facecolor("white")
    for row_index, record in enumerate(rows):
        for column_index, (axis, image) in enumerate(zip(axes[row_index], record["panels"])):
            axis.imshow(image)
            axis.set_xticks([])
            axis.set_yticks([])
            for spine in axis.spines.values():
                spine.set_color("#555555")
                spine.set_linewidth(0.45)
            if row_index == 0:
                axis.set_title(labels[column_index], fontsize=8.0, pad=4)
        axes[row_index, 0].set_ylabel(
            f"{record['time_seconds']:.2f} s\n|gyro|={record['gyro_norm_rad_s']:.3f} rad/s\nRPM={record['actual_rpm']:.0f}",
            fontsize=7.2,
            labelpad=8,
        )
    figure.suptitle("DSEC real-trajectory appendix: native events and offline optical modulation", fontsize=11, fontweight="semibold", y=0.985)
    figure.text(0.5, 0.012, "Native DSEC is an unaligned real-event reference; all other columns are offline RGB+IMU simulation outputs from the stored video.", ha="center", fontsize=7.2, color="#374151")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output.with_suffix(".png"), bbox_inches="tight", facecolor="white")
    figure.savefig(args.output.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(figure)

    serializable = {
        "source_video": str(args.video),
        "source_metadata": str(args.metadata),
        "scene": "DSEC thun_00_a",
        "simulation_resolution": metadata.get("simulation_resolution"),
        "event_window_ms": metadata.get("event_window_ms"),
        "panel_order": labels,
        "selection": "three stored video frames at approximately 10th, 50th, and 90th gyro-magnitude quantiles",
        "rows": [{key: value for key, value in record.items() if key != "panels"} for record in rows],
        "outputs": [str(args.output.with_suffix(".png")), str(args.output.with_suffix(".pdf"))],
    }
    args.output.with_suffix(".json").write_text(json.dumps(serializable, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(serializable, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
