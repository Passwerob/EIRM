#!/usr/bin/env python3
"""Stream a fixed/IMU-conditioned AMI comparison directly to an MP4.

The script keeps the threshold-event reference state across frames but never
stores the full generated event stream. This bounds memory and disk usage for
long demos. An optional raw DSEC H5 stream is rendered on the same timeline.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from collections import deque
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
import imageio.v3 as imageio
import numpy as np
import torch

from ami_simulator.dsec import read_dsec_events_h5, read_imu_csv
from ami_simulator.geometry import RotatingWedgePrism
from ami_simulator.pipeline import rgb_to_linear_luminance
from ami_simulator.rotation import ConstantRotation, ImuGearRotation
from ami_simulator.types import CameraIntrinsics, EventStream
from ami_simulator.warp import PrismWarper


def _timestamps(path: Path, unit: str) -> np.ndarray:
    rows = [row for row in csv.reader(path.read_text(encoding="utf-8").splitlines()) if row]
    try:
        values = np.asarray([float(row[-1]) for row in rows], dtype=np.float64)
    except ValueError:
        rows.pop(0)
        values = np.asarray([float(row[-1]) for row in rows], dtype=np.float64)
    return values * {"s": 1.0, "ms": 1e-3, "us": 1e-6, "ns": 1e-9}[unit]


class EventImageAccumulator:
    """Stateful threshold crossing and trailing-window count images."""

    def __init__(self, height: int, width: int, threshold: float, window_s: float) -> None:
        self.height = height
        self.width = width
        self.threshold = float(threshold)
        self.window_s = float(window_s)
        self.reference: np.ndarray | None = None
        self.entries: deque[tuple[float, np.ndarray, np.ndarray]] = deque()
        self.positive = np.zeros((height, width), dtype=np.int32)
        self.negative = np.zeros((height, width), dtype=np.int32)
        self.total_events = 0

    def update(self, intensity: np.ndarray, timestamp: float) -> tuple[np.ndarray, np.ndarray, int]:
        log_image = np.log(np.clip(np.asarray(intensity, dtype=np.float64), 1e-6, 1.0))
        if self.reference is None:
            self.reference = log_image.copy()
            return self.positive, self.negative, 0
        delta = log_image - self.reference
        positive = np.floor(np.maximum(delta, 0.0) / self.threshold + 1e-6).astype(np.int16)
        negative = np.floor(np.maximum(-delta, 0.0) / self.threshold + 1e-6).astype(np.int16)
        self.reference += (positive.astype(np.float64) - negative.astype(np.float64)) * self.threshold
        count = int(positive.sum(dtype=np.int64) + negative.sum(dtype=np.int64))
        self.total_events += count
        self.entries.append((timestamp, positive, negative))
        self.positive += positive
        self.negative += negative
        while self.entries and self.entries[0][0] < timestamp - self.window_s:
            _, old_positive, old_negative = self.entries.popleft()
            self.positive -= old_positive
            self.negative -= old_negative
        return self.positive, self.negative, count


def _event_counts(events: EventStream, height: int, width: int) -> tuple[np.ndarray, np.ndarray]:
    positive = np.zeros((height, width), dtype=np.int32)
    negative = np.zeros((height, width), dtype=np.int32)
    valid = (events.x >= 0) & (events.x < width) & (events.y >= 0) & (events.y < height)
    x, y, polarity = events.x[valid], events.y[valid], events.p[valid]
    mask = polarity > 0
    np.add.at(positive, (y[mask], x[mask]), 1)
    np.add.at(negative, (y[~mask], x[~mask]), 1)
    return positive, negative


def _render_events(positive: np.ndarray, negative: np.ndarray) -> np.ndarray:
    positive_level = np.log1p(positive.astype(np.float32)) / np.log(6.0)
    negative_level = np.log1p(negative.astype(np.float32)) / np.log(6.0)
    positive_level = np.clip(positive_level, 0.0, 1.0)
    negative_level = np.clip(negative_level, 0.0, 1.0)
    canvas = np.full((*positive.shape, 3), 20.0, dtype=np.float32)
    canvas[..., 0] += 225.0 * positive_level + 20.0 * negative_level
    canvas[..., 1] += 105.0 * positive_level + 145.0 * negative_level
    canvas[..., 2] += 20.0 * positive_level + 225.0 * negative_level
    return np.clip(canvas, 0, 255).astype(np.uint8)


def _panel(image: np.ndarray, title: str, detail: str) -> np.ndarray:
    output = image.copy()
    cv2.rectangle(output, (0, 0), (output.shape[1], 38), (8, 8, 8), thickness=-1)
    cv2.putText(output, title, (12, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (245, 245, 245), 1, cv2.LINE_AA)
    cv2.rectangle(output, (0, output.shape[0] - 30), (output.shape[1], output.shape[0]), (8, 8, 8), thickness=-1)
    cv2.putText(output, detail, (12, output.shape[0] - 9), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (225, 225, 225), 1, cv2.LINE_AA)
    return output


def _interpolated_rgb(paths: list[Path], source_times: np.ndarray, timestamp: float, width: int, height: int) -> np.ndarray:
    right = int(np.searchsorted(source_times, timestamp, side="right"))
    right = min(max(right, 1), len(source_times) - 1)
    left = right - 1
    alpha = float((timestamp - source_times[left]) / (source_times[right] - source_times[left]))
    first = cv2.resize(np.asarray(imageio.imread(paths[left]))[..., :3], (width, height), interpolation=cv2.INTER_AREA).astype(np.float32)
    second = cv2.resize(np.asarray(imageio.imread(paths[right]))[..., :3], (width, height), interpolation=cv2.INTER_AREA).astype(np.float32)
    return np.clip(first * (1.0 - alpha) + second * alpha, 0, 255).astype(np.uint8)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rgb-dir", type=Path, required=True)
    parser.add_argument("--timestamps", type=Path, required=True)
    parser.add_argument("--timestamp-unit", choices=["s", "ms", "us", "ns"], default="us")
    parser.add_argument("--imu-csv", type=Path, required=True)
    parser.add_argument("--imu-timestamp-unit", choices=["s", "ms", "us", "ns"], default="us")
    parser.add_argument("--dsec-events", type=Path, default=None)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--frames", type=int, default=120)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--window-ms", type=float, default=50.0)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--source-width", type=int, default=1440)
    parser.add_argument("--source-height", type=int, default=1080)
    parser.add_argument("--fx", type=float, default=1164.6238115833075)
    parser.add_argument("--fy", type=float, default=1164.6238115833075)
    parser.add_argument("--cx", type=float, default=713.5791168212891)
    parser.add_argument("--cy", type=float, default=570.9349365234375)
    parser.add_argument("--base-rpm", type=float, default=720.0)
    parser.add_argument("--contrast-threshold", type=float, default=0.2)
    parser.add_argument("--speed-thresholds", type=float, nargs=4, default=(0.025, 0.032, 0.040, 0.048))
    args = parser.parse_args()
    if args.frames < 2 or args.fps <= 0 or args.window_ms <= 0:
        raise SystemExit("frames >= 2, fps > 0, and window-ms > 0 are required")

    paths = sorted(path for path in args.rgb_dir.iterdir() if path.suffix.lower() in {".png", ".jpg", ".jpeg"})
    source_times = _timestamps(args.timestamps, args.timestamp_unit)[: len(paths)]
    start_time = float(source_times[0])
    output_times = start_time + np.arange(args.frames, dtype=np.float64) / args.fps
    if output_times[-1] > source_times[-1]:
        raise SystemExit("RGB sequence does not cover the requested output duration")
    needed = int(np.searchsorted(source_times, output_times[-1], side="right")) + 1
    paths, source_times = paths[:needed], source_times[:needed]
    imu = read_imu_csv(args.imu_csv, timestamp_unit=args.imu_timestamp_unit, gyro_unit="rad/s")
    if output_times[0] < imu.t[0] or output_times[-1] > imu.t[-1]:
        raise SystemExit("IMU does not cover the requested output timeline")

    source_intrinsics = CameraIntrinsics(args.source_width, args.source_height, args.fx, args.fy, args.cx, args.cy)
    intrinsics = source_intrinsics.resized(args.width, args.height)
    prism = RotatingWedgePrism(intrinsics, 0.5)
    fixed_rotation = ConstantRotation(args.base_rpm)
    gear_rotation = ImuGearRotation(imu, base_rpm=args.base_rpm, speed_thresholds_rad_s=tuple(args.speed_thresholds))
    fixed_warper = PrismWarper(prism, fixed_rotation, height=args.height, width=args.width, reference_time=start_time, padding_mode="border")
    gear_warper = PrismWarper(prism, gear_rotation, height=args.height, width=args.width, reference_time=start_time, padding_mode="border")
    fixed_events = EventImageAccumulator(args.height, args.width, args.contrast_threshold, args.window_ms * 1e-3)
    gear_events = EventImageAccumulator(args.height, args.width, args.contrast_threshold, args.window_ms * 1e-3)

    panel_count = 4 if args.dsec_events else 3
    args.output.parent.mkdir(parents=True, exist_ok=True)
    ffmpeg = subprocess.Popen(
        ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-vcodec", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{args.width * panel_count}x{args.height}", "-r", f"{args.fps:.6f}", "-i", "-", "-an", "-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p", str(args.output)],
        stdin=subprocess.PIPE,
    )
    frame_stats = []
    try:
        for index, timestamp in enumerate(output_times):
            rgb = _interpolated_rgb(paths, source_times, float(timestamp), args.width, args.height)
            luminance = rgb_to_linear_luminance(rgb[None])[0]
            tensor = torch.from_numpy(luminance).view(1, 1, args.height, args.width)
            with torch.inference_mode():
                fixed_sensor, _ = fixed_warper.warp(tensor, np.asarray([timestamp]))
                gear_sensor, _ = gear_warper.warp(tensor, np.asarray([timestamp]))
            fixed_positive, fixed_negative, fixed_new = fixed_events.update(fixed_sensor[0, 0].numpy(), float(timestamp))
            gear_positive, gear_negative, gear_new = gear_events.update(gear_sensor[0, 0].numpy(), float(timestamp))
            motion = float(np.interp(timestamp, imu.t, np.linalg.norm(imu.gyro, axis=1)))
            imu_index = min(int(np.searchsorted(imu.t, timestamp, side="right") - 1), len(imu.t) - 1)
            gear_index = int(gear_rotation.gear_indices[max(imu_index, 0)])
            rpm = args.base_rpm * gear_rotation.gear_scales[gear_index]
            elapsed = float(timestamp - start_time)
            panels = [_panel(rgb, "RGB", f"t={elapsed:5.2f}s  gyro={motion:.4f} rad/s")]
            dvs_count = None
            if args.dsec_events:
                dvs = read_dsec_events_h5(args.dsec_events, start_time=float(timestamp - args.window_ms * 1e-3), end_time=float(timestamp))
                dvs_positive, dvs_negative = _event_counts(dvs, args.height, args.width)
                dvs_count = len(dvs)
                panels.append(_panel(_render_events(dvs_positive, dvs_negative), "DSEC DVS", f"{args.window_ms:g}ms events={dvs_count:,}"))
            panels.extend([
                _panel(_render_events(fixed_positive, fixed_negative), "Fixed AMI", f"RPM={args.base_rpm:.0f}  new events={fixed_new:,}"),
                _panel(_render_events(gear_positive, gear_negative), "IMU Gear", f"gear={gear_index}  RPM={rpm:.0f}  new events={gear_new:,}"),
            ])
            frame = np.ascontiguousarray(np.concatenate(panels, axis=1))
            assert ffmpeg.stdin is not None
            ffmpeg.stdin.write(frame.tobytes())
            frame_stats.append({"frame": index, "time_seconds": elapsed, "gyro_norm_rad_s": motion, "gear": gear_index, "rpm": rpm, "dvs_window_events": dvs_count, "fixed_new_events": fixed_new, "imu_gear_new_events": gear_new})
            if index == 0 or (index + 1) % 10 == 0:
                print(f"rendered {index + 1}/{args.frames}", flush=True)
    finally:
        if ffmpeg.stdin is not None:
            ffmpeg.stdin.close()
        return_code = ffmpeg.wait()
        if return_code != 0:
            raise RuntimeError(f"ffmpeg failed with exit code {return_code}")

    metadata = {
        "output": str(args.output),
        "frames": args.frames,
        "fps": args.fps,
        "duration_seconds": args.frames / args.fps,
        "resolution": [args.width * panel_count, args.height],
        "simulation_resolution": [args.width, args.height],
        "event_window_ms": args.window_ms,
        "fixed_total_events": fixed_events.total_events,
        "imu_gear_total_events": gear_events.total_events,
        "gear_counts_at_video_frames": [int(sum(item["gear"] == gear for item in frame_stats)) for gear in range(len(gear_rotation.gear_scales))],
        "frames_detail": frame_stats,
    }
    args.output.with_suffix(".json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in metadata.items() if key != "frames_detail"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
