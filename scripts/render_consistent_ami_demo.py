#!/usr/bin/env python3
"""Render a phase-sampled, same-source Passive/Fixed/IMU AMI comparison.

Event formation runs on an internal timeline constrained by prism phase change,
while the MP4 remains 30 FPS. Generated event panels share the same source,
camera plane, threshold, timestamps, and trailing accumulation window. A raw
DSEC stream may be shown as an explicitly unaligned reference panel.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
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
import torch.nn.functional as functional

from ami_simulator.dsec import read_dsec_events_h5, read_imu_csv
from ami_simulator.geometry import RotatingWedgePrism
from ami_simulator.pipeline import rgb_to_linear_luminance
from ami_simulator.rotation import ConstantRotation, ImuGearRotation
from ami_simulator.types import CameraIntrinsics, EventStream
from ami_simulator.warp import PrismWarper

EVENT_COLORS = {
    "dsec": (180, 180, 180),
    "passive": (80, 210, 130),
    "fixed": (245, 165, 55),
    "gear": (225, 85, 190),
}


def _timestamps(path: Path, unit: str) -> np.ndarray:
    rows = [row for row in csv.reader(path.read_text(encoding="utf-8").splitlines()) if row]
    try:
        values = np.asarray([float(row[-1]) for row in rows], dtype=np.float64)
    except ValueError:
        values = np.asarray([float(row[-1]) for row in rows[1:]], dtype=np.float64)
    return values * {"s": 1.0, "ms": 1e-3, "us": 1e-6, "ns": 1e-9}[unit]


class EventWindow:
    """Stateful threshold formation with exact internal-sample window expiry."""

    def __init__(self, height: int, width: int, threshold: float, window_s: float) -> None:
        self.threshold = float(threshold)
        self.window_s = float(window_s)
        self.reference: np.ndarray | None = None
        self.entries: deque[tuple[float, np.ndarray, np.ndarray]] = deque()
        self.positive = np.zeros((height, width), dtype=np.float32)
        self.negative = np.zeros((height, width), dtype=np.float32)
        self.total_events = 0

    def update(
        self,
        intensity: np.ndarray,
        timestamp: float,
        *,
        display_positive: np.ndarray | None = None,
        display_negative: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        log_image = np.log(np.clip(np.asarray(intensity, dtype=np.float64), 1e-6, 1.0))
        if self.reference is None:
            self.reference = log_image.copy()
            return np.zeros_like(log_image, dtype=np.int32), np.zeros_like(log_image, dtype=np.int32)
        delta = log_image - self.reference
        positive = np.floor(np.maximum(delta, 0.0) / self.threshold + 1e-6).astype(np.int16)
        negative = np.floor(np.maximum(-delta, 0.0) / self.threshold + 1e-6).astype(np.int16)
        self.reference += (positive.astype(np.float64) - negative.astype(np.float64)) * self.threshold
        self.total_events += int(positive.sum(dtype=np.int64) + negative.sum(dtype=np.int64))
        shown_positive = positive if display_positive is None else np.asarray(display_positive, dtype=np.float32)
        shown_negative = negative if display_negative is None else np.asarray(display_negative, dtype=np.float32)
        self.entries.append((float(timestamp), shown_positive, shown_negative))
        self.positive += shown_positive
        self.negative += shown_negative
        while self.entries and self.entries[0][0] < timestamp - self.window_s:
            _, old_positive, old_negative = self.entries.popleft()
            self.positive -= old_positive
            self.negative -= old_negative
        return positive, negative

    def stats(self) -> tuple[int, float]:
        count = int(round(float(self.positive.sum(dtype=np.float64) + self.negative.sum(dtype=np.float64))))
        coverage = float(np.count_nonzero(self.positive + self.negative) / self.positive.size)
        return count, coverage


def _compensate_count_maps(positive: np.ndarray, negative: np.ndarray, grid: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
    """Map positive and negative count images in one grid-sample call."""
    source = torch.from_numpy(np.stack((positive, negative)).astype(np.float32)).unsqueeze(0)
    with torch.inference_mode():
        restored = functional.grid_sample(source, grid, mode="bilinear", padding_mode="zeros", align_corners=False)
    return restored[0, 0].numpy(), restored[0, 1].numpy()


def _prism_grid(prism: RotatingWedgePrism, theta: float, height: int, width: int) -> torch.Tensor:
    v, u = torch.meshgrid(torch.arange(height, dtype=torch.float32), torch.arange(width, dtype=torch.float32), indexing="ij")
    reference_uv = torch.stack((u, v), dim=-1)
    sensor_uv = prism.forward_points(reference_uv, float(theta))
    grid = torch.stack((2.0 * (sensor_uv[..., 0] + 0.5) / width - 1.0, 2.0 * (sensor_uv[..., 1] + 0.5) / height - 1.0), dim=-1)
    return grid.unsqueeze(0)


def _event_counts(events: EventStream, height: int, width: int) -> tuple[np.ndarray, np.ndarray]:
    positive = np.zeros((height, width), dtype=np.int32)
    negative = np.zeros((height, width), dtype=np.int32)
    valid = (events.x >= 0) & (events.x < width) & (events.y >= 0) & (events.y < height)
    x, y, polarity = events.x[valid], events.y[valid], events.p[valid]
    mask = polarity > 0
    np.add.at(positive, (y[mask], x[mask]), 1)
    np.add.at(negative, (y[~mask], x[~mask]), 1)
    return positive, negative


def _slice_events(events: EventStream, start_time: float, end_time: float) -> EventStream:
    """Select an event interval from a time-sorted stream without reopening its source."""
    start = int(np.searchsorted(events.t, float(start_time), side="left"))
    end = int(np.searchsorted(events.t, float(end_time), side="right"))
    return EventStream(events.x[start:end], events.y[start:end], events.t[start:end], events.p[start:end])


def _render_events(positive: np.ndarray, negative: np.ndarray, saturation_count: float) -> np.ndarray:
    denominator = np.log1p(float(saturation_count))
    positive_level = np.clip(np.log1p(positive.astype(np.float32)) / denominator, 0.0, 1.0)
    negative_level = np.clip(np.log1p(negative.astype(np.float32)) / denominator, 0.0, 1.0)
    canvas = np.full((*positive.shape, 3), 18.0, dtype=np.float32)
    canvas[..., 0] += 225.0 * positive_level + 20.0 * negative_level
    canvas[..., 1] += 105.0 * positive_level + 145.0 * negative_level
    canvas[..., 2] += 20.0 * positive_level + 225.0 * negative_level
    return np.clip(canvas, 0, 255).astype(np.uint8)


def _panel(image: np.ndarray, title: str, detail: str, width: int, height: int) -> np.ndarray:
    interpolation = cv2.INTER_AREA if image.shape[1] >= width else cv2.INTER_LINEAR
    output = cv2.resize(image, (width, height), interpolation=interpolation)
    cv2.rectangle(output, (0, 0), (width, 36), (7, 7, 7), thickness=-1)
    cv2.putText(output, title, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (245, 245, 245), 1, cv2.LINE_AA)
    cv2.rectangle(output, (0, height - 29), (width, height), (7, 7, 7), thickness=-1)
    cv2.putText(output, detail, (10, height - 9), cv2.FONT_HERSHEY_SIMPLEX, 0.43, (225, 225, 225), 1, cv2.LINE_AA)
    return output


def _load_rgb(paths: list[Path], width: int, height: int) -> np.ndarray:
    frames = []
    for path in paths:
        frame = np.asarray(imageio.imread(path))[..., :3]
        frames.append(cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA))
    return np.stack(frames).astype(np.uint8)


def _interpolate_rgb(frames: np.ndarray, source_times: np.ndarray, times: np.ndarray) -> np.ndarray:
    right = np.searchsorted(source_times, times, side="right")
    right = np.clip(right, 1, len(source_times) - 1)
    left = right - 1
    alpha = ((times - source_times[left]) / (source_times[right] - source_times[left])).astype(np.float32)
    return frames[left].astype(np.float32) * (1.0 - alpha[:, None, None, None]) + frames[right].astype(np.float32) * alpha[:, None, None, None]


def _draw_chart(
    canvas: np.ndarray,
    bounds: tuple[int, int, int, int],
    title: str,
    times: np.ndarray,
    index: int,
    series: list[tuple[str, np.ndarray, tuple[int, int, int]]],
    y_range: tuple[float, float],
    thresholds: tuple[float, ...] = (),
) -> None:
    x0, y0, x1, y1 = bounds
    cv2.rectangle(canvas, (x0, y0), (x1, y1), (18, 18, 18), thickness=-1)
    cv2.putText(canvas, title, (x0 + 12, y0 + 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (235, 235, 235), 1, cv2.LINE_AA)
    plot = (x0 + 48, y0 + 34, x1 - 12, y1 - 28)
    px0, py0, px1, py1 = plot
    for fraction in (0.0, 0.5, 1.0):
        y = int(py1 - fraction * (py1 - py0))
        cv2.line(canvas, (px0, y), (px1, y), (48, 48, 48), 1)
    low, high = y_range
    for value in thresholds:
        if low <= value <= high:
            y = int(py1 - (value - low) / (high - low) * (py1 - py0))
            cv2.line(canvas, (px0, y), (px1, y), (80, 80, 80), 1, cv2.LINE_AA)
    duration = max(float(times[-1]), 1e-9)
    for label_index, (label, values, color) in enumerate(series):
        valid = np.isfinite(values[: index + 1])
        if np.count_nonzero(valid) >= 2:
            x = px0 + np.clip(times[: index + 1][valid] / duration, 0.0, 1.0) * (px1 - px0)
            y = py1 - np.clip((values[: index + 1][valid] - low) / (high - low), 0.0, 1.0) * (py1 - py0)
            points = np.column_stack((x, y)).astype(np.int32)
            cv2.polylines(canvas, [points], False, color, 2, cv2.LINE_AA)
        value = values[index]
        text = f"{label} {value:.3g}" if np.isfinite(value) else label
        cv2.putText(canvas, text, (px0 + label_index * 130, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.39, color, 1, cv2.LINE_AA)
    marker_x = int(px0 + times[index] / duration * (px1 - px0))
    cv2.line(canvas, (marker_x, py0), (marker_x, py1), (220, 220, 220), 1, cv2.LINE_AA)


def _dashboard(history: dict[str, np.ndarray], times: np.ndarray, index: int, width: int, height: int, thresholds: tuple[float, ...], event_max: float) -> np.ndarray:
    canvas = np.full((height, width, 3), 12, dtype=np.uint8)
    chart_width = width // 5
    _draw_chart(canvas, (0, 0, chart_width, height), "IMU motion (rad/s)", times, index, [("raw", history["gyro"], (165, 165, 165)), ("EMA", history["smoothed"], (60, 210, 245))], (0.0, 0.065), thresholds)
    _draw_chart(canvas, (chart_width, 0, 2 * chart_width, height), "Prism speed (RPM)", times, index, [("fixed", history["fixed_rpm"], EVENT_COLORS["fixed"]), ("IMU", history["rpm"], EVENT_COLORS["gear"])], (0.0, 1200.0))
    _draw_chart(canvas, (2 * chart_width, 0, 3 * chart_width, height), "Events / 50 ms", times, index, [("DSEC", history["dsec_events"], EVENT_COLORS["dsec"]), ("pass", history["passive_events"], EVENT_COLORS["passive"]), ("fixed", history["fixed_events"], EVENT_COLORS["fixed"]), ("IMU", history["gear_events"], EVENT_COLORS["gear"])], (0.0, event_max))
    _draw_chart(canvas, (3 * chart_width, 0, 4 * chart_width, height), "Spatial coverage", times, index, [("pass", history["passive_coverage"], EVENT_COLORS["passive"]), ("fixed", history["fixed_coverage"], EVENT_COLORS["fixed"]), ("IMU", history["gear_coverage"], EVENT_COLORS["gear"])], (0.0, 1.0))
    _draw_chart(canvas, (4 * chart_width, 0, width - 1, height), "Flicker index", times, index, [("fixed raw", history["fixed_raw_flicker"], EVENT_COLORS["fixed"]), ("IMU raw", history["gear_raw_flicker"], EVENT_COLORS["gear"]), ("fixed comp", history["fixed_comp_flicker"], (245, 210, 95)), ("IMU comp", history["gear_comp_flicker"], (100, 230, 205))], (0.0, 1.0))
    return canvas


def _internal_times(start: float, end: float, fixed: ConstantRotation, gear: ImuGearRotation, phase_step_rad: float) -> np.ndarray:
    endpoints = np.asarray([start, end], dtype=np.float64)
    fixed_delta = abs(float(np.diff(fixed.theta_values(endpoints))[0]))
    gear_delta = abs(float(np.diff(gear.theta_values(endpoints))[0]))
    subdivisions = max(1, int(math.ceil(max(fixed_delta, gear_delta) / phase_step_rad)))
    return np.linspace(start, end, subdivisions + 1, dtype=np.float64)[1:]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rgb-dir", type=Path, required=True)
    parser.add_argument("--timestamps", type=Path, required=True)
    parser.add_argument("--timestamp-unit", choices=["s", "ms", "us", "ns"], default="us")
    parser.add_argument("--imu-csv", type=Path, required=True)
    parser.add_argument("--imu-timestamp-unit", choices=["s", "ms", "us", "ns"], default="us")
    parser.add_argument("--dsec-events", type=Path, default=None)
    parser.add_argument("--display-space", choices=("compensated", "raw", "both"), default="compensated", help="Display generated events on the reference plane, sensor plane, or both.")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--frames", type=int, default=120)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--window-ms", type=float, default=50.0)
    parser.add_argument("--phase-step-deg", type=float, default=5.0)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--panel-width", type=int, default=512)
    parser.add_argument("--panel-height", type=int, default=384)
    parser.add_argument("--chart-height", type=int, default=240)
    parser.add_argument("--source-width", type=int, default=1440)
    parser.add_argument("--source-height", type=int, default=1080)
    parser.add_argument("--fx", type=float, default=1164.6238115833075)
    parser.add_argument("--fy", type=float, default=1164.6238115833075)
    parser.add_argument("--cx", type=float, default=713.5791168212891)
    parser.add_argument("--cy", type=float, default=570.9349365234375)
    parser.add_argument("--base-rpm", type=float, default=720.0)
    parser.add_argument("--contrast-threshold", type=float, default=0.2)
    parser.add_argument("--render-saturation-count", type=float, default=16.0)
    parser.add_argument("--playback-speed", type=float, default=1.0, help="Physical-time speed relative to video time. Values below one create honest slow motion.")
    parser.add_argument("--speed-thresholds", type=float, nargs=4, default=(0.025, 0.032, 0.040, 0.048))
    parser.add_argument("--event-chart-max", type=float, default=2_500_000.0)
    parser.add_argument("--torch-threads", type=int, default=8, help="CPU threads for torch grid sampling; small image warps are often faster with a bounded pool.")
    args = parser.parse_args()
    if args.frames < 2 or min(args.fps, args.window_ms, args.phase_step_deg, args.render_saturation_count, args.event_chart_max, args.playback_speed) <= 0:
        raise SystemExit("frames >= 2 and positive fps/window/phase/render/chart/playback parameters are required")
    if args.torch_threads < 1:
        raise SystemExit("torch-threads must be positive")
    torch.set_num_threads(args.torch_threads)
    torch.set_num_interop_threads(1)

    all_paths = sorted(path for path in args.rgb_dir.iterdir() if path.suffix.lower() in {".png", ".jpg", ".jpeg"})
    all_source_times = _timestamps(args.timestamps, args.timestamp_unit)[: len(all_paths)]
    warmup_s = args.window_ms * 1e-3
    simulation_start = float(all_source_times[0])
    output_start = simulation_start + warmup_s
    output_times = output_start + np.arange(args.frames, dtype=np.float64) / args.fps * args.playback_speed
    if output_times[-1] > all_source_times[-1]:
        raise SystemExit("RGB sequence does not cover warmup plus requested output duration")
    needed = int(np.searchsorted(all_source_times, output_times[-1], side="right")) + 1
    paths, source_times = all_paths[:needed], all_source_times[:needed]
    frames = _load_rgb(paths, args.width, args.height)
    imu = read_imu_csv(args.imu_csv, timestamp_unit=args.imu_timestamp_unit, gyro_unit="rad/s")
    if simulation_start < imu.t[0] or output_times[-1] > imu.t[-1]:
        raise SystemExit("IMU does not cover warmup plus requested output duration")

    source_intrinsics = CameraIntrinsics(args.source_width, args.source_height, args.fx, args.fy, args.cx, args.cy)
    intrinsics = source_intrinsics.resized(args.width, args.height)
    prism = RotatingWedgePrism(intrinsics, 0.5)
    fixed_rotation = ConstantRotation(args.base_rpm)
    thresholds = tuple(float(value) for value in args.speed_thresholds)
    gear_rotation = ImuGearRotation(imu, base_rpm=args.base_rpm, speed_thresholds_rad_s=thresholds)
    dsec_stream = None
    if args.dsec_events:
        dsec_stream = read_dsec_events_h5(args.dsec_events, start_time=simulation_start, end_time=float(output_times[-1]))
    fixed_warper = PrismWarper(prism, fixed_rotation, height=args.height, width=args.width, reference_time=simulation_start, padding_mode="border")
    gear_warper = PrismWarper(prism, gear_rotation, height=args.height, width=args.width, reference_time=simulation_start, padding_mode="border")
    raw_windows = {name: EventWindow(args.height, args.width, args.contrast_threshold, warmup_s) for name in ("passive", "fixed", "gear")}
    compensated_windows = {name: EventWindow(args.height, args.width, args.contrast_threshold, warmup_s) for name in ("passive", "fixed", "gear")}

    initial_rgb = _interpolate_rgb(frames, source_times, np.asarray([simulation_start]))
    initial_luminance = rgb_to_linear_luminance(initial_rgb)[0]
    initial_tensor = torch.from_numpy(initial_luminance).view(1, 1, args.height, args.width)
    with torch.inference_mode():
        initial_fixed, _ = fixed_warper.warp(initial_tensor, np.asarray([simulation_start]))
        initial_gear, _ = gear_warper.warp(initial_tensor, np.asarray([simulation_start]))
    for name, intensity in (("passive", initial_luminance), ("fixed", initial_fixed[0, 0].numpy()), ("gear", initial_gear[0, 0].numpy())):
        raw_windows[name].update(intensity, simulation_start)
        compensated_windows[name].update(intensity, simulation_start)

    history = {name: np.full(args.frames, np.nan, dtype=np.float64) for name in ("gyro", "smoothed", "rpm", "fixed_rpm", "dsec_events", "passive_events", "fixed_events", "gear_events", "passive_coverage", "fixed_coverage", "gear_coverage", "fixed_raw_flicker", "gear_raw_flicker", "fixed_comp_flicker", "gear_comp_flicker")}
    elapsed_times = np.arange(args.frames, dtype=np.float64) / args.fps
    generated_spaces = ("raw", "compensated") if args.display_space == "both" else (args.display_space,)
    panel_count = (1 if args.dsec_events else 0) + 1 + len(generated_spaces) * 3
    video_width = args.panel_width * panel_count
    video_height = args.panel_height + args.chart_height
    args.output.parent.mkdir(parents=True, exist_ok=True)
    ffmpeg = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-vcodec", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{video_width}x{video_height}", "-r", f"{args.fps:.6f}", "-i", "-", "-an", "-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p", str(args.output)], stdin=subprocess.PIPE)
    phase_step_rad = math.radians(args.phase_step_deg)
    previous_time = simulation_start
    internal_sample_count = 1
    grid_cache: dict[tuple[str, float], torch.Tensor] = {}
    frame_stats = []
    previous_maps: dict[str, tuple[np.ndarray, np.ndarray] | None] = {"fixed_raw": None, "gear_raw": None, "fixed_comp": None, "gear_comp": None}
    try:
        for index, timestamp in enumerate(output_times):
            internal_times = _internal_times(previous_time, float(timestamp), fixed_rotation, gear_rotation, phase_step_rad)
            internal_sample_count += len(internal_times)
            rgb_batch = _interpolate_rgb(frames, source_times, internal_times)
            luminance = rgb_to_linear_luminance(rgb_batch)
            for value, sample_time in zip(luminance, internal_times):
                raw_windows["passive"].update(value, float(sample_time))
                compensated_windows["passive"].update(value, float(sample_time))
            tensor = torch.from_numpy(luminance).unsqueeze(1)
            with torch.inference_mode():
                fixed_sensor, _ = fixed_warper.warp(tensor, internal_times)
            fixed_theta = fixed_rotation.theta_values(internal_times, reference_time=simulation_start)
            for value, sample_time, theta in zip(fixed_sensor[:, 0].numpy(), internal_times, fixed_theta):
                positive, negative = raw_windows["fixed"].update(value, float(sample_time))
                if args.display_space != "raw":
                    grid_key = ("fixed", round(float(theta), 7))
                    grid = grid_cache.setdefault(grid_key, _prism_grid(prism, float(theta), args.height, args.width))
                    comp_positive, comp_negative = _compensate_count_maps(positive, negative, grid)
                    compensated_windows["fixed"].update(value, float(sample_time), display_positive=comp_positive, display_negative=comp_negative)
            del fixed_sensor
            with torch.inference_mode():
                gear_sensor, _ = gear_warper.warp(tensor, internal_times)
            gear_theta = gear_rotation.theta_values(internal_times, reference_time=simulation_start)
            for value, sample_time, theta in zip(gear_sensor[:, 0].numpy(), internal_times, gear_theta):
                positive, negative = raw_windows["gear"].update(value, float(sample_time))
                if args.display_space != "raw":
                    grid_key = ("gear", round(float(theta), 7))
                    grid = grid_cache.setdefault(grid_key, _prism_grid(prism, float(theta), args.height, args.width))
                    comp_positive, comp_negative = _compensate_count_maps(positive, negative, grid)
                    compensated_windows["gear"].update(value, float(sample_time), display_positive=comp_positive, display_negative=comp_negative)
            del gear_sensor, tensor

            imu_index = max(0, min(int(np.searchsorted(imu.t, timestamp, side="right") - 1), len(imu.t) - 1))
            gyro = float(np.interp(timestamp, imu.t, np.linalg.norm(imu.gyro, axis=1)))
            smoothed = float(np.interp(timestamp, imu.t, gear_rotation.smoothed_motion_rad_s))
            gear_index = int(gear_rotation.gear_indices[imu_index])
            rpm = float(gear_rotation.omega_values(np.asarray([timestamp]))[0] * 60.0 / (2.0 * np.pi))
            passive_count, passive_coverage = raw_windows["passive"].stats()
            fixed_count, fixed_coverage = raw_windows["fixed"].stats()
            gear_count, gear_coverage = raw_windows["gear"].stats()
            fixed_comp_count, fixed_comp_coverage = compensated_windows["fixed"].stats()
            gear_comp_count, gear_comp_coverage = compensated_windows["gear"].stats()
            current_maps = {
                "fixed_raw": (raw_windows["fixed"].positive, raw_windows["fixed"].negative),
                "gear_raw": (raw_windows["gear"].positive, raw_windows["gear"].negative),
                "fixed_comp": (compensated_windows["fixed"].positive, compensated_windows["fixed"].negative),
                "gear_comp": (compensated_windows["gear"].positive, compensated_windows["gear"].negative),
            }
            flicker: dict[str, float] = {}
            for name, (positive, negative) in current_maps.items():
                previous = previous_maps[name]
                if previous is None:
                    flicker[name] = 0.0
                else:
                    current = positive + negative
                    old = previous[0] + previous[1]
                    flicker[name] = float(np.mean(np.abs(current - old)) / max(float(np.mean(current) + np.mean(old)), 1.0))
                previous_maps[name] = (positive.copy(), negative.copy())
            dsec_count = np.nan
            dsec_positive = dsec_negative = None
            if dsec_stream is not None:
                dsec = _slice_events(dsec_stream, float(timestamp - warmup_s), float(timestamp))
                dsec_positive, dsec_negative = _event_counts(dsec, args.height, args.width)
                dsec_count = float(len(dsec))

            values = {
                "gyro": gyro,
                "smoothed": smoothed,
                "rpm": rpm,
                "fixed_rpm": args.base_rpm,
                "dsec_events": dsec_count,
                "passive_events": passive_count,
                "fixed_events": fixed_count,
                "gear_events": gear_count,
                "passive_coverage": passive_coverage,
                "fixed_coverage": fixed_coverage,
                "gear_coverage": gear_coverage,
                "fixed_raw_flicker": flicker["fixed_raw"],
                "gear_raw_flicker": flicker["gear_raw"],
                "fixed_comp_flicker": flicker["fixed_comp"],
                "gear_comp_flicker": flicker["gear_comp"],
            }
            for name, value in values.items():
                history[name][index] = value

            rgb_output = np.clip(rgb_batch[-1], 0, 255).astype(np.uint8)
            panels = [_panel(rgb_output, "RGB source", f"t={elapsed_times[index]:.2f}s  same camera trajectory", args.panel_width, args.panel_height)]
            if args.dsec_events and dsec_positive is not None and dsec_negative is not None:
                panels.append(_panel(_render_events(dsec_positive, dsec_negative, args.render_saturation_count), "DSEC reference (unaligned)", f"50ms events={int(dsec_count):,}", args.panel_width, args.panel_height))
            generated_panels = []
            for space in generated_spaces:
                source = raw_windows if space == "raw" else compensated_windows
                fixed_display_count, fixed_display_coverage = source["fixed"].stats()
                gear_display_count, gear_display_coverage = source["gear"].stats()
                generated_panels.extend([
                    _panel(_render_events(source["passive"].positive, source["passive"].negative, args.render_saturation_count), f"Passive ({space})", f"50ms events={passive_count:,}  coverage={passive_coverage:.1%}", args.panel_width, args.panel_height),
                    _panel(_render_events(source["fixed"].positive, source["fixed"].negative, args.render_saturation_count), f"Fixed AMI ({space})", f"50ms events={fixed_display_count:,}  RPM={args.base_rpm:.0f}", args.panel_width, args.panel_height),
                    _panel(_render_events(source["gear"].positive, source["gear"].negative, args.render_saturation_count), f"IMU Gear ({space})", f"50ms events={gear_display_count:,}  gear={gear_index}  RPM={rpm:.0f}", args.panel_width, args.panel_height),
                ])
            panels.extend(generated_panels)
            top = np.concatenate(panels, axis=1)
            charts = _dashboard(history, elapsed_times, index, video_width, args.chart_height, thresholds, args.event_chart_max)
            frame = np.ascontiguousarray(np.concatenate((top, charts), axis=0))
            assert ffmpeg.stdin is not None
            ffmpeg.stdin.write(frame.tobytes())
            frame_stats.append({"frame": index, "time_seconds": float(elapsed_times[index]), "absolute_time_seconds": float(timestamp), "internal_samples": int(len(internal_times)), "gyro_norm_rad_s": gyro, "smoothed_motion_rad_s": smoothed, "gear": gear_index, "actual_rpm": rpm, "dsec_reference_events_50ms": None if np.isnan(dsec_count) else int(dsec_count), "passive_events_50ms": passive_count, "fixed_events_50ms": fixed_count, "imu_gear_events_50ms": gear_count, "passive_coverage": passive_coverage, "fixed_coverage": fixed_coverage, "imu_gear_coverage": gear_coverage, "fixed_raw_flicker": flicker["fixed_raw"], "imu_gear_raw_flicker": flicker["gear_raw"], "fixed_comp_flicker": flicker["fixed_comp"], "imu_gear_comp_flicker": flicker["gear_comp"]})
            previous_time = float(timestamp)
            if index == 0 or (index + 1) % 10 == 0:
                print(f"rendered {index + 1}/{args.frames} (internal samples: {internal_sample_count})", flush=True)
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
        "physical_time_seconds": float(output_times[-1] - simulation_start),
        "playback_speed": args.playback_speed,
        "video_resolution": [video_width, video_height],
        "simulation_resolution": [args.width, args.height],
        "event_window_ms": args.window_ms,
        "phase_step_deg_max": args.phase_step_deg,
        "internal_sample_count": internal_sample_count,
        "display_space": args.display_space,
        "comparison_contract": "Passive, Fixed AMI, and IMU Gear use identical RGB source, camera plane, internal timestamps, threshold, and 50 ms event window. Raw panels show sensor-plane events; compensated panels map only the known prism motion to the reference plane. DSEC is an unaligned reference only.",
        "frames_detail": frame_stats,
    }
    args.output.with_suffix(".json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in metadata.items() if key != "frames_detail"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
