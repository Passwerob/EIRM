#!/usr/bin/env python3
"""2x2 video: RGB + inverse-warped polarity events, open-loop vs camera NFB.

Replays logged path_s / camera_rate / rpm from camera_speed_nfb.json. Does not
overwrite m0 or the RGB-only mp4.
"""

from __future__ import annotations

import argparse
import json
from collections import deque
from pathlib import Path

import cv2
import numpy as np
import torch

from ami_simulator.geometry import RotatingWedgePrism
from ami_simulator.pipeline import load_rgb_sequence, rgb_to_linear_luminance
from ami_simulator.types import EventStream
from scripts.evaluate_eveyes_camera_speed_nfb import DT, OBJECTIVE, _sample
from scripts.evaluate_eveyes_v2_downstream import _intrinsics, _scene_path
from scripts.evaluate_phase_continuous_control import _simulate_action

MAX_COUNT_GOOD = 18.0
MAX_COUNT_STORM = 5.0


def _count_map(y: np.ndarray, x: np.ndarray, height: int, width: int, *, splat: int = 0) -> np.ndarray:
    counts = np.zeros((height, width), dtype=np.float32)
    if len(x) == 0:
        return counts
    np.add.at(counts, (y, x), 1.0)
    if splat <= 0:
        return counts
    padded = np.pad(counts, splat)
    size = 2 * splat + 1
    windows = np.stack([padded[dy : dy + height, dx : dx + width] for dy in range(size) for dx in range(size)], axis=0)
    return windows.max(axis=0)


def _concat(streams: list[EventStream]) -> EventStream:
    if not streams:
        return EventStream.empty()
    return EventStream(
        np.concatenate([np.asarray(stream.x) for stream in streams]),
        np.concatenate([np.asarray(stream.y) for stream in streams]),
        np.concatenate([np.asarray(stream.t) for stream in streams]),
        np.concatenate([np.asarray(stream.p) for stream in streams]),
    )


def _events_to_map(events: EventStream, height: int, width: int, *, storm: bool = False) -> np.ndarray:
    canvas = np.zeros((height, width, 3), dtype=np.float32)
    if not len(events.x):
        return canvas.astype(np.uint8)
    x = np.rint(np.asarray(events.x)).astype(np.int32)
    y = np.rint(np.asarray(events.y)).astype(np.int32)
    p = np.asarray(events.p)
    valid = (x >= 0) & (x < width) & (y >= 0) & (y < height)
    x, y, p = x[valid], y[valid], p[valid]
    pos = p > 0
    splat = 2 if storm else 0
    max_count = MAX_COUNT_STORM if storm else MAX_COUNT_GOOD
    scale = 1.0 / np.log1p(max_count)
    positive = _count_map(y[pos], x[pos], height, width, splat=splat)
    negative = _count_map(y[~pos], x[~pos], height, width, splat=splat)
    on = np.clip(np.log1p(positive) * scale, 0.0, 1.0)
    off = np.clip(np.log1p(negative) * scale, 0.0, 1.0)
    if storm:
        density = np.clip(np.log1p(positive + negative) * scale, 0.0, 1.0)
        canvas[..., 0] = 255.0 * np.clip(on + 0.9 * density, 0.0, 1.0)
        canvas[..., 1] = 255.0 * np.clip(0.2 * on + 0.2 * off + 0.85 * density, 0.0, 1.0)
        canvas[..., 2] = 255.0 * np.clip(off + 0.9 * density, 0.0, 1.0)
    else:
        on = np.where(positive > 0, np.maximum(on, 0.7), 0.0)
        off = np.where(negative > 0, np.maximum(off, 0.7), 0.0)
        canvas[..., 0] = 255.0 * on
        canvas[..., 2] = 255.0 * off
    return canvas.astype(np.uint8)


def _hud(image: np.ndarray, lines: list[str]) -> np.ndarray:
    canvas = image.copy()
    y = 28
    for line in lines:
        cv2.putText(canvas, line, (16, y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(canvas, line, (16, y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 1, cv2.LINE_AA)
        y += 26
    return canvas


def _sample_rgb(frames: np.ndarray, path_s: float) -> np.ndarray:
    last = len(frames) - 1
    path_s = float(np.clip(path_s, 0.0, last))
    index = int(np.floor(path_s))
    next_index = min(index + 1, last)
    alpha = path_s - index
    blended = (1.0 - alpha) * frames[index].astype(np.float32) + alpha * frames[next_index].astype(np.float32)
    return np.clip(blended, 0, 255).astype(np.uint8)


class _Replay:
    def __init__(
        self,
        luminance: np.ndarray,
        prism: RotatingWedgePrism,
        rows: list[dict],
        *,
        rpm_fallback: float,
        device: str,
        plane: str = "reference",
        storm: bool = False,
        stack: int = 1,
        rpm_override: float | None = None,
    ):
        self.luminance = luminance
        self.prism = prism
        self.rows = rows
        self.device = device
        self.plane = plane
        self.storm = storm
        self.rpm_override = rpm_override
        self.buffer: deque[EventStream] = deque(maxlen=max(1, stack))
        self.phase = 0.0
        self.previous_active = None
        self.sensor_reference = None
        self.previous_rpm = float(rpm_override if rpm_override is not None else (rows[0]["rpm"] if rows else rpm_fallback))
        self.index = 0
        self.last_event_map = None
        self.last_row = rows[-1] if rows else {}

    def step(self) -> tuple[np.ndarray, EventStream | None, dict]:
        if self.index >= len(self.rows):
            return self.last_event_map, None, self.last_row
        row = self.rows[self.index]
        path_s = float(row["path_s"])
        rate = float(row["camera_rate"])
        rpm = float(self.rpm_override if self.rpm_override is not None else row["rpm"])
        lum_a = _sample(self.luminance, path_s)
        lum_b = _sample(self.luminance, path_s + rate)
        pair = np.stack((lum_a, lum_b), axis=0)
        now = float(row["t0"])
        times = np.asarray((now, now + DT), dtype=np.float64)
        valid = np.ones((2, pair.shape[1], pair.shape[2]), dtype=bool)
        result = _simulate_action(
            pair,
            times,
            valid,
            self.prism,
            index=0,
            rpm=rpm,
            phase=self.phase,
            previous_active=self.previous_active,
            previous_rpm=self.previous_rpm,
            target_density=0.2,
            objective=OBJECTIVE,
            contrast_threshold=0.2,
            max_phase_step_deg=30.0,
            sensor_reference_log=self.sensor_reference,
            device=self.device,
        )
        events = result["sensor_events"] if self.plane == "sensor" else result["reference_events"]
        self.buffer.append(events)
        stacked = _concat(list(self.buffer))
        event_map = _events_to_map(stacked, pair.shape[1], pair.shape[2], storm=self.storm)
        self.phase = float(result["phase_end"])
        self.sensor_reference = np.asarray(result["sensor_reference_log"])
        self.previous_active = result["active"]
        self.previous_rpm = rpm
        self.index += 1
        row = {**row, "event_count_now": int(len(stacked.x)), "rpm_draw": rpm}
        self.last_event_map = event_map
        self.last_row = row
        return event_map, stacked, row


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("results/eveyes_motion_nfb/camera_speed_nfb.json"))
    parser.add_argument("--dataset-root", type=Path, default=Path("data/eveyes_motion_nfb_ready"))
    parser.add_argument("--scene", default="test/Scene_005/co_motion")
    parser.add_argument("--left", default="open_loop_camera")
    parser.add_argument("--right", default="camera_speed_nfb")
    parser.add_argument("--output", type=Path, default=Path("results/eveyes_motion_nfb/figures/camera_nfb_storm_vs_ours.mp4"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--max-frames", type=int, default=480)
    parser.add_argument("--left-plane", default="sensor", choices=("sensor", "reference"))
    parser.add_argument("--right-plane", default="reference", choices=("sensor", "reference"))
    parser.add_argument("--left-rpm", type=float, default=1080.0)
    parser.add_argument("--left-stack", type=int, default=3)
    parser.add_argument("--storm-left", action="store_true", default=True)
    args = parser.parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit("cuda requested but unavailable")
    report = json.loads(args.input.read_text(encoding="utf-8"))
    block = report["scenes"][args.scene]
    open_rows = block[args.left]["rows"]
    nfb_rows = block[args.right]["rows"]
    split, scene, regime = args.scene.split("/")
    scene_path = _scene_path(args.dataset_root, split, scene, regime)
    frames, _timestamps = load_rgb_sequence(scene_path / "images", scene_path / "timestamps.csv", timestamp_unit="s", max_frames=args.max_frames)
    rgb = np.clip(frames, 0, 255).astype(np.uint8)
    luminance = rgb_to_linear_luminance(frames)
    prism = RotatingWedgePrism(_intrinsics(scene_path), 0.5)
    height, width = rgb[0].shape[:2]
    open_replay = _Replay(
        luminance,
        prism,
        open_rows,
        rpm_fallback=360.0,
        device=args.device,
        plane=args.left_plane,
        storm=bool(args.storm_left),
        stack=args.left_stack,
        rpm_override=args.left_rpm,
    )
    nfb_replay = _Replay(
        luminance,
        prism,
        nfb_rows,
        rpm_fallback=180.0,
        device=args.device,
        plane=args.right_plane,
        storm=False,
        stack=1,
        rpm_override=None,
    )
    n = max(len(open_rows), len(nfb_rows))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    raw = args.output.with_name(args.output.stem + "_raw.mp4")
    writer = cv2.VideoWriter(str(raw), cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (width * 2, height * 2))
    if not writer.isOpened():
        raise SystemExit(f"failed to open {raw}")
    open_event_map = nfb_event_map = None
    open_row = nfb_row = None
    for index in range(n):
        if index < len(open_rows):
            open_event_map, _events, open_row = open_replay.step()
        if index < len(nfb_rows):
            nfb_event_map, _events, nfb_row = nfb_replay.step()
        open_rgb = _hud(
            _sample_rgb(rgb, float(open_row["path_s"])),
            [
                "open loop RGB",
                f"t={open_row['t0']:.2f}s  rate={open_row['camera_rate']:.1f}  {open_row['rpm']:.0f} RPM",
                f"path={open_row['path_s']:.0f}",
            ],
        )
        nfb_rgb = _hud(
            _sample_rgb(rgb, float(nfb_row["path_s"])),
            [
                "ours RGB",
                f"t={nfb_row['t0']:.2f}s  rate={nfb_row['camera_rate']:.1f}  {nfb_row['rpm']:.0f} RPM",
                f"path={nfb_row['path_s']:.0f}",
            ],
        )
        open_evt = _hud(
            open_event_map,
            [
                "STORM  sensor plane",
                f"{open_row.get('rpm_draw', args.left_rpm):.0f} RPM  {open_row.get('event_count_now', open_row['event_count']):.0f} events",
            ],
        )
        nfb_evt = _hud(
            nfb_event_map,
            [
                "OURS  inverse-warped",
                f"{nfb_row['rpm']:.0f} RPM  {nfb_row.get('event_count_now', nfb_row['event_count']):.0f} events",
            ],
        )
        frame = np.concatenate(
            (np.concatenate((open_rgb, nfb_rgb), axis=1), np.concatenate((open_evt, nfb_evt), axis=1)),
            axis=0,
        )
        writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        if index % 40 == 0:
            print(f"  frame={index}/{n} open_rpm={open_row['rpm']:.0f} nfb_rpm={nfb_row['rpm']:.0f} nfb_path={nfb_row['path_s']:.1f}", flush=True)
    writer.release()
    print(raw, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
