#!/usr/bin/env python3
"""Paper qualitative events on object_only: camera frozen, one moving car.

Top row: RGB at 0 / 2 / 6 s, city locked. Bottom row: raw polarity event maps (black background). Storm is an
exaggerated 1080 RPM sensor-plane stack so it reads as a blizzard; 180 stays
inverse-warped.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

from ami_simulator.geometry import RotatingWedgePrism
from ami_simulator.pipeline import load_rgb_sequence, rgb_to_linear_luminance
from ami_simulator.types import EventStream
from scripts.evaluate_dl3dv_downstream import _ami_windows
from scripts.evaluate_dvs_v2e_ours import _read_npz, _slice
from scripts.evaluate_eveyes_v2_downstream import _intrinsics, _scene_path

READY = Path("data/eveyes_comotion_v2_ready")
CACHE = Path("results/ami_next/eveyes_v2_baseline_events")
CACHE_WINDOWS = Path("results/ami_next/eveyes_event_qualitative_windows")
DVS = CACHE / "test_Scene_005_object_only_f480_w960_dvs_voltmeter.npz"
RGB_TIMES = (0.0, 2.0, 6.0)
EVENT_T = 0.5
STORM_RPM = 1080.0
STORM_WINDOWS = 3
MAX_COUNT_GOOD = 18.0
MAX_COUNT_STORM = 5.0
WARMUP = 8


def _frame_index(t_seconds: float) -> int:
    return min(int(round(t_seconds * 60.0)), 478)


def _count_map(y: np.ndarray, x: np.ndarray, height: int, width: int, *, splat: int) -> np.ndarray:
    counts = np.zeros((height, width), dtype=np.float32)
    if len(x) == 0:
        return counts
    np.add.at(counts, (y, x), 1.0)
    if splat <= 0:
        return counts
    padded = np.pad(counts, splat)
    size = 2 * splat + 1
    windows = np.stack(
        [padded[dy : dy + height, dx : dx + width] for dy in range(size) for dx in range(size)],
        axis=0,
    )
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
    if not len(events):
        return canvas.astype(np.uint8)
    x = np.rint(np.asarray(events.x)).astype(np.int32)
    y = np.rint(np.asarray(events.y)).astype(np.int32)
    p = np.asarray(events.p)
    valid = (x >= 0) & (x < width) & (y >= 0) & (y < height)
    x, y, p = x[valid], y[valid], p[valid]
    pos = p > 0
    splat = 2 if storm else 0
    max_count = MAX_COUNT_STORM if storm else MAX_COUNT_GOOD
    positive = _count_map(y[pos], x[pos], height, width, splat=splat)
    negative = _count_map(y[~pos], x[~pos], height, width, splat=splat)
    scale = 1.0 / np.log1p(max_count)
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


def _ami_window(
    scene: Path,
    hero: int,
    rpm: float,
    device: str,
    *,
    plane: str = "reference",
    n_windows: int = 1,
) -> EventStream:
    cache = CACHE_WINDOWS / f"{scene.parent.name}_{scene.name}_f{hero}_rpm{int(rpm)}_{plane}_n{n_windows}.npz"
    if cache.exists():
        return _read_npz(cache)
    start = max(0, hero - max(WARMUP, n_windows - 1))
    frames, timestamps = load_rgb_sequence(scene / "images", scene / "timestamps.csv", timestamp_unit="s", max_frames=hero + 2)
    timestamps = timestamps[: len(frames)]
    luminance = rgb_to_linear_luminance(frames[start:])
    times = timestamps[start:]
    valid = np.ones((len(luminance), luminance.shape[1], luminance.shape[2]), dtype=bool)
    prism = RotatingWedgePrism(_intrinsics(scene), 0.5)
    reference, sensor, _ = _ami_windows(luminance, times, valid, prism, rpm, device=device)
    streams = sensor if plane == "sensor" else reference
    last = hero - start
    first = last - (n_windows - 1)
    events = _concat(list(streams[first : last + 1]))
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, x=np.asarray(events.x), y=np.asarray(events.y), t=np.asarray(events.t), p=np.asarray(events.p))
    return events


def _panel(axis, image: np.ndarray) -> None:
    axis.imshow(image)
    axis.set_xticks([])
    axis.set_yticks([])
    for spine in axis.spines.values():
        spine.set_linewidth(0.4)
        spine.set_color("0.35")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=READY)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", type=Path, default=Path("results/ami_next/figures/eveyes_event_qualitative"))
    args = parser.parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit("cuda requested but unavailable")
    torch.set_num_threads(8)
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans"],
            "font.size": 8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    scene = _scene_path(args.dataset_root, "test", "Scene_005", "object_only")
    hero = _frame_index(EVENT_T)
    frames, timestamps = load_rgb_sequence(scene / "images", scene / "timestamps.csv", timestamp_unit="s", max_frames=hero + 2)
    timestamps = timestamps[: len(frames)]
    rgb = np.asarray(Image.open(scene / "images" / f"{hero:06d}.png").convert("RGB"))
    height, width = rgb.shape[:2]
    stills = [
        np.asarray(Image.open(scene / "images" / f"{_frame_index(t):06d}.png").convert("RGB"))
        for t in RGB_TIMES
    ]
    idle = _slice(_read_npz(DVS), float(timestamps[hero]), float(timestamps[hero + 1]))
    print(f"AMI 180 / {int(STORM_RPM)} sensor  object_only  t={EVENT_T:.2f}  window={hero}", flush=True)
    storm = _ami_window(scene, hero, STORM_RPM, args.device, plane="sensor", n_windows=STORM_WINDOWS)
    good = _ami_window(scene, hero, 180.0, args.device, plane="reference", n_windows=1)
    event_maps = (
        ("Idle (DVS)", _events_to_map(idle, height, width), int(len(idle))),
        ("too fast", _events_to_map(storm, height, width, storm=True), int(len(storm))),
        ("180 RPM (Ours)", _events_to_map(good, height, width), int(len(good))),
    )
    stats = {
        "regime": "object_only",
        "rgb_times": list(RGB_TIMES),
        "event_t": EVENT_T,
        "window": hero,
        "idle": int(len(idle)),
        "storm": int(len(storm)),
        "good": int(len(good)),
        "storm_rpm": STORM_RPM,
        "storm_plane": "sensor",
        "storm_windows": STORM_WINDOWS,
    }
    print(json.dumps(stats, sort_keys=True), flush=True)

    fig = plt.figure(figsize=(7.15, 3.55), dpi=220)
    grid = fig.add_gridspec(2, 3, wspace=0.04, hspace=0.22, left=0.07, right=0.995, top=0.88, bottom=0.08)
    fig.suptitle("object only  ·  camera frozen, one moving car", fontsize=9.0, y=0.98, color="0.15")
    for col, (t_seconds, image) in enumerate(zip(RGB_TIMES, stills)):
        axis = fig.add_subplot(grid[0, col])
        _panel(axis, image)
        axis.set_title(f"{t_seconds:.1f} s", fontsize=8.0, pad=3)
        if col == 0:
            axis.set_ylabel("RGB", fontsize=7.6, labelpad=4, color="0.15")
    for col, (title, image, count) in enumerate(event_maps):
        axis = fig.add_subplot(grid[1, col])
        _panel(axis, image)
        axis.set_title(title, fontsize=8.0, pad=3)
        if col == 1:
            axis.set_xlabel(f"{count:,} events  ·  {int(STORM_RPM)} RPM", fontsize=6.5, labelpad=1.2, color="0.35")
        else:
            axis.set_xlabel(f"{count:,} events  ·  {EVENT_T:.1f} s", fontsize=6.5, labelpad=1.2, color="0.35")
        if col == 0:
            axis.set_ylabel("events", fontsize=7.6, labelpad=4, color="0.15")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    pdf = args.output.with_suffix(".pdf")
    png = args.output.with_suffix(".png")
    fig.savefig(pdf, bbox_inches="tight", pad_inches=0.03)
    fig.savefig(png, bbox_inches="tight", pad_inches=0.03, dpi=220)
    plt.close(fig)
    sidecar = args.output.with_suffix(".json")
    sidecar.write_text(
        json.dumps(
            {
                "background": "black polarity frame, no RGB underlay",
                "storm": "1080 RPM, sensor plane, 3 windows, saturated vis",
                "max_count_good": MAX_COUNT_GOOD,
                "max_count_storm": MAX_COUNT_STORM,
                "warp": "reference plane",
                "panels": [stats],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(json.dumps({"pdf": str(pdf), "png": str(png), "json": str(sidecar)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
