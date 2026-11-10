#!/usr/bin/env python3
"""Scene_001 2x2: too little motion vs too much motion, RGB and events.

Too little is the real clip: camera frozen, sharp RGB, sparse DVS (car only).
Too much is a synthesized camera sweep on the same frame: RGB motion blur,
noisy events from the same trajectory. Qualitative exhibit, not Table 1.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from ami_simulator.events import ThresholdEventSimulator
from ami_simulator.types import EventStream
from scripts.evaluate_dvs_v2e_ours import _read_npz, _slice
from scripts.plot_eveyes_scene001_dvs_vs_ours import CACHE, _load
from scripts.render_eveyes_scene001_storm_vs_ours import SCENE

HERO_T = 0.0
BLUR_SAMPLES = 17
BLUR_PX = 32.0
BLUR_DEG = -8.0
SHAKE_ROT_DEG = 0.6
EVENT_STEPS = 4
EVENT_THRESHOLD = 0.25
MAX_COUNT_STORM = 22.0


def _count_map(y: np.ndarray, x: np.ndarray, height: int, width: int, *, splat: int) -> np.ndarray:
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


def _events_to_map(events: EventStream, height: int, width: int, *, storm: bool) -> np.ndarray:
    canvas = np.zeros((height, width, 3), dtype=np.float32)
    if not len(events.x):
        return canvas.astype(np.uint8)
    x = np.rint(np.asarray(events.x)).astype(np.int32)
    y = np.rint(np.asarray(events.y)).astype(np.int32)
    p = np.asarray(events.p)
    valid = (x >= 0) & (x < width) & (y >= 0) & (y < height)
    x, y, p = x[valid], y[valid], p[valid]
    pos = p > 0
    splat = 1
    max_count = MAX_COUNT_STORM if storm else 18.0
    scale = 1.0 / np.log1p(max_count)
    positive = _count_map(y[pos], x[pos], height, width, splat=splat)
    negative = _count_map(y[~pos], x[~pos], height, width, splat=splat)
    on = np.clip(np.log1p(positive) * scale, 0.0, 1.0)
    off = np.clip(np.log1p(negative) * scale, 0.0, 1.0)
    if storm:
        density = np.clip(np.log1p(positive + negative) * scale, 0.0, 1.0)
        canvas[..., 0] = 255.0 * np.clip(on + 0.35 * density, 0.0, 1.0)
        canvas[..., 1] = 255.0 * np.clip(0.12 * density, 0.0, 1.0)
        canvas[..., 2] = 255.0 * np.clip(off + 0.35 * density, 0.0, 1.0)
    else:
        on = np.where(positive > 0, np.maximum(on, 0.75), 0.0)
        off = np.where(negative > 0, np.maximum(off, 0.75), 0.0)
        canvas[..., 0] = 255.0 * on
        canvas[..., 2] = 255.0 * off
    return canvas.astype(np.uint8)


def _warp(image: np.ndarray, dx: float, dy: float, angle: float) -> np.ndarray:
    height, width = image.shape[:2]
    matrix = cv2.getRotationMatrix2D((width / 2.0, height / 2.0), float(angle), 1.0)
    matrix[0, 2] += float(dx)
    matrix[1, 2] += float(dy)
    return cv2.warpAffine(image, matrix, (width, height), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)


def _trajectory(n: int, amplitude: float, angle_deg: float, rot_deg: float) -> np.ndarray:
    span = np.linspace(-0.5, 0.5, n, dtype=np.float64)
    rad = np.deg2rad(angle_deg)
    dx = span * amplitude * np.cos(rad)
    dy = span * amplitude * np.sin(rad)
    dtheta = span * rot_deg
    return np.stack((dx, dy, dtheta), axis=1)


def _motion_blur_rgb(rgb: np.ndarray) -> np.ndarray:
    poses = _trajectory(BLUR_SAMPLES, BLUR_PX, BLUR_DEG, SHAKE_ROT_DEG)
    acc = np.zeros(rgb.shape, dtype=np.float32)
    for dx, dy, angle in poses:
        acc += _warp(rgb, dx, dy, angle).astype(np.float32)
    return np.clip(acc / len(poses), 0, 255).astype(np.uint8)


def _motion_events(luminance: np.ndarray) -> EventStream:
    poses = _trajectory(EVENT_STEPS, BLUR_PX, BLUR_DEG, SHAKE_ROT_DEG)
    frames = np.stack([_warp(luminance, dx, dy, angle) for dx, dy, angle in poses], axis=0)
    frames = np.clip(frames, 1e-6, 1.0).astype(np.float32)
    # ~2 ms between warps: a fast camera sweep, not a 60 Hz take.
    times = np.arange(EVENT_STEPS, dtype=np.float64) * 0.002
    return ThresholdEventSimulator(contrast_threshold=EVENT_THRESHOLD).simulate(frames, times)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", type=Path, default=SCENE)
    parser.add_argument("--t", type=float, default=HERO_T)
    parser.add_argument("--max-frames", type=int, default=8)
    parser.add_argument("--output", type=Path, default=Path("results/eveyes_scene001/figures/scene001_too_little_too_much"))
    parser.add_argument("--dump-dir", type=Path, default=Path("results/eveyes_scene001/figures/motion_regimes"))
    args = parser.parse_args()
    rgb, timestamps, luminance = _load(args.scene, args.max_frames)
    hero = int(np.argmin(np.abs(timestamps - float(args.t))))
    hero = min(hero, len(rgb) - 2)
    height, width = rgb[hero].shape[:2]
    dvs = _slice(_read_npz(CACHE), float(timestamps[hero]), float(timestamps[hero + 1]))
    little_rgb = rgb[hero]
    little_evt = _events_to_map(dvs, height, width, storm=False)
    much_rgb = _motion_blur_rgb(little_rgb)
    much_stream = _motion_events(luminance[hero])
    much_evt = _events_to_map(much_stream, height, width, storm=True)
    print(
        {
            "t": float(timestamps[hero]),
            "sparse_dvs": int(len(dvs.x)),
            "noisy_events": int(len(much_stream.x)),
        },
        flush=True,
    )
    args.dump_dir.mkdir(parents=True, exist_ok=True)
    panels = {
        "too_little_rgb": little_rgb,
        "too_little_events": little_evt,
        "too_much_rgb": much_rgb,
        "too_much_events": much_evt,
    }
    for name, image in panels.items():
        Image.fromarray(image).save(args.dump_dir / f"{name}.png")
    fig, axes = plt.subplots(2, 2, figsize=(10.4, 5.8), dpi=140)
    images = ((little_rgb, much_rgb), (little_evt, much_evt))
    for row in range(2):
        for col in range(2):
            axis = axes[row, col]
            axis.imshow(images[row][col])
            axis.set_xticks([])
            axis.set_yticks([])
            if row == 0:
                axis.set_title(("Too little motion", "Too much motion")[col], fontsize=11, pad=6)
            if col == 0:
                axis.set_ylabel(("RGB", "Events")[row], fontsize=10)
            for spine in axis.spines.values():
                spine.set_visible(False)
    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output.with_suffix(".pdf"))
    fig.savefig(args.output.with_suffix(".png"), dpi=170)
    plt.close(fig)
    print(args.output.with_suffix(".png"), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
