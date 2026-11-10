#!/usr/bin/env python3
"""Obvious camera-speed exhibit: open loop vs 0.5× keep-360 NFB.

Big speed label and path progress bar. Does not overwrite the dual-NFB mp4.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from scripts.evaluate_eveyes_v2_downstream import _scene_path


def _sample_rgb(frames: np.ndarray, path_s: float) -> np.ndarray:
    last = len(frames) - 1
    path_s = float(np.clip(path_s, 0.0, last))
    index = int(np.floor(path_s))
    next_index = min(index + 1, last)
    alpha = path_s - index
    blended = (1.0 - alpha) * frames[index].astype(np.float32) + alpha * frames[next_index].astype(np.float32)
    return np.clip(blended, 0, 255).astype(np.uint8)


def _bar(image: np.ndarray, frac: float, *, label: str, color: tuple[int, int, int]) -> np.ndarray:
    canvas = image.copy()
    height, width = canvas.shape[:2]
    x0, y0, x1, y1 = 24, height - 42, width - 24, height - 18
    cv2.rectangle(canvas, (x0, y0), (x1, y1), (0, 0, 0), -1)
    fill = x0 + int((x1 - x0) * float(np.clip(frac, 0.0, 1.0)))
    cv2.rectangle(canvas, (x0, y0), (fill, y1), color, -1)
    cv2.rectangle(canvas, (x0, y0), (x1, y1), (255, 255, 255), 1)
    cv2.putText(canvas, label, (x0, y0 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(canvas, label, (x0, y0 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    return canvas


def _hud(image: np.ndarray, title: str, row: dict, path_last: float) -> np.ndarray:
    canvas = image.copy()
    rate = float(row["camera_rate"])
    lines = [
        title,
        f"{rate:.1f}x camera   {row['rpm']:.0f} RPM",
        f"t={row['t0']:.2f}s   path {row['path_s']:.0f}/{path_last:.0f}",
    ]
    y = 36
    for i, line in enumerate(lines):
        scale = 1.05 if i == 0 else 0.72
        cv2.putText(canvas, line, (18, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(canvas, line, (18, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), 2 if i == 0 else 1, cv2.LINE_AA)
        y += 34 if i == 0 else 28
    return _bar(canvas, float(row["path_s"]) / max(path_last, 1.0), label="path progress", color=(40, 180, 255) if rate >= 0.9 else (40, 220, 80))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("results/eveyes_motion_nfb/camera_speed_nfb.json"))
    parser.add_argument("--dataset-root", type=Path, default=Path("data/eveyes_motion_nfb_ready"))
    parser.add_argument("--scene", default="test/Scene_005/co_motion")
    parser.add_argument("--left", default="open_loop_camera")
    parser.add_argument("--right", default="camera_nfb_keep360")
    parser.add_argument("--left-title", default="OPEN LOOP  1.0x")
    parser.add_argument("--right-title", default="CAMERA NFB  0.5x  360 RPM")
    parser.add_argument("--output", type=Path, default=Path("results/eveyes_motion_nfb/figures/camera_nfb_keep360.mp4"))
    parser.add_argument("--fps", type=int, default=30)
    args = parser.parse_args()
    report = json.loads(args.input.read_text(encoding="utf-8"))
    block = report["scenes"][args.scene]
    left_rows = block[args.left]["rows"]
    right_rows = block[args.right]["rows"]
    split, scene, regime = args.scene.split("/")
    image_dir = _scene_path(args.dataset_root, split, scene, regime) / "images"
    frames = np.stack([np.asarray(Image.open(path).convert("RGB")) for path in sorted(image_dir.glob("*.png"))])
    path_last = float(len(frames) - 1)
    n = max(len(left_rows), len(right_rows))
    height, width = frames[0].shape[:2]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(args.output), cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (width * 2, height))
    if not writer.isOpened():
        raise SystemExit(f"failed to open {args.output}")
    for index in range(n):
        left_row = left_rows[min(index, len(left_rows) - 1)]
        right_row = right_rows[min(index, len(right_rows) - 1)]
        left = _hud(_sample_rgb(frames, float(left_row["path_s"])), args.left_title, left_row, path_last)
        right = _hud(_sample_rgb(frames, float(right_row["path_s"])), args.right_title, right_row, path_last)
        writer.write(cv2.cvtColor(np.concatenate((left, right), axis=1), cv2.COLOR_RGB2BGR))
    writer.release()
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
