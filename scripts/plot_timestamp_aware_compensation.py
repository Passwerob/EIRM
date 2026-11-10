#!/usr/bin/env python3
"""Supplementary figure: timestamp-aware inverse compensation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from ami_simulator.event_quality import compute_reference_edge_mask
from ami_simulator.geometry import RotatingWedgePrism
from ami_simulator.pipeline import load_rgb_sequence, rgb_to_linear_luminance
from ami_simulator.types import EventStream
from scripts.evaluate_dl3dv_downstream import _ami_windows, _intrinsics
from scripts.plot_eveyes_event_qualitative import _events_to_map


DEFAULT_ROOT = Path("data/dl3dv_large5")
DEFAULT_OUT = Path("results/ami_next/figures/timestamp_aware_compensation")


def _resize_rgb(frames: np.ndarray, width: int) -> np.ndarray:
    if frames.shape[2] == width:
        return frames
    height = int(round(frames.shape[1] * width / frames.shape[2]))
    return np.stack([cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA) for frame in frames])


def _constant_compensation(events: EventStream, prism: RotatingWedgePrism, theta: float) -> EventStream:
    if not len(events):
        return EventStream.empty()
    points = torch.from_numpy(np.stack([events.x, events.y], axis=1).astype(np.float32))
    with torch.inference_mode():
        reference = prism.inverse_points(points, float(theta)).cpu().numpy()
    return EventStream(np.rint(reference[:, 0]), np.rint(reference[:, 1]), events.t, events.p)


def _edge_image(luminance: np.ndarray) -> np.ndarray:
    mask = compute_reference_edge_mask(luminance, edge_quantile=0.72)
    # Keep edge structure black on white so this panel reads as a reference,
    # rather than as another event visualization.
    edge = np.full(mask.shape, 245, dtype=np.uint8)
    edge[mask] = 28
    return edge


def _crop_box(rgb: np.ndarray, edge: np.ndarray, crop_w: int = 104, crop_h: int = 76) -> tuple[int, int, int, int]:
    """Choose a high-frequency window, with a mild preference for the center."""
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    score_map = cv2.GaussianBlur(np.hypot(gx, gy), (0, 0), 2.0)
    h, w = gray.shape
    best = (-np.inf, 0, 0)
    # Avoid crops dominated by image borders while retaining high-frequency
    # architecture, signage, foliage, and object contours.
    for y in range(12, max(13, h - crop_h - 8), 5):
        for x in range(8, max(9, w - crop_w - 8), 5):
            patch_score = float(score_map[y : y + crop_h, x : x + crop_w].mean())
            edge_fraction = float(edge[y : y + crop_h, x : x + crop_w].mean() < 100)
            center_penalty = 0.10 * abs((x + crop_w * 0.5) / w - 0.5)
            score = patch_score + 0.12 * edge_fraction - center_penalty
            if score > best[0]:
                best = (score, x, y)
    _, x, y = best
    return int(x), int(y), int(x + crop_w), int(y + crop_h)


def _events_map(events: EventStream, shape: tuple[int, int]) -> np.ndarray:
    return _events_to_map(events, shape[0], shape[1])


def _draw_panel(ax: plt.Axes, image: np.ndarray, *, border: str = "0.28", lw: float = 0.45) -> None:
    if image.ndim == 2:
        ax.imshow(image, cmap="gray", vmin=0, vmax=255, interpolation="nearest")
    else:
        ax.imshow(image, interpolation="nearest")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(lw)
        spine.set_color(border)


def _draw_crop_panel(ax: plt.Axes, maps: list[np.ndarray], labels: list[str], box: tuple[int, int, int, int]) -> None:
    # A single column containing the same crop for each of the three variants.
    crops = [image[box[1] : box[3], box[0] : box[2]] for image in maps]
    gap = np.full((3, crops[0].shape[1], 3), 235, dtype=np.uint8)
    stacked = np.vstack([crops[0], gap, crops[1], gap, crops[2]])
    _draw_panel(ax, stacked, border="0.28", lw=0.45)
    h = crops[0].shape[0]
    for i, label in enumerate(labels):
        y = i * (h + gap.shape[0]) + 4
        ax.text(3, y, label, color="white", fontsize=5.1, ha="left", va="top", weight="bold",
                bbox=dict(facecolor="black", edgecolor="none", alpha=0.62, pad=1.0))


def _scene_block(scene: Path, hero: int, width: int, rpm: float, angle_deg: float, device: str, crop_override: tuple[int, int, int, int] | None = None) -> dict:
    frames, timestamps = load_rgb_sequence(scene / "images", scene / "timestamps.csv", timestamp_unit="s", max_frames=24)
    timestamps = timestamps[: len(frames)]
    frames = _resize_rgb(np.asarray(frames), width)
    luminance = rgb_to_linear_luminance(frames)
    hero = min(int(hero), len(timestamps) - 2)
    intrinsics = _intrinsics(scene)
    if width != intrinsics.width:
        intrinsics = intrinsics.resized(width, luminance.shape[1])
    prism = RotatingWedgePrism(intrinsics, angle_deg)
    masks = np.ones((len(luminance), luminance.shape[1], luminance.shape[2]), dtype=bool)
    # All three displayed variants originate from this one sensor stream.
    reference, sensor, _ = _ami_windows(luminance, timestamps, masks, prism, rpm, device=device)
    sensor_events = sensor[hero]
    per_event = reference[hero]
    start, end = float(timestamps[hero]), float(timestamps[hero + 1])
    theta_mid = float(rpm) * 2.0 * np.pi / 60.0 * (end - start) * 0.5
    interval = _constant_compensation(sensor_events, prism, theta_mid)
    rgb = np.clip(frames[hero], 0, 255).astype(np.uint8)
    edge = _edge_image(luminance[hero])
    crop = crop_override if crop_override is not None else _crop_box(rgb, edge)
    maps = [_events_map(item, rgb.shape[:2]) for item in (sensor_events, interval, per_event)]
    return {
        "scene": scene.name,
        "rgb": rgb,
        "edge": edge,
        "maps": maps,
        "crop": crop,
        "t0": start,
        "t1": end,
        "counts": [int(len(sensor_events)), int(len(interval)), int(len(per_event))],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--scenes", nargs=2, default=("screen-000083", "screen-000108"))
    parser.add_argument("--hero", type=int, default=12, help="Fallback frame index used for every scene.")
    parser.add_argument("--heroes", nargs="*", type=int, default=None, metavar="FRAME",
                        help="Optional per-scene frame indices.")
    parser.add_argument("--eval-width", type=int, default=480)
    parser.add_argument("--rpm", type=float, default=180.0)
    parser.add_argument("--prism-angle-deg", type=float, default=2.0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--crop-boxes", nargs="*", default=None, metavar="X0,Y0,X1,Y1|auto",
                        help="Optional per-scene crop boxes at eval resolution; use auto for automatic selection.")
    args = parser.parse_args()
    if len(args.scenes) != 2:
        raise SystemExit("Exactly two scenes are required for the two-row supplementary figure.")
    torch.set_num_threads(8)
    if args.heroes is not None and len(args.heroes) not in (0, len(args.scenes)):
        raise SystemExit("--heroes must provide one frame index per scene.")
    heroes = list(args.heroes) if args.heroes else [args.hero] * len(args.scenes)
    crop_boxes = []
    if args.crop_boxes is not None:
        if len(args.crop_boxes) not in (0, len(args.scenes)):
            raise SystemExit("--crop-boxes must provide one X0,Y0,X1,Y1 box per scene.")
        for value in args.crop_boxes:
            if value.lower() == "auto":
                crop_boxes.append(None)
                continue
            try:
                box = tuple(int(part) for part in value.split(","))
            except ValueError as exc:
                raise SystemExit(f"Invalid crop box {value!r}; expected X0,Y0,X1,Y1.") from exc
            if len(box) != 4:
                raise SystemExit(f"Invalid crop box {value!r}; expected X0,Y0,X1,Y1.")
            crop_boxes.append(box)
    if not crop_boxes:
        crop_boxes = [None] * len(args.scenes)
    blocks = [_scene_block(args.dataset_root / name, heroes[index], args.eval_width, args.rpm, args.prism_angle_deg, args.device, crop_boxes[index]) for index, name in enumerate(args.scenes)]

    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans"],
        "font.size": 7.0,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })
    fig = plt.figure(figsize=(11.0, 4.35), dpi=240, facecolor="white")
    fig.suptitle("Timestamp-aware compensation", fontsize=10.5, fontweight="bold", color="0.10", y=0.985)
    grid = fig.add_gridspec(
        2, 6, width_ratios=[1, 1, 1, 1, 1, 1.02],
        left=0.075, right=0.995, top=0.875, bottom=0.075,
        wspace=0.045, hspace=0.23,
    )
    headers = ["RGB reference", "Reference edge", "No compensation", "Interval-level\ncompensation", "Per-event\ncompensation", "Zoomed crop"]
    row_labels = ["Scene A", "Scene B"]
    for row, block in enumerate(blocks):
        images = [block["rgb"], block["edge"], *block["maps"]]
        rgb_ax = None
        for col, image in enumerate(images):
            ax = fig.add_subplot(grid[row, col])
            _draw_panel(ax, image)
            if col == 0:
                rgb_ax = ax
            if row == 0:
                ax.set_title(headers[col], fontsize=7.0, color="0.12", pad=4, linespacing=1.0)
            if col == 0:
                ax.set_ylabel(row_labels[row], fontsize=7.3, fontweight="bold", color="0.12", labelpad=8)
        # Show the crop location on the full RGB reference with a thin red box.
        x0, y0, x1, y1 = block["crop"]
        assert rgb_ax is not None
        rgb_ax.add_patch(plt.Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, edgecolor="#c83e3e", linewidth=0.75))
        ax_crop = fig.add_subplot(grid[row, 5])
        _draw_crop_panel(ax_crop, block["maps"], ["no", "interval", "per-event"], block["crop"])
        if row == 0:
            ax_crop.set_title(headers[5], fontsize=7.0, color="0.12", pad=4)

    # A short reading cue belongs in the caption area, not in a redundant legend.
    fig.text(0.075, 0.023, "Same generated event stream; only the inverse-warp evaluation time changes.  Red box marks the shared high-frequency crop.",
             ha="left", va="bottom", fontsize=6.3, color="0.30")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.03)
    fig.savefig(args.output.with_suffix(".png"), dpi=240, bbox_inches="tight", pad_inches=0.03)
    meta = {
        "title": "Timestamp-aware compensation",
        "scenes": [block["scene"] for block in blocks],
        "hero_frames": heroes,
        "eval_width": int(args.eval_width),
        "rpm": float(args.rpm),
        "prism_angle_deg": float(args.prism_angle_deg),
        "crop_boxes_xyxy": {block["scene"]: list(block["crop"]) for block in blocks},
        "event_counts_no_interval_per_event": {block["scene"]: block["counts"] for block in blocks},
        "outputs": [str(args.output.with_suffix(".png")), str(args.output.with_suffix(".pdf"))],
    }
    args.output.with_suffix(".json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    plt.close(fig)
    print(json.dumps(meta, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
