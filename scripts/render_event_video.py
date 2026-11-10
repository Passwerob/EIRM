#!/usr/bin/env python3
"""Render RGB and one or more event HDF5 streams into a comparison video.

Each event stream is accumulated over the same trailing time window at every
RGB timestamp.  The renderer accepts either a single HDF5 file or a directory
of per-frame HDF5 files (as produced by ``prepare_dsec_scene.py``).
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
import imageio.v2 as imageio
import numpy as np

from ami_simulator.dsec import read_dsec_events_h5
from ami_simulator.events import read_events_h5
from ami_simulator.types import EventStream


def _load_events(path: Path, *, dsec: bool = False) -> EventStream:
    files = sorted(path.glob("*.h5")) if path.is_dir() else [path]
    if not files:
        raise FileNotFoundError(f"No HDF5 files found under {path}")
    streams = [read_dsec_events_h5(file) if dsec else read_events_h5(file) for file in files]
    if len(streams) == 1:
        return streams[0]
    return EventStream(
        np.concatenate([stream.x for stream in streams]),
        np.concatenate([stream.y for stream in streams]),
        np.concatenate([stream.t for stream in streams]),
        np.concatenate([stream.p for stream in streams]),
    ).sorted_by_time()


def _read_times(path: Path, unit: str) -> np.ndarray:
    text = path.read_text(encoding="utf-8").splitlines()
    rows = [row for row in csv.reader(text) if any(field.strip() for field in row)]
    try:
        values = np.asarray([[float(value) for value in row] for row in rows], dtype=np.float64).reshape(-1)
    except ValueError:
        header = [value.strip().lower() for value in rows.pop(0)]
        preferred = {"s": ("timestamp_s", "timestamp", "time_s"), "ms": ("timestamp_ms", "time_ms"), "us": ("timestamp_us", "timestamp", "time_us"), "ns": ("timestamp_ns", "time_ns")}[unit]
        index = next((header.index(name) for name in preferred if name in header), None)
        if index is None:
            index = len(header) - 1
        values = np.asarray([float(row[index]) for row in rows], dtype=np.float64)
    scale = {"s": 1.0, "ms": 1e-3, "us": 1e-6, "ns": 1e-9}[unit]
    return values * scale


def _event_frame(events: EventStream, timestamp: float, window_s: float, height: int, width: int, max_count: int, coordinate_space: str = "raw", splat_mode: str = "nearest") -> np.ndarray:
    start = timestamp - window_s
    left = int(np.searchsorted(events.t, start, side="left"))
    right = int(np.searchsorted(events.t, timestamp, side="right"))
    canvas = np.full((height, width, 3), 18, dtype=np.float32)
    if right <= left:
        return canvas
    if coordinate_space == "reference" and {"x_reference", "y_reference"}.issubset(events.extras):
        x = events.extras["x_reference"][left:right].astype(np.float32, copy=False)
        y = events.extras["y_reference"][left:right].astype(np.float32, copy=False)
    else:
        x = events.x[left:right].astype(np.float32, copy=False)
        y = events.y[left:right].astype(np.float32, copy=False)
    p = events.p[left:right]
    valid = (x >= 0) & (x <= width - 1) & (y >= 0) & (y <= height - 1)
    x, y, p = x[valid], y[valid], p[valid]
    if not len(x):
        return canvas
    positive = np.zeros((height, width), dtype=np.float32)
    negative = np.zeros((height, width), dtype=np.float32)
    if splat_mode == "nearest":
        xi, yi = np.rint(x).astype(np.int32), np.rint(y).astype(np.int32)
        pos = p > 0
        np.add.at(positive, (yi[pos], xi[pos]), 1.0)
        np.add.at(negative, (yi[~pos], xi[~pos]), 1.0)
    else:
        x0, y0 = np.floor(x).astype(np.int32), np.floor(y).astype(np.int32)
        x1, y1 = np.minimum(x0 + 1, width - 1), np.minimum(y0 + 1, height - 1)
        wx, wy = x - x0, y - y0
        pos = p > 0
        for xx, yy, weight in ((x0, y0, (1.0 - wx) * (1.0 - wy)), (x1, y0, wx * (1.0 - wy)), (x0, y1, (1.0 - wx) * wy), (x1, y1, wx * wy)):
            np.add.at(positive, (yy[pos], xx[pos]), weight[pos])
            np.add.at(negative, (yy[~pos], xx[~pos]), weight[~pos])
    # Log-compressed per-polarity counts avoid saturation when a short window
    # contains millions of events while preserving sharp event contours.
    denominator = np.log1p(float(max_count))
    positive_level = np.clip(np.log1p(positive) / denominator, 0.0, 1.0)
    negative_level = np.clip(np.log1p(negative) / denominator, 0.0, 1.0)
    canvas[..., 0] += 225.0 * positive_level + 20.0 * negative_level
    canvas[..., 1] += 105.0 * positive_level + 145.0 * negative_level
    canvas[..., 2] += 20.0 * positive_level + 225.0 * negative_level
    return np.clip(canvas, 0, 255).astype(np.uint8)


def _event_coordinates(events: EventStream, left: int, right: int, width: int, height: int, coordinate_space: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if coordinate_space == "reference" and {"x_reference", "y_reference"}.issubset(events.extras):
        x = events.extras["x_reference"][left:right].astype(np.float32, copy=False)
        y = events.extras["y_reference"][left:right].astype(np.float32, copy=False)
    else:
        x = events.x[left:right].astype(np.float32, copy=False)
        y = events.y[left:right].astype(np.float32, copy=False)
    p = events.p[left:right]
    valid = (x >= 0) & (x <= width - 1) & (y >= 0) & (y <= height - 1)
    return x[valid], y[valid], p[valid]


def _time_surface_frame(events: EventStream, timestamp: float, window_s: float, height: int, width: int, tau_s: float, coordinate_space: str) -> np.ndarray:
    left = int(np.searchsorted(events.t, timestamp - window_s, side="left"))
    right = int(np.searchsorted(events.t, timestamp, side="right"))
    canvas = np.full((height, width, 3), 18.0, dtype=np.float32)
    if right <= left:
        return canvas.astype(np.uint8)
    # Keep each event's timestamp aligned with its selected coordinate.
    if coordinate_space == "reference" and {"x_reference", "y_reference"}.issubset(events.extras):
        ex = events.extras["x_reference"][left:right].astype(np.float32, copy=False)
        ey = events.extras["y_reference"][left:right].astype(np.float32, copy=False)
    else:
        ex = events.x[left:right].astype(np.float32, copy=False)
        ey = events.y[left:right].astype(np.float32, copy=False)
    valid = (ex >= 0) & (ex <= width - 1) & (ey >= 0) & (ey <= height - 1)
    ex, ey, ep, et = ex[valid], ey[valid], events.p[left:right][valid], events.t[left:right][valid]
    exi, eyi = np.rint(ex).astype(np.int32), np.rint(ey).astype(np.int32)
    latest_positive = np.full((height, width), -np.inf, dtype=np.float64)
    latest_negative = np.full((height, width), -np.inf, dtype=np.float64)
    pos = ep > 0
    np.maximum.at(latest_positive, (eyi[pos], exi[pos]), et[pos])
    np.maximum.at(latest_negative, (eyi[~pos], exi[~pos]), et[~pos])
    positive_level = np.where(np.isfinite(latest_positive), np.exp(-(timestamp - latest_positive) / max(tau_s, 1e-9)), 0.0).astype(np.float32)
    negative_level = np.where(np.isfinite(latest_negative), np.exp(-(timestamp - latest_negative) / max(tau_s, 1e-9)), 0.0).astype(np.float32)
    canvas[..., 0] += 225.0 * positive_level + 20.0 * negative_level
    canvas[..., 1] += 105.0 * positive_level + 145.0 * negative_level
    canvas[..., 2] += 20.0 * positive_level + 225.0 * negative_level
    return np.clip(canvas, 0, 255).astype(np.uint8)


def _voxel_grid_frame(events: EventStream, timestamp: float, window_s: float, height: int, width: int, bins: int, max_count: int, coordinate_space: str, splat_mode: str) -> np.ndarray:
    """Render a full-resolution temporal voxel projection with recency weights."""
    left = int(np.searchsorted(events.t, timestamp - window_s, side="left"))
    right = int(np.searchsorted(events.t, timestamp, side="right"))
    if right <= left:
        return np.full((height, width, 3), 18, dtype=np.uint8)
    x, y, p = _event_coordinates(events, left, right, width, height, coordinate_space)
    times = events.t[left:right]
    if coordinate_space == "reference" and {"x_reference", "y_reference"}.issubset(events.extras):
        ex = events.extras["x_reference"][left:right].astype(np.float32, copy=False)
        ey = events.extras["y_reference"][left:right].astype(np.float32, copy=False)
    else:
        ex, ey = events.x[left:right].astype(np.float32), events.y[left:right].astype(np.float32)
    valid = (ex >= 0) & (ex <= width - 1) & (ey >= 0) & (ey <= height - 1)
    ex, ey, ep, et = ex[valid], ey[valid], events.p[left:right][valid], times[valid]
    bin_index = np.minimum(bins - 1, np.floor((et - (timestamp - window_s)) / window_s * bins).astype(np.int32))
    recent_weight = (bin_index.astype(np.float32) + 1.0) / bins
    xi, yi = np.rint(ex).astype(np.int32), np.rint(ey).astype(np.int32)
    positive = np.zeros((height, width), dtype=np.float32)
    negative = np.zeros((height, width), dtype=np.float32)
    pos = ep > 0
    np.add.at(positive, (yi[pos], xi[pos]), recent_weight[pos])
    np.add.at(negative, (yi[~pos], xi[~pos]), recent_weight[~pos])
    denominator = np.log1p(float(max_count))
    positive_level = np.clip(np.log1p(positive) / denominator, 0.0, 1.0)
    negative_level = np.clip(np.log1p(negative) / denominator, 0.0, 1.0)
    canvas = np.full((height, width, 3), 18.0, dtype=np.float32)
    canvas[..., 0] += 225.0 * positive_level + 20.0 * negative_level
    canvas[..., 1] += 105.0 * positive_level + 145.0 * negative_level
    canvas[..., 2] += 20.0 * positive_level + 225.0 * negative_level
    return np.clip(canvas, 0, 255).astype(np.uint8)


def _iwe_frame(events: EventStream, timestamp: float, window_s: float, height: int, width: int, max_count: int, coordinate_space: str, splat_mode: str) -> np.ndarray:
    """Grayscale image of warped events, matching the common IWE visualization."""
    left = int(np.searchsorted(events.t, timestamp - window_s, side="left"))
    right = int(np.searchsorted(events.t, timestamp, side="right"))
    canvas = np.full((height, width, 3), 255.0, dtype=np.float32)
    if right <= left:
        return canvas.astype(np.uint8)
    x, y, _ = _event_coordinates(events, left, right, width, height, coordinate_space)
    if not len(x):
        return canvas.astype(np.uint8)
    xi, yi = np.rint(x).astype(np.int32), np.rint(y).astype(np.int32)
    counts = np.zeros((height, width), dtype=np.float32)
    np.add.at(counts, (yi, xi), 1.0)
    level = np.clip(np.log1p(counts) / np.log1p(float(max_count)), 0.0, 1.0)
    gray = 255.0 - 235.0 * level
    return np.repeat(gray[..., None], 3, axis=-1).astype(np.uint8)


def _label(image: np.ndarray, text: str, detail: str) -> np.ndarray:
    output = image.copy()
    cv2.rectangle(output, (0, 0), (output.shape[1], 34), (8, 8, 8), thickness=-1)
    cv2.putText(output, text, (10, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (245, 245, 245), 1, cv2.LINE_AA)
    cv2.putText(output, detail, (10, output.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (220, 220, 220), 1, cv2.LINE_AA)
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rgb-dir", type=Path, required=True)
    parser.add_argument("--timestamps", type=Path, required=True)
    parser.add_argument("--timestamp-unit", choices=["s", "ms", "us", "ns"], default="us")
    parser.add_argument("--events", type=Path, action="append", required=True, help="Event H5 file or directory; repeat for more panels.")
    parser.add_argument("--labels", nargs="*", default=None, help="Panel labels in the same order as --events.")
    parser.add_argument("--dsec-events", action="store_true", help="Interpret --events as raw DSEC H5 files/directories.")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--window-ms", type=float, default=50.0)
    parser.add_argument("--fps", type=float, default=None)
    parser.add_argument("--max-count", type=int, default=4)
    parser.add_argument("--coordinate-space", choices=["raw", "reference"], default="raw", help="Render raw x/y or per-event x_reference/y_reference coordinates from H5.")
    parser.add_argument("--splat-mode", choices=["nearest", "bilinear"], default="nearest", help="Rasterization for floating-point reference coordinates.")
    parser.add_argument("--representation", choices=["event_map", "time_surface", "voxel_grid", "iwe"], default="event_map", help="Event visualization representation.")
    parser.add_argument("--tau-ms", type=float, default=10.0, help="Time-surface decay constant.")
    parser.add_argument("--voxel-bins", type=int, default=5, help="Number of temporal tiles for voxel-grid visualization.")
    parser.add_argument("--scale", type=float, default=0.75, help="Output spatial scale, keeping source aspect ratio.")
    args = parser.parse_args()
    if args.window_ms <= 0 or args.scale <= 0 or args.max_count <= 0 or args.tau_ms <= 0 or args.voxel_bins < 2:
        raise SystemExit("window-ms, scale, max-count, tau-ms must be positive and voxel-bins >= 2")
    if args.representation == "iwe":
        args.coordinate_space = "reference"
    if args.labels is not None and len(args.labels) not in (0, len(args.events)):
        raise SystemExit("--labels must have one label per --events")
    labels = args.labels if args.labels else [path.stem for path in args.events]
    image_paths = sorted(path for path in args.rgb_dir.iterdir() if path.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"})
    if args.max_frames is not None:
        image_paths = image_paths[: args.max_frames]
    if len(image_paths) < 2:
        raise SystemExit("Need at least two RGB frames")
    timestamps = _read_times(args.timestamps, args.timestamp_unit)[: len(image_paths)]
    streams = [_load_events(path, dsec=args.dsec_events) for path in args.events]
    first = np.asarray(imageio.imread(image_paths[0]))[..., :3]
    height, width = first.shape[:2]
    output_width, output_height = int(width * args.scale), int(height * args.scale)
    panels = []
    for label, stream in zip(labels, streams):
        panels.append((label, stream))
    fps = float(args.fps or (1.0 / np.median(np.diff(timestamps))))
    fps = float(np.clip(fps, 1.0, 60.0))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    video_width = output_width * (1 + len(panels))
    ffmpeg = subprocess.Popen(
        ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-vcodec", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{video_width}x{output_height}", "-r", f"{fps:.6f}", "-i", "-", "-an", "-c:v", "libx264", "-pix_fmt", "yuv420p", str(args.output)],
        stdin=subprocess.PIPE,
    )
    try:
        for index, (image_path, timestamp) in enumerate(zip(image_paths, timestamps)):
            rgb = np.asarray(imageio.imread(image_path))[..., :3]
            rgb = cv2.resize(rgb, (output_width, output_height), interpolation=cv2.INTER_AREA)
            rgb = _label(rgb, "RGB", f"t={timestamp:.6f}s  frame={index:04d}")
            event_panels = []
            for label, stream in panels:
                if args.representation == "event_map":
                    rendered = _event_frame(stream, float(timestamp), args.window_ms * 1e-3, height, width, args.max_count, args.coordinate_space, args.splat_mode)
                elif args.representation == "iwe":
                    rendered = _iwe_frame(stream, float(timestamp), args.window_ms * 1e-3, height, width, args.max_count, "reference", args.splat_mode)
                elif args.representation == "time_surface":
                    rendered = _time_surface_frame(stream, float(timestamp), args.window_ms * 1e-3, height, width, args.tau_ms * 1e-3, args.coordinate_space)
                else:
                    rendered = _voxel_grid_frame(stream, float(timestamp), args.window_ms * 1e-3, height, width, args.voxel_bins, args.max_count, args.coordinate_space, args.splat_mode)
                rendered = cv2.resize(rendered, (output_width, output_height), interpolation=cv2.INTER_NEAREST)
                count = int(np.count_nonzero((stream.t >= timestamp - args.window_ms * 1e-3) & (stream.t <= timestamp)))
                event_panels.append(_label(rendered, label, f"{args.window_ms:g} ms  events={count:,}"))
            frame = np.concatenate([rgb, *event_panels], axis=1)
            assert ffmpeg.stdin is not None
            ffmpeg.stdin.write(np.ascontiguousarray(frame).tobytes())
            if index == 0 or (index + 1) % 10 == 0:
                print(f"rendered {index + 1}/{len(image_paths)} frames", flush=True)
    finally:
        if ffmpeg.stdin is not None:
            ffmpeg.stdin.close()
        return_code = ffmpeg.wait()
        if return_code != 0:
            raise RuntimeError(f"ffmpeg failed with exit code {return_code}")
    metadata = {"rgb_frames": len(image_paths), "fps": fps, "window_ms": args.window_ms, "representation": args.representation, "tau_ms": args.tau_ms, "voxel_bins": args.voxel_bins, "coordinate_space": args.coordinate_space, "splat_mode": args.splat_mode, "timestamp_driven_mapping": args.coordinate_space == "reference", "events": [{"path": str(path), "label": label, "count": len(stream), "has_reference_coordinates": {"x_reference", "y_reference"}.issubset(stream.extras)} for path, label, stream in zip(args.events, labels, streams)]}
    args.output.with_suffix(".json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(args.output), **metadata}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
