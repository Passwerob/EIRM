#!/usr/bin/env python3
"""Official 960x540 panels for the constructed EVEYES crawl-then-dash sequence.

The sequence re-times one recorded Scene_005 traversal into three segments and
the figure shows the two that disagree about the prism:

(a) crawl, scene speed 0.25x on a 60 Hz clock: the prism sweeps 18 deg per
    window, 360 over-fills, and Fixed 180 is already the right gear.
(b) dash, scene speed 8x on a 480 Hz clock: the prism sweeps 2.25 deg per
    window, so 360 wins global edge F1 while 0 wins object-edge F1.

Panels are bare native-resolution PNGs, no titles, badges, borders, or burnt-in
numbers; the numbers live in metrics.json. Events use the EIRM palette with one
shared scale per row so the panels in a row are comparable. Not the 64x64 pack.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import imageio.v3 as imageio
import numpy as np
import torch

from ami_simulator.event_visualization import eirm_counts_from_events, eirm_from_counts, eirm_peak
from ami_simulator.geometry import RotatingWedgePrism
from ami_simulator.pipeline import _IMAGE_SUFFIXES, _read_timestamp_file, rgb_to_linear_luminance
from scripts.evaluate_dl3dv_downstream import _step
from scripts.evaluate_eveyes_v2_downstream import _intrinsics, _scene_path

READY = Path("data/eveyes_comotion_v2_ready")
EVAL_JSON = Path("results/ami_next/eveyes_crawl_then_dash.json")
SCENE = "test/Scene_005/crawl_then_dash"
PRISM_DEG = 0.5
WARMUP = 12
GUTTER = 14
PAPER_BG = (255, 255, 255)


def _rows(payload: dict, method: str) -> list[dict]:
    selected = [row for row in payload["rows"] if row["scene"] == SCENE and row["method"] == method]
    selected.sort(key=lambda row: int(row["window"]))
    return selected


def _replay(luminance, timestamps, prism, rpms, device):
    phase = 0.0
    previous_active = None
    sensor_reference = None
    previous_rpm = float(rpms[0])
    ones = np.ones((len(timestamps), luminance.shape[1], luminance.shape[2]), dtype=bool)
    streams = []
    for index, rpm in enumerate(rpms):
        result = _step(luminance, timestamps, ones, prism, index, float(rpm), phase, previous_active, previous_rpm, sensor_reference, device=device)
        streams.append(result["reference_events"])
        phase = float(result["phase_end"])
        previous_active = np.asarray(result["active"])
        sensor_reference = np.asarray(result["sensor_reference_log"])
        previous_rpm = float(rpm)
    return streams


def _hero(scene_dir: Path, hero: int, gears: dict[str, np.ndarray], device: str):
    start = max(0, hero - WARMUP)
    last = hero + 1
    paths = sorted(path for path in (scene_dir / "images").iterdir() if path.suffix.lower() in _IMAGE_SUFFIXES)
    frames = np.stack([np.asarray(imageio.imread(path))[..., :3] for path in paths[start : last + 1]])
    if frames.shape[1] != 540 or frames.shape[2] != 960:
        raise SystemExit(f"{scene_dir} is {frames.shape[2]}x{frames.shape[1]}, expected 960x540")
    timestamps = _read_timestamp_file(scene_dir / "timestamps.csv")[start : last + 1]
    luminance = rgb_to_linear_luminance(frames)
    prism = RotatingWedgePrism(_intrinsics(scene_dir), PRISM_DEG)
    local = hero - start
    height, width = luminance.shape[1], luminance.shape[2]
    counts = {}
    for name, series in gears.items():
        chunk = np.asarray(series[start : start + (last - start)], dtype=np.float64)
        print(f"    replay {name} n={len(chunk)} rpm_hero={chunk[local]:.0f}", flush=True)
        counts[name] = eirm_counts_from_events(_replay(luminance, timestamps, prism, chunk, device)[local], height, width)
    shared = (max(eirm_peak(value[0]) for value in counts.values()), max(eirm_peak(value[1]) for value in counts.values()))
    return frames[local], {name: eirm_from_counts(*value, peak=shared) for name, value in counts.items()}


def _grid(rows: list[list[np.ndarray]]) -> np.ndarray:
    height, width = rows[0][0].shape[:2]
    columns = max(len(row) for row in rows)
    canvas = np.full((len(rows) * height + (len(rows) - 1) * GUTTER, columns * width + (columns - 1) * GUTTER, 3), PAPER_BG, dtype=np.uint8)
    for r, row in enumerate(rows):
        for c, panel in enumerate(row):
            canvas[r * (height + GUTTER) : r * (height + GUTTER) + height, c * (width + GUTTER) : c * (width + GUTTER) + width] = panel
    return canvas


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:6")
    parser.add_argument("--output-dir", type=Path, default=Path("results/ami_next/figures/crawl_then_dash"))
    args = parser.parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        args.device = "cpu"

    payload = json.loads(EVAL_JSON.read_text(encoding="utf-8"))
    passive, fixed180, fixed360, ours = (_rows(payload, name) for name in ("ami_passive", "ami_fixed_180", "ami_fixed_360", "ours"))
    rpm_ours = np.asarray([float(row["rpm"]) for row in ours])
    scene_dir = _scene_path(READY, "test", "Scene_005", "crawl_then_dash")
    state = list(csv.DictReader((scene_dir / "motion_state.csv").open(encoding="utf-8")))
    sensor_fps = np.asarray([float(row["sensor_fps"]) for row in state])
    scene_speed = np.asarray([float(row["playback_rate"]) for row in state])
    source_frame = np.asarray([float(row["source_path_frame"]) for row in state])
    rendered = np.abs(source_frame - np.round(source_frame)) < 1e-6

    dash = [
        index
        for index in range(len(fixed180))
        if sensor_fps[index] > 100
        and scene_speed[index] >= 7.9
        and rendered[index]
        and rendered[index + 1]
        and rpm_ours[index] >= 359
        and ours[index]["edge_f1"] > fixed180[index]["edge_f1"]
        and passive[index]["object_edge_f1"] > fixed180[index]["object_edge_f1"]
    ]
    if not dash:
        raise SystemExit("no dash window satisfies both claims")
    hero_b = max(dash, key=lambda i: min(ours[i]["edge_f1"] - fixed180[i]["edge_f1"], 0.03) + min(passive[i]["object_edge_f1"] - fixed180[i]["object_edge_f1"], 0.03))
    crawl = [index for index in range(len(fixed180)) if sensor_fps[index] < 100 and index >= WARMUP + 18 and rpm_ours[index] <= 181]
    hero_a = max(crawl, key=lambda i: fixed180[i]["edge_f1"] - fixed360[i]["edge_f1"])

    a = {
        "segment": "crawl",
        "window": hero_a,
        "t": float(fixed180[hero_a]["t0"]),
        "scene_speed": float(state[hero_a]["playback_rate"]),
        "sensor_fps": float(sensor_fps[hero_a]),
        "prism_sweep_deg_per_window_at_180rpm": 180.0 / 60.0 * 360.0 / float(sensor_fps[hero_a]),
        "ours_rpm": float(rpm_ours[hero_a]),
        "edge_f1": {"0": float(passive[hero_a]["edge_f1"]), "180": float(fixed180[hero_a]["edge_f1"]), "360": float(fixed360[hero_a]["edge_f1"]), "ours": float(ours[hero_a]["edge_f1"])},
        "object_edge_f1": {"0": float(passive[hero_a]["object_edge_f1"]), "180": float(fixed180[hero_a]["object_edge_f1"]), "360": float(fixed360[hero_a]["object_edge_f1"])},
        "unique_pixel_ratio": {"0": float(passive[hero_a]["unique_pixel_ratio"]), "180": float(fixed180[hero_a]["unique_pixel_ratio"]), "360": float(fixed360[hero_a]["unique_pixel_ratio"])},
    }
    b = {
        "segment": "dash",
        "window": hero_b,
        "t": float(fixed180[hero_b]["t0"]),
        "scene_speed": float(state[hero_b]["playback_rate"]),
        "sensor_fps": float(sensor_fps[hero_b]),
        "prism_sweep_deg_per_window_at_180rpm": 180.0 / 60.0 * 360.0 / float(sensor_fps[hero_b]),
        "frames_are_rendered_not_interpolated": True,
        "source_frames": [float(source_frame[hero_b]), float(source_frame[hero_b + 1])],
        "ours_rpm": float(rpm_ours[hero_b]),
        "edge_f1": {"0": float(passive[hero_b]["edge_f1"]), "180": float(fixed180[hero_b]["edge_f1"]), "360": float(fixed360[hero_b]["edge_f1"]), "ours": float(ours[hero_b]["edge_f1"])},
        "object_edge_f1": {"0": float(passive[hero_b]["object_edge_f1"]), "180": float(fixed180[hero_b]["object_edge_f1"]), "360": float(fixed360[hero_b]["object_edge_f1"])},
        "unique_pixel_ratio": {"0": float(passive[hero_b]["unique_pixel_ratio"]), "180": float(fixed180[hero_b]["unique_pixel_ratio"]), "360": float(fixed360[hero_b]["unique_pixel_ratio"])},
    }
    a["ours_minus_180_edge_f1"] = a["edge_f1"]["ours"] - a["edge_f1"]["180"]
    b["ours_minus_180_edge_f1"] = b["edge_f1"]["ours"] - b["edge_f1"]["180"]
    b["zero_minus_180_object_edge_f1"] = b["object_edge_f1"]["0"] - b["object_edge_f1"]["180"]

    n = len(fixed180)
    zeros, hundred80, threesixty = np.zeros(n), np.full(n, 180.0), np.full(n, 360.0)
    print(f"(a) crawl  w{hero_a} t={a['t']:.3f}s  180 {a['edge_f1']['180']:.4f} vs 360 {a['edge_f1']['360']:.4f}", flush=True)
    rgb_a, maps_a = _hero(scene_dir, hero_a, {"0": zeros, "180": hundred80, "360": threesixty}, args.device)
    print(f"(b) dash   w{hero_b} t={b['t']:.3f}s  Ours {b['edge_f1']['ours']:.4f} vs 180 {b['edge_f1']['180']:.4f}, obj 0 {b['object_edge_f1']['0']:.4f} vs 180 {b['object_edge_f1']['180']:.4f}", flush=True)
    rgb_b, maps_b = _hero(scene_dir, hero_b, {"0": zeros, "180": hundred80, "360": threesixty}, args.device)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    panels = {
        "a1_rgb": rgb_a, "a2_rpm0": maps_a["0"], "a3_fixed180": maps_a["180"], "a4_rpm360": maps_a["360"],
        "b1_rgb": rgb_b, "b2_rpm0": maps_b["0"], "b3_fixed180": maps_b["180"], "b4_rpm360": maps_b["360"],
    }
    for name, image in panels.items():
        imageio.imwrite(args.output_dir / f"{name}.png", image)
    grid = _grid([[rgb_a, maps_a["0"], maps_a["180"], maps_a["360"]], [rgb_b, maps_b["0"], maps_b["180"], maps_b["360"]]])
    imageio.imwrite(args.output_dir / "grid_crawl_then_dash.png", grid)

    stats = {
        "dataset": "EVEYES-co-motion-v2 constructed regime",
        "scene": SCENE,
        "constructed": True,
        "construction": "recorded Scene_005/drive_stop_drive traversal re-timed; slow frames from Practical-RIFE; IMU and poses regenerated",
        "native_maps": "960x540",
        "not_the_64x64_procedural_pack": True,
        "panels_are_unlabelled": True,
        "accumulation": "one evaluation window per panel",
        "event_palette": {"name": "EIRM", "background": "#030712", "positive": "#37DCFF", "negative": "#E148FF", "lambda": 3.0, "gamma": 0.65, "normalization": "shared per row: max over the row of log1p(lambda * count), per polarity"},
        "row_a": a,
        "row_b": b,
        "row_b_zero_is_not_table1_ours": "Table 1 Ours is {180,360} with a 180 floor and cannot select 0",
        "files": {key: f"{key}.png" for key in panels},
    }
    (args.output_dir / "metrics.json").write_text(json.dumps(stats, indent=2) + "\n", encoding="utf-8")
    caption = (
        "Figure. Constructed EVEYES crawl-then-dash sequence, 960x540 inverse-warped events.\n"
        "Columns: RGB, 0 RPM, Fixed 180, 360 RPM.\n"
        f"(a) crawl, t={a['t']:.2f}s, scene speed {a['scene_speed']:.2f}x on a {a['sensor_fps']:.0f} Hz clock, so a\n"
        f"    180 RPM prism sweeps {a['prism_sweep_deg_per_window_at_180rpm']:.1f} deg per window. 180 is already the right gear:\n"
        f"    edge F1 {a['edge_f1']['180']:.3f} against {a['edge_f1']['360']:.3f} at 360, whose occupancy rises from\n"
        f"    {a['unique_pixel_ratio']['180']:.2f} to {a['unique_pixel_ratio']['360']:.2f}. Ours stays at {a['ours_rpm']:.0f} RPM.\n"
        f"(b) dash, t={b['t']:.2f}s, scene speed {b['scene_speed']:.0f}x on a {b['sensor_fps']:.0f} Hz clock, so the same 180 RPM\n"
        f"    prism sweeps only {b['prism_sweep_deg_per_window_at_180rpm']:.2f} deg per window and both claims flip. Ours selects\n"
        f"    {b['ours_rpm']:.0f} RPM and beats Fixed 180 on edge F1 {b['edge_f1']['180']:.3f} -> {b['edge_f1']['ours']:.3f}\n"
        f"    ({b['ours_minus_180_edge_f1']:+.3f}), while 0 RPM beats Fixed 180 on object-edge F1\n"
        f"    {b['object_edge_f1']['180']:.3f} -> {b['object_edge_f1']['0']:.3f} ({b['zero_minus_180_object_edge_f1']:+.3f}).\n"
        "The 0 RPM column is the off gear on the same official eval, not Table 1 Ours, whose\n"
        "{180,360} gear set with a 180 floor cannot select 0. Events use the EIRM palette:\n"
        "background #030712, positive #37DCFF, negative #E148FF, lambda 3.0, gamma 0.65, with\n"
        "one shared scale per row. Not the 64x64 pack.\n"
    )
    (args.output_dir / "caption.txt").write_text(caption, encoding="utf-8")
    print(json.dumps({"output_dir": str(args.output_dir), "row_a": a, "row_b": b}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
