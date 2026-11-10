#!/usr/bin/env python3
"""Official EVEYES 960×540 panels for the two-sided "why not always 180" figure.

(a) camera-only t=0.37 s: paper Ours at 360 beats Fixed 180 on global F1 / recall.
(b) drive-stop-drive t=7.00 s: 0 RPM beats Fixed 180 on object-edge F1.

Panels are written as bare native-resolution PNGs with no titles, badges,
borders, or burnt-in numbers, so the figure can be laid out downstream. All
numbers live in metrics.json. Events use the EIRM palette. Not the 64×64 pack.

Row (b) is the off gear on the same official eval, not Table 1 Ours
({180,360} with a 180 floor cannot select 0).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import imageio.v3 as imageio
import numpy as np
import torch

from ami_simulator.event_visualization import eirm_from_events
from ami_simulator.geometry import RotatingWedgePrism
from ami_simulator.pipeline import _IMAGE_SUFFIXES, _read_timestamp_file, rgb_to_linear_luminance
from ami_simulator.types import EventStream
from scripts.evaluate_dl3dv_downstream import _step
from scripts.evaluate_eveyes_v2_downstream import _intrinsics, _scene_path
from scripts.render_eveyes_camera_nfb_event_video import _concat

READY = Path("data/eveyes_comotion_v2_ready")
CAM_JSON = Path("results/ami_next/eveyes_v2_regimes_test_ours_plus_dsd.json")
DSD_JSON = Path("results/ami_next/eveyes_v2_drive_stop_drive_ours_plus_dsd.json")
PRISM_DEG = 0.5
WARMUP = 12
ACCUM = 3
GUTTER = 14
PAPER_BG = (255, 255, 255)


def _rows(payload: dict, scene: str, method: str) -> list[dict]:
    selected = [row for row in payload["rows"] if row["scene"] == scene and row["method"] == method]
    selected.sort(key=lambda row: int(row["window"]))
    return selected


def _load_slice(scene: Path, start: int, last_frame: int) -> tuple[np.ndarray, np.ndarray]:
    directory = scene / "images"
    image_paths = sorted(path for path in directory.iterdir() if path.suffix.lower() in _IMAGE_SUFFIXES)
    selected = image_paths[start : last_frame + 1]
    if len(selected) < 2:
        raise SystemExit(f"{scene} slice {start}:{last_frame} has {len(selected)} frames")
    frames = np.stack([np.asarray(imageio.imread(path))[..., :3] for path in selected])
    timestamps = _read_timestamp_file(scene / "timestamps.csv")[start : last_frame + 1]
    if frames.shape[1] != 540 or frames.shape[2] != 960:
        raise SystemExit(f"{scene} is {frames.shape[2]}x{frames.shape[1]}, expected 960x540")
    return frames, timestamps


def _replay(luminance, timestamps, prism, rpms: np.ndarray, device: str) -> list[EventStream]:
    phase = 0.0
    previous_active = None
    sensor_reference = None
    previous_rpm = float(rpms[0])
    ones = np.ones((len(timestamps), luminance.shape[1], luminance.shape[2]), dtype=bool)
    streams = []
    for index, rpm in enumerate(rpms):
        result = _step(
            luminance,
            timestamps,
            ones,
            prism,
            index,
            float(rpm),
            phase,
            previous_active,
            previous_rpm,
            sensor_reference,
            device=device,
        )
        streams.append(result["reference_events"])
        phase = float(result["phase_end"])
        previous_active = np.asarray(result["active"])
        sensor_reference = np.asarray(result["sensor_reference_log"])
        previous_rpm = float(rpm)
    return streams


def _hero_maps(scene_dir: Path, hero: int, n_windows: int, rpms: dict[str, np.ndarray], device: str):
    start = max(0, hero - WARMUP)
    last_frame = min(n_windows, hero + 1)
    print(f"  load {scene_dir} frames {start}..{last_frame} hero={hero}", flush=True)
    frames, timestamps = _load_slice(scene_dir, start, last_frame)
    luminance = rgb_to_linear_luminance(frames)
    prism = RotatingWedgePrism(_intrinsics(scene_dir), PRISM_DEG)
    local = hero - start
    height, width = luminance.shape[1], luminance.shape[2]
    maps = {}
    n_sim = last_frame - start
    for name, series in rpms.items():
        chunk = np.asarray(series[start : start + n_sim], dtype=np.float64)
        print(f"    replay {name} n={len(chunk)} rpm0={chunk[0]:.0f} rpm_hero={chunk[local]:.0f}", flush=True)
        streams = _replay(luminance, timestamps, prism, chunk, device)
        first = max(0, local - (ACCUM - 1))
        maps[name] = eirm_from_events(_concat(list(streams[first : local + 1])), height, width)
    return frames[local], maps


def _grid(rows: list[list[np.ndarray]]) -> np.ndarray:
    height, width = rows[0][0].shape[:2]
    n_cols = max(len(row) for row in rows)
    canvas_h = len(rows) * height + (len(rows) - 1) * GUTTER
    canvas_w = n_cols * width + (n_cols - 1) * GUTTER
    canvas = np.full((canvas_h, canvas_w, 3), PAPER_BG, dtype=np.uint8)
    for r, row in enumerate(rows):
        for c, panel in enumerate(row):
            top = r * (height + GUTTER)
            left = c * (width + GUTTER)
            canvas[top : top + height, left : left + width] = panel
    return canvas


def _save(image: np.ndarray, stem: Path) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    imageio.imwrite(stem.with_suffix(".png"), image)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:6")
    parser.add_argument("--output-dir", type=Path, default=Path("results/ami_next/figures/paper_two_sided_180"))
    args = parser.parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        args.device = "cpu"

    cam = json.loads(CAM_JSON.read_text(encoding="utf-8"))
    dsd = json.loads(DSD_JSON.read_text(encoding="utf-8"))
    if cam.get("dataset") != "EVEYES-co-motion-v2" or dsd.get("dataset") != "EVEYES-co-motion-v2":
        raise SystemExit("expected EVEYES json")
    cam_scene = "test/Scene_005/camera_only"
    dsd_scene = "test/Scene_005/drive_stop_drive"
    cam_ours = _rows(cam, cam_scene, "ours")
    cam_180 = _rows(cam, cam_scene, "ami_fixed_180")
    dsd_0 = _rows(dsd, dsd_scene, "ami_passive")
    dsd_180 = _rows(dsd, dsd_scene, "ami_fixed_180")
    t_cam = np.asarray([float(row["t0"]) for row in cam_ours])
    t_dsd = np.asarray([float(row["t0"]) for row in dsd_0])
    hero_a = int(np.argmin(np.abs(t_cam - 0.367)))
    hero_b = int(np.argmin(np.abs(t_dsd - 7.000)))
    rpm_ours = np.asarray([float(row["rpm"]) for row in cam_ours])
    if rpm_ours[hero_a] < 359:
        raise SystemExit(f"row (a) Ours is {rpm_ours[hero_a]:.0f} RPM")
    if float(dsd_0[hero_b]["object_edge_f1"]) <= float(dsd_180[hero_b]["object_edge_f1"]):
        raise SystemExit("row (b) object-edge F1 is not 0 > 180")

    a = {
        "t": float(t_cam[hero_a]),
        "window": hero_a,
        "ours_rpm": float(rpm_ours[hero_a]),
        "f1_180": float(cam_180[hero_a]["edge_f1"]),
        "f1_ours": float(cam_ours[hero_a]["edge_f1"]),
        "rec_180": float(cam_180[hero_a]["edge_recall"]),
        "rec_ours": float(cam_ours[hero_a]["edge_recall"]),
        "occ_180": float(cam_180[hero_a]["unique_pixel_ratio"]),
        "occ_ours": float(cam_ours[hero_a]["unique_pixel_ratio"]),
    }
    a["df1"] = a["f1_ours"] - a["f1_180"]
    a["drec"] = a["rec_ours"] - a["rec_180"]
    b = {
        "t": float(t_dsd[hero_b]),
        "window": hero_b,
        "obj_180": float(dsd_180[hero_b]["object_edge_f1"]),
        "obj_0": float(dsd_0[hero_b]["object_edge_f1"]),
        "f1_180": float(dsd_180[hero_b]["edge_f1"]),
        "f1_0": float(dsd_0[hero_b]["edge_f1"]),
        "occ_180": float(dsd_180[hero_b]["unique_pixel_ratio"]),
        "occ_0": float(dsd_0[hero_b]["unique_pixel_ratio"]),
        "pur_180": float(dsd_180[hero_b]["target_purity"]),
        "pur_0": float(dsd_0[hero_b]["target_purity"]),
    }
    b["dobj"] = b["obj_0"] - b["obj_180"]

    print(f"(a) camera_only t={a['t']:.3f} Ours={a['ours_rpm']:.0f}  F1 {a['f1_180']:.3f}->{a['f1_ours']:.3f}", flush=True)
    rgb_a, maps_a = _hero_maps(
        _scene_path(READY, "test", "Scene_005", "camera_only"),
        hero_a,
        len(cam_ours),
        {"180": np.full(len(cam_ours), 180.0), "ours": rpm_ours},
        args.device,
    )
    print(f"(b) drive_stop_drive t={b['t']:.3f} 0 vs 180  obj-F1 {b['obj_180']:.3f}->{b['obj_0']:.3f}", flush=True)
    rgb_b, maps_b = _hero_maps(
        _scene_path(READY, "test", "Scene_005", "drive_stop_drive"),
        hero_b,
        len(dsd_0),
        {"180": np.full(len(dsd_180), 180.0), "0": np.zeros(len(dsd_0))},
        args.device,
    )

    panels = {
        "a1_rgb": rgb_a,
        "a2_fixed180": maps_a["180"],
        "a3_ours360": maps_a["ours"],
        "b1_rgb": rgb_b,
        "b2_fixed180": maps_b["180"],
        "b3_rpm0": maps_b["0"],
    }
    for name, image in panels.items():
        _save(image, args.output_dir / name)
    _save(_grid([[rgb_a, maps_a["180"], maps_a["ours"]], [rgb_b, maps_b["180"], maps_b["0"]]]), args.output_dir / "grid_two_sided")
    _save(_grid([[rgb_a, maps_a["180"], maps_a["ours"]]]), args.output_dir / "grid_row_a")

    stats = {
        "dataset": "EVEYES-co-motion-v2",
        "native_maps": "960x540",
        "not_the_64x64_procedural_pack": True,
        "event_palette": {
            "name": "EIRM",
            "background": "#030712",
            "positive": "#37DCFF",
            "negative": "#E148FF",
            "lambda": 3.0,
            "gamma": 0.65,
            "normalization": "per-panel max of log1p(lambda * count)",
        },
        "panels_are_unlabelled": True,
        "row_a": a,
        "row_b": b,
        "row_b_not_table1_ours": True,
        "row_b_metric": "object_edge_f1",
        "row_b_global_f1_still_180_better": True,
        "files": {
            "a_rgb": "a1_rgb.png",
            "a_fixed_180": "a2_fixed180.png",
            "a_ours_360": "a3_ours360.png",
            "b_rgb": "b1_rgb.png",
            "b_fixed_180": "b2_fixed180.png",
            "b_rpm_0": "b3_rpm0.png",
        },
    }
    (args.output_dir / "metrics.json").write_text(json.dumps(stats, indent=2) + "\n", encoding="utf-8")
    caption = (
        "Figure. EVEYES 960x540 inverse-warped events, same Fixed 180, two short instants.\n"
        f"(a) camera-only t={a['t']:.2f}s: paper Ours selects 360 RPM and beats Fixed 180\n"
        f"    F1 {a['f1_180']:.3f} -> {a['f1_ours']:.3f} ({a['df1']:+.3f}), recall {a['rec_180']:.2f} -> {a['rec_ours']:.2f} ({a['drec']:+.2f}),\n"
        f"    occupancy {a['occ_180']:.2f} -> {a['occ_ours']:.2f}.\n"
        f"(b) drive-stop-drive t={b['t']:.2f}s: 180 over-fills the frame. 0 RPM is better on\n"
        f"    object-edge F1 {b['obj_180']:.3f} -> {b['obj_0']:.3f} ({b['dobj']:+.3f}) and occupancy\n"
        f"    {b['occ_180']:.2f} -> {b['occ_0']:.2f}; global F1 stays higher at 180 ({b['f1_180']:.3f} vs {b['f1_0']:.3f}).\n"
        "Row (b) is the off gear on the same official eval, not Table 1 Ours ({180,360} with a\n"
        "180 floor). Events use the EIRM palette: background #030712, positive #37DCFF,\n"
        "negative #E148FF, lambda 3.0, gamma 0.65. Not the 64x64 pack.\n"
    )
    (args.output_dir / "caption.txt").write_text(caption, encoding="utf-8")
    print(json.dumps({"output_dir": str(args.output_dir), **stats}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
