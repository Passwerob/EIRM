#!/usr/bin/env python3
"""EVEYES object_only: why a fixed 180 RPM is over-excited.

Camera is frozen, the car already moves. Natural events (0 RPM) stay on the
object. Fixed 180 still paints the static city. Paper Ours has a 180 floor,
so it matches 180 here — this figure is the sensing mismatch, not a claim
that current Ours downshifts to 0.

Panels are bare native-resolution PNGs with no titles, badges, borders, or
burnt-in numbers, so the figure can be laid out downstream. Numbers live in
metrics.json. Events use the EIRM palette.

Data: data/eveyes_comotion_v2_ready/test/Scene_005/object_only
Eval occupancy: results/ami_next/eveyes_v2_regimes_test_ours_plus_dsd.json
Prism 0.5 deg, reference-plane events, same palette for every method.
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
from ami_simulator.pipeline import load_rgb_sequence, rgb_to_linear_luminance
from scripts.evaluate_dl3dv_downstream import _ami_windows, _dvs_windows
from scripts.evaluate_eveyes_v2_downstream import _intrinsics, _scene_path
from scripts.render_eveyes_camera_nfb_event_video import _concat

READY = Path("data/eveyes_comotion_v2_ready")
EVAL_JSON = Path("results/ami_next/eveyes_v2_regimes_test_ours_plus_dsd.json")
SCENE_ID = "test/Scene_005/object_only"
HERO_T = 3.9833333333333334
WARMUP = 8
ACCUM = 3
PRISM_DEG = 0.5
GUTTER = 14
PAPER_BG = (255, 255, 255)


def _method_rows(payload: dict, method: str) -> list[dict]:
    rows = [row for row in payload["rows"] if row["scene"] == SCENE_ID and row["method"] == method]
    rows.sort(key=lambda row: int(row["window"]))
    return rows


def _grid(panels: list[np.ndarray]) -> np.ndarray:
    height, width = panels[0].shape[:2]
    canvas_w = len(panels) * width + (len(panels) - 1) * GUTTER
    canvas = np.full((height, canvas_w, 3), PAPER_BG, dtype=np.uint8)
    for index, panel in enumerate(panels):
        left = index * (width + GUTTER)
        canvas[:, left : left + width] = panel
    return canvas


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:6")
    parser.add_argument("--output-dir", type=Path, default=Path("results/ami_next/figures/object_only_180_overexcited"))
    args = parser.parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        args.device = "cpu"

    payload = json.loads(EVAL_JSON.read_text(encoding="utf-8"))
    if payload.get("dataset") != "EVEYES-co-motion-v2":
        raise SystemExit("expected EVEYES regimes json")
    metrics = {
        name: _method_rows(payload, key)
        for name, key in (("0", "ami_passive"), ("180", "ami_fixed_180"), ("360", "ami_fixed_360"), ("ours", "ours"))
    }
    t = np.asarray([float(row["t0"]) for row in metrics["180"]], dtype=np.float64)
    hero = int(np.argmin(np.abs(t - HERO_T)))

    scene = _scene_path(READY, "test", "Scene_005", "object_only")
    start = max(0, hero - max(WARMUP, ACCUM - 1))
    frames, timestamps = load_rgb_sequence(scene / "images", scene / "timestamps.csv", timestamp_unit="s", max_frames=hero + 2)
    timestamps = timestamps[: len(frames)]
    luminance = rgb_to_linear_luminance(frames[start:])
    times = timestamps[start:]
    valid = np.ones((len(luminance), luminance.shape[1], luminance.shape[2]), dtype=bool)
    prism = RotatingWedgePrism(_intrinsics(scene), PRISM_DEG)
    print(f"EVEYES object_only  hero={hero} t={t[hero]:.3f}  {args.device}", flush=True)
    local = hero - start
    maps = {}
    for rpm, key in ((0.0, "0"), (180.0, "180"), (360.0, "360")):
        print(f"  simulate {int(rpm)} RPM", flush=True)
        if rpm == 0.0:
            reference, _ = _dvs_windows(luminance, times)
        else:
            reference, _, _ = _ami_windows(luminance, times, valid, prism, rpm, device=args.device)
        first = max(0, local - (ACCUM - 1))
        maps[key] = eirm_from_events(_concat(list(reference[first : local + 1])), luminance.shape[1], luminance.shape[2])

    rgb = frames[hero]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    panels = {"p1_rgb": rgb, "p2_rpm0": maps["0"], "p3_fixed180": maps["180"], "p4_fixed360": maps["360"]}
    for name, image in panels.items():
        imageio.imwrite((args.output_dir / name).with_suffix(".png"), image)
    imageio.imwrite(args.output_dir / "grid.png", _grid([rgb, maps["0"], maps["180"], maps["360"]]))

    stats = {
        "dataset": "EVEYES-co-motion-v2",
        "scene": SCENE_ID,
        "native_maps": "960x540",
        "not_the_64x64_procedural_pack": True,
        "prism_angle_deg": PRISM_DEG,
        "panels_are_unlabelled": True,
        "event_palette": {
            "name": "EIRM",
            "background": "#030712",
            "positive": "#37DCFF",
            "negative": "#E148FF",
            "lambda": 3.0,
            "gamma": 0.65,
            "normalization": "per-panel max of log1p(lambda * count)",
        },
        "hero_t": float(t[hero]),
        "hero_window": hero,
        **{
            f"{field}_{name}": float(metrics[name][hero][field])
            for name in ("0", "180", "360", "ours")
            for field in ("unique_pixel_ratio", "edge_f1", "object_edge_f1", "target_purity")
        },
        **{f"background_event_count_{name}": float(metrics[name][hero]["background_event_count"]) for name in ("0", "180", "360")},
        "ours_rpm": float(metrics["ours"][hero]["rpm"]),
        "ours_stays_at_180": True,
        "note": "180 over-excites the static background. Current EVEYES Ours cannot go below 180.",
        "files": {"rgb": "p1_rgb.png", "rpm_0": "p2_rpm0.png", "fixed_180": "p3_fixed180.png", "fixed_360": "p4_fixed360.png"},
    }
    (args.output_dir / "metrics.json").write_text(json.dumps(stats, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output_dir": str(args.output_dir), **stats}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
