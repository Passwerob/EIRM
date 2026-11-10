#!/usr/bin/env python3
"""Search EVEYES re-timings for a window where 0 RPM beats Fixed 180.

RGB is Blender-rendered offline, so a new motion can only be a re-timing of
already rendered frames. Two knobs:

  stride  take every k-th rendered frame  -> k x larger displacement per window
  dt      declare the frames closer in time -> the prism sweeps less per window

0 RPM events depend only on the two frames, so dt isolates the prism effect.

Scoring is the official EVEYES protocol (evaluate_eveyes_v2_downstream._score):
image-wide edge F1 at quantile 0.75 plus the object-ROI metrics.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import imageio.v3 as imageio
import numpy as np
import torch

from ami_simulator.geometry import RotatingWedgePrism
from ami_simulator.pipeline import _IMAGE_SUFFIXES, rgb_to_linear_luminance
from scripts.evaluate_dl3dv_downstream import _ami_windows, _dvs_windows
from scripts.evaluate_eveyes_v2_downstream import _intrinsics, _score, _scene_path

READY = Path("data/eveyes_comotion_v2_ready")
PRISM_DEG = 0.5
GEARS = ((0.0, "ami_passive"), (180.0, "ami_fixed_180"), (360.0, "ami_fixed_360"))


def _read_indexed(scene: Path, indices: np.ndarray) -> tuple[np.ndarray, np.ndarray | None]:
    directory = scene / "images"
    image_paths = sorted(path for path in directory.iterdir() if path.suffix.lower() in _IMAGE_SUFFIXES)
    frames = np.stack([np.asarray(imageio.imread(image_paths[index]))[..., :3] for index in indices])
    mask_dir = scene / "object_masks"
    masks = None
    if mask_dir.is_dir():
        mask_paths = sorted(mask_dir.glob("*.png"))
        if len(mask_paths) >= int(indices.max()) + 1:
            masks = np.stack([cv2.imread(str(mask_paths[index]), cv2.IMREAD_GRAYSCALE) > 127 for index in indices])
    return frames, masks


def _run(scene: Path, scene_id: str, indices: np.ndarray, dt: float, device: str) -> list[dict]:
    frames, masks = _read_indexed(scene, indices)
    luminance = rgb_to_linear_luminance(frames)
    timestamps = np.arange(len(frames), dtype=np.float64) * dt
    valid = np.ones((len(luminance), luminance.shape[1], luminance.shape[2]), dtype=bool)
    prism = RotatingWedgePrism(_intrinsics(scene), PRISM_DEG)
    rows = []
    for rpm, method in GEARS:
        if rpm == 0.0:
            reference, rpms = _dvs_windows(luminance, timestamps)
        else:
            reference, _, rpms = _ami_windows(luminance, timestamps, valid, prism, rpm, device=device)
        rows.extend(_score(reference, luminance, valid, timestamps, method, scene_id, rpms, masks))
    return rows


def _compare(rows: list[dict], config: str) -> dict:
    by_method = {}
    for row in rows:
        by_method.setdefault(row["method"], []).append(row)
    for method in by_method:
        by_method[method].sort(key=lambda row: int(row["window"]))
    g0, g180, g360 = by_method["ami_passive"], by_method["ami_fixed_180"], by_method["ami_fixed_360"]
    f1_0 = np.asarray([float(row["edge_f1"]) for row in g0])
    f1_180 = np.asarray([float(row["edge_f1"]) for row in g180])
    f1_360 = np.asarray([float(row["edge_f1"]) for row in g360])
    obj_0 = np.asarray([float(row.get("object_edge_f1", np.nan)) for row in g0])
    obj_180 = np.asarray([float(row.get("object_edge_f1", np.nan)) for row in g180])
    occ_0 = np.asarray([float(row["unique_pixel_ratio"]) for row in g0])
    occ_180 = np.asarray([float(row["unique_pixel_ratio"]) for row in g180])
    d0 = f1_0 - f1_180
    d360 = f1_360 - f1_180
    dobj = obj_0 - obj_180
    return {
        "config": config,
        "windows": len(f1_0),
        "mean_f1_0": float(f1_0.mean()),
        "mean_f1_180": float(f1_180.mean()),
        "mean_f1_360": float(f1_360.mean()),
        "mean_occ_0": float(occ_0.mean()),
        "mean_occ_180": float(occ_180.mean()),
        "best_global_0_minus_180": float(np.nanmax(d0)),
        "n_global_0_beats_180": int(np.count_nonzero(d0 > 0)),
        "best_global_360_minus_180": float(np.nanmax(d360)),
        "n_global_360_beats_180": int(np.count_nonzero(d360 > 0)),
        "best_object_0_minus_180": float(np.nanmax(dobj)) if np.any(np.isfinite(dobj)) else None,
        "n_object_0_beats_180": int(np.count_nonzero(dobj > 0)) if np.any(np.isfinite(dobj)) else None,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:6")
    parser.add_argument("--windows", type=int, default=24)
    parser.add_argument("--output", type=Path, default=Path("results/ami_next/eveyes_two_sided_construction_search.json"))
    parser.add_argument("--scenes", nargs="*", default=("test/Scene_005",), help="split/Scene entries to probe.")
    parser.add_argument("--dt-only", action="store_true", help="Probe only the dt-compression configs, which are the ones that flip the object metric.")
    args = parser.parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        args.device = "cpu"

    base_dt = 1.0 / 60.0
    configs = []
    if args.dt_only:
        for compress in (4, 8):
            configs.append(("camera_only", 1, compress, 0))
            configs.append(("drive_stop_drive", 1, compress, 400))
    else:
        for stride in (1, 2, 4, 8):
            configs.append(("camera_only", stride, 1, 0))
        for compress in (2, 4, 8):
            configs.append(("camera_only", 1, compress, 0))
        for stride in (1, 4, 8):
            configs.append(("object_only", stride, 1, 200))
        configs.append(("object_only", 4, 4, 200))
        configs.append(("drive_stop_drive", 1, 1, 400))
        configs.append(("drive_stop_drive", 4, 1, 400))

    summaries = []
    detail = {}
    for scene_key in args.scenes:
        split, scene_name = scene_key.split("/")
        for regime, stride, compress, start in configs:
            scene = _scene_path(READY, split, scene_name, regime)
            indices = start + stride * np.arange(args.windows + 1)
            indices = indices[indices < 480]
            if len(indices) < 4:
                continue
            dt = base_dt / compress
            name = f"{split}_{scene_name}_{regime}_stride{stride}_dt{compress}x_start{start}"
            print(f"== {name}  frames {indices[0]}..{indices[-1]}  dt={dt:.6f}", flush=True)
            rows = _run(scene, f"constructed/{name}", indices, dt, args.device)
            summary = _compare(rows, name)
            summaries.append(summary)
            detail[name] = rows
            print(
                f"   F1 0/180/360 = {summary['mean_f1_0']:.3f} / {summary['mean_f1_180']:.3f} / {summary['mean_f1_360']:.3f}"
                f"   global 0>180 in {summary['n_global_0_beats_180']}/{summary['windows']}"
                f"   object 0>180 in {summary['n_object_0_beats_180']}/{summary['windows']}",
                flush=True,
            )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(
            {
                "dataset": "EVEYES-co-motion-v2",
                "scenes": list(args.scenes),
                "protocol": "evaluate_eveyes_v2_downstream._score",
                "prism_angle_deg": PRISM_DEG,
                "note": "stride re-indexes rendered frames; dt compresses the declared window duration",
                "summaries": summaries,
                "rows": detail,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summaries, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
