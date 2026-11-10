#!/usr/bin/env python3
"""Append CRUISE IMU=0 sequences onto the frozen v2 F1 oracles.

IMU bounds stay the v2 co_motion gyro range, so zero gyro maps to imu_motion=0.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from scripts.build_eveyes_v2_f1_oracles import GEAR_180, GEAR_360, _imu_fraction, _imu_raw, _load_result_rows, _ready_path, _window_table
from ami_simulator.pipeline import load_rgb_sequence, rgb_to_linear_luminance
from scripts.evaluate_dl3dv_downstream import _visual_motion
from scripts.evaluate_phase_continuous_control import GEAR_RPM
from scripts.train_inverse_motion_policy import FEATURE_DIM

CRUISE = (
    ("train", "Scene_001", "camera_only"),
    ("train", "Scene_001", "object_only"),
    ("valid", "Scene_004", "camera_only"),
    ("valid", "Scene_004", "object_only"),
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v2-manifest", type=Path, default=Path("results/eveyes_v2_regime_f1_oracles/manifest.json"))
    parser.add_argument("--cruise-root", type=Path, default=Path("data/eveyes_cruise_imu0_ready"))
    parser.add_argument("--cruise-results", type=Path, default=Path("results/ami_next/eveyes_cruise_imu0_ami.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/eveyes_v2_plus_cruise_oracles"))
    parser.add_argument("--max-frames", type=int, default=480)
    args = parser.parse_args()
    v2 = json.loads(args.v2_manifest.read_text(encoding="utf-8"))
    imu_bounds = tuple(float(value) for value in v2["imu_motion_bounds"])
    rows = _load_result_rows([args.cruise_results])
    args.output_dir.mkdir(parents=True, exist_ok=True)
    records = list(v2["oracles"])
    teacher_stats = dict(v2.get("teacher_stats", {}))
    for split, scene, regime in CRUISE:
        scene_id = f"{split}/{scene}/{regime}"
        root = _ready_path(args.cruise_root, split, scene, regime)
        frames, timestamps = load_rgb_sequence(root / "images", root / "timestamps.csv", timestamp_unit="s", max_frames=args.max_frames)
        timestamps = timestamps[: len(frames)]
        visual = _visual_motion(rgb_to_linear_luminance(frames), timestamps)
        imu_raw = _imu_raw(root / "imu.csv", timestamps)
        imu = _imu_fraction(imu_raw, imu_bounds)
        rows_180 = _window_table(rows, (scene_id,), "ami_fixed_180")
        rows_360 = _window_table(rows, (scene_id,), "ami_fixed_360")
        count = min(len(visual), len(rows_180), len(rows_360))
        n_gears = len(GEAR_RPM)
        edge_f1 = np.zeros((count, n_gears), dtype=np.float64)
        densities = np.zeros((count, n_gears), dtype=np.float64)
        overlap = np.zeros((count, n_gears), dtype=np.float64)
        unique_ratio = np.zeros((count, n_gears), dtype=np.float64)
        for index in range(count):
            edge_f1[index, GEAR_180] = float(rows_180[index]["edge_f1"])
            edge_f1[index, GEAR_360] = float(rows_360[index]["edge_f1"])
            unique_ratio[index, GEAR_180] = float(rows_180[index]["unique_pixel_ratio"])
            unique_ratio[index, GEAR_360] = float(rows_360[index]["unique_pixel_ratio"])
            densities[index, GEAR_180] = unique_ratio[index, GEAR_180]
            densities[index, GEAR_360] = unique_ratio[index, GEAR_360]
        output = args.output_dir / f"{split}_{scene}_{regime}.npz"
        np.savez_compressed(
            output,
            visual_motion=visual[:count].astype(np.float64),
            imu_motion=imu[:count].astype(np.float64),
            imu_raw=imu_raw[:count].astype(np.float64),
            densities=densities,
            unique_ratio=unique_ratio,
            temporal_overlap=overlap,
            edge_f1=edge_f1,
            timestamps=timestamps[1 : count + 1],
        )
        records.append({"scene": scene_id, "condition": regime, "split": split, "output": str(output), "windows": count, "source": "eveyes_full_CRUISE_imu0"})
        teacher_stats[scene_id] = {
            "windows": count,
            "mean_f1_180": float(edge_f1[:, GEAR_180].mean()),
            "mean_f1_360": float(edge_f1[:, GEAR_360].mean()),
            "mean_visual_motion": float(visual[:count].mean()),
            "mean_imu_motion": float(imu[:count].mean()),
            "teacher_360_frac": float(np.mean(edge_f1[:, GEAR_360] + 1e-12 >= edge_f1[:, GEAR_180])),
        }
        print(f"wrote {output} {json.dumps(teacher_stats[scene_id], sort_keys=True)}")
    manifest = {
        **v2,
        "dataset": "EVEYES-co-motion-v2-plus-full-CRUISE-imu0",
        "imu_motion_bounds": list(imu_bounds),
        "imu_bounds_source": v2.get("imu_bounds_source"),
        "cruise_source": "data/eveyes_full CRUISE, 50mm, gyro identically 0",
        "oracles": records,
        "teacher_stats": teacher_stats,
        "feature_dim": FEATURE_DIM,
    }
    path = args.output_dir / "manifest.json"
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"manifest": str(path), "cruise": {key: teacher_stats[key] for key in teacher_stats if "Scene_001" in key or ("Scene_004" in key and "train" not in str(teacher_stats))}}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
