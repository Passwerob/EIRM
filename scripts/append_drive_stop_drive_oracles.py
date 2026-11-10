#!/usr/bin/env python3
"""Append train/valid drive_stop_drive windows onto frozen v2 F1 oracles.

Does not add test/Scene_005/drive_stop_drive — that clip stays eval-only.
IMU bounds stay the v2 co_motion gyro range.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
from PIL import Image

from scripts.build_eveyes_v2_f1_oracles import GEAR_180, GEAR_360, _imu_fraction, _imu_raw, _load_result_rows, _ready_path, _window_table
from scripts.evaluate_phase_continuous_control import GEAR_RPM
from scripts.train_inverse_motion_policy import FEATURE_DIM

DSD = (
    ("train", "Scene_002", "drive_stop_drive"),
    ("valid", "Scene_004", "drive_stop_drive"),
)


def _visual_streaming(image_dir: Path, timestamps: np.ndarray, n_frames: int) -> np.ndarray:
    """Native-resolution visual motion without stacking all RGB in memory."""
    def luminance(index: int) -> np.ndarray:
        rgb = np.asarray(Image.open(image_dir / f"{index:06d}.png").convert("RGB"), dtype=np.float32) / 255.0
        linear = np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)
        return np.clip(0.2126 * linear[..., 0] + 0.7152 * linear[..., 1] + 0.0722 * linear[..., 2], 0.0, 1.0)

    log_prev = np.log(np.clip(luminance(0), 1e-6, None))
    visual = np.empty(n_frames - 1, dtype=np.float64)
    for index in range(1, n_frames):
        log_now = np.log(np.clip(luminance(index), 1e-6, None))
        visual[index - 1] = float(np.quantile(np.abs(log_now - log_prev), 0.9) / (timestamps[index] - timestamps[index - 1]))
        log_prev = log_now
    return visual


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v2-manifest", type=Path, default=Path("results/eveyes_v2_regime_f1_oracles/manifest.json"))
    parser.add_argument("--dataset-root", type=Path, default=Path("data/eveyes_comotion_v2_ready"))
    parser.add_argument("--dsd-results", nargs="*", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("results/eveyes_v2_plus_dsd_oracles"))
    parser.add_argument("--max-frames", type=int, default=480)
    args = parser.parse_args()
    v2 = json.loads(args.v2_manifest.read_text(encoding="utf-8"))
    imu_bounds = tuple(float(value) for value in v2["imu_motion_bounds"])
    rows = _load_result_rows(list(args.dsd_results))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for record in v2["oracles"]:
        source = Path(str(record["output"]))
        destination = args.output_dir / source.name
        if source.exists() and not destination.exists():
            shutil.copy2(source, destination)
    records = []
    for record in v2["oracles"]:
        copy = dict(record)
        copy["output"] = str(args.output_dir / Path(str(record["output"])).name)
        records.append(copy)
    teacher_stats = dict(v2.get("teacher_stats", {}))
    for split, scene, regime in DSD:
        scene_id = f"{split}/{scene}/{regime}"
        root = _ready_path(args.dataset_root, split, scene, regime)
        timestamps = np.genfromtxt(root / "timestamps.csv", delimiter=",", names=True)
        times = np.asarray(timestamps["timestamp_seconds"] if "timestamp_seconds" in timestamps.dtype.names else timestamps["timestamp_s"], dtype=np.float64)[: args.max_frames]
        visual = _visual_streaming(root / "images", times, len(times))
        imu_raw = _imu_raw(root / "imu.csv", times)
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
            timestamps=times[1 : count + 1],
        )
        records.append({"scene": scene_id, "condition": regime, "split": split, "output": str(output), "windows": count, "source": "drive_stop_drive"})
        freeze = slice(150, 329)
        teacher_stats[scene_id] = {
            "windows": count,
            "mean_f1_180": float(edge_f1[:, GEAR_180].mean()),
            "mean_f1_360": float(edge_f1[:, GEAR_360].mean()),
            "mean_visual_motion": float(visual[:count].mean()),
            "mean_imu_motion": float(imu[:count].mean()),
            "teacher_360_frac": float(np.mean(edge_f1[:, GEAR_360] + 1e-12 >= edge_f1[:, GEAR_180])),
            "freeze_teacher_360_frac": float(np.mean(edge_f1[freeze, GEAR_360] + 1e-12 >= edge_f1[freeze, GEAR_180])),
            "freeze_mean_visual": float(visual[freeze].mean()),
            "drive_mean_visual": float(np.concatenate([visual[:150], visual[329:count]]).mean()) if count > 329 else float("nan"),
        }
        print(f"wrote {output} {json.dumps(teacher_stats[scene_id], sort_keys=True)}")
    manifest = {
        **v2,
        "dataset": "EVEYES-co-motion-v2-plus-drive-stop-drive-train-valid",
        "oracles": records,
        "teacher_stats": teacher_stats,
        "feature_dim": FEATURE_DIM,
        "drive_stop_drive": "train/Scene_002 and valid/Scene_004 only; test Scene_005 remains eval-only",
    }
    path = args.output_dir / "manifest.json"
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"manifest": str(path), "dsd": {key: teacher_stats[key] for key in teacher_stats if "drive_stop_drive" in key}}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
