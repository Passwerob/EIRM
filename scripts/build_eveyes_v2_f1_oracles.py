#!/usr/bin/env python3
"""Build EVEYES native 960x540 F1-argmax oracles from Table 1 and regime dumps.

Teacher labels need per-window edge F1 at 180 and 360 RPM. co_motion scores
come from results/ami_next/eveyes_v2_downstream_{train,valid,test}.json.
camera_only / object_only scores come from --extra-results JSON files.
This script does not resimulate the prism. It only adds causal visual/IMU
features.

Storm feedback stores unique_pixel_ratio in the densities slot. Count/pixel
density at 960x540 is ~1.5-3 and would latch the 0.75 storm brake on every
window, forcing a 180 RPM teacher.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from ami_simulator.dsec import read_imu_csv
from ami_simulator.pipeline import load_rgb_sequence, rgb_to_linear_luminance
from scripts.evaluate_dl3dv_downstream import _visual_motion
from scripts.evaluate_phase_continuous_control import GEAR_RPM
from scripts.train_inverse_motion_policy import FEATURE_DIM

SCENES = (
    ("train", "Scene_002"),
    ("valid", "Scene_004"),
    ("test", "Scene_005"),
)
REGIMES = ("co_motion", "camera_only", "object_only", "drive_stop_drive")
GEAR_360 = int(np.argmin(np.abs(GEAR_RPM - 360.0)))
GEAR_180 = int(np.argmin(np.abs(GEAR_RPM - 180.0)))
GEAR_0 = int(np.argmin(np.abs(GEAR_RPM - 0.0)))


def _scene_aliases(scene_id: str) -> tuple[str, ...]:
    parts = scene_id.split("/")
    if len(parts) == 3 and parts[2] == "co_motion":
        return (scene_id, f"{parts[0]}/{parts[1]}")
    if len(parts) == 2:
        return (scene_id, f"{scene_id}/co_motion")
    return (scene_id,)


def _ready_path(root: Path, split: str, scene: str, regime: str) -> Path:
    nested = root / split / scene / regime
    if (nested / "images").exists():
        return nested
    legacy = root / split / scene
    if regime == "co_motion" and (legacy / "images").exists():
        return legacy
    raise SystemExit(f"missing ready sequence {nested}")


def _imu_raw(imu_csv: Path, timestamps: np.ndarray) -> np.ndarray:
    imu = read_imu_csv(imu_csv, timestamp_unit="s", gyro_unit="rad/s")
    motion = np.linalg.norm(imu.gyro, axis=1)
    windows = np.empty(len(timestamps) - 1, dtype=np.float64)
    for index, (start, end) in enumerate(zip(timestamps[:-1], timestamps[1:])):
        left = int(np.searchsorted(imu.t, start, side="left"))
        right = int(np.searchsorted(imu.t, end, side="right"))
        values = motion[left:right]
        windows[index] = float(values.mean() if len(values) else np.interp((start + end) * 0.5, imu.t, motion))
    return windows


def _imu_fraction(raw: np.ndarray, bounds: tuple[float, float]) -> np.ndarray:
    low, high = bounds
    return np.clip((raw - low) / max(high - low, 1e-9), 0.0, 1.0)


def _window_table(rows: list[dict[str, object]], scene_ids: tuple[str, ...], method: str) -> list[dict[str, object]]:
    for scene_id in sorted(scene_ids, key=len, reverse=True):
        selected = [row for row in rows if row["scene"] == scene_id and row["method"] == method]
        if not selected:
            continue
        selected.sort(key=lambda row: int(row["window"]))
        if [int(row["window"]) for row in selected] != list(range(len(selected))):
            raise ValueError(f"{scene_id} {method} is missing a contiguous window table.")
        return selected
    raise ValueError(f"{scene_ids} {method} is missing a contiguous window table.")


def _load_result_rows(paths: list[Path]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        rows.extend(payload["rows"])
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=Path("data/eveyes_comotion_v2_ready"))
    parser.add_argument("--results-dir", type=Path, default=Path("results/ami_next"))
    parser.add_argument("--extra-results", nargs="*", type=Path, default=(), help="Regime JSONs, e.g. eveyes_v2_regimes_train_valid.json eveyes_v2_regimes_test.json")
    parser.add_argument("--regimes", nargs="*", default=REGIMES, choices=REGIMES)
    parser.add_argument("--output-dir", type=Path, default=Path("results/eveyes_v2_regime_f1_oracles"))
    parser.add_argument("--max-frames", type=int, default=480)
    parser.add_argument("--require-zero-gear", action="store_true", help="Fail if a regime has no ami_passive rows, instead of leaving the 0 gear empty.")
    args = parser.parse_args()
    if args.max_frames < 2:
        raise SystemExit("max-frames >= 2")
    if "comotion_controlled" in str(args.dataset_root):
        raise SystemExit("Refusing the 64x64 procedural pack.")

    result_paths = [args.results_dir / f"eveyes_v2_downstream_{split}.json" for split, _ in SCENES]
    result_paths.extend(args.extra_results)
    missing = [path for path in result_paths if not path.exists()]
    if missing:
        raise SystemExit(f"missing result JSON: {missing}")
    all_rows = _load_result_rows(result_paths)

    train_scene = _ready_path(args.dataset_root, "train", "Scene_002", "co_motion")
    frames, timestamps = load_rgb_sequence(train_scene / "images", train_scene / "timestamps.csv", timestamp_unit="s", max_frames=args.max_frames)
    timestamps = timestamps[: len(frames)]
    train_imu_raw = _imu_raw(train_scene / "imu.csv", timestamps)
    imu_bounds = (float(np.min(train_imu_raw)), float(np.max(train_imu_raw)))
    if imu_bounds[1] <= imu_bounds[0]:
        raise SystemExit("Train IMU has no range; cannot normalize imu_motion.")

    records = []
    teacher_stats = {}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for split, scene in SCENES:
        for regime in args.regimes:
            scene_id = f"{split}/{scene}/{regime}"
            aliases = _scene_aliases(scene_id)
            root = _ready_path(args.dataset_root, split, scene, regime)
            frames, timestamps = load_rgb_sequence(root / "images", root / "timestamps.csv", timestamp_unit="s", max_frames=args.max_frames)
            timestamps = timestamps[: len(frames)]
            luminance = rgb_to_linear_luminance(frames)
            visual = _visual_motion(luminance, timestamps)
            imu_raw = _imu_raw(root / "imu.csv", timestamps)
            imu = _imu_fraction(imu_raw, imu_bounds)
            tables = {GEAR_180: _window_table(all_rows, aliases, "ami_fixed_180"), GEAR_360: _window_table(all_rows, aliases, "ami_fixed_360")}
            try:
                tables[GEAR_0] = _window_table(all_rows, aliases, "ami_passive")
            except ValueError:
                if args.require_zero_gear:
                    raise SystemExit(f"{scene_id} has no ami_passive rows; run the 0 RPM eval first")
                print(f"  {scene_id}: no ami_passive rows, the 0 gear stays empty")
            count = min([len(visual)] + [len(rows) for rows in tables.values()])
            if count != len(visual) or any(count != len(rows) for rows in tables.values()):
                raise ValueError(f"{scene_id} window count mismatch visual={len(visual)} " + " ".join(f"g{GEAR_RPM[key]:.0f}={len(rows)}" for key, rows in tables.items()))
            n_gears = len(GEAR_RPM)
            edge_f1 = np.zeros((count, n_gears), dtype=np.float64)
            object_edge_f1 = np.zeros((count, n_gears), dtype=np.float64)
            target_purity = np.zeros((count, n_gears), dtype=np.float64)
            object_area = np.zeros((count,), dtype=np.float64)
            densities = np.zeros((count, n_gears), dtype=np.float64)
            overlap = np.zeros((count, n_gears), dtype=np.float64)
            unique_ratio = np.zeros((count, n_gears), dtype=np.float64)
            for gear, rows in tables.items():
                for index in range(count):
                    edge_f1[index, gear] = float(rows[index]["edge_f1"])
                    unique_ratio[index, gear] = float(rows[index]["unique_pixel_ratio"])
                    densities[index, gear] = unique_ratio[index, gear]
                    object_edge_f1[index, gear] = float(rows[index].get("object_edge_f1", 0.0))
                    target_purity[index, gear] = float(rows[index].get("target_purity", 0.0))
                    object_area[index] = float(rows[index].get("object_area", 0.0))
            has_object = bool(object_edge_f1.any())
            wins_360 = int(np.sum(edge_f1[:, GEAR_360] + 1e-12 >= edge_f1[:, GEAR_180]))
            output = args.output_dir / f"{split}_{scene}_{regime}.npz"
            np.savez_compressed(
                output,
                visual_motion=visual.astype(np.float64),
                imu_motion=imu.astype(np.float64),
                imu_raw=imu_raw.astype(np.float64),
                densities=densities,
                unique_ratio=unique_ratio,
                temporal_overlap=overlap,
                edge_f1=edge_f1,
                object_edge_f1=object_edge_f1,
                target_purity=target_purity,
                object_area=object_area,
                timestamps=timestamps[1 : count + 1],
            )
            records.append({"scene": scene_id, "condition": regime, "split": split, "output": str(output), "windows": count})
            teacher_stats[scene_id] = {
                "windows": count,
                "mean_f1_180": float(edge_f1[:, GEAR_180].mean()),
                "mean_f1_360": float(edge_f1[:, GEAR_360].mean()),
                "oracle_argmax_f1": float(np.maximum(edge_f1[:, GEAR_180], edge_f1[:, GEAR_360]).mean()),
                "teacher_360_frac": wins_360 / count,
                "mean_visual_motion": float(visual.mean()),
                "mean_imu_motion": float(imu.mean()),
                "mean_unique_ratio_180": float(unique_ratio[:, GEAR_180].mean()),
                "mean_unique_ratio_360": float(unique_ratio[:, GEAR_360].mean()),
                "has_zero_gear": GEAR_0 in tables,
                "has_object_metric": has_object,
                "mean_f1_0": float(edge_f1[:, GEAR_0].mean()) if GEAR_0 in tables else None,
                "mean_object_f1": {rpm: float(object_edge_f1[:, gear].mean()) for rpm, gear in (("0", GEAR_0), ("180", GEAR_180), ("360", GEAR_360)) if gear in tables} if has_object else None,
                "object_argmax_frac": {
                    rpm: float(np.mean(np.argmax(object_edge_f1[:, [GEAR_0, GEAR_180, GEAR_360]], axis=1) == column))
                    for column, rpm in enumerate(("0", "180", "360"))
                }
                if has_object and GEAR_0 in tables
                else None,
            }
            print(f"wrote {output} {json.dumps(teacher_stats[scene_id], sort_keys=True)}")

    manifest = {
        "dataset": "EVEYES-co-motion-v2-regime-f1-oracles",
        "source_resolution": [960, 540],
        "eval_width": 960,
        "native_resolution": True,
        "prism_angle_deg": 0.5,
        "max_frames": args.max_frames,
        "feature_dim": FEATURE_DIM,
        "gear_rpm": GEAR_RPM.tolist(),
        "gear_360_index": GEAR_360,
        "gear_180_index": GEAR_180,
        "gear_0_index": GEAR_0,
        "imu_motion_bounds": list(imu_bounds),
        "imu_bounds_source": "train/Scene_002/co_motion gyro norm",
        "imu_timestamp_unit": "s",
        "storm_density_feature": "unique_pixel_ratio",
        "temporal_overlap_in_oracle": "zeros_overlap_was_not_stored_in_table1_json",
        "not_the_64x64_procedural_pack": True,
        "teacher_source": "native_960x540_ami_fixed_180_and_360_window_edge_f1",
        "regimes": list(args.regimes),
        "oracles": records,
        "teacher_stats": teacher_stats,
    }
    path = args.output_dir / "manifest.json"
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"manifest": str(path), "imu_motion_bounds": list(imu_bounds), "teacher_stats": teacher_stats}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
