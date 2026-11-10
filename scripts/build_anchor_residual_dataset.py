#!/usr/bin/env python3
"""Collect deployment-state residual labels around a fixed 180 RPM anchor.

Each scene is rolled out with the anchor action. At every causal window, all
candidate actions are simulated at the exact current prism phase and their
useful-objective differences to the anchor are stored. This closes the
oracle-path/state-distribution mismatch of the earlier Q-policy dataset.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from scripts.evaluate_phase_continuous_control import GEAR_RPM, _load_scene, _simulate_action
from ami_simulator.geometry import RotatingWedgePrism


ANCHOR_GEAR = 3  # GEAR_RPM[3] == 180 RPM
RESIDUAL_GEAR_INDICES = (4, 2)  # 0 RPM and 360 RPM
FEATURE_DIM = 20  # 9 event/IMU state + 8 flow + 2 phase + dwell


def _state(
    oracle: np.lib.npyio.NpzFile,
    *,
    index: int,
    previous: dict[str, float] | None,
    phase: float,
    dwell_seconds: float,
    previous_rpm: float,
) -> np.ndarray:
    values = np.zeros(FEATURE_DIM, dtype=np.float32)
    if previous is not None:
        values[0] = float(oracle["imu_motion"][index - 1])
        values[1] = float(previous["event_density"]) / 10.0
        values[2] = float(previous["edge_coverage"])
        values[3] = float(previous["structural_ratio"])
        values[4] = float(previous["unique_pixel_ratio"])
        values[5] = float(previous["temporal_overlap"])
        values[6] = float(previous["polarity_imbalance"])
        values[7] = float(previous["event_efficiency"])
        values[8] = float(previous_rpm / GEAR_RPM[0])
        if "flow_features" in oracle:
            values[9:17] = np.asarray(oracle["flow_features"][index - 1], dtype=np.float32)
    values[17] = float(np.sin(phase))
    values[18] = float(np.cos(phase))
    values[19] = float(dwell_seconds)
    return values


def _collect_scene(
    scene: Path,
    oracle_path: Path,
    objective: dict[str, float],
    *,
    max_phase_step_deg: float,
) -> dict[str, np.ndarray | str | float | int]:
    luminance, timestamps, masks, intrinsics, oracle = _load_scene(scene, oracle_path)
    try:
        count = len(timestamps) - 1
        if "flow_features" not in oracle:
            raise ValueError(f"Oracle {oracle_path} lacks required flow_features.")
        prism = RotatingWedgePrism(intrinsics, 2.0)
        states = np.zeros((count, FEATURE_DIM), dtype=np.float32)
        candidate_utilities = np.zeros((count, len(RESIDUAL_GEAR_INDICES) + 1), dtype=np.float32)
        candidate_event_efficiency = np.zeros((count, len(RESIDUAL_GEAR_INDICES) + 1), dtype=np.float32)
        candidate_edge_f1 = np.zeros((count, len(RESIDUAL_GEAR_INDICES) + 1), dtype=np.float32)
        candidate_edge_coverage = np.zeros((count, len(RESIDUAL_GEAR_INDICES) + 1), dtype=np.float32)
        anchor_utilities = np.zeros(count, dtype=np.float32)
        phase = 0.0
        previous_active = None
        previous_metrics = None
        previous_rpm = float(GEAR_RPM[ANCHOR_GEAR])
        dwell_seconds = 0.0
        for index in range(count):
            states[index] = _state(oracle, index=index, previous=previous_metrics, phase=phase, dwell_seconds=dwell_seconds, previous_rpm=previous_rpm)
            candidates = [
                _simulate_action(
                    luminance,
                    timestamps,
                    masks,
                    prism,
                    index=index,
                    rpm=float(rpm),
                    phase=phase,
                    previous_active=previous_active,
                    previous_rpm=previous_rpm,
                    target_density=float(oracle["target_density"][index]),
                    objective=objective,
                    contrast_threshold=0.2,
                    max_phase_step_deg=max_phase_step_deg,
                )
                for rpm in (0.0, 180.0, 360.0)
            ]
            anchor = candidates[1]
            anchor_utilities[index] = float(anchor["utility"])
            candidate_utilities[index] = np.asarray([candidate["utility"] for candidate in candidates], dtype=np.float32)
            candidate_event_efficiency[index] = np.asarray([candidate["metrics"]["event_efficiency"] for candidate in candidates], dtype=np.float32)
            candidate_edge_f1[index] = np.asarray([candidate["metrics"]["edge_f1"] for candidate in candidates], dtype=np.float32)
            candidate_edge_coverage[index] = np.asarray([candidate["metrics"]["edge_coverage"] for candidate in candidates], dtype=np.float32)
            previous_metrics = {key: float(anchor["metrics"][key]) for key in ("event_density", "edge_coverage", "structural_ratio", "unique_pixel_ratio", "temporal_overlap", "polarity_imbalance", "event_efficiency")}
            previous_active = anchor["active"]
            phase = float(anchor["phase_end"])
            previous_rpm = 180.0
            dwell_seconds += float(timestamps[index + 1] - timestamps[index])
        return {
            "features": states,
            "candidate_rpm": np.asarray([0.0, 180.0, 360.0], dtype=np.float64),
            "candidate_utility": candidate_utilities,
            "candidate_event_efficiency": candidate_event_efficiency,
            "candidate_edge_f1": candidate_edge_f1,
            "candidate_edge_coverage": candidate_edge_coverage,
            "anchor_utility": anchor_utilities,
            "anchor_event_efficiency": candidate_event_efficiency[:, 1],
            "anchor_edge_f1": candidate_edge_f1[:, 1],
            "anchor_edge_coverage": candidate_edge_coverage[:, 1],
            "delta_utility": candidate_utilities[:, (0, 2)] - anchor_utilities[:, None],
            "delta_event_efficiency": candidate_event_efficiency[:, (0, 2)] - candidate_event_efficiency[:, 1, None],
            "delta_edge_f1": candidate_edge_f1[:, (0, 2)] - candidate_edge_f1[:, 1, None],
            "delta_edge_coverage": candidate_edge_coverage[:, (0, 2)] - candidate_edge_coverage[:, 1, None],
            "imu_motion": np.asarray(oracle["imu_motion"], dtype=np.float32),
            "timestamps": np.asarray(timestamps[1:], dtype=np.float64),
            "scene": scene.name,
        }
    finally:
        oracle.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("results/comotion_v2_learned/manifest.json"))
    parser.add_argument("--dataset-root", type=Path, default=Path("data/comotion_controlled_v2"))
    parser.add_argument("--output-root", type=Path, default=Path("results/comotion_anchor_residual_dataset"))
    parser.add_argument("--split", choices=("train", "valid", "test", "all"), default="all")
    parser.add_argument("--max-scenes-per-split", type=int, default=None, help="Optional cap for a fast prototype; scenes are taken in manifest order.")
    parser.add_argument("--max-phase-step-deg", type=float, default=30.0)
    parser.add_argument("--event-efficiency-weight", type=float, default=None, help="Override the source objective weight; zero keeps repeated-event efficiency diagnostic-only.")
    args = parser.parse_args()
    if args.max_phase_step_deg <= 0.0 or (args.max_scenes_per_split is not None and args.max_scenes_per_split < 1) or (args.event_efficiency_weight is not None and args.event_efficiency_weight < 0.0):
        parser.error("max-phase-step-deg must be positive and max-scenes-per-split must be positive")
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    objective = dict(manifest["configuration"])
    if args.event_efficiency_weight is not None:
        objective["event_efficiency_weight"] = float(args.event_efficiency_weight)
    records: list[dict[str, object]] = []
    split_counts: dict[str, int] = {}
    for record in manifest["oracles"]:
        split = str(record["split"])
        if args.split != "all" and split != args.split:
            continue
        if args.max_scenes_per_split is not None and split_counts.get(split, 0) >= args.max_scenes_per_split:
            continue
        split_counts[split] = split_counts.get(split, 0) + 1
        output = args.output_root / split / f"{record['scene']}.npz"
        output.parent.mkdir(parents=True, exist_ok=True)
        if output.exists():
            records.append({"split": split, "scene": record["scene"], "condition": record["condition"], "seed": record["seed"], "output": str(output), "windows": int(record["windows"]), "reused": True})
            continue
        collected = _collect_scene(args.dataset_root / str(record["scene"]), Path(str(record["output"])), objective, max_phase_step_deg=args.max_phase_step_deg)
        np.savez_compressed(output, **{key: value for key, value in collected.items() if isinstance(value, np.ndarray)})
        records.append({"split": split, "scene": record["scene"], "condition": record["condition"], "seed": record["seed"], "output": str(output), "windows": int(record["windows"]), "reused": False})
        print(f"wrote {output}")
    output_manifest = {"dataset": "AMI anchor residual deployment-state dataset", "source_manifest": str(args.manifest), "configuration": objective, "anchor_rpm": 180.0, "candidate_rpm": [0.0, 180.0, 360.0], "feature_dim": FEATURE_DIM, "feature_layout": {"imu_and_event_state": [0, 9], "flow": [9, 17], "phase_sin_cos": [17, 19], "dwell_seconds": 19}, "phase_policy": "anchor_phase_integrated_after_every_180rpm_action", "records": records}
    args.output_root.mkdir(parents=True, exist_ok=True)
    (args.output_root / "manifest.json").write_text(json.dumps(output_manifest, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"records": len(records), "output_root": str(args.output_root)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
