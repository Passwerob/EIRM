#!/usr/bin/env python3
"""Audit scene-disjoint EveYES RGB/timestamp/IMU contracts and motion state."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from ami_active_sensing.data import discover_eveyes_scenes, load_scene_imu, load_scene_rgb
from ami_active_sensing.motion.motion_features import extract_flow_features, extract_imu_features
from ami_active_sensing.motion.optical_flow import FarnebackFlowBackend


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("NEW/results/eveyes_imu_audit.json"))
    parser.add_argument("--sample-stride", type=int, default=30)
    args = parser.parse_args()
    if args.sample_stride <= 0:
        raise SystemExit("sample-stride must be positive")

    scenes = discover_eveyes_scenes(args.dataset)
    flow_backend = FarnebackFlowBackend()
    records: list[dict[str, object]] = []
    for scene in scenes:
        imu = load_scene_imu(scene)
        imu_features = extract_imu_features(imu)
        indices = list(range(0, len(scene.rgb_paths), args.sample_stride))
        if indices[-1] != len(scene.rgb_paths) - 1:
            indices.append(len(scene.rgb_paths) - 1)
        frames = load_scene_rgb(scene, indices)
        sampled_times = scene.timestamps[indices]
        flow_features: list[dict[str, float]] = []
        for index in range(1, len(frames)):
            result = flow_backend.estimate(frames[index - 1], frames[index], dt_seconds=float(sampled_times[index] - sampled_times[index - 1]))
            flow_features.append(extract_flow_features(result, low_motion_threshold_px_s=1.0, magnitude_histogram_max_px_s=100.0, reference_intensity=frames[index - 1]))
        flow_keys = tuple(flow_features[0]) if flow_features else ()
        mean_flow = {key: float(np.mean([row[key] for row in flow_features])) for key in flow_keys}
        records.append({
            "scene_id": scene.scene_id,
            "split": scene.split,
            "motion_regime": scene.motion_regime,
            "sequence_id": scene.sequence_id,
            "frame_count": len(scene.rgb_paths),
            "duration_s": float(scene.timestamps[-1] - scene.timestamps[0]),
            "rgb_shape": [scene.intrinsics.height, scene.intrinsics.width, 3],
            "imu_sample_count": len(imu.t),
            "imu_nonzero_gyro_fraction": float(np.mean(np.linalg.norm(imu.gyro, axis=1) > 1e-9)),
            "imu_features": imu_features,
            "sample_stride": args.sample_stride,
            "flow_sample_count": len(flow_features),
            "mean_flow_features": mean_flow,
            "motion_config": scene.motion_config,
        })
        print(json.dumps({"sequence_id": scene.sequence_id, "imu_nonzero_gyro_fraction": records[-1]["imu_nonzero_gyro_fraction"], "flow_mean_px_s": mean_flow.get("flow_mean_px_s", 0.0)}), flush=True)

    result = {
        "status": "synthetic_rendered_dataset_contract_audit",
        "provenance": "Blender-rendered RGB with trajectory-derived deterministic IMU",
        "dataset": str(args.dataset),
        "scene_count": len(records),
        "split_counts": {split: sum(row["split"] == split for row in records) for split in sorted({str(row["split"]) for row in records})},
        "records": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False), encoding="utf-8")
    print(json.dumps({"output": str(args.output), "scene_count": len(records), "split_counts": result["split_counts"]}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
