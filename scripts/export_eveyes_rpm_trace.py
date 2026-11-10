#!/usr/bin/env python3
"""Export Ours RPM vs time so switching is a figure, not a mean.

Closed-loop traces come from evaluate_eveyes_v2_downstream.py rows.
The concatenated object → camera → object trace is a policy I/O plot:
it shows the net returning to 180 when visual motion collapses again.
Caption that splice as three takes, not one Blender shot.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from ami_simulator.motion_net import TinyCausalGearPolicy
from scripts.evaluate_inverse_motion_control import _choose_rpm
from scripts.evaluate_phase_continuous_control import GEAR_RPM
from scripts.train_inverse_motion_policy import FEATURE_DIM


def _ours_rows(payload: dict[str, object], scene: str) -> list[dict[str, object]]:
    rows = [row for row in payload["rows"] if row["scene"] == scene and row["method"] == "ours"]
    rows.sort(key=lambda row: int(row["window"]))
    return rows


def _bin_mean(values: np.ndarray, width: int) -> list[float]:
    return [float(values[index : index + width].mean()) for index in range(0, len(values), width)]


def _closed_loop(payload: dict[str, object], scene: str, bin_frames: int) -> dict[str, object] | None:
    rows = _ours_rows(payload, scene)
    if not rows:
        return None
    rpm = np.asarray([float(row["rpm"]) for row in rows], dtype=np.float64)
    switches = int(np.sum(rpm[1:] != rpm[:-1])) if len(rpm) > 1 else 0
    return {
        "scene": scene,
        "windows": len(rpm),
        "mean_rpm": float(rpm.mean()),
        "frac_360": float(np.mean(rpm >= 359.0)),
        "switches": switches,
        "bin_seconds": bin_frames / 60.0,
        "binned_mean_rpm": [round(value, 1) for value in _bin_mean(rpm, bin_frames)],
    }


def _concat_policy(
    checkpoint: dict[str, object],
    model: torch.nn.Module,
    paths: list[tuple[Path, str]],
    *,
    dwell: float,
    confidence: float,
) -> dict[str, object]:
    visual_parts: list[np.ndarray] = []
    imu_parts: list[np.ndarray] = []
    occupancy_parts: list[np.ndarray] = []
    tags: list[str] = []
    for path, tag in paths:
        with np.load(path) as data:
            visual_parts.append(np.asarray(data["visual_motion"], dtype=np.float64))
            imu_parts.append(np.asarray(data["imu_motion"], dtype=np.float64))
            occupancy_parts.append(np.asarray(data["unique_ratio"][:, int(np.argmin(np.abs(GEAR_RPM - 180.0)))], dtype=np.float64))
            tags.extend([tag] * len(data["visual_motion"]))
    visual = np.concatenate(visual_parts)
    imu = np.concatenate(imu_parts)
    occupancy = np.concatenate(occupancy_parts)
    gears = tuple(float(value) for value in checkpoint["candidate_rpm"])
    thresholds = tuple(float(value) for value in checkpoint["motion_thresholds"])
    features: list[np.ndarray] = []
    rpms: list[float] = []
    previous_occupancy = 0.0
    previous_rpm = 180.0
    last_switch = -dwell
    dt = 1.0 / 60.0
    for index in range(len(visual)):
        motion = float(visual[index - 1] if index else 0.0)
        imu_value = float(imu[index - 1] if index else 0.0)
        features.append(np.asarray((motion, imu_value, previous_occupancy, 0.0), dtype=np.float32))
        rpm = _choose_rpm(
            "inverse_learned",
            motion=motion,
            thresholds=thresholds,
            previous_density=previous_occupancy,
            previous_overlap=0.0,
            features=features,
            model=model,
            checkpoint=checkpoint,
            gears=gears,
            quality_floor=bool(checkpoint.get("quality_floor", False)),
            teacher=str(checkpoint.get("teacher", "f1-argmax")),
            confidence=confidence,
            require_imu=False,
        )
        time = index * dt
        if index and time - last_switch < dwell:
            rpm = previous_rpm
        if rpm != previous_rpm:
            last_switch = time
        rpms.append(float(rpm))
        previous_rpm = float(rpm)
        previous_occupancy = float(occupancy[index])
    rpm_values = np.asarray(rpms, dtype=np.float64)
    sticky = np.empty_like(rpm_values)
    hold = 180.0
    for index, motion in enumerate(np.r_[0.0, visual[:-1]]):
        if motion < 2.0:
            hold = 180.0
        elif motion > 20.0:
            hold = 360.0
        sticky[index] = hold
    seconds = len(rpm_values) // 60
    return {
        "order": [tag for _, tag in paths],
        "dwell_seconds": dwell,
        "windows": len(rpm_values),
        "switches": int(np.sum(rpm_values[1:] != rpm_values[:-1])),
        "visual_hysteresis_switches": int(np.sum(sticky[1:] != sticky[:-1])),
        "binned_mean_rpm": [round(float(rpm_values[index * 60 : (index + 1) * 60].mean()), 1) for index in range(seconds)],
        "binned_visual_hysteresis_rpm": [round(float(sticky[index * 60 : (index + 1) * 60].mean()), 1) for index in range(seconds)],
        "binned_visual": [round(float(visual[index * 60 : (index + 1) * 60].mean()), 2) for index in range(seconds)],
        "note": "object_only is replayed after camera_only to show the downshift. Caption as spliced takes.",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ours-json", type=Path, default=Path("results/ami_next/eveyes_v2_regimes_test_ours_m0.json"))
    parser.add_argument("--student", type=Path, default=Path("results/eveyes_v2_regime_f1_argmax_m0/policy.pt"))
    parser.add_argument("--oracle-dir", type=Path, default=Path("results/eveyes_v2_regime_f1_oracles"))
    parser.add_argument("--output", type=Path, default=Path("results/ami_next/eveyes_v2_rpm_trace.json"))
    parser.add_argument("--dwell", type=float, default=0.30)
    parser.add_argument("--confidence", type=float, default=0.50)
    args = parser.parse_args()
    payload = json.loads(args.ours_json.read_text(encoding="utf-8")) if args.ours_json.exists() else {"rows": []}
    checkpoint = torch.load(args.student, map_location="cpu", weights_only=False)
    model = TinyCausalGearPolicy(feature_dim=FEATURE_DIM, hidden_dim=int(checkpoint["hidden_dim"]), num_gears=int(checkpoint["num_gears"]))
    model.load_state_dict(checkpoint["model"])
    model.eval()
    oracle = args.oracle_dir
    report = {
        "student": str(args.student),
        "ours_json": str(args.ours_json),
        "closed_loop": {
            scene: _closed_loop(payload, scene, 30)
            for scene in ("test/Scene_005/object_only", "test/Scene_005/camera_only", "test/Scene_005/co_motion")
        },
        "spliced_object_camera_object": _concat_policy(
            checkpoint,
            model,
            [
                (oracle / "test_Scene_005_object_only.npz", "object_only"),
                (oracle / "test_Scene_005_camera_only.npz", "camera_only"),
                (oracle / "test_Scene_005_object_only.npz", "object_only"),
            ],
            dwell=args.dwell,
            confidence=args.confidence,
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("closed_loop", "spliced_object_camera_object")}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
