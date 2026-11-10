#!/usr/bin/env python3
"""DEPRECATED for the camera-speed experiment.

Prism-only inverse RPM. The camera-motion branch is
scripts/evaluate_eveyes_camera_speed_nfb.py
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

from ami_simulator.geometry import RotatingWedgePrism
from ami_simulator.inverse_motion import spearman_rho
from ami_simulator.pipeline import load_rgb_sequence, rgb_to_linear_luminance
from scripts.evaluate_eveyes_v2_downstream import _intrinsics, _scene_path, _structure_metrics
from scripts.evaluate_phase_continuous_control import _simulate_action

READY = Path("data/eveyes_motion_nfb_ready")
OBJECTIVE = {
    "event_efficiency_weight": 0.0,
    "overlap_weight": 0.0,
    "density_weight": 0.0,
    "invalid_fraction_weight": 0.0,
    "rpm_effort_weight": 0.0,
    "rpm_change_weight": 0.0,
}
SPEED_CUT_MPS = 0.5
STORM_ON = 0.45
STORM_OFF = 0.38
DWELL_S = 0.25
FLOOR_RPM = 180.0
FAST_RPM = 360.0


def _load_speed(scene: Path, n_windows: int) -> np.ndarray:
    rows = list(csv.DictReader((scene / "pose_speed.csv").open(encoding="utf-8")))
    speed = np.asarray([float(row["speed_mps"]) for row in rows], dtype=np.float64)
    if len(speed) < n_windows:
        raise SystemExit(f"{scene} pose_speed.csv has {len(speed)} rows, need {n_windows}")
    return speed[:n_windows]


def _choose_rpm(
    *,
    speed_mps: float,
    occupancy: float,
    previous_rpm: float,
    storm_latch: bool,
    now: float,
    last_switch: float,
    storm_brake: bool,
) -> tuple[float, bool]:
    latch = bool(storm_latch)
    if storm_brake:
        if occupancy >= STORM_ON:
            latch = True
        elif occupancy <= STORM_OFF:
            latch = False
    want = FAST_RPM if speed_mps < SPEED_CUT_MPS and not latch else FLOOR_RPM
    if now - last_switch < DWELL_S:
        return float(previous_rpm), latch
    return float(want), latch


def _rollout(scene: Path, *, storm_brake: bool, device: str, max_frames: int) -> dict[str, object]:
    frames, timestamps = load_rgb_sequence(scene / "images", scene / "timestamps.csv", timestamp_unit="s", max_frames=max_frames)
    timestamps = timestamps[: len(frames)]
    luminance = rgb_to_linear_luminance(frames)
    n_windows = len(timestamps) - 1
    speed = _load_speed(scene, n_windows)
    valid = np.ones((len(luminance), luminance.shape[1], luminance.shape[2]), dtype=bool)
    prism = RotatingWedgePrism(_intrinsics(scene), 0.5)
    phase = 0.0
    previous_active = None
    sensor_reference = None
    previous_rpm = FLOOR_RPM
    last_switch = float(timestamps[0]) - DWELL_S
    storm_latch = False
    occupancy = 0.0
    rows: list[dict[str, object]] = []
    for index in range(n_windows):
        rpm, storm_latch = _choose_rpm(
            speed_mps=float(speed[index]),
            occupancy=occupancy,
            previous_rpm=previous_rpm,
            storm_latch=storm_latch,
            now=float(timestamps[index]),
            last_switch=last_switch,
            storm_brake=storm_brake,
        )
        if rpm != previous_rpm:
            last_switch = float(timestamps[index])
        result = _simulate_action(
            luminance,
            timestamps,
            valid,
            prism,
            index=index,
            rpm=rpm,
            phase=phase,
            previous_active=previous_active,
            previous_rpm=previous_rpm,
            target_density=0.2,
            objective=OBJECTIVE,
            contrast_threshold=0.2,
            max_phase_step_deg=30.0,
            sensor_reference_log=sensor_reference,
            device=device,
        )
        metrics, previous_active, _ = _structure_metrics(
            result["reference_events"], luminance[index], valid[index], previous_active
        )
        occupancy = float(metrics["unique_pixel_ratio"])
        phase = float(result["phase_end"])
        sensor_reference = np.asarray(result["sensor_reference_log"])
        rows.append(
            {
                "window": index,
                "t0": float(timestamps[index]),
                "t1": float(timestamps[index + 1]),
                "speed_mps": float(speed[index]),
                "rpm": float(rpm),
                "storm_latch": bool(storm_latch),
                **metrics,
            }
        )
        previous_rpm = float(rpm)
    rpms = np.asarray([row["rpm"] for row in rows], dtype=np.float64)
    speeds = np.asarray([row["speed_mps"] for row in rows], dtype=np.float64)
    occ = np.asarray([row["unique_pixel_ratio"] for row in rows], dtype=np.float64)
    return {
        "n_windows": n_windows,
        "mean_rpm": float(rpms.mean()),
        "frac_360": float(np.mean(rpms >= 330.0)),
        "n_switches": int(np.sum(np.abs(np.diff(rpms)) > 1.0)),
        "mean_speed_mps": float(speeds.mean()),
        "mean_occupancy": float(occ.mean()),
        "mean_edge_f1": float(np.mean([row["edge_f1"] for row in rows])),
        "mean_event_count": float(np.mean([row["event_count"] for row in rows])),
        "spearman_speed_rpm": spearman_rho(speeds, rpms) if len(rpms) >= 3 else 0.0,
        "rows": rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=READY)
    parser.add_argument("--output", type=Path, default=Path("results/eveyes_motion_nfb/test_rollout.json"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-frames", type=int, default=480)
    parser.add_argument("--splits", nargs="+", default=("test",))
    parser.add_argument(
        "--regimes",
        nargs="+",
        default=("object_only", "camera_only", "co_motion", "drive_stop_drive"),
    )
    args = parser.parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit("cuda requested but unavailable")
    torch.set_num_threads(8)
    scenes = [
        (split, "Scene_005" if split == "test" else ("Scene_002" if split == "train" else "Scene_004"), regime)
        for split in args.splits
        for regime in args.regimes
    ]
    policies = (
        ("inverse_speed", False),
        ("inverse_speed_storm", True),
    )
    report: dict[str, object] = {
        "dataset": str(args.dataset_root),
        "actuator": "prism_rpm",
        "motion_signal": "pose_speed_mps",
        "speed_cut_mps": SPEED_CUT_MPS,
        "storm_on": STORM_ON,
        "storm_off": STORM_OFF,
        "dwell_s": DWELL_S,
        "not_m0": True,
        "scenes": {},
    }
    for split, scene, regime in scenes:
        path = _scene_path(args.dataset_root, split, scene, regime)
        scene_id = f"{split}/{scene}/{regime}"
        report["scenes"][scene_id] = {}
        for name, storm_brake in policies:
            print(f"{scene_id}  {name}", flush=True)
            payload = _rollout(path, storm_brake=storm_brake, device=args.device, max_frames=args.max_frames)
            summary = {key: value for key, value in payload.items() if key != "rows"}
            report["scenes"][scene_id][name] = payload
            print(json.dumps({"scene": scene_id, "policy": name, **summary}, sort_keys=True), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(args.output)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
