#!/usr/bin/env python3
"""One-step camera-path NFB oracles.

Default teacher is occupancy deficit + F1 slack so 1.5× is used when events
are scarce. For each window the teacher scores {0.5, 1.0, 1.5}× from the
current path state, then executes the winning rate. Prism RPM is frozen
(default 180) so labels isolate the camera actuator.
Does not overwrite Table 1 / m0 or the rule-based camera_speed_nfb.json.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from ami_simulator.camera_nfb import FEATURE_NAMES, load_pose_speed, pair_visual_motion, resize_luminance, sample_path, sample_pose_speed
from ami_simulator.geometry import RotatingWedgePrism
from ami_simulator.inverse_motion import CAMERA_DEFICIT_OCCUPANCY, CAMERA_HOLD_RATE, CAMERA_RATES, CAMERA_STORM_OCCUPANCY, CAMERA_UPSHIFT_SLACK, deficit_camera_rate, f1_argmax_camera_rate
from ami_simulator.pipeline import load_rgb_sequence, rgb_to_linear_luminance
from scripts.evaluate_eveyes_camera_speed_nfb import DT, OBJECTIVE
from scripts.evaluate_eveyes_v2_downstream import _intrinsics, _scene_path, _structure_metrics
from scripts.evaluate_phase_continuous_control import _simulate_action

SCENES = (("train", "Scene_002"), ("valid", "Scene_004"), ("test", "Scene_005"))
REGIMES = ("co_motion", "camera_only", "object_only", "drive_stop_drive")


def _simulate_rate(
    luminance: np.ndarray,
    path_s: float,
    rate: float,
    *,
    now: float,
    prism: RotatingWedgePrism,
    rpm: float,
    phase: float,
    previous_active,
    previous_rpm: float,
    sensor_reference,
    device: str,
) -> dict[str, object]:
    last = len(luminance) - 1
    used = float(min(rate, last - path_s))
    lum_a = sample_path(luminance, path_s)
    lum_b = sample_path(luminance, path_s + used)
    pair = np.stack((lum_a, lum_b), axis=0)
    times = np.asarray((now, now + DT), dtype=np.float64)
    valid = np.ones((2, pair.shape[1], pair.shape[2]), dtype=bool)
    result = _simulate_action(
        pair,
        times,
        valid,
        prism,
        index=0,
        rpm=float(rpm),
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
    metrics, active, _ = _structure_metrics(result["reference_events"], lum_a, valid[0], previous_active)
    return {
        "result": result,
        "metrics": metrics,
        "active": active,
        "visual_motion": pair_visual_motion(lum_a, lum_b, DT),
        "used_rate": used,
    }


def _rollout_scene(
    scene: Path,
    *,
    rpm: float,
    prism_angle_deg: float,
    eval_width: int,
    max_frames: int,
    max_windows: int,
    storm_occupancy: float,
    upshift_margin: float,
    deficit_occupancy: float,
    upshift_slack: float,
    teacher: str,
    device: str,
) -> dict[str, object]:
    frames, timestamps = load_rgb_sequence(scene / "images", scene / "timestamps.csv", timestamp_unit="s", max_frames=max_frames)
    timestamps = timestamps[: len(frames)]
    luminance = rgb_to_linear_luminance(frames)
    intrinsics = _intrinsics(scene)
    luminance, intrinsics = resize_luminance(luminance, intrinsics, eval_width)
    speed_t, speed_v = load_pose_speed(scene)
    prism = RotatingWedgePrism(intrinsics, prism_angle_deg)
    last = len(luminance) - 1
    path_s = 0.0
    occupancy = 0.0
    phase = 0.0
    previous_active = None
    sensor_reference = None
    previous_rpm = float(rpm)
    rows: list[dict[str, object]] = []
    for window in range(max_windows):
        remaining = last - path_s
        if remaining < 0.5 - 1e-9:
            break
        now = window * DT
        branches: dict[float, dict[str, object]] = {}
        for rate in CAMERA_RATES:
            branches[rate] = _simulate_rate(
                luminance,
                path_s,
                rate,
                now=now,
                prism=prism,
                rpm=rpm,
                phase=phase,
                previous_active=previous_active,
                previous_rpm=previous_rpm,
                sensor_reference=sensor_reference,
                device=device,
            )
        if teacher == "deficit":
            label = deficit_camera_rate(
                float(branches[0.5]["metrics"]["edge_f1"]),
                float(branches[1.0]["metrics"]["edge_f1"]),
                float(branches[1.5]["metrics"]["edge_f1"]),
                previous_occupancy=occupancy,
                deficit_occupancy=deficit_occupancy,
                storm_occupancy=storm_occupancy,
                upshift_slack=upshift_slack,
                upshift_margin=upshift_margin,
            )
        else:
            label = f1_argmax_camera_rate(
                float(branches[0.5]["metrics"]["edge_f1"]),
                float(branches[1.0]["metrics"]["edge_f1"]),
                float(branches[1.5]["metrics"]["edge_f1"]),
                previous_occupancy=occupancy,
                upshift_margin=upshift_margin,
                storm_occupancy=storm_occupancy,
            )
        chosen = branches[label]
        metrics = chosen["metrics"]
        rows.append(
            {
                "window": window,
                "t0": now,
                "path_s": float(path_s),
                "label_rate": float(label),
                "executed_rate": float(chosen["used_rate"]),
                "visual_motion": float(chosen["visual_motion"]),
                "speed_mps": sample_pose_speed(speed_t, speed_v, path_s, timestamps),
                "occupancy": float(metrics["unique_pixel_ratio"]),
                "edge_f1": np.asarray([float(branches[rate]["metrics"]["edge_f1"]) for rate in CAMERA_RATES], dtype=np.float64),
                "unique_pixel_ratio": np.asarray([float(branches[rate]["metrics"]["unique_pixel_ratio"]) for rate in CAMERA_RATES], dtype=np.float64),
                "event_count": np.asarray([float(branches[rate]["metrics"]["event_count"]) for rate in CAMERA_RATES], dtype=np.float64),
            }
        )
        path_s = min(last, path_s + float(chosen["used_rate"]))
        occupancy = float(metrics["unique_pixel_ratio"])
        phase = float(chosen["result"]["phase_end"])
        sensor_reference = np.asarray(chosen["result"]["sensor_reference_log"])
        previous_active = chosen["active"]
        previous_rpm = float(rpm)
        if window % 40 == 0:
            print(f"  w={window} path={path_s:.1f} label={label:.1f} occ={occupancy:.3f} f1={metrics['edge_f1']:.3f}", flush=True)
    labels = np.asarray([row["label_rate"] for row in rows], dtype=np.float64)
    return {
        "n_windows": len(rows),
        "path_progress": float(path_s / last) if last else 0.0,
        "label_frac": {str(rate): float(np.mean(labels == rate)) if len(labels) else 0.0 for rate in CAMERA_RATES},
        "mean_executed_rate": float(np.mean([row["executed_rate"] for row in rows])) if rows else 0.0,
        "mean_occupancy": float(np.mean([row["occupancy"] for row in rows])) if rows else 0.0,
        "mean_teacher_f1": float(np.mean([row["edge_f1"][CAMERA_RATES.index(row["label_rate"])] for row in rows])) if rows else 0.0,
        "rows": rows,
    }


def _write_npz(path: Path, payload: dict[str, object]) -> None:
    rows = payload["rows"]
    np.savez_compressed(
        path,
        path_s=np.asarray([row["path_s"] for row in rows], dtype=np.float64),
        label_rate=np.asarray([row["label_rate"] for row in rows], dtype=np.float64),
        executed_rate=np.asarray([row["executed_rate"] for row in rows], dtype=np.float64),
        visual_motion=np.asarray([row["visual_motion"] for row in rows], dtype=np.float64),
        speed_mps=np.asarray([row["speed_mps"] for row in rows], dtype=np.float64),
        occupancy=np.asarray([row["occupancy"] for row in rows], dtype=np.float64),
        edge_f1=np.stack([row["edge_f1"] for row in rows]) if rows else np.zeros((0, 3), dtype=np.float64),
        unique_pixel_ratio=np.stack([row["unique_pixel_ratio"] for row in rows]) if rows else np.zeros((0, 3), dtype=np.float64),
        event_count=np.stack([row["event_count"] for row in rows]) if rows else np.zeros((0, 3), dtype=np.float64),
        candidate_rates=np.asarray(CAMERA_RATES, dtype=np.float64),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=Path("data/eveyes_motion_nfb_ready"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/eveyes_motion_nfb/camera_nfb_oracles_deficit"))
    parser.add_argument("--splits", nargs="+", default=("train", "valid", "test"))
    parser.add_argument("--regimes", nargs="+", default=("co_motion",))
    parser.add_argument("--prism-rpm", type=float, default=180.0)
    parser.add_argument("--prism-angle-deg", type=float, default=0.5)
    parser.add_argument("--eval-width", type=int, default=320)
    parser.add_argument("--max-frames", type=int, default=480)
    parser.add_argument("--max-windows", type=int, default=1000)
    parser.add_argument("--teacher", choices=("deficit", "f1-argmax"), default="deficit")
    parser.add_argument("--storm-occupancy", type=float, default=CAMERA_STORM_OCCUPANCY)
    parser.add_argument("--deficit-occupancy", type=float, default=CAMERA_DEFICIT_OCCUPANCY)
    parser.add_argument("--upshift-slack", type=float, default=CAMERA_UPSHIFT_SLACK)
    parser.add_argument("--upshift-margin", type=float, default=0.0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--merge", action="store_true")
    args = parser.parse_args()
    if args.eval_width < 32 or args.max_frames < 3:
        raise SystemExit("eval-width >= 32 and max-frames >= 3")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit("cuda requested but unavailable")
    torch.set_num_threads(8)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_dir / "manifest.json"
    manifest: dict[str, object]
    if args.merge and manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    else:
        manifest = {
            "dataset": str(args.dataset_root),
            "teacher": "deficit-camera-rate" if args.teacher == "deficit" else "f1-argmax-camera-rate",
            "candidate_rates": list(CAMERA_RATES),
            "feature_names": list(FEATURE_NAMES),
            "prism_rpm": args.prism_rpm,
            "prism_angle_deg": args.prism_angle_deg,
            "eval_width": args.eval_width,
            "storm_occupancy": args.storm_occupancy,
            "deficit_occupancy": args.deficit_occupancy,
            "upshift_slack": args.upshift_slack,
            "upshift_margin": args.upshift_margin,
            "hold_on_warmup": CAMERA_HOLD_RATE,
            "oracles": [],
        }
    existing = {str(record["scene"]) for record in manifest["oracles"]}
    for split, scene in SCENES:
        if split not in args.splits:
            continue
        for regime in args.regimes:
            scene_id = f"{split}/{scene}/{regime}"
            if args.merge and scene_id in existing:
                print(f"skip existing {scene_id}", flush=True)
                continue
            path = _scene_path(args.dataset_root, split, scene, regime)
            print(scene_id, flush=True)
            payload = _rollout_scene(
                path,
                rpm=args.prism_rpm,
                prism_angle_deg=args.prism_angle_deg,
                eval_width=args.eval_width,
                max_frames=args.max_frames,
                max_windows=args.max_windows,
                storm_occupancy=args.storm_occupancy,
                upshift_margin=args.upshift_margin,
                deficit_occupancy=args.deficit_occupancy,
                upshift_slack=args.upshift_slack,
                teacher=args.teacher,
                device=args.device,
            )
            output = args.output_dir / f"{split}_{scene}_{regime}.npz"
            _write_npz(output, payload)
            record = {
                "scene": scene_id,
                "split": split,
                "condition": regime,
                "output": str(output),
                **{key: value for key, value in payload.items() if key != "rows"},
            }
            manifest["oracles"] = [item for item in manifest["oracles"] if item["scene"] != scene_id]
            manifest["oracles"].append(record)
            manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
            print(json.dumps({key: record[key] for key in ("scene", "n_windows", "label_frac", "mean_executed_rate", "mean_occupancy", "mean_teacher_f1")}, sort_keys=True), flush=True)
    print(json.dumps({"output": str(manifest_path), "n_oracles": len(manifest["oracles"])}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
