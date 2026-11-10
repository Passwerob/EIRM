#!/usr/bin/env python3
"""Camera-path NFB: rule three-action or learned F1-argmax gears.

Rule controller steps {0.5, 1.0, 1.5}× from occupancy and its one-step trend.
Learned controller imitates a one-step edge-F1 teacher over the same rates.
The camera never fully stops. Inverse warp undoes the prism only.

Optional prism NFB still downshifts 360→180 on storm occupancy. Learned
camera experiments freeze prism RPM (default 180) so the actuator is isolated.
Does not overwrite m0.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from ami_simulator.camera_nfb import (
    ARCHITECTURE,
    FEATURE_DIM,
    camera_feature,
    choose_learned_camera_rate,
    load_pose_speed,
    normalize_speed,
    pair_visual_motion,
    resize_luminance,
    sample_path,
    sample_pose_speed,
)
from ami_simulator.geometry import RotatingWedgePrism
from ami_simulator.motion_net import TinyCausalGearPolicy
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
DT = 1.0 / 60.0
STORM_ON = 0.50
STORM_OFF = 0.48
DWELL_S = 0.20
CAMERA_RATES = (0.5, 1.0, 1.5)
CAMERA_OCC_LOW = 0.35
CAMERA_OCC_HIGH = 0.50
CAMERA_TREND = 0.03
PRISM_FAST = 360.0
PRISM_SLOW = 180.0


def _sample(volume: np.ndarray, path_s: float) -> np.ndarray:
    return sample_path(volume, path_s)


def _camera_action(occupancy: float, previous_occupancy: float) -> str:
    """Three-action camera feedback: accelerate, hold, or decelerate.

    The occupancy band controls the operating point; its one-step trend makes
    the controller react before a rapidly growing event field reaches storm.
    """
    delta = float(occupancy) - float(previous_occupancy)
    if occupancy >= CAMERA_OCC_HIGH or (occupancy >= CAMERA_OCC_LOW and delta >= CAMERA_TREND):
        return "decelerate"
    if occupancy <= CAMERA_OCC_LOW and delta <= 0.0:
        return "accelerate"
    return "hold"


def _step_camera_rate(rate: float, occupancy: float, previous_occupancy: float = 0.0) -> float:
    levels = list(CAMERA_RATES)
    slot = min(range(len(levels)), key=lambda i: abs(levels[i] - rate))
    action = _camera_action(occupancy, previous_occupancy)
    if action == "decelerate":
        slot = max(0, slot - 1)
    elif action == "accelerate":
        slot = min(len(levels) - 1, slot + 1)
    return float(levels[slot])


def _nudge_actuators(rate: float, rpm: float, occupancy: float, previous_occupancy: float = 0.0, *, camera_nfb: bool, prism_nfb: bool) -> tuple[float, float]:
    """Apply the three-action camera controller and optional prism brake."""
    if camera_nfb:
        rate = _step_camera_rate(rate, occupancy, previous_occupancy)
    if prism_nfb:
        if occupancy >= STORM_ON:
            rpm = PRISM_SLOW
    return rate, rpm


def _load_learned(path: Path) -> tuple[TinyCausalGearPolicy, dict[str, object]]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint.get("architecture") != ARCHITECTURE:
        raise SystemExit(f"Expected {ARCHITECTURE}, got {checkpoint.get('architecture')!r}")
    model = TinyCausalGearPolicy(feature_dim=FEATURE_DIM, hidden_dim=int(checkpoint["hidden_dim"]), num_gears=int(checkpoint["num_gears"]))
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model, checkpoint


def _rate_summary(rates: np.ndarray) -> dict[str, float]:
    if len(rates) == 0:
        return {"frac_rate_0.5": 0.0, "frac_rate_1.0": 0.0, "frac_rate_1.5": 0.0, "n_rate_switches": 0}
    nearest = np.asarray([min((0.5, 1.0, 1.5), key=lambda value: abs(value - rate)) for rate in rates], dtype=np.float64)
    return {
        "frac_rate_0.5": float(np.mean(nearest == 0.5)),
        "frac_rate_1.0": float(np.mean(nearest == 1.0)),
        "frac_rate_1.5": float(np.mean(nearest == 1.5)),
        "n_rate_switches": int(np.sum(np.abs(np.diff(nearest)) > 1e-6)),
    }


def _rollout(
    scene: Path,
    *,
    camera_nfb: bool,
    prism_nfb: bool,
    rpm: float,
    device: str,
    max_frames: int,
    max_windows: int,
    eval_width: int | None = None,
    learned: tuple[TinyCausalGearPolicy, dict[str, object]] | None = None,
    confidence: float = 0.0,
) -> dict[str, object]:
    frames, timestamps = load_rgb_sequence(scene / "images", scene / "timestamps.csv", timestamp_unit="s", max_frames=max_frames)
    timestamps = timestamps[: len(frames)]
    luminance = rgb_to_linear_luminance(frames)
    intrinsics = _intrinsics(scene)
    if eval_width is not None:
        luminance, intrinsics = resize_luminance(luminance, intrinsics, eval_width)
    prism = RotatingWedgePrism(intrinsics, 0.5)
    speed_t, speed_v = load_pose_speed(scene)
    last = len(luminance) - 1
    path_s = 0.0
    camera_rate = 1.0
    current_rpm = float(rpm)
    last_switch = -DWELL_S
    occupancy = 0.0
    previous_occupancy = 0.0
    phase = 0.0
    previous_active = None
    sensor_reference = None
    previous_rpm = float(rpm)
    camera_action = "hold"
    lagged_visual = 0.0
    lagged_speed = 0.0
    features: list[np.ndarray] = []
    model, checkpoint = learned if learned is not None else (None, None)
    speed_bounds = tuple(float(value) for value in checkpoint["speed_bounds"]) if checkpoint is not None else (0.0, 1.0)
    rows: list[dict[str, object]] = []
    for window in range(max_windows):
        if path_s >= last - 1e-6:
            break
        now = window * DT
        if model is not None and checkpoint is not None:
            features.append(camera_feature(lagged_visual, lagged_speed, occupancy, camera_rate))
        if now - last_switch >= DWELL_S:
            if model is not None and checkpoint is not None:
                camera_action = "learned"
                new_rate = choose_learned_camera_rate(features, model, checkpoint, previous_occupancy=occupancy, confidence=confidence)
                new_rpm = float(rpm)
                if prism_nfb and occupancy >= STORM_ON:
                    new_rpm = PRISM_SLOW
            else:
                camera_action = _camera_action(occupancy, previous_occupancy)
                new_rate, new_rpm = _nudge_actuators(camera_rate, current_rpm, occupancy, previous_occupancy, camera_nfb=camera_nfb, prism_nfb=prism_nfb)
                if not camera_nfb:
                    new_rate = 1.0
                if not prism_nfb:
                    new_rpm = float(rpm)
            new_rate = float(min(new_rate, max(last - path_s, 0.5)))
            if new_rate != camera_rate or new_rpm != current_rpm:
                last_switch = now
            camera_rate, current_rpm = new_rate, new_rpm
        lum_a = _sample(luminance, path_s)
        lum_b = _sample(luminance, path_s + camera_rate)
        pair = np.stack((lum_a, lum_b), axis=0)
        times = np.asarray((now, now + DT), dtype=np.float64)
        valid = np.ones((2, pair.shape[1], pair.shape[2]), dtype=bool)
        result = _simulate_action(
            pair,
            times,
            valid,
            prism,
            index=0,
            rpm=float(current_rpm),
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
        metrics, previous_active, _ = _structure_metrics(result["reference_events"], lum_a, valid[0], previous_active)
        previous_occupancy, occupancy = occupancy, float(metrics["unique_pixel_ratio"])
        lagged_visual = pair_visual_motion(lum_a, lum_b, DT)
        lagged_speed = normalize_speed(sample_pose_speed(speed_t, speed_v, path_s, timestamps), speed_bounds)
        phase = float(result["phase_end"])
        sensor_reference = np.asarray(result["sensor_reference_log"])
        rows.append(
            {
                "window": window,
                "t0": now,
                "path_s": float(path_s),
                "camera_rate": float(camera_rate),
                "camera_action": camera_action if camera_nfb or model is not None else "hold",
                "rpm": float(current_rpm),
                **metrics,
            }
        )
        path_s = min(last, path_s + camera_rate)
        previous_rpm = float(current_rpm)
        if window % 40 == 0:
            print(f"  w={window} path={path_s:.1f} rate={camera_rate:.1f} rpm={current_rpm:.0f} occ={occupancy:.3f}", flush=True)
    rates = np.asarray([row["camera_rate"] for row in rows], dtype=np.float64)
    occ = np.asarray([row["unique_pixel_ratio"] for row in rows], dtype=np.float64)
    rpms = np.asarray([row["rpm"] for row in rows], dtype=np.float64)
    return {
        "n_windows": len(rows),
        "path_progress": float(path_s / last) if last else 0.0,
        "mean_camera_rate": float(rates.mean()) if len(rates) else 0.0,
        "frac_camera_stopped": float(np.mean(rates < 0.25)) if len(rates) else 0.0,
        "mean_rpm": float(rpms.mean()) if len(rpms) else 0.0,
        "frac_360": float(np.mean(rpms >= 330.0)) if len(rpms) else 0.0,
        "mean_occupancy": float(occ.mean()) if len(occ) else 0.0,
        "mean_event_count": float(np.mean([row["event_count"] for row in rows])) if rows else 0.0,
        "mean_edge_f1": float(np.mean([row["edge_f1"] for row in rows])) if rows else 0.0,
        **_rate_summary(rates),
        "rows": rows,
    }


def _policy_spec(name: str, prism_rpm: float | None) -> tuple[bool, bool, float, bool]:
    learned = name == "learned_camera_nfb"
    camera_nfb = name not in {"open_loop_camera"}
    prism_nfb = name == "camera_speed_nfb"
    if prism_rpm is not None:
        rpm = float(prism_rpm)
    elif name in {"rule_camera_nfb", "learned_camera_nfb"}:
        rpm = PRISM_SLOW
    elif name == "open_loop_camera":
        rpm = PRISM_FAST
    else:
        rpm = PRISM_FAST
    if name == "rule_camera_nfb":
        camera_nfb = True
        prism_nfb = False
    if name == "learned_camera_nfb":
        camera_nfb = True
        prism_nfb = False
    if name == "camera_nfb_keep360":
        camera_nfb = True
        prism_nfb = False
        rpm = PRISM_FAST if prism_rpm is None else float(prism_rpm)
    if name == "camera_speed_nfb":
        camera_nfb = True
        prism_nfb = True
        rpm = PRISM_FAST if prism_rpm is None else float(prism_rpm)
    if name == "open_loop_camera":
        camera_nfb = False
        prism_nfb = False
    return camera_nfb, prism_nfb, float(rpm), learned


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=READY)
    parser.add_argument("--output", type=Path, default=Path("results/eveyes_motion_nfb/camera_speed_nfb.json"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-frames", type=int, default=480)
    parser.add_argument("--max-windows", type=int, default=1000)
    parser.add_argument("--eval-width", type=int, default=0, help="0 keeps native resolution")
    parser.add_argument("--prism-rpm", type=float, default=None)
    parser.add_argument("--student", type=Path, default=None)
    parser.add_argument("--confidence", type=float, default=0.0)
    parser.add_argument("--regimes", nargs="+", default=("co_motion",))
    parser.add_argument("--merge", action="store_true", help="keep existing scene/policy blocks")
    parser.add_argument(
        "--policies",
        nargs="+",
        default=("open_loop_camera", "camera_speed_nfb", "camera_nfb_keep360"),
    )
    args = parser.parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit("cuda requested but unavailable")
    if "learned_camera_nfb" in args.policies and args.student is None:
        raise SystemExit("learned_camera_nfb requires --student")
    torch.set_num_threads(8)
    learned = _load_learned(args.student) if args.student is not None else None
    eval_width = None if args.eval_width <= 0 else args.eval_width
    report: dict[str, object]
    meta = {
        "dataset": str(args.dataset_root),
        "branches": {
            "prism": "frozen unless camera_speed_nfb dual storm downshift",
            "camera": "rule three-action or learned F1-argmax over {0.5, 1.0, 1.5}",
        },
        "storm_on": STORM_ON,
        "storm_off": STORM_OFF,
        "dwell_s": DWELL_S,
        "eval_width": eval_width,
        "prism_rpm": args.prism_rpm,
        "student": None if args.student is None else str(args.student),
        "not_m0": True,
    }
    if args.merge and args.output.exists():
        report = json.loads(args.output.read_text(encoding="utf-8"))
        report.update(meta)
        report.setdefault("scenes", {})
    else:
        report = {**meta, "scenes": {}}
    known = {"open_loop_camera", "camera_speed_nfb", "camera_nfb_keep360", "rule_camera_nfb", "learned_camera_nfb"}
    unknown = [name for name in args.policies if name not in known]
    if unknown:
        raise SystemExit(f"unknown policies {unknown}")
    for regime in args.regimes:
        path = _scene_path(args.dataset_root, "test", "Scene_005", regime)
        scene_id = f"test/Scene_005/{regime}"
        report.setdefault("scenes", {})
        report["scenes"].setdefault(scene_id, {})
        for name in args.policies:
            if args.merge and name in report["scenes"][scene_id]:
                print(f"skip existing {scene_id} {name}", flush=True)
                continue
            camera_nfb, prism_nfb, rpm, use_learned = _policy_spec(name, args.prism_rpm)
            print(f"{scene_id}  {name}  rpm={rpm:.0f}", flush=True)
            payload = _rollout(
                path,
                camera_nfb=camera_nfb,
                prism_nfb=prism_nfb,
                rpm=rpm,
                device=args.device,
                max_frames=args.max_frames,
                max_windows=args.max_windows,
                eval_width=eval_width,
                learned=learned if use_learned else None,
                confidence=args.confidence,
            )
            summary = {key: value for key, value in payload.items() if key != "rows"}
            report["scenes"][scene_id][name] = payload
            print(json.dumps({"scene": scene_id, "policy": name, **summary}, sort_keys=True), flush=True)
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(args.output)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
