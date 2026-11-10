#!/usr/bin/env python3
"""DL3DV-large5 downstream protocol: DVS-Voltmeter, v2e, fixed AMI, and Ours.

Ours is the frozen Blender F1-argmax gear policy with IMU gate, run causally
on DL3DV RGB + pose-derived IMU. AMI events are inverse-warped to the
reference plane before scoring. A sensor-plane (uncompensated) row exists
only as a warp diagnostic and is not a method in the paper table.
"""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

import cv2
import numpy as np
import torch

from ami_simulator.dsec import read_imu_csv
from ami_simulator.events import ThresholdEventSimulator
from ami_simulator.geometry import RotatingWedgePrism
from ami_simulator.motion_net import TinyCausalGearPolicy
from ami_simulator.pipeline import load_rgb_sequence, rgb_to_linear_luminance
from ami_simulator.types import CameraIntrinsics
from scripts.evaluate_dvs_v2e_ours import _external, _read_npz, _slice, _window_metrics
from scripts.evaluate_inverse_motion_control import _choose_rpm, temporal_overlap_enabled
from scripts.evaluate_phase_continuous_control import _simulate_action
from scripts.train_inverse_motion_policy import FEATURE_DIM, WINDOW

OBJECTIVE = {
    "event_efficiency_weight": 0.0,
    "overlap_weight": 0.5,
    "density_weight": 0.1,
    "invalid_fraction_weight": 0.0,
    "rpm_effort_weight": 0.0,
    "rpm_change_weight": 0.0,
}
SCALAR_KEYS = (
    "edge_f1",
    "edge_precision",
    "edge_recall",
    "spatial_matching_f1",
    "feature_repeatability",
    "spatial_precision",
    "spatial_recall",
    "event_count",
    "event_keypoints",
    "rgb_keypoints",
    "temporal_support",
)
# Training count/pixel density is not resolution-stable. Occupancy is the
# quantity the storm brake was written to measure.
IMU_MOTION_BOUNDS = (0.0, 1.8706787256548878)
VISUAL_MOTION_QUANTILE = 0.9


def _intrinsics(scene: Path) -> CameraIntrinsics:
    transforms = json.loads((scene / "transforms.json").read_text(encoding="utf-8"))
    return CameraIntrinsics(int(transforms["w"]), int(transforms["h"]), float(transforms["fl_x"]), float(transforms["fl_y"]), float(transforms["cx"]), float(transforms["cy"]))


def _visual_motion(luminance: np.ndarray, timestamps: np.ndarray) -> np.ndarray:
    delta = np.quantile(np.abs(np.diff(np.log(np.clip(luminance, 1e-6, None)), axis=0)), VISUAL_MOTION_QUANTILE, axis=(1, 2)).astype(np.float64)
    return delta / np.diff(timestamps)


def _imu_fraction(imu_csv: Path, timestamps: np.ndarray) -> np.ndarray:
    imu = read_imu_csv(imu_csv, timestamp_unit="us", gyro_unit="rad/s")
    motion = np.linalg.norm(imu.gyro, axis=1)
    windows = np.empty(len(timestamps) - 1, dtype=np.float64)
    for index, (start, end) in enumerate(zip(timestamps[:-1], timestamps[1:])):
        left = int(np.searchsorted(imu.t, start, side="left"))
        right = int(np.searchsorted(imu.t, end, side="right"))
        values = motion[left:right]
        windows[index] = float(values.mean() if len(values) else np.interp((start + end) * 0.5, imu.t, motion))
    low, high = IMU_MOTION_BOUNDS
    return np.clip((windows - low) / max(high - low, 1e-9), 0.0, 1.0)


def _step(luminance, timestamps, masks, prism, index, rpm, phase, previous_active, previous_rpm, sensor_reference, device="cpu", contrast_threshold=0.2):
    return _simulate_action(
        luminance,
        timestamps,
        masks,
        prism,
        index=index,
        rpm=float(rpm),
        phase=phase,
        previous_active=previous_active,
        previous_rpm=previous_rpm,
        target_density=0.2,
        objective=OBJECTIVE,
        contrast_threshold=contrast_threshold,
        max_phase_step_deg=30.0,
        sensor_reference_log=sensor_reference,
        device=device,
    )


def _ami_windows(luminance, timestamps, masks, prism, rpm: float, device="cpu", contrast_threshold=0.2) -> tuple[list[object], list[object], list[float]]:
    reference: list[object] = []
    sensor: list[object] = []
    phase = 0.0
    previous_active = None
    sensor_reference = None
    previous_rpm = float(rpm)
    for index in range(len(timestamps) - 1):
        result = _step(luminance, timestamps, masks, prism, index, rpm, phase, previous_active, previous_rpm, sensor_reference, device=device, contrast_threshold=contrast_threshold)
        reference.append(result["reference_events"])
        sensor.append(result["sensor_events"])
        phase = float(result["phase_end"])
        previous_active = np.asarray(result["active"])
        sensor_reference = np.asarray(result["sensor_reference_log"])
        previous_rpm = float(rpm)
    return reference, sensor, [float(rpm)] * (len(timestamps) - 1)


def _dvs_windows(luminance, timestamps, *, contrast_threshold: float = 0.2) -> tuple[list[object], list[float]]:
    """Generate direct DVS events without an AMI warp or inverse compensation.

    This is the explicit 0-RPM baseline: the RGB luminance sequence is fed to
    the threshold event model directly, with its reference level carried across
    windows. The returned streams are already in the camera/reference image
    coordinates and must not pass through ``compensate_events``.
    """
    simulator = ThresholdEventSimulator(contrast_threshold)
    streams: list[object] = []
    reference_log = None
    for index in range(len(timestamps) - 1):
        events, reference_log = simulator.simulate_with_state(
            np.asarray(luminance[index : index + 2]),
            np.asarray(timestamps[index : index + 2]),
            reference_log=reference_log,
        )
        streams.append(events)
    return streams, [0.0] * len(streams)


def _ours_windows(
    luminance,
    timestamps,
    masks,
    prism,
    *,
    visual: np.ndarray,
    imu: np.ndarray,
    model: TinyCausalGearPolicy,
    checkpoint: dict[str, object],
    confidence: float,
    require_imu: bool,
    dwell: float,
    device="cpu",
) -> tuple[list[object], list[float]]:
    streams: list[object] = []
    rpms: list[float] = []
    features: list[np.ndarray] = []
    phase = 0.0
    previous_active = None
    sensor_reference = None
    previous_rpm = 180.0
    last_switch = float(timestamps[0]) - dwell
    previous_occupancy = 0.0
    previous_overlap = 0.0
    teacher = str(checkpoint.get("teacher", "f1-argmax"))
    thresholds = tuple(float(value) for value in checkpoint["motion_thresholds"])
    gears = tuple(float(value) for value in checkpoint["candidate_rpm"])
    quality_floor = bool(checkpoint.get("quality_floor", False))
    use_overlap = temporal_overlap_enabled(checkpoint)
    for index in range(len(timestamps) - 1):
        motion = float(visual[index - 1] if index else 0.0)
        imu_value = float(imu[index - 1] if index else 0.0)
        overlap_feature = previous_overlap if use_overlap else 0.0
        features.append(np.asarray((motion, imu_value, previous_occupancy, overlap_feature), dtype=np.float32))
        rpm = _choose_rpm(
            "inverse_learned",
            motion=motion,
            thresholds=thresholds,
            previous_density=previous_occupancy,
            previous_overlap=overlap_feature,
            features=features,
            model=model,
            checkpoint=checkpoint,
            gears=gears,
            quality_floor=quality_floor,
            teacher=teacher,
            confidence=confidence,
            require_imu=require_imu,
        )
        if index and float(timestamps[index] - last_switch) < dwell:
            rpm = previous_rpm
        result = _step(luminance, timestamps, masks, prism, index, rpm, phase, previous_active, previous_rpm, sensor_reference, device=device)
        streams.append(result["reference_events"])
        rpms.append(float(rpm))
        if rpm != previous_rpm:
            last_switch = float(timestamps[index + 1])
        phase = float(result["phase_end"])
        previous_active = np.asarray(result["active"])
        sensor_reference = np.asarray(result["sensor_reference_log"])
        previous_rpm = float(rpm)
        previous_occupancy = float(result["metrics"]["unique_pixel_ratio"])
        previous_overlap = float(result["metrics"]["temporal_overlap"])
    return streams, rpms


def _score(windows, luminance, masks, timestamps, method, scene_id, rpms=None) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    previous = None
    for index, events in enumerate(windows):
        metrics, previous = _window_metrics(events, luminance[index], masks[index], previous)
        row = {"scene": scene_id, "window": index, "method": method, "t0": float(timestamps[index]), "t1": float(timestamps[index + 1]), **metrics}
        if rpms is not None:
            row["rpm"] = float(rpms[index])
        rows.append(row)
    return rows


def _mean(rows: list[dict[str, object]], method: str) -> dict[str, float]:
    selected = [row for row in rows if row["method"] == method]
    summary = {key: float(np.mean([float(row[key]) for row in selected])) for key in SCALAR_KEYS}
    if selected and "rpm" in selected[0]:
        summary["mean_rpm"] = float(np.mean([float(row["rpm"]) for row in selected]))
        summary["frac_360"] = float(np.mean([float(row["rpm"]) >= 359.0 for row in selected]))
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=Path("data/dl3dv_large5"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scenes", nargs="*", default=("screen-000083", "screen-000087", "screen-000089", "screen-000095", "screen-000108"))
    parser.add_argument("--max-frames", type=int, default=24)
    parser.add_argument("--eval-width", type=int, default=160)
    parser.add_argument("--prism-angle-deg", type=float, default=2.0)
    parser.add_argument("--student", type=Path, default=Path("results/comotion_f1_argmax_policy/policy.pt"))
    parser.add_argument("--confidence", type=float, default=0.70)
    parser.add_argument("--require-imu", action="store_true", default=True)
    parser.add_argument("--no-require-imu", action="store_false", dest="require_imu")
    parser.add_argument("--dwell", type=float, default=0.025)
    parser.add_argument("--v2e-root", type=Path, default=Path("/data/fcr/vendor/v2e"))
    parser.add_argument("--v2e-python", type=Path, default=Path("/data/fcr/miniconda3/envs/v2e/bin/python"))
    parser.add_argument("--dvs-root", type=Path, default=Path("/data/fcr/code/event_vggt/Data_Part/DVS-Voltmeter"))
    parser.add_argument("--dvs-python", type=Path, default=Path("/data/fcr/miniconda3/envs/dvsvolt/bin/python"))
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--cache", type=Path, default=Path("results/ami_next/dl3dv_baseline_events"))
    parser.add_argument("--torch-threads", type=int, default=8)
    args = parser.parse_args()
    if args.max_frames < WINDOW + 2 or args.eval_width < 32 or args.torch_threads < 1:
        raise SystemExit(f"max-frames must be >= {WINDOW + 2} so the causal policy has a full history window")
    torch.set_num_threads(args.torch_threads)
    torch.set_num_interop_threads(1)

    checkpoint = torch.load(args.student, map_location="cpu", weights_only=True)
    if checkpoint.get("architecture") != "tiny_inverse_motion_gear":
        raise SystemExit("Expected the F1-argmax tiny inverse-motion gear checkpoint.")
    model = TinyCausalGearPolicy(feature_dim=FEATURE_DIM, hidden_dim=int(checkpoint["hidden_dim"]), num_gears=int(checkpoint["num_gears"]))
    model.load_state_dict(checkpoint["model"])
    model.eval()

    methods = ("dvs_voltmeter", "v2e", "ami_passive", "ami_fixed_180", "ami_fixed_360", "ours", "ami_fixed_180_sensor", "ami_fixed_360_sensor")
    paper_methods = ("dvs_voltmeter", "v2e", "ami_passive", "ami_fixed_180", "ami_fixed_360", "ours")
    rows: list[dict[str, object]] = []
    for scene_id in args.scenes:
        scene = args.dataset_root / scene_id
        frames, timestamps = load_rgb_sequence(scene / "images", scene / "timestamps.csv", timestamp_unit="s", max_frames=args.max_frames)
        timestamps = timestamps[: len(frames)]
        intrinsics = _intrinsics(scene)
        if args.eval_width != intrinsics.width:
            eval_height = int(round(intrinsics.height * args.eval_width / intrinsics.width))
            frames = np.stack([cv2.resize(frame, (args.eval_width, eval_height), interpolation=cv2.INTER_AREA) for frame in frames])
            intrinsics = intrinsics.resized(args.eval_width, eval_height)
        luminance = rgb_to_linear_luminance(frames)
        masks = np.ones((len(luminance), luminance.shape[1], luminance.shape[2]), dtype=bool)
        prism = RotatingWedgePrism(intrinsics, args.prism_angle_deg)
        visual = _visual_motion(luminance, timestamps)
        imu = _imu_fraction(scene / "imu.csv", timestamps)
        streams: dict[str, list[object]] = {}
        rpm_logs: dict[str, list[float]] = {}
        with tempfile.TemporaryDirectory(prefix="ami_dl3dv_") as temporary:
            for method in ("dvs_voltmeter", "v2e"):
                cached = args.cache / f"{scene_id}_f{args.max_frames}_w{args.eval_width}_{method}.npz"
                if cached.exists():
                    events = _read_npz(cached)
                else:
                    events = _external(method, luminance, timestamps, args, Path(temporary))
                    cached.parent.mkdir(parents=True, exist_ok=True)
                    np.savez_compressed(cached, x=np.asarray(events.x), y=np.asarray(events.y), t=np.asarray(events.t), p=np.asarray(events.p))
                streams[method] = [_slice(events, float(timestamps[index]), float(timestamps[index + 1])) for index in range(len(timestamps) - 1)]
        streams["ami_passive"], _, rpm_logs["ami_passive"] = _ami_windows(luminance, timestamps, masks, prism, 0.0)
        streams["ami_fixed_180"], streams["ami_fixed_180_sensor"], rpm_logs["ami_fixed_180"] = _ami_windows(luminance, timestamps, masks, prism, 180.0)
        rpm_logs["ami_fixed_180_sensor"] = rpm_logs["ami_fixed_180"]
        streams["ami_fixed_360"], streams["ami_fixed_360_sensor"], rpm_logs["ami_fixed_360"] = _ami_windows(luminance, timestamps, masks, prism, 360.0)
        rpm_logs["ami_fixed_360_sensor"] = rpm_logs["ami_fixed_360"]
        streams["ours"], rpm_logs["ours"] = _ours_windows(
            luminance,
            timestamps,
            masks,
            prism,
            visual=visual,
            imu=imu,
            model=model,
            checkpoint=checkpoint,
            confidence=args.confidence,
            require_imu=args.require_imu,
            dwell=args.dwell,
        )
        for method, windows in streams.items():
            rows.extend(_score(windows, luminance, masks, timestamps, method, scene_id, rpm_logs.get(method)))
        ours_rpm = rpm_logs["ours"]
        print(
            f"evaluated {scene_id} frames={len(frames)} size={luminance.shape[2]}x{luminance.shape[1]} "
            f"ours_mean_rpm={float(np.mean(ours_rpm)):.1f} ours_frac_360={float(np.mean([r >= 359 for r in ours_rpm])):.2f} "
            f"mean_imu={float(np.mean(imu)):.3f} mean_visual={float(np.mean(visual)):.2f}"
        )

    aggregate = {method: _mean(rows, method) for method in methods}
    per_scene = {
        scene_id: {method: _mean([row for row in rows if row["scene"] == scene_id], method) for method in methods}
        for scene_id in args.scenes
    }
    report = {
        "protocol": "ami_dl3dv_large5_downstream_v2",
        "dataset": "DL3DV-large5",
        "timeline": "nominal_from_explicit_fps_30",
        "eval_width": args.eval_width,
        "max_frames": args.max_frames,
        "prism_angle_deg": args.prism_angle_deg,
        "student": str(args.student),
        "confidence": args.confidence,
        "require_imu": args.require_imu,
        "dwell": args.dwell,
        "history_window": WINDOW,
        "storm_density_feature": "unique_pixel_ratio",
        "imu_motion_bounds": list(IMU_MOTION_BOUNDS),
        "methods": list(paper_methods),
        "diagnostic_methods": ["ami_fixed_180_sensor", "ami_fixed_360_sensor"],
        "metrics": list(SCALAR_KEYS),
        "no_object_mask": True,
        "warp": "AMI paper rows use inverse-warped reference-plane events. ami_fixed_*_sensor is the same stream before compensation.",
        "limitations": [
            "DL3DV has no measured capture timestamps; 30 FPS is an explicit assumption.",
            "No object masks, so TEO is not scored. Image-wide edge F1 is Table 1.",
            "This is camera-motion in the wild, not controlled co-motion.",
            "IMU is pose-derived, not a physical IMU.",
            "The policy was trained on Blender co-motion; DL3DV is a transfer check.",
            "Storm feedback uses unique-pixel occupancy because count/pixel density is not resolution-stable from 64x64 Blender to 160x90 DL3DV.",
        ],
        "aggregate": aggregate,
        "paper_aggregate": {method: aggregate[method] for method in paper_methods},
        "per_scene": per_scene,
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"paper_aggregate": report["paper_aggregate"], "warp_diagnostic": {key: aggregate[key] for key in ("ami_fixed_180", "ami_fixed_180_sensor", "ami_fixed_360", "ami_fixed_360_sensor")}}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
