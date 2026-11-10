#!/usr/bin/env python3
"""Build resumable DL3DV Practical-RIFE time-scale oracles and train an RPM policy."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as functional

from ami_simulator.dsec import read_imu_csv
from ami_simulator.event_quality import QUALITY_PROTOCOL_VERSION
from ami_simulator.pipeline import load_rgb_sequence, rgb_to_linear_luminance


DEFAULT_TRAIN = ("screen-000083", "screen-000087", "screen-000089")
DEFAULT_VALID = ("screen-000095",)
DEFAULT_TEST = ("screen-000108",)


def _scene_motion(scene: Path, scales: list[float], max_frames: int, resize_width: int, spatial_quantile: float) -> tuple[list[float], list[float]]:
    frames, timestamps = load_rgb_sequence(scene / "images", scene / "timestamps.csv", timestamp_unit="s", max_frames=max_frames)
    if resize_width and resize_width < frames.shape[2]:
        height = int(round(frames.shape[1] * resize_width / frames.shape[2]))
        values = torch.from_numpy(frames.astype(np.float32)).permute(0, 3, 1, 2)
        frames = functional.interpolate(values, size=(height, resize_width), mode="bilinear", align_corners=False).permute(0, 2, 3, 1).numpy()
    luminance = rgb_to_linear_luminance(frames)
    contrast = np.quantile(np.abs(np.diff(np.log(np.clip(luminance, 1e-6, None)), axis=0)), spatial_quantile, axis=(1, 2))
    imu = read_imu_csv(scene / "imu.csv", timestamp_unit="us", gyro_unit="rad/s")
    gyro = np.linalg.norm(imu.gyro, axis=1)
    visual, inertial = [], []
    for scale in scales:
        visual.extend((contrast / (np.diff(timestamps) * scale)).tolist())
        inertial.extend((gyro / scale).tolist())
    return visual, inertial


def _bounds(values: list[float]) -> tuple[float, float]:
    low, high = np.quantile(np.asarray(values, dtype=np.float64), [0.05, 0.95])
    if high <= low:
        high = low + max(abs(float(low)) * 1e-3, 1e-6)
    return float(low), float(high)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=Path("data/dl3dv_large5"))
    parser.add_argument("--practical-rife-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("results/dl3dv_rife_multiscale"))
    parser.add_argument("--train-scenes", nargs="+", default=list(DEFAULT_TRAIN))
    parser.add_argument("--valid-scenes", nargs="+", default=list(DEFAULT_VALID))
    parser.add_argument("--test-scenes", nargs="+", default=list(DEFAULT_TEST))
    parser.add_argument("--time-scales", type=float, nargs="+", default=(0.25, 0.5, 1.0, 2.0, 4.0))
    parser.add_argument("--interpolation-fps", type=float, default=240.0)
    parser.add_argument("--max-phase-step-deg", type=float, default=30.0)
    parser.add_argument("--max-frames", type=int, default=12)
    parser.add_argument("--resize-width", type=int, default=120)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--threads-per-worker", type=int, default=8)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--window", type=int, default=8)
    parser.add_argument("--event-efficiency-weight", type=float, default=0.5)
    parser.add_argument("--density-weight", type=float, default=0.1)
    parser.add_argument("--rpm-effort-weight", type=float, default=0.05)
    parser.add_argument("--rpm-change-weight", type=float, default=0.05)
    parser.add_argument("--minimum-dwell-seconds", type=float, default=0.05)
    parser.add_argument("--switch-margin", type=float, default=0.02)
    parser.add_argument("--flow-features", action="store_true", help="Store pre-prism dense-flow summaries and train with them.")
    parser.add_argument("--policy-architecture", choices=("tcn", "lstm"), default="tcn")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.workers < 1 or args.threads_per_worker < 1 or args.max_frames < 2 or args.resize_width < 16 or args.interpolation_fps <= 0.0 or any(value <= 0.0 for value in args.time_scales) or min(args.event_efficiency_weight, args.density_weight, args.rpm_effort_weight, args.rpm_change_weight, args.minimum_dwell_seconds, args.switch_margin) < 0.0:
        parser.error("Workers/threads/time scales/FPS must be positive, max-frames >=2, and resize-width >=16.")
    groups = {"train": args.train_scenes, "valid": args.valid_scenes, "test": args.test_scenes}
    all_scenes = [scene for values in groups.values() for scene in values]
    if len(set(all_scenes)) != len(all_scenes):
        parser.error("Train, validation, and test scene IDs must be disjoint.")
    for scene in all_scenes:
        if not (args.dataset_root / scene).is_dir():
            parser.error(f"Missing scene: {args.dataset_root / scene}")
    if args.fp16 and not args.device.startswith("cuda"):
        parser.error("--fp16 requires a CUDA device.")

    visual_values, imu_values = [], []
    for scene_id in args.train_scenes:
        visual, inertial = _scene_motion(args.dataset_root / scene_id, list(args.time_scales), args.max_frames, args.resize_width, 0.9)
        visual_values.extend(visual)
        imu_values.extend(inertial)
    visual_bounds, imu_bounds = _bounds(visual_values), _bounds(imu_values)
    args.output.mkdir(parents=True, exist_ok=True)
    jobs: list[dict[str, object]] = []
    for split, scene_ids in groups.items():
        for scene_id in scene_ids:
            for scale in args.time_scales:
                scale_name = f"{scale:g}".replace(".", "p")
                destination = args.output / split / scene_id / f"scale_{scale_name}.npz"
                jobs.append({"split": split, "scene": scene_id, "time_scale": float(scale), "output": destination})

    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(Path(__file__).resolve().parents[1])
    environment["OMP_NUM_THREADS"] = str(args.threads_per_worker)
    environment["MKL_NUM_THREADS"] = str(args.threads_per_worker)

    def run_job(job: dict[str, object]) -> dict[str, object]:
        destination = Path(job["output"])
        report_path = destination.with_suffix(".json")
        if destination.exists() and report_path.exists() and not args.overwrite:
            report = json.loads(report_path.read_text(encoding="utf-8"))
            flow_ready = bool(report.get("flow_features", {}).get("enabled", False))
            if not args.flow_features or flow_ready:
                return {**job, "status": "reused", "windows": report.get("windows"), "label_counts": report.get("label_counts")}
        destination.parent.mkdir(parents=True, exist_ok=True)
        command = [
            sys.executable, "scripts/build_imu_gear_oracle.py",
            "--scene-root", str(args.dataset_root / str(job["scene"])),
            "--imu-csv", str(args.dataset_root / str(job["scene"]) / "imu.csv"),
            "--timestamp-unit", "s", "--imu-timestamp-unit", "us",
            "--time-scale", str(job["time_scale"]),
            "--interpolation-fps", str(args.interpolation_fps),
            "--vfi", "practical_rife", "--practical-rife-root", str(args.practical_rife_root),
            "--rife-device", args.device, "--motion-source", "max",
            "--visual-motion-bounds", str(visual_bounds[0]), str(visual_bounds[1]),
            "--imu-motion-bounds", str(imu_bounds[0]), str(imu_bounds[1]),
            "--max-phase-step-deg", str(args.max_phase_step_deg),
            "--event-efficiency-weight", str(args.event_efficiency_weight),
            "--density-weight", str(args.density_weight),
            "--rpm-effort-weight", str(args.rpm_effort_weight),
            "--rpm-change-weight", str(args.rpm_change_weight),
            "--minimum-dwell-seconds", str(args.minimum_dwell_seconds),
            "--switch-margin", str(args.switch_margin),
            "--store-active-masks",
            "--max-frames", str(args.max_frames), "--resize-width", str(args.resize_width),
            "--output", str(destination),
        ]
        if args.fp16:
            command.append("--rife-fp16")
        if args.flow_features:
            command.append("--flow-features")
        completed = subprocess.run(command, cwd=Path(__file__).resolve().parents[1], env=environment, text=True, capture_output=True)
        if completed.returncode != 0:
            raise RuntimeError(f"Oracle failed for {job['scene']} scale={job['time_scale']}:\n{completed.stderr}\n{completed.stdout}")
        report = json.loads(report_path.read_text(encoding="utf-8"))
        return {**job, "status": "generated", "windows": report["windows"], "label_counts": report["label_counts"]}

    records: list[dict[str, object]] = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(run_job, job): job for job in jobs}
        for future in as_completed(futures):
            record = future.result()
            records.append(record)
            print(json.dumps({key: str(value) if isinstance(value, Path) else value for key, value in record.items()}, sort_keys=True), flush=True)

    train_oracles = [str(record["output"]) for record in records if record["split"] == "train"]
    valid_oracles = [str(record["output"]) for record in records if record["split"] == "valid"]
    policy_path = args.output / "causal_rpm_policy.pt"
    train_command = [sys.executable, "scripts/train_causal_rpm_policy.py"]
    for oracle in sorted(train_oracles):
        train_command.extend(("--train-oracle", oracle))
    for oracle in sorted(valid_oracles):
        train_command.extend(("--valid-oracle", oracle))
    train_command.extend(("--window", str(args.window), "--epochs", str(args.epochs), "--output", str(policy_path)))
    train_command.extend(("--architecture", args.policy_architecture))
    if args.flow_features:
        train_command.append("--use-flow")
    completed = subprocess.run(train_command, cwd=Path(__file__).resolve().parents[1], env=environment, text=True, capture_output=True)
    if completed.returncode != 0:
        raise RuntimeError(f"Policy training failed:\n{completed.stderr}\n{completed.stdout}")
    policy_report = json.loads(policy_path.with_suffix(".json").read_text(encoding="utf-8"))
    manifest = {
        "dataset": "DL3DV-large5",
        "augmentation": "nominal_timeline_scale_plus_Practical-RIFE_direct_arbitrary_timestep",
        "quality_protocol_version": QUALITY_PROTOCOL_VERSION,
        "composite_score_role": "provisional_controller_surrogate_not_benchmark_ground_truth",
        "splits": groups,
        "time_scales": list(args.time_scales),
        "same_scene_variants_share_split": True,
        "visual_motion_bounds": visual_bounds,
        "imu_motion_bounds": imu_bounds,
        "motion_bounds_source": "training_scenes_only",
        "configuration": {"base_rpm": 720.0, "interpolation_fps": args.interpolation_fps, "max_phase_step_deg": args.max_phase_step_deg, "max_frames": args.max_frames, "resize_width": args.resize_width, "device": args.device, "fp16": args.fp16, "flow_features": args.flow_features, "policy_architecture": args.policy_architecture, "event_efficiency_weight": args.event_efficiency_weight, "density_weight": args.density_weight, "overlap_weight": 0.5, "invalid_fraction_weight": 0.25, "rpm_effort_weight": args.rpm_effort_weight, "rpm_change_weight": args.rpm_change_weight, "minimum_dwell_seconds": args.minimum_dwell_seconds, "switch_margin": args.switch_margin},
        "oracles": [{key: str(value) if isinstance(value, Path) else value for key, value in record.items()} for record in sorted(records, key=lambda value: (str(value["split"]), str(value["scene"]), float(value["time_scale"])))],
        "policy_report": policy_report,
        "limitations": ["DL3DV timestamps are nominal, not measured capture times.", "RIFE frames are interpolated observations, not new endpoint geometry.", "Test oracles are generated for evaluation only and never enter policy training."],
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"output": str(args.output), "oracles": len(records), "policy": str(policy_path), "valid_mae_rpm": policy_report["valid_mae_rpm"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
