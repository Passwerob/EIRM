#!/usr/bin/env python3
"""Motion–excitation sweep: fixed 0/180/360, causal 3-gear policy, per-window oracle.

Procedural 64x64 pack only. AMI events are inverse-warped to the reference
plane before F1/TEO. Does not retrain. Does not overwrite EVEYES Table 1.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import cv2
import imageio.v3 as imageio
import numpy as np
import torch

from ami_simulator.dsec import read_imu_csv
from ami_simulator.event_quality import compute_reference_edge_mask, event_quality_metrics, target_observability_metrics
from ami_simulator.geometry import RotatingWedgePrism
from ami_simulator.motion_net import TinyCausalGearPolicy
from ami_simulator.pipeline import load_rgb_sequence, rgb_to_linear_luminance
from ami_simulator.types import CameraIntrinsics
from scripts.evaluate_dl3dv_downstream import _visual_motion
from scripts.evaluate_inverse_motion_control import _choose_rpm, temporal_overlap_enabled
from scripts.evaluate_phase_continuous_control import _simulate_action
from scripts.run_control_headroom_stateful import RolloutState
from scripts.train_inverse_motion_policy import FEATURE_DIM, WINDOW

GEARS = (0.0, 180.0, 360.0)
OBJECTIVE = {
    "event_efficiency_weight": 0.0,
    "overlap_weight": 0.0,
    "density_weight": 0.0,
    "invalid_fraction_weight": 0.0,
    "rpm_effort_weight": 0.0,
    "rpm_change_weight": 0.0,
}
DWELL = 0.025
CONFIDENCE = 0.5
STORM = 0.75


def _intrinsics(scene: Path, width: int, height: int) -> CameraIntrinsics:
    data = json.loads((scene / "camera_intrinsics.json").read_text(encoding="utf-8"))
    return CameraIntrinsics(width, height, float(data["fx"]), float(data["fy"]), float(data["cx"]), float(data["cy"]))


def _imu_windows(scene: Path, timestamps: np.ndarray, bounds: tuple[float, float]) -> np.ndarray:
    imu = read_imu_csv(scene / "imu.csv", timestamp_unit="s", gyro_unit="rad/s")
    motion = np.linalg.norm(imu.gyro, axis=1)
    out = np.empty(len(timestamps) - 1, dtype=np.float64)
    for index, (start, end) in enumerate(zip(timestamps[:-1], timestamps[1:])):
        left = int(np.searchsorted(imu.t, start, side="left"))
        right = int(np.searchsorted(imu.t, end, side="right"))
        values = motion[left:right]
        raw = float(values.mean() if len(values) else np.interp((start + end) * 0.5, imu.t, motion))
        low, high = bounds
        out[index] = float(np.clip((raw - low) / max(high - low, 1e-9), 0.0, 1.0))
    return out


def _farneback_median(prev: np.ndarray, nxt: np.ndarray, mask: np.ndarray | None = None) -> float:
    flow = cv2.calcOpticalFlowFarneback(
        np.clip(prev * 255.0, 0, 255).astype(np.uint8),
        np.clip(nxt * 255.0, 0, 255).astype(np.uint8),
        None,
        0.5,
        3,
        9,
        3,
        5,
        1.2,
        0,
    )
    mag = np.hypot(flow[..., 0], flow[..., 1])
    if mask is not None:
        mag = mag[mask]
    mag = mag[np.isfinite(mag)]
    return float(np.median(mag)) if mag.size else 0.0


def _commanded_speed(traj: dict[str, np.ndarray], timestamps: np.ndarray, key_x: str, key_y: str) -> np.ndarray:
    dt = np.diff(timestamps)
    dx = np.diff(traj[key_x])
    dy = np.diff(traj[key_y])
    return np.hypot(dx, dy) / np.maximum(dt, 1e-9)


def _score_window(events, luminance: np.ndarray, mask: np.ndarray, t0: float, t1: float) -> dict[str, float]:
    height, width = luminance.shape
    edges = compute_reference_edge_mask(luminance, edge_quantile=0.75)
    wide = event_quality_metrics(events, reference_edge_mask=edges, image_shape=(height, width))
    teo = target_observability_metrics(
        events,
        target_mask=mask,
        target_edge_mask=edges & mask,
        image_shape=(height, width),
        time_range=(t0, t1),
    )
    return {
        "edge_f1": float(wide["edge_f1"]),
        "edge_recall": float(wide["edge_coverage"]),
        "edge_precision": float(wide["edge_precision"]),
        "unique_pixel_ratio": float(wide["unique_pixel_ratio"]),
        "event_count": float(wide["event_count"]),
        "temporal_overlap": float(wide["temporal_overlap"]),
        "teo": float(teo["teo"]),
        "coverage_C": float(teo["target_coverage"]),
        "purity_Pi": float(teo["target_purity"]),
        "tau": float(teo["target_temporal_support"]),
        "target_f1": float(teo["target_f1"]),
    }


def _fixed_rollout(luminance, timestamps, masks, prism, rpm: float, device: str) -> list[dict[str, object]]:
    phase = 0.0
    previous_active = None
    sensor_reference = None
    previous_rpm = float(rpm)
    rows = []
    ones = np.ones_like(masks)
    for index in range(len(timestamps) - 1):
        result = _simulate_action(
            luminance,
            timestamps,
            ones,
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
        scores = _score_window(result["reference_events"], luminance[index], masks[index], float(timestamps[index]), float(timestamps[index + 1]))
        scores["rpm"] = float(rpm)
        scores["event_density"] = float(result["metrics"]["event_density"])
        scores["temporal_overlap"] = float(result["metrics"]["temporal_overlap"])
        rows.append(scores)
        phase = float(result["phase_end"])
        previous_active = np.asarray(result["active"])
        sensor_reference = np.asarray(result["sensor_reference_log"])
        previous_rpm = float(rpm)
    return rows


def _adaptive_rollout(
    luminance,
    timestamps,
    masks,
    prism,
    *,
    visual: np.ndarray,
    imu: np.ndarray,
    model: TinyCausalGearPolicy,
    checkpoint: dict[str, object],
    device: str,
) -> list[dict[str, object]]:
    gears = tuple(float(v) for v in checkpoint["candidate_rpm"])
    teacher = str(checkpoint.get("teacher", "f1-argmax-3"))
    quality_floor = bool(checkpoint.get("quality_floor", False))
    thresholds = tuple(float(v) for v in checkpoint["motion_thresholds"])
    use_overlap = temporal_overlap_enabled(checkpoint)
    state = RolloutState(0.0, None, None, 180.0, float(timestamps[0]) - DWELL)
    features: list[np.ndarray] = []
    ones = np.ones_like(masks)
    rows = []
    for index in range(len(timestamps) - 1):
        motion = float(visual[index - 1] if index else 0.0)
        previous_density = 0.0 if state.previous_metrics is None else float(state.previous_metrics["event_density"])
        previous_overlap = 0.0 if state.previous_metrics is None else float(state.previous_metrics["temporal_overlap"])
        overlap_feature = previous_overlap if use_overlap else 0.0
        feature = np.asarray((motion, float(imu[index - 1] if index else 0.0), previous_density, overlap_feature), dtype=np.float32)
        features.append(feature)
        rpm = _choose_rpm(
            "inverse_learned",
            motion=motion,
            thresholds=thresholds,
            previous_density=previous_density,
            previous_overlap=overlap_feature,
            features=features,
            model=model,
            checkpoint=checkpoint,
            gears=gears,
            quality_floor=quality_floor,
            teacher=teacher,
            confidence=CONFIDENCE,
            require_imu=False,
        )
        if index and float(timestamps[index] - state.last_switch_time) < DWELL:
            rpm = state.previous_rpm
        result = _simulate_action(
            luminance,
            timestamps,
            ones,
            prism,
            index=index,
            rpm=rpm,
            phase=state.phase,
            previous_active=state.previous_active,
            previous_rpm=state.previous_rpm,
            target_density=0.2,
            objective=OBJECTIVE,
            contrast_threshold=0.2,
            max_phase_step_deg=30.0,
            sensor_reference_log=state.sensor_reference_log,
            device=device,
        )
        scores = _score_window(result["reference_events"], luminance[index], masks[index], float(timestamps[index]), float(timestamps[index + 1]))
        scores["rpm"] = float(rpm)
        scores["event_density"] = float(result["metrics"]["event_density"])
        scores["temporal_overlap"] = float(result["metrics"]["temporal_overlap"])
        rows.append(scores)
        switched = rpm != state.previous_rpm
        state = RolloutState(
            phase=float(result["phase_end"]),
            previous_active=np.asarray(result["active"]).copy(),
            sensor_reference_log=np.asarray(result["sensor_reference_log"]).copy(),
            previous_rpm=float(rpm),
            last_switch_time=float(timestamps[index + 1]) if switched else state.last_switch_time,
            previous_metrics=scores,
        )
    return rows


def _oracle_rows(fixed: dict[float, list[dict[str, object]]], score_key: str) -> list[dict[str, object]]:
    n = len(fixed[0.0])
    out = []
    for index in range(n):
        scores = {rpm: float(fixed[rpm][index][score_key]) for rpm in GEARS}
        best = max(GEARS, key=lambda rpm: scores[rpm])
        row = dict(fixed[best][index])
        row["rpm"] = float(best)
        row["oracle_score"] = float(scores[best])
        row["score_0"] = scores[0.0]
        row["score_180"] = scores[180.0]
        row["score_360"] = scores[360.0]
        out.append(row)
    return out


def _mean_row(rows: list[dict[str, object]], keys: tuple[str, ...]) -> dict[str, float]:
    out = {key: float(np.mean([float(row[key]) for row in rows])) for key in keys}
    rpms = np.asarray([float(row["rpm"]) for row in rows], dtype=np.float64)
    out["mean_rpm"] = float(rpms.mean())
    out["frac_0"] = float(np.mean(np.isclose(rpms, 0.0)))
    out["frac_180"] = float(np.mean(np.isclose(rpms, 180.0)))
    out["frac_360"] = float(np.mean(np.isclose(rpms, 360.0)))
    out["n_switches"] = int(np.sum(np.abs(np.diff(rpms)) > 1e-6))
    return out


SCORE_KEYS = (
    "edge_f1",
    "edge_recall",
    "edge_precision",
    "unique_pixel_ratio",
    "event_count",
    "teo",
    "coverage_C",
    "purity_Pi",
    "tau",
    "target_f1",
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=Path("data/motion_excitation_sweep"))
    parser.add_argument("--student", type=Path, default=Path("results/comotion_f1_argmax_0_180_360/policy.pt"))
    parser.add_argument("--output", type=Path, default=Path("results/ami_next/motion_excitation_sweep"))
    parser.add_argument("--device", default="cuda:6")
    parser.add_argument("--prism-angle-deg", type=float, default=2.0)
    parser.add_argument("--skip-grid", action="store_true")
    args = parser.parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        print("cuda unavailable; CPU", flush=True)
        args.device = "cpu"

    manifest = json.loads((args.dataset_root / "manifest.json").read_text(encoding="utf-8"))
    checkpoint = torch.load(args.student, map_location="cpu", weights_only=True)
    if checkpoint.get("architecture") != "tiny_inverse_motion_gear":
        raise SystemExit(f"unexpected architecture {checkpoint.get('architecture')}")
    model = TinyCausalGearPolicy(feature_dim=FEATURE_DIM, hidden_dim=int(checkpoint["hidden_dim"]), num_gears=int(checkpoint["num_gears"]))
    model.load_state_dict(checkpoint["model"])
    model.eval()
    imu_bounds = (float(checkpoint["imu_motion_bounds"][0]), float(checkpoint["imu_motion_bounds"][1]))

    args.output.mkdir(parents=True, exist_ok=True)
    raw_dir = args.output / "per_scene"
    raw_dir.mkdir(parents=True, exist_ok=True)

    scene_rows: list[dict[str, object]] = []
    for record in manifest["records"]:
        if args.skip_grid and record.get("sweep") == "grid":
            continue
        scene = args.dataset_root / str(record["scene"])
        frames, timestamps = load_rgb_sequence(scene / "images", scene / "timestamps.csv", timestamp_unit="s")
        timestamps = timestamps[: len(frames)]
        luminance = rgb_to_linear_luminance(frames)
        mask_paths = sorted((scene / "object_masks").glob("*.png"))[: len(frames)]
        masks = np.stack([np.asarray(imageio.imread(path)) > 127 for path in mask_paths])
        height, width = luminance.shape[1], luminance.shape[2]
        prism = RotatingWedgePrism(_intrinsics(scene, width, height), args.prism_angle_deg)
        visual = _visual_motion(luminance, timestamps)
        imu = _imu_windows(scene, timestamps, imu_bounds)
        traj = dict(np.load(scene / "image_plane_trajectory.npz"))
        n = len(timestamps) - 1
        flow_full = np.asarray([_farneback_median(luminance[i], luminance[i + 1]) for i in range(n)], dtype=np.float64)
        flow_obj = np.asarray([_farneback_median(luminance[i], luminance[i + 1], masks[i]) for i in range(n)], dtype=np.float64)
        cmd_bg = _commanded_speed(traj, timestamps, "background_x", "background_y")
        cmd_obj = _commanded_speed(traj, timestamps, "object_x", "object_y")
        cmd_rel = np.hypot(
            np.diff(traj["object_x"]) - np.diff(traj["background_x"]),
            np.diff(traj["object_y"]) - np.diff(traj["background_y"]),
        ) / np.maximum(np.diff(timestamps), 1e-9)

        print(f"eval {scene.name}", flush=True)
        fixed = {rpm: _fixed_rollout(luminance, timestamps, masks, prism, rpm, args.device) for rpm in GEARS}
        adaptive = _adaptive_rollout(
            luminance,
            timestamps,
            masks,
            prism,
            visual=visual,
            imu=imu,
            model=model,
            checkpoint=checkpoint,
            device=args.device,
        )
        oracle_teo = _oracle_rows(fixed, "teo")
        oracle_f1 = _oracle_rows(fixed, "edge_f1")

        methods = {
            "fixed_0": fixed[0.0],
            "fixed_180": fixed[180.0],
            "fixed_360": fixed[360.0],
            "adaptive": adaptive,
            "oracle_teo": oracle_teo,
            "oracle_f1": oracle_f1,
        }
        window_dump = []
        for index in range(n):
            item = {
                "window": index,
                "t0": float(timestamps[index]),
                "t1": float(timestamps[index + 1]),
                "m_nat_flow_px": float(flow_full[index]),
                "m_nat_object_flow_px": float(flow_obj[index]),
                "m_cmd_bg_px_s": float(cmd_bg[index]),
                "m_cmd_obj_px_s": float(cmd_obj[index]),
                "m_cmd_rel_px_s": float(cmd_rel[index]),
                "visual_motion": float(visual[index]),
                "imu_frac": float(imu[index]),
            }
            for name, rows in methods.items():
                item[name] = rows[index]
            window_dump.append(item)

        out_scene = {
            "scene": record["scene"],
            "sweep": record.get("sweep"),
            "seed": record.get("seed"),
            "condition": record.get("condition"),
            "motion_scale": record.get("motion_scale"),
            "object_alpha": record.get("object_alpha"),
            "grid_camera_scale": record.get("grid_camera_scale"),
            "grid_object_scale": record.get("grid_object_scale"),
            "mean_m_nat_flow_px": float(flow_full.mean()),
            "mean_m_nat_object_flow_px": float(flow_obj.mean()),
            "mean_m_cmd_bg_px_s": float(cmd_bg.mean()),
            "mean_m_cmd_rel_px_s": float(cmd_rel.mean()),
            "methods": {name: _mean_row(rows, SCORE_KEYS) for name, rows in methods.items()},
        }
        (raw_dir / f"{record['scene']}.json").write_text(json.dumps({"summary": out_scene, "windows": window_dump}, indent=2) + "\n", encoding="utf-8")
        scene_rows.append(out_scene)
        print(
            json.dumps(
                {
                    "scene": record["scene"],
                    "m_nat": out_scene["mean_m_nat_flow_px"],
                    "teo0": out_scene["methods"]["fixed_0"]["teo"],
                    "teo180": out_scene["methods"]["fixed_180"]["teo"],
                    "teo360": out_scene["methods"]["fixed_360"]["teo"],
                    "teoA": out_scene["methods"]["adaptive"]["teo"],
                    "oracle": out_scene["methods"]["oracle_teo"]["teo"],
                    "adapt_rpm": out_scene["methods"]["adaptive"]["mean_rpm"],
                    "oracle_rpm": out_scene["methods"]["oracle_teo"]["mean_rpm"],
                },
                sort_keys=True,
            ),
            flush=True,
        )

    config = {
        "protocol": "ami_motion_excitation_sweep_v1",
        "dataset": str(args.dataset_root),
        "student": str(args.student),
        "architecture": checkpoint.get("architecture"),
        "teacher": checkpoint.get("teacher"),
        "candidate_rpm": list(checkpoint.get("candidate_rpm", [])),
        "feature_names": checkpoint.get("feature_names"),
        "history_window": WINDOW,
        "hidden_dim": checkpoint.get("hidden_dim"),
        "quality_floor": checkpoint.get("quality_floor"),
        "prism_angle_deg": args.prism_angle_deg,
        "compensation": "per-event inverse prism warp to RGB reference plane",
        "oracle": "per-window argmax among independent fixed-0/180/360 rollouts; not a deployable policy",
        "adaptive": "causal TinyCausalGearPolicy; lagged visual/IMU/occupancy/overlap; no GT mask / future / oracle score",
        "note": "EVEYES Table 1 Ours remains {180,360} with 180 floor. This experiment uses the procedural 3-gear checkpoint.",
    }
    (args.output / "config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    (args.output / "scenes.json").write_text(json.dumps(scene_rows, indent=2) + "\n", encoding="utf-8")
    with (args.output / "scenes.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "scene",
                "sweep",
                "seed",
                "motion_scale",
                "object_alpha",
                "m_nat_flow_px",
                "m_nat_object_flow_px",
                "m_cmd_rel_px_s",
                "method",
                "teo",
                "edge_f1",
                "edge_recall",
                "edge_precision",
                "coverage_C",
                "purity_Pi",
                "tau",
                "event_count",
                "unique_pixel_ratio",
                "mean_rpm",
                "frac_0",
                "frac_180",
                "frac_360",
            ]
        )
        for scene in scene_rows:
            for method, stats in scene["methods"].items():
                writer.writerow(
                    [
                        scene["scene"],
                        scene["sweep"],
                        scene["seed"],
                        scene.get("motion_scale"),
                        scene.get("object_alpha"),
                        scene["mean_m_nat_flow_px"],
                        scene["mean_m_nat_object_flow_px"],
                        scene["mean_m_cmd_rel_px_s"],
                        method,
                        stats["teo"],
                        stats["edge_f1"],
                        stats["edge_recall"],
                        stats["edge_precision"],
                        stats["coverage_C"],
                        stats["purity_Pi"],
                        stats["tau"],
                        stats["event_count"],
                        stats["unique_pixel_ratio"],
                        stats["mean_rpm"],
                        stats["frac_0"],
                        stats["frac_180"],
                        stats["frac_360"],
                    ]
                )
    print({"scenes": len(scene_rows), "output": str(args.output)})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
