#!/usr/bin/env python3
"""Evaluate prism-speed control with downstream edge F1 as a primary metric.

Policies can invert visual motion, follow an excitation-deficit rule
(camera moving but events scarce → 360 RPM), or imitate an F1 teacher.
The 180 RPM floor is required for quality teachers so structure is not
turned off.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from ami_simulator.geometry import RotatingWedgePrism
from ami_simulator.inverse_motion import INVERSE_RPM, QUALITY_FLOOR_RPM, excitation_deficit_rpm, inverse_rpm, quality_inverse_rpm, motion_thresholds, spearman_rho
from ami_simulator.motion_net import TinyCausalGearPolicy
from scripts.evaluate_phase_continuous_control import _load_scene
from scripts.run_control_headroom_stateful import RolloutState, _transition
from scripts.train_inverse_motion_policy import FEATURE_DIM, WINDOW, _records, _scene_arrays

STORM_DENSITY = 0.75
STORM_OVERLAP = 0.65


def temporal_overlap_enabled(checkpoint: dict[str, object] | None) -> bool:
    """EVEYES F1 oracles stored temporal_overlap as zeros.

    Feeding live overlap into that checkpoint is OOD and trips the 0.65 storm
    brake, which silently collapses a gear policy to 180 RPM.
    """
    if checkpoint is None:
        return True
    if "use_temporal_overlap" in checkpoint:
        return bool(checkpoint["use_temporal_overlap"])
    return "eveyes" not in str(checkpoint.get("dataset", "")).lower()


def _choose_rpm(
    policy: str,
    *,
    motion: float,
    thresholds: tuple[float, float],
    previous_density: float,
    previous_overlap: float,
    features: list[np.ndarray],
    model: TinyCausalGearPolicy | None,
    checkpoint: dict[str, object] | None,
    gears: tuple[float, ...],
    quality_floor: bool,
    teacher: str,
    confidence: float = 0.5,
    require_imu: bool = False,
) -> float:
    floor = QUALITY_FLOOR_RPM if quality_floor else 0.0
    imu_threshold = float(checkpoint.get("imu_threshold", 0.3)) if checkpoint is not None else 0.3
    density_threshold = float(checkpoint.get("density_threshold", 0.3)) if checkpoint is not None else 0.3
    imu = float(features[-1][1]) if features else 0.0
    if policy == "fixed_0":
        return 0.0
    if policy == "fixed_180":
        return 180.0
    if policy == "fixed_360":
        return 360.0
    if policy == "deficit_rule":
        return excitation_deficit_rpm(
            imu,
            previous_density,
            previous_overlap,
            imu_threshold=imu_threshold,
            density_threshold=density_threshold,
            storm_density=STORM_DENSITY,
            storm_overlap=STORM_OVERLAP,
        )
    if policy == "inverse_visual":
        if quality_floor:
            return quality_inverse_rpm(motion, thresholds, previous_density=previous_density, previous_overlap=previous_overlap, storm_density=STORM_DENSITY, storm_overlap=STORM_OVERLAP)
        return inverse_rpm(motion, thresholds, previous_density=previous_density, previous_overlap=previous_overlap, storm_density=STORM_DENSITY, storm_overlap=STORM_OVERLAP)
    if policy != "inverse_learned" or model is None or checkpoint is None:
        raise ValueError(f"Unknown policy {policy!r}")
    if len(features) < WINDOW:
        if teacher in {"f1-argmax", "f1-argmax-3", "three-gear"}:
            return 180.0
        if teacher == "deficit":
            return excitation_deficit_rpm(
                imu,
                previous_density,
                previous_overlap,
                imu_threshold=imu_threshold,
                density_threshold=density_threshold,
                storm_density=STORM_DENSITY,
                storm_overlap=STORM_OVERLAP,
            )
        if quality_floor:
            return quality_inverse_rpm(motion, thresholds, previous_density=previous_density, previous_overlap=previous_overlap, storm_density=STORM_DENSITY, storm_overlap=STORM_OVERLAP)
        return inverse_rpm(motion, thresholds, previous_density=previous_density, previous_overlap=previous_overlap, storm_density=STORM_DENSITY, storm_overlap=STORM_OVERLAP)
    mean = np.asarray(checkpoint["feature_mean"], dtype=np.float32)
    std = np.asarray(checkpoint["feature_std"], dtype=np.float32)
    clip = float(checkpoint.get("feature_clip", 5.0))
    history = np.clip((np.stack(features[-WINDOW:]) - mean) / std, -clip, clip)
    with torch.inference_mode():
        logits = model(torch.from_numpy(history[None].astype(np.float32)))[0]
        probs = torch.softmax(logits, dim=0)
        gear = int(torch.argmax(logits).item())
        if float(probs[gear].item()) < float(confidence):
            gear = 1 if quality_floor else int(np.argmin(np.abs(np.asarray(gears) - 180.0)))
    rpm = float(gears[gear])
    if require_imu and imu < imu_threshold:
        rpm = QUALITY_FLOOR_RPM if quality_floor else min(rpm, 180.0)
    if previous_density >= STORM_DENSITY or previous_overlap >= STORM_OVERLAP:
        rpm = min(rpm, 180.0)
    if previous_density >= 1.5 * STORM_DENSITY:
        rpm = floor
    return float(max(rpm, floor))


def _run(
    scene: tuple[np.ndarray, np.ndarray, np.ndarray, object],
    oracle: np.lib.npyio.NpzFile,
    arrays: dict[str, np.ndarray],
    objective: dict[str, float],
    *,
    policy: str,
    thresholds: tuple[float, float],
    model: TinyCausalGearPolicy | None = None,
    checkpoint: dict[str, object] | None = None,
    gears: tuple[float, ...] = INVERSE_RPM,
    quality_floor: bool = False,
    teacher: str = "inverse",
    dwell: float = 0.025,
    confidence: float = 0.5,
    require_imu: bool = False,
    sticky: bool = False,
) -> dict[str, object]:
    luminance, timestamps, masks, intrinsics = scene
    prism = RotatingWedgePrism(intrinsics, 2.0)
    state = RolloutState(0.0, None, None, 180.0, float(timestamps[0]) - dwell)
    features: list[np.ndarray] = []
    rows: list[dict[str, float]] = []
    rpms: list[float] = []
    motions: list[float] = []
    sticky_rpm: float | None = None
    use_overlap = temporal_overlap_enabled(checkpoint)
    for index in range(len(timestamps) - 1):
        motion = float(arrays["visual"][index - 1] if index else 0.0)
        previous_density = 0.0 if state.previous_metrics is None else float(state.previous_metrics["event_density"])
        previous_overlap = 0.0 if state.previous_metrics is None else float(state.previous_metrics["temporal_overlap"])
        overlap_feature = previous_overlap if use_overlap else 0.0
        feature = np.asarray((motion, float(arrays["imu"][index - 1] if index else 0.0), previous_density, overlap_feature), dtype=np.float32)
        features.append(feature)
        rpm = _choose_rpm(policy, motion=motion, thresholds=thresholds, previous_density=previous_density, previous_overlap=overlap_feature, features=features, model=model, checkpoint=checkpoint, gears=gears, quality_floor=quality_floor, teacher=teacher, confidence=confidence, require_imu=require_imu)
        if sticky and policy in {"inverse_learned", "deficit_rule"}:
            if previous_density >= STORM_DENSITY or overlap_feature >= STORM_OVERLAP:
                sticky_rpm = 180.0
            elif sticky_rpm is None or rpm >= 360.0:
                sticky_rpm = rpm
            rpm = float(sticky_rpm)
        if index and float(timestamps[index] - state.last_switch_time) < dwell:
            rpm = state.previous_rpm
        result, next_state = _transition(scene, oracle, objective, prism, state, index, rpm, dwell=dwell)
        metrics = {key: float(value) for key, value in result["metrics"].items()}
        rows.append(metrics)
        rpms.append(float(rpm))
        motions.append(motion)
        state = next_state
    rpm_values = np.asarray(rpms, dtype=np.float64)
    motion_values = np.asarray(motions, dtype=np.float64)
    densities = np.asarray([row["event_density"] for row in rows], dtype=np.float64)
    overlaps = np.asarray([row["temporal_overlap"] for row in rows], dtype=np.float64)
    coverage = np.asarray([row["edge_coverage"] for row in rows], dtype=np.float64)
    f1 = np.asarray([row["edge_f1"] for row in rows], dtype=np.float64)
    order = np.argsort(motion_values)
    tertile = np.array_split(order, 3)
    bins = {}
    for name, indices in zip(("slow", "mid", "fast"), tertile):
        bins[name] = {
            "mean_motion": float(motion_values[indices].mean()) if len(indices) else float("nan"),
            "mean_rpm": float(rpm_values[indices].mean()) if len(indices) else float("nan"),
            "mean_event_density": float(densities[indices].mean()) if len(indices) else float("nan"),
            "mean_temporal_overlap": float(overlaps[indices].mean()) if len(indices) else float("nan"),
            "mean_edge_coverage": float(coverage[indices].mean()) if len(indices) else float("nan"),
            "mean_edge_f1": float(f1[indices].mean()) if len(indices) else float("nan"),
            "storm_rate": float(np.mean((densities[indices] >= STORM_DENSITY) | (overlaps[indices] >= STORM_OVERLAP))) if len(indices) else float("nan"),
        }
    return {
        "windows": len(rows),
        "mean_rpm": float(rpm_values.mean()),
        "mean_event_density": float(densities.mean()),
        "mean_temporal_overlap": float(overlaps.mean()),
        "mean_edge_coverage": float(coverage.mean()),
        "mean_edge_f1": float(f1.mean()),
        "storm_rate": float(np.mean((densities >= STORM_DENSITY) | (overlaps >= STORM_OVERLAP))),
        "spearman_motion_rpm": spearman_rho(motion_values, rpm_values),
        "slow_minus_fast_rpm": float(bins["slow"]["mean_rpm"] - bins["fast"]["mean_rpm"]),
        "bins": bins,
        "rpm_hist": {str(value): float(np.mean(np.isclose(rpm_values, value))) for value in sorted(set(gears) | {0.0, 180.0, 360.0})},
        "rpms": [float(value) for value in rpm_values],
    }


def _mean(rows: list[dict[str, object]], key: str) -> float:
    return float(np.mean([float(row[key]) for row in rows]))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("results/comotion_v2_learned/manifest.json"))
    parser.add_argument("--dataset-root", type=Path, default=Path("data/comotion_controlled_v2"))
    parser.add_argument("--student", type=Path, default=None)
    parser.add_argument("--split", choices=("train", "valid", "test"), default="test")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--confidence", type=float, default=0.5, help="Minimum softmax mass required to leave 180 RPM.")
    parser.add_argument("--require-imu", action="store_true", help="Forbid 360 RPM unless lagged IMU exceeds the deficit IMU threshold.")
    parser.add_argument("--sticky", action="store_true", help="Keep an upshift until an event storm, to avoid per-window chatter.")
    parser.add_argument("--policies", nargs="*", default=None)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    train_visual = np.concatenate([_scene_arrays(Path(str(record["output"])))["visual"] for record in _records(manifest, "train")])
    thresholds = motion_thresholds(train_visual)
    checkpoint = None
    model = None
    if args.student is not None:
        checkpoint = torch.load(args.student, map_location="cpu", weights_only=True)
        if checkpoint.get("architecture") != "tiny_inverse_motion_gear" or int(checkpoint.get("feature_dim", 0)) != FEATURE_DIM:
            raise ValueError("Expected tiny_inverse_motion_gear checkpoint.")
        model = TinyCausalGearPolicy(feature_dim=FEATURE_DIM, hidden_dim=int(checkpoint["hidden_dim"]), num_gears=int(checkpoint["num_gears"]))
        model.load_state_dict(checkpoint["model"])
        model.eval()
        thresholds = tuple(float(value) for value in checkpoint["motion_thresholds"])
    quality_floor = bool(checkpoint.get("quality_floor", False)) if checkpoint is not None else False
    teacher = str(checkpoint.get("teacher", "quality-floor" if quality_floor else "inverse")) if checkpoint is not None else "inverse"
    gears = tuple(float(value) for value in checkpoint["candidate_rpm"]) if checkpoint is not None else INVERSE_RPM
    objective = {**manifest["configuration"], "event_efficiency_weight": 0.0}
    if teacher in {"f1-argmax-3", "three-gear", "inverse"}:
        policies = ("fixed_0", "fixed_180", "fixed_360")
        if teacher != "f1-argmax-3":
            policies = policies + ("inverse_visual",)
    elif teacher in {"f1-argmax", "deficit"}:
        policies = ("fixed_180", "fixed_360", "deficit_rule")
    else:
        policies = ("fixed_180", "fixed_360", "inverse_visual")
    if model is not None:
        policies = policies + ("inverse_learned",)
    if args.policies is not None:
        allowed = set(args.policies)
        policies = tuple(policy for policy in policies if policy in allowed)
        if not policies:
            raise ValueError(f"No matching policies in {args.policies}.")
    rows: list[dict[str, object]] = []
    for record in _records(manifest, args.split):
        luminance, timestamps, masks, intrinsics, oracle = _load_scene(args.dataset_root / str(record["scene"]), Path(str(record["output"])))
        arrays = _scene_arrays(Path(str(record["output"])))
        try:
            scene = (luminance, timestamps, masks, intrinsics)
            for policy in policies:
                result = _run(
                    scene,
                    oracle,
                    arrays,
                    objective,
                    policy=policy,
                    thresholds=thresholds,
                    model=model,
                    checkpoint=checkpoint,
                    gears=gears,
                    quality_floor=quality_floor,
                    teacher=teacher,
                    confidence=args.confidence,
                    require_imu=args.require_imu,
                    sticky=args.sticky,
                )
                rows.append({"scene": record["scene"], "condition": record["condition"], "seed": int(record["seed"]), "policy": policy, **{key: value for key, value in result.items() if key != "bins"}, "bins": result["bins"]})
        finally:
            oracle.close()
    aggregate: dict[str, dict[str, float]] = {}
    for policy in policies:
        selected = [row for row in rows if row["policy"] == policy]
        summary = {key: _mean(selected, key) for key in ("mean_rpm", "mean_event_density", "mean_temporal_overlap", "mean_edge_coverage", "mean_edge_f1", "storm_rate", "spearman_motion_rpm", "slow_minus_fast_rpm")}
        for rpm in (0.0, 180.0, 360.0):
            summary[f"frac_{int(rpm)}"] = float(np.mean([float(row["rpm_hist"].get(str(rpm), 0.0)) for row in selected]))
        for bin_name in ("slow", "mid", "fast"):
            summary[f"{bin_name}_rpm"] = float(np.mean([float(row["bins"][bin_name]["mean_rpm"]) for row in selected]))
            summary[f"{bin_name}_density"] = float(np.mean([float(row["bins"][bin_name]["mean_event_density"]) for row in selected]))
            summary[f"{bin_name}_storm_rate"] = float(np.mean([float(row["bins"][bin_name]["storm_rate"]) for row in selected]))
            summary[f"{bin_name}_coverage"] = float(np.mean([float(row["bins"][bin_name]["mean_edge_coverage"]) for row in selected]))
            summary[f"{bin_name}_edge_f1"] = float(np.mean([float(row["bins"][bin_name]["mean_edge_f1"]) for row in selected]))
        for condition in ("camera_only", "object_only", "co_motion", "counter_motion"):
            subset = [row for row in selected if row["condition"] == condition]
            if subset:
                summary[f"{condition}_edge_f1"] = float(np.mean([float(row["mean_edge_f1"]) for row in subset]))
                for rpm in (0.0, 180.0, 360.0):
                    summary[f"{condition}_frac_{int(rpm)}"] = float(np.mean([float(row["rpm_hist"].get(str(rpm), 0.0)) for row in subset]))
        aggregate[policy] = summary
    inverse = aggregate.get("inverse_learned") or aggregate.get("deficit_rule") or aggregate.get("inverse_visual")
    if inverse is None:
        inverse = next(iter(aggregate.values()))
    behavior_check = {
        "learned_or_visual_spearman_motion_rpm": inverse["spearman_motion_rpm"],
        "slow_rpm_minus_fast_rpm": inverse["slow_minus_fast_rpm"],
        "fast_bin_storm_vs_fixed360": inverse["fast_storm_rate"] - aggregate["fixed_360"]["fast_storm_rate"],
        "slow_bin_coverage_vs_fixed180": inverse["slow_coverage"] - aggregate["fixed_180"]["slow_coverage"],
        "mean_edge_f1_vs_fixed180": inverse["mean_edge_f1"] - aggregate["fixed_180"]["mean_edge_f1"],
        "co_motion_edge_f1_vs_fixed180": inverse.get("co_motion_edge_f1", float("nan")) - aggregate["fixed_180"].get("co_motion_edge_f1", float("nan")),
        "object_only_edge_f1_vs_fixed180": inverse.get("object_only_edge_f1", float("nan")) - aggregate["fixed_180"].get("object_only_edge_f1", float("nan")),
        "inverts_motion": bool(inverse["slow_minus_fast_rpm"] > 0.0 and inverse["spearman_motion_rpm"] < 0.0),
        "beats_fixed180_f1": bool(inverse["mean_edge_f1"] > aggregate["fixed_180"]["mean_edge_f1"]),
        "beats_fixed360_f1": bool(inverse["mean_edge_f1"] > aggregate["fixed_360"]["mean_edge_f1"]),
    }
    if "fixed_0" in aggregate:
        behavior_check["mean_edge_f1_vs_fixed0"] = inverse["mean_edge_f1"] - aggregate["fixed_0"]["mean_edge_f1"]
        behavior_check["beats_fixed0_f1"] = bool(inverse["mean_edge_f1"] > aggregate["fixed_0"]["mean_edge_f1"])
    if "deficit_rule" in aggregate:
        behavior_check["deficit_mean_edge_f1_vs_fixed180"] = aggregate["deficit_rule"]["mean_edge_f1"] - aggregate["fixed_180"]["mean_edge_f1"]
        behavior_check["deficit_co_motion_edge_f1_vs_fixed180"] = aggregate["deficit_rule"].get("co_motion_edge_f1", float("nan")) - aggregate["fixed_180"].get("co_motion_edge_f1", float("nan"))
        behavior_check["deficit_beats_fixed180_f1"] = bool(aggregate["deficit_rule"]["mean_edge_f1"] > aggregate["fixed_180"]["mean_edge_f1"])
    report = {
        "evaluation": "prism-speed control with downstream edge F1",
        "split": args.split,
        "teacher": teacher,
        "motion_source": "causal lagged visual_motion, IMU, and event-storm / deficit feedback",
        "quality_floor": quality_floor,
        "motion_thresholds": list(thresholds),
        "candidate_rpm": list(gears),
        "storm_density": STORM_DENSITY,
        "storm_overlap": STORM_OVERLAP,
        "confidence": args.confidence,
        "require_imu": args.require_imu,
        "sticky": args.sticky,
        "student": None if args.student is None else str(args.student),
        "aggregate": aggregate,
        "behavior_check": behavior_check,
        "rows": rows,
        "limitations": [
            "visual_motion is a global brightness-change score; co-motion background motion can look 'fast' even when the object is static.",
            "On the procedural pack the teacher may select 0 RPM. That is allowed when quality_floor is false.",
            "These F1 numbers are object-mask scores on 64x64 procedural scenes. Do not copy them into the EVEYES Table 1.",
            "The F1 teacher picks the lowest RPM among {0, 180, 360} with the best teacher edge F1; storm occupancy refuses 360.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"split": args.split, "aggregate": aggregate, "behavior_check": report["behavior_check"]}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
