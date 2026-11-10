#!/usr/bin/env python3
"""Same-source downstream comparison: DVS-Voltmeter, v2e, and current AMI control.

DVS-Voltmeter and v2e rows are reused from the frozen protocol JSON so the
external simulators are not rerun. Current Ours and fixed 180 RPM are rolled
out with the same window scorer as that protocol.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from ami_simulator.geometry import RotatingWedgePrism
from ami_simulator.inverse_motion import QUALITY_INVERSE_RPM
from ami_simulator.motion_net import TinyCausalGearPolicy
from scripts.evaluate_dvs_v2e_ours import _window_metrics
from scripts.evaluate_inverse_motion_control import _choose_rpm
from scripts.evaluate_phase_continuous_control import _load_scene
from scripts.run_control_headroom_stateful import RolloutState, _transition
from scripts.train_inverse_motion_policy import FEATURE_DIM, _records, _scene_arrays

METRIC_KEYS = (
    "edge_f1",
    "edge_precision",
    "edge_recall",
    "event_count",
    "temporal_support",
    "spatial_matching_f1",
    "feature_repeatability",
    "descriptor_true_matches",
)
CONDITIONS = ("camera_only", "object_only", "co_motion", "counter_motion")


def _score_streams(windows: list[object], luminance: np.ndarray, masks: np.ndarray, scene: str, condition: str, method: str) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    previous = None
    for index, events in enumerate(windows):
        metrics, previous = _window_metrics(events, luminance[index], masks[index], previous)
        rows.append({"scene": scene, "condition": condition, "window": index, "method": method, **metrics})
    return rows


def _rollout(
    scene: tuple[np.ndarray, np.ndarray, np.ndarray, object],
    oracle: np.lib.npyio.NpzFile,
    arrays: dict[str, np.ndarray],
    objective: dict[str, float],
    *,
    policy: str,
    model: TinyCausalGearPolicy | None,
    checkpoint: dict[str, object] | None,
    confidence: float,
    require_imu: bool,
    dwell: float,
) -> list[object]:
    luminance, timestamps, _masks, intrinsics = scene
    prism = RotatingWedgePrism(intrinsics, 2.0)
    state = RolloutState(0.0, None, None, 180.0, float(timestamps[0]) - dwell)
    features: list[np.ndarray] = []
    streams: list[object] = []
    teacher = str(checkpoint.get("teacher", "f1-argmax")) if checkpoint is not None else "f1-argmax"
    thresholds = tuple(float(value) for value in checkpoint["motion_thresholds"]) if checkpoint is not None else (0.0, 1.0)
    for index in range(len(timestamps) - 1):
        motion = float(arrays["visual"][index - 1] if index else 0.0)
        previous_density = 0.0 if state.previous_metrics is None else float(state.previous_metrics["event_density"])
        previous_overlap = 0.0 if state.previous_metrics is None else float(state.previous_metrics["temporal_overlap"])
        feature = np.asarray((motion, float(arrays["imu"][index - 1] if index else 0.0), previous_density, previous_overlap), dtype=np.float32)
        features.append(feature)
        rpm = _choose_rpm(
            policy,
            motion=motion,
            thresholds=thresholds,
            previous_density=previous_density,
            previous_overlap=previous_overlap,
            features=features,
            model=model,
            checkpoint=checkpoint,
            gears=QUALITY_INVERSE_RPM,
            quality_floor=True,
            teacher=teacher,
            confidence=confidence,
            require_imu=require_imu,
        )
        if index and float(timestamps[index] - state.last_switch_time) < dwell:
            rpm = state.previous_rpm
        result, state = _transition(scene, oracle, objective, prism, state, index, rpm, dwell=dwell)
        streams.append(result["reference_events"])
    return streams


def _mean(rows: list[dict[str, object]], key: str) -> float:
    return float(np.mean([float(row[key]) for row in rows])) if rows else float("nan")


def _summarize(rows: list[dict[str, object]], methods: tuple[str, ...]) -> dict[str, dict[str, float]]:
    summary: dict[str, dict[str, float]] = {}
    for method in methods:
        selected = [row for row in rows if row["method"] == method]
        block = {key: _mean(selected, key) for key in METRIC_KEYS}
        for condition in CONDITIONS:
            subset = [row for row in selected if row["condition"] == condition]
            for key in METRIC_KEYS:
                block[f"{condition}_{key}"] = _mean(subset, key)
        summary[method] = block
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("results/comotion_v2_learned/manifest.json"))
    parser.add_argument("--dataset-root", type=Path, default=Path("data/comotion_controlled_v2"))
    parser.add_argument("--baseline", type=Path, default=Path("results/ami_next/downstream_dvs_v2e_ours_test.json"))
    parser.add_argument("--student", type=Path, default=Path("results/comotion_f1_argmax_policy/policy.pt"))
    parser.add_argument("--split", choices=("valid", "test"), default="test")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--confidence", type=float, default=0.70)
    parser.add_argument("--require-imu", action="store_true", default=True)
    parser.add_argument("--no-require-imu", action="store_false", dest="require_imu")
    parser.add_argument("--dwell", type=float, default=0.025)
    args = parser.parse_args()

    baseline = json.loads(args.baseline.read_text(encoding="utf-8"))
    if baseline.get("protocol") != "ami_downstream_dvs_v2e_ours_v1":
        raise ValueError("Baseline JSON must be the frozen DVS/v2e/ours protocol.")
    reused: list[dict[str, object]] = []
    for row in baseline["rows"]:
        method = str(row["method"])
        mapped = {"dvs_voltmeter": "dvs_voltmeter", "v2e": "v2e", "ours": "ours_horizon2_q"}[method]
        reused.append({**row, "method": mapped})

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    checkpoint = torch.load(args.student, map_location="cpu", weights_only=True)
    if checkpoint.get("architecture") != "tiny_inverse_motion_gear":
        raise ValueError("Expected the F1 / inverse-motion gear checkpoint.")
    model = TinyCausalGearPolicy(feature_dim=FEATURE_DIM, hidden_dim=int(checkpoint["hidden_dim"]), num_gears=int(checkpoint["num_gears"]))
    model.load_state_dict(checkpoint["model"])
    model.eval()
    objective = {**manifest["configuration"], "event_efficiency_weight": 0.0}
    rolled: list[dict[str, object]] = []
    for record in _records(manifest, args.split):
        luminance, timestamps, masks, intrinsics, oracle = _load_scene(args.dataset_root / str(record["scene"]), Path(str(record["output"])))
        arrays = _scene_arrays(Path(str(record["output"])))
        scene = (luminance, timestamps, masks, intrinsics)
        try:
            for policy, method in (("fixed_180", "fixed_180"), ("inverse_learned", "ours_f1_imu")):
                streams = _rollout(
                    scene,
                    oracle,
                    arrays,
                    objective,
                    policy=policy,
                    model=model,
                    checkpoint=checkpoint,
                    confidence=args.confidence,
                    require_imu=args.require_imu,
                    dwell=args.dwell,
                )
                rolled.extend(_score_streams(streams, luminance, masks, str(record["scene"]), str(record["condition"]), method))
        finally:
            oracle.close()
        print(f"evaluated {record['scene']}")

    methods = ("dvs_voltmeter", "v2e", "fixed_180", "ours_horizon2_q", "ours_f1_imu")
    rows = reused + rolled
    aggregate = _summarize(rows, methods)
    ours = aggregate["ours_f1_imu"]
    report = {
        "protocol": "ami_downstream_dvs_v2e_f1_ours_v1",
        "split": args.split,
        "baseline": str(args.baseline),
        "student": str(args.student),
        "confidence": args.confidence,
        "require_imu": args.require_imu,
        "dwell": args.dwell,
        "methods": list(methods),
        "aggregate": aggregate,
        "decision": {
            "ours_vs_dvs_edge_f1": ours["edge_f1"] - aggregate["dvs_voltmeter"]["edge_f1"],
            "ours_vs_v2e_edge_f1": ours["edge_f1"] - aggregate["v2e"]["edge_f1"],
            "ours_vs_dvs_co_motion_edge_f1": ours["co_motion_edge_f1"] - aggregate["dvs_voltmeter"]["co_motion_edge_f1"],
            "ours_vs_v2e_co_motion_edge_f1": ours["co_motion_edge_f1"] - aggregate["v2e"]["co_motion_edge_f1"],
            "ours_vs_dvs_object_only_edge_f1": ours["object_only_edge_f1"] - aggregate["dvs_voltmeter"]["object_only_edge_f1"],
            "ours_vs_v2e_object_only_edge_f1": ours["object_only_edge_f1"] - aggregate["v2e"]["object_only_edge_f1"],
            "passive_co_motion_edge_f1_is_zero": bool(aggregate["dvs_voltmeter"]["co_motion_edge_f1"] < 1e-6 and aggregate["v2e"]["co_motion_edge_f1"] < 1e-6),
            "beats_dvs_mean_f1": bool(ours["edge_f1"] > aggregate["dvs_voltmeter"]["edge_f1"]),
            "beats_v2e_mean_f1": bool(ours["edge_f1"] > aggregate["v2e"]["edge_f1"]),
            "beats_dvs_co_motion_f1": bool(ours["co_motion_edge_f1"] > aggregate["dvs_voltmeter"]["co_motion_edge_f1"]),
            "beats_v2e_co_motion_f1": bool(ours["co_motion_edge_f1"] > aggregate["v2e"]["co_motion_edge_f1"]),
        },
        "fairness": "same source RGB, timestamps, windows, resolution, object-ROI edge mask, and scorer; DVS/v2e native events from frozen protocol; Ours inverse-warped to the reference plane",
        "limitations": [
            "Native event budgets are not matched. DVS emits fewer events than v2e/Ours; co-motion failure of DVS/v2e is not a count artifact (they still emit events, just not on the object).",
            "Contrast-maximization (CM-IWE) is not included in this run.",
            "Synthetic controlled-motion pack; no prism hardware validation.",
            "Object-only is a negative control: passive DVS/v2e already see object motion, so prism excitation can lower precision.",
        ],
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    printable = {"split": args.split, "aggregate": {method: {key: aggregate[method][key] for key in ("edge_f1", "edge_precision", "edge_recall", "spatial_matching_f1", "event_count", "co_motion_edge_f1", "object_only_edge_f1", "camera_only_edge_f1", "counter_motion_edge_f1")} for method in methods}, "decision": report["decision"]}
    print(json.dumps(printable, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
