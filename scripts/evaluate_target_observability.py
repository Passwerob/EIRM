#!/usr/bin/env python3
"""Compare DVS-Voltmeter, v2e, and current AMI on Target Event Observability."""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

import numpy as np
import torch

from ami_simulator.event_quality import TARGET_OBSERVABILITY_VERSION, compute_reference_edge_mask, target_observability_metrics
from ami_simulator.geometry import RotatingWedgePrism
from ami_simulator.inverse_motion import QUALITY_INVERSE_RPM
from ami_simulator.motion_net import TinyCausalGearPolicy
from scripts.evaluate_dvs_v2e_f1_ours import _rollout
from scripts.evaluate_dvs_v2e_ours import _external, _slice
from scripts.evaluate_phase_continuous_control import _load_scene
from scripts.train_inverse_motion_policy import FEATURE_DIM, _records, _scene_arrays

CONDITIONS = ("camera_only", "object_only", "co_motion", "counter_motion")
SCALAR_KEYS = ("teo", "target_f1", "target_coverage", "target_purity", "target_temporal_support", "edge_f1", "background_event_count", "target_event_count")
ALL_METHODS = ("dvs_voltmeter", "v2e", "fixed_180", "fixed_360", "ours_f1_imu")
AMI_POLICIES = (("fixed_180", "fixed_180"), ("fixed_360", "fixed_360"), ("inverse_learned", "ours_f1_imu"))
EXTERNAL_METHODS = ("dvs_voltmeter", "v2e")


def _cache_path(root: Path, split: str, scene: str, method: str) -> Path:
    return root / split / scene / f"{method}.npz"


def _write_events(path: Path, events) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, x=np.asarray(events.x), y=np.asarray(events.y), t=np.asarray(events.t), p=np.asarray(events.p))


def _read_cached(path: Path):
    from ami_simulator.types import EventStream

    with np.load(path) as data:
        return EventStream(data["x"], data["y"], data["t"], data["p"]).sorted_by_time()


def _window_scores(events, luminance: np.ndarray, mask: np.ndarray, start: float, end: float) -> dict[str, float]:
    from ami_simulator.event_quality import event_quality_metrics

    edges = compute_reference_edge_mask(luminance, edge_quantile=0.75) & mask
    teo = target_observability_metrics(events, target_mask=mask, target_edge_mask=edges, image_shape=luminance.shape, time_range=(start, end))
    wide = event_quality_metrics(events, reference_edge_mask=edges, image_shape=luminance.shape)
    return {
        "teo": float(teo["teo"]),
        "target_f1": float(teo["target_f1"]),
        "target_coverage": float(teo["target_coverage"]),
        "target_purity": float(teo["target_purity"]),
        "target_temporal_support": float(teo["target_temporal_support"]),
        "edge_f1": float(wide["edge_f1"]),
        "background_event_count": int(teo["background_event_count"]),
        "target_event_count": int(teo["target_event_count"]),
    }


def _summarize(rows: list[dict[str, object]], methods: tuple[str, ...]) -> dict[str, dict[str, float]]:
    summary: dict[str, dict[str, float]] = {}
    for method in methods:
        selected = [row for row in rows if row["method"] == method]
        block = {key: float(np.mean([float(row[key]) for row in selected])) for key in SCALAR_KEYS}
        for condition in CONDITIONS:
            subset = [row for row in selected if row["condition"] == condition]
            for key in SCALAR_KEYS:
                block[f"{condition}_{key}"] = float(np.mean([float(row[key]) for row in subset])) if subset else float("nan")
        summary[method] = block
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("results/comotion_v2_learned/manifest.json"))
    parser.add_argument("--dataset-root", type=Path, default=Path("data/comotion_controlled_v2"))
    parser.add_argument("--student", type=Path, default=Path("results/comotion_f1_argmax_policy/policy.pt"))
    parser.add_argument("--split", choices=("valid", "test"), default="test")
    parser.add_argument("--cache", type=Path, default=Path("results/ami_next/baseline_events"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--v2e-root", type=Path, default=Path("/data/fcr/vendor/v2e"))
    parser.add_argument("--v2e-python", type=Path, default=Path("/data/fcr/miniconda3/envs/v2e/bin/python"))
    parser.add_argument("--dvs-root", type=Path, default=Path("/data/fcr/code/event_vggt/Data_Part/DVS-Voltmeter"))
    parser.add_argument("--dvs-python", type=Path, default=Path("/data/fcr/miniconda3/envs/dvsvolt/bin/python"))
    parser.add_argument("--confidence", type=float, default=0.70)
    parser.add_argument("--require-imu", action="store_true", default=True)
    parser.add_argument("--no-require-imu", action="store_false", dest="require_imu")
    parser.add_argument("--dwell", type=float, default=0.025)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--max-scenes", type=int, default=0)
    parser.add_argument("--only-methods", nargs="+", default=None, help="Score only these methods; others can be merged from --merge-from.")
    parser.add_argument("--merge-from", type=Path, default=None, help="Reuse existing TEO rows for methods not being re-scored.")
    args = parser.parse_args()

    requested = tuple(args.only_methods) if args.only_methods else ALL_METHODS
    unknown = [method for method in requested if method not in ALL_METHODS]
    if unknown:
        parser.error(f"Unknown methods: {unknown}. Choose from {ALL_METHODS}.")
    methods = requested
    rows: list[dict[str, object]] = []
    if args.merge_from is not None:
        previous = json.loads(args.merge_from.read_text(encoding="utf-8"))
        rows.extend(row for row in previous["rows"] if row["method"] not in methods)

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    checkpoint = torch.load(args.student, map_location="cpu", weights_only=True)
    model = TinyCausalGearPolicy(feature_dim=FEATURE_DIM, hidden_dim=int(checkpoint["hidden_dim"]), num_gears=int(checkpoint["num_gears"]))
    model.load_state_dict(checkpoint["model"])
    model.eval()
    objective = {**manifest["configuration"], "event_efficiency_weight": 0.0}
    completed = 0
    ami_jobs = [(policy, method) for policy, method in AMI_POLICIES if method in methods]
    external_jobs = [method for method in EXTERNAL_METHODS if method in methods]
    for record in _records(manifest, args.split):
        if args.max_scenes and completed >= args.max_scenes:
            break
        luminance, timestamps, masks, intrinsics, oracle = _load_scene(args.dataset_root / str(record["scene"]), Path(str(record["output"])))
        arrays = _scene_arrays(Path(str(record["output"])))
        scene = (luminance, timestamps, masks, intrinsics)
        try:
            streams: dict[str, list[object]] = {}
            if external_jobs:
                with tempfile.TemporaryDirectory(prefix="ami_teo_") as temporary:
                    for method in external_jobs:
                        cached = _cache_path(args.cache, args.split, str(record["scene"]), method)
                        if cached.exists():
                            events = _read_cached(cached)
                        else:
                            events = _external(method, luminance, timestamps, args, Path(temporary))
                            _write_events(cached, events)
                        streams[method] = [_slice(events, float(timestamps[index]), float(timestamps[index + 1])) for index in range(len(timestamps) - 1)]
            for policy, method in ami_jobs:
                streams[method] = _rollout(
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
            for method, windows in streams.items():
                for index, events in enumerate(windows):
                    scores = _window_scores(events, luminance[index], masks[index], float(timestamps[index]), float(timestamps[index + 1]))
                    rows.append({"scene": record["scene"], "condition": record["condition"], "window": index, "method": method, **scores})
        finally:
            oracle.close()
        print(f"evaluated {record['scene']}")
        completed += 1

    methods_out = tuple(method for method in ALL_METHODS if any(row["method"] == method for row in rows))
    aggregate = _summarize(rows, methods_out)
    ours = aggregate.get("ours_f1_imu", {})
    decision: dict[str, float | bool] = {}
    if ours:
        for name, key in (("dvs", "dvs_voltmeter"), ("v2e", "v2e"), ("fixed180", "fixed_180"), ("fixed360", "fixed_360")):
            if key not in aggregate:
                continue
            decision[f"ours_teo_vs_{name}"] = ours["teo"] - aggregate[key]["teo"]
            decision[f"ours_object_only_teo_vs_{name}"] = ours["object_only_teo"] - aggregate[key]["object_only_teo"]
            decision[f"ours_co_motion_teo_vs_{name}"] = ours["co_motion_teo"] - aggregate[key]["co_motion_teo"]
            decision[f"beats_{name}_teo"] = bool(ours["teo"] > aggregate[key]["teo"])
            decision[f"beats_{name}_object_only_teo"] = bool(ours["object_only_teo"] > aggregate[key]["object_only_teo"])
            decision[f"beats_{name}_co_motion_teo"] = bool(ours["co_motion_teo"] > aggregate[key]["co_motion_teo"])
    report = {
        "protocol": TARGET_OBSERVABILITY_VERSION,
        "split": args.split,
        "student": str(args.student),
        "confidence": args.confidence,
        "require_imu": args.require_imu,
        "methods": list(methods_out),
        "aggregate": aggregate,
        "decision": decision,
        "definition": {
            "coverage": "unique target-edge pixels with events / target-edge pixels",
            "purity": "unique target-edge pixels with events / unique target pixels with events",
            "temporal_support": "fraction of the control window with at least one target-edge event",
            "teo": "target_f1(coverage, purity) * temporal_support",
            "background_events": "ignored by TEO; they still appear in image-wide edge_f1",
        },
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    printable = {method: {key: aggregate[method][key] for key in ("teo", "target_f1", "target_coverage", "target_purity", "target_temporal_support", "edge_f1", "object_only_teo", "co_motion_teo", "camera_only_teo", "counter_motion_teo")} for method in methods_out}
    print(json.dumps({"split": args.split, "aggregate": printable, "decision": report["decision"]}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
