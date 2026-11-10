#!/usr/bin/env python3
"""Build read-only appendix summaries from existing JSON/CSV artifacts."""

from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/ami_next"


def _mean(rows: list[dict], key: str) -> float | None:
    values = [float(row[key]) for row in rows if key in row]
    return float(sum(values) / len(values)) if values else None


def build_dsec() -> dict:
    source = ROOT / "results/dsec_consistent_ami_compensated_120f_4s_stable.json"
    data = json.loads(source.read_text(encoding="utf-8"))
    rows = data["frames_detail"]
    methods = {
        "Passive": {
            "event_count": _mean(rows, "passive_events_50ms"),
            "spatial_coverage": _mean(rows, "passive_coverage"),
            "raw_flicker": None,
            "compensated_flicker": None,
            "mean_rpm": None,
        },
        "Fixed AMI (stored label)": {
            "event_count": _mean(rows, "fixed_events_50ms"),
            "spatial_coverage": _mean(rows, "fixed_coverage"),
            "raw_flicker": _mean(rows, "fixed_raw_flicker"),
            "compensated_flicker": _mean(rows, "fixed_comp_flicker"),
            "mean_rpm": None,
        },
        "IMU Gear / adaptive": {
            "event_count": _mean(rows, "imu_gear_events_50ms"),
            "spatial_coverage": _mean(rows, "imu_gear_coverage"),
            "raw_flicker": _mean(rows, "imu_gear_raw_flicker"),
            "compensated_flicker": _mean(rows, "imu_gear_comp_flicker"),
            "mean_rpm": _mean(rows, "actual_rpm"),
        },
        "Native DSEC reference": {
            "event_count": _mean(rows, "dsec_reference_events_50ms"),
            "spatial_coverage": None,
            "raw_flicker": None,
            "compensated_flicker": None,
            "mean_rpm": None,
        },
    }
    report = {
        "source": str(source),
        "sequence": "DSEC thun_00_a",
        "frames": len(rows),
        "event_window_ms": data.get("event_window_ms"),
        "simulation_resolution": data.get("simulation_resolution"),
        "aggregation": "arithmetic mean over stored frames_detail rows",
        "methods": methods,
        "limitations": [
            "Native DSEC has only event count in this result; coverage/flicker are not defined for a like-for-like comparison.",
            "The stored JSON does not record the Fixed AMI base RPM. The generator default is 720 RPM, so this output must not be labeled Fixed 180 RPM.",
            "The optical modulation is offline RGB+IMU simulation; native DSEC is an unaligned real-event reference.",
        ],
    }
    path = OUT / "dsec_appendix_statistics.json"
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    with (OUT / "dsec_appendix_statistics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["method", "event_count", "spatial_coverage", "raw_flicker", "compensated_flicker", "mean_rpm"])
        writer.writeheader()
        for method, values in methods.items():
            writer.writerow({"method": method, **values})
    tex = [
        "% Generated from dsec_appendix_statistics.json",
        "\\begin{tabular}{lrrrrr}",
        "Method & Events/window & Coverage & Raw flicker & Comp. flicker & Mean RPM \\\\",
        "\\hline",
    ]
    for method, values in methods.items():
        def fmt(value: float | None) -> str:
            return "--" if value is None else f"{value:.4f}"
        tex.append(f"{method} & {fmt(values['event_count'])} & {fmt(values['spatial_coverage'])} & {fmt(values['raw_flicker'])} & {fmt(values['compensated_flicker'])} & {fmt(values['mean_rpm'])} \\\\")
    tex.append("\\end{tabular}")
    (OUT / "dsec_appendix_statistics.tex").write_text("\n".join(tex) + "\n", encoding="utf-8")
    return report


def build_comotion() -> dict:
    f1_path = OUT / "main_f1_test_current.json"
    teo_path = OUT / "main_teo_test_current.json"
    f1 = json.loads(f1_path.read_text(encoding="utf-8"))["rows"]
    teo = json.loads(teo_path.read_text(encoding="utf-8"))["rows"]
    selected: list[dict] = []
    for method in ("dvs_voltmeter", "v2e"):
        rows = [row for row in teo if row["method"] == method and row["condition"] == "co_motion" and row["scene"] == "test_seed09_co_motion"][:3]
        for row in rows:
            f1_row = next(item for item in f1 if item["method"] == method and item["condition"] == "co_motion" and item["scene"] == row["scene"] and item["window"] == row["window"])
            selected.append({
                "method": method,
                "scene": row["scene"],
                "window": int(row["window"]),
                "event_count": int(f1_row["event_count"]),
                "target_event_count": int(row["target_event_count"]),
                "background_event_count": int(row["background_event_count"]),
                "coverage_C": float(row["target_coverage"]),
                "purity_Pi": float(row["target_purity"]),
                "temporal_support_tau": float(row["target_temporal_support"]),
                "teo": float(row["teo"]),
                "image_edge_f1": float(f1_row["edge_f1"]),
            })
    report = {
        "source_f1": str(f1_path),
        "source_teo": str(teo_path),
        "selection": "first three windows of test_seed09_co_motion for each passive method",
        "rows": selected,
        "interpretation": "Nonzero total/background activity with zero target support; F1/TEO are zero because target-edge overlap is zero.",
    }
    (OUT / "co_motion_passive_failure_summary.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    with (OUT / "co_motion_passive_failure_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(selected[0]))
        writer.writeheader(); writer.writerows(selected)
    tex = [
        "% Generated from co_motion_passive_failure_summary.json",
        "\\begin{tabular}{llrrrrrrrr}",
        "Method & Window & Events & Target & Background & $C$ & $\\Pi$ & $\\tau$ & TEO & Edge F1 \\\\",
        "\\hline",
    ]
    for row in selected:
        tex.append(f"{row['method']} & {row['window']} & {row['event_count']} & {row['target_event_count']} & {row['background_event_count']} & {row['coverage_C']:.3f} & {row['purity_Pi']:.3f} & {row['temporal_support_tau']:.3f} & {row['teo']:.3f} & {row['image_edge_f1']:.3f} \\\\")
    tex.append("\\end{tabular}")
    (OUT / "co_motion_passive_failure_summary.tex").write_text("\n".join(tex) + "\n", encoding="utf-8")
    return report


def build_motion() -> dict:
    files = [OUT / "motion_excitation_sweep" / "scenes.json", OUT / "counter_motion_excitation_sweep" / "scenes.json"]
    counts = {}
    ids = {}
    for path in files:
        records = json.loads(path.read_text(encoding="utf-8"))
        by_sweep: dict[str, dict[str, object]] = {}
        for record in records:
            sweep = str(record["sweep"])
            block = by_sweep.setdefault(sweep, {"records": 0, "seeds": set(), "scene_ids": [], "points": Counter()})
            block["records"] += 1
            block["seeds"].add(int(record["seed"]))
            block["scene_ids"].append(str(record["scene"]))
            point = (record.get("motion_scale"), record.get("object_alpha"), record.get("grid_camera_scale"), record.get("grid_object_scale"))
            block["points"][str(point)] += 1
        for sweep, block in by_sweep.items():
            counts[sweep] = {
                "records": block["records"],
                "seed_count": len(block["seeds"]),
                "seeds": sorted(block["seeds"]),
                "point_count": len(block["points"]),
                "records_per_point": sorted(set(block["points"].values())),
            }
            ids[sweep] = sorted(block["scene_ids"])
    report = {
        "sources": [str(path) for path in files],
        "aggregation": "plot points are arithmetic means over records sharing the plotted motion coordinate; each point has three seed-indexed rendered scene realizations.",
        "condition_counts": counts,
        "seed_indexed_scene_ids": ids,
        "fixed_methods": ["fixed_0", "fixed_180", "fixed_360"],
        "adaptive_method": "adaptive",
        "metrics_present_in_records": ["teo", "event_count", "unique_pixel_ratio", "coverage_C", "purity_Pi", "tau", "target_f1"],
        "no_error_bars": True,
    }
    (OUT / "motion_excitation_audit_summary.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def build_compensation() -> dict:
    sources = [OUT / "compensation_ablation_camera_only.json", OUT / "dl3dv_compensation_ablation.json"]
    report = {
        "sources": [str(path) for path in sources],
        "existing_qualitative_figure": str(OUT / "figures/timestamp_aware_compensation_temple_best.png"),
        "existing_qualitative_pdf": str(OUT / "figures/timestamp_aware_compensation_temple_best.pdf"),
        "visualization_script": str(ROOT / "scripts/plot_timestamp_aware_compensation.py"),
        "generated_event_file": None,
        "generated_event_source": "computed by _ami_windows from data/dl3dv_large5 RGB/timestamps; no sensor-event NPZ is serialized for this compensation figure",
        "same_event_stream": True,
        "variants": ["no_compensation", "interval_compensation", "per_event_compensation"],
        "notes": "The evaluator constructs all three views from result['sensor_events']; interval uses midpoint inverse warp and per-event uses each event timestamp. Valid in-image counts can differ after warping, but the source event identity is shared.",
        "scenes": ["EVEYES Scene_005 camera_only", "DL3DV-large5 screen-000083/087/089/095/108"],
    }
    (OUT / "compensation_appendix_audit.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def build_missing(dsec: dict) -> str:
    text = """APPENDIX EXPERIMENT ARTIFACTS AUDIT\n=================================\n\nGenerated from existing repository outputs only. No training or full benchmark rerun was performed.\n\nDSEC\n----\nAvailable: native DSEC event stream, passive generated stream, Fixed AMI generated stream, IMU Gear/adaptive generated stream, compensated/raw display variants, per-frame event counts, spatial coverage, raw flicker, compensated flicker, gyro magnitude, and actual RPM.\n\nProduced artifacts:\n  results/ami_next/figures/dsec_appendix_qualitative.png\n  results/ami_next/figures/dsec_appendix_qualitative.pdf\n  results/ami_next/figures/dsec_appendix_qualitative.json\n  results/ami_next/dsec_appendix_statistics.json\n  results/ami_next/dsec_appendix_statistics.csv\n  results/ami_next/dsec_appendix_statistics.tex\n\nDSEC limitation: the stored JSON does not record the Fixed AMI base RPM. The generator default is 720 RPM, so the stored result cannot safely be labeled Fixed 180 RPM. Native DSEC coverage/flicker are not placed in the same comparable table because those definitions are absent. No DSEC F1/TEO/FSIM/LPIPS was added.\n\nCO-MOTION PASSIVE DIAGNOSTIC\n-----------------------------\nProduced artifacts:\n  results/ami_next/co_motion_passive_failure_summary.json\n  results/ami_next/co_motion_passive_failure_summary.csv\n  results/ami_next/co_motion_passive_failure_summary.tex\n\nThe selected rows are three windows from test_seed09_co_motion for DVS-Voltmeter and v2e. They have nonzero total/background events but zero target events, C, Pi, tau, TEO, and image-edge F1.\n\nCOMPENSATION QUALITATIVE\n------------------------\nExisting artifacts reused:\n  results/ami_next/figures/timestamp_aware_compensation_temple_best.png\n  results/ami_next/figures/timestamp_aware_compensation_temple_best.pdf\n  scripts/plot_timestamp_aware_compensation.py\n  results/ami_next/compensation_ablation_camera_only.json\n  results/ami_next/dl3dv_compensation_ablation.json\n\nThe evaluator confirms a shared sensor-event stream with no, interval-midpoint, and per-event inverse warp variants. No new sensing condition was generated.\n\nMOTION-EXCITATION AUDIT\n------------------------\nProduced artifact:\n  results/ami_next/motion_excitation_audit_summary.json\n\nExisting sources:\n  results/ami_next/motion_excitation_sweep/scenes.json\n  results/ami_next/counter_motion_excitation_sweep/scenes.json\n\nEach plotted point is the arithmetic mean over three seed-indexed rendered scene realizations. These are not described as independent random trials or statistical significance experiments. No new error bars were generated.\n\nMISSING-DATA REPORT\n-------------------\n1. A DSEC result explicitly labeled Fixed 180 RPM cannot be confirmed from the stored long-video JSON; the stored Fixed AMI base RPM is not recorded and the generator default is 720 RPM.\n2. A strict native-DSEC versus Fixed180 versus adaptive table with identical named columns is therefore not claimed.\n3. DSEC formal F1/TEO/FSIM/LPIPS is unavailable under an existing protocol and was not invented.\n4. Controller-specific Fixed360/EIRM TEO is unavailable in the existing controller JSON.\n5. Confirmed provenance of object_masks as Blender GT is unavailable from the evaluator metadata.\n\nMinimum reruns if those missing claims become necessary:\n  - rerun the existing DSEC renderer with an explicitly recorded --base-rpm 180 and the existing IMU Gear path;\n  - run the existing TEO evaluator on the controller-ablation streams;\n  - do not expand to new benchmarks or new metrics.\n"""
    path = OUT / "APPENDIX_EXPERIMENT_ARTIFACTS_AUDIT.txt"
    path.write_text(text, encoding="utf-8")
    return text


def main() -> int:
    dsec = build_dsec()
    build_comotion()
    build_motion()
    build_compensation()
    build_missing(dsec)
    print("wrote appendix audit artifacts")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
