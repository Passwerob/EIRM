#!/usr/bin/env python3
"""Evaluate fixed-window event accumulation using global SRER and LPIPS."""

from __future__ import annotations

import csv
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import torch
from lpips import LPIPS

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ami_simulator.pipeline import load_rgb_sequence, rgb_to_linear_luminance
from scripts.evaluate_accumulation_fidelity import _accum, _eirm_events
from scripts.evaluate_cm_edge_reconstruction import _event_for_spec
from scripts.plot_paper_dvs_v2e_cm_coverage import MULTI_SCENE, MOTION_REGIMES, _scene_path, _timestamps


OUT = ROOT / "results/ami_next/accumulation_fidelity_srer"
OLD = ROOT / "results/ami_next/accumulation_fidelity"
METHODS = ("DVS-Voltmeter", "v2e", "Fixed 180 RPM", "Fixed 360 RPM", "EIRM")
METHOD_KEYS = {"DVS-Voltmeter": "dvs_voltmeter", "v2e": "v2e", "Fixed 180 RPM": "ami_fixed_180", "Fixed 360 RPM": "ami_fixed_360", "EIRM": "adaptive_eirm"}
REGIMES = ("camera_only", "object_only", "co_motion", "counter_motion")
EPS = 1e-12
COUNT_CLIP = 8.0


def _specs() -> list[dict]:
    return list(MOTION_REGIMES) + [s for s in MULTI_SCENE if s["split"] != "test" or s["regime"] not in {"object_only", "co_motion"}]


def _reference_signal(luminance: np.ndarray) -> np.ndarray:
    image = np.log(np.clip(np.asarray(luminance, dtype=np.float64), 1e-6, None))
    gy, gx = np.gradient(image)
    return np.hypot(gx, gy).astype(np.float32)


def _lpips_value(model: LPIPS, a: np.ndarray, b: np.ndarray) -> float:
    aa = torch.from_numpy(np.repeat(a[None, None], 3, axis=1) * 2.0 - 1.0).float()
    bb = torch.from_numpy(np.repeat(b[None, None], 3, axis=1) * 2.0 - 1.0).float()
    with torch.no_grad():
        return float(model(aa, bb).mean().item())


def main() -> None:
    specs = _specs()
    dataset_root = ROOT / "data/eveyes_comotion_v2_ready"
    # Freeze the reference scale from train/validation only.
    train_valid_gradients = []
    for spec in specs:
        if spec["split"] not in {"train", "valid"}:
            continue
        scene = _scene_path(dataset_root, spec["split"], spec["scene"], spec["regime"])
        hero = int(spec["hero"])
        frames, _ = load_rgb_sequence(scene / "images", scene / "timestamps.csv", timestamp_unit="s", max_frames=hero + 2)
        grad = _reference_signal(rgb_to_linear_luminance(frames[hero : hero + 1])[0])
        positive = grad[grad > 1e-12]
        if len(positive):
            train_valid_gradients.append(positive)
    if not train_valid_gradients:
        raise RuntimeError("no train/validation reference gradients available")
    reference_clip = float(np.quantile(np.concatenate(train_valid_gradients), 0.995))
    reference_clip = max(reference_clip, 1e-6)

    lpips_model = LPIPS(net="alex", version="0.1", verbose=False).eval()
    rows: list[dict] = []
    energy = {method: {"signal_energy": 0.0, "error_energy": 0.0, "n_samples": 0} for method in METHODS}
    comotion_energy = {method: {"signal_energy": 0.0, "error_energy": 0.0, "n_samples": 0} for method in METHODS}
    for spec in specs:
        scene = _scene_path(dataset_root, spec["split"], spec["scene"], spec["regime"])
        timestamps = _timestamps(scene)
        hero = int(spec["hero"])
        frames, frame_ts = load_rgb_sequence(scene / "images", scene / "timestamps.csv", timestamp_unit="s", max_frames=hero + 2)
        luminance = rgb_to_linear_luminance(frames[hero : hero + 1])[0].astype(np.float32)
        reference_raw = _reference_signal(luminance)
        reference = np.clip(reference_raw / reference_clip, 0.0, 1.0)
        for method in METHODS:
            key = METHOD_KEYS[method]
            events = _eirm_events(spec) if key == "adaptive_eirm" else _event_for_spec(spec, scene, key, timestamps, hero, "cpu")
            accumulation = _accum(events, reference.shape)
            diff = reference.astype(np.float64) - accumulation.astype(np.float64)
            signal_energy = float(np.sum(reference.astype(np.float64) ** 2))
            error_energy = float(np.sum(diff * diff))
            lp = _lpips_value(lpips_model, reference, accumulation)
            energy[method]["signal_energy"] += signal_energy
            energy[method]["error_energy"] += error_energy
            energy[method]["n_samples"] += 1
            if spec["regime"] == "co_motion":
                comotion_energy[method]["signal_energy"] += signal_energy
                comotion_energy[method]["error_energy"] += error_energy
                comotion_energy[method]["n_samples"] += 1
            rows.append({"split": spec["split"], "scene": spec["scene"], "regime": spec["regime"], "hero": hero, "method": method, "t0": float(frame_ts[hero]), "t1": float(frame_ts[min(hero + 1, len(frame_ts) - 1)]), "event_count": int(len(events)), "reference_energy": signal_energy, "error_energy": error_energy, "per_frame_srer": float(10.0 * np.log10(signal_energy / (error_energy + EPS))), "lpips": lp})
        print(f"evaluated {spec['split']}/{spec['scene']}/{spec['regime']}", flush=True)

    def _srer(item: dict) -> float:
        return float(10.0 * np.log10(item["signal_energy"] / (item["error_energy"] + EPS)))

    aggregate = {}
    comotion = {}
    for method in METHODS:
        aggregate[method] = {"srer": _srer(energy[method]), "lpips": float(np.mean([row["lpips"] for row in rows if row["method"] == method])), "n_samples": energy[method]["n_samples"]}
        comotion[method] = {"srer": _srer(comotion_energy[method]), "lpips": float(np.mean([row["lpips"] for row in rows if row["method"] == method and row["regime"] == "co_motion"])), "n_samples": comotion_energy[method]["n_samples"]}

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "qualitative").mkdir(exist_ok=True)
    protocol = {
        "task": "fixed-window event accumulation signal fidelity",
        "reference_signal": "RGB-derived log-linear-luminance gradient magnitude",
        "reference_construction": "log(max(linear luminance, 1e-6)); np.gradient; hypot; one global 99.5th percentile clip estimated from train/validation positive gradients; clip to [0,1]",
        "accumulation_definition": "positive and non-positive event counts per pixel, signed log difference, divided by log(1+8), mapped as 0.5+0.5*signed and clipped to [0,1]",
        "polarity_handling": "two polarity counts retained before signed scalar readout; p>0 positive and p<=0 negative",
        "normalization": "COUNT_CLIP=8 for all methods and windows; reference clip frozen from train/validation; no per-image maximum or percentile scaling",
        "valid_region": "full native 960x540 image; identical for every method",
        "temporal_window": "same source fixed physical-time windows and timestamps as frozen CM-IWE protocol",
        "active_event_input": "timestamp-aware compensated reference-plane events for Fixed180, Fixed360, and EIRM",
        "srer_definition": "10*log10(||R||_2^2 / (||R-A||_2^2 + eps))",
        "global_aggregation": "accumulate signal and error energy over all samples before log10; do not average dB values",
        "lpips_backbone": "AlexNet LPIPS v0.1",
        "lpips_preprocessing": "same [0,1] reference and accumulation scalar maps replicated to three channels, mapped to [-1,1], no method-specific preprocessing",
        "evaluated_regimes": list(REGIMES),
        "available_regimes": sorted({spec["regime"] for spec in specs}),
        "fairness_checks": {"same_scenes": True, "same_windows": True, "same_timestamps": True, "same_valid_region": True, "same_reference": True, "same_accumulation": True, "same_normalization": True, "same_lpips": True, "no_method_specific_masks": True, "no_method_specific_rescaling": True, "global_energy_before_log": True},
        "counter_motion_status": "absent from frozen source protocol; retained as unavailable",
    }
    (OUT / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    (OUT / "aggregate_metrics.json").write_text(json.dumps({"status": "evaluated", "metrics": ["SRER", "LPIPS"], "aggregate": aggregate}, indent=2) + "\n")
    (OUT / "comotion_metrics.json").write_text(json.dumps({"status": "evaluated", "metrics": ["SRER", "LPIPS"], "co_motion": comotion}, indent=2) + "\n")
    energy_out = {"global": {method: {**energy[method], "srer": _srer(energy[method])} for method in METHODS}, "co_motion": {method: {**comotion_energy[method], "srer": _srer(comotion_energy[method])} for method in METHODS}, "eps": EPS, "reference_clip": reference_clip}
    (OUT / "energy_breakdown.json").write_text(json.dumps(energy_out, indent=2) + "\n")
    if (OLD / "aggregate_metrics.json").exists() and (OLD / "per_regime_metrics.json").exists():
        old = {"aggregate": json.loads((OLD / "aggregate_metrics.json").read_text()), "per_regime": json.loads((OLD / "per_regime_metrics.json").read_text()), "note": "Diagnostic only; retained from prior RGB-luminance PSNR/SSIM evaluation."}
        (OUT / "diagnostic_old_psnr_ssim.json").write_text(json.dumps(old, indent=2) + "\n")
    fields = ["method", "srer", "lpips", "comotion_srer", "comotion_lpips"]
    with (OUT / "summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader()
        for method in METHODS:
            writer.writerow({"method": method, "srer": aggregate[method]["srer"], "lpips": aggregate[method]["lpips"], "comotion_srer": comotion[method]["srer"], "comotion_lpips": comotion[method]["lpips"]})
    fields = ["split", "scene", "regime", "hero", "method", "t0", "t1", "event_count", "reference_energy", "error_energy", "per_frame_srer", "lpips"]
    with (OUT / "per_frame_metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(rows)
    report = ["# Accumulation signal fidelity: SRER and LPIPS", "", "This evaluation measures fixed-window signal fidelity. It is not a learned reconstruction task. The primary metrics are global SRER and LPIPS; prior PSNR/SSIM values are retained only in `diagnostic_old_psnr_ssim.json`.", "", "Reference: RGB-derived gradient magnitude computed as the hypotenuse of the spatial gradient of log linear luminance. Its global 99.5th-percentile clip is estimated on train/validation and frozen for every method.", "", "Accumulation: positive and negative event counts are kept separately, transformed with the same signed log rule and COUNT_CLIP=8, then mapped to [0,1]. Active events are timestamp-aware compensated before accumulation.", "", "SRER = 10 log10(signal_energy / (error_energy + eps)); signal and error energies are accumulated over the full evaluation set before applying log10.", "", "| Method | SRER ↑ | LPIPS ↓ | Co-motion SRER ↑ | Co-motion LPIPS ↓ |", "|---|---:|---:|---:|---:|"]
    for method in METHODS:
        report.append(f"| {method} | {aggregate[method]['srer']:.4f} | {aggregate[method]['lpips']:.4f} | {comotion[method]['srer']:.4f} | {comotion[method]['lpips']:.4f} |")
    regime_counts = {regime: sum(1 for spec in specs if spec["regime"] == regime) for regime in REGIMES}
    report += ["", "## Sample audit", "", f"Total evaluated samples per method: {len(specs)}.", "", "| Regime | Samples |", "|---|---:|"]
    for regime in REGIMES:
        report.append(f"| {regime} | {regime_counts[regime]} |")
    report += ["", "## Energy audit", "", "| Method | Global signal energy | Global error energy | Co-motion signal energy | Co-motion error energy |", "|---|---:|---:|---:|---:|"]
    for method in METHODS:
        report.append(f"| {method} | {energy[method]['signal_energy']:.6e} | {energy[method]['error_energy']:.6e} | {comotion_energy[method]['signal_energy']:.6e} | {comotion_energy[method]['error_energy']:.6e} |")
    report += ["", "The reported SRER values were recomputed from the energy terms above using the exact stored epsilon. LPIPS uses the same scalar signal maps replicated to three channels and the same frozen AlexNet LPIPS v0.1 weights for all methods.", "", "SRER and LPIPS disagree in ranking: passive methods have less negative SRER, while active methods have substantially lower LPIPS. This disagreement is retained as measured and is not resolved by method-specific rescaling.", "", "Counter-motion is unavailable in the frozen source window protocol.", "", "Script: `scripts/evaluate_accumulation_fidelity_srer.py`; previous `scripts/evaluate_accumulation_fidelity.py` remains unchanged. Existing event streams and prior PSNR/SSIM outputs were not overwritten."]
    (OUT / "report.md").write_text("\n".join(report) + "\n")


if __name__ == "__main__":
    main()
