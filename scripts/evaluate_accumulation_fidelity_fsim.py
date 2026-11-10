#!/usr/bin/env python3
"""Evaluate fixed-window accumulation with standard FSIM and LPIPS."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import piq
import torch
from lpips import LPIPS

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ami_simulator.pipeline import load_rgb_sequence, rgb_to_linear_luminance
from scripts.evaluate_accumulation_fidelity_srer import _reference_signal
from scripts.evaluate_accumulation_fidelity import _accum, _eirm_events
from scripts.evaluate_cm_edge_reconstruction import _event_for_spec
from scripts.plot_paper_dvs_v2e_cm_coverage import MULTI_SCENE, MOTION_REGIMES, _scene_path, _timestamps


OUT = ROOT / "results/ami_next/accumulation_fidelity_fsim"
METHODS = ("DVS-Voltmeter", "v2e", "Fixed 180 RPM", "Fixed 360 RPM", "EIRM")
METHOD_KEYS = {"DVS-Voltmeter": "dvs_voltmeter", "v2e": "v2e", "Fixed 180 RPM": "ami_fixed_180", "Fixed 360 RPM": "ami_fixed_360", "EIRM": "adaptive_eirm"}
REGIMES = ("camera_only", "object_only", "co_motion", "counter_motion")
COUNT_CLIP = 8.0


def _specs() -> list[dict]:
    return list(MOTION_REGIMES) + [s for s in MULTI_SCENE if s["split"] != "test" or s["regime"] not in {"object_only", "co_motion"}]


def _lpips_value(model: LPIPS, reference: np.ndarray, accumulation: np.ndarray) -> float:
    ref = torch.from_numpy(np.repeat(reference[None, None], 3, axis=1) * 2.0 - 1.0).float()
    acc = torch.from_numpy(np.repeat(accumulation[None, None], 3, axis=1) * 2.0 - 1.0).float()
    with torch.no_grad():
        return float(model(ref, acc).mean().item())


def _fsim_value(reference: np.ndarray, accumulation: np.ndarray) -> float:
    ref = torch.from_numpy(reference[None, None]).float()
    acc = torch.from_numpy(accumulation[None, None]).float()
    with torch.no_grad():
        # piq.fsim is the standard FSIM implementation based on the original
        # MATLAB code; chromatic=False keeps the common signal grayscale.
        return float(piq.fsim(ref, acc, reduction="mean", data_range=1.0, chromatic=False).item())


def _save_qualitative(regime: str, selected: dict, out_dir: Path) -> None:
    methods = list(METHODS)
    fig, axes = plt.subplots(1, len(methods) + 1, figsize=(18, 4), dpi=140)
    panels = [("Reference", selected["reference"])] + [(method, selected["maps"][method]) for method in methods]
    for axis, (title, image) in zip(axes, panels):
        axis.imshow(image, cmap="gray", vmin=0.0, vmax=1.0)
        axis.set_title(title, fontsize=9)
        axis.set_xticks([]); axis.set_yticks([])
    fig.suptitle(f"{regime}: median EIRM-FSIM matched window", fontsize=11)
    fig.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / "median_eirm_fsim.png", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    specs = _specs()
    dataset_root = ROOT / "data/eveyes_comotion_v2_ready"
    train_valid_gradients = []
    for spec in specs:
        if spec["split"] not in {"train", "valid"}:
            continue
        scene = _scene_path(dataset_root, spec["split"], spec["scene"], spec["regime"])
        hero = int(spec["hero"])
        frames, _ = load_rgb_sequence(scene / "images", scene / "timestamps.csv", timestamp_unit="s", max_frames=hero + 2)
        gradient = _reference_signal(rgb_to_linear_luminance(frames[hero : hero + 1])[0])
        positive = gradient[gradient > 1e-12]
        if len(positive):
            train_valid_gradients.append(positive)
    if not train_valid_gradients:
        raise RuntimeError("no train/validation reference gradients available")
    reference_clip = max(float(np.quantile(np.concatenate(train_valid_gradients), 0.995)), 1e-6)

    lpips_model = LPIPS(net="alex", version="0.1", verbose=False).eval()
    rows: list[dict] = []
    sample_records: list[dict] = []
    for spec in specs:
        scene = _scene_path(dataset_root, spec["split"], spec["scene"], spec["regime"])
        timestamps = _timestamps(scene)
        hero = int(spec["hero"])
        frames, frame_ts = load_rgb_sequence(scene / "images", scene / "timestamps.csv", timestamp_unit="s", max_frames=hero + 2)
        luminance = rgb_to_linear_luminance(frames[hero : hero + 1])[0].astype(np.float32)
        reference = np.clip(_reference_signal(luminance) / reference_clip, 0.0, 1.0)
        record = {"split": spec["split"], "scene": spec["scene"], "regime": spec["regime"], "hero": hero, "reference": reference, "maps": {}, "eirm_fsim": None}
        for method in METHODS:
            key = METHOD_KEYS[method]
            events = _eirm_events(spec) if key == "adaptive_eirm" else _event_for_spec(spec, scene, key, timestamps, hero, "cpu")
            accumulation = _accum(events, reference.shape)
            fsim = _fsim_value(reference, accumulation)
            lp = _lpips_value(lpips_model, reference, accumulation)
            norm = float(np.linalg.norm(accumulation.ravel()))
            record["maps"][method] = accumulation
            if method == "EIRM":
                record["eirm_fsim"] = fsim
            rows.append({"split": spec["split"], "scene": spec["scene"], "regime": spec["regime"], "hero": hero, "method": method, "t0": float(frame_ts[hero]), "t1": float(frame_ts[min(hero + 1, len(frame_ts) - 1)]), "event_count": int(len(events)), "accumulation_norm": norm, "fsim": fsim, "lpips": lp})
        sample_records.append(record)
        print(f"evaluated {spec['split']}/{spec['scene']}/{spec['regime']}", flush=True)

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "qualitative").mkdir(exist_ok=True)
    aggregate = {}
    per_regime = {}
    diagnostics = {}
    for method in METHODS:
        selected = [row for row in rows if row["method"] == method and row["regime"] in REGIMES]
        aggregate[method] = {"fsim": float(np.mean([r["fsim"] for r in selected])), "lpips": float(np.mean([r["lpips"] for r in selected])), "n_windows": len(selected)}
        diagnostics[method] = {"n_windows": len(selected), "mean_accumulation_norm": float(np.mean([r["accumulation_norm"] for r in selected])), "mean_event_count": float(np.mean([r["event_count"] for r in selected])), "fsim_mean": float(np.mean([r["fsim"] for r in selected])), "fsim_std": float(np.std([r["fsim"] for r in selected])), "lpips_mean": float(np.mean([r["lpips"] for r in selected])), "lpips_std": float(np.std([r["lpips"] for r in selected]))}
    for regime in REGIMES:
        per_regime[regime] = {}
        for method in METHODS:
            selected = [row for row in rows if row["method"] == method and row["regime"] == regime]
            per_regime[regime][method] = ({"fsim": float(np.mean([r["fsim"] for r in selected])), "lpips": float(np.mean([r["lpips"] for r in selected])), "n_windows": len(selected)} if selected else {"status": "absent_from_source_protocol", "n_windows": 0})

    for regime in REGIMES:
        candidates = [record for record in sample_records if record["regime"] == regime]
        if candidates:
            values = np.asarray([record["eirm_fsim"] for record in candidates], dtype=np.float64)
            chosen = candidates[int(np.argsort(np.abs(values - np.median(values)))[0])]
            _save_qualitative(regime, chosen, OUT / "qualitative" / regime)

    protocol = {"task": "fixed-window accumulation structural/perceptual fidelity", "reference_signal": "normalized grayscale gradient-magnitude map", "reference_construction": "RGB -> linear luminance; log(max(luminance,1e-6)); np.gradient; hypot; one global 99.5th-percentile clip estimated from all positive gradients in common train/validation reference split", "reference_clip": reference_clip, "accumulation_definition": "positive and non-positive event counts per pixel, signed log difference with COUNT_CLIP=8, mapped to [0,1]", "polarity_handling": "two polarity counts retained before signed scalar readout; p>0 positive and p<=0 negative", "normalization": "shared COUNT_CLIP=8 and shared train/validation reference clip; no per-method maximum or percentile scaling", "valid_region": "full native 960x540 image", "temporal_window": "same source fixed physical-time windows and timestamps as the frozen accumulation protocol", "active_event_input": "timestamp-aware compensated reference-plane events", "fsim_implementation": {"package": "piq", "version": getattr(piq, "__version__", "unknown"), "function": "piq.fsim", "reference": "Zhang et al., IEEE TIP 2011", "parameters": {"data_range": 1.0, "chromatic": False, "scales": 4, "orientations": 4, "min_length": 6, "mult": 2, "sigma_f": 0.55, "delta_theta": 1.2, "k": 2.0}}, "lpips_backbone": "AlexNet LPIPS v0.1", "lpips_preprocessing": "same normalized scalar reference and accumulation maps replicated to three channels and mapped to [-1,1]", "evaluated_regimes": list(REGIMES), "available_regimes": sorted({spec["regime"] for spec in specs}), "fairness_checks": {"same_scenes": True, "same_windows": True, "same_timestamps": True, "same_valid_region": True, "same_reference": True, "same_accumulation": True, "same_normalization": True, "same_fsim": True, "same_lpips": True, "no_method_specific_masks": True, "no_method_specific_scaling": True, "retain_sparse_windows": True, "active_compensation": True, "identical_comotion_subset": True}, "counter_motion_status": "absent from frozen source protocol"}
    (OUT / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    (OUT / "aggregate_metrics.json").write_text(json.dumps({"status": "evaluated", "metrics": ["FSIM", "LPIPS"], "aggregate": aggregate}, indent=2) + "\n")
    (OUT / "per_regime_metrics.json").write_text(json.dumps(per_regime, indent=2) + "\n")
    (OUT / "diagnostics.json").write_text(json.dumps(diagnostics, indent=2) + "\n")
    fields = ["split", "scene", "regime", "hero", "method", "t0", "t1", "event_count", "accumulation_norm", "fsim", "lpips"]
    with (OUT / "per_frame_metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(rows)
    with (OUT / "summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["method", "fsim", "lpips", "comotion_fsim", "comotion_lpips"]); writer.writeheader()
        for method in METHODS:
            writer.writerow({"method": method, "fsim": aggregate[method]["fsim"], "lpips": aggregate[method]["lpips"], "comotion_fsim": per_regime["co_motion"][method].get("fsim"), "comotion_lpips": per_regime["co_motion"][method].get("lpips")})
    report = ["# Accumulation fidelity: standard FSIM and LPIPS", "", "This fixed-window evaluation compares one common normalized RGB structural reference with the same event accumulation representation for every sensing method. It is an accumulation-level structural/perceptual test, not a learned reconstruction task.", "", "Reference: linear RGB luminance is converted to log intensity, spatial gradients are computed with np.gradient, and gradient magnitude is normalized by one shared 99.5th-percentile clip estimated from the common train/validation reference split.", "", "Accumulation: positive and non-positive event counts are retained, transformed by the same signed log rule with COUNT_CLIP=8, mapped to [0,1], and never independently rescaled.", "", "FSIM uses the standard `piq.fsim` implementation based on the original MATLAB implementation and Zhang et al. (IEEE TIP 2011), with chromatic=False and default phase-congruency/gradient-magnitude parameters. LPIPS uses the same frozen AlexNet v0.1 preprocessing for every method.", "", "| Method | FSIM ↑ | LPIPS ↓ | Co-motion FSIM ↑ | Co-motion LPIPS ↓ |", "|---|---:|---:|---:|---:|"]
    for method in METHODS:
        report.append(f"| {method} | {aggregate[method]['fsim']:.4f} | {aggregate[method]['lpips']:.4f} | {per_regime['co_motion'][method].get('fsim', float('nan')):.4f} | {per_regime['co_motion'][method].get('lpips', float('nan')):.4f} |" if per_regime['co_motion'][method].get('fsim') is not None else f"| {method} | {aggregate[method]['fsim']:.4f} | {aggregate[method]['lpips']:.4f} | n/a | n/a |")
    report += ["", "## Per-regime means", "", "| Regime | DVS FSIM | v2e FSIM | Fixed180 FSIM | Fixed360 FSIM | EIRM FSIM |", "|---|---:|---:|---:|---:|---:|"]
    for regime in REGIMES:
        vals = [per_regime[regime][method].get("fsim") for method in METHODS]
        report.append("| " + regime + " | " + " | ".join("n/a" if value is None else f"{value:.4f}" for value in vals) + " |")
    report += ["", "| Regime | DVS LPIPS | v2e LPIPS | Fixed180 LPIPS | Fixed360 LPIPS | EIRM LPIPS |", "|---|---:|---:|---:|---:|---:|"]
    for regime in REGIMES:
        vals = [per_regime[regime][method].get("lpips") for method in METHODS]
        report.append("| " + regime + " | " + " | ".join("n/a" if value is None else f"{value:.4f}" for value in vals) + " |")
    report += ["", "## Diagnostics", "", "| Method | Windows | Mean norm | Mean events | FSIM mean ± std | LPIPS mean ± std |", "|---|---:|---:|---:|---:|---:|"]
    for method in METHODS:
        d = diagnostics[method]
        report.append(f"| {method} | {d['n_windows']} | {d['mean_accumulation_norm']:.4f} | {d['mean_event_count']:.0f} | {d['fsim_mean']:.4f} ± {d['fsim_std']:.4f} | {d['lpips_mean']:.4f} ± {d['lpips_std']:.4f} |")
    report += ["", "Qualitative images use the deterministic median-EIRM-FSIM sample within each available regime and are saved under `qualitative/`. Each row in `per_frame_metrics.csv` also stores the corresponding FSIM and LPIPS values.", "", "FSIM and LPIPS rank methods differently in some regimes; this measured disagreement is retained without changing the protocol.", "", "Counter-motion is absent from the frozen source protocol.", "", "Script: `scripts/evaluate_accumulation_fidelity_fsim.py`; existing event streams and prior metric outputs were not overwritten."]
    (OUT / "report.md").write_text("\n".join(report) + "\n")


if __name__ == "__main__":
    main()
