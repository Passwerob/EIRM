#!/usr/bin/env python3
"""Evaluate fixed-window event accumulation against the matched RGB target."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
from lpips import LPIPS

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ami_simulator.pipeline import load_rgb_sequence, rgb_to_linear_luminance
from scripts.evaluate_cm_edge_reconstruction import _event_for_spec
from scripts.evaluate_dvs_v2e_ours import _read_npz
from scripts.plot_paper_dvs_v2e_cm_coverage import MULTI_SCENE, MOTION_REGIMES, _scene_path, _timestamps


OUT = ROOT / "results/ami_next/accumulation_fidelity"
METHODS = ("dvs_voltmeter", "v2e", "ami_fixed_180", "ami_fixed_360", "adaptive_eirm")
LABELS = {"dvs_voltmeter": "DVS-Voltmeter", "v2e": "v2e", "ami_fixed_180": "Fixed180", "ami_fixed_360": "Fixed360", "adaptive_eirm": "EIRM"}
REGIMES = ("camera_only", "object_only", "co_motion", "counter_motion")
COUNT_CLIP = 8.0
EIRM_EVENTS = ROOT / "results/ami_next/cm_iwe_protocol_matched/events"


def _eirm_events(spec: dict):
    path = EIRM_EVENTS / f"{spec['split']}_{spec['scene']}_{spec['regime']}_f{int(spec['hero'])}.npz"
    if not path.exists():
        raise FileNotFoundError(f"missing frozen EIRM event cache: {path}")
    return _read_npz(path)


def _ssim(a: np.ndarray, b: np.ndarray) -> float:
    # Global SSIM with fixed constants; no image-dependent normalization.
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    c1, c2 = 0.01**2, 0.03**2
    mux, muy = a.mean(), b.mean()
    vx, vy = a.var(), b.var()
    cov = ((a - mux) * (b - muy)).mean()
    return float(((2 * mux * muy + c1) * (2 * cov + c2)) / ((mux * mux + muy * muy + c1) * (vx + vy + c2)))


def _accum(events, shape: tuple[int, int]) -> np.ndarray:
    h, w = shape
    pos = np.zeros((h, w), dtype=np.float32)
    neg = np.zeros((h, w), dtype=np.float32)
    x = np.rint(np.asarray(events.x)).astype(np.int64)
    y = np.rint(np.asarray(events.y)).astype(np.int64)
    p = np.asarray(events.p)
    valid = (x >= 0) & (x < w) & (y >= 0) & (y < h)
    np.add.at(pos, (y[valid & (p > 0)], x[valid & (p > 0)]), 1.0)
    np.add.at(neg, (y[valid & (p <= 0)], x[valid & (p <= 0)]), 1.0)
    # One fixed log compression shared by every method and window.
    signed = (np.log1p(pos) - np.log1p(neg)) / np.log1p(COUNT_CLIP)
    return np.clip(0.5 + 0.5 * signed, 0.0, 1.0).astype(np.float32)


def _metrics(pred: np.ndarray, target: np.ndarray, lpips_model: LPIPS) -> dict[str, float]:
    diff = pred.astype(np.float64) - target.astype(np.float64)
    mse = float(np.mean(diff * diff))
    psnr = float(10.0 * np.log10(1.0 / max(mse, 1e-12)))
    ssim = _ssim(pred, target)
    p = torch.from_numpy(np.repeat(pred[None, None], 3, axis=1) * 2.0 - 1.0).float()
    t = torch.from_numpy(np.repeat(target[None, None], 3, axis=1) * 2.0 - 1.0).float()
    with torch.no_grad():
        lp = float(lpips_model(p, t).mean().item())
    return {"psnr": psnr, "ssim": ssim, "lpips": lp}


def main() -> None:
    # Keep the same eight source windows used by the frozen CM-IWE package.
    specs = list(MOTION_REGIMES) + [s for s in MULTI_SCENE if s["split"] != "test" or s["regime"] not in {"object_only", "co_motion"}]
    lpips_model = LPIPS(net="alex", version="0.1", verbose=False).eval()
    rows = []
    for spec in specs:
        scene = _scene_path(ROOT / "data/eveyes_comotion_v2_ready", spec["split"], spec["scene"], spec["regime"])
        timestamps = _timestamps(scene)
        hero = int(spec["hero"])
        frames, frame_ts = load_rgb_sequence(scene / "images", scene / "timestamps.csv", timestamp_unit="s", max_frames=hero + 2)
        target = np.asarray(rgb_to_linear_luminance(frames[hero : hero + 1])[0], dtype=np.float32)
        for method in METHODS:
            events = _eirm_events(spec) if method == "adaptive_eirm" else _event_for_spec(spec, scene, method, timestamps, hero, "cpu")
            pred = _accum(events, target.shape)
            metric = _metrics(pred, target, lpips_model)
            rows.append({"split": spec["split"], "scene": spec["scene"], "regime": spec["regime"], "hero": hero, "method": LABELS[method], "t0": float(frame_ts[hero]), "t1": float(frame_ts[min(hero + 1, len(frame_ts) - 1)]), "event_count": int(len(events)), **metric})
        print(f"evaluated {spec['split']}/{spec['scene']}/{spec['regime']}", flush=True)

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "qualitative").mkdir(exist_ok=True)
    methods = [LABELS[m] for m in METHODS]
    aggregate = {}
    for method in methods:
        selected = [r for r in rows if r["method"] == method and r["regime"] in REGIMES]
        aggregate[method] = {k: float(np.mean([r[k] for r in selected])) for k in ("psnr", "ssim", "lpips", "event_count")}
    per_regime = {regime: {method: ({k: float(np.mean([r[k] for r in selected])) for k in ("psnr", "ssim", "lpips", "event_count")} if (selected := [r for r in rows if r["method"] == method and r["regime"] == regime]) else {"status": "absent_from_source_protocol"}) for method in methods} for regime in REGIMES}
    protocol = {"task": "fixed-window event accumulation fidelity diagnostic", "event_representation": "two-polarity count accumulation collapsed only after shared signed log compression", "count_clip": COUNT_CLIP, "normalization": "one global fixed clip; no per-image maximum normalization", "rgb_target": "linear luminance from the same target RGB timestamp, replicated to three channels", "metrics": ["PSNR", "SSIM", "LPIPS-Alex-v0.1"], "metric_implementation": {"psnr": "10*log10(1/MSE) on [0,1] luminance", "ssim": "global SSIM with C1=0.01^2 and C2=0.03^2", "lpips": "lpips package Alex v0.1, eval mode, frozen local weights"}, "lpips_weights": "local lpips package weights/v0.1/alex.pth", "windows": "same source windows as frozen CM-IWE package", "active_compensation": "same timestamp-aware reference-plane events as CM-IWE", "aggregation": "arithmetic mean over windows; counter_motion unavailable in source protocol", "limitation": "diagnostic of fixed accumulation fidelity, not a learned independent downstream model"}
    (OUT / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    (OUT / "aggregate_metrics.json").write_text(json.dumps({"status": "evaluated", "aggregate": aggregate}, indent=2) + "\n")
    (OUT / "per_regime_metrics.json").write_text(json.dumps(per_regime, indent=2) + "\n")
    fields = ["split", "scene", "regime", "hero", "method", "t0", "t1", "event_count", "psnr", "ssim", "lpips"]
    for name in ("per_window_metrics.csv", "per_frame_metrics.csv"):
        with (OUT / name).open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader(); writer.writerows(rows)
    report = ["# Fixed-window event accumulation fidelity", "", "This is a supplementary diagnostic paired with the CM-IWE geometric downstream result. Every method uses the same physical windows, target timestamps, signed polarity accumulation, fixed global logarithmic clip, and metric implementations. No method-specific intensity scaling or per-image maximum normalization is used.", "", "| Method | PSNR ↑ | SSIM ↑ | LPIPS ↓ | Mean events |", "|---|---:|---:|---:|---:|"]
    for method, m in aggregate.items(): report.append(f"| {method} | {m['psnr']:.4f} | {m['ssim']:.4f} | {m['lpips']:.4f} | {m['event_count']:.0f} |")
    report += ["", "## Per-regime results", "", "| Regime | DVS PSNR | v2e PSNR | Fixed180 PSNR | Fixed360 PSNR | EIRM PSNR | EIRM SSIM | EIRM LPIPS |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for regime in REGIMES:
        vals = per_regime[regime]
        if vals[methods[0]].get("status"):
            report.append(f"| {regime} | n/a | n/a | n/a | n/a | n/a | n/a | n/a |")
        else:
            report.append("| " + regime + " | " + " | ".join([f"{vals[m]['psnr']:.4f}" for m in methods[:4]] + [f"{vals['EIRM']['psnr']:.4f}", f"{vals['EIRM']['ssim']:.4f}", f"{vals['EIRM']['lpips']:.4f}"]) + " |")
    report += ["", "The values measure similarity between a fixed event accumulation representation and RGB luminance. They should be interpreted as representation fidelity, not as evidence that a learned RGB reconstruction model transfers better.", "", "Counter-motion is absent from the frozen source window protocol and is reported as unavailable in the per-regime file."]
    (OUT / "report.md").write_text("\n".join(report) + "\n")


if __name__ == "__main__":
    main()
