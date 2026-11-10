#!/usr/bin/env python3
"""Per-segment official-protocol AMI evaluation for a constructed EVEYES regime.

Runs Fixed {0,180,360} RPM over the full constructed timeline with the paper
primitive (_ami_windows + _score), then aggregates per playback segment read
from motion_state.csv. Verifies that a construction exposes a segment where
360 beats 180 (image-wide edge F1) AND a segment where 0 beats 180 (object-edge
F1), the two-sided regime needed for adaptive gear selection.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

from ami_simulator.geometry import RotatingWedgePrism
from ami_simulator.pipeline import load_rgb_sequence, rgb_to_linear_luminance
from scripts.evaluate_dl3dv_downstream import _ami_windows, _dvs_windows
from scripts.evaluate_eveyes_v2_downstream import _intrinsics, _load_object_masks, _score

GEARS = ((0.0, "ami_passive"), (180.0, "ami_fixed_180"), (360.0, "ami_fixed_360"))


def _segments(scene: Path, n_frames: int):
    """Group frames into contiguous runs of identical (playback_rate, sensor_fps)."""
    rows = list(csv.DictReader((scene / "motion_state.csv").open()))
    rate = np.asarray([float(r["playback_rate"]) for r in rows])
    fps = np.asarray([float(r["sensor_fps"]) for r in rows])
    keys = [f"rate{rate[i]:g}_fps{fps[i]:g}" for i in range(len(rows))]
    out = []
    run_index = -1
    previous_key = None
    for i in range(n_frames):
        key = keys[i] if i < len(keys) else keys[-1]
        if key != previous_key:
            run_index += 1
            previous_key = key
        out.append(f"s{run_index}")
    return np.asarray(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default="test/Scene_005/crawl_then_dash")
    ap.add_argument("--dataset-root", type=Path, default=Path("data/eveyes_comotion_v2_ready"))
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--threads", type=int, default=32)
    ap.add_argument("--output", type=Path, default=None)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    split, name, regime = args.scene.split("/")
    scene = args.dataset_root / split / name / regime
    frames, timestamps = load_rgb_sequence(scene / "images", scene / "timestamps.csv", timestamp_unit="s")
    lum = rgb_to_linear_luminance(frames)
    intr = _intrinsics(scene)
    prism = RotatingWedgePrism(intr, 0.5)
    masks = _load_object_masks(scene, len(lum), intr.width, intr.height)
    seg_labels = _segments(scene, len(frames))
    n_windows = len(timestamps) - 1
    win_seg = seg_labels[:n_windows]
    valid = np.ones((n_windows, intr.height, intr.width), dtype=bool)
    out = {"scene": args.scene, "frames": len(frames), "windows": n_windows,
           "resolution": [intr.width, intr.height], "per_gear": {}}
    for rpm, method in GEARS:
        if rpm == 0.0:
            ref_events, rpms = _dvs_windows(lum, timestamps)
        else:
            ref_events, _, rpms = _ami_windows(lum, timestamps, valid, prism, rpm, device=args.device)
        rows = _score(ref_events, lum[:-1], valid, timestamps, method, args.scene, rpms, masks[:-1] if masks is not None else None)
        for r in rows:
            r["segment"] = str(win_seg[int(r["window"])])
        out["per_gear"][method] = rows
        print(f"  done rpm={rpm}", flush=True)
    # per-segment aggregates
    seg_names = sorted(set(win_seg.tolist()))
    agg = {}
    for seg in seg_names:
        idx = np.where(win_seg == seg)[0]
        entry = {"windows": int(len(idx))}
        for rpm, method in GEARS:
            rows = out["per_gear"][method]
            sel = [rows[i] for i in idx]
            entry[method] = {
                "edge_f1": float(np.mean([r["edge_f1"] for r in sel])),
                "edge_precision": float(np.mean([r["edge_precision"] for r in sel])),
                "edge_recall": float(np.mean([r["edge_recall"] for r in sel])),
                "event_count": float(np.mean([r["event_count"] for r in sel])),
            }
            if masks is not None:
                selm = [r for r in sel if r.get("object_area", 0) >= 0.05]
                if selm:
                    entry[method]["object_edge_f1"] = float(np.mean([r["object_edge_f1"] for r in selm]))
                    entry[method]["object_edge_recall"] = float(np.mean([r["object_edge_recall"] for r in selm]))
                    entry[method]["teo"] = float(np.mean([r["teo"] for r in selm]))
                    entry[method]["target_purity"] = float(np.mean([r["target_purity"] for r in selm]))
                    entry[method]["object_windows"] = len(selm)
        agg[seg] = entry
    out["per_segment"] = agg
    # verdict: per segment, does 360 beat 180 (edge F1) and does 0 beat 180 (object F1)?
    verdict = {}
    best360, best0 = None, None
    for seg, entry in agg.items():
        f1_0 = entry["ami_passive"].get("edge_f1", 0.0)
        f1_180 = entry["ami_fixed_180"].get("edge_f1", 0.0)
        f1_360 = entry["ami_fixed_360"].get("edge_f1", 0.0)
        o0 = entry["ami_passive"].get("object_edge_f1")
        o180 = entry["ami_fixed_180"].get("object_edge_f1")
        e = {
            "d360_edge_f1": f1_360 - f1_180,
            "d0_edge_f1": f1_0 - f1_180,
        }
        if o0 is not None and o180 is not None:
            e["d0_object_edge_f1"] = o0 - o180
        verdict[seg] = e
        if e["d360_edge_f1"] > 0 and (best360 is None or e["d360_edge_f1"] > best360[1]):
            best360 = (seg, e["d360_edge_f1"])
        if e.get("d0_object_edge_f1", -1) > 0 and (best0 is None or e["d0_object_edge_f1"] > best0[1]):
            best0 = (seg, e["d0_object_edge_f1"])
    out["verdict"] = verdict
    out["best_360_segment"] = best360
    out["best_0_segment"] = best0
    out["two_sided_ok"] = best360 is not None and best0 is not None
    op = args.output or Path(f"results/ami_next/constructed_eval_{regime}.json")
    op.parent.mkdir(parents=True, exist_ok=True)
    op.write_text(json.dumps(out, indent=2))
    print(json.dumps({"per_segment": agg, "verdict": verdict, "two_sided_ok": out["two_sided_ok"]}, indent=2))


if __name__ == "__main__":
    raise SystemExit(main())
