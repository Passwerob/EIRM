#!/usr/bin/env python3
"""Probe per-regime AMI RPM behavior on native 960x540 EVEYES.

Simulates {0,180,360} RPM across the full native timeline of each regime and
reports image-wide edge F1 and (when masks exist) object-target metrics. Used
to justify the fused gear-excels construction. Read-only; writes one JSON.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np
import torch

from ami_simulator.event_quality import compute_reference_edge_mask, event_quality_metrics, target_observability_metrics
from ami_simulator.geometry import RotatingWedgePrism
from ami_simulator.pipeline import load_rgb_sequence, rgb_to_linear_luminance
from ami_simulator.types import CameraIntrinsics
from scripts.evaluate_dl3dv_downstream import _ami_windows, _dvs_windows, _visual_motion


def _intrinsics(scene: Path) -> CameraIntrinsics:
    calib = json.loads((scene / "camera_intrinsics.json").read_text())
    return CameraIntrinsics(int(calib["width"]), int(calib["height"]), float(calib["fx"]), float(calib["fy"]), float(calib["cx"]), float(calib["cy"]))


def _masks(scene: Path, n: int, w: int, h: int):
    paths = sorted((scene / "object_masks").glob("*.png"))[:n] if (scene / "object_masks").is_dir() else []
    if len(paths) != n:
        return None
    out = []
    for p in paths:
        img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        if img.shape != (h, w):
            img = cv2.resize(img, (w, h), interpolation=cv2.INTER_NEAREST)
        out.append(img > 127)
    return np.stack(out)


def _full_masks(w: int, h: int, n: int):
    return np.ones((n, h, w), dtype=bool)


def _score(events_per_window, luminance, valid, timestamps, object_masks):
    rows = []
    prev = None
    for i, events in enumerate(events_per_window):
        edge = compute_reference_edge_mask(luminance[i], edge_quantile=0.75) & valid[i]
        q = event_quality_metrics(events, reference_edge_mask=edge, previous_active_pixels=prev, image_shape=luminance[i].shape)
        active = np.unique(events.y.astype(np.int64) * luminance[i].shape[1] + events.x.astype(np.int64)) if len(events) else np.empty(0, np.int64)
        prev = active
        row = {"window": i, "edge_f1": float(q["edge_f1"]), "edge_precision": float(q["edge_precision"]),
               "edge_recall": float(q["edge_coverage"]), "event_count": int(len(events))}
        if object_masks is not None:
            target = np.asarray(object_masks[i], dtype=bool)
            oedge = edge & target
            oq = event_quality_metrics(events, reference_edge_mask=oedge, image_shape=luminance[i].shape)
            teo = target_observability_metrics(events, target_mask=target, target_edge_mask=oedge,
                                               image_shape=luminance[i].shape,
                                               time_range=(float(timestamps[i]), float(timestamps[i + 1])))
            row.update({"object_area": float(target.mean()), "object_edge_f1": float(oq["edge_f1"]),
                        "object_edge_precision": float(oq["edge_precision"]), "object_edge_recall": float(oq["edge_coverage"]),
                        "teo": float(teo["teo"]), "target_coverage": float(teo["target_coverage"]),
                        "target_event_count": int(teo["target_event_count"]),
                        "background_event_count": int(teo["background_event_count"])})
        rows.append(row)
    return rows


def _summary(rows, object_masks):
    out = {"edge_f1": float(np.mean([r["edge_f1"] for r in rows])),
           "edge_precision": float(np.mean([r["edge_precision"] for r in rows])),
           "edge_recall": float(np.mean([r["edge_recall"] for r in rows])),
           "event_count": float(np.mean([r["event_count"] for r in rows]))}
    if object_masks is not None:
        sel = [r for r in rows if r["object_area"] >= 0.05]
        for k in ("object_edge_f1", "object_edge_precision", "object_edge_recall", "teo", "target_coverage"):
            out[k] = float(np.mean([r[k] for r in sel])) if sel else 0.0
        out["object_windows"] = len(sel)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-root", type=Path, default=Path("data/eveyes_comotion_v2_ready"))
    ap.add_argument("--split", default="test")
    ap.add_argument("--scene", default="Scene_005")
    ap.add_argument("--regimes", nargs="*", default=["co_motion", "object_only", "camera_only", "drive_stop_drive"])
    ap.add_argument("--rpms", type=float, nargs="*", default=[0.0, 180.0, 360.0])
    ap.add_argument("--max-frames", type=int, default=480)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--output", type=Path, default=Path("results/ami_next/probe_segment_rpm.json"))
    args = ap.parse_args()
    torch.set_num_threads(8)
    report = {"resolution": [960, 540], "rpm_sweep": args.rpms, "regimes": {}}
    for regime in args.regimes:
        scene = args.dataset_root / args.split / args.scene / regime
        if not (scene / "images").is_dir():
            continue
        frames, timestamps = load_rgb_sequence(scene / "images", scene / "timestamps.csv", timestamp_unit="s", max_frames=args.max_frames)
        timestamps = timestamps[: len(frames)]
        lum = rgb_to_linear_luminance(frames)
        intr = _intrinsics(scene)
        prism = RotatingWedgePrism(intr, 0.5)
        omasks = _masks(scene, len(frames), intr.width, intr.height)
        full = _full_masks(intr.width, intr.height, len(frames) - 1)
        visual = _visual_motion(lum, timestamps)
        regime_out = {"n_frames": len(frames), "mean_visual_motion": float(np.mean(visual)), "rpm": {}}
        for rpm in args.rpms:
            if rpm == 0.0:
                ref_events, _ = _dvs_windows(lum, timestamps)
            else:
                ref_events, _, _ = _ami_windows(lum, timestamps, full, prism, rpm, device=args.device)
            rows = _score(ref_events, lum[:-1], full, timestamps, omasks[:-1] if omasks is not None else None)
            regime_out["rpm"][str(int(rpm))] = _summary(rows, omasks)
        report["regimes"][regime] = regime_out
        print(f"[probe] {regime} done  visual={regime_out['mean_visual_motion']:.3f}", flush=True)
        for rpm, s in regime_out["rpm"].items():
            print(f"    rpm={rpm:>4} edgeF1={s['edge_f1']:.4f} objF1={s.get('object_edge_f1', float('nan')):.4f} teo={s.get('teo', float('nan')):.4f} ev={s['event_count']:.0f}", flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True))
    print(f"wrote {args.output}")


if __name__ == "__main__":
    raise SystemExit(main())
