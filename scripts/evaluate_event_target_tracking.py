#!/usr/bin/env python3
"""Event-only target proposal and short-horizon tracking benchmark.

The detector never reads target masks. Masks are used only after proposal
generation to score IoU, recall, center error, and temporal continuity.
"""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

import cv2
import numpy as np
import torch

from ami_simulator.geometry import RotatingWedgePrism
from ami_simulator.pipeline import load_rgb_sequence, rgb_to_linear_luminance
from ami_simulator.types import EventStream
from scripts.evaluate_dl3dv_downstream import _ami_windows, _ours_windows, _visual_motion
from scripts.evaluate_eveyes_v2_downstream import _imu_fraction, _intrinsics, _load_object_masks, _scene_path
from scripts.evaluate_dvs_v2e_ours import _external, _read_npz, _slice
from scripts.train_inverse_motion_policy import FEATURE_DIM
from ami_simulator.motion_net import TinyCausalGearPolicy


REGIMES = ("camera_only", "object_only", "co_motion")
METHODS = ("dvs_voltmeter", "v2e", "ami_passive", "ami_fixed_180", "ami_fixed_360", "ours")


def _proposal(events: EventStream, shape: tuple[int, int], threshold: int = 1) -> tuple[int, int, int, int] | None:
    height, width = shape
    if not len(events):
        return None
    x = np.asarray(events.x, dtype=np.int64)
    y = np.asarray(events.y, dtype=np.int64)
    valid = (x >= 0) & (x < width) & (y >= 0) & (y < height)
    counts = np.zeros((height, width), dtype=np.uint16)
    np.add.at(counts, (y[valid], x[valid]), 1)
    active = (counts >= threshold).astype(np.uint8)
    active = cv2.morphologyEx(active, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    num, labels, stats, _ = cv2.connectedComponentsWithStats(active, connectivity=8)
    candidates = []
    for index in range(1, num):
        x0, y0, w, h, area = stats[index]
        if area < 20:
            continue
        mass = float(counts[labels == index].sum())
        # Compact, event-rich components are the deterministic proposal score.
        score = mass / max(float(area) ** 0.35, 1.0)
        candidates.append((score, int(x0), int(y0), int(w), int(h)))
    if not candidates:
        ys, xs = np.where(active > 0)
        if len(xs) == 0:
            return None
        return int(xs.min()), int(ys.min()), int(xs.max() - xs.min() + 1), int(ys.max() - ys.min() + 1)
    _, x0, y0, w, h = max(candidates, key=lambda item: item[0])
    return x0, y0, w, h


def _box(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max() - xs.min() + 1), int(ys.max() - ys.min() + 1)


def _iou(a, b) -> float:
    if a is None or b is None:
        return 0.0
    ax, ay, aw, ah = a; bx, by, bw, bh = b
    x0, y0 = max(ax, bx), max(ay, by)
    x1, y1 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    inter = max(0, x1 - x0) * max(0, y1 - y0)
    union = aw * ah + bw * bh - inter
    return float(inter / union) if union else 0.0


def _center_error(pred, target, shape) -> float:
    if pred is None or target is None:
        return 1.0
    px, py, pw, ph = pred; tx, ty, tw, th = target
    pc = np.asarray((px + pw / 2, py + ph / 2), dtype=float)
    tc = np.asarray((tx + tw / 2, ty + th / 2), dtype=float)
    return float(np.linalg.norm(pc - tc) / np.linalg.norm(shape[::-1]))


def _score_streams(streams, masks, shape, scene, method):
    rows = []
    previous_pred = None
    for index, events in enumerate(streams):
        pred = _proposal(events, shape)
        target = _box(np.asarray(masks[index], dtype=bool))
        iou = _iou(pred, target)
        rows.append({"scene": scene, "method": method, "window": index, "iou": iou, "recall_iou50": float(iou >= 0.5), "center_error": _center_error(pred, target, shape), "has_proposal": float(pred is not None), "continuity": float(previous_pred is not None and pred is not None)})
        previous_pred = pred
    return rows


def _mean(rows):
    return {key: float(np.mean([float(row[key]) for row in rows])) for key in ("iou", "recall_iou50", "center_error", "has_proposal", "continuity")}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, default=Path("data/eveyes_comotion_v2_ready"))
    parser.add_argument("--scene", default="test/Scene_005")
    parser.add_argument("--regimes", nargs="*", default=REGIMES)
    parser.add_argument("--max-frames", type=int, default=120)
    parser.add_argument("--eval-width", type=int, default=960)
    parser.add_argument("--student", type=Path, default=Path("results/eveyes_v2_regime_f1_argmax_policy/policy.pt"))
    parser.add_argument("--confidence", type=float, default=0.50)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", type=Path, default=Path("results/ami_next/event_target_tracking_test.json"))
    args = parser.parse_args()
    torch.set_num_threads(8)
    torch.set_num_interop_threads(1)
    split, scene_name = args.scene.split("/")
    checkpoint = torch.load(args.student, map_location="cpu", weights_only=True)
    model = TinyCausalGearPolicy(feature_dim=FEATURE_DIM, hidden_dim=int(checkpoint["hidden_dim"]), num_gears=int(checkpoint["num_gears"]))
    model.load_state_dict(checkpoint["model"]); model.eval()
    rows = []
    for regime in args.regimes:
        scene_id = f"{split}/{scene_name}/{regime}"
        scene = _scene_path(args.dataset_root, split, scene_name, regime)
        frames, timestamps = load_rgb_sequence(scene / "images", scene / "timestamps.csv", timestamp_unit="s", max_frames=args.max_frames)
        timestamps = timestamps[: len(frames)]
        intrinsics = _intrinsics(scene)
        if args.eval_width != intrinsics.width:
            height = int(round(intrinsics.height * args.eval_width / intrinsics.width))
            frames = np.stack([cv2.resize(frame, (args.eval_width, height), interpolation=cv2.INTER_AREA) for frame in frames])
            intrinsics = intrinsics.resized(args.eval_width, height)
        luminance = rgb_to_linear_luminance(frames)
        masks = _load_object_masks(scene, len(frames), intrinsics.width, intrinsics.height)
        if masks is None:
            raise SystemExit(f"{scene_id} has no masks; generate pose-derived masks before this benchmark")
        valid = np.ones((len(frames), intrinsics.height, intrinsics.width), dtype=bool)
        prism = RotatingWedgePrism(intrinsics, 0.5)
        streams = {}
        cache_base = Path("results/ami_next/eveyes_v2_baseline_events")
        cache_name = {"camera_only": f"test_{scene_name}_camera_only_f480_w960_{{}}.npz", "object_only": f"test_{scene_name}_object_only_f480_w960_{{}}.npz", "co_motion": f"test_{scene_name}_f480_w960_{{}}.npz"}[regime]
        for method in ("dvs_voltmeter", "v2e"):
            path = cache_base / cache_name.format(method)
            streams[method] = [_slice(_read_npz(path), float(timestamps[i]), float(timestamps[i + 1])) for i in range(len(timestamps) - 1)]
        streams["ami_passive"], _, _ = _ami_windows(luminance, timestamps, valid, prism, 0.0, device=args.device)
        streams["ami_fixed_180"], _, _ = _ami_windows(luminance, timestamps, valid, prism, 180.0, device=args.device)
        streams["ami_fixed_360"], _, _ = _ami_windows(luminance, timestamps, valid, prism, 360.0, device=args.device)
        visual = _visual_motion(luminance, timestamps)
        imu = _imu_fraction(scene / "imu.csv", timestamps, (float(checkpoint["imu_motion_bounds"][0]), float(checkpoint["imu_motion_bounds"][1])))
        streams["ours"], _ = _ours_windows(luminance, timestamps, valid, prism, visual=visual, imu=imu, model=model, checkpoint=checkpoint, confidence=args.confidence, require_imu=False, dwell=0.025, device=args.device)
        for method in METHODS:
            rows.extend(_score_streams(streams[method], masks, (intrinsics.height, intrinsics.width), scene_id, method))
        print(f"evaluated {scene_id}", flush=True)
    aggregate = {method: _mean([row for row in rows if row["method"] == method]) for method in METHODS}
    per_regime = {regime: {method: _mean([row for row in rows if row["scene"].endswith('/' + regime) and row["method"] == method]) for method in METHODS} for regime in args.regimes}
    report = {"protocol": "ami_event_target_tracking_v1", "scene": args.scene, "regimes": list(args.regimes), "max_frames": args.max_frames, "detector": "event-count connected-component proposal; thresholds fixed before test", "masks": "target masks used only for scoring", "aggregate": aggregate, "per_regime": per_regime, "rows": rows}
    args.output.parent.mkdir(parents=True, exist_ok=True); args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"aggregate": aggregate, "per_regime": per_regime}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
