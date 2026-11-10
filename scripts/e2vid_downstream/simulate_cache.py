#!/usr/bin/env python3
"""Simulate reference-plane events for No Prism / 180 / 360 / EIRM.

Uses the frozen paper controller and the existing AMI inverse-warp pipeline.
Does not modify gears, dwell, or policy weights. Scene splits are applied
before the four modulation variants are generated.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

from ami_simulator.geometry import RotatingWedgePrism  # noqa: E402
from ami_simulator.motion_net import TinyCausalGearPolicy  # noqa: E402
from ami_simulator.pipeline import load_rgb_sequence, rgb_to_linear_luminance  # noqa: E402
from ami_simulator.types import EventStream  # noqa: E402
from protocol import (  # noqa: E402
    CONFIDENCE,
    CONTRAST_THRESHOLD,
    DATASET_ROOT,
    DWELL_S,
    EVAL_CONDITIONS,
    OUT_ROOT,
    POLICY_PATH,
    PRISM_ANGLE_DEG,
    REQUIRE_IMU,
    cache_path,
    scene_ids,
)
from scripts.evaluate_dl3dv_downstream import _ami_windows, _dvs_windows, _ours_windows, _visual_motion  # noqa: E402
from scripts.evaluate_eveyes_v2_downstream import _imu_fraction, _intrinsics, _scene_path  # noqa: E402
from scripts.train_inverse_motion_policy import FEATURE_DIM  # noqa: E402


def _concat(windows: list[EventStream], width: int, height: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    ptr = [0]
    xs, ys, ts, ps = [], [], [], []
    for window in windows:
        x = np.asarray(window.x, dtype=np.int32)
        y = np.asarray(window.y, dtype=np.int32)
        t = np.asarray(window.t, dtype=np.float64)
        p = np.asarray(window.p, dtype=np.int8)
        valid = (x >= 0) & (x < width) & (y >= 0) & (y < height)
        xs.append(x[valid].astype(np.uint16))
        ys.append(y[valid].astype(np.uint16))
        ts.append(t[valid].astype(np.float32))
        ps.append(p[valid])
        ptr.append(ptr[-1] + int(valid.sum()))
    if not xs:
        empty_u = np.empty(0, np.uint16)
        return empty_u, empty_u, np.empty(0, np.float32), np.empty(0, np.int8), np.asarray(ptr, np.int64)
    return (
        np.concatenate(xs),
        np.concatenate(ys),
        np.concatenate(ts),
        np.concatenate(ps),
        np.asarray(ptr, np.int64),
    )


def _load_policy(path: Path) -> tuple[TinyCausalGearPolicy, dict]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    model = TinyCausalGearPolicy(
        feature_dim=FEATURE_DIM,
        hidden_dim=int(checkpoint["hidden_dim"]),
        num_gears=int(checkpoint["num_gears"]),
    )
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model, checkpoint


def simulate_one(
    scene_id: str,
    condition: str,
    *,
    dataset_root: Path,
    max_frames: int | None,
    device: str,
    policy: TinyCausalGearPolicy,
    checkpoint: dict,
) -> dict:
    split, scene_name, regime = scene_id.split("/")
    scene = _scene_path(dataset_root, split, scene_name, regime)
    frames, timestamps = load_rgb_sequence(
        scene / "images",
        scene / "timestamps.csv",
        timestamp_unit="s",
        max_frames=max_frames,
    )
    timestamps = timestamps[: len(frames)]
    luminance = rgb_to_linear_luminance(frames)
    intrinsics = _intrinsics(scene)
    valid = np.ones((len(frames) - 1, intrinsics.height, intrinsics.width), dtype=bool)
    prism = RotatingWedgePrism(intrinsics, PRISM_ANGLE_DEG)
    rpms: list[float]
    if condition == "no_prism":
        windows, rpms = _dvs_windows(luminance, timestamps, contrast_threshold=CONTRAST_THRESHOLD)
    elif condition == "fixed_180":
        windows, _, rpms = _ami_windows(luminance, timestamps, valid, prism, 180.0, device=device, contrast_threshold=CONTRAST_THRESHOLD)
    elif condition == "fixed_360":
        windows, _, rpms = _ami_windows(luminance, timestamps, valid, prism, 360.0, device=device, contrast_threshold=CONTRAST_THRESHOLD)
    elif condition in {"eirm", "eirm_imu", "eirm_no_imu"}:
        visual = _visual_motion(luminance, timestamps)
        imu_bounds = (float(checkpoint["imu_motion_bounds"][0]), float(checkpoint["imu_motion_bounds"][1]))
        imu = _imu_fraction(scene / "imu.csv", timestamps, imu_bounds)
        if condition == "eirm_no_imu":
            imu = np.zeros_like(imu)
            use_gate = False
        elif condition == "eirm_imu":
            use_gate = True
        else:
            use_gate = REQUIRE_IMU
        windows, rpms = _ours_windows(
            luminance,
            timestamps,
            valid,
            prism,
            visual=visual,
            imu=imu,
            model=policy,
            checkpoint=checkpoint,
            confidence=CONFIDENCE,
            require_imu=use_gate,
            dwell=DWELL_S,
            device=device,
        )
    else:
        raise ValueError(condition)
    x, y, t, p, ptr = _concat(windows, intrinsics.width, intrinsics.height)
    motion = _visual_motion(luminance, timestamps)
    t0 = np.asarray(timestamps[:-1], dtype=np.float64)
    t1 = np.asarray(timestamps[1:], dtype=np.float64)
    return {
        "luminance": luminance.astype(np.float16),
        "timestamps": timestamps.astype(np.float64),
        "x": x,
        "y": y,
        "t": t,
        "p": p,
        "window_ptr": ptr,
        "rpm": np.asarray(rpms, dtype=np.float32),
        "t0": t0,
        "t1": t1,
        "visual_motion": motion.astype(np.float32),
        "width": np.int32(intrinsics.width),
        "height": np.int32(intrinsics.height),
        "n_frames": np.int32(len(frames)),
        "n_windows": np.int32(len(windows)),
        "n_events": np.int64(len(t)),
        "scene_id": np.asarray(scene_id),
        "condition": np.asarray(condition),
        "regime": np.asarray(regime),
        "split": np.asarray(split),
        "event_space": np.asarray("reference_plane_inverse_warped" if condition != "no_prism" else "camera_plane_dvs"),
        "policy": np.asarray(str(POLICY_PATH) if condition.startswith("eirm") else ""),
        "confidence": np.float32(CONFIDENCE),
        "require_imu": np.bool_(condition == "eirm_imu" or (condition == "eirm" and REQUIRE_IMU)),
        "imu_feature": np.asarray("zeroed" if condition == "eirm_no_imu" else "lagged_imu_motion"),
        "dwell_s": np.float32(DWELL_S),
        "prism_angle_deg": np.float32(PRISM_ANGLE_DEG),
        "image_dir": np.asarray(str(scene / "images")),
        "timestamps_csv": np.asarray(str(scene / "timestamps.csv")),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=DATASET_ROOT)
    parser.add_argument("--output", type=Path, default=OUT_ROOT / "cache")
    parser.add_argument("--splits", nargs="*", default=("train", "valid", "test"))
    parser.add_argument("--conditions", nargs="*", default=list(EVAL_CONDITIONS))
    parser.add_argument("--scene-ids", nargs="*", default=None)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--extra-test", action="store_true")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    wanted = args.scene_ids or [sid for split in args.splits for sid in scene_ids(split, extra_test=args.extra_test)]
    args.output.mkdir(parents=True, exist_ok=True)
    policy, checkpoint = _load_policy(POLICY_PATH)
    index_path = args.output / "index.json"
    index = []
    if index_path.exists():
        try:
            index = json.loads(index_path.read_text(encoding="utf-8")).get("records", [])
        except json.JSONDecodeError:
            index = []
    for scene_id in wanted:
        for condition in args.conditions:
            dest = cache_path(args.output, scene_id, condition)
            rec = {
                "scene_id": scene_id,
                "condition": condition,
                "path": str(dest),
            }
            if dest.exists() and not args.overwrite:
                print(f"skip existing {dest.name}", flush=True)
                rec["status"] = "exists"
                index.append(rec)
                continue
            print(f"simulate {scene_id} {condition} max_frames={args.max_frames}", flush=True)
            payload = simulate_one(
                scene_id,
                condition,
                dataset_root=args.dataset_root,
                max_frames=args.max_frames,
                device=args.device,
                policy=policy,
                checkpoint=checkpoint,
            )
            gray_path = args.output / f"{scene_id.replace('/', '__')}__luminance.npz"
            if not gray_path.exists():
                np.savez_compressed(
                    gray_path,
                    luminance=payload.pop("luminance"),
                    timestamps=payload.pop("timestamps"),
                )
            else:
                payload.pop("luminance", None)
                payload.pop("timestamps", None)
            np.savez_compressed(dest, **payload)
            rec.update({"status": "wrote", "n_events": int(payload["n_events"]), "n_windows": int(payload["n_windows"])})
            print(f"  {int(payload['n_events'])} events / {int(payload['n_windows'])} windows -> {dest.name}", flush=True)
            index.append(rec)
    meta = {
        "dataset": "EVEYES-co-motion-v2",
        "event_space": "AMI conditions are inverse-warped to the RGB reference plane; No Prism is native DVS on the same plane",
        "controller": str(POLICY_PATH),
        "confidence": CONFIDENCE,
        "require_imu": REQUIRE_IMU,
        "dwell_s": DWELL_S,
        "prism_angle_deg": PRISM_ANGLE_DEG,
        "max_frames": args.max_frames,
        "records": index,
    }
    seen = {(str(r.get("scene_id")), str(r.get("condition"))): i for i, r in enumerate(index)}
    # `index` already includes this process's records; keep unique scene/condition.
    unique = []
    uniq_keys = set()
    for rec in index:
        key = (str(rec.get("scene_id")), str(rec.get("condition")))
        if key in uniq_keys:
            unique = [u for u in unique if (u.get("scene_id"), u.get("condition")) != key]
        unique.append(rec)
        uniq_keys.add(key)
    meta["records"] = unique
    tmp = index_path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    tmp.replace(index_path)
    print("wrote", index_path, "n=", len(unique), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
