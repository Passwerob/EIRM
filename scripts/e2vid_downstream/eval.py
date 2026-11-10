#!/usr/bin/env python3
"""Evaluate one E2VID checkpoint on all four acquisition conditions.

The same weights are used for No Prism / Fixed-180 / Fixed-360 / EIRM.
Windows follow RGB timestamps. Reconstruction target is linear luminance
on the shared reference plane. Metrics use per-image affine alignment.
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

from data import load_sequence  # noqa: E402
from metrics import MetricBundle, summarize, temporal_pair_metrics  # noqa: E402
from model_io import forward_frame, load_e2vid  # noqa: E402
from protocol import EVAL_CONDITIONS, MOTION_REGIME_MAP, OUT_ROOT, WARMUP_FRAMES, scene_ids  # noqa: E402


def _load_weights(path: Path, device: torch.device):
    raw = torch.load(path, map_location="cpu", weights_only=False)
    if "state_dict" in raw and "arch" in raw:
        model = load_e2vid(device=device)
        # Mixed-trained checkpoints share E2VIDRecurrent keys; zero-shot uses the original tar.
        missing, unexpected = model.load_state_dict(raw["state_dict"], strict=False)
        if missing:
            print("missing", missing[:8], flush=True)
        if unexpected:
            print("unexpected", unexpected[:8], flush=True)
        return model.to(device).eval()
    return load_e2vid(path, device=device).eval()


def _detach_states(states):
    if states is None:
        return None
    detached = []
    for state in states:
        if state is None:
            detached.append(None)
        elif isinstance(state, tuple):
            detached.append(tuple(t.detach() for t in state))
        else:
            detached.append(state.detach())
    return detached


@torch.inference_mode()
def reconstruct_sequence(model, cache, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    """Replay one sequence. States are detached; the 479-step graph is not kept."""
    states = None
    recs = []
    gts = []
    for index in range(cache.n_windows):
        voxel = torch.from_numpy(cache.voxel(index)[None]).to(device)
        recon, states = forward_frame(model, voxel, states)
        states = _detach_states(states)
        recs.append(recon[0, 0].float().cpu().numpy())
        gts.append(cache.target(index))
        del voxel, recon
    return np.stack(recs), np.stack(gts)


def motion_bin(value: float, edges: tuple[float, float]) -> str:
    lo, hi = edges
    if value <= lo:
        return "weak"
    if value >= hi:
        return "strong"
    return "moderate"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--cache", type=Path, default=OUT_ROOT / "cache")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--splits", nargs="*", default=("test",))
    parser.add_argument("--conditions", nargs="*", default=list(EVAL_CONDITIONS))
    parser.add_argument("--extra-test", action="store_true")
    parser.add_argument("--stride", type=int, default=1, help="Evaluate every Nth window after a full recurrent rollout.")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--save-recon", action="store_true")
    parser.add_argument("--tag", default="")
    parser.add_argument("--scene-ids", nargs="*", default=None)
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = _load_weights(args.weights, device)
    metrics = MetricBundle(device)
    args.output.mkdir(parents=True, exist_ok=True)

    # Motion tertiles are frozen from the training-split No-Prism visual motion
    # so reconstruction quality cannot define the groups.
    motions = []
    train_ids = [s for s in (args.scene_ids or scene_ids("train")) if s.startswith("train/")]
    for scene_id in train_ids or scene_ids("train"):
        path = args.cache / f"{scene_id.replace('/', '__')}__no_prism.npz"
        if not path.exists():
            path = args.cache / f"{scene_id.replace('/', '__')}__fixed_180.npz"
        if not path.exists():
            continue
        with np.load(path) as z:
            motions.append(np.asarray(z["visual_motion"], dtype=np.float64))
    all_motion = np.concatenate(motions) if motions else np.asarray([0.0])
    lo, hi = np.quantile(all_motion, [1 / 3, 2 / 3])

    rows = []
    recon_index = []
    for split in args.splits:
        ids = args.scene_ids or scene_ids(split, extra_test=args.extra_test)
        for scene_id in ids:
            if scene_id.split("/")[0] != split:
                continue
            for condition in args.conditions:
                cache = load_sequence(args.cache, scene_id, condition)
                print(f"eval {scene_id} {condition} windows={cache.n_windows}", flush=True)
                recs, gts = reconstruct_sequence(model, cache, device)
                prev_pred = None
                prev_gt = None
                for index in range(cache.n_windows):
                    if index % args.stride != 0:
                        continue
                    frame = metrics.frame_metrics(recs[index], gts[index])
                    rpm = float(cache.rpm[index])
                    switched = bool(index > 0 and abs(float(cache.rpm[index]) - float(cache.rpm[index - 1])) > 1.0)
                    row = {
                        "split": split,
                        "scene_id": scene_id,
                        "regime": scene_id.split("/")[-1],
                        "condition": condition,
                        "window": index,
                        "warmup": bool(index < WARMUP_FRAMES),
                        "t0": float(cache.t0[index]),
                        "t1": float(cache.t1[index]),
                        "rpm": rpm,
                        "switched": switched,
                        "n_events": int(cache.ptr[index + 1] - cache.ptr[index]),
                        "visual_motion": float(cache.visual_motion[index]),
                        "motion_regime": MOTION_REGIME_MAP.get(scene_id.split("/")[-1], "unknown"),
                        "motion_bin": motion_bin(float(cache.visual_motion[index]), (float(lo), float(hi))),
                        **frame,
                    }
                    if prev_pred is not None:
                        row.update(temporal_pair_metrics(prev_pred, recs[index], prev_gt, gts[index]))
                    rows.append(row)
                    prev_pred, prev_gt = recs[index], gts[index]
                if args.save_recon:
                    dest = args.output / "recons" / f"{scene_id.replace('/', '__')}__{condition}.npz"
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    np.savez_compressed(
                        dest,
                        recon=recs.astype(np.float16),
                        gt=gts.astype(np.float16),
                        rpm=cache.rpm,
                        t1=cache.t1,
                    )
                    recon_index.append(str(dest))

    tables = {}
    for condition in args.conditions:
        selected = [r for r in rows if r["condition"] == condition and not r.get("warmup")]
        tables[condition] = {
            "all": summarize(selected),
            "per_scene": {
                sid: summarize([r for r in selected if r["scene_id"] == sid])
                for sid in sorted({r["scene_id"] for r in selected})
            },
            "per_motion_regime": {
                name: summarize([r for r in selected if r["motion_regime"] == name])
                for name in sorted({r["motion_regime"] for r in selected})
            },
            "per_motion_bin": {
                name: summarize([r for r in selected if r["motion_bin"] == name])
                for name in sorted({r["motion_bin"] for r in selected})
            },
            "transitions": summarize([r for r in selected if r["switched"]]),
        }
    report = {
        "weights": str(args.weights),
        "tag": args.tag,
        "affine_alignment": "per-image scale+shift of reconstruction to linear-luminance GT",
        "event_space": "reference plane (inverse-warped AMI; native DVS for No Prism)",
        "lpips_error": metrics.lpips_error,
        "motion_tertiles_from_train": [float(lo), float(hi)],
        "n_rows": len(rows),
        "tables": tables,
        "rows": rows,
        "recons": recon_index,
    }
    (args.output / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("wrote", args.output / "metrics.json", flush=True)
    for condition, block in tables.items():
        s = block["all"]
        print(
            f"{condition:12s}  PSNR {s.get('psnr_mean', float('nan')):.3f}  "
            f"SSIM {s.get('ssim_mean', float('nan')):.3f}  "
            f"LPIPS {s.get('lpips_mean', float('nan')):.3f}  n={int(s.get('n', 0))}",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
