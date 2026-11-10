#!/usr/bin/env python3
"""EVEYES co-motion v2 downstream: official DVS-Voltmeter, v2e, AMI, and Ours.

This is the Blender 960x540 pack. Do not point this script at
data/comotion_controlled_v2 (64x64 procedural toy data).

AMI events are inverse-warped to the reference plane. Image-wide edge F1 is
the Table 1 metric. When object_masks exist they are used only at scoring
(object-ROI F1 and TEO), never as AMI validity masks and never as controller
input. ORB is not scored.

Ours is the F1-argmax gear policy trained on native EVEYES 180/360 window F1,
not the 64x64 procedural checkpoint.
"""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

import cv2
import numpy as np
import torch

from ami_simulator.dsec import read_imu_csv
from ami_simulator.event_quality import compute_reference_edge_mask, event_quality_metrics, target_observability_metrics
from ami_simulator.geometry import RotatingWedgePrism
from ami_simulator.motion_net import TinyCausalGearPolicy
from ami_simulator.pipeline import load_rgb_sequence, rgb_to_linear_luminance
from ami_simulator.types import CameraIntrinsics
from scripts.evaluate_dl3dv_downstream import _ami_windows, _dvs_windows, _ours_windows, _visual_motion
from scripts.evaluate_inverse_motion_control import temporal_overlap_enabled
from scripts.evaluate_dvs_v2e_ours import _external, _read_npz, _slice
from scripts.train_inverse_motion_policy import FEATURE_DIM, WINDOW

SCALAR_KEYS = ("edge_f1", "edge_precision", "edge_recall", "event_count", "unique_pixel_ratio")
OBJECT_KEYS = (
    "object_edge_f1",
    "object_edge_precision",
    "object_edge_recall",
    "teo",
    "target_coverage",
    "target_purity",
    "target_f1",
    "background_event_count",
    "target_event_count",
)
OBJECT_MEAN_AREA = 0.05
SCENES = (
    ("train", "Scene_002"),
    ("valid", "Scene_004"),
    ("test", "Scene_005"),
)
REGIMES = ("co_motion", "camera_only", "object_only", "counter_motion", "drive_stop_drive")
BASELINE_METHODS = ("dvs_voltmeter", "v2e", "ami_passive", "ami_fixed_180", "ami_fixed_360", "ami_fixed_180_sensor")
PAPER_METHODS = ("dvs_voltmeter", "v2e", "ami_passive", "ami_fixed_180", "ami_fixed_360", "ours")


def _intrinsics(scene: Path) -> CameraIntrinsics:
    calibration = json.loads((scene / "camera_intrinsics.json").read_text(encoding="utf-8"))
    return CameraIntrinsics(int(calibration["width"]), int(calibration["height"]), float(calibration["fx"]), float(calibration["fy"]), float(calibration["cx"]), float(calibration["cy"]))


def _imu_fraction(imu_csv: Path, timestamps: np.ndarray, bounds: tuple[float, float]) -> np.ndarray:
    imu = read_imu_csv(imu_csv, timestamp_unit="s", gyro_unit="rad/s")
    motion = np.linalg.norm(imu.gyro, axis=1)
    windows = np.empty(len(timestamps) - 1, dtype=np.float64)
    for index, (start, end) in enumerate(zip(timestamps[:-1], timestamps[1:])):
        left = int(np.searchsorted(imu.t, start, side="left"))
        right = int(np.searchsorted(imu.t, end, side="right"))
        values = motion[left:right]
        windows[index] = float(values.mean() if len(values) else np.interp((start + end) * 0.5, imu.t, motion))
    low, high = bounds
    return np.clip((windows - low) / max(high - low, 1e-9), 0.0, 1.0)


def _scene_path(root: Path, split: str, scene: str, regime: str) -> Path:
    nested = root / split / scene / regime
    if (nested / "images").exists():
        return nested
    legacy = root / split / scene
    if regime == "co_motion" and (legacy / "images").exists():
        return legacy
    raise SystemExit(f"missing ready sequence {nested}")


def _scene_aliases(scene_id: str) -> tuple[str, ...]:
    parts = scene_id.split("/")
    if len(parts) == 3 and parts[2] == "co_motion":
        return (scene_id, f"{parts[0]}/{parts[1]}")
    return (scene_id,)


def _load_object_masks(scene: Path, n_frames: int, width: int, height: int) -> np.ndarray | None:
    paths = sorted((scene / "object_masks").glob("*.png"))[:n_frames]
    if not paths:
        return None
    if len(paths) != n_frames:
        raise SystemExit(f"{scene} has {len(paths)} object masks for {n_frames} frames")
    masks = []
    for path in paths:
        image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise SystemExit(f"failed to read {path}")
        if image.shape != (height, width):
            image = cv2.resize(image, (width, height), interpolation=cv2.INTER_NEAREST)
        masks.append(image > 127)
    return np.stack(masks)


def _structure_metrics(events, reference, mask, previous_active=None):
    edge = compute_reference_edge_mask(reference, edge_quantile=0.75) & mask
    quality = event_quality_metrics(events, reference_edge_mask=edge, previous_active_pixels=previous_active, image_shape=reference.shape)
    if len(events):
        active = np.unique(events.y.astype(np.int64) * reference.shape[1] + events.x.astype(np.int64))
    else:
        active = np.empty(0, np.int64)
    return {
        "edge_f1": float(quality["edge_f1"]),
        "edge_precision": float(quality["edge_precision"]),
        "edge_recall": float(quality["edge_coverage"]),
        "event_count": len(events),
        "unique_pixel_ratio": float(quality["unique_pixel_ratio"]),
    }, active, edge


def _score(windows, luminance, valid_masks, timestamps, method, scene_id, rpms=None, object_masks=None):
    rows = []
    previous = None
    for index, events in enumerate(windows):
        metrics, previous, edge = _structure_metrics(events, luminance[index], valid_masks[index], previous)
        row = {
            "scene": scene_id,
            "window": index,
            "method": method,
            "t0": float(timestamps[index]),
            "t1": float(timestamps[index + 1]),
            **metrics,
        }
        if rpms is not None:
            row["rpm"] = float(rpms[index])
        if object_masks is not None:
            target = np.asarray(object_masks[index], dtype=bool)
            row["object_area"] = float(target.mean())
            object_edge = edge & target
            object_quality = event_quality_metrics(events, reference_edge_mask=object_edge, image_shape=luminance[index].shape)
            teo = target_observability_metrics(
                events,
                target_mask=target,
                target_edge_mask=object_edge,
                image_shape=luminance[index].shape,
                time_range=(float(timestamps[index]), float(timestamps[index + 1])),
            )
            row["object_edge_f1"] = float(object_quality["edge_f1"])
            row["object_edge_precision"] = float(object_quality["edge_precision"])
            row["object_edge_recall"] = float(object_quality["edge_coverage"])
            row["teo"] = float(teo["teo"])
            row["target_coverage"] = float(teo["target_coverage"])
            row["target_purity"] = float(teo["target_purity"])
            row["target_f1"] = float(teo["target_f1"])
            row["background_event_count"] = int(teo["background_event_count"])
            row["target_event_count"] = int(teo["target_event_count"])
        rows.append(row)
    return rows


def _mean(rows, method):
    selected = [row for row in rows if row["method"] == method]
    if not selected:
        raise ValueError(f"no rows for method {method}")
    summary = {key: float(np.mean([float(row[key]) for row in selected])) for key in SCALAR_KEYS}
    rpm_rows = [row for row in selected if "rpm" in row]
    if rpm_rows:
        summary["mean_rpm"] = float(np.mean([float(row["rpm"]) for row in rpm_rows]))
        summary["frac_360"] = float(np.mean([float(row["rpm"]) >= 359.0 for row in rpm_rows]))
    object_rows = [row for row in selected if "object_edge_f1" in row and float(row.get("object_area", 0.0)) >= OBJECT_MEAN_AREA]
    if object_rows:
        for key in OBJECT_KEYS:
            summary[key] = float(np.mean([float(row[key]) for row in object_rows]))
        summary["object_windows"] = len(object_rows)
        summary["object_area"] = float(np.mean([float(row["object_area"]) for row in object_rows]))
    return summary


def _reuse_rows(baseline_rows: list[dict[str, object]], scene_id: str) -> list[dict[str, object]]:
    aliases = set(_scene_aliases(scene_id))
    reused = [row for row in baseline_rows if row["scene"] in aliases and row["method"] != "ours"]
    remapped = []
    for row in reused:
        copy = dict(row)
        copy["scene"] = scene_id
        remapped.append(copy)
    return remapped


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=Path("data/eveyes_comotion_v2_ready"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--splits", nargs="*", default=("test",))
    parser.add_argument("--regimes", nargs="*", default=("co_motion",), choices=REGIMES)
    parser.add_argument("--scene-ids", nargs="*", default=None, help="Explicit split/scene/regime ids. Overrides --splits/--regimes.")
    parser.add_argument("--max-frames", type=int, default=8)
    parser.add_argument("--eval-width", type=int, default=960, help="960 is native. Smaller only for CPU smoke, and must be recorded.")
    parser.add_argument("--prism-angle-deg", type=float, default=0.5)
    parser.add_argument("--v2e-root", type=Path, default=Path("/data/fcr/vendor/v2e"))
    parser.add_argument("--v2e-python", type=Path, default=Path("/data/fcr/miniconda3/envs/v2e/bin/python"))
    parser.add_argument("--dvs-root", type=Path, default=Path("/data/fcr/code/event_vggt/Data_Part/DVS-Voltmeter"))
    parser.add_argument("--dvs-python", type=Path, default=Path("/data/fcr/miniconda3/envs/dvsvolt/bin/python"))
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--cache", type=Path, default=Path("results/ami_next/eveyes_v2_baseline_events"))
    parser.add_argument("--torch-threads", type=int, default=8)
    parser.add_argument("--student", type=Path, default=None, help="EVEYES-native F1-argmax checkpoint. Required for Ours.")
    parser.add_argument("--baseline-json", type=Path, default=None, help="Reuse DVS/v2e/fixed-AMI rows for sequences that already exist in the JSON.")
    parser.add_argument("--confidence", type=float, default=0.50, help="Softmax mass required to leave 180 RPM. 0.70 plus a 0.01 F1 margin collapses Ours to always-180.")
    parser.add_argument("--require-imu", action="store_true", default=True)
    parser.add_argument("--no-require-imu", action="store_false", dest="require_imu")
    parser.add_argument("--dwell", type=float, default=0.025)
    parser.add_argument("--device", default="cpu", help="cpu or cuda:N. Warp and AMI events follow this device.")
    parser.add_argument("--skip-dvs-v2e", action="store_true", help="Only run AMI gears. Use for teacher oracles.")
    parser.add_argument("--ami-rpms", nargs="*", type=float, default=None, help="Subset of AMI gears, e.g. 180 360. Default is 0, 180, 360.")
    args = parser.parse_args()
    if args.max_frames < 2 or args.eval_width < 32 or args.torch_threads < 1:
        raise SystemExit("max-frames >= 2, eval-width >= 32, torch-threads >= 1")
    if args.student is not None and args.max_frames < WINDOW + 2:
        raise SystemExit(f"max-frames must be >= {WINDOW + 2} so the causal policy has a full history window")
    if "comotion_controlled" in str(args.dataset_root):
        raise SystemExit("Refusing the 64x64 procedural pack. Point at data/eveyes_comotion_v2_ready.")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit(f"Requested {args.device} but torch.cuda.is_available() is False.")
    torch.set_num_threads(args.torch_threads)
    torch.set_num_interop_threads(1)
    ami_rpms = {float(value) for value in (args.ami_rpms if args.ami_rpms is not None else (0.0, 180.0, 360.0))}

    model = None
    checkpoint: dict[str, object] | None = None
    imu_bounds = None
    if args.student is not None:
        checkpoint = torch.load(args.student, map_location="cpu", weights_only=True)
        if checkpoint.get("architecture") != "tiny_inverse_motion_gear":
            raise SystemExit("Expected the F1-argmax tiny inverse-motion gear checkpoint.")
        if "64x64" in str(checkpoint.get("dataset", "")) or "comotion_controlled" in str(args.student):
            raise SystemExit("Refusing the 64x64 procedural checkpoint. Train on EVEYES native oracles.")
        model = TinyCausalGearPolicy(feature_dim=FEATURE_DIM, hidden_dim=int(checkpoint["hidden_dim"]), num_gears=int(checkpoint["num_gears"]))
        model.load_state_dict(checkpoint["model"])
        model.eval()
        if "imu_motion_bounds" not in checkpoint:
            raise SystemExit("EVEYES Ours checkpoint must store imu_motion_bounds. Do not use the 64x64 [0, 1.87] gate.")
        imu_bounds = (float(checkpoint["imu_motion_bounds"][0]), float(checkpoint["imu_motion_bounds"][1]))

    wanted = {name for name in args.splits}
    if args.scene_ids:
        scene_ids = list(args.scene_ids)
        for scene_id in scene_ids:
            if len(scene_id.split("/")) != 3:
                raise SystemExit(f"scene-ids must be split/scene/regime, got {scene_id}")
    else:
        scene_ids = [f"{split}/{scene}/{regime}" for split, scene in SCENES if split in wanted for regime in args.regimes]
    methods = list(BASELINE_METHODS)
    paper_methods = list(PAPER_METHODS)
    if args.student is None:
        paper_methods = [method for method in paper_methods if method != "ours"]
    rows: list[dict[str, object]] = []
    baseline_rows: list[dict[str, object]] = []
    if args.baseline_json is not None:
        baseline_rows = json.loads(args.baseline_json.read_text(encoding="utf-8"))["rows"]

    def _write(path: Path) -> dict[str, object]:
        present = {str(row["method"]) for row in rows}
        used = [method for method in methods + (["ours"] if args.student is not None else []) if method in present]
        paper = [method for method in paper_methods if method in present]
        present_scenes = [scene_id for scene_id in scene_ids if any(row["scene"] == scene_id for row in rows)]
        aggregate = {method: _mean(rows, method) for method in used} if rows else {}
        per_scene = {
            scene_id: {method: _mean([row for row in rows if row["scene"] == scene_id], method) for method in used}
            for scene_id in present_scenes
        }
        report = {
            "protocol": "ami_eveyes_v2_downstream_v3",
            "dataset": "EVEYES-co-motion-v2",
            "source_resolution": [960, 540],
            "eval_width": args.eval_width,
            "native_resolution": args.eval_width == 960,
            "max_frames": args.max_frames,
            "fps": 60.0,
            "prism_angle_deg": args.prism_angle_deg,
            "timeline": "rgb_60hz_render_clock",
            "device": args.device,
            "condition": list(args.regimes),
            "student": None if args.student is None else str(args.student),
            "confidence": args.confidence,
            "use_temporal_overlap": None if checkpoint is None else temporal_overlap_enabled(checkpoint),
            "upshift_margin": None if checkpoint is None else checkpoint.get("upshift_margin"),
            "require_imu": args.require_imu,
            "dwell": args.dwell,
            "history_window": WINDOW,
            "imu_motion_bounds": None if imu_bounds is None else list(imu_bounds),
            "storm_density_feature": None if checkpoint is None else checkpoint.get("storm_density_feature"),
            "methods": list(paper),
            "object_mask_protocol": "AABB neighbor masks scored only when screen area >= 0.05; AMI simulation is always image-wide",
            "no_object_mask": not any("object_edge_f1" in row for row in rows),
            "not_the_64x64_procedural_pack": True,
            "warp": "AMI paper rows use inverse-warped reference-plane events.",
            "aggregate": aggregate,
            "paper_aggregate": {} if len(set(args.regimes)) > 1 else {method: aggregate[method] for method in paper},
            "aggregate_note": "Do not average image-wide F1 across motion regimes. Read per_scene.",
            "per_scene": per_scene,
            "rows": rows,
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
        return report

    for scene_id in scene_ids:
        split, scene_name, regime = scene_id.split("/")
        scene = _scene_path(args.dataset_root, split, scene_name, regime)
        frames, timestamps = load_rgb_sequence(scene / "images", scene / "timestamps.csv", timestamp_unit="s", max_frames=args.max_frames)
        timestamps = timestamps[: len(frames)]
        intrinsics = _intrinsics(scene)
        if args.eval_width != intrinsics.width:
            eval_height = int(round(intrinsics.height * args.eval_width / intrinsics.width))
            frames = np.stack([cv2.resize(frame, (args.eval_width, eval_height), interpolation=cv2.INTER_AREA) for frame in frames])
            intrinsics = intrinsics.resized(args.eval_width, eval_height)
        luminance = rgb_to_linear_luminance(frames)
        valid = np.ones((len(luminance), luminance.shape[1], luminance.shape[2]), dtype=bool)
        object_masks = _load_object_masks(scene, len(luminance), luminance.shape[2], luminance.shape[1])
        prism = RotatingWedgePrism(intrinsics, args.prism_angle_deg)
        reused = _reuse_rows(baseline_rows, scene_id) if baseline_rows else []
        if reused:
            rows.extend(reused)
            print(f"reused {scene_id} baseline windows={len(reused)} from {args.baseline_json}")
        else:
            streams: dict[str, list[object]] = {}
            rpm_logs: dict[str, list[float]] = {}
            if not args.skip_dvs_v2e:
                with tempfile.TemporaryDirectory(prefix="ami_eveyes_") as temporary:
                    for method in ("dvs_voltmeter", "v2e"):
                        cached = args.cache / f"{scene_id.replace('/', '_')}_f{args.max_frames}_w{args.eval_width}_{method}.npz"
                        if cached.exists():
                            events = _read_npz(cached)
                        else:
                            events = _external(method, luminance, timestamps, args, Path(temporary))
                            cached.parent.mkdir(parents=True, exist_ok=True)
                            np.savez_compressed(cached, x=np.asarray(events.x), y=np.asarray(events.y), t=np.asarray(events.t), p=np.asarray(events.p))
                        streams[method] = [_slice(events, float(timestamps[index]), float(timestamps[index + 1])) for index in range(len(timestamps) - 1)]
            if 0.0 in ami_rpms:
                streams["ami_passive"], rpm_logs["ami_passive"] = _dvs_windows(luminance, timestamps)
            if 180.0 in ami_rpms:
                streams["ami_fixed_180"], streams["ami_fixed_180_sensor"], rpm_logs["ami_fixed_180"] = _ami_windows(luminance, timestamps, valid, prism, 180.0, device=args.device)
                rpm_logs["ami_fixed_180_sensor"] = rpm_logs["ami_fixed_180"]
            if 360.0 in ami_rpms:
                streams["ami_fixed_360"], _, rpm_logs["ami_fixed_360"] = _ami_windows(luminance, timestamps, valid, prism, 360.0, device=args.device)
            for method, windows in streams.items():
                rows.extend(_score(windows, luminance, valid, timestamps, method, scene_id, rpm_logs.get(method), object_masks))
        if model is not None and checkpoint is not None and imu_bounds is not None:
            visual = _visual_motion(luminance, timestamps)
            imu = _imu_fraction(scene / "imu.csv", timestamps, imu_bounds)
            ours, ours_rpm = _ours_windows(
                luminance,
                timestamps,
                valid,
                prism,
                visual=visual,
                imu=imu,
                model=model,
                checkpoint=checkpoint,
                confidence=args.confidence,
                require_imu=args.require_imu,
                dwell=args.dwell,
                device=args.device,
            )
            rows.extend(_score(ours, luminance, valid, timestamps, "ours", scene_id, ours_rpm, object_masks))
            print(
                f"evaluated {scene_id} frames={len(frames)} size={luminance.shape[2]}x{luminance.shape[1]} "
                f"ours_mean_rpm={float(np.mean(ours_rpm)):.1f} "
                f"ours_frac_0={float(np.mean([rpm < 1 for rpm in ours_rpm])):.2f} "
                f"ours_frac_180={float(np.mean([179 <= rpm <= 181 for rpm in ours_rpm])):.2f} "
                f"ours_frac_360={float(np.mean([rpm >= 359 for rpm in ours_rpm])):.2f} "
                f"mean_imu={float(np.mean(imu)):.3f} mean_visual={float(np.mean(visual)):.2f}"
            )
        else:
            print(f"evaluated {scene_id} frames={len(frames)} size={luminance.shape[2]}x{luminance.shape[1]} masks={'yes' if object_masks is not None else 'no'}")
        report = _write(args.output)
        print(json.dumps({"scene": scene_id, "paper": {method: report["per_scene"][scene_id][method] for method in report["methods"] if method in report["per_scene"][scene_id]}}, indent=2, sort_keys=True))

    report = _write(args.output)
    summary = {"paper_aggregate": report["paper_aggregate"], "per_scene": {key: {method: report["per_scene"][key][method] for method in report["methods"]} for key in report["per_scene"]}}
    if "ami_fixed_180" in report["aggregate"] and "ami_fixed_180_sensor" in report["aggregate"]:
        summary["warp_180"] = {"reference": report["aggregate"]["ami_fixed_180"], "sensor": report["aggregate"]["ami_fixed_180_sensor"]}
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
