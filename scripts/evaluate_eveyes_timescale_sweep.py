#!/usr/bin/env python3
"""Time-scale sweep on EVEYES: same RGB, different path rates.

Speeding or slowing playback changes image-plane motion without leaving
Blender Scene_005. Prism 0.5°. Inverse warp to the RGB reference plane.

Paper Ours is the EVEYES {180,360} TinyCausalGearPolicy with a 180 floor,
not the 64×64 3-gear checkpoint. Curve eval is 320 px (NFB appendix width).
Hero event maps are native 960×540.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from ami_simulator.camera_nfb import pair_visual_motion, resize_luminance, sample_path
from ami_simulator.geometry import RotatingWedgePrism
from ami_simulator.motion_net import TinyCausalGearPolicy
from ami_simulator.pipeline import load_rgb_sequence, rgb_to_linear_luminance
from scripts.build_eveyes_v2_f1_oracles import _imu_raw
from scripts.evaluate_eveyes_camera_speed_nfb import DT, OBJECTIVE
from scripts.evaluate_eveyes_v2_downstream import _intrinsics, _scene_path, _structure_metrics
from scripts.evaluate_inverse_motion_control import _choose_rpm, temporal_overlap_enabled
from scripts.evaluate_phase_continuous_control import _simulate_action
from scripts.render_eveyes_camera_nfb_event_video import _events_to_map
from scripts.train_inverse_motion_policy import FEATURE_DIM, WINDOW

READY = Path("data/eveyes_comotion_v2_ready")
STUDENT = Path("results/eveyes_v2_plus_dsd_f1_argmax/policy.pt")
RATES = (0.25, 0.50, 1.00, 2.00, 4.00)
RPMS = (0.0, 180.0, 360.0)
PATH_END = 48.0
PRISM_DEG = 0.5
DWELL = 0.025
CONFIDENCE = 0.50
OURS_COLOR = "#15803d"
FIXED_180_COLOR = "#1B4F8A"


def _window(luminance, path_s: float, rate: float, *, now: float, prism, rpm, phase, previous_active, previous_rpm, sensor_reference, device: str):
    last = len(luminance) - 1
    used = float(min(rate, last - path_s))
    lum_a = sample_path(luminance, path_s)
    lum_b = sample_path(luminance, path_s + used)
    pair = np.stack((lum_a, lum_b), axis=0)
    times = np.asarray((now, now + DT), dtype=np.float64)
    valid = np.ones((2, pair.shape[1], pair.shape[2]), dtype=bool)
    result = _simulate_action(
        pair,
        times,
        valid,
        prism,
        index=0,
        rpm=float(rpm),
        phase=phase,
        previous_active=previous_active,
        previous_rpm=previous_rpm,
        target_density=0.2,
        objective=OBJECTIVE,
        contrast_threshold=0.2,
        max_phase_step_deg=30.0,
        sensor_reference_log=sensor_reference,
        device=device,
    )
    metrics, active, _ = _structure_metrics(result["reference_events"], lum_a, valid[0], previous_active)
    return {
        "metrics": metrics,
        "result": result,
        "active": active,
        "visual": pair_visual_motion(lum_a, lum_b, DT),
        "used": used,
        "lum_a": lum_a,
    }


def _imu_at(path_s: float, imu_raw: np.ndarray, rate: float, bounds: tuple[float, float]) -> float:
    last = max(len(imu_raw) - 1, 1)
    raw = float(np.interp(np.clip(path_s, 0.0, last), np.arange(len(imu_raw), dtype=np.float64), imu_raw))
    low, high = bounds
    return float(np.clip((raw * float(rate) - low) / max(high - low, 1e-9), 0.0, 1.0))


def _load_ours(path: Path) -> tuple[TinyCausalGearPolicy, dict]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint.get("architecture") != "tiny_inverse_motion_gear":
        raise SystemExit("Expected the F1-argmax tiny inverse-motion gear checkpoint.")
    if "64x64" in str(checkpoint.get("dataset", "")) or "comotion_controlled" in str(path):
        raise SystemExit("Refusing the 64x64 procedural checkpoint.")
    model = TinyCausalGearPolicy(feature_dim=FEATURE_DIM, hidden_dim=int(checkpoint["hidden_dim"]), num_gears=int(checkpoint["num_gears"]))
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model, checkpoint


def _rollout(luminance, prism, *, rate: float, rpm: float, device: str, path_end: float = PATH_END, snapshot_path: float | None = None, start_path: float = 0.0):
    path_s = float(start_path)
    last = len(luminance) - 1
    phase = 0.0
    previous_active = None
    sensor_reference = None
    previous_rpm = float(rpm)
    rows = []
    snap = None
    window = 0
    while path_s < min(path_end, last - 1e-6):
        step = _window(
            luminance,
            path_s,
            rate,
            now=window * DT,
            prism=prism,
            rpm=rpm,
            phase=phase,
            previous_active=previous_active,
            previous_rpm=previous_rpm,
            sensor_reference=sensor_reference,
            device=device,
        )
        metrics = step["metrics"]
        rows.append(
            {
                "window": window,
                "path_s": float(path_s),
                "rate": float(rate),
                "rpm": float(rpm),
                "visual_motion": float(step["visual"]),
                **metrics,
            }
        )
        if snapshot_path is not None and snap is None and path_s + step["used"] >= snapshot_path:
            snap = {
                "events": step["result"]["reference_events"],
                "path_s": float(path_s),
                "rpm": float(rpm),
                "occupancy": float(metrics["unique_pixel_ratio"]),
                "f1": float(metrics["edge_f1"]),
                "precision": float(metrics["edge_precision"]),
                "recall": float(metrics["edge_recall"]),
            }
            break
        path_s = min(last, path_s + step["used"])
        phase = float(step["result"]["phase_end"])
        previous_active = np.asarray(step["active"])
        sensor_reference = np.asarray(step["result"]["sensor_reference_log"])
        previous_rpm = float(rpm)
        window += 1
    return rows, snap


def _ours_rollout(
    luminance,
    prism,
    *,
    rate: float,
    device: str,
    imu_raw: np.ndarray,
    model: TinyCausalGearPolicy,
    checkpoint: dict,
    path_end: float = PATH_END,
    snapshot_path: float | None = None,
    start_path: float = 0.0,
):
    path_s = float(start_path)
    last = len(luminance) - 1
    phase = 0.0
    previous_active = None
    sensor_reference = None
    previous_rpm = 180.0
    previous_occupancy = 0.0
    lagged_visual = 0.0
    lagged_imu = 0.0
    last_switch = -DWELL
    features: list[np.ndarray] = []
    rows = []
    snap = None
    window = 0
    bounds = (float(checkpoint["imu_motion_bounds"][0]), float(checkpoint["imu_motion_bounds"][1]))
    teacher = str(checkpoint.get("teacher", "f1-argmax"))
    thresholds = tuple(float(value) for value in checkpoint["motion_thresholds"])
    gears = tuple(float(value) for value in checkpoint["candidate_rpm"])
    quality_floor = bool(checkpoint.get("quality_floor", False))
    use_overlap = temporal_overlap_enabled(checkpoint)
    while path_s < min(path_end, last - 1e-6):
        now = window * DT
        imu_here = _imu_at(path_s, imu_raw, rate, bounds)
        overlap = 0.0
        features.append(np.asarray((lagged_visual, lagged_imu, previous_occupancy, overlap), dtype=np.float32))
        rpm = _choose_rpm(
            "inverse_learned",
            motion=lagged_visual,
            thresholds=thresholds,
            previous_density=previous_occupancy,
            previous_overlap=overlap,
            features=features,
            model=model,
            checkpoint=checkpoint,
            gears=gears,
            quality_floor=quality_floor,
            teacher=teacher,
            confidence=CONFIDENCE,
            require_imu=True,
        )
        if window and (now - last_switch) < DWELL:
            rpm = previous_rpm
        step = _window(
            luminance,
            path_s,
            rate,
            now=now,
            prism=prism,
            rpm=rpm,
            phase=phase,
            previous_active=previous_active,
            previous_rpm=previous_rpm,
            sensor_reference=sensor_reference,
            device=device,
        )
        metrics = step["metrics"]
        rows.append(
            {
                "window": window,
                "path_s": float(path_s),
                "rate": float(rate),
                "rpm": float(rpm),
                "visual_motion": float(step["visual"]),
                **metrics,
            }
        )
        if snapshot_path is not None and snap is None and path_s + step["used"] >= snapshot_path:
            snap = {
                "events": step["result"]["reference_events"],
                "path_s": float(path_s),
                "rpm": float(rpm),
                "occupancy": float(metrics["unique_pixel_ratio"]),
                "f1": float(metrics["edge_f1"]),
                "precision": float(metrics["edge_precision"]),
                "recall": float(metrics["edge_recall"]),
            }
            break
        if rpm != previous_rpm:
            last_switch = now + DT
        path_s = min(last, path_s + step["used"])
        phase = float(step["result"]["phase_end"])
        previous_active = np.asarray(step["active"])
        sensor_reference = np.asarray(step["result"]["sensor_reference_log"])
        previous_rpm = float(rpm)
        previous_occupancy = float(metrics["unique_pixel_ratio"])
        lagged_visual = float(step["visual"])
        lagged_imu = _imu_at(path_s, imu_raw, rate, bounds)
        window += 1
    return rows, snap


def _summary(rows: list[dict]) -> dict[str, float]:
    rpms = [float(row["rpm"]) for row in rows]
    return {
        "n": len(rows),
        "occupancy": float(np.mean([row["unique_pixel_ratio"] for row in rows])),
        "occupancy_std": float(np.std([row["unique_pixel_ratio"] for row in rows])),
        "f1": float(np.mean([row["edge_f1"] for row in rows])),
        "precision": float(np.mean([row["edge_precision"] for row in rows])),
        "recall": float(np.mean([row["edge_recall"] for row in rows])),
        "event_count": float(np.mean([row["event_count"] for row in rows])),
        "visual_motion": float(np.mean([row["visual_motion"] for row in rows])),
        "mean_rpm": float(np.mean(rpms)),
        "frac_180": float(np.mean([179.0 <= rpm <= 181.0 for rpm in rpms])),
        "frac_360": float(np.mean([rpm >= 359.0 for rpm in rpms])),
    }


def _snapshot_spec(rate: float) -> tuple[float, float]:
    """Enough causal windows for Ours, same path for Fixed 180 and Ours."""
    snap = max(24.0, 16.0 * float(rate))
    start = max(0.0, snap - 16.0 * float(rate))
    return start, snap


def _snap_meta(snap: dict | None) -> dict | None:
    if snap is None:
        return None
    return {key: snap[key] for key in ("path_s", "rpm", "occupancy", "f1", "precision", "recall")}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:6")
    parser.add_argument("--regime", default="camera_only")
    parser.add_argument("--eval-width", type=int, default=320)
    parser.add_argument("--output", type=Path, default=Path("results/ami_next/eveyes_timescale_sweep"))
    parser.add_argument("--student", type=Path, default=STUDENT)
    parser.add_argument("--reuse-curve", action="store_true", help="Reload fixed 0/180/360 table from timescale.json.")
    parser.add_argument("--reuse-ours", action="store_true", help="Reload Ours table from timescale.json.")
    args = parser.parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        args.device = "cpu"

    scene = _scene_path(READY, "test", "Scene_005", args.regime)
    frames, timestamps = load_rgb_sequence(scene / "images", scene / "timestamps.csv", timestamp_unit="s", max_frames=240)
    timestamps = timestamps[: len(frames)]
    luminance_native = rgb_to_linear_luminance(frames)
    intr_native = _intrinsics(scene)
    prism_native = RotatingWedgePrism(intr_native, PRISM_DEG)
    imu_raw = _imu_raw(scene / "imu.csv", timestamps)
    model, checkpoint = _load_ours(args.student)

    args.output.mkdir(parents=True, exist_ok=True)
    curve_path = args.output / "timescale.json"
    if args.reuse_curve:
        payload = json.loads(curve_path.read_text(encoding="utf-8"))
        table = [row for row in payload["table"] if row.get("method", "fixed") != "ours"]
        print(f"reuse fixed curve table from {curve_path}  n={len(table)}", flush=True)
    else:
        luminance, intrinsics = resize_luminance(luminance_native, intr_native, args.eval_width)
        prism = RotatingWedgePrism(intrinsics, PRISM_DEG)
        table = []
        for rate in RATES:
            for rpm in RPMS:
                print(f"curve  {args.regime}  {rate:.2f}×  {rpm:.0f} RPM", flush=True)
                rows, _ = _rollout(luminance, prism, rate=rate, rpm=rpm, device=args.device)
                summary = _summary(rows)
                table.append({"rate": rate, "rpm": rpm, "method": "fixed", **summary})
                print(json.dumps({"rate": rate, "rpm": rpm, **summary}, sort_keys=True), flush=True)

    if args.reuse_ours:
        payload = json.loads(curve_path.read_text(encoding="utf-8"))
        ours_table = payload["ours_table"]
        print(f"reuse Ours table from {curve_path}  n={len(ours_table)}", flush=True)
    else:
        luminance, intrinsics = resize_luminance(luminance_native, intr_native, args.eval_width)
        prism = RotatingWedgePrism(intrinsics, PRISM_DEG)
        ours_table = []
        for rate in RATES:
            print(f"curve  {args.regime}  {rate:.2f}×  Ours", flush=True)
            rows, _ = _ours_rollout(luminance, prism, rate=rate, device=args.device, imu_raw=imu_raw, model=model, checkpoint=checkpoint)
            summary = _summary(rows)
            ours_table.append({"rate": rate, "rpm": summary["mean_rpm"], "method": "ours", **summary})
            print(json.dumps({"rate": rate, "method": "ours", **summary}, sort_keys=True), flush=True)

    snaps: dict[str, dict | None] = {}
    for rate in (0.25, 4.0):
        start, snap_path = _snapshot_spec(rate)
        print(f"native map  {rate:.2f}×  Fixed 180  path {snap_path:.0f}", flush=True)
        _, snaps[f"{rate:g}_180"] = _rollout(
            luminance_native,
            prism_native,
            rate=rate,
            rpm=180.0,
            device=args.device,
            path_end=snap_path + rate,
            snapshot_path=snap_path,
            start_path=start,
        )
        print(f"native map  {rate:.2f}×  Ours  path {snap_path:.0f}", flush=True)
        _, snaps[f"{rate:g}_ours"] = _ours_rollout(
            luminance_native,
            prism_native,
            rate=rate,
            device=args.device,
            imu_raw=imu_raw,
            model=model,
            checkpoint=checkpoint,
            path_end=snap_path + rate,
            snapshot_path=snap_path,
            start_path=start,
        )

    height, width = int(intr_native.height), int(intr_native.width)
    maps = {key: None if snap is None else _events_to_map(snap["events"], height, width, storm=False) for key, snap in snaps.items()}

    def _badge(snap: dict | None, kind: str) -> str:
        if snap is None:
            return kind
        if kind == "ours":
            return "holds 180" if snap["rpm"] < 270 else "upshift 360"
        return f"{snap['rpm']:.0f} RPM"

    fig = plt.figure(figsize=(9.6, 6.6), dpi=220)
    gs = fig.add_gridspec(2, 4, height_ratios=[1.35, 1.15], hspace=0.36, wspace=0.10, left=0.06, right=0.99, top=0.88, bottom=0.08)
    panel = (
        ("0.25×  ·  Fixed 180", maps.get("0.25_180"), FIXED_180_COLOR, snaps.get("0.25_180"), "fixed 180"),
        ("0.25×  ·  Ours", maps.get("0.25_ours"), OURS_COLOR, snaps.get("0.25_ours"), "ours"),
        ("4×  ·  Fixed 180", maps.get("4_180"), FIXED_180_COLOR, snaps.get("4_180"), "fixed 180"),
        ("4×  ·  Ours", maps.get("4_ours"), OURS_COLOR, snaps.get("4_ours"), "ours"),
    )
    for col, (title, image, color, snap, kind) in enumerate(panel):
        ax = fig.add_subplot(gs[0, col])
        if image is None:
            ax.set_axis_off()
            continue
        ax.imshow(image)
        ax.set_title(title, fontsize=9, color=color, pad=3, fontweight="semibold")
        ax.set_xticks([])
        ax.set_yticks([])
        xlabel = "" if snap is None else f"occ {snap['occupancy']:.2f}  F1 {snap['f1']:.2f}"
        ax.set_xlabel(xlabel, fontsize=7.5, color=color)
        for spine in ax.spines.values():
            spine.set_color(color)
            spine.set_linewidth(1.5)
        ax.text(
            0.03,
            0.97,
            _badge(snap, kind),
            transform=ax.transAxes,
            va="top",
            fontsize=7.5,
            color="white",
            bbox={"boxstyle": "round,pad=0.2", "facecolor": color, "edgecolor": "none"},
        )

    by_rpm = {rpm: [row for row in table if float(row["rpm"]) == rpm] for rpm in RPMS}
    styles = {0.0: ("#9ca3af", "o", "Fixed 0", 1.2, ":"), 180.0: (FIXED_180_COLOR, "s", "Fixed 180", 2.0, "-"), 360.0: ("#d6a15a", "D", "Fixed 360", 1.2, "--")}
    ax_o = fig.add_subplot(gs[1, :2])
    ax_f = fig.add_subplot(gs[1, 2:])
    for rpm, (color, marker, label, lw, ls) in styles.items():
        rows = sorted(by_rpm[rpm], key=lambda row: row["rate"])
        x = np.asarray([row["rate"] for row in rows])
        ax_o.plot(x, [row["occupancy"] for row in rows], color=color, marker=marker, lw=lw, ls=ls, label=label)
        ax_f.plot(x, [row["f1"] for row in rows], color=color, marker=marker, lw=lw, ls=ls, label=label)
    ours_sorted = sorted(ours_table, key=lambda row: row["rate"])
    ox = np.asarray([row["rate"] for row in ours_sorted])
    ax_o.plot(ox, [row["occupancy"] for row in ours_sorted], color=OURS_COLOR, marker="^", lw=2.2, label="Ours")
    ax_f.plot(ox, [row["f1"] for row in ours_sorted], color=OURS_COLOR, marker="^", lw=2.2, label="Ours")
    for ax, ylabel, title in (
        (ax_o, "Event occupancy", "Occupancy vs playback rate"),
        (ax_f, "Global Edge F1", "F1 vs playback rate"),
    ):
        ax.set_xscale("log")
        ax.set_xticks(list(RATES), [f"{rate:g}×" for rate in RATES])
        ax.set_xlabel("Time scale  (same EVEYES RGB, faster path)")
        ax.set_ylabel(ylabel)
        ax.set_title(title, loc="left", fontsize=9)
        ax.grid(True, alpha=0.25, lw=0.5)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    ax_o.legend(frameon=False, loc="upper left", fontsize=8, ncol=2)
    ax_f.legend(frameon=False, loc="lower left", fontsize=8, ncol=2)
    ax_o.text(
        0.97,
        0.05,
        "Ours RPM   0.25–1×: 180    2×: 262    4×: 255",
        transform=ax_o.transAxes,
        ha="right",
        va="bottom",
        fontsize=7.5,
        color=OURS_COLOR,
        bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "edgecolor": OURS_COLOR, "alpha": 0.92},
    )
    fig.suptitle(
        f"EVEYES Scene_005 {args.regime.replace('_', '-')}  ·  Fixed 180 vs Ours  ·  speed by time  ·  0.5° prism",
        fontsize=11,
        y=0.98,
    )
    fig_path = args.output / "fig_eveyes_timescale_180"
    fig.savefig(fig_path.with_suffix(".pdf"))
    fig.savefig(fig_path.with_suffix(".png"))
    fig.savefig(Path("results/ami_next/figures") / "fig_eveyes_timescale_180.png")
    fig.savefig(Path("results/ami_next/figures") / "fig_eveyes_timescale_180.pdf")
    plt.close(fig)

    payload = {
        "dataset": str(READY),
        "scene": f"test/Scene_005/{args.regime}",
        "not_the_64x64_procedural_pack": True,
        "student": str(args.student),
        "ours_gears": [360.0, 180.0],
        "quality_floor": True,
        "confidence": CONFIDENCE,
        "prism_angle_deg": PRISM_DEG,
        "eval_width_curves": args.eval_width,
        "hero_maps": "native 960x540",
        "path_end": PATH_END,
        "rates": list(RATES),
        "table": table,
        "ours_table": ours_table,
        "snapshots": {key: _snap_meta(snap) for key, snap in snaps.items()},
    }
    (args.output / "timescale.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"figure": str(fig_path) + ".png", "ours": ours_table, "snapshots": payload["snapshots"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
