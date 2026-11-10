#!/usr/bin/env python3
"""One figure: fixed 180 occupancy storm vs causal 3-gear suppression.

Uses a switching object-only trajectory so image speed explicitly
accelerates and decelerates. AMI events are inverse-warped. Same
colormap for every method — density differences are real counts.
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

from ami_simulator.geometry import RotatingWedgePrism
from ami_simulator.motion_net import TinyCausalGearPolicy
from ami_simulator.pipeline import load_rgb_sequence, rgb_to_linear_luminance
from ami_simulator.types import EventStream
from scripts.evaluate_dl3dv_downstream import _visual_motion
from scripts.evaluate_motion_excitation_sweep import (
    CONFIDENCE,
    DWELL,
    GEARS,
    OBJECTIVE,
    _adaptive_rollout,
    _commanded_speed,
    _fixed_rollout,
    _imu_windows,
    _intrinsics,
)
from scripts.evaluate_phase_continuous_control import _simulate_action
from scripts.generate_comotion_scenes import _render_scene
from scripts.render_eveyes_camera_nfb_event_video import _concat, _events_to_map
from scripts.train_inverse_motion_policy import FEATURE_DIM

ACCUM = 3


def _rollout_events(luminance, timestamps, masks, prism, rpms, device: str) -> list[EventStream]:
    phase = 0.0
    previous_active = None
    sensor_reference = None
    previous_rpm = float(rpms[0])
    streams = []
    ones = np.ones_like(masks)
    for index, rpm in enumerate(rpms):
        result = _simulate_action(
            luminance,
            timestamps,
            ones,
            prism,
            index=index,
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
        streams.append(result["reference_events"])
        phase = float(result["phase_end"])
        previous_active = np.asarray(result["active"])
        sensor_reference = np.asarray(result["sensor_reference_log"])
        previous_rpm = float(rpm)
    return streams


def _stack_map(streams: list[EventStream], index: int, height: int, width: int) -> np.ndarray:
    first = max(0, index - (ACCUM - 1))
    return _events_to_map(_concat(streams[first : index + 1]), height, width, storm=False)


def _upscale(image: np.ndarray, scale: int = 8) -> np.ndarray:
    return np.repeat(np.repeat(image, scale, axis=0), scale, axis=1)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=Path("results/ami_next/motion_excitation_sweep"))
    parser.add_argument("--student", type=Path, default=Path("results/comotion_f1_argmax_0_180_360/policy.pt"))
    parser.add_argument("--device", default="cuda:6")
    parser.add_argument("--seed", type=int, default=101)
    parser.add_argument("--frames", type=int, default=160)
    parser.add_argument("--scale", type=float, default=3.0)
    args = parser.parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        args.device = "cpu"

    data_root = args.output_root / "storm_switching"
    data_root.mkdir(parents=True, exist_ok=True)
    meta = _render_scene(
        data_root,
        split="test",
        seed=args.seed,
        condition="object_only",
        frames=args.frames,
        fps=120.0,
        size=64,
        motion_profile="switching",
        motion_scale=args.scale,
        scene_name=f"switch_object_s{args.scale:.0f}_seed{args.seed}",
    )
    scene = data_root / str(meta["scene"])
    frames, timestamps = load_rgb_sequence(scene / "images", scene / "timestamps.csv", timestamp_unit="s")
    timestamps = timestamps[: len(frames)]
    luminance = rgb_to_linear_luminance(frames)
    import imageio.v3 as imageio

    mask_paths = sorted((scene / "object_masks").glob("*.png"))[: len(frames)]
    masks = np.stack([np.asarray(imageio.imread(path)) > 127 for path in mask_paths])
    height, width = luminance.shape[1], luminance.shape[2]
    prism = RotatingWedgePrism(_intrinsics(scene, width, height), 2.0)
    traj = dict(np.load(scene / "image_plane_trajectory.npz"))
    speed = _commanded_speed(traj, timestamps, "object_x", "object_y")

    checkpoint = torch.load(args.student, map_location="cpu", weights_only=True)
    model = TinyCausalGearPolicy(feature_dim=FEATURE_DIM, hidden_dim=int(checkpoint["hidden_dim"]), num_gears=int(checkpoint["num_gears"]))
    model.load_state_dict(checkpoint["model"])
    model.eval()
    imu_bounds = (float(checkpoint["imu_motion_bounds"][0]), float(checkpoint["imu_motion_bounds"][1]))
    visual = _visual_motion(luminance, timestamps)
    imu = _imu_windows(scene, timestamps, imu_bounds)

    print("rollout fixed + adaptive", flush=True)
    fixed = {rpm: _fixed_rollout(luminance, timestamps, masks, prism, rpm, args.device) for rpm in GEARS}
    adaptive = _adaptive_rollout(
        luminance,
        timestamps,
        masks,
        prism,
        visual=visual,
        imu=imu,
        model=model,
        checkpoint=checkpoint,
        device=args.device,
    )
    rpms = {
        0.0: np.zeros(len(timestamps) - 1),
        180.0: np.full(len(timestamps) - 1, 180.0),
        360.0: np.full(len(timestamps) - 1, 360.0),
        "adaptive": np.asarray([row["rpm"] for row in adaptive], dtype=np.float64),
    }
    streams = {
        0.0: _rollout_events(luminance, timestamps, masks, prism, rpms[0.0], args.device),
        180.0: _rollout_events(luminance, timestamps, masks, prism, rpms[180.0], args.device),
        "adaptive": _rollout_events(luminance, timestamps, masks, prism, rpms["adaptive"], args.device),
    }

    occ = {
        name: np.asarray([row["unique_pixel_ratio"] for row in rows], dtype=np.float64)
        for name, rows in (("fixed_0", fixed[0.0]), ("fixed_180", fixed[180.0]), ("fixed_360", fixed[360.0]), ("adaptive", adaptive))
    }
    t = timestamps[:-1]
    excess = occ["fixed_180"] - occ["adaptive"]
    hero = int(np.argmax(excess))
    rgb = np.clip(frames[hero] * 255.0 if frames[hero].max() <= 1.0 else frames[hero], 0, 255).astype(np.uint8)
    if rgb.ndim == 2:
        rgb = np.stack([rgb] * 3, axis=-1)

    maps = {
        "Fixed 180": _upscale(_stack_map(streams[180.0], hero, height, width)),
        "Adaptive": _upscale(_stack_map(streams["adaptive"], hero, height, width)),
        "Fixed 0": _upscale(_stack_map(streams[0.0], hero, height, width)),
    }
    rgb_show = _upscale(rgb)

    fig = plt.figure(figsize=(8.8, 6.35), dpi=200)
    gs = fig.add_gridspec(
        3,
        3,
        height_ratios=[1.55, 1.05, 0.95],
        hspace=0.28,
        wspace=0.12,
        left=0.075,
        right=0.91,
        top=0.91,
        bottom=0.08,
    )

    panel_spec = (
        ("Fixed 180 RPM", maps["Fixed 180"], f"occupancy {occ['fixed_180'][hero]:.2f}", "#1B4F8A", "background event storm"),
        ("Adaptive", maps["Adaptive"], f"occupancy {occ['adaptive'][hero]:.2f}  ·  {rpms['adaptive'][hero]:.0f} RPM", "#15803d", "storm suppressed"),
        ("Fixed 0 RPM", maps["Fixed 0"], f"occupancy {occ['fixed_0'][hero]:.2f}", "#6b7280", "object motion only"),
    )
    for col, (title, image, xlabel, color, badge) in enumerate(panel_spec):
        ax = fig.add_subplot(gs[0, col])
        ax.imshow(image)
        ax.set_title(title, fontsize=10, pad=4, color=color, fontweight="semibold")
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_color(color)
            spine.set_linewidth(1.6)
        ax.set_xlabel(xlabel, fontsize=8, color=color)
        ax.text(
            0.03,
            0.97,
            badge,
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=7.5,
            color="white",
            bbox={"boxstyle": "round,pad=0.25", "facecolor": color, "edgecolor": "none", "alpha": 0.92},
        )
        if col == 0:
            inset = ax.inset_axes([0.68, 0.68, 0.30, 0.30])
            inset.imshow(rgb_show)
            inset.set_xticks([])
            inset.set_yticks([])
            inset.set_title("RGB", fontsize=6, pad=1, color="white")
            for spine in inset.spines.values():
                spine.set_color("white")
                spine.set_linewidth(0.8)

    ax_o = fig.add_subplot(gs[1, :])
    ax_r = fig.add_subplot(gs[2, :], sharex=ax_o)
    fast = speed >= np.median(speed)
    in_fast = False
    start = 0.0
    labeled = False
    for index in range(len(t)):
        if fast[index] and not in_fast:
            start = float(t[index])
            in_fast = True
        elif in_fast and (not fast[index] or index == len(t) - 1):
            end = float(t[index] if not fast[index] else t[index] + np.diff(timestamps)[-1])
            ax_o.axvspan(start, end, color="#fde68a", alpha=0.38, lw=0, zorder=0, label="faster object motion" if not labeled else None)
            ax_r.axvspan(start, end, color="#fde68a", alpha=0.38, lw=0, zorder=0)
            labeled = True
            in_fast = False
    ax_o.plot(t, occ["fixed_0"], color="#6b7280", lw=1.4, label="Fixed 0")
    ax_o.plot(t, occ["fixed_180"], color="#1B4F8A", lw=2.0, label="Fixed 180")
    ax_o.plot(t, occ["fixed_360"], color="#b45309", lw=1.3, alpha=0.85, label="Fixed 360")
    ax_o.plot(t, occ["adaptive"], color="#15803d", lw=2.0, label="Adaptive")
    ax_o.axvline(t[hero], color="#111827", ls="--", lw=0.9)
    ax_o.set_ylabel("Event occupancy")
    ax_o.set_ylim(0, min(1.0, max(occ["fixed_360"].max(), occ["fixed_180"].max()) * 1.22))
    ax_o.legend(frameon=False, ncol=5, loc="upper left", fontsize=8)
    ax_o.grid(True, alpha=0.25, lw=0.5)
    ax_o.spines["top"].set_visible(False)
    ax_o.spines["right"].set_visible(False)
    ax_o.set_title("Occupancy vs time  ·  yellow = faster object motion", fontsize=9, loc="left", pad=3)

    ax_r.step(t, rpms["adaptive"], where="post", color="#15803d", lw=1.9, label="Adaptive RPM")
    ax_r.axhline(180.0, color="#1B4F8A", ls="--", lw=1.1, label="Fixed 180")
    ax_r.axvline(t[hero], color="#111827", ls="--", lw=0.9)
    ax_r.set_ylim(-25, 400)
    ax_r.set_yticks([0, 180, 360])
    ax_r.set_ylabel("RPM")
    ax_r.set_xlabel("Time (s)")
    ax_speed = ax_r.twinx()
    ax_speed.plot(t, speed, color="#6b7280", lw=1.25, label="object image speed")
    ax_speed.set_ylabel("Object speed (px/s)")
    ax_r.grid(True, alpha=0.25, lw=0.5)
    ax_r.spines["top"].set_visible(False)
    handles_r, labels_r = ax_r.get_legend_handles_labels()
    handles_s, labels_s = ax_speed.get_legend_handles_labels()
    ax_r.legend(handles_r + handles_s, labels_r + labels_s, frameon=False, ncol=3, loc="upper left", fontsize=8)

    fig.suptitle("Fixed 180 keeps exciting the static background; the causal policy downshifts through accel / decel", fontsize=10.5, y=0.98)

    stats = {
        "scene": meta["scene"],
        "hero_window": hero,
        "hero_t": float(t[hero]),
        "hero_speed_px_s": float(speed[hero]),
        "occ_0": float(occ["fixed_0"][hero]),
        "occ_180": float(occ["fixed_180"][hero]),
        "occ_360": float(occ["fixed_360"][hero]),
        "occ_adaptive": float(occ["adaptive"][hero]),
        "adaptive_rpm_hero": float(rpms["adaptive"][hero]),
        "mean_occ_180": float(occ["fixed_180"].mean()),
        "mean_occ_adaptive": float(occ["adaptive"].mean()),
        "mean_occ_0": float(occ["fixed_0"].mean()),
        "mean_adaptive_rpm": float(rpms["adaptive"].mean()),
        "note": "object_only switching; camera still; extra occupancy of 180/360 is prism events on the static background",
    }
    fig_dir = args.output_root / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    out = fig_dir / "fig_fixed180_storm_vs_adaptive"
    fig.savefig(out.with_suffix(".pdf"))
    fig.savefig(out.with_suffix(".png"))
    plt.close(fig)
    (args.output_root / "storm_switching_stats.json").write_text(json.dumps(stats, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"figure": str(out) + ".png", **stats}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
