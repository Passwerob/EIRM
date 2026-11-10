#!/usr/bin/env python3
"""Make EIRM-style mechanism figures for 360>180 and 0-RPM storm response."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ami_simulator.event_visualization import eirm_counts_from_events, eirm_from_counts, eirm_peak
from ami_simulator.geometry import RotatingWedgePrism
from ami_simulator.pipeline import load_rgb_sequence, rgb_to_linear_luminance
from scripts.evaluate_dl3dv_downstream import _ami_windows, _dvs_windows
from scripts.evaluate_eveyes_v2_downstream import _intrinsics

METHODS = (("ami_passive", 0.0, "0 RPM", "#374151"), ("ami_fixed_180", 180.0, "180 RPM", "#1677b8"), ("ami_fixed_360", 360.0, "360 RPM", "#d45b0a"))


def _render(positive: np.ndarray, negative: np.ndarray, peak: tuple[float, float]) -> np.ndarray:
    return eirm_from_counts(positive, negative, lam=3.0, gamma=0.65, peak=peak).astype(np.float32) / 255.0


def _rows(payload: dict, method: str, segment: str) -> list[dict]:
    return [row for row in payload["per_gear"][method] if row["segment"] == segment]


def _choose_window(payload: dict, segment: str, criterion: str) -> int:
    rows = {method: _rows(payload, method, segment) for method, _, _, _ in METHODS}
    windows = [int(row["window"]) for row in rows["ami_passive"]]
    if criterion == "360_beats_180":
        candidates = [w for w in windows if next(r for r in rows["ami_fixed_360"] if int(r["window"]) == w)["edge_f1"] > next(r for r in rows["ami_fixed_180"] if int(r["window"]) == w)["edge_f1"]]
    else:
        candidates = windows
    if not candidates:
        raise RuntimeError(f"no candidate windows for {segment}/{criterion}")
    totals = {w: sum(next(r for r in rows[m] if int(r["window"]) == w)["event_count"] for m, _, _, _ in METHODS) for w in candidates}
    median = float(np.median(list(totals.values())))
    return min(candidates, key=lambda w: (abs(totals[w] - median), w))


def _make_row(axs, rgb: np.ndarray, rendered: dict[str, np.ndarray], payload: dict, segment: str, window: int, fps: float, label: str) -> dict:
    axs[0].imshow(np.clip(rgb, 0, 255).astype(np.uint8)); axs[0].set_title("RGB reference", fontsize=11, fontweight="semibold")
    axs[0].set_xticks([]); axs[0].set_yticks([])
    stats = {"segment": segment, "window": window, "fps": fps, "label": label}
    for index, (method, _, method_label, color) in enumerate(METHODS, start=1):
        axs[index].imshow(rendered[method]); row = next(r for r in payload["per_gear"][method] if int(r["window"]) == window)
        axs[index].set_title(method_label, color=color, fontsize=11, fontweight="semibold")
        axs[index].set_xlabel(f"{row['event_count'] / 1e6:.2f}M events\nocc. {row['unique_pixel_ratio']:.3f}", color=color, fontsize=8.5, labelpad=4)
        axs[index].set_xticks([]); axs[index].set_yticks([])
        for spine in axs[index].spines.values(): spine.set_color(color); spine.set_linewidth(1.4)
        stats[method] = {"event_count": int(row["event_count"]), "edge_f1": float(row["edge_f1"]), "object_edge_f1": float(row["object_edge_f1"]), "unique_pixel_ratio": float(row["unique_pixel_ratio"])}
    axs[0].set_ylabel(label, fontsize=10, fontweight="semibold", labelpad=10)
    return stats


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", type=Path, default=Path("data/eveyes_comotion_v2_ready/test/Scene_005/camera_passby_complete"))
    parser.add_argument("--evaluation", type=Path, default=Path("results/ami_next/constructed_eval_camera_passby_complete.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/ami_next/figures/camera_passby_complete_clean"))
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    payload = json.loads(args.evaluation.read_text(encoding="utf-8"))
    frames, timestamps = load_rgb_sequence(args.scene / "images", args.scene / "timestamps.csv", timestamp_unit="s")
    motion = list(csv.DictReader((args.scene / "motion_state.csv").open(encoding="utf-8")))
    fps = np.asarray([float(row["sensor_fps"]) for row in motion])
    boundaries = [0]
    for i in range(1, len(fps)):
        if fps[i] != fps[i - 1]: boundaries.append(i)
    boundaries.append(len(fps) - 1)
    segments = {"normal": ("s0", _choose_window(payload, "s0", "360_beats_180"), float(fps[0]), "480 FPS · 360 beats 180"), "storm": ("s1", _choose_window(payload, "s1", "storm"), float(fps[-1]), "7.5 FPS · 0 suppresses storm")}
    luminance = rgb_to_linear_luminance(frames)
    intrinsics = _intrinsics(args.scene)
    prism = RotatingWedgePrism(intrinsics, 0.5)
    valid = np.ones((len(frames) - 1, intrinsics.height, intrinsics.width), dtype=bool)
    streams = {}
    for method, rpm, _, _ in METHODS:
        if rpm == 0.0:
            streams[method], _ = _dvs_windows(luminance, timestamps)
        else:
            streams[method], _, _ = _ami_windows(luminance, timestamps, valid, prism, rpm, device=args.device)
        print(f"computed {method}", flush=True)
    rendered_rows = {}
    metadata = {"scene": str(args.scene), "evaluation": str(args.evaluation), "display": "repository EIRM palette: background #030712, positive #37DCFF, negative #E148FF; shared per-row log1p(3*count) scale; gamma=0.65", "selection": "normal: median total event count among windows where Fixed360 edge F1 > Fixed180; storm: median total event count in s1", "rows": {}}
    for name, (segment, window, sensor_fps, label) in segments.items():
        maps = {method: eirm_counts_from_events(streams[method][window], intrinsics.height, intrinsics.width) for method, _, _, _ in METHODS}
        peak = (max(eirm_peak(pos, lam=3.0) for pos, _ in maps.values()), max(eirm_peak(neg, lam=3.0) for _, neg in maps.values()))
        rendered = {method: _render(pos, neg, peak) for method, (pos, neg) in maps.items()}
        rendered_rows[name] = (frames[min(window, len(frames) - 1)], rendered)
        metadata["rows"][name] = {
            "segment": segment,
            "window": window,
            "fps": sensor_fps,
            "label": label,
            **{method: {key: float(next(r for r in payload["per_gear"][method] if int(r["window"]) == window)[key]) for key in ("event_count", "edge_f1", "object_edge_f1", "unique_pixel_ratio")} for method, _, _, _ in METHODS},
        }

    # Re-render cleanly because metadata row collection above should not own axes.
    figure, axes = plt.subplots(2, 4, figsize=(16, 8.2), dpi=220, facecolor="white", gridspec_kw={"wspace": 0.025, "hspace": 0.22})
    for row_index, (name, (segment, window, sensor_fps, label)) in enumerate(segments.items()):
        rgb, rendered = rendered_rows[name]
        _make_row(axes[row_index], rgb, rendered, payload, segment, window, sensor_fps, label)
    figure.suptitle("Camera pass-by: 360 beats 180 in the normal clock; 0 suppresses the low-FPS storm", fontsize=14, fontweight="semibold", y=0.98)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    combined = args.output_dir / "fig_camera_passby_mechanism_eirm"
    figure.savefig(combined.with_suffix(".png"), bbox_inches="tight", facecolor="white")
    figure.savefig(combined.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(figure)

    # Also export each row for two-column paper layouts.
    for name, (segment, window, sensor_fps, label) in segments.items():
        rgb, rendered = rendered_rows[name]
        fig, axs = plt.subplots(1, 4, figsize=(16, 4.2), dpi=220, facecolor="white", gridspec_kw={"wspace": 0.025})
        _make_row(axs, rgb, rendered, payload, segment, window, sensor_fps, label)
        fig.suptitle(f"Camera pass-by · {label}", fontsize=13, fontweight="semibold", y=0.98)
        path = args.output_dir / f"fig_camera_passby_{name}_eirm"
        fig.savefig(path.with_suffix(".png"), bbox_inches="tight", facecolor="white")
        fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
        plt.close(fig)
    metadata["files"] = {"combined": str(combined), "normal": str(args.output_dir / "fig_camera_passby_normal_eirm"), "storm": str(args.output_dir / "fig_camera_passby_storm_eirm")}
    (args.output_dir / "fig_camera_passby_mechanism_eirm.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metadata, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
