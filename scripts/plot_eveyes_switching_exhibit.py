#!/usr/bin/env python3
"""Paper controller exhibit: three locked regimes plus one-take freeze.

Uses the frozen m0 checkpoint traces. 0.5 s RPM bins; do not plot the raw
16 ms square wave. Caption as conditioning, not an F1 win.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

import matplotlib

from scripts.plot_eveyes_drive_stop_drive import _timestamps, _visual_motion

matplotlib.use("Agg")

READY = Path("data/eveyes_comotion_v2_ready/test/Scene_005")
ORACLE = Path("results/eveyes_v2_regime_f1_oracles")
FREEZE = (2.5, 5.5)
FRAME_TIMES = (0.0, 1.5, 4.0, 7.5)
DSD_THUMBS = (0.0, 2.5, 3.25, 4.75, 7.98)


def _rows(path: Path) -> list[dict[str, object]]:
    return json.loads(path.read_text(encoding="utf-8"))["rows"]


def _series(rows: list[dict[str, object]], scene: str, method: str, key: str) -> np.ndarray:
    selected = [row for row in rows if row["scene"] == scene and row["method"] == method]
    selected.sort(key=lambda row: int(row["window"]))
    return np.asarray([float(row[key]) for row in selected], dtype=np.float64)


def _bin(values: np.ndarray, width: int = 30) -> tuple[np.ndarray, np.ndarray]:
    centers = []
    means = []
    for start in range(0, len(values), width):
        chunk = values[start : start + width]
        centers.append((start + 0.5 * len(chunk)) / 60.0)
        means.append(float(chunk.mean()))
    return np.asarray(centers), np.asarray(means)


def _strip(regime: str, times: tuple[float, ...]) -> tuple[np.ndarray, list[np.ndarray]]:
    images = []
    for t in times:
        frame = min(int(round(t * 60.0)), 479)
        images.append(np.asarray(Image.open(READY / regime / "images" / f"{frame:06d}.png").convert("RGB")))
    return np.concatenate(images, axis=1), images


def _rpm_axis(axes, rpm: np.ndarray, visual: np.ndarray, *, teacher: np.ndarray | None, freeze: bool):
    t = np.arange(len(rpm)) / 60.0
    t_bin, rpm_bin = _bin(rpm)
    _, vis_bin = _bin(visual[: len(rpm)])
    if freeze:
        axes.axvspan(FREEZE[0], FREEZE[1], color="0.92", lw=0, zorder=0)
        axes.axvline(FREEZE[0], color="0.45", lw=0.8, ls="--")
        axes.axvline(FREEZE[1], color="0.45", lw=0.8, ls="--")
        axes.text((FREEZE[0] + FREEZE[1]) * 0.5, 365, "camera frozen", ha="center", va="top", fontsize=7, color="0.35")
    axes.fill_between(t, 180.0, rpm, step="pre", color="0.85", linewidth=0)
    if teacher is not None:
        _, teacher_bin = _bin(teacher[: len(rpm)])
        axes.plot(t_bin, teacher_bin, color="0.45", lw=1.2, ls=":", label="F1 teacher (0.5 s)")
    axes.plot(t_bin, rpm_bin, color="#1d4ed8", lw=2.0, label="Ours RPM (0.5 s)")
    axes.set_ylim(150, 390)
    axes.set_xlim(0, 8)
    axes.set_ylabel("RPM")
    axes.set_yticks([180, 270, 360])
    axes.axhline(180, color="0.6", lw=0.6, ls=":")
    axes.axhline(360, color="0.6", lw=0.6, ls=":")
    vis_ax = axes.twinx()
    vis_ax.plot(t_bin, vis_bin, color="#b45309", lw=1.1, ls="--", label="visual motion")
    vis_ax.set_ylabel("visual")
    vis_ax.set_ylim(-2, 90)
    return vis_ax


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--regimes-json", type=Path, default=Path("results/ami_next/eveyes_v2_regimes_test_ours_m0.json"))
    parser.add_argument("--dsd-json", type=Path, default=Path("results/ami_next/eveyes_v2_drive_stop_drive_ours_m0.json"))
    parser.add_argument("--output", type=Path, default=Path("results/ami_next/figures/eveyes_switching_exhibit"))
    args = parser.parse_args()
    regimes_rows = _rows(args.regimes_json)
    dsd_rows = _rows(args.dsd_json)
    scene_dsd = "test/Scene_005/drive_stop_drive"
    rpm_dsd = _series(dsd_rows, scene_dsd, "ours", "rpm")
    f180 = _series(dsd_rows, scene_dsd, "ami_fixed_180", "edge_f1")
    f360 = _series(dsd_rows, scene_dsd, "ami_fixed_360", "edge_f1")
    teacher = np.where(f360 + 1e-12 >= f180, 360.0, 180.0)
    visual_path = READY / "drive_stop_drive"
    times = _timestamps(visual_path / "timestamps.csv")
    visual_dsd = _visual_motion(visual_path / "images", times, min(len(times), 480))[: len(rpm_dsd)]

    fig = plt.figure(figsize=(7.4, 9.6), dpi=160)
    outer = fig.add_gridspec(4, 1, height_ratios=[1.15, 1, 1, 1], hspace=0.38)

    panels = [
        ("drive_stop_drive", "one take: drive / freeze / drive", DSD_THUMBS, rpm_dsd, visual_dsd, teacher, True, dsd_rows),
        ("object_only", "object only (camera frozen)", FRAME_TIMES, None, None, None, False, regimes_rows),
        ("camera_only", "camera only (parked neighbor)", FRAME_TIMES, None, None, None, False, regimes_rows),
        ("co_motion", "co-motion (locked relative pose)", FRAME_TIMES, None, None, None, False, regimes_rows),
    ]

    for row, (regime, title, thumbs_t, rpm, visual, teacher_rpm, freeze, rows) in enumerate(panels):
        block = outer[row].subgridspec(1, 2, width_ratios=[1.2, 1.65], wspace=0.16)
        thumbs = fig.add_subplot(block[0, 0])
        axes = fig.add_subplot(block[0, 1])
        strip, images = _strip(regime, thumbs_t)
        thumbs.imshow(strip)
        thumbs.set_axis_off()
        letter = chr(ord("a") + row)
        thumbs.set_title(f"({letter})  {title}", loc="left", fontsize=9, pad=4)
        for index, t in enumerate(thumbs_t):
            thumbs.text((index + 0.5) * images[0].shape[1], images[0].shape[0] + 16, f"{t:.2f}s", ha="center", va="top", fontsize=6.5, clip_on=False)
        if rpm is None:
            scene = f"test/Scene_005/{regime}"
            rpm = _series(rows, scene, "ours", "rpm")
            visual = np.load(ORACLE / f"test_Scene_005_{regime}.npz")["visual_motion"]
            teacher_rpm = None
        vis_ax = _rpm_axis(axes, rpm, visual, teacher=teacher_rpm, freeze=freeze)
        if row == 0:
            lines, labels = axes.get_legend_handles_labels()
            lines2, labels2 = vis_ax.get_legend_handles_labels()
            axes.legend(lines + lines2, labels + labels2, loc="upper right", fontsize=6.5, frameon=False)
        if row == 3:
            axes.set_xlabel("time (s)")

    fig.suptitle("Causal prism speed is conditioned on visual motion, not a fixed gear", fontsize=11, y=0.995)
    fig.subplots_adjust(left=0.07, right=0.90, top=0.97, bottom=0.04)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    pdf = args.output.with_suffix(".pdf")
    png = args.output.with_suffix(".png")
    fig.savefig(pdf)
    fig.savefig(png)
    plt.close(fig)
    freeze = rpm_dsd[150:330]
    print(json.dumps({
        "pdf": str(pdf),
        "png": str(png),
        "checkpoint": "results/eveyes_v2_regime_f1_argmax_m0/policy.pt",
        "dsd_freeze_mean_rpm": float(freeze.mean()),
        "dsd_freeze_frac_360": float(np.mean(freeze >= 359.0)),
        "dsd_teacher_freeze_frac_360": float(np.mean(teacher[150:330] >= 359.0)),
        "not_plus_dsd": True,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
