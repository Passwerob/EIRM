#!/usr/bin/env python3
"""Plot camera-speed NFB: occupancy, path progress, camera rate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _series(rows: list[dict[str, object]], key: str) -> tuple[np.ndarray, np.ndarray]:
    t = np.asarray([float(row["t0"]) for row in rows], dtype=np.float64)
    y = np.asarray([float(row[key]) for row in rows], dtype=np.float64)
    return t, y


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("results/eveyes_motion_nfb/camera_speed_nfb.json"))
    parser.add_argument("--scene", default="test/Scene_005/co_motion")
    parser.add_argument("--output", type=Path, default=Path("results/eveyes_motion_nfb/figures/camera_speed_nfb"))
    args = parser.parse_args()
    report = json.loads(args.input.read_text(encoding="utf-8"))
    block = report["scenes"][args.scene]
    fig, axes = plt.subplots(4, 1, figsize=(7.2, 6.2), sharex=True, dpi=160)
    styles = (
        ("open_loop_camera", "0.45", "open loop"),
        ("rule_camera_nfb", "C1", "rule three-action"),
        ("learned_camera_nfb", "C3", "learned F1-argmax"),
        ("camera_speed_nfb", "C2", "dual NFB (legacy)"),
        ("camera_nfb_keep360", "C0", "camera NFB (keep 360)"),
    )
    for name, color, label in styles:
        if name not in block:
            continue
        rows = block[name]["rows"]
        t, occ = _series(rows, "unique_pixel_ratio")
        _, rate = _series(rows, "camera_rate")
        _, path = _series(rows, "path_s")
        _, rpm = _series(rows, "rpm")
        axes[0].plot(t, occ, color=color, lw=1.2, label=label)
        axes[1].plot(t, rate, color=color, lw=1.4, label=label, drawstyle="steps-post")
        axes[2].plot(t, rpm, color=color, lw=1.4, label=label, drawstyle="steps-post")
        axes[3].plot(t, path, color=color, lw=1.2, label=label)
    axes[0].axhline(report["storm_on"], color="0.7", ls="--", lw=0.8)
    axes[0].set_ylabel("occupancy")
    axes[1].set_ylabel("camera rate")
    axes[1].set_yticks([0.5, 1.0, 1.5])
    axes[2].set_ylabel("prism RPM")
    axes[2].set_yticks([180, 360])
    axes[3].set_ylabel("path index")
    axes[3].set_xlabel("time (s)")
    axes[0].set_title(f"{args.scene}  ·  two-actuator NFB")
    axes[0].legend(frameon=False, fontsize=8)
    for axis in axes:
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(args.output.with_suffix(".pdf"))
    fig.savefig(args.output.with_suffix(".png"), dpi=180)
    plt.close(fig)
    print(args.output.with_suffix(".pdf"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
