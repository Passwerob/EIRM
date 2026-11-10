#!/usr/bin/env python3
"""Show the 0/180/360 gear split by motion regime.

This is the qualitative 3-gear figure for the procedural pack. Do not use it
as an EVEYES Table 1 number.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


CONDITIONS = (
    ("camera_only", "camera only"),
    ("object_only", "object only"),
    ("co_motion", "co-motion"),
    ("counter_motion", "counter-motion"),
)
GEARS = (0.0, 180.0, 360.0)
COLORS = ("#6b7280", "#1B4F8A", "#b45309")


def _hist(rows: list[dict[str, object]], policy: str, condition: str) -> np.ndarray:
    selected = [row for row in rows if row["policy"] == policy and row["condition"] == condition]
    if not selected:
        return np.zeros(3, dtype=np.float64)
    return np.asarray([np.mean([float(row["rpm_hist"].get(str(rpm), 0.0)) for row in selected]) for rpm in GEARS], dtype=np.float64)


def _trace(rows: list[dict[str, object]], policy: str, condition: str) -> np.ndarray | None:
    selected = [row for row in rows if row["policy"] == policy and row["condition"] == condition and "rpms" in row]
    if not selected:
        return None
    selected.sort(key=lambda row: (int(row.get("seed", 0)), str(row["scene"])))
    return np.asarray(selected[0]["rpms"], dtype=np.float64)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-json", type=Path, default=Path("results/comotion_f1_argmax_0_180_360/test.json"))
    parser.add_argument("--output", type=Path, default=Path("results/comotion_f1_argmax_0_180_360/figures/three_gear_by_regime"))
    args = parser.parse_args()
    payload = json.loads(args.eval_json.read_text(encoding="utf-8"))
    rows = payload["rows"]
    args.output.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 4, figsize=(11.2, 5.4), dpi=160, gridspec_kw={"height_ratios": [1.15, 1.0]})
    x = np.arange(3)
    width = 0.36
    for col, (condition, title) in enumerate(CONDITIONS):
        ax = axes[0, col]
        learned = _hist(rows, "inverse_learned", condition)
        ax.bar(x, learned, color=COLORS, width=0.72, edgecolor="white", linewidth=0.4)
        ax.set_ylim(0, 1.05)
        ax.set_xticks(x)
        ax.set_xticklabels(["0", "180", "360"], fontsize=8)
        ax.set_title(title, fontsize=9, pad=4)
        if col == 0:
            ax.set_ylabel("fraction of windows", fontsize=8)
        else:
            ax.set_yticklabels([])
        ax.tick_params(length=0)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)

        ax_t = axes[1, col]
        rpm = _trace(rows, "inverse_learned", condition)
        if rpm is None:
            ax_t.set_axis_off()
            continue
        t = np.arange(len(rpm))
        ax_t.step(t, rpm, where="post", color="#1B4F8A", lw=1.4)
        ax_t.set_ylim(-20, 400)
        ax_t.set_yticks([0, 180, 360])
        ax_t.set_xlim(0, max(len(rpm) - 1, 1))
        ax_t.axhline(0, color="0.8", lw=0.5)
        ax_t.axhline(180, color="0.8", lw=0.5)
        ax_t.axhline(360, color="0.8", lw=0.5)
        if col == 0:
            ax_t.set_ylabel("RPM", fontsize=8)
        else:
            ax_t.set_yticklabels([])
        ax_t.set_xlabel("window", fontsize=8)
        for spine in ("top", "right"):
            ax_t.spines[spine].set_visible(False)

    fig.suptitle("Ours selects 0 / 180 / 360 RPM by motion regime", fontsize=11, y=0.98)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    pdf = args.output.with_suffix(".pdf")
    png = args.output.with_suffix(".png")
    fig.savefig(pdf)
    fig.savefig(png)
    plt.close(fig)
    print(json.dumps({"pdf": str(pdf), "png": str(png)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
