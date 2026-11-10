#!/usr/bin/env python3
"""ICLR/CVPR-style 2x4 motion-dependent optical excitation figure."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


COLORS = {"fixed_0": "#6b7280", "fixed_180": "#2563a6", "fixed_360": "#c96a1b"}
CONDITIONS = (
    ("camera", "Camera-only", "motion_scale"),
    ("object", "Object-only", "motion_scale"),
    ("co_motion", "Co-motion", "motion_scale"),
    ("counter", "Counter-motion", "motion_scale"),
)


def _style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.5,
            "axes.titlesize": 9.0,
            "axes.labelsize": 8.5,
            "xtick.labelsize": 7.3,
            "ytick.labelsize": 7.3,
            "legend.fontsize": 8.0,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.facecolor": "white",
        }
    )


def _records(path: Path) -> list[dict]:
    return json.loads(path.read_text(encoding="utf-8"))


def _aggregate(records: list[dict], sweep: str) -> list[dict]:
    groups: dict[float, list[dict]] = defaultdict(list)
    for record in records:
        if sweep == "co_motion":
            # The mismatch sweep fixes the camera scale and varies object_alpha.
            # Use relative-motion magnitude as the plotted motion scale.
            x_value = abs(1.0 - float(record["object_alpha"]))
        else:
            x_value = float(record["motion_scale"])
        groups[x_value].append(record)
    output = []
    for x, rows in sorted(groups.items()):
        point = {"x": x, "n_seeds": len(rows), "methods": {}}
        for method in ("fixed_0", "fixed_180", "fixed_360", "adaptive", "oracle_f1"):
            if method not in rows[0]["methods"]:
                continue
            fields = rows[0]["methods"][method]
            point["methods"][method] = {
                field: float(np.mean([float(row["methods"][method][field]) for row in rows]))
                for field, value in fields.items()
                if isinstance(value, (int, float))
            }
        output.append(point)
    return output


def _modal_action(method: dict[str, float]) -> float:
    fractions = (method.get("frac_0", 0.0), method.get("frac_180", 0.0), method.get("frac_360", 0.0))
    return float((0.0, 180.0, 360.0)[int(np.argmax(fractions))])


def _line(ax, x, y, color, marker="o", linestyle="-", label=None) -> None:
    ax.plot(x, y, color=color, marker=marker, markersize=3.6, linewidth=1.45, linestyle=linestyle, label=label)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, default=Path("results/ami_next/motion_excitation_sweep/scenes.json"))
    parser.add_argument("--counter", type=Path, default=Path("results/ami_next/counter_motion_excitation_sweep/scenes.json"))
    parser.add_argument("--output", type=Path, default=Path("results/ami_next/figures/fig_motion_dependent_optical_excitation_2x4"))
    args = parser.parse_args()
    _style()
    base = _records(args.base)
    counter = _records(args.counter)
    data = {
        "camera": _aggregate([row for row in base if row.get("sweep") == "camera"], "camera"),
        "object": _aggregate([row for row in base if row.get("sweep") == "object"], "object"),
        "co_motion": _aggregate([row for row in base if row.get("sweep") == "mismatch"], "co_motion"),
        "counter": _aggregate(counter, "counter"),
    }
    activity_values = []
    for key, _, _ in CONDITIONS:
        for point in data[key]:
            for method in ("fixed_180", "fixed_360"):
                activity_values.append(point["methods"][method]["event_count"])
    activity_denominator = max(activity_values)

    fig, axes = plt.subplots(2, 4, figsize=(13.2, 5.7), dpi=260, sharex=False)
    fig.patch.set_facecolor("white")
    fig.subplots_adjust(left=0.075, right=0.995, top=0.84, bottom=0.20, wspace=0.23, hspace=0.45)

    for col, (key, title, _) in enumerate(CONDITIONS):
        points = data[key]
        x = np.asarray([point["x"] for point in points], dtype=float)

        # Observation quality row.
        ax = axes[0, col]
        for method, color in (("fixed_0", COLORS["fixed_0"]), ("fixed_180", COLORS["fixed_180"]), ("fixed_360", COLORS["fixed_360"])):
            y = np.asarray([point["methods"][method]["teo"] for point in points], dtype=float)
            _line(ax, x, y, color, label={"fixed_0": "0 RPM", "fixed_180": "180 RPM", "fixed_360": "360 RPM"}[method])
        ax.set_title(title, fontweight="semibold", color="#1f2937", pad=5)
        ax.set_ylim(-0.03, 0.98)
        ax.set_xlim(float(x.min()), float(x.max()))
        ax.grid(axis="y", color="#d1d5db", alpha=0.55, linewidth=0.55)
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
        if col == 0:
            ax.set_ylabel("TEO $\\uparrow$")
        else:
            ax.set_yticklabels([])
        ax.set_xticklabels([])

        # Activity row.
        ax = axes[1, col]
        for method, color in (("fixed_0", COLORS["fixed_0"]), ("fixed_180", COLORS["fixed_180"]), ("fixed_360", COLORS["fixed_360"])):
            y = np.asarray([point["methods"][method]["event_count"] / activity_denominator for point in points], dtype=float)
            _line(ax, x, y, color, label=method)
        ax.set_ylim(-0.02, 1.05)
        ax.set_xlim(float(x.min()), float(x.max()))
        ax.grid(axis="y", color="#d1d5db", alpha=0.55, linewidth=0.55)
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
        if col == 0:
            ax.set_ylabel("Event activity $\\uparrow$")
        else:
            ax.set_yticklabels([])
        ax.set_xlabel("Motion scale")
        if key == "co_motion":
            ax.set_xticks([0, 1, 2, 3, 4, 6])

    fig.text(0.012, 0.675, "(a) Observation quality", rotation=90, ha="center", va="center", fontsize=10.5, fontweight="semibold", color="#111827")
    fig.text(0.012, 0.365, "(b) Event activity", rotation=90, ha="center", va="center", fontsize=10.5, fontweight="semibold", color="#111827")
    fig.suptitle("Motion-dependent optical excitation", fontsize=14, fontweight="semibold", color="#111827", y=0.965)
    fig.text(0.5, 0.925, "Optical excitation changes both target observability and event activity across relative-motion regimes.", ha="center", va="center", fontsize=8.4, color="#4b5563")

    quality_handles = [
        plt.Line2D([0], [0], color=COLORS["fixed_0"], marker="o", lw=1.5, markersize=4, label="0 RPM"),
        plt.Line2D([0], [0], color=COLORS["fixed_180"], marker="o", lw=1.5, markersize=4, label="180 RPM"),
        plt.Line2D([0], [0], color=COLORS["fixed_360"], marker="o", lw=1.5, markersize=4, label="360 RPM"),
    ]
    activity_handles = [
        plt.Line2D([0], [0], color=COLORS["fixed_0"], marker="o", lw=1.5, markersize=4, label="0 RPM"),
        plt.Line2D([0], [0], color=COLORS["fixed_180"], marker="o", lw=1.5, markersize=4, label="180 RPM"),
        plt.Line2D([0], [0], color=COLORS["fixed_360"], marker="o", lw=1.5, markersize=4, label="360 RPM"),
    ]
    fig.legend(handles=quality_handles, loc="lower center", bbox_to_anchor=(0.27, 0.075), ncol=3, frameon=False, title="Observation quality", title_fontsize=8)
    fig.legend(handles=activity_handles, loc="lower center", bbox_to_anchor=(0.73, 0.075), ncol=3, frameon=False, title="Event activity", title_fontsize=8)
    fig.text(0.5, 0.025, "Activity is event count normalized by the global maximum across fixed 180/360 curves and all four regimes.", ha="center", va="center", fontsize=7.2, color="#6b7280")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    fig.savefig(args.output.with_suffix(".png"), dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    curve_metadata = {}
    for key, title, _ in CONDITIONS:
        points = data[key]
        curve_metadata[key] = {
            "title": title,
            "motion_scale": [float(point["x"]) for point in points],
            "n_seeds_per_point": [int(point["n_seeds"]) for point in points],
            "teo": {
                method: [float(point["methods"][method]["teo"]) for point in points]
                for method in ("fixed_0", "fixed_180", "fixed_360")
            },
            "event_count": {
                method: [float(point["methods"][method]["event_count"]) for point in points]
                for method in ("fixed_0", "fixed_180", "fixed_360")
            },
            "event_activity_normalized": {
                method: [float(point["methods"][method]["event_count"] / activity_denominator) for point in points]
                for method in ("fixed_0", "fixed_180", "fixed_360")
            },
        }
    meta = {
        "source_base": str(args.base),
        "source_counter": str(args.counter),
        "output_pdf": str(args.output.with_suffix(".pdf")),
        "output_png": str(args.output.with_suffix(".png")),
        "conditions": [title for _, title, _ in CONDITIONS],
        "quality_metric": "TEO",
        "activity_metric": "event_count normalized by global fixed-180/fixed-360 maximum; plotted for 0/180/360 RPM",
        "action_panel": "not included in this 2x4 figure; see fig_motion_dependent_optical_excitation.pdf for oracle/EIRM action analysis",
        "counter_motion_protocol": "same procedural renderer, 3 seeds, 11 motion scales, counter_motion object warp = -0.8 * camera warp",
        "colors": {"0_RPM": COLORS["fixed_0"], "180_RPM": COLORS["fixed_180"], "360_RPM": COLORS["fixed_360"]},
        "axis_limits": {"teo": [-0.03, 0.98], "event_activity_normalized": [-0.02, 1.05]},
        "curves": curve_metadata,
    }
    args.output.with_name(args.output.name + "_meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(meta, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
