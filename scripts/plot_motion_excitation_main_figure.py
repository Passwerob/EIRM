#!/usr/bin/env python3
"""Paper-style 3x3 grouped figure for motion-dependent excitation.

The source is the real multi-seed motion-excitation sweep. The figure has
three grouped panels (TEO, activity, preferred action), each with camera,
object, and mismatch condition subplots.
"""

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
LABELS = {"fixed_0": "0 RPM", "fixed_180": "180 RPM", "fixed_360": "360 RPM"}
CONDITIONS = (
    ("camera", "Camera motion", "motion_scale"),
    ("object", "Object motion", "motion_scale"),
    ("mismatch", "Co-motion mismatch", "object_alpha"),
)


def _style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.5,
            "axes.titlesize": 9.0,
            "axes.labelsize": 8.5,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "legend.fontsize": 8.0,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.facecolor": "white",
        }
    )


def _load_points(path: Path) -> dict[str, list[dict]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    grouped: dict[str, dict[float, list[dict]]] = defaultdict(lambda: defaultdict(list))
    for row in raw:
        sweep = str(row["sweep"])
        x_key = "object_alpha" if sweep == "mismatch" else "motion_scale"
        grouped[sweep][float(row[x_key])].append(row)
    output: dict[str, list[dict]] = {}
    for sweep, by_x in grouped.items():
        points = []
        for x, rows in sorted(by_x.items()):
            point = {"x": x, "n_seeds": len(rows), "methods": {}}
            for method in ("fixed_0", "fixed_180", "fixed_360", "adaptive", "oracle_f1"):
                if method not in rows[0]["methods"]:
                    continue
                fields = rows[0]["methods"][method].keys()
                point["methods"][method] = {
                    field: float(np.mean([float(row["methods"][method][field]) for row in rows]))
                    for field in fields
                    if isinstance(rows[0]["methods"][method][field], (float, int))
                }
            points.append(point)
        output[sweep] = points
    return output


def _dominant_action(method: dict[str, float]) -> float:
    fractions = [method.get("frac_0", 0.0), method.get("frac_180", 0.0), method.get("frac_360", 0.0)]
    return float((0.0, 180.0, 360.0)[int(np.argmax(fractions))])


def _panel_title(ax, title: str) -> None:
    ax.set_title(title, loc="left", pad=5, fontweight="semibold", color="#1f2937")
    ax.grid(axis="y", color="#d1d5db", alpha=0.55, linewidth=0.55)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#6b7280")
    ax.spines["bottom"].set_color("#6b7280")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("results/ami_next/motion_excitation_sweep/scenes.json"))
    parser.add_argument("--output", type=Path, default=Path("results/ami_next/figures/fig_motion_dependent_optical_excitation"))
    args = parser.parse_args()
    _style()
    points = _load_points(args.input)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    all_activity = []
    for sweep, _, _ in CONDITIONS:
        for point in points[sweep]:
            for method in ("fixed_180", "fixed_360"):
                all_activity.append(point["methods"][method]["event_count"])
    activity_scale = max(all_activity)

    fig = plt.figure(figsize=(13.2, 5.7), dpi=240, facecolor="white")
    outer = fig.add_gridspec(1, 3, width_ratios=(1, 1, 1), wspace=0.26, left=0.055, right=0.985, top=0.85, bottom=0.19)
    panel_titles = ["(a) TEO vs. motion scale", "(b) Event activity vs. motion scale", "(c) Preferred action"]
    panel_axes = []

    for panel_index, ((sweep, _, x_key), title) in enumerate(zip(CONDITIONS, panel_titles)):
        inner = outer[panel_index].subgridspec(3, 1, hspace=0.55)
        axes = [fig.add_subplot(inner[row, 0]) for row in range(3)]
        panel_axes.append(axes)
        for row, (condition_name, _, _) in enumerate(CONDITIONS):
            points_row = points[condition_name]
            x = np.asarray([p["x"] for p in points_row], dtype=float)
            ax = axes[row]
            ax.set_facecolor("#ffffff")
            if panel_index == 0:
                for method in ("fixed_0", "fixed_180", "fixed_360"):
                    y = np.asarray([p["methods"][method]["teo"] for p in points_row], dtype=float)
                    ax.plot(x, y, color=COLORS[method], marker="o", markersize=3.3, linewidth=1.45, label=LABELS[method])
                ax.set_ylim(-0.03, 0.98)
                ax.set_ylabel("TEO $\\uparrow$" if row == 1 else "")
                if row == 0:
                    ax.set_ylabel("TEO $\\uparrow$")
                if row == 2:
                    ax.set_xlabel("Motion scale" if condition_name != "mismatch" else r"Mismatch $\alpha$")
                _panel_title(ax, condition_name.replace("_", " ").title())
            elif panel_index == 1:
                for method in ("fixed_180", "fixed_360"):
                    y = np.asarray([p["methods"][method]["event_count"] / activity_scale for p in points_row], dtype=float)
                    ax.plot(x, y, color=COLORS[method], marker="o", markersize=3.3, linewidth=1.45, label=LABELS[method])
                ax.set_ylim(-0.02, 1.05)
                if row == 0:
                    ax.set_ylabel("Activity (norm.) $\\uparrow$")
                if row == 2:
                    ax.set_xlabel("Motion scale" if condition_name != "mismatch" else r"Mismatch $\alpha$")
                _panel_title(ax, condition_name.replace("_", " ").title())
            else:
                oracle = np.asarray([_dominant_action(p["methods"]["oracle_f1"]) for p in points_row], dtype=float)
                eirm = np.asarray([_dominant_action(p["methods"]["adaptive"]) for p in points_row], dtype=float)
                order = np.argsort(x)
                ax.step(x[order], oracle[order], where="mid", color="#6b7280", linestyle=(0, (3, 2)), linewidth=1.55, label="Oracle")
                ax.step(x[order], eirm[order], where="mid", color="#2563a6", linewidth=1.75, label="EIRM")
                ax.scatter(x, oracle, color="#6b7280", s=16, zorder=3)
                ax.scatter(x, eirm, color="#2563a6", s=16, zorder=3)
                ax.set_ylim(-18, 378)
                ax.set_yticks([0, 180, 360])
                if row == 0:
                    ax.set_ylabel("Selected RPM")
                if row == 2:
                    ax.set_xlabel("Motion scale" if condition_name != "mismatch" else r"Mismatch $\alpha$")
                _panel_title(ax, condition_name.replace("_", " ").title())
            ax.set_xlim(float(x.min()), float(x.max()))
            if row < 2:
                ax.set_xticklabels([])
            if condition_name == "mismatch":
                ax.set_xticks([0.0, 0.5, 1.0, 1.5])

        axes[0].text(-0.18, 1.22, title, transform=axes[0].transAxes, fontsize=11, fontweight="semibold", color="#111827", va="bottom")

    handles = [plt.Line2D([0], [0], color=COLORS[m], marker="o", lw=1.6, markersize=4, label=LABELS[m]) for m in ("fixed_0", "fixed_180", "fixed_360")]
    fig.legend(handles=handles, loc="lower left", bbox_to_anchor=(0.06, 0.055), ncol=3, frameon=False, title="Optical excitation", title_fontsize=8)
    action_handles = [
        plt.Line2D([0], [0], color="#6b7280", linestyle=(0, (3, 2)), lw=1.6, label="Oracle"),
        plt.Line2D([0], [0], color="#2563a6", lw=1.8, label="EIRM"),
    ]
    fig.legend(handles=action_handles, loc="lower right", bbox_to_anchor=(0.985, 0.055), ncol=2, frameon=False, title="Preferred action", title_fontsize=8)
    fig.text(0.5, 0.012, r"Motion scale is the commanded image-plane scale for camera/object sweeps and $\alpha$ for camera/object mismatch. Activity is event count normalized by the global 180/360 maximum.", ha="center", va="bottom", fontsize=7.4, color="#4b5563")

    fig.savefig(args.output.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    fig.savefig(args.output.with_suffix(".png"), dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    manifest = {
        "source": str(args.input),
        "output_pdf": str(args.output.with_suffix(".pdf")),
        "output_png": str(args.output.with_suffix(".png")),
        "conditions": [item[1] for item in CONDITIONS],
        "panel_metrics": ["TEO", "event_count_normalized_global_180_360_max", "oracle_f1_vs_adaptive_modal_action"],
        "activity_normalization": {"denominator": "maximum event_count across all fixed 180/360 curves and all plotted conditions", "value": activity_scale},
        "action_definition": "Oracle=modal oracle_f1 gear; EIRM=modal adaptive action fraction at each motion scale",
        "source_note": "The sweep is a controlled procedural benchmark; trends are intended for qualitative motion-dependent analysis, not hardware calibration.",
    }
    args.output.with_name(args.output.name + "_meta.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
