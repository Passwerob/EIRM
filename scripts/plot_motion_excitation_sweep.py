#!/usr/bin/env python3
"""Plot motion–excitation response curves from the real sweep JSON.

No smoothing, no invented points. Seed-mean with seed std error bars.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

METHODS = (
    ("fixed_0", "Fixed 0", "#6b7280", "o"),
    ("fixed_180", "Fixed 180", "#1B4F8A", "s"),
    ("fixed_360", "Fixed 360", "#b45309", "D"),
    ("adaptive", "Adaptive", "#15803d", "^"),
    ("oracle_teo", "Oracle (TEO)", "#111827", "x"),
)
F1_METHODS = (
    ("fixed_0", "Fixed 0", "#6b7280", "o"),
    ("fixed_180", "Fixed 180", "#1B4F8A", "s"),
    ("fixed_360", "Fixed 360", "#b45309", "D"),
    ("adaptive", "Adaptive", "#15803d", "^"),
    ("oracle_f1", "Oracle (F1)", "#111827", "x"),
)
METRIC_PANELS = (
    ("teo", "TEO"),
    ("edge_f1", "Global Edge F1"),
    ("edge_recall", "Recall"),
    ("edge_precision", "Precision"),
    ("coverage_C", "Coverage C"),
    ("purity_Pi", "Purity Π"),
    ("tau", "Temporal support τ"),
    ("unique_pixel_ratio", "Event occupancy"),
    ("event_count", "Event count"),
)


def _style() -> None:
    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "legend.fontsize": 8,
            "figure.dpi": 160,
            "savefig.bbox": "tight",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def _rows(scenes: list[dict], sweep: str) -> list[dict]:
    return [row for row in scenes if row.get("sweep") == sweep]


def _group_key(row: dict, sweep: str) -> float:
    if sweep == "mismatch":
        return float(row["object_alpha"])
    if sweep == "grid":
        return (float(row["grid_camera_scale"]), float(row["grid_object_scale"]))
    return float(row["motion_scale"])


def _aggregate(rows: list[dict], sweep: str) -> list[dict]:
    groups: dict[object, list[dict]] = {}
    for row in rows:
        groups.setdefault(_group_key(row, sweep), []).append(row)
    out = []
    for key in sorted(groups, key=lambda item: item if not isinstance(item, tuple) else item):
        items = groups[key]
        entry: dict[str, object] = {
            "sweep": sweep,
            "n_seeds": len(items),
            "motion_scale": None if sweep == "mismatch" else (key[0] if sweep == "grid" else float(key)),
            "object_alpha": float(key) if sweep == "mismatch" else None,
            "grid_camera_scale": key[0] if sweep == "grid" else None,
            "grid_object_scale": key[1] if sweep == "grid" else None,
            "m_nat_flow_px": float(np.mean([float(r["mean_m_nat_flow_px"]) for r in items])),
            "m_nat_flow_px_std": float(np.std([float(r["mean_m_nat_flow_px"]) for r in items])),
            "m_nat_object_flow_px": float(np.mean([float(r["mean_m_nat_object_flow_px"]) for r in items])),
            "m_nat_object_flow_px_std": float(np.std([float(r["mean_m_nat_object_flow_px"]) for r in items])),
            "m_cmd_bg_px_s": float(np.mean([float(r.get("mean_m_cmd_bg_px_s", 0.0)) for r in items])),
            "m_cmd_rel_px_s": float(np.mean([float(r["mean_m_cmd_rel_px_s"]) for r in items])),
            "methods": {},
        }
        method_names = items[0]["methods"].keys()
        for name in method_names:
            stats: dict[str, float] = {}
            for field in items[0]["methods"][name]:
                values = np.asarray([float(r["methods"][name][field]) for r in items], dtype=np.float64)
                stats[field] = float(values.mean())
                stats[f"{field}_std"] = float(values.std(ddof=0))
            entry["methods"][name] = stats
        teo = {rpm: float(entry["methods"][f"fixed_{int(rpm)}"]["teo"]) for rpm in (0, 180, 360)}
        f1 = {rpm: float(entry["methods"][f"fixed_{int(rpm)}"]["edge_f1"]) for rpm in (0, 180, 360)}
        entry["oracle_gear_teo"] = int(max(teo, key=teo.get))
        entry["oracle_gear_f1"] = int(max(f1, key=f1.get))
        for name in ("fixed_0", "fixed_180", "fixed_360", "adaptive"):
            entry["methods"][name]["regret_teo"] = float(entry["methods"]["oracle_teo"]["teo"] - entry["methods"][name]["teo"])
            entry["methods"][name]["regret_f1"] = float(entry["methods"]["oracle_f1"]["edge_f1"] - entry["methods"][name]["edge_f1"])
        out.append(entry)
    return out


def _xs(points: list[dict], sweep: str, which: str) -> np.ndarray:
    if which == "m_nat":
        key = "m_nat_object_flow_px" if sweep in {"object", "mismatch"} else "m_nat_flow_px"
        return np.asarray([float(p[key]) for p in points], dtype=np.float64)
    if which == "cmd":
        key = "m_cmd_rel_px_s" if sweep == "object" else "m_cmd_bg_px_s"
        return np.asarray([float(p[key]) for p in points], dtype=np.float64)
    if which == "alpha":
        return np.asarray([float(p["object_alpha"]) for p in points], dtype=np.float64)
    if which == "scale":
        return np.asarray([float(p["motion_scale"]) for p in points], dtype=np.float64)
    raise ValueError(which)


def _ys(points: list[dict], method: str, metric: str) -> tuple[np.ndarray, np.ndarray]:
    mean = np.asarray([float(p["methods"][method][metric]) for p in points], dtype=np.float64)
    std = np.asarray([float(p["methods"][method].get(f"{metric}_std", 0.0)) for p in points], dtype=np.float64)
    return mean, std


def _line(ax, x, y, std, *, label, color, marker, step=False) -> None:
    order = np.argsort(x)
    x, y, std = x[order], y[order], std[order]
    if step:
        ax.step(x, y, where="mid", color=color, lw=1.8, label=label)
    else:
        ax.plot(x, y, color=color, marker=marker, ms=4.5, lw=1.4, label=label)
        if np.nanmax(std) > 0:
            ax.fill_between(x, y - std, y + std, color=color, alpha=0.12, linewidth=0)


def _save(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path.with_suffix(".pdf"))
    fig.savefig(path.with_suffix(".png"))
    plt.close(fig)


def _response(points_map: dict[str, list[dict]], metric: str, ylabel: str, out: Path, methods=METHODS, x_which="m_nat") -> None:
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.6), sharey=True)
    for ax, sweep, title in (
        (axes[0], "camera", "Sweep A: camera motion (object locked)"),
        (axes[1], "object", "Sweep B: object motion (camera still)"),
    ):
        points = points_map[sweep]
        x = _xs(points, sweep, x_which)
        for method, label, color, marker in methods:
            if method not in points[0]["methods"]:
                continue
            y, std = _ys(points, method, metric)
            _line(ax, x, y, std, label=label, color=color, marker=marker)
        ax.set_title(title)
        ax.set_xlabel("Commanded image-plane speed (px/s)")
        ax.grid(True, alpha=0.25, lw=0.5)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
    axes[0].set_ylabel(ylabel)
    axes[1].legend(frameon=False, loc="best")
    fig.tight_layout()
    _save(fig, out)


def _mismatch(points: list[dict], out: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.6), sharey=True)
    for ax, which, xlabel in (
        (axes[0], "alpha", r"Co-motion coefficient $\alpha=v_\mathrm{object}/v_\mathrm{camera}$"),
        (axes[1], "m_nat", "Measured object-region natural flow (px / frame)"),
    ):
        x = _xs(points, "mismatch", which)
        for method, label, color, marker in METHODS:
            y, std = _ys(points, method, "teo")
            _line(ax, x, y, std, label=label, color=color, marker=marker)
        if which == "alpha":
            ax.axvline(1.0, color="#9ca3af", ls="--", lw=0.9)
            ax.set_title("vs commanded α  (α=1 is image-plane lock)")
        else:
            ax.set_title("vs measured natural object motion")
        ax.set_xlabel(xlabel)
        ax.grid(True, alpha=0.25, lw=0.5)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
    axes[0].set_ylabel("TEO")
    axes[1].legend(frameon=False, loc="best")
    fig.tight_layout()
    _save(fig, out)


def _gear_transition(points_map: dict[str, list[dict]], out: Path) -> None:
    fig, axes = plt.subplots(3, 2, figsize=(9.2, 6.2), sharey=True)
    specs = (
        ("camera", "cmd", "Camera-motion sweep"),
        ("object", "cmd", "Object-motion sweep"),
        ("mismatch", "alpha", r"Co-motion mismatch vs $\alpha$"),
    )
    for row, (sweep, which, title) in enumerate(specs):
        points = points_map[sweep]
        x = _xs(points, sweep, which)
        order = np.argsort(x)
        for col, (gear_key, score_name) in enumerate((("oracle_gear_teo", "TEO"), ("oracle_gear_f1", "F1"))):
            ax = axes[row, col]
            y = np.asarray([int(p[gear_key]) for p in points], dtype=np.float64)
            ax.step(x[order], y[order], where="post", color="#111827", lw=2.0)
            ax.scatter(x, y, c="#111827", s=18, zorder=3)
            ax.set_yticks([0, 180, 360])
            ax.set_title(f"{title} — oracle {score_name}", loc="left")
            if col == 0:
                ax.set_ylabel("Oracle gear")
            ax.grid(True, axis="x", alpha=0.25, lw=0.5)
            for spine in ("top", "right"):
                ax.spines[spine].set_visible(False)
    axes[2, 0].set_xlabel(r"image-plane speed or $\alpha$")
    axes[2, 1].set_xlabel(r"image-plane speed or $\alpha$")
    fig.tight_layout()
    _save(fig, out)


def _policy_vs_oracle(points_map: dict[str, list[dict]], out: Path) -> None:
    fig, axes = plt.subplots(3, 2, figsize=(9.2, 6.6), sharex="row")
    specs = (("camera", "cmd"), ("object", "cmd"), ("mismatch", "alpha"))
    titles = ("Camera sweep", "Object sweep", r"Mismatch vs $\alpha$")
    for row, ((sweep, which), title) in enumerate(zip(specs, titles)):
        points = points_map[sweep]
        x = _xs(points, sweep, which)
        order = np.argsort(x)
        x = x[order]
        oracle = np.asarray([int(p["oracle_gear_teo"]) for p in points], dtype=np.float64)[order]
        axes[row, 0].step(x, oracle, where="post", color="#111827", lw=2.0)
        axes[row, 0].scatter(x, oracle, c="#111827", s=16)
        axes[row, 0].set_yticks([0, 180, 360])
        axes[row, 0].set_ylabel("RPM")
        axes[row, 0].set_title(f"{title} — oracle gear", loc="left")
        stacked = np.vstack(
            [
                np.asarray([float(p["methods"]["adaptive"]["frac_0"]) for p in points])[order],
                np.asarray([float(p["methods"]["adaptive"]["frac_180"]) for p in points])[order],
                np.asarray([float(p["methods"]["adaptive"]["frac_360"]) for p in points])[order],
            ]
        )
        axes[row, 1].stackplot(x, stacked, colors=("#6b7280", "#1B4F8A", "#b45309"), labels=("0", "180", "360"), alpha=0.9)
        mean_rpm = np.asarray([float(p["methods"]["adaptive"]["mean_rpm"]) for p in points], dtype=np.float64)[order]
        ax_r = axes[row, 1].twinx()
        ax_r.plot(x, mean_rpm, color="#111827", lw=1.3, marker="o", ms=3.5)
        ax_r.set_ylim(-20, 400)
        ax_r.set_yticks([0, 180, 360])
        axes[row, 1].set_ylim(0, 1)
        axes[row, 1].set_ylabel("policy fraction")
        axes[row, 1].set_title(f"{title} — adaptive mix", loc="left")
        for ax in (axes[row, 0], axes[row, 1]):
            for spine in ("top",):
                ax.spines[spine].set_visible(False)
            ax.grid(True, axis="x", alpha=0.25, lw=0.5)
    axes[0, 1].legend(frameon=False, loc="upper right", ncol=3)
    axes[2, 0].set_xlabel("motion / α")
    axes[2, 1].set_xlabel("motion / α")
    fig.tight_layout()
    _save(fig, out)


def _regret(points_map: dict[str, list[dict]], out: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(10.4, 3.4), sharey=True)
    specs = (("camera", "cmd", "Camera"), ("object", "cmd", "Object"), ("mismatch", "alpha", r"Mismatch $\alpha$"))
    colors = {"fixed_0": "#6b7280", "fixed_180": "#1B4F8A", "fixed_360": "#b45309", "adaptive": "#15803d"}
    markers = {"fixed_0": "o", "fixed_180": "s", "fixed_360": "D", "adaptive": "^"}
    for ax, (sweep, which, title) in zip(axes, specs):
        points = points_map[sweep]
        x = _xs(points, sweep, which)
        for name, label in (("fixed_0", "Fixed 0"), ("fixed_180", "Fixed 180"), ("fixed_360", "Fixed 360"), ("adaptive", "Adaptive")):
            y = np.asarray([float(p["methods"][name]["regret_teo"]) for p in points], dtype=np.float64)
            std = np.zeros_like(y)
            _line(ax, x, y, std, label=label, color=colors[name], marker=markers[name])
        ax.set_title(title)
        ax.set_xlabel("Natural motion or α")
        ax.grid(True, alpha=0.25, lw=0.5)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
    axes[0].set_ylabel("TEO regret  (oracle − method)")
    axes[2].legend(frameon=False, loc="best")
    fig.tight_layout()
    _save(fig, out)


def _diagnostics(points_map: dict[str, list[dict]], out: Path) -> None:
    fig, axes = plt.subplots(3, 3, figsize=(10.8, 8.4))
    sweep = "camera"
    points = points_map[sweep]
    x = _xs(points, sweep, "cmd")
    for ax, (metric, title) in zip(axes.ravel(), METRIC_PANELS):
        for method, label, color, marker in METHODS:
            if method == "oracle_teo" and metric not in {"teo", "edge_f1"}:
                continue
            y, std = _ys(points, method, metric)
            _line(ax, x, y, std, label=label, color=color, marker=marker)
        ax.set_title(title)
        ax.grid(True, alpha=0.25, lw=0.5)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
    axes[-1, 1].set_xlabel("Camera-sweep commanded image-plane speed (px/s)")
    axes[0, 2].legend(frameon=False, fontsize=7)
    fig.suptitle("Camera-motion sweep diagnostics (reference plane)", fontsize=11, y=1.01)
    fig.tight_layout()
    _save(fig, out)


def _phase(points: list[dict], which: str, out: Path) -> None:
    cams = sorted({float(p["grid_camera_scale"]) for p in points})
    objs = sorted({float(p["grid_object_scale"]) for p in points})
    grid = np.full((len(objs), len(cams)), np.nan)
    for p in points:
        i = objs.index(float(p["grid_object_scale"]))
        j = cams.index(float(p["grid_camera_scale"]))
        if which == "oracle":
            grid[i, j] = float(p["oracle_gear_teo"])
        elif which == "policy":
            grid[i, j] = float(p["methods"]["adaptive"]["mean_rpm"])
        else:
            grid[i, j] = float(p["methods"]["adaptive"]["mean_rpm"]) - float(p["oracle_gear_teo"])
    fig, ax = plt.subplots(figsize=(4.6, 4.0))
    cmap_name = "coolwarm" if which == "mismatch" else "viridis"
    im = ax.imshow(grid, origin="lower", cmap=plt.colormaps[cmap_name], vmin=-360 if which == "mismatch" else 0, vmax=360)
    ax.set_xticks(range(len(cams)), [f"{v:g}" for v in cams])
    ax.set_yticks(range(len(objs)), [f"{v:g}" for v in objs])
    ax.set_xlabel("camera motion scale")
    ax.set_ylabel("object motion scale")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    _save(fig, out)


def _write_tables(aggregated: dict[str, list[dict]], root: Path) -> None:
    payload = {sweep: points for sweep, points in aggregated.items()}
    (root / "aggregated.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    with (root / "aggregated.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "sweep",
                "motion_scale",
                "object_alpha",
                "m_nat_flow_px",
                "m_nat_object_flow_px",
                "m_cmd_bg_px_s",
                "m_cmd_rel_px_s",
                "method",
                "teo",
                "teo_std",
                "edge_f1",
                "edge_f1_std",
                "mean_rpm",
                "frac_0",
                "frac_180",
                "frac_360",
                "oracle_gear_teo",
                "regret_teo",
            ]
        )
        for sweep, points in aggregated.items():
            for point in points:
                for method, stats in point["methods"].items():
                    writer.writerow(
                        [
                            sweep,
                            point.get("motion_scale"),
                            point.get("object_alpha"),
                            point["m_nat_flow_px"],
                            point["m_nat_object_flow_px"],
                            point.get("m_cmd_bg_px_s"),
                            point.get("m_cmd_rel_px_s"),
                            method,
                            stats.get("teo"),
                            stats.get("teo_std"),
                            stats.get("edge_f1"),
                            stats.get("edge_f1_std"),
                            stats.get("mean_rpm"),
                            stats.get("frac_0"),
                            stats.get("frac_180"),
                            stats.get("frac_360"),
                            point.get("oracle_gear_teo"),
                            stats.get("regret_teo"),
                        ]
                    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("results/ami_next/motion_excitation_sweep/scenes.json"))
    parser.add_argument("--output", type=Path, default=Path("results/ami_next/motion_excitation_sweep/figures"))
    args = parser.parse_args()
    scenes = json.loads(args.input.read_text(encoding="utf-8"))
    _style()
    aggregated = {sweep: _aggregate(_rows(scenes, sweep), sweep) for sweep in ("camera", "object", "mismatch", "grid") if _rows(scenes, sweep)}
    _write_tables(aggregated, args.input.parent)
    fig_root = args.output
    _response(aggregated, "teo", "TEO", fig_root / "fig_motion_response_teo", x_which="cmd")
    _response(aggregated, "edge_f1", "Global Edge F1", fig_root / "fig_motion_response_f1", methods=F1_METHODS, x_which="cmd")
    _mismatch(aggregated["mismatch"], fig_root / "fig_comotion_mismatch_sweep")
    _gear_transition(aggregated, fig_root / "fig_oracle_gear_transition")
    _policy_vs_oracle(aggregated, fig_root / "fig_policy_gear_vs_motion")
    _regret(aggregated, fig_root / "fig_oracle_regret")
    _diagnostics(aggregated, fig_root / "fig_motion_response_diagnostics")
    if "grid" in aggregated:
        _phase(aggregated["grid"], "oracle", fig_root / "fig_oracle_gear_phase_map")
        _phase(aggregated["grid"], "policy", fig_root / "fig_policy_gear_phase_map")
        _phase(aggregated["grid"], "mismatch", fig_root / "fig_policy_oracle_mismatch")
    summary = {}
    for sweep, points in aggregated.items():
        if sweep == "grid":
            continue
        regrets = {
            name: {
                "mean": float(np.mean([p["methods"][name]["regret_teo"] for p in points])),
                "max": float(np.max([p["methods"][name]["regret_teo"] for p in points])),
                "mean_f1": float(np.mean([p["methods"][name]["regret_f1"] for p in points])),
                "max_f1": float(np.max([p["methods"][name]["regret_f1"] for p in points])),
            }
            for name in ("fixed_0", "fixed_180", "fixed_360", "adaptive")
        }
        summary[sweep] = {
            "oracle_gears_teo": [int(p["oracle_gear_teo"]) for p in points],
            "oracle_gears_f1": [int(p["oracle_gear_f1"]) for p in points],
            "adaptive_mean_rpm": [float(p["methods"]["adaptive"]["mean_rpm"]) for p in points],
            "regret": regrets,
        }
    (args.input.parent / "analysis_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print({"figures": str(fig_root), "sweeps": list(aggregated)})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
