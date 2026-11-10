#!/usr/bin/env python3
"""Build TEO / FSIM / LPIPS report with C, P (Purity), tau, event density, and CI."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "ami_next"

METHOD_LABELS = {
    "dvs_voltmeter": "DVS-Voltmeter",
    "v2e": "v2e",
    "fixed_180": "Fixed 180 RPM",
    "fixed_360": "Fixed 360 RPM",
    "ours_f1_imu": "EIRM",
}

TEO_METHODS = ("dvs_voltmeter", "v2e", "fixed_180", "fixed_360", "ours_f1_imu")
FSIM_METHODS = ("dvs_voltmeter", "v2e", "fixed_180", "fixed_360", "ours_f1_imu")
FSIM_LABELS = {
    "dvs_voltmeter": "DVS-Voltmeter",
    "v2e": "v2e",
    "fixed_180": "Fixed 180 RPM",
    "fixed_360": "Fixed 360 RPM",
    "ours_f1_imu": "EIRM",
}

TEO_FIELDS = {
    "teo": "teo",
    "C": "target_coverage",
    "P": "target_purity",
    "tau": "target_temporal_support",
}

PAPER_REGIMES = ("camera_only", "object_only", "co_motion", "counter_motion")
FSIM_TEST_REGIMES = ("camera_only", "object_only", "co_motion")
IMAGE_PIXELS = 64 * 64


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def scene_means(rows: list[dict], method: str, key: str) -> dict[str, float]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        if row["method"] != method:
            continue
        grouped[str(row["scene"])].append(float(row[key]))
    return {scene: float(np.mean(values)) for scene, values in grouped.items()}


def bootstrap_ci(values: list[float] | dict[str, float], *, samples: int = 5000, seed: int = 17) -> dict[str, float]:
    if isinstance(values, dict):
        values = list(values.values())
    arr = np.asarray(values, dtype=float)
    if len(arr) == 0:
        return {"mean": float("nan"), "ci95_low": float("nan"), "ci95_high": float("nan"), "n": 0}
    if len(arr) == 1:
        value = float(arr[0])
        return {"mean": value, "ci95_low": value, "ci95_high": value, "n": 1}
    rng = np.random.default_rng(seed)
    boots = np.asarray(
        [float(np.mean(rng.choice(arr, size=len(arr), replace=True))) for _ in range(samples)],
        dtype=float,
    )
    return {
        "mean": float(np.mean(arr)),
        "ci95_low": float(np.quantile(boots, 0.025)),
        "ci95_high": float(np.quantile(boots, 0.975)),
        "n": int(len(arr)),
    }


def fmt(value: float, digits: int = 4) -> str:
    return f"{value:.{digits}f}"


def fmt_ci(stat: dict[str, float], digits: int = 4) -> str:
    return f"{fmt(stat['mean'], digits)} [{fmt(stat['ci95_low'], digits)}, {fmt(stat['ci95_high'], digits)}]"


def fmt_pm(mean: float, std: float, digits: int = 4) -> str:
    return f"{mean:.{digits}f} ± {std:.{digits}f}"


def enrich_teo_rows(rows: list[dict]) -> list[dict]:
    enriched: list[dict] = []
    for row in rows:
        event_count = float(row["target_event_count"]) + float(row["background_event_count"])
        enriched.append({**row, "event_count": event_count, "event_density": event_count / IMAGE_PIXELS})
    return enriched


def summarize_teo(rows: list[dict]) -> dict[str, dict[str, dict[str, float]]]:
    summary: dict[str, dict[str, dict[str, float]]] = {}
    for method in TEO_METHODS:
        summary[method] = {}
        for short, key in TEO_FIELDS.items():
            summary[method][short] = bootstrap_ci(scene_means(rows, method, key))
        summary[method]["event_density"] = bootstrap_ci(scene_means(rows, method, "event_density"))
    return summary


def summarize_fsim(csv_path: Path) -> dict[str, dict[str, object]]:
    rows: list[dict[str, str]] = []
    with csv_path.open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["split"] != "test" or row["regime"] not in FSIM_TEST_REGIMES:
                continue
            rows.append(row)

    summary: dict[str, dict[str, object]] = {}
    for method, label in FSIM_LABELS.items():
        selected = [row for row in rows if row["method"] == label]
        if not selected:
            continue
        fsims = np.asarray([float(row["fsim"]) for row in selected], dtype=float)
        lpips = np.asarray([float(row["lpips"]) for row in selected], dtype=float)
        summary[method] = {
            "fsim_mean": float(np.mean(fsims)),
            "fsim_std": float(np.std(fsims, ddof=1)) if len(fsims) > 1 else 0.0,
            "lpips_mean": float(np.mean(lpips)),
            "lpips_std": float(np.std(lpips, ddof=1)) if len(lpips) > 1 else 0.0,
            "n_windows": len(selected),
            "regimes": sorted({row["regime"] for row in selected}),
        }
    return summary


def build_text(teo: dict, fsim: dict) -> str:
    lines = [
        "Metric report with uncertainty (existing outputs only)",
        "=====================================================",
        "",
        "Notation:",
        "  C   = target-edge coverage",
        "  P   = target purity (same Pi / target_purity used in prior TEO experiments)",
        "  tau = temporal support",
        "  TEO = F1(C, P) * tau",
        "  event density = (target + background events) / (64 x 64 pixels)",
        "",
        "TEO block (held-out controlled test, 12 scenes = 3 seeds x 4 regimes):",
        "  uncertainty = paired scene-level bootstrap 95% CI over scene means",
        "",
    ]
    for method in TEO_METHODS:
        lines.append(f"{METHOD_LABELS[method]}")
        for key in ("teo", "C", "P", "tau", "event_density"):
            lines.append(f"  {key:14s}: {fmt_ci(teo[method][key])}")
        lines.append("")
    lines.extend(
        [
            "FSIM / LPIPS block (EVEYES test hero windows, 3 paper regimes):",
            "  uncertainty = window-level standard deviation (n=3 windows per method)",
            "  do not call this a scene bootstrap CI",
            "",
        ]
    )
    for method in FSIM_METHODS:
        if method not in fsim:
            continue
        block = fsim[method]
        lines.append(
            f"{METHOD_LABELS.get(method, method)}: "
            f"FSIM {fmt_pm(block['fsim_mean'], block['fsim_std'])}, "
            f"LPIPS {fmt_pm(block['lpips_mean'], block['lpips_std'])}, "
            f"n={block['n_windows']}"
        )
    return "\n".join(lines) + "\n"


def build_tex(teo: dict, fsim: dict) -> str:
    lines = [
        r"\begin{table}[t]",
        r"    \centering",
        r"    \caption{Target observability with component breakdown. $P$ denotes target purity (same $\Pi$ factor used in prior TEO experiments). Scene-level bootstrap 95\% CI over 12 held-out scenes.}",
        r"    \label{tab:teo_components_ci}",
        r"    \small",
        r"    \setlength{\tabcolsep}{3.5pt}",
        r"    \begin{tabular}{lcccccc}",
        r"        \toprule",
        r"        Method & TEO $\uparrow$ & $C$ $\uparrow$ & $P$ $\uparrow$ & $\tau$ $\uparrow$ & Density & CI note \\",
        r"        \midrule",
    ]
    for method in TEO_METHODS:
        label = METHOD_LABELS[method]
        stats = teo[method]
        lines.append(
            "        "
            + " & ".join(
                [
                    label,
                    fmt(stats["teo"]["mean"]),
                    fmt(stats["C"]["mean"]),
                    fmt(stats["P"]["mean"]),
                    fmt(stats["tau"]["mean"]),
                    fmt(stats["event_density"]["mean"]),
                    f"[{fmt(stats['teo']['ci95_low'])}, {fmt(stats['teo']['ci95_high'])}]",
                ]
            )
            + r" \\"
        )
    lines.extend(
        [
            r"        \bottomrule",
            r"    \end{tabular}",
            r"\end{table}",
            "",
            r"\begin{table}[t]",
            r"    \centering",
            r"    \caption{Accumulation fidelity on EVEYES test hero windows. Values are mean $\pm$ window std over three paper regimes; this is not a scene bootstrap interval.}",
            r"    \label{tab:fsim_lpips_uncertainty}",
            r"    \small",
            r"    \begin{tabular}{lccc}",
            r"        \toprule",
            r"        Method & FSIM $\uparrow$ & LPIPS $\downarrow$ & Windows \\",
            r"        \midrule",
        ]
    )
    for method in FSIM_METHODS:
        if method not in fsim:
            continue
        block = fsim[method]
        lines.append(
            "        "
            + " & ".join(
                [
                    METHOD_LABELS.get(method, method),
                    fmt_pm(block["fsim_mean"], block["fsim_std"]),
                    fmt_pm(block["lpips_mean"], block["lpips_std"]),
                    str(block["n_windows"]),
                ]
            )
            + r" \\"
        )
    lines.extend([r"        \bottomrule", r"    \end{tabular}", r"\end{table}", ""])
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--teo", type=Path, default=RESULTS / "main_teo_test_current.json")
    parser.add_argument("--fsim-csv", type=Path, default=RESULTS / "accumulation_fidelity_fsim" / "per_frame_metrics.csv")
    parser.add_argument("--output-json", type=Path, default=RESULTS / "metric_report_with_ci.json")
    parser.add_argument("--output-txt", type=Path, default=RESULTS / "metric_report_with_ci.txt")
    parser.add_argument("--output-tex", type=Path, default=RESULTS / "metric_report_with_ci.tex")
    args = parser.parse_args()

    teo_rows = enrich_teo_rows(load_json(args.teo)["rows"])
    teo_summary = summarize_teo(teo_rows)
    fsim_summary = summarize_fsim(args.fsim_csv)

    payload = {
        "protocol": "ami_metric_report_with_ci_v1",
        "notation": {
            "C": "target_coverage",
            "P": "target_purity (same Pi / Purity factor in prior TEO experiments)",
            "tau": "target_temporal_support",
            "teo": "F1(C, P) * tau",
            "event_density": "total events / (64 * 64)",
        },
        "uncertainty": {
            "teo": "scene-level bootstrap 95% CI over 12 held-out scenes",
            "fsim_lpips": "window-level mean ± std; not a scene bootstrap CI",
        },
        "sources": {
            "teo": str(args.teo),
            "fsim_lpips": str(args.fsim_csv),
        },
        "teo": {method: {key: teo_summary[method][key] for key in ("teo", "C", "P", "tau", "event_density")} for method in TEO_METHODS},
        "fsim_lpips": fsim_summary,
    }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    args.output_txt.write_text(build_text(teo_summary, fsim_summary), encoding="utf-8")
    args.output_tex.write_text(build_tex(teo_summary, fsim_summary), encoding="utf-8")
    print(json.dumps({"json": str(args.output_json), "txt": str(args.output_txt), "tex": str(args.output_tex)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
