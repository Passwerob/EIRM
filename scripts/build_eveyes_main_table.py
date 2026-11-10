#!/usr/bin/env python3
"""Build the same-source edge table from native EVEYES results only."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "results/ami_next/eveyes_v2_regimes_test_ours_plus_dsd.json"
OUT_TEX = ROOT / "results/ami_next/eveyes_same_source_edge_reconstruction.tex"
OUT_JSON = ROOT / "results/ami_next/eveyes_same_source_edge_reconstruction.json"

PAPER_REGIMES = ("camera_only", "co_motion", "object_only", "counter_motion")
SUBSET_COLUMNS = (
    ("co_motion", "co_motion_f1", "Co-motion F1"),
    ("object_only", "object_only_f1", "Object-only F1"),
    ("counter_motion", "counter_motion_f1", "Counter-motion F1"),
)


def mean(rows: list[dict], key: str) -> float:
    return sum(float(row[key]) for row in rows) / len(rows)


def _regime(row: dict) -> str:
    return str(row["scene"]).rsplit("/", 1)[-1]


def main() -> int:
    report = json.loads(SOURCE.read_text(encoding="utf-8"))
    methods = [
        ("dvs_voltmeter", "DVS-Voltmeter"),
        ("v2e", "v2e"),
        ("ami_fixed_180", "Fixed 180 RPM"),
        ("ours", "EIRM"),
    ]
    present = {name for row in report["rows"] if (name := _regime(row)) in PAPER_REGIMES}
    missing = [name for name in PAPER_REGIMES if name not in present]
    paper_rows = [row for row in report["rows"] if _regime(row) in PAPER_REGIMES]
    subset = [item for item in SUBSET_COLUMNS if item[0] in present]
    summary = {}
    for method, label in methods:
        selected = [row for row in paper_rows if row["method"] == method]
        if not selected:
            raise SystemExit(f"no paper rows for {method}")
        condition_rows = {
            name: [row for row in selected if _regime(row) == name]
            for name in PAPER_REGIMES
        }
        entry = {
            "label": label,
            "mean_f1": mean(selected, "edge_f1"),
            "precision": mean(selected, "edge_precision"),
            "recall": mean(selected, "edge_recall"),
            "windows": len(selected),
            "regimes": sorted({_regime(row) for row in selected}),
        }
        for name, key, _ in SUBSET_COLUMNS:
            rows = condition_rows[name]
            entry[key] = None if not rows else mean(rows, "edge_f1")
        summary[method] = entry

    payload = {
        "source": str(SOURCE),
        "dataset": report["dataset"],
        "protocol": report["protocol"],
        "source_resolution": report["source_resolution"],
        "eval_width": report["eval_width"],
        "split": "test/Scene_005",
        "regimes": [name for name in PAPER_REGIMES if name in present],
        "missing_regimes": missing,
        "aggregation": (
            f"arithmetic mean over 479 temporal windows per available regime; "
            f"Mean columns use {len(present)} regime(s): "
            + ", ".join(name for name in PAPER_REGIMES if name in present)
        ),
        "methods": summary,
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    if missing:
        print("missing Table-1 regimes (do not fill from 64x64 or drive_stop_drive): " + ", ".join(missing))

    keys = ("mean_f1", "precision", "recall") + tuple(key for _, key, _ in subset)
    ranks = {}
    for key in keys:
        ordered = sorted(((method, summary[method][key]) for method, _ in methods), key=lambda item: item[1], reverse=True)
        ranks[key] = {ordered[0][0]: "bold", ordered[1][0]: "italic"}

    def cell(method: str, key: str) -> str:
        value = f"{summary[method][key]:.4f}"
        style = ranks[key].get(method)
        if style == "bold":
            return r"\textbf{" + value + "}"
        if style == "italic":
            return r"\textit{" + value + "}"
        return value

    n_mean = len(present)
    regime_words = {
        "camera_only": "camera-only",
        "co_motion": "co-motion",
        "object_only": "object-only",
        "counter_motion": "counter-motion",
    }
    mean_names = ", ".join(regime_words[name] for name in PAPER_REGIMES if name in present)
    header = "Method & Mean F1 & Precision & Recall & " + " & ".join(label for _, _, label in subset) + r" \\"
    align = "l" + "c" * (3 + len(subset))
    missing_comment = (
        "% counter_motion ABSENT: Mean uses the available regimes only. Do not substitute 64x64 or drive_stop_drive."
        if "counter_motion" in missing
        else "% Mean columns average camera_only, co_motion, object_only, and counter_motion windows."
    )
    caption = (
        r"    \caption{\textbf{Same-source edge reconstruction on native EVEYES-co-motion-v2.} "
        f"Mean columns average the {mean_names} test regimes of Scene_005 "
        f"({n_mean} regime"
        + ("s" if n_mean != 1 else "")
        + r"); subset columns report the corresponding temporal-window subsets. "
        r"\textbf{Best} and \textit{second-best} results are shown in bold and italics.}"
    )
    lines = [
        "% Native EVEYES-co-motion-v2; generated by scripts/build_eveyes_main_table.py",
        missing_comment,
        r"\begin{table}[!ht]",
        r"    \centering",
        caption,
        r"    \label{tab:downstream_main_eveyes}",
        r"    \small",
        r"    \setlength{\tabcolsep}{" + ("4.0pt" if len(subset) >= 3 else "5.0pt") + "}",
        r"    \begin{tabular}{" + align + "}",
        r"        \toprule",
        r"        " + header,
        r"        \midrule",
    ]
    for method, label in methods:
        row = f"        {label} & " + " & ".join(cell(method, key) for key in keys) + r" \\"
        if method == "ours":
            lines.append(r"        \rowcolor{ourgray}")
        lines.append(row)
    lines.extend([r"        \bottomrule", r"    \end{tabular}", r"\end{table}"])
    OUT_TEX.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({k: payload[k] for k in payload if k != "methods"}, indent=2))
    print(json.dumps({method: {key: summary[method][key] for key in ("mean_f1", "precision", "recall", "co_motion_f1", "object_only_f1", "counter_motion_f1", "windows")} for method, _ in methods}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
