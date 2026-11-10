#!/usr/bin/env python3
"""Build tables and publication figures from E2VID downstream eval JSONs."""

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

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from metrics import affine_align  # noqa: E402
from protocol import CONDITION_LABELS, EVAL_CONDITIONS, IMU_ABLATION, SEEDS  # noqa: E402

ORDER = list(EVAL_CONDITIONS)
COLORS = {
    "no_prism": "#7a7a7a",
    "fixed_180": "#4c72b0",
    "fixed_360": "#55a868",
    "eirm": "#c44e52",
    "eirm_imu": "#8172b3",
    "eirm_no_imu": "#ccb974",
}


def _present(report: dict) -> list[str]:
    tables = report.get("tables", {})
    preferred = list(ORDER) + [c for c in IMU_ABLATION if c not in ORDER]
    return [c for c in preferred if c in tables] + [c for c in tables if c not in preferred]


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _fmt(value: float, digits: int = 3) -> str:
    if value is None or not np.isfinite(value):
        return "n/a"
    return f"{value:.{digits}f}"


def _mean_std(values: list[float]) -> tuple[float, float]:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan"), float("nan")
    if arr.size == 1:
        return float(arr[0]), 0.0
    return float(arr.mean()), float(arr.std(ddof=1))


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def zero_shot_table(report: dict, metric: str) -> tuple[str, list[dict]]:
    rows_md = []
    rows_csv = []
    for condition in _present(report):
        block = report["tables"][condition]["all"]
        rows_md.append([CONDITION_LABELS[condition], _fmt(block[f"{metric}_mean"])])
        rows_csv.append({"input": condition, "mean": block[f"{metric}_mean"], "std": block[f"{metric}_std"], "n": block["n"]})
    header = ["Input", f"{metric.upper()}"]
    if metric == "lpips":
        header[1] += " ↓"
    else:
        header[1] += " ↑"
    return markdown_table(header, rows_md), rows_csv


def mixed_table(reports: dict[int, dict], metric: str) -> tuple[str, list[dict]]:
    header = ["Input"] + [f"Seed {s}" for s in SEEDS] + ["Mean ± Std"]
    rows_md = []
    rows_csv = []
    for condition in _present(next(iter(reports.values()))):
        vals = [reports[s]["tables"][condition]["all"][f"{metric}_mean"] for s in SEEDS]
        mean, std = _mean_std(vals)
        rows_md.append([CONDITION_LABELS[condition], *[_fmt(v) for v in vals], f"{_fmt(mean)} ± {_fmt(std)}"])
        rec = {"input": condition, "mean": mean, "std": std}
        for seed, value in zip(SEEDS, vals):
            rec[f"seed_{seed}"] = value
        rows_csv.append(rec)
    title = metric.upper() + (" ↓" if metric == "lpips" else " ↑")
    header[0] = f"Input ({title})"
    return markdown_table(header, rows_md), rows_csv


def to_display(image: np.ndarray) -> np.ndarray:
    """Gamma-encode linear luminance for viewing only. Metrics stay linear."""
    return np.clip(np.asarray(image, dtype=np.float32), 0.0, 1.0) ** (1.0 / 2.2)


def panel(axis, image, title: str, vmin=0.0, vmax=1.0, cmap="gray") -> None:
    vis = to_display(image)
    if vis.ndim == 3:
        axis.imshow(np.clip(vis, 0, 1))
    else:
        axis.imshow(vis, cmap=cmap, vmin=vmin, vmax=vmax)
    axis.set_title(title, fontsize=9)
    axis.set_xticks([])
    axis.set_yticks([])


def figure_recon(cache_dir: Path, eval_dir: Path, out: Path, scene_id: str) -> None:
    from data import load_sequence

    rec_dir = eval_dir / "recons"
    cache = load_sequence(cache_dir, scene_id, "fixed_180")
    fig, axes = plt.subplots(5, 3, figsize=(9.6, 12.4), dpi=140)
    times = (0.45, 0.70, 0.92)
    for col, frac in enumerate(times):
        idx = int(np.clip(frac * (cache.n_windows - 1), 0, cache.n_windows - 1))
        gt = cache.target(idx)
        for row, condition in enumerate(["gt", *ORDER]):
            ax = axes[row, col]
            if condition == "gt":
                panel(ax, gt, f"GT  t={cache.t1[idx]:.2f}s")
                continue
            path = rec_dir / f"{scene_id.replace('/', '__')}__{condition}.npz"
            if not path.exists():
                ax.axis("off")
                continue
            with np.load(path) as z:
                rec = np.asarray(z["recon"], dtype=np.float32)
                t1 = np.asarray(z["t1"])
            aligned, _, _ = affine_align(rec[min(idx, len(rec) - 1)], gt)
            panel(ax, aligned, f"{CONDITION_LABELS[condition]}  t={t1[min(idx, len(t1) - 1)]:.2f}s")
    fig.suptitle(f"E2VID reconstruction  {scene_id}  (same checkpoint)", fontsize=11)
    fig.tight_layout()
    fig.savefig(out.with_suffix(".png"))
    fig.savefig(out.with_suffix(".pdf"))
    plt.close(fig)


def figure_motion(report: dict, out: Path, metric: str = "psnr") -> None:
    regimes = ["weak", "moderate", "mixed", "mixed_change"]
    conditions = _present(report)
    x = np.arange(len(regimes))
    width = 0.8 / max(len(conditions), 1)
    fig, ax = plt.subplots(figsize=(8.4, 3.6), dpi=140)
    for i, condition in enumerate(conditions):
        vals = []
        for regime in regimes:
            block = report["tables"][condition]["per_motion_regime"].get(regime, {})
            vals.append(block.get(f"{metric}_mean", np.nan))
        ax.bar(x + (i - (len(conditions) - 1) / 2) * width, vals, width=width, label=CONDITION_LABELS.get(condition, condition), color=COLORS.get(condition, "#333333"))
    ax.set_xticks(x)
    ax.set_xticklabels(["weak\n(object_only)", "moderate\n(camera_only)", "mixed\n(co_motion)", "mixed-change\n(drive_stop_drive)"])
    ax.set_ylabel(metric.upper() + (" ↓" if metric == "lpips" else " ↑"))
    ax.legend(frameon=False, ncol=4, fontsize=8)
    ax.set_title("Motion-regime reconstruction (same mixed-trained checkpoint)")
    fig.tight_layout()
    fig.savefig(out.with_suffix(".png"))
    fig.savefig(out.with_suffix(".pdf"))
    plt.close(fig)


def figure_temporal(report: dict, out: Path, scene_id: str, metric: str = "psnr") -> None:
    fig, ax = plt.subplots(figsize=(9.2, 3.4), dpi=140)
    switch_t = []
    for condition in _present(report):
        rows = [r for r in report["rows"] if r["scene_id"] == scene_id and r["condition"] == condition]
        rows = sorted(rows, key=lambda r: r["t1"])
        if not rows:
            continue
        ax.plot([r["t1"] for r in rows], [r[metric] for r in rows], color=COLORS.get(condition, "#333333"), label=CONDITION_LABELS.get(condition, condition), lw=1.4)
        if condition in {"eirm", "eirm_imu", "eirm_no_imu"} and not switch_t:
            switch_t = [r["t1"] for r in rows if r.get("switched")]
    for t in switch_t:
        ax.axvline(t, color="#c44e52", ls="--", lw=0.8, alpha=0.7)
    if switch_t:
        ax.plot([], [], color="#c44e52", ls="--", lw=0.8, label="EIRM switch")
    ax.set_xlabel("time (s)")
    ax.set_ylabel(metric.upper())
    ax.set_title(f"Temporal {metric.upper()}  {scene_id}")
    ax.legend(frameon=False, fontsize=8, ncol=3)
    fig.tight_layout()
    fig.savefig(out.with_suffix(".png"))
    fig.savefig(out.with_suffix(".pdf"))
    plt.close(fig)


def figure_failures(report: dict, out: Path) -> None:
    by_key = {}
    for row in report["rows"]:
        key = (row["scene_id"], row["window"])
        by_key.setdefault(key, {})[row["condition"]] = row
    failures = []
    for key, block in by_key.items():
        if "fixed_180" not in block or "eirm" not in block:
            continue
        delta = block["fixed_180"]["psnr"] - block["eirm"]["psnr"]
        if delta > 0.5:
            failures.append((delta, key, block))
    failures.sort(reverse=True)
    fig, ax = plt.subplots(figsize=(7.2, 3.4), dpi=140)
    if not failures:
        ax.text(0.5, 0.5, "No windows where Fixed-180 PSNR exceeds EIRM by > 0.5 dB", ha="center", va="center")
        ax.axis("off")
    else:
        top = failures[:40]
        ax.hist([d for d, _, _ in failures], bins=20, color="#4c72b0")
        ax.set_xlabel("Fixed-180 PSNR − EIRM PSNR (dB)")
        ax.set_ylabel("windows")
        ax.set_title(f"Failure mass: {len(failures)} windows where Fixed-180 > EIRM")
        (out.parent / "failure_windows.json").write_text(
            json.dumps(
                [
                    {
                        "scene_id": block["eirm"]["scene_id"],
                        "window": block["eirm"]["window"],
                        "t1": block["eirm"]["t1"],
                        "delta_psnr": delta,
                        "eirm_rpm": block["eirm"]["rpm"],
                        "switched": block["eirm"]["switched"],
                    }
                    for delta, _, block in top
                ],
                indent=2,
            ),
            encoding="utf-8",
        )
    fig.tight_layout()
    fig.savefig(out.with_suffix(".png"))
    fig.savefig(out.with_suffix(".pdf"))
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--zero-shot", type=Path, required=True)
    parser.add_argument("--mixed", nargs="*", default=[])
    parser.add_argument("--cache", type=Path, default=None)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--hero-scene", default="test/Scene_005/drive_stop_drive")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    zero = _load(args.zero_shot)
    mixed = {}
    for path in args.mixed:
        path = Path(path)
        data = _load(path)
        seed = int(data.get("tag") or 0)
        if seed not in SEEDS:
            # infer from folder name mixed_seed42
            for s in SEEDS:
                if f"seed{s}" in str(path):
                    seed = s
        mixed[seed] = data

    md = ["# EIRM → E2VID downstream reconstruction", "", "## Protocol", ""]
    md.append("- Event space: inverse-warped AMI reference plane; No Prism is native DVS on the same RGB plane.")
    md.append("- Target: linear luminance of the RGB frame at the window end.")
    md.append("- Metrics: per-image affine (scale+shift) then PSNR / SSIM / LPIPS.")
    md.append("- Mixed training: one shared E2VID per seed on Fixed-180/360/EIRM only. No Prism is eval-only.")
    md.append("- Zero-shot vs mixed-trained is **not** a training-matched comparison.")
    md.append("")
    md.append("## Table 1. Original E2VID zero-shot")
    md.append("")
    for metric in ("psnr", "ssim", "lpips"):
        table, rows = zero_shot_table(zero, metric)
        md.append(f"### {metric.upper()}")
        md.append("")
        md.append(table)
        md.append("")
        write_csv(args.output / f"table1_zeroshot_{metric}.csv", rows, list(rows[0].keys()))

    if mixed and set(SEEDS) <= set(mixed):
        md.append("## Table 2. Mixed-trained E2VID (shared weights per seed)")
        md.append("")
        for metric in ("psnr", "ssim", "lpips"):
            table, rows = mixed_table(mixed, metric)
            md.append(f"### {metric.upper()}")
            md.append("")
            md.append(table)
            md.append("")
            write_csv(args.output / f"table2_mixed_{metric}.csv", rows, list(rows[0].keys()))

    hero = mixed.get(SEEDS[0]) or (next(iter(mixed.values())) if mixed else zero)
    md.append("## Table 3. Motion-regime PSNR")
    md.append("")
    header = ["Input", "weak", "moderate", "mixed", "mixed_change"]
    body = []
    for condition in _present(hero):
        body.append(
            [CONDITION_LABELS.get(condition, condition)]
            + [_fmt(hero["tables"][condition]["per_motion_regime"].get(name, {}).get("psnr_mean", float("nan"))) for name in ("weak", "moderate", "mixed", "mixed_change")]
        )
    md.append(markdown_table(header, body))
    md.append("")
    figure_motion(hero, args.output / "fig_motion_regime")
    figure_temporal(hero, args.output / "fig_temporal_psnr", args.hero_scene)
    if "fixed_180" in hero.get("tables", {}) and "eirm" in hero.get("tables", {}):
        figure_failures(hero, args.output / "fig_failures_180_gt_eirm")
    recon_root = Path(str(args.mixed[0])).parent if args.mixed else args.zero_shot.parent
    if args.cache and (recon_root / "recons").exists():
        figure_recon(args.cache, recon_root, args.output / "fig_recon_compare", args.hero_scene)

    findings = [
        "## Findings",
        "",
        "- Report the numbers as measured. EIRM is not assumed to beat Fixed-180.",
        "- Switching artifacts, if present, appear as PSNR dips at the dashed EIRM switch times.",
        "- Zero-shot E2VID was trained on natural event cameras, not prism-modulated streams; mixed training is a domain-adaptation experiment, not a claim that EIRM is a better reconstruction method.",
    ]
    md.extend(findings)
    (args.output / "REPORT.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("wrote", args.output / "REPORT.md", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
