#!/usr/bin/env python3
"""Qualitative real-capture figures for data_pc_time (no RGB-event FSIM)."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ami_simulator.event_visualization import eirm_from_counts, eirm_peak

DATA = ROOT / "data_pc_time"
OUT = ROOT / "results/ami_next/real_pc_time_qualitative"
H, W = 480, 640
WINDOW_MS = 33.0
HERO_T_MS = 5000.0
SNAPSHOT_MS = (2000.0, 5000.0, 10000.0)
ACCUM_MS = 99.0  # three 33 ms bins

SESSIONS = (
    ("m1_p0_stop_on_car_20261106_000558_880", "stop / car", 0),
    ("m2_p180_stop_on_car_20261106_000943_432", "stop / car", 180),
    ("m3_p360_stop_on_car_20261106_001129_598", "stop / car", 360),
    ("m4_p0_move_on_car_20261106_001303_123", "move / car", 0),
    ("m5_p180_move_on_car_20261106_001512_353", "move / car", 180),
    ("m6_p360_move_on_car_20261106_001757_355", "move / car", 360),
    ("m7_p0_move_on_ground_20261106_002146_453", "move / ground", 0),
    ("m8_p180_move_on_ground_20261106_002321_261", "move / ground", 180),
    ("m9_p360_move_on_ground_20261106_003750_022", "move / ground", 360),
)


def _style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "savefig.facecolor": "white",
            "pdf.fonttype": 42,
        }
    )


def _load_csv(path: Path) -> list[dict]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def _frame_for_time(index_rows: list[dict], t_ms: float) -> int:
    centers = np.array(
        [(float(row["t_rel_start_ms"]) + float(row["t_rel_end_ms"])) * 0.5 for row in index_rows]
    )
    return int(np.argmin(np.abs(centers - t_ms)))


def _slice_events(h5_path: Path, start: int, stop: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with h5py.File(h5_path, "r") as handle:
        x = np.rint(handle["x"][start:stop]).astype(np.int32)
        y = np.rint(handle["y"][start:stop]).astype(np.int32)
        p = np.asarray(handle["p"][start:stop])
    return x, y, p


def _counts(x: np.ndarray, y: np.ndarray, p: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    positive = np.zeros((H, W), dtype=np.float32)
    negative = np.zeros((H, W), dtype=np.float32)
    if len(x) == 0:
        return positive, negative
    valid = (x >= 0) & (x < W) & (y >= 0) & (y < H)
    x, y, p = x[valid], y[valid], p[valid]
    on = p > 0
    if np.any(on):
        np.add.at(positive, (y[on], x[on]), 1.0)
    if np.any(~on):
        np.add.at(negative, (y[~on], x[~on]), 1.0)
    return positive, negative


def _range_counts(session_dir: Path, index_rows: list[dict], start_frame: int, n_frames: int, kind: str) -> tuple[np.ndarray, np.ndarray, int]:
    start_frame = max(0, start_frame)
    end_frame = min(len(index_rows) - 1, start_frame + n_frames - 1)
    first = index_rows[start_frame]
    last = index_rows[end_frame]
    key_start = "raw_start" if kind == "raw" else "inverse_start"
    key_stop = "raw_stop" if kind == "raw" else "inverse_stop"
    start = int(first[key_start])
    stop = int(last[key_stop])
    name = "events_raw.h5" if kind == "raw" else "events_inverse.h5"
    x, y, p = _slice_events(session_dir / name, start, stop)
    return (*_counts(x, y, p), int(stop - start))


def _nearest_rgb(session_dir: Path, index_rows: list[dict], rgb_rows: list[dict], frame: int) -> Image.Image:
    rgb_index = int(index_rows[frame]["nearest_rgb_index"])
    rgb_index = int(np.clip(rgb_index, 0, len(rgb_rows) - 1))
    image = Image.open(session_dir / "rgb" / rgb_rows[rgb_index]["file"])
    return ImageOps.exif_transpose(image).convert("RGB")


def _shared_peak(pairs: list[tuple[np.ndarray, np.ndarray]]) -> tuple[float, float]:
    pos = max(eirm_peak(pos) for pos, _ in pairs)
    neg = max(eirm_peak(neg) for _, neg in pairs)
    return pos, neg


def _panel(axis, image: np.ndarray | Image.Image, title: str, *, cmap=None) -> None:
    if isinstance(image, Image.Image):
        axis.imshow(image)
    else:
        axis.imshow(image, interpolation="nearest")
    axis.set_title(title, fontsize=9, pad=4)
    axis.set_xticks([])
    axis.set_yticks([])
    for spine in axis.spines.values():
        spine.set_linewidth(0.4)
        spine.set_color("#9ca3af")


def _load_session(name: str) -> dict:
    session_dir = DATA / name
    return {
        "dir": session_dir,
        "index": _load_csv(session_dir / "index.csv"),
        "rgb": _load_csv(session_dir / "rgb_time.csv"),
        "imu": _load_csv(session_dir / "imu.csv"),
        "meta": json.loads((session_dir / "metadata.json").read_text()),
    }


def fig_rgb_grid(sessions: dict[str, dict]) -> None:
    fig, axes = plt.subplots(3, 3, figsize=(11.2, 7.4), dpi=140)
    platforms = ("stop / car", "move / car", "move / ground")
    rpms = (0, 180, 360)
    lookup = {(plat, rpm): name for name, plat, rpm in SESSIONS}
    for row, plat in enumerate(platforms):
        for col, rpm in enumerate(rpms):
            name = lookup[(plat, rpm)]
            data = sessions[name]
            frame = _frame_for_time(data["index"], HERO_T_MS)
            image = _nearest_rgb(data["dir"], data["index"], data["rgb"], frame)
            title = f"{rpm} rpm" if row == 0 else ""
            _panel(axes[row, col], image, title)
            if col == 0:
                axes[row, col].set_ylabel(plat, fontsize=9)
    fig.suptitle(f"Phone RGB near t = {HERO_T_MS/1000:.1f} s (not geometrically registered to events)", fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT / "fig_rgb_3x3.png", bbox_inches="tight")
    plt.close(fig)


def fig_event_grids(sessions: dict[str, dict], n_frames: int, suffix: str, title: str) -> dict:
    platforms = ("stop / car", "move / car", "move / ground")
    rpms = (0, 180, 360)
    lookup = {(plat, rpm): name for name, plat, rpm in SESSIONS}
    packed = {}
    for plat in platforms:
        for rpm in rpms:
            name = lookup[(plat, rpm)]
            data = sessions[name]
            frame = _frame_for_time(data["index"], HERO_T_MS)
            packed[(plat, rpm)] = _range_counts(data["dir"], data["index"], frame, n_frames, "raw")
    peak = _shared_peak([(pos, neg) for pos, neg, _ in packed.values()])
    fig, axes = plt.subplots(3, 3, figsize=(11.2, 8.2), dpi=140)
    stats = {}
    for row, plat in enumerate(platforms):
        for col, rpm in enumerate(rpms):
            pos, neg, n_events = packed[(plat, rpm)]
            image = eirm_from_counts(pos, neg, peak=peak)
            occupied = float(((pos + neg) > 0).mean())
            _panel(axes[row, col], image, f"{rpm} rpm · {n_events/1000:.1f}k ev · occ {occupied:.0%}" if row == 0 else f"{n_events/1000:.1f}k ev · occ {occupied:.0%}")
            if col == 0:
                axes[row, col].set_ylabel(plat, fontsize=9)
            stats[f"{plat}|{rpm}"] = {"events": n_events, "occupied": occupied, "max": float((pos + neg).max())}
    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT / f"fig_raw_events_3x3_{suffix}.png", bbox_inches="tight")
    plt.close(fig)
    return stats


def fig_compensation(sessions: dict[str, dict]) -> list[dict]:
    rows_spec = [
        ("m2_p180_stop_on_car_20261106_000943_432", "180 rpm · stop / car"),
        ("m5_p180_move_on_car_20261106_001512_353", "180 rpm · move / car"),
        ("m8_p180_move_on_ground_20261106_002321_261", "180 rpm · move / ground"),
        ("m6_p360_move_on_car_20261106_001757_355", "360 rpm · move / car"),
        ("m9_p360_move_on_ground_20261106_003750_022", "360 rpm · move / ground"),
        ("m3_p360_stop_on_car_20261106_001129_598", "360 rpm · stop / car"),
    ]
    n_frames = int(round(ACCUM_MS / WINDOW_MS))
    records = []
    fig, axes = plt.subplots(len(rows_spec), 2, figsize=(8.6, 14.4), dpi=140)
    for row, (name, label) in enumerate(rows_spec):
        data = sessions[name]
        frame = _frame_for_time(data["index"], HERO_T_MS)
        raw = _range_counts(data["dir"], data["index"], frame, n_frames, "raw")
        inv = _range_counts(data["dir"], data["index"], frame, n_frames, "inverse")
        peak = _shared_peak([(raw[0], raw[1]), (inv[0], inv[1])])
        raw_occ = float(((raw[0] + raw[1]) > 0).mean())
        inv_occ = float(((inv[0] + inv[1]) > 0).mean())
        raw_max = float((raw[0] + raw[1]).max())
        inv_max = float((inv[0] + inv[1]).max())
        _panel(axes[row, 0], eirm_from_counts(raw[0], raw[1], peak=peak), f"{label}\nraw  occ {raw_occ:.0%}  peak {raw_max:.0f}")
        _panel(axes[row, 1], eirm_from_counts(inv[0], inv[1], peak=peak), f"inverse  occ {inv_occ:.0%}  peak {inv_max:.0f}")
        records.append(
            {
                "session": name,
                "label": label,
                "raw_events": raw[2],
                "inverse_events": inv[2],
                "raw_occupied": raw_occ,
                "inverse_occupied": inv_occ,
                "raw_peak": raw_max,
                "inverse_peak": inv_max,
            }
        )
    fig.suptitle(f"Raw vs inverse-compensated events, {ACCUM_MS:.0f} ms at t = {HERO_T_MS/1000:.1f} s (shared scale per row)", fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT / "fig_compensation_raw_vs_inverse.png", bbox_inches="tight")
    plt.close(fig)
    return records


def fig_snapshots(sessions: dict[str, dict]) -> None:
    name = "m6_p360_move_on_car_20261106_001757_355"
    data = sessions[name]
    n_frames = int(round(ACCUM_MS / WINDOW_MS))
    fig, axes = plt.subplots(3, 3, figsize=(11.0, 8.6), dpi=140)
    for col, t_ms in enumerate(SNAPSHOT_MS):
        frame = _frame_for_time(data["index"], t_ms)
        rgb = _nearest_rgb(data["dir"], data["index"], data["rgb"], frame)
        raw = _range_counts(data["dir"], data["index"], frame, n_frames, "raw")
        inv = _range_counts(data["dir"], data["index"], frame, n_frames, "inverse")
        peak = _shared_peak([(raw[0], raw[1]), (inv[0], inv[1])])
        _panel(axes[0, col], rgb, f"t = {t_ms/1000:.0f} s")
        _panel(axes[1, col], eirm_from_counts(raw[0], raw[1], peak=peak), f"raw {raw[2]/1000:.1f}k")
        _panel(axes[2, col], eirm_from_counts(inv[0], inv[1], peak=peak), f"inverse {inv[2]/1000:.1f}k")
        if col == 0:
            axes[0, col].set_ylabel("RGB")
            axes[1, col].set_ylabel("raw events")
            axes[2, col].set_ylabel("inverse")
    fig.suptitle("m6 360 rpm move / car: same 99 ms window at three times", fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT / "fig_time_snapshots_m6.png", bbox_inches="tight")
    plt.close(fig)


def fig_rates(sessions: dict[str, dict]) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(10.4, 8.0), dpi=140, sharex=True)
    colors = {0: "#6b7280", 180: "#2563a6", 360: "#c96a1b"}
    platforms = ("stop / car", "move / car", "move / ground")
    for ax, plat in zip(axes, platforms):
        for name, session_plat, rpm in SESSIONS:
            if session_plat != plat:
                continue
            rows = sessions[name]["index"]
            t = np.array([float(row["t_rel_start_ms"]) / 1000.0 for row in rows])
            n = np.array([int(row["n_raw"]) for row in rows], dtype=float)
            keps = n / (WINDOW_MS / 1000.0) / 1000.0
            ax.plot(t, keps, color=colors[rpm], lw=1.15, label=f"{rpm} rpm")
        ax.set_ylabel("keps")
        ax.set_title(plat, loc="left")
        ax.legend(frameon=False, ncol=3)
        ax.set_xlim(0, 15)
        ax.axvspan(6.96, 7.16, color="#ef4444", alpha=0.12)
    axes[0].axvspan(6.96, 7.16, color="#ef4444", alpha=0.18)
    axes[-1].set_xlabel("time (s)")
    fig.suptitle("Per 33 ms window event rate. Red band marks m3 USB dropout near 7 s.", fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT / "fig_event_rate.png", bbox_inches="tight")
    plt.close(fig)


def fig_imu(sessions: dict[str, dict]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 3.8), dpi=140)
    colors = {"stop / car": "#6b7280", "move / car": "#c96a1b", "move / ground": "#2563a6"}
    for name, plat, rpm in SESSIONS:
        imu = sessions[name]["imu"]
        t = np.array([float(row["t_rel_ms"]) / 1000.0 for row in imu])
        ax = np.array([float(row["ax_m_s2"]) for row in imu])
        ay = np.array([float(row["ay_m_s2"]) for row in imu])
        az = np.array([float(row["az_m_s2"]) for row in imu])
        mag = np.sqrt(ax * ax + ay * ay + az * az)
        ang = np.unwrap(np.array([float(row["angle_rad"]) for row in imu]))
        axes[0].plot(t, mag, color=colors[plat], lw=0.7, alpha=0.85 if rpm == 0 else 0.45)
        axes[1].plot(t, np.degrees(ang), color={0: "#6b7280", 180: "#2563a6", 360: "#c96a1b"}[rpm], lw=0.9)
    axes[0].set_title("IMU |a| (car vs ground vibration)")
    axes[0].set_xlabel("time (s)")
    axes[0].set_ylabel("m/s²")
    axes[1].set_title("Encoder angle (unwrap)")
    axes[1].set_xlabel("time (s)")
    axes[1].set_ylabel("deg")
    fig.tight_layout()
    fig.savefig(OUT / "fig_imu_accel_angle.png", bbox_inches="tight")
    plt.close(fig)


def fig_dropout(sessions: dict[str, dict]) -> None:
    name = "m3_p360_stop_on_car_20261106_001129_598"
    data = sessions[name]
    n_frames = int(round(ACCUM_MS / WINDOW_MS))
    times = (6500.0, 7000.0, 7500.0)
    fig, axes = plt.subplots(1, 3, figsize=(10.8, 3.4), dpi=140)
    packs = []
    for t_ms in times:
        frame = _frame_for_time(data["index"], t_ms)
        packs.append(_range_counts(data["dir"], data["index"], frame, n_frames, "raw"))
    peak = _shared_peak([(pos, neg) for pos, neg, _ in packs])
    for ax, t_ms, (pos, neg, n_events) in zip(axes, times, packs):
        _panel(ax, eirm_from_counts(pos, neg, peak=peak), f"t = {t_ms/1000:.1f} s · {n_events} events")
    fig.suptitle("m3 360 rpm stop / car: USB dropout around 7.0 s", fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT / "fig_m3_dropout.png", bbox_inches="tight")
    plt.close(fig)


def fig_zero_rpm_control(sessions: dict[str, dict]) -> None:
    names = [
        ("m1_p0_stop_on_car_20261106_000558_880", "0 rpm stop / car"),
        ("m4_p0_move_on_car_20261106_001303_123", "0 rpm move / car"),
        ("m7_p0_move_on_ground_20261106_002146_453", "0 rpm move / ground"),
    ]
    n_frames = int(round(ACCUM_MS / WINDOW_MS))
    packs = []
    rgbs = []
    for name, _ in names:
        data = sessions[name]
        frame = _frame_for_time(data["index"], HERO_T_MS)
        rgbs.append(_nearest_rgb(data["dir"], data["index"], data["rgb"], frame))
        packs.append(_range_counts(data["dir"], data["index"], frame, n_frames, "raw"))
    peak = _shared_peak([(pos, neg) for pos, neg, _ in packs])
    fig, axes = plt.subplots(2, 3, figsize=(11.0, 6.2), dpi=140)
    for col, ((_, label), rgb, (pos, neg, n_events)) in enumerate(zip(names, rgbs, packs)):
        _panel(axes[0, col], rgb, label)
        _panel(axes[1, col], eirm_from_counts(pos, neg, peak=peak), f"raw {n_events/1000:.1f}k")
    fig.suptitle("0 rpm control: residual events from scene/platform motion, not the spinner", fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT / "fig_zero_rpm_control.png", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    _style()
    OUT.mkdir(parents=True, exist_ok=True)
    sessions = {name: _load_session(name) for name, _, _ in SESSIONS}
    fig_rgb_grid(sessions)
    stats_33 = fig_event_grids(
        sessions,
        1,
        "33ms",
        f"Raw events, 33 ms window at t = {HERO_T_MS/1000:.1f} s (one shared EIRM scale)",
    )
    stats_99 = fig_event_grids(
        sessions,
        int(round(ACCUM_MS / WINDOW_MS)),
        "99ms",
        f"Raw events, {ACCUM_MS:.0f} ms window at t = {HERO_T_MS/1000:.1f} s (one shared EIRM scale)",
    )
    compensation = fig_compensation(sessions)
    fig_snapshots(sessions)
    fig_rates(sessions)
    fig_imu(sessions)
    fig_dropout(sessions)
    fig_zero_rpm_control(sessions)
    report = {
        "hero_t_ms": HERO_T_MS,
        "accum_ms": ACCUM_MS,
        "raw_33ms": stats_33,
        "raw_99ms": stats_99,
        "compensation": compensation,
        "notes": [
            "RGB and events are not geometrically registered.",
            "Event timestamps are USB chunk receive times.",
            "m3 has a USB dropout near 7 s; hero window is 5 s.",
        ],
    }
    (OUT / "qualitative_stats.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
