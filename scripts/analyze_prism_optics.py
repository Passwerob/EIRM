#!/usr/bin/env python3
"""Paper/appendix optical analysis of the rotating wedge (EVEYES 0.5°)."""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ami_simulator.geometry import RotatingWedgePrism
from ami_simulator.types import CameraIntrinsics

OUT = ROOT / "results/ami_next/prism_optics"
DWELL_S = 0.025
PRISM_DEG = 0.5
GEARS_RPM = (180.0, 360.0)
N_BK7 = 1.5168


def _intrinsics_eveyes() -> CameraIntrinsics:
    return CameraIntrinsics(960, 540, 426.6666666666667, 426.6666666666667, 480.0, 270.0)


def _scale_intrinsics(k: CameraIntrinsics, width: int, height: int) -> CameraIntrinsics:
    return k.resized(width, height)


def orbit_radius_px(k: CameraIntrinsics, prism_deg: float, uv: np.ndarray) -> dict:
    model = RotatingWedgePrism(k, prism_deg)
    pts = torch.tensor(uv, dtype=torch.float32)
    thetas = torch.linspace(0.0, 2 * math.pi, 361)
    shifted = []
    for th in thetas:
        shifted.append(model.forward_points(pts, th).numpy())
    shifted = np.stack(shifted, axis=0)
    center = shifted.mean(axis=0)
    radii = np.linalg.norm(shifted - center[None], axis=-1)
    paraxial = float(k.fx * math.tan(math.radians(prism_deg)))
    return {
        "paraxial_R_px": paraxial,
        "mean_R_px": float(radii.mean()),
        "min_R_px": float(radii.min()),
        "max_R_px": float(radii.max()),
        "orbit": shifted,
        "uv": uv,
    }


def kinematics(R: float, rpm: float, dwell: float) -> dict:
    omega = rpm * 2.0 * math.pi / 60.0
    period = 60.0 / rpm if rpm else float("inf")
    arc_rad = omega * dwell
    return {
        "rpm": rpm,
        "omega_rad_s": omega,
        "rev_per_s": rpm / 60.0,
        "period_s": period,
        "dwell_s": dwell,
        "dwell_arc_deg": math.degrees(arc_rad),
        "dwell_arc_px": R * arc_rad,
        "tangential_px_s": R * omega,
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    k = _intrinsics_eveyes()
    k_vga = _scale_intrinsics(k, 640, 480)
    delta = math.radians(PRISM_DEG)
    apex_bk7 = PRISM_DEG / (N_BK7 - 1.0)
    hfov = 2.0 * math.degrees(math.atan((k.width / 2.0) / k.fx))
    vfov = 2.0 * math.degrees(math.atan((k.height / 2.0) / k.fy))

    samples = {
        "principal": np.array([[k.cx, k.cy]], dtype=np.float32),
        "mid_right": np.array([[k.cx + 200.0, k.cy]], dtype=np.float32),
        "corner": np.array([[k.width - 1.0, k.height - 1.0]], dtype=np.float32),
    }
    orbits = {name: orbit_radius_px(k, PRISM_DEG, uv) for name, uv in samples.items()}
    R = orbits["principal"]["paraxial_R_px"]

    gear_rows = {int(rpm): kinematics(R, rpm, DWELL_S) for rpm in GEARS_RPM}
    rpm_scan = np.linspace(0.0, 720.0, 145)
    arc_scan = np.array([kinematics(R, float(rpm), DWELL_S)["dwell_arc_px"] for rpm in rpm_scan])
    speed_scan = np.array([kinematics(R, float(rpm), DWELL_S)["tangential_px_s"] for rpm in rpm_scan])

    # FOV variation of orbit radius
    us = np.linspace(0, k.width - 1, 17)
    vs = np.linspace(0, k.height - 1, 11)
    uu, vv = np.meshgrid(us, vs)
    grid = np.stack((uu.ravel(), vv.ravel()), axis=1).astype(np.float32)
    field = orbit_radius_px(k, PRISM_DEG, grid)

    # invalid border: pixels whose 0.5° orbit leaves the image at some phase
    model = RotatingWedgePrism(k, PRISM_DEG)
    v_pix, u_pix = torch.meshgrid(
        torch.arange(k.height, dtype=torch.float32),
        torch.arange(k.width, dtype=torch.float32),
        indexing="ij",
    )
    sensor = torch.stack((u_pix, v_pix), dim=-1)
    invalid_any = torch.zeros((k.height, k.width), dtype=torch.bool)
    for th in torch.linspace(0, 2 * math.pi, 24):
        src = model.inverse_points(sensor, th)
        invalid_any |= (
            (src[..., 0] < 0)
            | (src[..., 0] > k.width - 1)
            | (src[..., 1] < 0)
            | (src[..., 1] > k.height - 1)
        )
    invalid_frac = float(invalid_any.float().mean())

    report = {
        "model": "paraxial ray-deflection wedge (Rodrigues about in-plane axis)",
        "code": "ami_simulator.geometry.RotatingWedgePrism; prism_angle_deg is the deflection, not apex",
        "eveyes": {
            "width": k.width,
            "height": k.height,
            "fx": k.fx,
            "fy": k.fy,
            "cx": k.cx,
            "cy": k.cy,
            "hfov_deg": hfov,
            "vfov_deg": vfov,
            "implied_sensor_width_mm_if_f16mm": 16.0 * k.width / k.fx,
        },
        "wedge": {
            "deflection_deg": PRISM_DEG,
            "deflection_mrad": 1000.0 * delta,
            "thin_wedge_apex_BK7_deg": apex_bk7,
            "n_BK7": N_BK7,
            "paraxial_R_px": R,
            "R_over_fx": R / k.fx,
        },
        "orbits_px": {
            name: {k: v for k, v in row.items() if k != "orbit" and k != "uv"}
            for name, row in orbits.items()
        },
        "field_R_px": {"min": field["min_R_px"], "mean": field["mean_R_px"], "max": field["max_R_px"]},
        "dwell_s": DWELL_S,
        "gears": gear_rows,
        "invalid_border_fraction": invalid_frac,
        "vga_same_fov": {
            "width": 640,
            "height": 480,
            "fx": k_vga.fx,
            "paraxial_R_px": float(k_vga.fx * math.tan(delta)),
            "note": "same angular FOV as EVEYES 960x540; SilkyEvCam is 640x480 but lens fx is not this unless FOV matches",
        },
        "why_180_360": {
            "rule": "1-DoF discrete octave: dwell arc ~1-2 px at 180, twice that at 360",
            "arc_180_px": gear_rows[180]["dwell_arc_px"],
            "arc_360_px": gear_rows[360]["dwell_arc_px"],
            "speed_180_px_s": gear_rows[180]["tangential_px_s"],
            "speed_360_px_s": gear_rows[360]["tangential_px_s"],
        },
    }
    (OUT / "prism_optics.json").write_text(json.dumps(report, indent=2))

    # Figure 1: orbit of principal point
    fig, ax = plt.subplots(figsize=(5.2, 5.0))
    o = orbits["principal"]["orbit"][:, 0]
    ax.plot(o[:, 0], o[:, 1], color="#4c72b0", lw=1.6)
    ax.scatter([k.cx], [k.cy], c="#111827", s=28, zorder=3)
    ax.set_aspect("equal")
    ax.set_xlabel("u (px)")
    ax.set_ylabel("v (px)")
    ax.set_title(f"Principal-point orbit  ·  {PRISM_DEG}°  ·  R={R:.2f} px")
    ax.grid(alpha=0.3, lw=0.5)
    fig.tight_layout()
    fig.savefig(OUT / "fig_orbit_principal.png", dpi=180)
    plt.close(fig)

    # Figure 2: dwell arc vs RPM
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.6))
    axes[0].plot(rpm_scan, arc_scan, color="#111827", lw=1.8)
    for rpm, color in ((180, "#4c72b0"), (360, "#55a868")):
        y = kinematics(R, rpm, DWELL_S)["dwell_arc_px"]
        axes[0].scatter([rpm], [y], c=color, s=42, zorder=3)
        axes[0].annotate(f"{int(rpm)} RPM\n{y:.2f} px", (rpm, y), textcoords="offset points", xytext=(8, 6), fontsize=8, color=color)
    axes[0].axhspan(1.0, 2.0, color="#4c72b0", alpha=0.12)
    axes[0].set_xlabel("RPM")
    axes[0].set_ylabel(f"arc in {1000 * DWELL_S:.0f} ms dwell (px)")
    axes[0].set_title("Control-window arc")
    axes[0].set_xlim(0, 720)
    axes[1].plot(rpm_scan, speed_scan, color="#111827", lw=1.8)
    for rpm, color in ((180, "#4c72b0"), (360, "#55a868")):
        y = kinematics(R, rpm, DWELL_S)["tangential_px_s"]
        axes[1].scatter([rpm], [y], c=color, s=42, zorder=3)
        axes[1].annotate(f"{y:.0f} px/s", (rpm, y), textcoords="offset points", xytext=(8, 6), fontsize=8, color=color)
    axes[1].set_xlabel("RPM")
    axes[1].set_ylabel("tangential speed (px/s)")
    axes[1].set_title("Image-plane excitation speed")
    axes[1].set_xlim(0, 720)
    fig.suptitle(f"EVEYES 960×540, fx={k.fx:.1f} px, deflection {PRISM_DEG}°", fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT / "fig_rpm_arc_speed.png", dpi=180)
    plt.close(fig)

    # Figure 3: displacement field at one phase
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    th = torch.tensor(0.0)
    src = model.forward_points(sensor, th).numpy()
    du = src[..., 0] - u_pix.numpy()
    dv = src[..., 1] - v_pix.numpy()
    step = 24
    ax.quiver(
        u_pix.numpy()[::step, ::step],
        v_pix.numpy()[::step, ::step],
        du[::step, ::step],
        dv[::step, ::step],
        angles="xy",
        scale_units="xy",
        scale=0.08,
        color="#4c72b0",
        width=0.003,
    )
    ax.set_xlim(0, k.width)
    ax.set_ylim(k.height, 0)
    ax.set_aspect("equal")
    ax.set_title(r"Forward deflection field at $\theta=0$ (scale exaggerated)")
    ax.set_xlabel("u (px)")
    ax.set_ylabel("v (px)")
    fig.tight_layout()
    fig.savefig(OUT / "fig_deflection_field.png", dpi=180)
    plt.close(fig)

    print(json.dumps({k: report[k] for k in report if k != "eveyes"}, indent=2)[:2500])
    print("R_px", R)
    print("180", gear_rows[180])
    print("360", gear_rows[360])
    print("invalid_frac", invalid_frac)
    print("field R", field["min_R_px"], field["mean_R_px"], field["max_R_px"])
    print("wrote", OUT)


if __name__ == "__main__":
    main()
