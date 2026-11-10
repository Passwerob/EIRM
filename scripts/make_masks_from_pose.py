#!/usr/bin/env python3
"""Generate binary neighbor masks for a constructed regime by projecting the
rendered neighbor box into the camera. Used for sources (e.g. co_motion) that
were shipped without Cryptomatte object masks. Box dims come from the render
motion_config (vehicle_width_m); length/height use the Aventador proportions
already consistent with the existing masks (validated against camera_only).
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np


def _read_pose(path: Path):
    rows = list(csv.DictReader(path.open()))
    pos = np.asarray([[float(r[f"position_{a}"]) for a in "xyz"] for r in rows])
    quat = np.asarray([[float(r[f"quaternion_{a}"]) for a in ("w", "x", "y", "z")] for r in rows])
    return pos, quat


def _qmat(q):
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


def _corners(center, yaw, w, l, h, z0):
    # ego/vehicle frame: x lateral, y forward, z up
    hx, hy, hz = w / 2, l / 2, h
    local = np.array([[sx * hx, sy * hy, z0 + sz * hz] for sx in (-1, 1) for sy in (-1, 1) for sz in (0, 1)])
    c, s = np.cos(yaw), np.sin(yaw)
    R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    return local @ R.T + center


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("scene", type=Path)
    ap.add_argument("--width-m", type=float, default=2.0)
    ap.add_argument("--length-m", type=float, default=4.6)
    ap.add_argument("--height-m", type=float, default=1.15)
    ap.add_argument("--out", default="object_masks")
    args = ap.parse_args()
    cam_pos, cam_q = _read_pose(args.scene / "camera_pose.csv")
    nb_pos, nb_q = _read_pose(args.scene / "neighbor_pose.csv")
    intr = json.loads((args.scene / "camera_intrinsics.json").read_text())
    fx, fy, cx, cy, W, H = intr["fx"], intr["fy"], intr["cx"], intr["cy"], int(intr["width"]), int(intr["height"])
    out = args.scene / args.out
    out.mkdir(parents=True, exist_ok=True)
    for i in range(len(cam_pos)):
        Rw = _qmat(cam_q[i])  # world_from_camera rotation (Blender cam: +Z up? use as-is)
        # Blender camera looks along -Z in its frame. Build cam_from_world.
        Rc = Rw
        Pc = cam_pos[i]
        # corners in world
        yaw = np.arctan2(Rw[1, 0], Rw[0, 0]) * 0  # neighbor yaw from its own quat
        Rn = _qmat(nb_q[i])
        n_yaw = np.arctan2(Rn[1, 0], Rn[0, 0])
        corners = _corners(nb_pos[i], n_yaw, args.width_m, args.length_m, args.height_m, 0.0)
        # to camera coords: Blender cam +X right, +Y up, -Z forward
        pts = (corners - Pc) @ Rc
        z = -pts[:, 2]
        valid = z > 0.2
        if not valid.any():
            mask = np.zeros((H, W), np.uint8)
        else:
            u = fx * pts[:, 0] / z + cx
            v = cy - fy * pts[:, 1] / z
            u0, u1 = int(np.floor(u[valid].min())), int(np.ceil(u[valid].max()))
            v0, v1 = int(np.floor(v[valid].min())), int(np.ceil(v[valid].max()))
            u0, u1 = max(0, u0), min(W - 1, u1)
            v0, v1 = max(0, v0), min(H - 1, v1)
            mask = np.zeros((H, W), np.uint8)
            if u1 > u0 and v1 > v0:
                mask[v0:v1, u0:u1] = 255
        cv2.imwrite(str(out / f"{i:06d}.png"), mask)
    print(f"wrote {len(cam_pos)} masks to {out}")


if __name__ == "__main__":
    main()
