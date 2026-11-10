#!/usr/bin/env python3
"""Render a motion-scale pack for the motion–excitation response experiment.

Does not overwrite data/comotion_controlled_v2. Procedural 64x64 renderer
with an explicit motion_scale and co-motion mismatch alpha. Not EVEYES.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from scripts.generate_comotion_scenes import _render_scene

SCALES = (0.00, 0.10, 0.25, 0.50, 0.75, 1.00, 1.50, 2.00, 3.00, 4.00, 6.00)
ALPHAS = (0.00, 0.25, 0.50, 0.75, 0.90, 0.95, 1.00, 1.05, 1.10, 1.25, 1.50)
GRID_SCALES = (0.00, 0.25, 0.50, 1.00, 2.00, 4.00)


def _name(sweep: str, seed: int, *, scale: float | None = None, alpha: float | None = None, cam: float | None = None, obj: float | None = None) -> str:
    if sweep == "mismatch":
        return f"test_seed{seed:02d}_mismatch_a{alpha:.2f}_s{scale:.2f}"
    if sweep == "grid":
        return f"test_seed{seed:02d}_grid_c{cam:.2f}_o{obj:.2f}"
    return f"test_seed{seed:02d}_{sweep}_s{scale:.2f}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("data/motion_excitation_sweep"))
    parser.add_argument("--seeds", type=int, nargs="+", default=(101, 102, 103))
    parser.add_argument("--frames", type=int, default=64)
    parser.add_argument("--fps", type=float, default=120.0)
    parser.add_argument("--size", type=int, default=64)
    parser.add_argument("--skip-grid", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, object]] = []

    for seed in args.seeds:
        for scale in SCALES:
            for sweep, condition in (("camera", "camera_only"), ("object", "object_only")):
                meta = _render_scene(
                    args.output,
                    split="test",
                    seed=seed,
                    condition=condition,
                    frames=args.frames,
                    fps=args.fps,
                    size=args.size,
                    motion_profile="smooth",
                    motion_scale=scale,
                    scene_name=_name(sweep, seed, scale=scale),
                )
                meta["sweep"] = sweep
                records.append(meta)
        for alpha in ALPHAS:
            meta = _render_scene(
                args.output,
                split="test",
                seed=seed,
                condition="co_motion",
                frames=args.frames,
                fps=args.fps,
                size=args.size,
                motion_profile="smooth",
                motion_scale=1.0,
                object_alpha=alpha,
                scene_name=_name("mismatch", seed, scale=1.0, alpha=alpha),
            )
            meta["sweep"] = "mismatch"
            records.append(meta)
        if not args.skip_grid:
            for cam in GRID_SCALES:
                for obj in GRID_SCALES:
                    # Independent same-direction warps: background=cam, object=obj * same traj.
                    # Implemented as mismatch with camera scale = cam and alpha such that
                    # (1-alpha)*cam = obj  =>  alpha = 1 - obj/cam when cam>0.
                    if cam == 0.0:
                        meta = _render_scene(
                            args.output,
                            split="test",
                            seed=seed,
                            condition="object_only",
                            frames=args.frames,
                            fps=args.fps,
                            size=args.size,
                            motion_profile="smooth",
                            motion_scale=obj,
                            scene_name=_name("grid", seed, cam=cam, obj=obj),
                        )
                    else:
                        alpha = 1.0 - (obj / cam)
                        meta = _render_scene(
                            args.output,
                            split="test",
                            seed=seed,
                            condition="co_motion",
                            frames=args.frames,
                            fps=args.fps,
                            size=args.size,
                            motion_profile="smooth",
                            motion_scale=cam,
                            object_alpha=alpha,
                            scene_name=_name("grid", seed, cam=cam, obj=obj),
                        )
                    meta["sweep"] = "grid"
                    meta["grid_camera_scale"] = float(cam)
                    meta["grid_object_scale"] = float(obj)
                    records.append(meta)

    manifest = {
        "dataset": "ami_motion_excitation_sweep_v1",
        "scientific_role": "controlled 64x64 motion sweep, not EVEYES / not photorealistic Table 1",
        "renderer": "scripts/generate_comotion_scenes.py _render_scene",
        "prism_angle_deg_for_eval": 2.0,
        "seeds": list(args.seeds),
        "scales": list(SCALES),
        "alphas": list(ALPHAS),
        "grid_scales": list(GRID_SCALES),
        "frames": args.frames,
        "fps": args.fps,
        "size": args.size,
        "motion_unit": "image-plane translation amplitude relative to the original pack (~7 px)",
        "records": records,
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "scenes": len(records)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
