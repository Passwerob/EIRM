#!/usr/bin/env python3
"""Generate the missing counter-motion branch of the motion sweep."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from scripts.generate_comotion_scenes import _render_scene
from scripts.generate_motion_excitation_sweep import SCALES


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("data/counter_motion_excitation_sweep"))
    parser.add_argument("--seeds", type=int, nargs="+", default=(101, 102, 103))
    parser.add_argument("--frames", type=int, default=64)
    parser.add_argument("--fps", type=float, default=120.0)
    parser.add_argument("--size", type=int, default=64)
    args = parser.parse_args()
    records = []
    for seed in args.seeds:
        for scale in SCALES:
            scene_name = f"test_seed{seed:02d}_counter_s{scale:.2f}"
            meta = _render_scene(
                args.output,
                split="test",
                seed=seed,
                condition="counter_motion",
                frames=args.frames,
                fps=args.fps,
                size=args.size,
                motion_profile="smooth",
                motion_scale=scale,
                scene_name=scene_name,
            )
            meta["sweep"] = "counter"
            records.append(meta)
    manifest = {
        "dataset": "ami_motion_excitation_counter_v1",
        "scientific_role": "controlled 64x64 counter-motion branch for the motion-excitation figure",
        "seeds": list(args.seeds),
        "scales": list(SCALES),
        "frames": args.frames,
        "fps": args.fps,
        "size": args.size,
        "records": records,
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (args.output / "scenes.json").write_text(json.dumps(records, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "scenes": len(records)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
