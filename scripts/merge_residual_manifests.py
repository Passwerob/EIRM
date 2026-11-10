#!/usr/bin/env python3
"""Merge residual datasets while preserving split and scene provenance."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _records(path: Path) -> list[dict[str, object]]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    values = manifest.get("records")
    if not isinstance(values, list):
        raise ValueError(f"{path} does not contain residual records")
    return [dict(record) for record in values]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--anchor-manifest", type=Path, required=True)
    parser.add_argument("--dagger-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    anchor = _records(args.anchor_manifest)
    dagger = _records(args.dagger_manifest)
    # Keep both train distributions, but use the anchor dataset as the source
    # for validation/test to prevent accidental evaluation on policy rollouts.
    train = [record for record in anchor + dagger if record["split"] == "train"]
    held_out = [record for record in anchor if record["split"] in ("valid", "test")]
    records = train + held_out
    feature_dims = {int(json.loads(path.read_text(encoding="utf-8")).get("feature_dim", 20)) for path in (args.anchor_manifest, args.dagger_manifest)}
    if len(feature_dims) != 1:
        raise ValueError(f"Anchor and DAgger feature dimensions differ: {feature_dims}")
    output = {
        "dataset": "AMI anchor residual mixed anchor plus DAgger dataset",
        "anchor_manifest": str(args.anchor_manifest),
        "dagger_manifest": str(args.dagger_manifest),
        "anchor_rpm": 180.0,
        "candidate_rpm": [0.0, 180.0, 360.0],
        "feature_dim": feature_dims.pop(),
        "records": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"train_records": len(train), "held_out_records": len(held_out), "output": str(args.output)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
