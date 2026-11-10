"""Create an explicitly labelled nominal timeline for RGB-only sequence exports."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


_IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff"}


def main() -> int:
    parser = argparse.ArgumentParser(description="Create a nominal timestamp CSV; this does not recover capture timestamps.")
    parser.add_argument("--image-dir", required=True, type=Path)
    parser.add_argument("--fps", required=True, type=float, help="Explicitly assumed source frame rate.")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.fps <= 0:
        raise SystemExit("--fps must be positive.")
    images = sorted(path for path in args.image_dir.iterdir() if path.suffix.lower() in _IMAGE_SUFFIXES)
    if len(images) < 2:
        raise SystemExit("At least two source images are required.")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["index", "timestamp_seconds", "timeline"])
        for index in range(len(images)):
            writer.writerow([index, f"{index / args.fps:.17g}", "nominal_from_explicit_fps"])
    print(f"Wrote {len(images)} nominal timestamps at {args.fps:g} FPS to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
