#!/usr/bin/env python3
"""Extract selected stored IMU bags from DSEC's large ZIP via HTTP ranges.

The official ``lidar_imu.zip`` contains uncompressed per-area bags. This
script reads only the ZIP directory and requested byte ranges, avoiding a
full 28 GB download.
"""

from __future__ import annotations

import argparse
import struct
import urllib.request
from pathlib import Path


URL = "https://download.ifi.uzh.ch/rpg/DSEC/lidar_imu.zip"
ZIP64_EOCD = b"PK\x06\x06"
CENTRAL = b"PK\x01\x02"
LOCAL = b"PK\x03\x04"


def fetch_range(url: str, start: int, end: int) -> bytes:
    request = urllib.request.Request(url, headers={"Range": f"bytes={start}-{end}"})
    with urllib.request.urlopen(request, timeout=120) as response:
        data = response.read()
    expected = end - start + 1
    if len(data) != expected:
        raise RuntimeError(f"Range request returned {len(data)} bytes, expected {expected}.")
    return data


def central_entries() -> tuple[int, dict[str, tuple[int, int, int]]]:
    tail_size = 1 << 20
    request = urllib.request.Request(URL, headers={"Range": f"bytes=-{tail_size}"})
    with urllib.request.urlopen(request, timeout=120) as response:
        tail = response.read()
        content_range = response.headers.get("Content-Range", "")
    if "/" not in content_range:
        raise RuntimeError("DSEC server did not return Content-Range for the ZIP tail.")
    total_size = int(content_range.rsplit("/", 1)[1])
    eocd = tail.rfind(ZIP64_EOCD)
    if eocd < 0:
        raise RuntimeError("ZIP64 end-of-directory record was not found.")
    _, _, _, _, _, _, _, _, directory_size, directory_offset = struct.unpack_from("<4sQ2H2I2Q2Q", tail, eocd)
    directory = fetch_range(URL, directory_offset, directory_offset + directory_size - 1)
    entries: dict[str, tuple[int, int, int]] = {}
    position = 0
    while position < len(directory):
        if directory[position : position + 4] != CENTRAL:
            raise RuntimeError(f"Invalid central directory signature at {position}.")
        values = struct.unpack_from("<4s6H3I5H2I", directory, position)
        _, _, _, _, _, _, _, _, compressed_size, _, name_len, extra_len, comment_len, _, _, _, local_offset = values
        name_start = position + 46
        name = directory[name_start : name_start + name_len].decode("utf-8")
        extra = directory[name_start + name_len : name_start + name_len + extra_len]
        if local_offset == 0xFFFFFFFF:
            offset = 0
            cursor = 0
            while cursor + 4 <= len(extra):
                field, field_len = struct.unpack_from("<HH", extra, cursor)
                payload = extra[cursor + 4 : cursor + 4 + field_len]
                cursor += 4 + field_len
                if field == 1:
                    # Uncompressed files need only the local-header offset here.
                    offset = struct.unpack_from("<Q", payload, 0)[0]
                    break
            local_offset = offset
        entries[name] = (int(local_offset), int(compressed_size), int(total_size))
        position += 46 + name_len + extra_len + comment_len
    return total_size, entries


def extract_entry(name: str, destination: Path, entries: dict[str, tuple[int, int, int]]) -> None:
    if name not in entries:
        raise KeyError(f"Entry {name!r} was not found in DSEC lidar_imu.zip.")
    local_offset, compressed_size, _ = entries[name]
    local_header = fetch_range(URL, local_offset, local_offset + 30 - 1)
    if local_header[:4] != LOCAL:
        raise RuntimeError(f"Invalid local header for {name}.")
    _, _, _, method, _, _, _, _, _, name_len, extra_len = struct.unpack("<4s5H3I2H", local_header)
    if method != 0:
        raise RuntimeError(f"Entry {name} is compressed; range extraction expects stored data.")
    data_start = local_offset + 30 + name_len + extra_len
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")
    completed = temporary.stat().st_size if temporary.exists() else 0
    if completed > compressed_size:
        raise RuntimeError(f"Partial file {temporary} is larger than the ZIP entry.")
    with temporary.open("ab") as handle:
        chunk_size = 32 << 20
        for start in range(completed, compressed_size, chunk_size):
            end = min(compressed_size, start + chunk_size) - 1
            handle.write(fetch_range(URL, data_start + start, data_start + end))
    temporary.replace(destination)
    if destination.stat().st_size != compressed_size:
        raise RuntimeError(f"Extracted {destination} has an unexpected size.")
    print(f"wrote {destination} ({compressed_size / (1 << 30):.2f} GiB)")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--area", action="append", required=True, help="DSEC IMU area, e.g. interlaken_00 or zurich_city_02; repeat this option.")
    args = parser.parse_args()
    _, entries = central_entries()
    for area in args.area:
        extract_entry(f"data/{area}/lidar_imu.bag", args.output_root / area / "lidar_imu.bag", entries)
        extract_entry(f"data/{area}/cam_to_lidar.yaml", args.output_root / area / "cam_to_lidar.yaml", entries)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
