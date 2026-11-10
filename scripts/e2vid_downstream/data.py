"""Load cached reference-plane events and RGB luminance targets."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from protocol import NUM_BINS, SEQ_LEN, cache_path
from voxels import events_to_voxel_grid, normalize_nonzero


@dataclass
class SequenceCache:
    scene_id: str
    condition: str
    x: np.ndarray
    y: np.ndarray
    t: np.ndarray
    p: np.ndarray
    ptr: np.ndarray
    rpm: np.ndarray
    t0: np.ndarray
    t1: np.ndarray
    visual_motion: np.ndarray
    luminance: np.ndarray
    width: int
    height: int

    @property
    def n_windows(self) -> int:
        return int(len(self.rpm))

    def window_events(self, index: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        a, b = int(self.ptr[index]), int(self.ptr[index + 1])
        return self.x[a:b], self.y[a:b], self.t[a:b], self.p[a:b]

    def voxel(self, index: int, *, x0: int = 0, y0: int = 0, width: int | None = None, height: int | None = None) -> np.ndarray:
        width = self.width if width is None else width
        height = self.height if height is None else height
        x, y, t, p = self.window_events(index)
        if x0 or y0:
            x = x.astype(np.int32) - x0
            y = y.astype(np.int32) - y0
        return normalize_nonzero(
            events_to_voxel_grid(t, x, y, p, num_bins=NUM_BINS, width=width, height=height)
        )

    def target(self, index: int) -> np.ndarray:
        """GT for events in (t_i, t_{i+1}] is the luminance at frame i+1."""
        return np.asarray(self.luminance[index + 1], dtype=np.float32)


def load_sequence(cache_dir: Path, scene_id: str, condition: str) -> SequenceCache:
    path = cache_path(cache_dir, scene_id, condition)
    with np.load(path, allow_pickle=True) as z:
        width = int(z["width"])
        height = int(z["height"])
        payload = {
            "x": z["x"],
            "y": z["y"],
            "t": z["t"],
            "p": z["p"],
            "ptr": z["window_ptr"],
            "rpm": z["rpm"],
            "t0": z["t0"],
            "t1": z["t1"],
            "visual_motion": z["visual_motion"],
        }
    gray_path = cache_dir / f"{scene_id.replace('/', '__')}__luminance.npz"
    with np.load(gray_path) as g:
        luminance = np.asarray(g["luminance"], dtype=np.float32)
    return SequenceCache(
        scene_id=scene_id,
        condition=condition,
        width=width,
        height=height,
        luminance=luminance,
        **payload,
    )


def sample_crop(width: int, height: int, crop: int, rng: np.random.Generator) -> tuple[int, int]:
    if width == crop and height == crop:
        return 0, 0
    x0 = int(rng.integers(0, max(width - crop + 1, 1)))
    y0 = int(rng.integers(0, max(height - crop + 1, 1)))
    return x0, y0


def sequence_item(cache: SequenceCache, start: int, *, crop: int, x0: int, y0: int, seq_len: int = SEQ_LEN) -> dict:
    voxels = []
    targets = []
    rpms = []
    motions = []
    for offset in range(seq_len):
        index = start + offset
        voxel = cache.voxel(index, x0=x0, y0=y0, width=crop, height=crop)
        target = cache.target(index)[y0 : y0 + crop, x0 : x0 + crop]
        voxels.append(voxel)
        targets.append(target)
        rpms.append(float(cache.rpm[index]))
        motions.append(float(cache.visual_motion[index]))
    return {
        "voxel": np.stack(voxels, axis=0).astype(np.float32),
        "target": np.stack(targets, axis=0).astype(np.float32),
        "rpm": np.asarray(rpms, dtype=np.float32),
        "visual_motion": np.asarray(motions, dtype=np.float32),
        "scene_id": cache.scene_id,
        "condition": cache.condition,
        "start": start,
        "x0": x0,
        "y0": y0,
        "t0": float(cache.t0[start]),
        "t1": float(cache.t1[start + seq_len - 1]),
    }
