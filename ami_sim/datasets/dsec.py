"""DSEC RGB sequence and calibration readers with timestamp-preserving semantics."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Iterator, Literal

import imageio.v3 as iio
import numpy as np
import yaml

from ami_sim.types import CameraIntrinsics

CameraSide = Literal["left", "right"]
_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
_UNIT_SCALE = {"s": 1.0, "ms": 1e-3, "us": 1e-6, "ns": 1e-9}


def _timestamp_scale(values: np.ndarray, unit: str) -> float:
    if unit != "auto":
        try:
            return _UNIT_SCALE[unit]
        except KeyError as error:
            raise ValueError(f"Unsupported timestamp unit {unit!r}.") from error
    if len(values) < 2:
        return 1.0
    median_step = float(np.median(np.diff(values.astype(np.float64))))
    # DSEC timestamps are commonly integer microseconds. Infer from increments,
    # not absolute time, so a zero-based and epoch-based sequence behave alike.
    if median_step >= 1e6:
        return 1e-9
    if median_step >= 1e3:
        return 1e-6
    if median_step >= 1.0:
        return 1e-3
    return 1.0


def _read_timestamps(path: Path) -> np.ndarray:
    if path.suffix == ".npy":
        values = np.load(path)
    elif path.suffix == ".npz":
        payload = np.load(path)
        key = "timestamps" if "timestamps" in payload else list(payload.files)[0]
        values = payload[key]
    else:
        try:
            values = np.loadtxt(path, delimiter="," if path.suffix == ".csv" else None)
        except ValueError:
            # Some exports include a header or a frame-index column. Keep the
            # last numeric field on every non-empty line instead of discarding
            # the sequence or coercing timestamps to a constant FPS.
            rows: list[float] = []
            for line in path.read_text(encoding="utf-8").splitlines():
                fields = [field for field in re.split(r"[,\\s]+", line.strip()) if field]
                numeric = []
                for field in fields:
                    try:
                        numeric.append(float(field))
                    except ValueError:
                        continue
                if numeric:
                    rows.append(numeric[-1])
            if not rows:
                raise ValueError(f"No numeric timestamps found in {path}.")
            values = np.asarray(rows, dtype=np.float64)
        if values.ndim > 1:
            values = values[:, -1]
    return np.asarray(values).reshape(-1)


def _intrinsics_from_mapping(payload: dict, camera: str) -> CameraIntrinsics:
    camera_key = {"left": "cam0", "right": "cam1"}.get(camera, camera)
    node = payload.get(camera_key, payload.get(camera, payload))
    intrinsics = node.get("intrinsics") if isinstance(node, dict) else None
    resolution = node.get("resolution") if isinstance(node, dict) else None
    if intrinsics is None and isinstance(node, dict):
        matrix = node.get("camera_matrix", node.get("K"))
        if isinstance(matrix, dict):
            matrix = matrix.get("data")
        if matrix is not None:
            matrix = np.asarray(matrix, dtype=float).reshape(3, 3)
            intrinsics = [matrix[0, 0], matrix[1, 1], matrix[0, 2], matrix[1, 2]]
    if intrinsics is None or resolution is None:
        raise ValueError(
            f"Could not find intrinsics and resolution for {camera_key!r} in calibration YAML."
        )
    fx, fy, cx, cy = map(float, intrinsics[:4])
    width, height = map(int, resolution[:2])
    return CameraIntrinsics(width, height, fx, fy, cx, cy)


@dataclass(frozen=True)
class DSECCameraCalibration:
    camera: str
    intrinsics: CameraIntrinsics
    source_path: Path

    @classmethod
    def from_file(cls, path: str | Path, camera: str = "left") -> "DSECCameraCalibration":
        source = Path(path)
        with source.open("r", encoding="utf-8") as handle:
            payload = yaml.safe_load(handle)
        if not isinstance(payload, dict):
            raise ValueError(f"Calibration file {source} does not contain a mapping.")
        return cls(camera=camera, intrinsics=_intrinsics_from_mapping(payload, camera), source_path=source)


@dataclass
class DSECRGBSequence:
    """A timestamped DSEC RGB camera sequence without any FPS assumption."""

    rgb_dir: Path
    frame_paths: list[Path]
    timestamps: np.ndarray
    raw_timestamps: np.ndarray
    timestamp_unit: str
    sequence_name: str
    camera: CameraSide
    resolution: tuple[int, int]
    calibration: DSECCameraCalibration | None = None

    @classmethod
    def from_paths(
        cls,
        rgb_dir: str | Path,
        timestamps_path: str | Path,
        *,
        sequence_name: str | None = None,
        camera: CameraSide = "left",
        timestamp_unit: str = "auto",
        calibration_path: str | Path | None = None,
        max_frames: int | None = None,
    ) -> "DSECRGBSequence":
        rgb_dir = Path(rgb_dir)
        frame_paths = sorted(path for path in rgb_dir.iterdir() if path.suffix.lower() in _IMAGE_SUFFIXES)
        if max_frames is not None:
            frame_paths = frame_paths[:max_frames]
        if not frame_paths:
            raise FileNotFoundError(f"No RGB images found in {rgb_dir}.")
        raw_timestamps = _read_timestamps(Path(timestamps_path))
        if len(raw_timestamps) < len(frame_paths):
            raise ValueError("Timestamp file has fewer entries than RGB frames.")
        raw_timestamps = raw_timestamps[:len(frame_paths)]
        timestamps = raw_timestamps.astype(np.float64) * _timestamp_scale(raw_timestamps, timestamp_unit)
        if len(timestamps) > 1 and not np.all(np.diff(timestamps) > 0):
            raise ValueError("DSEC RGB timestamps must be strictly increasing.")
        first_frame = np.asarray(iio.imread(frame_paths[0]))
        if first_frame.ndim != 3 or first_frame.shape[2] < 3:
            raise ValueError("DSEC RGB frames must be color images with at least three channels.")
        height, width = first_frame.shape[:2]
        calibration = (
            DSECCameraCalibration.from_file(calibration_path, camera)
            if calibration_path is not None else None
        )
        if calibration is not None and (calibration.intrinsics.width, calibration.intrinsics.height) != (width, height):
            calibration = DSECCameraCalibration(
                calibration.camera,
                calibration.intrinsics.resized(width, height),
                calibration.source_path,
            )
        return cls(
            rgb_dir=rgb_dir,
            frame_paths=frame_paths,
            timestamps=timestamps,
            raw_timestamps=raw_timestamps,
            timestamp_unit=timestamp_unit,
            sequence_name=sequence_name or rgb_dir.name,
            camera=camera,
            resolution=(width, height),
            calibration=calibration,
        )

    def __len__(self) -> int:
        return len(self.frame_paths)

    def frame(self, index: int) -> np.ndarray:
        frame = np.asarray(iio.imread(self.frame_paths[index]))
        if frame.shape[:2] != (self.resolution[1], self.resolution[0]):
            raise ValueError(f"Frame {self.frame_paths[index]} has a different resolution.")
        return frame[..., :3]

    def iter_frames(self) -> Iterator[np.ndarray]:
        for index in range(len(self)):
            yield self.frame(index)
