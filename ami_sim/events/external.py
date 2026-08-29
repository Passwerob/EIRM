"""Subprocess adapter shared by official DVS-Voltmeter and v2e integrations."""

from __future__ import annotations

import shlex
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import imageio.v3 as iio
import numpy as np

from .base import EventSimulatorBackend, normalise_events, read_events_h5
from ami_sim.types import EventStream


class ExternalEventSimulatorBackend(EventSimulatorBackend):
    def __init__(self, command: str | None, *, output_time_unit: str = "s", working_directory: str | Path | None = None) -> None:
        self.command = command
        self.output_time_unit = output_time_unit
        self.working_directory = Path(working_directory) if working_directory else None

    def _read_output(self, path: Path) -> EventStream:
        if path.suffix in {".h5", ".hdf5"}:
            return read_events_h5(path, time_unit=self.output_time_unit)
        if path.suffix == ".npz":
            payload = np.load(path)
            return normalise_events(payload["x"], payload["y"], payload["t"], payload["p"], time_unit=self.output_time_unit)
        if path.suffix == ".npy":
            payload = np.asarray(np.load(path))
            if payload.ndim != 2 or payload.shape[1] < 4:
                raise ValueError("NPY event output must have columns x, y, t, p.")
            return normalise_events(payload[:, 0], payload[:, 1], payload[:, 2], payload[:, 3], time_unit=self.output_time_unit)
        if path.suffix == ".csv":
            payload = np.loadtxt(path, delimiter=",")
            return normalise_events(payload[:, 0], payload[:, 1], payload[:, 2], payload[:, 3], time_unit=self.output_time_unit)
        raise ValueError(f"Unsupported external event output format: {path}")

    def simulate(self, frames: np.ndarray, timestamps: np.ndarray, config: dict[str, Any] | None = None) -> EventStream:
        del config
        if not self.command:
            raise RuntimeError(
                f"{self.name} is an external adapter and no --event-command was supplied. "
                "Install the official simulator and provide its command explicitly."
            )
        frames = np.asarray(frames, dtype=np.float32)
        timestamps = np.asarray(timestamps, dtype=np.float64)
        if frames.ndim != 3 or len(frames) != len(timestamps):
            raise ValueError("External event simulation expects [T,H,W] frames and matching timestamps.")
        with tempfile.TemporaryDirectory(prefix=f"{self.name}_", dir=self.working_directory) as temporary:
            root = Path(temporary)
            frames_dir = root / "frames"
            frames_dir.mkdir()
            for index, frame in enumerate(frames):
                np.save(frames_dir / f"{index:06d}.npy", frame)
            timestamps_path = root / "timestamps.npy"
            output_path = root / "events.h5"
            np.save(timestamps_path, timestamps)
            rendered = self.command.format(
                frames_dir=str(frames_dir), timestamps_path=str(timestamps_path), output_path=str(output_path)
            )
            command = shlex.split(rendered)
            subprocess.run(command, cwd=self.working_directory, check=True)
            if not output_path.exists():
                candidates = list(root.glob("events.*"))
                if len(candidates) != 1:
                    raise FileNotFoundError("External event command did not create {output_path} or a unique events.* file.")
                output_path = candidates[0]
            return self._read_output(output_path)

    def metadata(self) -> dict[str, object]:
        return {**super().metadata(), "command": self.command, "output_time_unit": self.output_time_unit}
