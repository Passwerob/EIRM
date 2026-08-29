"""Official DVS-Voltmeter adapters.

The subprocess adapter remains the compatibility path. ``DVSVoltmeterTorchBackend``
uses the same official simulator in-process and keeps its per-pixel stochastic
state on the selected torch device, avoiding temporary frame files and a child
process for CUDA runs.
"""

from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np
import torch

from ami_sim.types import EventStream
from .base import EventSimulatorBackend
from .external import ExternalEventSimulatorBackend


class DVSVoltmeterBackend(ExternalEventSimulatorBackend):
    name = "dvs_voltmeter"

    def __init__(self, command: str | None = None, *, output_time_unit: str = "s", working_directory: str | Path | None = None) -> None:
        super().__init__(command, output_time_unit=output_time_unit, working_directory=working_directory)


_SENSOR_PARAMETERS = {
    "DVS346": [0.00018 * 29250, 20, 0.0001, 1e-7, 5e-9, 0.00001],
    "DVS240": [0.000094 * 47065, 23, 0.0002, 1e-7, 5e-8, 0.00001],
}


class DVSVoltmeterTorchBackend(EventSimulatorBackend):
    """In-process official DVS-Voltmeter backend with GPU-resident state."""

    name = "dvs_voltmeter_torch"
    supports_torch = True

    def __init__(
        self,
        dvs_root: str | Path,
        *,
        camera_type: str = "DVS346",
        device: str | torch.device | None = None,
        transfer_chunk_frames: int = 16,
        seed: int = 0,
        sort_each_frame: bool = False,
        precision: str = "float64",
    ) -> None:
        root = Path(dvs_root)
        if camera_type not in _SENSOR_PARAMETERS:
            raise ValueError(f"Unsupported DVS camera type: {camera_type}")
        if transfer_chunk_frames <= 0:
            raise ValueError("transfer_chunk_frames must be positive.")
        self.dvs_root = root
        self.camera_type = camera_type
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.transfer_chunk_frames = int(transfer_chunk_frames)
        self.seed = int(seed)
        self.sort_each_frame = bool(sort_each_frame)
        if precision not in {"float32", "float64"}:
            raise ValueError("precision must be float32 or float64")
        self.precision = precision

    def _event_simulator(self):
        root = self.dvs_root.resolve()
        if not (root / "src" / "simulator.py").is_file():
            raise FileNotFoundError(f"DVS-Voltmeter checkout is missing its EventSim source: {root}")
        root_text = str(root)
        if root_text not in sys.path:
            sys.path.insert(0, root_text)
        from src.config import cfg  # type: ignore[import-not-found]
        from src.simulator import EventSim  # type: ignore[import-not-found]

        cfg.SENSOR.CAMERA_TYPE = self.camera_type
        cfg.SENSOR.K = _SENSOR_PARAMETERS[self.camera_type]
        # The official class only uses these names to construct a legacy output
        # path; the in-process backend owns serialization and never writes it.
        return EventSim(
            cfg=cfg,
            output_folder=".",
            video_name="eirm_inprocess",
            device=self.device,
            sort_events=self.sort_each_frame,
            compute_dtype=torch.float32 if self.precision == "float32" else torch.float64,
        )

    @staticmethod
    def _events_to_cpu(chunks: list[torch.Tensor]) -> list[np.ndarray]:
        if not chunks:
            return []
        merged = torch.cat(chunks, dim=0)
        order = torch.argsort(merged[:, 0], stable=True)
        values = merged[order].detach().cpu().numpy()
        return [values[:, 0], values[:, 1], values[:, 2], values[:, 3]]

    def simulate(self, frames: np.ndarray | torch.Tensor, timestamps: np.ndarray, config: dict[str, object] | None = None) -> EventStream:
        del config
        if isinstance(frames, torch.Tensor):
            tensor_frames = frames
        else:
            tensor_frames = torch.from_numpy(np.asarray(frames, dtype=np.float32))
        if tensor_frames.ndim != 3:
            raise ValueError("DVS-Voltmeter expects [T,H,W] intensity frames.")
        tensor_frames = tensor_frames.to(device=self.device, dtype=torch.float32, non_blocking=True)
        values = np.asarray(timestamps, dtype=np.float64).reshape(-1)
        if len(tensor_frames) != len(values):
            raise ValueError("Frame count does not match timestamp count.")
        if len(values) < 2 or not np.all(np.diff(values) > 0):
            raise ValueError("DVS-Voltmeter timestamps must be strictly increasing.")

        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        if self.device.type == "cuda":
            torch.cuda.manual_seed_all(self.seed)
        simulator = self._event_simulator()
        pending: list[torch.Tensor] = []
        cpu_chunks: list[np.ndarray] = []

        def flush() -> None:
            nonlocal pending
            columns = self._events_to_cpu(pending)
            if columns:
                cpu_chunks.append(np.stack(columns, axis=1))
            pending = []

        with torch.inference_mode():
            for index, (frame, timestamp) in enumerate(zip(tensor_frames, values)):
                image_u8 = torch.round(torch.clamp(frame, 0.0, 1.0) * 255.0).to(torch.uint8)
                event = simulator.generate_events(
                    image_u8,
                    int(np.rint(timestamp * 1e6)),
                    return_torch=True,
                )
                if isinstance(event, torch.Tensor) and event.numel():
                    pending.append(event)
                if (index + 1) % self.transfer_chunk_frames == 0:
                    flush()
        flush()
        simulator.reset()

        if not cpu_chunks:
            return EventStream.empty()
        values_us = np.concatenate(cpu_chunks, axis=0)
        order = np.argsort(values_us[:, 0], kind="stable")
        values_us = values_us[order]
        return EventStream(
            x=values_us[:, 1],
            y=values_us[:, 2],
            t=values_us[:, 0].astype(np.float64) * 1e-6,
            p=np.where(values_us[:, 3] > 0, 1, -1),
        )

    def metadata(self) -> dict[str, object]:
        return {
            **super().metadata(),
            "dvs_voltmeter_root": str(self.dvs_root),
            "camera_type": self.camera_type,
            "device": str(self.device),
            "transfer_chunk_frames": self.transfer_chunk_frames,
            "random_seed": self.seed,
            "sort_each_frame": self.sort_each_frame,
            "precision": self.precision,
            "input_intensity_range": "[0,1] converted to uint8 [0,255] on selected device",
            "input_timestamp_unit": "seconds converted to rounded microseconds",
        }
