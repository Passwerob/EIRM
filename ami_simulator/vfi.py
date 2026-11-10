"""Video interpolation backends with explicit arbitrary-timestep semantics."""

from __future__ import annotations

import inspect
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as functional


def validate_targets(targets: np.ndarray) -> np.ndarray:
    values = np.asarray(targets, dtype=np.float64).reshape(-1)
    if np.any((values <= 0.0) | (values >= 1.0)) or (len(values) > 1 and not np.all(np.diff(values) > 0.0)):
        raise ValueError("VFI targets must be strictly increasing normalized values inside (0, 1).")
    return values


class DebugLinearVFIBackend:
    """Non-research interpolation backend for deterministic tests only."""

    name = "debug_linear"

    def interpolate(self, first: np.ndarray, second: np.ndarray, targets: np.ndarray) -> np.ndarray:
        values = validate_targets(targets)
        left = np.asarray(first, dtype=np.float32)[..., :3]
        right = np.asarray(second, dtype=np.float32)[..., :3]
        if left.shape != right.shape:
            raise ValueError("VFI endpoint frames must have identical shapes.")
        if left.max(initial=0.0) > 1.0:
            left = left / 255.0
        if right.max(initial=0.0) > 1.0:
            right = right / 255.0
        return np.stack([left * (1.0 - value) + right * value for value in values]) if len(values) else np.empty((0, *left.shape), np.float32)

    def metadata(self) -> dict[str, object]:
        return {"vfi_backend": self.name, "recursive_interpolation": False, "debug_only": True}


class PracticalRIFEBackend:
    """Adapter for Practical-RIFE's direct arbitrary-timestep API."""

    name = "practical_rife"

    def __init__(self, repository_root: str | Path, checkpoint: str | Path | None = None, *, device: str | None = None, scale: float = 1.0, inference_batch_size: int = 8, fp16: bool = False) -> None:
        self.repository_root = Path(repository_root)
        self.checkpoint = Path(checkpoint) if checkpoint else None
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.scale = float(scale)
        self.inference_batch_size = int(inference_batch_size)
        self.fp16 = bool(fp16)
        if self.inference_batch_size <= 0 or self.scale <= 0.0:
            raise ValueError("RIFE scale and inference_batch_size must be positive.")
        if self.fp16 and self.device.type != "cuda":
            raise ValueError("Practical-RIFE fp16 requires CUDA.")
        self._model: object | None = None
        self._padding_multiple = 64

    def _load_model(self) -> object:
        if self._model is not None:
            return self._model
        if not self.repository_root.is_dir():
            raise FileNotFoundError(f"Practical-RIFE checkout is missing: {self.repository_root}")
        sys.path.insert(0, str(self.repository_root))
        try:
            try:
                from model.RIFE import Model  # type: ignore[import-not-found]
                source = "model.RIFE"
            except ImportError:
                from train_log.RIFE_HDv3 import Model  # type: ignore[import-not-found]
                source = "train_log.RIFE_HDv3"
        finally:
            sys.path.pop(0)
        try:
            model = Model(arbitrary=True)
            arbitrary = True
        except TypeError:
            model = Model()
            arbitrary = False
        checkpoint = self.checkpoint or self.repository_root / "train_log"
        if not checkpoint.is_dir():
            raise FileNotFoundError(f"Practical-RIFE checkpoint is missing: {checkpoint}")
        model.load_model(str(checkpoint), -1)
        model.eval()
        if self.device.type == "cuda":
            model.device()
            if hasattr(model, "flownet"):
                model.flownet.to(self.device)
                if self.fp16:
                    model.flownet.half()
        model_file = self.repository_root / "train_log" / "RIFE_HDv3.py"
        if model_file.is_file() and "[32/scale" in model_file.read_text(encoding="utf-8"):
            self._padding_multiple = 128
        self._model, self._model_source, self._arbitrary = model, source, arbitrary
        self._batched = source == "train_log.RIFE_HDv3" and hasattr(model, "flownet")
        return model

    @staticmethod
    def _timestep_parameter(model: object) -> str | None:
        parameters = inspect.signature(model.inference).parameters
        for name in ("timestep", "time", "t"):
            if name in parameters:
                return name
        return "timestep" if any(value.kind is inspect.Parameter.VAR_KEYWORD for value in parameters.values()) else None

    def _tensor(self, frame: np.ndarray) -> tuple[torch.Tensor, tuple[int, int]]:
        values = np.asarray(frame, dtype=np.float32)
        if values.ndim != 3 or values.shape[-1] < 3:
            raise ValueError("Practical-RIFE expects HWC RGB frames.")
        values = values[..., :3]
        if values.max(initial=0.0) > 1.0:
            values = values / 255.0
        dtype = torch.float16 if self.fp16 else torch.float32
        tensor = torch.from_numpy(values).permute(2, 0, 1).unsqueeze(0).to(self.device, dtype=dtype)
        height, width = values.shape[:2]
        pad_h = (self._padding_multiple - height % self._padding_multiple) % self._padding_multiple
        pad_w = (self._padding_multiple - width % self._padding_multiple) % self._padding_multiple
        return functional.pad(tensor, (0, pad_w, 0, pad_h), mode="replicate"), (height, width)

    def interpolate(self, first_frame: np.ndarray, second_frame: np.ndarray, targets: np.ndarray) -> np.ndarray:
        values = validate_targets(targets)
        if not len(values):
            return np.empty((0, *np.asarray(first_frame).shape), np.float32)
        model = self._load_model()
        parameter = self._timestep_parameter(model)
        if parameter is None:
            raise RuntimeError("Practical-RIFE checkout lacks a direct arbitrary-timestep API; recursive midpoint interpolation is disabled.")
        first, shape = self._tensor(first_frame)
        second, _ = self._tensor(second_frame)
        batch_size = self.inference_batch_size if self._batched and parameter == "timestep" else 1
        outputs: list[torch.Tensor] = []
        with torch.inference_mode():
            for begin in range(0, len(values), batch_size):
                batch = values[begin : begin + batch_size]
                count = len(batch)
                left = first.expand(count, -1, -1, -1) if count > 1 else first
                right = second.expand(count, -1, -1, -1) if count > 1 else second
                timestep: float | torch.Tensor = torch.as_tensor(batch, device=self.device, dtype=first.dtype).view(count, 1, 1, 1) if count > 1 else float(batch[0])
                scale = {"scale": self.scale} if "scale" in inspect.signature(model.inference).parameters else {}
                prediction = model.inference(left, right, **scale, **{parameter: timestep})
                if isinstance(prediction, (tuple, list)):
                    prediction = prediction[0]
                if prediction.ndim == 3:
                    prediction = prediction.unsqueeze(0)
                outputs.append(prediction[..., : shape[0], : shape[1]].float().permute(0, 2, 3, 1).cpu())
        return torch.cat(outputs).numpy()

    def metadata(self) -> dict[str, object]:
        return {"vfi_backend": self.name, "recursive_interpolation": False, "repository_root": str(self.repository_root), "checkpoint": str(self.checkpoint or self.repository_root / "train_log"), "device": str(self.device), "fp16": self.fp16, "model_source": getattr(self, "_model_source", None), "arbitrary_constructor": getattr(self, "_arbitrary", None)}


def resample_with_vfi(frames: np.ndarray, timestamps: np.ndarray, *, target_fps: float, backend: object) -> tuple[np.ndarray, np.ndarray]:
    """Query a VFI backend directly on every physical source interval."""
    source = np.asarray(frames)
    times = np.asarray(timestamps, dtype=np.float64).reshape(-1)
    if len(source) != len(times) or len(times) < 2 or not np.all(np.diff(times) > 0.0) or target_fps <= 0.0:
        raise ValueError("VFI resampling needs aligned frames, increasing timestamps, and positive target_fps.")
    output = [source[0].astype(np.float32) / (255.0 if source[0].max(initial=0.0) > 1.0 else 1.0)]
    output_times = [float(times[0])]
    for index, (start, end) in enumerate(zip(times[:-1], times[1:])):
        subdivisions = max(1, int(np.ceil((end - start) * target_fps)))
        fractions = np.arange(1, subdivisions, dtype=np.float64) / subdivisions
        if len(fractions):
            middle = backend.interpolate(source[index], source[index + 1], fractions)
            if len(middle) != len(fractions):
                raise ValueError("VFI backend returned an unexpected frame count.")
            output.extend(middle)
            output_times.extend((start + fractions * (end - start)).tolist())
        endpoint = source[index + 1].astype(np.float32)
        output.append(endpoint / (255.0 if endpoint.max(initial=0.0) > 1.0 else 1.0))
        output_times.append(float(end))
    result_times = np.asarray(output_times, dtype=np.float64)
    if not np.all(np.diff(result_times) > 0.0):
        raise RuntimeError("VFI resampling produced non-increasing timestamps.")
    return np.stack(output).astype(np.float32), result_times
