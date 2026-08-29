"""Adapter for the official hzwer/Practical-RIFE implementation."""

from __future__ import annotations

import inspect
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .base import VFIBackend, validate_targets


class PracticalRIFEBackend(VFIBackend):
    name = "practical_rife"

    def __init__(
        self,
        repository_root: str | Path,
        checkpoint: str | Path | None = None,
        *,
        scale: float = 1.0,
        device: str | None = None,
        inference_batch_size: int = 8,
        fp16: bool = False,
    ) -> None:
        self.repository_root = Path(repository_root)
        self.checkpoint = Path(checkpoint) if checkpoint else None
        self.scale = scale
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        if inference_batch_size <= 0:
            raise ValueError("inference_batch_size must be positive.")
        self.inference_batch_size = int(inference_batch_size)
        self.fp16 = bool(fp16)
        if self.fp16 and self.device.type != "cuda":
            raise ValueError("Practical-RIFE fp16 mode requires a CUDA device.")
        self._model: object | None = None
        self._padding_multiple = 64

    def _load_model(self) -> object:
        if self._model is not None:
            return self._model
        if not self.repository_root.is_dir():
            raise FileNotFoundError(
                f"Practical-RIFE checkout is missing: {self.repository_root}. "
                "Install https://github.com/hzwer/Practical-RIFE and pass --practical-rife-root."
            )
        sys.path.insert(0, str(self.repository_root))
        try:
            from model.RIFE import Model  # type: ignore[import-not-found]
            model_source = "model.RIFE"
        except ImportError as error:
            try:
                # Official 4.x checkpoint archives carry their inference module
                # alongside flownet.pkl in train_log rather than in model/.
                from train_log.RIFE_HDv3 import Model  # type: ignore[import-not-found]
                model_source = "train_log.RIFE_HDv3"
            except ImportError as fallback_error:
                raise ImportError(
                    f"Could not import model.RIFE or train_log.RIFE_HDv3 from {self.repository_root}."
                ) from fallback_error
        # Practical-RIFE exposes arbitrary-timestep inference through the
        # `arbitrary` constructor flag in current official revisions. Older
        # revisions may not accept it; they are still loaded for inspection but
        # will be rejected below if no direct timestep API is present.
        try:
            model = Model(arbitrary=True)
            arbitrary_constructor = True
        except TypeError:
            model = Model()
            arbitrary_constructor = False
        checkpoint = self.checkpoint or self.repository_root / "train_log"
        if not checkpoint.is_dir():
            raise FileNotFoundError(f"Practical-RIFE checkpoint is missing: {checkpoint}.")
        if model_source == "train_log.RIFE_HDv3" and not (checkpoint / "flownet.pkl").is_file():
            raise FileNotFoundError(f"Practical-RIFE checkpoint weight is missing: {checkpoint / 'flownet.pkl'}.")
        model.load_model(str(checkpoint), -1)
        model.eval()
        if self.device.type == "cuda":
            model.device()
            # The upstream Model.device() uses its module-level default CUDA
                # device. EIRM may explicitly select cuda:1, so move the actual
            # network to the adapter's requested device as the final step.
            if hasattr(model, "flownet"):
                model.flownet.to(self.device)
            if self.fp16 and hasattr(model, "flownet"):
                # Match Practical-RIFE's official --fp16 inference path. The
                # model and inputs use half precision on Tensor Core GPUs;
                # outputs are converted back to float32 at the API boundary.
                model.flownet.half()
        self._model = model
        self._arbitrary_constructor = arbitrary_constructor
        self._model_source = model_source
        self._supports_batched_timestep = model_source == "train_log.RIFE_HDv3" and hasattr(model, "flownet")
        # Practical-RIFE lite checkpoints use a coarser first pyramid scale
        # ([32, 16, 8, 4, 1]) and the upstream demo pads to 128. Full HDv3
        # checkpoints use [16, 8, 4, 2, 1] and work with 64-pixel padding.
        model_file = self.repository_root / "train_log" / "RIFE_HDv3.py"
        if model_file.is_file() and "[32/scale" in model_file.read_text(encoding="utf-8"):
            self._padding_multiple = 128
        return model

    def _tensor(self, frame: np.ndarray) -> tuple[torch.Tensor, tuple[int, int]]:
        array = np.asarray(frame)
        if array.ndim != 3 or array.shape[2] < 3:
            raise ValueError("Practical-RIFE expects an HWC RGB frame.")
        if array.dtype != np.float32:
            array = array.astype(np.float32)
        if array.max(initial=0.0) > 1.0:
            array /= 255.0
        tensor = torch.from_numpy(array[..., :3].transpose(2, 0, 1)).unsqueeze(0).to(self.device)
        height, width = tensor.shape[-2:]
        # Practical-RIFE 4.x uses five pyramid stages (down to 1/16
        # resolution) and its official inference script pads to 64 pixels.
        # Smaller padding can make intermediate tensors have mismatched
        # spatial sizes even when the input is divisible by 32.
        pad_h = (self._padding_multiple - height % self._padding_multiple) % self._padding_multiple
        pad_w = (self._padding_multiple - width % self._padding_multiple) % self._padding_multiple
        if pad_h or pad_w:
            tensor = F.pad(tensor, (0, pad_w, 0, pad_h), mode="replicate")
        return tensor, (height, width)

    @staticmethod
    def _direct_timestep_parameter(model: object) -> str | None:
        parameters = inspect.signature(model.inference).parameters
        for name in ("timestep", "time", "t"):
            if name in parameters:
                return name
        if any(parameter.kind is inspect.Parameter.VAR_KEYWORD for parameter in parameters.values()):
            return "timestep"
        return None

    def interpolate(self, frame0: np.ndarray, frame1: np.ndarray, target_timestamps: np.ndarray) -> np.ndarray:
        prediction = self.interpolate_torch(
            torch.from_numpy(np.asarray(frame0)),
            torch.from_numpy(np.asarray(frame1)),
            target_timestamps,
        )
        return prediction.detach().float().cpu().numpy()

    def _tensor_torch(self, frame: torch.Tensor) -> tuple[torch.Tensor, tuple[int, int]]:
        if frame.ndim != 3 or frame.shape[-1] < 3:
            raise ValueError("Practical-RIFE expects an HWC RGB tensor.")
        input_dtype = torch.float16 if self.fp16 else torch.float32
        tensor = frame[..., :3].to(device=self.device, dtype=input_dtype).permute(2, 0, 1).unsqueeze(0)
        if float(tensor.detach().amax().item()) > 1.0:
            tensor = tensor / 255.0
        height, width = tensor.shape[-2:]
        pad_h = (self._padding_multiple - height % self._padding_multiple) % self._padding_multiple
        pad_w = (self._padding_multiple - width % self._padding_multiple) % self._padding_multiple
        if pad_h or pad_w:
            tensor = F.pad(tensor, (0, pad_w, 0, pad_h), mode="replicate")
        return tensor, (height, width)

    def interpolate_torch(self, frame0: torch.Tensor, frame1: torch.Tensor, target_timestamps: np.ndarray) -> torch.Tensor:
        targets = validate_targets(target_timestamps)
        if len(targets) == 0:
            return torch.empty((0, *frame0.shape), dtype=torch.float32, device=self.device)
        model = self._load_model()
        parameter = self._direct_timestep_parameter(model)
        if parameter is None:
            raise RuntimeError(
                "This Practical-RIFE checkout has no direct arbitrary-timestep API. "
                "Recursive midpoint interpolation is intentionally disabled; use a compatible "
                "Practical-RIFE revision or choose an explicit backend."
            )
        first, original_shape = self._tensor_torch(frame0)
        second, _ = self._tensor_torch(frame1)
        outputs: list[torch.Tensor] = []
        # The official 4.x network accepts a batch of timestep values. This
        # matters for long fixed-count exports such as 40 source frames to
        # 480 output frames; legacy/custom adapters remain scalar-query based.
        batch_targets = self.inference_batch_size if getattr(self, "_supports_batched_timestep", False) and parameter == "timestep" else 1
        with torch.inference_mode():
            for begin in range(0, len(targets), batch_targets):
                end = min(begin + batch_targets, len(targets))
                count = end - begin
                first_input = first.expand(count, -1, -1, -1) if count > 1 else first
                second_input = second.expand(count, -1, -1, -1) if count > 1 else second
                if count > 1:
                    timestep = torch.as_tensor(targets[begin:end], dtype=first.dtype, device=self.device).view(count, 1, 1, 1)
                else:
                    timestep = float(targets[begin])
                inference_parameters = inspect.signature(model.inference).parameters
                scale_kwargs = {"scale": self.scale} if "scale" in inference_parameters else {}
                prediction = model.inference(first_input, second_input, **scale_kwargs, **{parameter: timestep})
                if isinstance(prediction, (tuple, list)):
                    prediction = prediction[0]
                if prediction.ndim == 3:
                    prediction = prediction.unsqueeze(0)
                prediction = prediction[..., :original_shape[0], :original_shape[1]]
                outputs.append(prediction.detach().float().permute(0, 2, 3, 1))
        return torch.cat(outputs, dim=0).float()

    def metadata(self) -> dict[str, object]:
        return {
            **super().metadata(),
            "repository_root": str(self.repository_root),
            "checkpoint": str(self.checkpoint) if self.checkpoint else str(self.repository_root / "train_log"),
            "scale": self.scale,
            "device": str(self.device),
            "arbitrary_constructor": getattr(self, "_arbitrary_constructor", None),
            "model_source": getattr(self, "_model_source", None),
            "inference_batch_size": self.inference_batch_size,
            "fp16": self.fp16,
            "padding_multiple": self._padding_multiple,
            "supports_batched_timestep": getattr(self, "_supports_batched_timestep", None),
        }
