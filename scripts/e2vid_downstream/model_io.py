"""Load the original E2VID recurrent UNet and run padded forward passes."""

from __future__ import annotations

import math
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from protocol import E2VID_ROOT, E2VID_WEIGHTS, NUM_ENCODERS


def _ensure_e2vid_path() -> None:
    root = str(E2VID_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)


def load_e2vid(weights: Path | None = None, device: torch.device | str = "cpu") -> nn.Module:
    _ensure_e2vid_path()
    from utils.loading_utils import load_model

    model = load_model(str(weights or E2VID_WEIGHTS))
    model.to(device)
    return model


def freeze_batchnorm(model: nn.Module) -> None:
    """Keep pretrained BN stats; batch-1 fine-tuning otherwise corrupts them."""
    for module in model.modules():
        if isinstance(module, nn.BatchNorm2d):
            module.eval()
            for param in module.parameters():
                param.requires_grad = False


def pad_hw(height: int, width: int, num_encoders: int = NUM_ENCODERS) -> tuple[int, int]:
    factor = 2 ** num_encoders
    return int(math.ceil(height / factor) * factor), int(math.ceil(width / factor) * factor)


def pad_tensor(x: torch.Tensor, num_encoders: int = NUM_ENCODERS) -> tuple[torch.Tensor, tuple[int, int]]:
    """Reflection-pad NCHW tensor to a multiple of 2^encoders. Returns padded, (H, W)."""
    _, _, height, width = x.shape
    pad_h, pad_w = pad_hw(height, width, num_encoders)
    top = (pad_h - height) // 2
    left = (pad_w - width) // 2
    bottom = pad_h - height - top
    right = pad_w - width - left
    if top == 0 and bottom == 0 and left == 0 and right == 0:
        return x, (height, width)
    return F.pad(x, (left, right, top, bottom), mode="reflect"), (height, width)


def crop_tensor(x: torch.Tensor, height: int, width: int) -> torch.Tensor:
    _, _, ph, pw = x.shape
    top = (ph - height) // 2
    left = (pw - width) // 2
    return x[:, :, top : top + height, left : left + width]


def forward_frame(
    model: nn.Module,
    voxel: torch.Tensor,
    states,
    *,
    num_encoders: int = NUM_ENCODERS,
):
    """voxel: [B, C, H, W] → recon [B, 1, H, W] in [0, 1], plus recurrent states."""
    padded, (height, width) = pad_tensor(voxel, num_encoders)
    recon, states = model(padded, states)
    return crop_tensor(recon, height, width), states
