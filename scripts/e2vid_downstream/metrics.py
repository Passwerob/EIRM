"""Reconstruction metrics. Photometry is not calibrated, so PSNR/SSIM/LPIPS
use a per-image affine (scale+shift) fit of the prediction to the GT.
Unaligned PSNR is stored as a diagnostic only.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn.functional as F


def _gaussian_window(channel: int, window_size: int = 11, sigma: float = 1.5, device=None, dtype=None) -> torch.Tensor:
    coords = torch.arange(window_size, device=device, dtype=dtype) - window_size // 2
    gauss = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    gauss = gauss / gauss.sum()
    window = gauss[:, None] * gauss[None, :]
    return window.expand(channel, 1, window_size, window_size).contiguous()


def ssim_torch(pred: torch.Tensor, target: torch.Tensor, data_range: float = 1.0) -> torch.Tensor:
    """pred/target: [B, 1, H, W] or [B, C, H, W] in [0, data_range]. Mean SSIM."""
    channel = pred.shape[1]
    window = _gaussian_window(channel, device=pred.device, dtype=pred.dtype)
    pad = window.shape[-1] // 2
    mu1 = F.conv2d(pred, window, padding=pad, groups=channel)
    mu2 = F.conv2d(target, window, padding=pad, groups=channel)
    mu1_sq, mu2_sq, mu12 = mu1 ** 2, mu2 ** 2, mu1 * mu2
    sigma1_sq = F.conv2d(pred * pred, window, padding=pad, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(target * target, window, padding=pad, groups=channel) - mu2_sq
    sigma12 = F.conv2d(pred * target, window, padding=pad, groups=channel) - mu12
    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2
    ssim_map = ((2 * mu12 + c1) * (2 * sigma12 + c2)) / ((mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2))
    return ssim_map.mean()


def affine_align(pred: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, float, float]:
    p = np.asarray(pred, dtype=np.float64).reshape(-1)
    g = np.asarray(target, dtype=np.float64).reshape(-1)
    if not np.isfinite(p).all() or p.std() < 1e-8:
        b = float(g.mean() - p.mean())
        aligned = np.asarray(pred, dtype=np.float64) + b
        return np.clip(aligned, 0.0, 1.0), 1.0, b
    a_mat = np.stack([p, np.ones_like(p)], axis=1)
    coef, *_ = np.linalg.lstsq(a_mat, g, rcond=None)
    a, b = float(coef[0]), float(coef[1])
    aligned = a * np.asarray(pred, dtype=np.float64) + b
    return np.clip(aligned, 0.0, 1.0), a, b


def psnr_np(pred: np.ndarray, target: np.ndarray) -> float:
    mse = float(np.mean((pred - target) ** 2))
    if mse <= 1e-12:
        return 99.0
    return float(10.0 * np.log10(1.0 / mse))


class MetricBundle:
    def __init__(self, device: torch.device):
        self.device = device
        self.lpips = None
        try:
            import lpips

            self.lpips = lpips.LPIPS(net="alex", verbose=False).to(device).eval()
            for param in self.lpips.parameters():
                param.requires_grad = False
        except Exception as exc:  # noqa: BLE001
            self.lpips_error = str(exc)
        else:
            self.lpips_error = None

    @torch.no_grad()
    def lpips_value(self, pred: np.ndarray, target: np.ndarray) -> float:
        if self.lpips is None:
            return float("nan")
        def to_n1(img: np.ndarray) -> torch.Tensor:
            t = torch.from_numpy(np.asarray(img, dtype=np.float32))
            if t.ndim == 2:
                t = t.unsqueeze(0)
            t = t.unsqueeze(0).to(self.device)
            t = t.repeat(1, 3, 1, 1) if t.shape[1] == 1 else t
            return t * 2.0 - 1.0

        return float(self.lpips(to_n1(pred), to_n1(target)).item())

    def frame_metrics(self, pred: np.ndarray, target: np.ndarray) -> dict[str, float]:
        pred = np.clip(np.asarray(pred, dtype=np.float64), 0.0, 1.0)
        target = np.clip(np.asarray(target, dtype=np.float64), 0.0, 1.0)
        aligned, scale, shift = affine_align(pred, target)
        pred_t = torch.from_numpy(aligned[None, None].astype(np.float32))
        gt_t = torch.from_numpy(target[None, None].astype(np.float32))
        return {
            "psnr": psnr_np(aligned, target),
            "psnr_raw": psnr_np(pred, target),
            "ssim": float(ssim_torch(pred_t, gt_t).item()),
            "lpips": self.lpips_value(aligned.astype(np.float32), target.astype(np.float32)),
            "affine_scale": scale,
            "affine_shift": shift,
        }


def temporal_pair_metrics(pred_a: np.ndarray, pred_b: np.ndarray, gt_a: np.ndarray, gt_b: np.ndarray) -> dict[str, float]:
    d_pred = pred_b - pred_a
    d_gt = gt_b - gt_a
    return {
        "temporal_rmse": float(np.sqrt(np.mean(d_pred ** 2))),
        "flicker_rmse": float(np.sqrt(np.mean((d_pred - d_gt) ** 2))),
        "gt_temporal_rmse": float(np.sqrt(np.mean(d_gt ** 2))),
    }


def summarize(rows: list[dict[str, Any]], keys: tuple[str, ...] = ("psnr", "ssim", "lpips")) -> dict[str, float]:
    out: dict[str, float] = {"n": float(len(rows))}
    for key in keys:
        values = np.asarray([float(row[key]) for row in rows if np.isfinite(float(row[key]))], dtype=np.float64)
        if values.size == 0:
            out[f"{key}_mean"] = float("nan")
            out[f"{key}_std"] = float("nan")
        else:
            out[f"{key}_mean"] = float(values.mean())
            out[f"{key}_std"] = float(values.std(ddof=1) if values.size > 1 else 0.0)
    return out
