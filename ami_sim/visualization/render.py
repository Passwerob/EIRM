"""Compact deterministic debug visualizations."""

from __future__ import annotations

from pathlib import Path

import imageio.v3 as iio
import numpy as np

from ami_sim.types import EventStream


def _uint8(frame: np.ndarray) -> np.ndarray:
    value = np.asarray(frame)
    if value.ndim == 2:
        value = np.repeat(value[..., None], 3, axis=-1)
    if value.dtype != np.uint8:
        value = np.clip(value * 255.0 if value.max(initial=0.0) <= 1.0 else value, 0, 255).astype(np.uint8)
    return value[..., :3]


def save_frame_sequence(frames: np.ndarray, directory: str | Path, *, fps: float | None = None, video_path: str | Path | None = None) -> None:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    values = np.asarray(frames)
    for index, frame in enumerate(values):
        iio.imwrite(directory / f"{index:06d}.png", _uint8(frame))
    if video_path is not None and fps is not None:
        iio.imwrite(video_path, np.stack([_uint8(frame) for frame in values]), fps=fps)


def save_event_image(events: EventStream, shape: tuple[int, int], path: str | Path, *, compensated: bool = False) -> None:
    height, width = shape
    image = np.zeros((height, width, 3), dtype=np.uint8)
    if compensated and "x_float" in events.extras:
        x = np.rint(events.extras["x_float"]).astype(np.int64)
        y = np.rint(events.extras["y_float"]).astype(np.int64)
    else:
        x, y = events.x.astype(np.int64), events.y.astype(np.int64)
    valid = (x >= 0) & (y >= 0) & (x < width) & (y < height)
    positive = valid & (events.p > 0)
    negative = valid & ~positive
    image[y[positive], x[positive], 0] = 255
    image[y[negative], x[negative], 2] = 255
    iio.imwrite(path, image)


def save_temporal_difference_video(intensity_frames: np.ndarray, path: str | Path, *, fps: float) -> None:
    """Write global-scale |delta log I| video to expose VFI flicker without per-frame scaling."""
    frames = np.asarray(intensity_frames, dtype=np.float32)
    if len(frames) < 2:
        difference = np.zeros((1, *frames.shape[1:]), dtype=np.float32)
    else:
        difference = np.abs(np.diff(np.log(np.clip(frames, 1e-6, 1.0)), axis=0))
    scale = float(np.percentile(difference, 99.5)) if np.any(difference) else 1.0
    visual = np.clip(difference / max(scale, 1e-8) * 255.0, 0, 255).astype(np.uint8)
    iio.imwrite(path, visual, fps=fps)


def save_panel(rgb: np.ndarray, vanilla: EventStream, raw: EventStream, compensated: EventStream, path: str | Path) -> None:
    rgb_image = _uint8(rgb)
    panel_shape = (rgb_image.shape[0], rgb_image.shape[1])
    panels = [rgb_image]
    for events, is_compensated in ((vanilla, False), (raw, False), (compensated, True)):
        image = np.zeros((*panel_shape, 3), dtype=np.uint8)
        if is_compensated and "x_float" in events.extras:
            x = np.rint(events.extras["x_float"]).astype(np.int64)
            y = np.rint(events.extras["y_float"]).astype(np.int64)
        else:
            x, y = events.x.astype(np.int64), events.y.astype(np.int64)
        valid = (x >= 0) & (y >= 0) & (x < panel_shape[1]) & (y < panel_shape[0])
        image[y[valid & (events.p > 0)], x[valid & (events.p > 0)], 0] = 255
        image[y[valid & (events.p < 0)], x[valid & (events.p < 0)], 2] = 255
        panels.append(image)
    iio.imwrite(path, np.concatenate(panels, axis=1))


def save_event_comparison_panel(
    rgb: np.ndarray,
    original_direct: EventStream,
    vanilla: EventStream,
    raw: EventStream,
    compensated: EventStream,
    path: str | Path,
) -> None:
    """Write RGB plus direct-original/VFI/AMI event stages in one strip."""
    rgb_image = _uint8(rgb)
    panel_shape = (rgb_image.shape[0], rgb_image.shape[1])
    panels = [rgb_image]
    for events, is_compensated in (
        (original_direct, False),
        (vanilla, False),
        (raw, False),
        (compensated, True),
    ):
        image = np.zeros((*panel_shape, 3), dtype=np.uint8)
        if is_compensated and "x_float" in events.extras:
            x = np.rint(events.extras["x_float"]).astype(np.int64)
            y = np.rint(events.extras["y_float"]).astype(np.int64)
        else:
            x, y = events.x.astype(np.int64), events.y.astype(np.int64)
        valid = (x >= 0) & (y >= 0) & (x < panel_shape[1]) & (y < panel_shape[0])
        image[y[valid & (events.p > 0)], x[valid & (events.p > 0)], 0] = 255
        image[y[valid & (events.p < 0)], x[valid & (events.p < 0)], 2] = 255
        panels.append(image)
    iio.imwrite(path, np.concatenate(panels, axis=1))


def save_event_pair_panel(
    rgb: np.ndarray,
    original_direct: EventStream,
    interpolated_direct: EventStream,
    path: str | Path,
) -> None:
    """Write RGB plus sparse-source and interpolated direct-event streams."""
    rgb_image = _uint8(rgb)
    panel_shape = (rgb_image.shape[0], rgb_image.shape[1])
    panels = [rgb_image]
    for events in (original_direct, interpolated_direct):
        image = np.zeros((*panel_shape, 3), dtype=np.uint8)
        x, y = events.x.astype(np.int64), events.y.astype(np.int64)
        valid = (x >= 0) & (y >= 0) & (x < panel_shape[1]) & (y < panel_shape[0])
        image[y[valid & (events.p > 0)], x[valid & (events.p > 0)], 0] = 255
        image[y[valid & (events.p < 0)], x[valid & (events.p < 0)], 2] = 255
        panels.append(image)
    iio.imwrite(path, np.concatenate(panels, axis=1))
