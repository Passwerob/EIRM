"""Reproducible diagnostics for the EIRM AMI simulator validation document.

The functions in this module are deliberately downstream of the simulator.  They
load canonical event streams, RGB/intensity frames, timestamps, and prism state
from a run directory and never alter event formation or the forward model.
"""

from __future__ import annotations

import csv
import json
import math
import shutil
from pathlib import Path
from typing import Any, Iterable

import cv2
import imageio.v3 as iio
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

from ami_sim.events import read_events_h5
from ami_sim.events import DVSVoltmeterBackend
from ami_sim.metrics import compare_event_streams
from ami_sim.pipeline import rgb_to_luminance
from ami_sim.prism import ConstantRotationPolicy, PaperSimplifiedPrismModel, PrismWarper, compensate_events
from ami_sim.types import CameraIntrinsics, EventStream


ANGLE_LABELS = ("0-15", "15-30", "30-45", "45-60", "60-75", "75-90")
ANGLE_EDGES = np.deg2rad(np.arange(0.0, 91.0, 15.0))


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(value), indent=2, sort_keys=True, allow_nan=False), encoding="utf-8")


def _read_csv_values(path: Path, key: str) -> np.ndarray:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        return np.empty(0, dtype=np.float64)
    return np.asarray([float(row[key]) for row in rows], dtype=np.float64)


def _read_frame_paths(directory: Path) -> list[Path]:
    return sorted(directory.glob("*.png"))


def _load_rgb(directory: Path, *, limit: int | None = None) -> np.ndarray:
    paths = _read_frame_paths(directory)
    if limit is not None:
        paths = paths[:limit]
    if not paths:
        raise FileNotFoundError(f"No PNG frames found in {directory}.")
    return np.stack([np.asarray(iio.imread(path))[..., :3] for path in paths], axis=0)


def _load_intensity(directory: Path) -> np.ndarray:
    paths = sorted(directory.glob("*.npy"))
    if not paths:
        raise FileNotFoundError(f"No intensity frames found in {directory}.")
    return np.stack([np.asarray(np.load(path), dtype=np.float32) for path in paths], axis=0)


def _event_file(root: Path, *names: str) -> Path:
    for name in names:
        path = root / name / "events.h5"
        if path.is_file():
            return path
    raise FileNotFoundError(f"No event stream found under {root}: {names}")


def _load_events(root: Path, *names: str) -> EventStream:
    return read_events_h5(_event_file(root, *names), time_unit="s")


def _window(events: EventStream, start: float, duration: float) -> EventStream:
    return events.masked((events.t >= start) & (events.t < start + duration))


def _coords(events: EventStream, compensated: bool = False) -> tuple[np.ndarray, np.ndarray]:
    if compensated and "x_float" in events.extras and "y_float" in events.extras:
        return events.extras["x_float"].astype(np.float64), events.extras["y_float"].astype(np.float64)
    return events.x.astype(np.float64), events.y.astype(np.float64)


def _safe_points(events: EventStream, shape: tuple[int, int], compensated: bool = False) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    height, width = shape
    x, y = _coords(events, compensated)
    valid = np.isfinite(x) & np.isfinite(y) & (x >= 0) & (y >= 0) & (x < width) & (y < height)
    return x, y, valid


def event_density_in_mask(events: EventStream, mask: np.ndarray, shape: tuple[int, int], *, compensated: bool = False) -> float:
    """Return events per pixel after assigning events to a spatial mask."""
    region = np.asarray(mask, dtype=bool)
    if region.shape != shape:
        raise ValueError("Event-density mask shape does not match the event image shape.")
    denominator = int(np.count_nonzero(region))
    if denominator == 0:
        return 0.0
    x, y, valid = _safe_points(events, shape, compensated)
    if not np.any(valid):
        return 0.0
    xi = np.rint(x[valid]).astype(np.int32)
    yi = np.rint(y[valid]).astype(np.int32)
    inside = (xi >= 0) & (yi >= 0) & (xi < shape[1]) & (yi < shape[0])
    return float(np.count_nonzero(region[yi[inside], xi[inside]]) / denominator)


def filter_events_to_mask(events: EventStream, mask: np.ndarray, shape: tuple[int, int], *, compensated: bool = False) -> EventStream:
    """Drop events outside a metric's valid image region without rounding early."""
    region = np.asarray(mask, dtype=bool)
    if region.shape != shape:
        raise ValueError("Valid-event mask shape does not match the event image shape.")
    x, y, coordinate_valid = _safe_points(events, shape, compensated)
    keep = np.zeros(len(events), dtype=bool)
    if not np.any(coordinate_valid):
        return events.masked(keep)
    indices = np.flatnonzero(coordinate_valid)
    xi = np.rint(x[coordinate_valid]).astype(np.int32)
    yi = np.rint(y[coordinate_valid]).astype(np.int32)
    inside = (xi >= 0) & (yi >= 0) & (xi < shape[1]) & (yi < shape[0])
    keep[indices[inside]] = region[yi[inside], xi[inside]]
    return events.masked(keep)


def support_map(events: EventStream, shape: tuple[int, int], *, radius: int = 1, compensated: bool = False) -> np.ndarray:
    """Rasterize an event stream with a small configurable matching radius."""
    height, width = shape
    x, y, valid = _safe_points(events, shape, compensated)
    result = np.zeros(shape, dtype=bool)
    if not np.any(valid):
        return result
    xi = np.rint(x[valid]).astype(np.int32)
    yi = np.rint(y[valid]).astype(np.int32)
    inside = (xi >= 0) & (yi >= 0) & (xi < width) & (yi < height)
    xi = xi[inside]
    yi = yi[inside]
    if not len(xi):
        return result
    result[yi, xi] = True
    if radius > 0:
        kernel = np.ones((2 * radius + 1, 2 * radius + 1), dtype=np.uint8)
        result = cv2.dilate(result.astype(np.uint8), kernel, iterations=1).astype(bool)
    return result


def sobel_edge_map(intensity: np.ndarray, valid_mask: np.ndarray, quantile: float = 0.80) -> tuple[np.ndarray, np.ndarray, float]:
    image = np.asarray(intensity, dtype=np.float32)
    gx = cv2.Sobel(image, cv2.CV_32F, 1, 0, ksize=3) / 8.0
    gy = cv2.Sobel(image, cv2.CV_32F, 0, 1, ksize=3) / 8.0
    magnitude = np.hypot(gx, gy)
    finite = magnitude[np.asarray(valid_mask, dtype=bool)]
    # A sparse synthetic line image contains mostly zero gradients.  Quantiling
    # all pixels would therefore return zero and classify the whole image as an
    # edge.  Quantile the positive gradient population while retaining the same
    # configurable rule for natural scenes.
    positive = finite[finite > 1e-8]
    threshold = float(np.quantile(positive, quantile)) if len(positive) else 0.0
    edge = (magnitude >= threshold) & np.asarray(valid_mask, dtype=bool)
    return edge, magnitude, threshold


def edge_distances(edge: np.ndarray, events: EventStream, *, compensated: bool = False) -> np.ndarray:
    distance = cv2.distanceTransform((~np.asarray(edge, dtype=bool)).astype(np.uint8), cv2.DIST_L2, 3)
    x, y, valid = _safe_points(events, edge.shape, compensated)
    if not np.any(valid):
        return np.empty(0, dtype=np.float32)
    xi = np.rint(x[valid]).astype(np.int32)
    yi = np.rint(y[valid]).astype(np.int32)
    inside = (xi >= 0) & (yi >= 0) & (xi < edge.shape[1]) & (yi < edge.shape[0])
    xi = xi[inside]
    yi = yi[inside]
    if not len(xi):
        return np.empty(0, dtype=np.float32)
    return distance[yi, xi].astype(np.float32, copy=False)


def distance_stats(edge: np.ndarray, events: EventStream, *, compensated: bool = False) -> dict[str, float]:
    values = edge_distances(edge, events, compensated=compensated)
    if len(values) == 0:
        return {"event_count": 0.0, "mean": float("nan"), "median": float("nan"), "p75": float("nan"), "p90": float("nan"), "band_width_p90": float("nan")}
    return {
        "event_count": float(len(values)),
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "p75": float(np.percentile(values, 75)),
        "p90": float(np.percentile(values, 90)),
        "band_width_p90": float(np.percentile(values, 90)),
    }


def _event_counts(events: EventStream) -> dict[str, int]:
    return {
        "events": int(len(events)),
        "positive": int(np.count_nonzero(events.p > 0)),
        "negative": int(np.count_nonzero(events.p < 0)),
    }


def _render_events(events: EventStream, shape: tuple[int, int], *, compensated: bool = False, polarity: str = "combined") -> np.ndarray:
    image = np.zeros((*shape, 3), dtype=np.uint8)
    x, y, valid = _safe_points(events, shape, compensated)
    if not np.any(valid):
        return image
    xi = np.rint(x[valid]).astype(np.int32)
    yi = np.rint(y[valid]).astype(np.int32)
    inside = (xi >= 0) & (yi >= 0) & (xi < shape[1]) & (yi < shape[0])
    xi = xi[inside]
    yi = yi[inside]
    p = events.p[valid][inside]
    if not len(xi):
        return image
    if polarity in {"on", "combined"}:
        on = p > 0
        image[yi[on], xi[on], 0] = 255
    if polarity in {"off", "combined"}:
        off = p < 0
        image[yi[off], xi[off], 2] = 255
    return image


def _save_rgb(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    value = np.asarray(image)
    if value.dtype != np.uint8:
        value = np.clip(value * 255.0 if value.max(initial=0.0) <= 1.0 else value, 0, 255).astype(np.uint8)
    iio.imwrite(path, value[..., :3])


def _save_figure(path: Path, panels: Iterable[np.ndarray], titles: Iterable[str], *, figsize: tuple[float, float], cmap: str | None = None) -> None:
    panels = list(panels)
    titles = list(titles)
    fig, axes = plt.subplots(1, len(panels), figsize=figsize, squeeze=False)
    for axis, image, title in zip(axes[0], panels, titles):
        axis.imshow(image, cmap=cmap)
        axis.set_title(title, fontsize=10)
        axis.axis("off")
    fig.tight_layout(pad=0.4)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160, bbox_inches="tight", facecolor="black")
    plt.close(fig)


def _intrinsics_from_metadata(metadata: dict[str, Any], shape: tuple[int, int]) -> CameraIntrinsics:
    prism = metadata.get("prism", {})
    node = prism.get("intrinsics", {}) if isinstance(prism, dict) else {}
    height, width = shape
    if node:
        return CameraIntrinsics(width, height, float(node["fx"]), float(node["fy"]), float(node["cx"]), float(node["cy"]))
    return CameraIntrinsics(width, height, float(width), float(height), (width - 1) / 2.0, (height - 1) / 2.0)


def _model_policy(metadata: dict[str, Any], shape: tuple[int, int]) -> tuple[PaperSimplifiedPrismModel, ConstantRotationPolicy, float]:
    intrinsics = _intrinsics_from_metadata(metadata, shape)
    prism_angle = float(metadata.get("prism_angle", metadata.get("options", {}).get("prism_angle_deg", 0.5)))
    rpm = float(metadata.get("rpm", metadata.get("options", {}).get("rotation_rpm", 720.0)))
    theta0 = float(metadata.get("reference_theta", metadata.get("options", {}).get("theta0", 0.0)))
    return PaperSimplifiedPrismModel(intrinsics, prism_angle), ConstantRotationPolicy(rpm), theta0


def _flow_and_motion(root: Path, intensity0: np.ndarray, valid_mask: np.ndarray, metadata: dict[str, Any], source_rgb: np.ndarray) -> dict[str, Any]:
    """Estimate natural flow on the source interval and a matching prism displacement."""
    shape = intensity0.shape
    if len(source_rgb) < 2:
        flow = np.zeros((*shape, 2), dtype=np.float32)
    else:
        a = cv2.cvtColor(np.asarray(source_rgb[0])[..., :3], cv2.COLOR_RGB2GRAY)
        b = cv2.cvtColor(np.asarray(source_rgb[1])[..., :3], cv2.COLOR_RGB2GRAY)
        flow = cv2.calcOpticalFlowFarneback(a, b, None, 0.5, 3, 21, 3, 5, 1.2, 0).astype(np.float32)
    source_ts_path = root / "source_timestamps.csv"
    source_ts = _read_csv_values(source_ts_path, "timestamp_seconds") if source_ts_path.exists() else np.asarray([0.0, 1.0 / 30.0])
    dt = float(source_ts[1] - source_ts[0]) if len(source_ts) > 1 else 1.0 / 30.0
    model, policy, theta0 = _model_policy(metadata, shape)
    y, x = torch.meshgrid(torch.arange(shape[0], dtype=torch.float32), torch.arange(shape[1], dtype=torch.float32), indexing="ij")
    points = torch.stack((x, y), dim=-1)
    theta = policy.states_at(np.asarray([float(source_ts[0]), float(source_ts[0] + dt)]), theta0=theta0, reference_time=float(source_ts[0]))
    forward0 = model.forward_points(points, float(theta[0].theta)).detach().cpu().numpy()
    forward1 = model.forward_points(points, float(theta[1].theta)).detach().cpu().numpy()
    prism = (forward1 - forward0).astype(np.float32)
    gx = cv2.Sobel(np.asarray(intensity0, dtype=np.float32), cv2.CV_32F, 1, 0, ksize=3) / 8.0
    gy = cv2.Sobel(np.asarray(intensity0, dtype=np.float32), cv2.CV_32F, 0, 1, ksize=3) / 8.0
    gradient = np.stack((gx, gy), axis=-1)
    edge, magnitude, _ = sobel_edge_map(intensity0, valid_mask, 0.80)
    natural_mag = np.linalg.norm(flow, axis=-1)
    prism_mag = np.linalg.norm(prism, axis=-1)
    effective = flow + prism
    effective_mag = np.linalg.norm(effective, axis=-1)
    denominator = np.maximum(natural_mag + prism_mag, 1e-8)
    cancellation = effective_mag / denominator
    gmag = np.linalg.norm(gradient, axis=-1)
    nmag = np.maximum(natural_mag, 1e-8)
    angle = np.arccos(np.clip(np.abs(np.sum(gradient * flow, axis=-1)) / np.maximum(gmag * nmag, 1e-8), 0.0, 1.0))
    edge_valid = edge & np.isfinite(angle)
    thresholds = (0.1, 0.25, 0.5)
    return {
        "natural_flow": flow,
        "prism_flow": prism,
        "effective_flow": effective,
        "gradient": gradient,
        "gradient_magnitude": magnitude,
        "edge": edge,
        "angle": angle,
        "cancellation": cancellation,
        "edge_valid": edge_valid,
        "source_interval_seconds": dt,
        "natural_motion_mean_px": float(np.mean(natural_mag[valid_mask])),
        "natural_motion_median_px": float(np.median(natural_mag[valid_mask])),
        "prism_motion_mean_px": float(np.mean(prism_mag[valid_mask])),
        "effective_motion_mean_px": float(np.mean(effective_mag[valid_mask])),
        "mean_gradient_motion_angle_deg": float(np.degrees(np.mean(angle[edge_valid]))) if np.any(edge_valid) else float("nan"),
        "near_cancellation_fraction": {str(t): float(np.mean(cancellation[edge_valid] < t)) if np.any(edge_valid) else 0.0 for t in thresholds},
        "valid_edge_pixels": int(edge_valid.sum()),
    }


def _motion_quantile_masks(flow: np.ndarray, valid_mask: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    """Create comparable low/high natural-motion masks from the RGB flow field."""
    magnitude = np.linalg.norm(np.asarray(flow, dtype=np.float32), axis=-1)
    valid_values = magnitude[np.asarray(valid_mask, dtype=bool)]
    if not len(valid_values):
        empty = np.zeros_like(valid_mask, dtype=bool)
        return empty, empty, {"low_motion_threshold_px": 0.0, "high_motion_threshold_px": 0.0}
    low_threshold, high_threshold = np.quantile(valid_values, (0.40, 0.60))
    valid = np.asarray(valid_mask, dtype=bool)
    return (
        valid & (magnitude <= low_threshold),
        valid & (magnitude >= high_threshold),
        {"low_motion_threshold_px": float(low_threshold), "high_motion_threshold_px": float(high_threshold)},
    )


def _angle_metrics(
    edge: np.ndarray,
    angle: np.ndarray,
    vanilla: EventStream,
    compensated: EventStream,
    valid_mask: np.ndarray,
    *,
    radius: int = 1,
) -> list[dict[str, Any]]:
    shape = edge.shape
    v_support = support_map(vanilla, shape, radius=radius)
    c_support = support_map(compensated, shape, radius=radius, compensated=True)
    new_support = c_support & ~v_support
    rows: list[dict[str, Any]] = []
    for index, label in enumerate(ANGLE_LABELS):
        selected = edge & valid_mask & (angle >= ANGLE_EDGES[index]) & (angle < ANGLE_EDGES[index + 1])
        denom = int(selected.sum())
        v_cov = float((selected & v_support).sum() / max(denom, 1))
        c_cov = float((selected & c_support).sum() / max(denom, 1))
        new_cov = float((selected & new_support).sum() / max(denom, 1))
        # Assign event density to the local angle bin at the rounded event pixel.
        def density(events: EventStream, compensated_flag: bool) -> float:
            x, y, valid = _safe_points(events, shape, compensated_flag)
            if not np.any(valid):
                return 0.0
            xi = np.rint(x[valid]).astype(np.int32)
            yi = np.rint(y[valid]).astype(np.int32)
            inside = (xi >= 0) & (yi >= 0) & (xi < shape[1]) & (yi < shape[0])
            local = selected[yi[inside], xi[inside]]
            return float(np.count_nonzero(local) / max(denom, 1))
        rows.append({
            "angle_bin": label,
            "angle_low_deg": float(index * 15),
            "angle_high_deg": float((index + 1) * 15),
            "edge_pixels": denom,
            "vanilla_coverage": v_cov,
            "compensated_coverage": c_cov,
            "ami_gain": c_cov - v_cov,
            "vanilla_event_density": density(vanilla, False),
            "compensated_event_density": density(compensated, True),
            "new_edge_coverage": new_cov,
            "mean_natural_angle_deg": float(np.degrees(np.mean(angle[selected]))) if denom else float("nan"),
        })
    return rows


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fields = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: _jsonable(v) for k, v in row.items()})


def _flicker_stats(intensity: np.ndarray, valid_mask: np.ndarray, *, contrast_threshold: float = 0.2) -> dict[str, Any]:
    log_frames = np.log(np.clip(np.asarray(intensity, dtype=np.float64), 1e-6, 1.0))
    differences = np.diff(log_frames, axis=0)
    values = differences[:, valid_mask].reshape(-1)
    abs_values = np.abs(values)
    pixel_median = np.median(np.abs(differences), axis=0)
    low_motion = valid_mask & (pixel_median <= np.percentile(pixel_median[valid_mask], 40))
    low_values = np.abs(differences[:, low_motion]).reshape(-1) if np.any(low_motion) else abs_values
    temporal_variance = np.var(log_frames[:, low_motion], axis=0) if np.any(low_motion) else np.asarray([0.0])
    return {
        "frame_count": int(len(intensity)),
        "mean_abs_log_difference": float(np.mean(abs_values)) if len(abs_values) else 0.0,
        "p95_abs_log_difference": float(np.percentile(abs_values, 95)) if len(abs_values) else 0.0,
        "temporal_variance_mean_low_motion": float(np.mean(temporal_variance)),
        "temporal_variance_p95_low_motion": float(np.percentile(temporal_variance, 95)),
        "fraction_above_dvs_threshold": float(np.mean(abs_values >= contrast_threshold)) if len(abs_values) else 0.0,
        "low_motion_p95_abs_log_difference": float(np.percentile(low_values, 95)) if len(low_values) else 0.0,
        "low_motion_pixel_fraction": float(np.mean(low_motion[valid_mask])) if np.any(valid_mask) else 0.0,
        "dvs_like_threshold": float(contrast_threshold),
        "criterion": "pixels below the 40th percentile of temporal median |delta log intensity|",
    }


def _synthetic_image(shape: tuple[int, int]) -> np.ndarray:
    height, width = shape
    image = np.full(shape, 0.05, dtype=np.float32)
    image[max(1, height // 3 - 2):height // 3 + 3, width // 10:width * 9 // 10] = 0.95
    image[height // 10:height * 9 // 10, max(1, width // 3 - 2):width // 3 + 3] = 0.95
    for y in range(height // 5, height * 4 // 5):
        x = width // 10 + int((y - height // 5) * (width * 0.8) / (height * 0.6))
        if 2 <= x < width - 2:
            image[y - 2:y + 3, x - 2:x + 3] = 0.95
    # A compact checkerboard/corner makes the inverse test cover junctions too.
    size = max(8, min(height, width) // 24)
    y0, x0 = height * 2 // 3, width * 2 // 3
    for row in range(5):
        for col in range(5):
            if (row + col) % 2 == 0:
                image[y0 + row * size:y0 + (row + 1) * size, x0 + col * size:x0 + (col + 1) * size] = 0.95
    return image


def _synthetic_metrics(edge: np.ndarray, events: EventStream, valid_mask: np.ndarray, *, compensated: bool) -> dict[str, Any]:
    events = filter_events_to_mask(events, valid_mask, edge.shape, compensated=compensated)
    values = edge_distances(edge, events, compensated=compensated)
    counts = _event_counts(events)
    if len(values) == 0:
        return {**counts, "mean_edge_distance": float("nan"), "median_edge_distance": float("nan"), "p90_edge_distance": float("nan"), "band_width_p90_px": float("nan"), "fwhm_like_width_px": float("nan")}
    hist, bins = np.histogram(values, bins=np.arange(0.0, 20.5, 0.5))
    half = hist.max() / 2.0 if len(hist) else 0.0
    nonzero = np.flatnonzero(hist >= half) if half else np.empty(0, dtype=np.int64)
    fwhm = float((bins[nonzero[-1] + 1] - bins[nonzero[0]]) if len(nonzero) else np.nan)
    return {
        **counts,
        "mean_edge_distance": float(np.mean(values)),
        "median_edge_distance": float(np.median(values)),
        "p90_edge_distance": float(np.percentile(values, 90)),
        "band_width_p90_px": float(np.percentile(values, 90)),
        "fwhm_like_width_px": fwhm,
    }


def run_synthetic_test(root: Path, validation: Path, metadata: dict[str, Any], shape: tuple[int, int], valid_mask: np.ndarray, *, use_official_dvs: bool = True) -> dict[str, Any]:
    synthetic_dir = validation / "synthetic_compensation"
    synthetic_dir.mkdir(parents=True, exist_ok=True)
    image = _synthetic_image(shape)
    model, policy, theta0 = _model_policy(metadata, shape)
    frame_count = 480
    duration = 1.3
    timestamps = np.linspace(0.0, duration, frame_count, dtype=np.float64)
    tensor = torch.from_numpy(np.repeat(image[None, None, ...], frame_count, axis=0))
    warper = PrismWarper(model, policy, shape[0], shape[1], theta0=theta0, reference_time=0.0, padding_mode="border", batch_size=8)
    warped, _ = warper.warp(tensor, timestamps)
    warped_np = warped[:, 0].detach().cpu().numpy().astype(np.float32)
    # Support both the EIRM layout and historical runs stored
    # directly under the workspace-level outputs directory.
    dvs_candidates = [
        root.parent.parent / "DVS-Voltmeter",
        Path(__file__).resolve().parents[2] / "DVS-Voltmeter",
    ]
    dvs_root = next((candidate for candidate in dvs_candidates if candidate.is_dir()), None)
    if use_official_dvs and dvs_root is not None:
        command = "python scripts/dvs_voltmeter_runner.py --frames {frames_dir} --timestamps {timestamps_path} --output {output_path} --dvs-root DVS-Voltmeter --camera-type DVS346 --seed 0"
        backend = DVSVoltmeterBackend(command, output_time_unit="s")
        raw = backend.simulate(warped_np, timestamps, {})
        backend_name = "official_dvs_voltmeter"
    else:
        from ami_sim.events import DebugThresholdEventSimulator
        raw = DebugThresholdEventSimulator(0.2).simulate(warped_np, timestamps, {})
        backend_name = "debug_threshold"
    compensated = compensate_events(raw, model, policy, theta0=theta0, reference_time=0.0)
    edge, _, edge_threshold = sobel_edge_map(image, valid_mask, 0.80)
    raw_valid = filter_events_to_mask(raw, valid_mask, shape)
    compensated_valid = filter_events_to_mask(compensated, valid_mask, shape, compensated=True)
    metrics = {
        "event_backend": backend_name,
        "frame_count": frame_count,
        "duration_seconds": duration,
        "edge_threshold_sobel_quantile_80": edge_threshold,
        "raw": _synthetic_metrics(edge, raw_valid, valid_mask, compensated=False),
        "compensated": _synthetic_metrics(edge, compensated_valid, valid_mask, compensated=True),
    }
    raw_image = _render_events(raw_valid, shape)
    comp_image = _render_events(compensated_valid, shape, compensated=True)
    rgb = np.repeat(image[..., None], 3, axis=-1)
    _save_rgb(synthetic_dir / "rgb.png", rgb)
    _save_rgb(synthetic_dir / "raw_ami.png", raw_image)
    _save_rgb(synthetic_dir / "compensated_ami.png", comp_image)
    _save_figure(synthetic_dir / "comparison.png", [rgb, raw_image, comp_image], ["Synthetic RGB", "Raw AMI", "Compensated AMI"], figsize=(13.5, 4.0))
    shutil.copy2(synthetic_dir / "comparison.png", validation / "figures" / "fig_04_compensation_synthetic.png")
    _write_json(synthetic_dir / "metrics.json", metrics)
    return metrics


def _analyze_fps_directory(
    path: Path,
    edge: np.ndarray,
    valid_mask: np.ndarray,
    *,
    window_ms: float,
    shape: tuple[int, int],
    low_motion_mask: np.ndarray | None = None,
    high_motion_mask: np.ndarray | None = None,
) -> dict[str, Any]:
    metadata_path = path / "metadata.json"
    metadata = json.loads(metadata_path.read_text()) if metadata_path.exists() else {}
    intensity = np.load(path / "interpolated_intensity" / "000000.npy")
    source = _load_events(path, "events_original_direct", "events_dvs_voltmeter_original_direct")
    vanilla = _load_events(path, "events_vanilla", "events_dvs_voltmeter_480_direct", "events_dvs_voltmeter_interpolated_480_direct")
    raw = _load_events(path, "events_ami_raw", "events_dvs_voltmeter_ami_raw")
    comp = _load_events(path, "events_ami_compensated", "events_dvs_voltmeter_ami_compensated")
    timestamps_path = path / "timestamps.csv"
    timestamps = _read_csv_values(timestamps_path, "timestamp_seconds") if timestamps_path.exists() else np.empty(0, dtype=np.float64)
    start = float(timestamps[0]) if len(timestamps) else 0.0
    duration = window_ms / 1000.0
    streams = {
        "source_direct": (source, False),
        "vanilla": (vanilla, False),
        "raw_ami": (raw, False),
        "compensated_ami": (comp, True),
    }
    result: dict[str, Any] = {"fps": metadata.get("options", {}).get("interp_fps", metadata.get("interp_fps_target", "unknown")), "window_ms": window_ms, "source_frames": int(metadata.get("source_timestamp_count", 0)), "frames": int(metadata.get("timestamp_count", 0))}
    windows: dict[str, tuple[EventStream, bool]] = {}
    for name, (events, comp_flag) in streams.items():
        window = filter_events_to_mask(_window(events, start, duration), valid_mask, shape, compensated=comp_flag)
        windows[name] = (window, comp_flag)
        support = support_map(window, shape, radius=1, compensated=comp_flag)
        edge_cov = float((edge & valid_mask & support).sum() / max(int((edge & valid_mask).sum()), 1))
        d = distance_stats(edge & valid_mask, window, compensated=comp_flag)
        result[f"{name}_events"] = int(len(window))
        result[f"{name}_positive"] = int(np.count_nonzero(window.p > 0))
        result[f"{name}_negative"] = int(np.count_nonzero(window.p < 0))
        result[f"{name}_polarity_ratio"] = float(np.count_nonzero(window.p > 0) / max(len(window), 1))
        result[f"{name}_edge_coverage"] = edge_cov
        result[f"{name}_mean_edge_distance"] = d["mean"]
        result[f"{name}_median_edge_distance"] = d["median"]
        if low_motion_mask is not None:
            result[f"{name}_low_motion_event_density"] = event_density_in_mask(window, low_motion_mask, shape, compensated=comp_flag)
        if high_motion_mask is not None:
            result[f"{name}_high_motion_event_density"] = event_density_in_mask(window, high_motion_mask, shape, compensated=comp_flag)
    v_support = support_map(windows["vanilla"][0], shape, radius=1)
    c_support = support_map(windows["compensated_ami"][0], shape, radius=1, compensated=True)
    edge_valid = edge & valid_mask
    new = c_support & ~v_support & edge_valid
    additional = max(0, len(windows["compensated_ami"][0]) - len(windows["vanilla"][0]))
    result["new_edge_coverage"] = float(new.sum() / max(int(edge_valid.sum()), 1))
    result["additional_events"] = int(additional)
    result["event_efficiency"] = float(new.sum() / max(additional, 1))
    return result


def _copy_temporal_video(root: Path, validation: Path) -> None:
    candidates = [root / "diagnostics" / "temporal_difference_video.mp4", root / "diagnostics" / "temporal_diff_video.mp4"]
    for source in candidates:
        if source.exists():
            shutil.copy2(source, validation / "temporal_diff_video.mp4")
            return


def run_validation(
    root: str | Path,
    *,
    output: str | Path | None = None,
    event_window_ms: float = 20.0,
    edge_quantile: float = 0.80,
    support_radius: int = 1,
    fps_root: str | Path | None = None,
    run_synthetic: bool = True,
    official_synthetic_dvs: bool = True,
    reference_root: str | Path | None = None,
) -> dict[str, Any]:
    """Run all downstream diagnostics for one completed AMI simulation."""
    root = Path(root)
    validation = Path(output) if output is not None else root / "ami_validation"
    figures = validation / "figures"
    polarity_dir = validation / "polarity"
    validation.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)
    polarity_dir.mkdir(parents=True, exist_ok=True)
    metadata = json.loads((root / "metadata.json").read_text(encoding="utf-8"))
    rgb_original = _load_rgb(root / "rgb_original")
    rgb_interpolated = _load_rgb(root / "rgb_interpolated", limit=1)
    intensity = _load_intensity(root / "interpolated_intensity")
    shape = intensity.shape[1:]
    valid_path = root / "valid_mask" / "valid_mask.png"
    valid_mask = np.asarray(iio.imread(valid_path)) > 0 if valid_path.exists() else np.ones(shape, dtype=bool)
    if valid_mask.shape != shape:
        valid_mask = cv2.resize(valid_mask.astype(np.uint8), (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST).astype(bool)
    edge, gradient_magnitude, edge_threshold = sobel_edge_map(intensity[0], valid_mask, edge_quantile)
    edge_valid = edge & valid_mask
    source = _load_events(root, "events_original_direct", "events_dvs_voltmeter_original_direct")
    vanilla = _load_events(root, "events_vanilla", "events_dvs_voltmeter_480_direct", "events_dvs_voltmeter_interpolated_480_direct")
    raw = _load_events(root, "events_ami_raw", "events_dvs_voltmeter_ami_raw")
    compensated = _load_events(root, "events_ami_compensated", "events_dvs_voltmeter_ami_compensated")
    timestamps_path = root / "timestamps.csv"
    timeline = _read_csv_values(timestamps_path, "timestamp_seconds") if timestamps_path.exists() else np.empty(0, dtype=np.float64)
    start = float(timeline[0]) if len(timeline) else 0.0
    event_quality_vs_reference = None
    if reference_root is not None:
        reference_root = Path(reference_root)
        reference_raw = _load_events(reference_root, "events_ami_raw", "events_dvs_voltmeter_ami_raw")
        event_quality_vs_reference = compare_event_streams(
            reference_raw,
            raw,
            shape,
            frame_timestamps=timeline if len(timeline) > 1 else None,
            candidate_requires_integer_microseconds="cuda_fast" in str(metadata.get("event_backend_name", "")),
        )
    streams = {
        "vanilla_low_fps": (source, False),
        "vanilla_480": (vanilla, False),
        "raw_ami_480": (raw, False),
        "compensated_ami_480": (compensated, True),
    }
    windows: dict[str, tuple[EventStream, bool]] = {}
    for name, (events, comp_flag) in streams.items():
        windows[name] = (
            filter_events_to_mask(
                _window(events, start, event_window_ms / 1000.0),
                valid_mask,
                shape,
                compensated=comp_flag,
            ),
            comp_flag,
        )
    distances: dict[str, dict[str, float]] = {}
    supports: dict[str, np.ndarray] = {}
    for name, (events, comp_flag) in windows.items():
        distances[name] = distance_stats(edge_valid, events, compensated=comp_flag)
        supports[name] = support_map(events, shape, radius=support_radius, compensated=comp_flag)
    summary_rows: list[dict[str, Any]] = []
    vanilla_support = supports["vanilla_480"]
    comp_support = supports["compensated_ami_480"]
    new_support = comp_support & ~vanilla_support & edge_valid
    additional_events = max(0, len(windows["compensated_ami_480"][0]) - len(windows["vanilla_480"][0]))
    for name, (events, comp_flag) in windows.items():
        coverage = float((edge_valid & supports[name]).sum() / max(int(edge_valid.sum()), 1))
        d = distances[name]
        summary_rows.append({
            "stream": name,
            "events": len(events),
            "positive": int(np.count_nonzero(events.p > 0)),
            "negative": int(np.count_nonzero(events.p < 0)),
            "edge_coverage": coverage,
            "mean_edge_dist": d["mean"],
            "median_edge_dist": d["median"],
            "p75_edge_dist": d["p75"],
            "p90_edge_dist": d["p90"],
            "new_coverage": float(new_support.sum() / max(int(edge_valid.sum()), 1)) if name == "compensated_ami_480" else "N/A",
            "additional_events": additional_events if name == "compensated_ami_480" else "N/A",
            "event_efficiency": float(new_support.sum() / max(additional_events, 1)) if name == "compensated_ami_480" else "N/A",
        })
    _write_csv(validation / "summary_metrics.csv", summary_rows)
    compensation_metrics = {
        "edge_method": "Sobel gradient magnitude",
        "edge_threshold_quantile": edge_quantile,
        "edge_threshold": edge_threshold,
        "edge_pixels": int(edge_valid.sum()),
        "valid_pixel_fraction": float(np.mean(valid_mask)),
        "event_window_ms": event_window_ms,
        "streams": {name: {**_event_counts(events), **distances[name]} for name, (events, _) in windows.items()},
    }
    _write_json(validation / "compensation_metrics.json", compensation_metrics)
    if event_quality_vs_reference is not None:
        _write_json(validation / "event_quality_vs_reference.json", event_quality_vs_reference)

    # Expose sensitivity to the accumulation window without changing the
    # primary comparison window used by the figures.
    window_rows: list[dict[str, Any]] = []
    for window_ms in (10.0, 20.0, 33.0, 50.0):
        duration = window_ms / 1000.0
        slices: dict[str, tuple[EventStream, bool]] = {
            name: (
                filter_events_to_mask(
                    _window(events, start, duration),
                    valid_mask,
                    shape,
                    compensated=comp_flag,
                ),
                comp_flag,
            )
            for name, (events, comp_flag) in streams.items()
        }
        row: dict[str, Any] = {"window_ms": window_ms}
        for name, (events, comp_flag) in slices.items():
            support = support_map(events, shape, radius=support_radius, compensated=comp_flag)
            d = distance_stats(edge_valid, events, compensated=comp_flag)
            row[f"{name}_events"] = len(events)
            row[f"{name}_edge_coverage"] = float((edge_valid & support).sum() / max(int(edge_valid.sum()), 1))
            row[f"{name}_mean_edge_distance"] = d["mean"]
            row[f"{name}_median_edge_distance"] = d["median"]
        v = support_map(slices["vanilla_480"][0], shape, radius=support_radius)
        c = support_map(slices["compensated_ami_480"][0], shape, radius=support_radius, compensated=True)
        new = c & ~v & edge_valid
        extra = max(0, len(slices["compensated_ami_480"][0]) - len(slices["vanilla_480"][0]))
        row["new_edge_coverage"] = float(new.sum() / max(int(edge_valid.sum()), 1))
        row["additional_events"] = extra
        row["event_efficiency"] = float(new.sum() / max(extra, 1))
        window_rows.append(row)
    _write_csv(validation / "window_metrics.csv", window_rows)

    # Required polarity assets and a labeled polarity figure.
    polarity_panels: list[np.ndarray] = []
    polarity_titles: list[str] = []
    for stream_name, (events, comp_flag) in (("vanilla", windows["vanilla_480"]), ("ami_raw", windows["raw_ami_480"]), ("ami_comp", windows["compensated_ami_480"])):
        for polarity in ("on", "off", "combined"):
            image = _render_events(events, shape, compensated=comp_flag, polarity=polarity)
            _save_rgb(polarity_dir / f"{stream_name}_{polarity}.png", image)
            polarity_panels.append(image)
            polarity_titles.append(f"{stream_name} {polarity.upper()}")
    fig, axes = plt.subplots(3, 3, figsize=(13.5, 10.0), squeeze=False)
    for axis, image, title in zip(axes.flat, polarity_panels, polarity_titles):
        axis.imshow(image)
        axis.set_title(title, fontsize=9)
        axis.axis("off")
    fig.tight_layout(pad=0.5)
    fig.savefig(figures / "fig_02_polarity.png", dpi=160, facecolor="black")
    plt.close(fig)
    polarity_rows: list[dict[str, Any]] = []
    for stream_name, (events, comp_flag) in (("vanilla_480", windows["vanilla_480"]), ("raw_ami_480", windows["raw_ami_480"]), ("compensated_ami_480", windows["compensated_ami_480"])):
        for p_value, label in ((1, "on"), (-1, "off")):
            subset = events.masked(events.p == p_value)
            d = distance_stats(edge_valid, subset, compensated=comp_flag)
            polarity_rows.append({"stream": stream_name, "polarity": label, **d})
    _write_csv(validation / "polarity_metrics.csv", polarity_rows)

    # Main and new-structure panels use exactly the same accumulation window.
    main_panels = [rgb_original[0], _render_events(windows["vanilla_low_fps"][0], shape), _render_events(windows["vanilla_480"][0], shape), _render_events(windows["raw_ami_480"][0], shape), _render_events(windows["compensated_ami_480"][0], shape, compensated=True)]
    _save_figure(figures / "fig_01_main_comparison.png", main_panels, ["RGB reference", "40-frame direct DVS", "480-frame no-prism DVS", "Raw AMI 480", "Compensated AMI 480"], figsize=(18.0, 4.1))
    rgb_aligned = np.asarray(rgb_original[0]).copy()
    rgb_aligned[~new_support] = (rgb_aligned[~new_support] * 0.15).astype(np.uint8)
    rgb_aligned[new_support] = np.asarray([0, 255, 0], dtype=np.uint8)
    new_binary = np.zeros((*shape, 3), dtype=np.uint8)
    new_binary[new_support] = np.asarray([0, 255, 0], dtype=np.uint8)
    _save_rgb(validation / "newly_recovered_rgb_aligned.png", rgb_aligned)
    _save_rgb(validation / "newly_recovered_support.png", new_binary)
    _save_figure(figures / "fig_03_new_structure.png", [rgb_original[0], _render_events(windows["vanilla_480"][0], shape), _render_events(windows["compensated_ami_480"][0], shape, compensated=True), new_binary], ["RGB", "480-frame no-prism DVS", "AMI compensated", "AMI-only RGB-aligned structure"], figsize=(15.0, 4.1))

    source_motion = _flow_and_motion(root, intensity[0], valid_mask, metadata, rgb_original)
    low_motion_mask, high_motion_mask, motion_density_thresholds = _motion_quantile_masks(source_motion["natural_flow"], valid_mask)
    angle_rows = _angle_metrics(edge, source_motion["angle"], windows["vanilla_480"][0], windows["compensated_ami_480"][0], valid_mask, radius=support_radius)
    _write_csv(validation / "angle_conditioned_metrics.csv", angle_rows)
    _write_json(
        validation / "relative_motion_metrics.json",
        {
            **{key: value for key, value in source_motion.items() if not isinstance(value, np.ndarray)},
            "fps_density_masks": {
                **motion_density_thresholds,
                "low_motion_pixel_fraction": float(np.mean(low_motion_mask[valid_mask])),
                "high_motion_pixel_fraction": float(np.mean(high_motion_mask[valid_mask])),
            },
        },
    )
    fig, ax = plt.subplots(figsize=(7.0, 4.5))
    x = np.arange(len(angle_rows))
    ax.plot(x, [row["vanilla_coverage"] for row in angle_rows], "o-", label="480-frame no-prism DVS")
    ax.plot(x, [row["compensated_coverage"] for row in angle_rows], "o-", label="Compensated AMI 480")
    ax.set_xticks(x, [row["angle_bin"] for row in angle_rows])
    ax.set_xlabel("Absolute gradient-motion angle (deg)")
    ax.set_ylabel("Edge coverage")
    ax.set_ylim(bottom=0)
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(figures / "fig_05_coverage_vs_angle.png", dpi=180)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(7.0, 4.5))
    ax.bar(x, [row["ami_gain"] for row in angle_rows], color="#20a060")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x, [row["angle_bin"] for row in angle_rows])
    ax.set_xlabel("Absolute gradient-motion angle (deg)")
    ax.set_ylabel("AMI coverage gain")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(figures / "fig_06_ami_gain_vs_angle.png", dpi=180)
    plt.close(fig)

    flicker = _flicker_stats(intensity, valid_mask, contrast_threshold=float(metadata.get("options", {}).get("event_window_ms", 20.0)) / 100.0)
    _write_json(validation / "temporal_flicker_stats.json", flicker)
    _copy_temporal_video(root, validation)
    # Keep an explicit static diagnostic copy if the source run used the older name.
    if not (validation / "temporal_diff_video.mp4").exists():
        iio.imwrite(validation / "temporal_diff_video.mp4", np.clip(np.abs(np.diff(np.log(np.clip(intensity, 1e-6, 1.0)), axis=0)) * 255.0, 0, 255).astype(np.uint8), fps=float(metadata.get("options", {}).get("interp_fps", 480.0)))

    synthetic = run_synthetic_test(root, validation, metadata, shape, valid_mask, use_official_dvs=official_synthetic_dvs) if run_synthetic else None

    fps_rows: list[dict[str, Any]] = []
    if fps_root is not None:
        fps_root = Path(fps_root)
        for fps in (240, 480, 960):
            path = fps_root / str(fps)
            if path.is_dir() and (path / "metadata.json").exists():
                fps_rows.append(
                    _analyze_fps_directory(
                        path,
                        edge,
                        valid_mask,
                        window_ms=event_window_ms,
                        shape=shape,
                        low_motion_mask=low_motion_mask,
                        high_motion_mask=high_motion_mask,
                    )
                )
    if not fps_rows:
        # The representative fixed-count run is still a valid 480-FPS row; mark
        # the other rates as pending rather than fabricating convergence data.
        fps_rows.append(
            _analyze_fps_directory(
                root,
                edge,
                valid_mask,
                window_ms=event_window_ms,
                shape=shape,
                low_motion_mask=low_motion_mask,
                high_motion_mask=high_motion_mask,
            )
        )
    _write_csv(validation / "fps_convergence.csv", fps_rows)
    available_fps = [row for row in fps_rows if str(row.get("fps")) in {"240", "480", "960", "240.0", "480.0", "960.0"}]
    if available_fps:
        labels = [str(row["fps"]).replace(".0", "") for row in available_fps]
        fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.0))
        x = np.arange(len(available_fps))
        axes[0].plot(x, [row["vanilla_events"] for row in available_fps], "o-", label="480-frame no-prism DVS")
        axes[0].plot(x, [row["compensated_ami_events"] for row in available_fps], "o-", label="Compensated AMI")
        axes[0].set_ylabel("Events in window")
        axes[1].plot(x, [row["vanilla_edge_coverage"] for row in available_fps], "o-", label="480-frame no-prism DVS")
        axes[1].plot(x, [row["compensated_ami_edge_coverage"] for row in available_fps], "o-", label="Compensated AMI")
        axes[1].set_ylabel("Edge coverage")
        for axis in axes:
            axis.set_xticks(x, labels)
            axis.set_xlabel("Interpolation FPS")
            axis.grid(alpha=0.25)
            axis.legend()
        fig.tight_layout()
        fig.savefig(figures / "fig_07_fps_convergence.png", dpi=180)
        plt.close(fig)

    config = {
        "source_run": str(root),
        "event_window_ms": event_window_ms,
        "edge_method": "Sobel magnitude quantile",
        "edge_quantile": edge_quantile,
        "support_radius_px": support_radius,
        "valid_mask": str(root / "valid_mask" / "valid_mask.png"),
        "event_files": {name: str(_event_file(root, *names)) for name, names in {
            "source_direct": ("events_original_direct", "events_dvs_voltmeter_original_direct"),
            "vanilla_480": ("events_vanilla", "events_dvs_voltmeter_480_direct", "events_dvs_voltmeter_interpolated_480_direct"),
            "raw_ami_480": ("events_ami_raw", "events_dvs_voltmeter_ami_raw"),
            "compensated_ami_480": ("events_ami_compensated", "events_dvs_voltmeter_ami_compensated"),
        }.items()},
        "fps_root": str(fps_root) if fps_root is not None else None,
    }
    with (validation / "config.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(_jsonable(config), handle, sort_keys=True)
    result = {
        "validation_dir": str(validation),
        "summary_metrics": summary_rows,
        "compensation": compensation_metrics,
        "relative_motion": {key: value for key, value in source_motion.items() if not isinstance(value, np.ndarray)},
        "angle_metrics": angle_rows,
        "flicker": flicker,
        "synthetic": synthetic,
        "fps_rows": fps_rows,
        "event_quality_vs_reference": event_quality_vs_reference,
    }
    _write_json(validation / "validation_summary.json", result)
    return result
