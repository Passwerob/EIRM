"""Causal RGB optical-flow features for natural relative-motion sensing."""

from __future__ import annotations

import numpy as np


FLOW_FEATURE_NAMES = (
    "median_magnitude",
    "p90_magnitude",
    "mean_dx",
    "mean_dy",
    "direction_coherence",
    "magnitude_std",
    "divergence_p90",
    "curl_p90",
)

TILED_FLOW_FEATURE_NAMES = tuple(
    f"tile_{tile}_{name}"
    for tile in ("00", "01", "10", "11")
    for name in FLOW_FEATURE_NAMES
)

MOTION_CLUSTER_FEATURE_NAMES = tuple(
    f"cluster_{index}_{name}"
    for index in range(3)
    for name in ("mean_dx", "mean_dy", "mean_magnitude", "p90_magnitude", "area_fraction", "centroid_x", "centroid_y", "coherence", "magnitude_std")
)

PRIVILEGED_ROI_FEATURE_NAMES = tuple(f"object_{name}" for name in FLOW_FEATURE_NAMES) + tuple(f"background_{name}" for name in FLOW_FEATURE_NAMES)
CAUSAL_BASE_FEATURE_DIM = 20
PRIVILEGED_ROI_FEATURE_DIM = len(PRIVILEGED_ROI_FEATURE_NAMES)
PRIVILEGED_ROI_Q_FEATURE_DIM = CAUSAL_BASE_FEATURE_DIM + PRIVILEGED_ROI_FEATURE_DIM


def _gray_uint8(image: np.ndarray) -> np.ndarray:
    values = np.asarray(image)
    if values.ndim == 3 and values.shape[-1] >= 3:
        values = 0.2126 * values[..., 0] + 0.7152 * values[..., 1] + 0.0722 * values[..., 2]
    if values.ndim != 2:
        raise ValueError("Expected an image shaped [H,W] or RGB [H,W,3].")
    values = values.astype(np.float32, copy=False)
    if float(np.nanmax(values, initial=0.0)) <= 1.0:
        values = values * 255.0
    return np.clip(values, 0.0, 255.0).astype(np.uint8)


def estimate_dense_flow(previous: np.ndarray, current: np.ndarray) -> np.ndarray:
    """Estimate deterministic Farneback flow from a pre-prism RGB pair."""
    try:
        import cv2
    except ImportError as error:  # pragma: no cover - depends on environment
        raise RuntimeError("Optical-flow features require opencv-python-headless.") from error
    first, second = _gray_uint8(previous), _gray_uint8(current)
    if first.shape != second.shape:
        raise ValueError("Optical-flow images must have matching height and width.")
    return cv2.calcOpticalFlowFarneback(
        first,
        second,
        None,
        pyr_scale=0.5,
        levels=3,
        winsize=15,
        iterations=3,
        poly_n=5,
        poly_sigma=1.2,
        flags=0,
    ).astype(np.float32)


def summarize_flow(
    flow: np.ndarray,
    reference_image: np.ndarray | None = None,
    region_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Return eight scale-stable flow statistics for one causal image pair.

    ``region_mask`` is intended for diagnostic region-conditioned summaries.
    When supplied, pixels outside the region are never used, including when
    the textured-pixel fallback is triggered.
    """
    values = np.asarray(flow, dtype=np.float32)
    if values.ndim != 3 or values.shape[-1] != 2 or min(values.shape[:2]) < 2:
        raise ValueError("Expected dense flow shaped [H,W,2].")
    height, width = values.shape[:2]
    diagonal = max(float(np.hypot(height, width)), 1.0)
    finite = np.isfinite(values).all(axis=-1)
    safe = np.where(finite[..., None], values, 0.0)
    if region_mask is None:
        region = np.ones((height, width), dtype=bool)
    else:
        region = np.asarray(region_mask, dtype=bool)
        if region.shape != (height, width):
            raise ValueError("region_mask must match the spatial flow shape.")
    magnitude = np.linalg.norm(safe, axis=-1) / diagonal
    if reference_image is None:
        textured = np.ones_like(magnitude, dtype=bool)
    else:
        gray = _gray_uint8(reference_image).astype(np.float32) / 255.0
        gx = np.gradient(gray, axis=1)
        gy = np.gradient(gray, axis=0)
        gradient = np.hypot(gx, gy)
        threshold = float(np.quantile(gradient, 0.5))
        textured = gradient >= max(threshold, 1e-4)
    valid = finite & textured & region
    if not np.any(valid):
        valid = finite & region
    if not np.any(valid):
        return np.zeros(len(FLOW_FEATURE_NAMES), dtype=np.float32)
    selected = safe[valid] / diagonal
    selected_magnitude = magnitude[valid]
    mean_vector = np.mean(selected, axis=0)
    directions = selected / np.maximum(np.linalg.norm(selected, axis=1, keepdims=True), 1e-6)
    coherence = float(np.linalg.norm(np.mean(directions, axis=0)))
    du_dy, du_dx = np.gradient(safe[..., 0])
    dv_dy, dv_dx = np.gradient(safe[..., 1])
    divergence = (du_dx + dv_dy)[valid] / diagonal
    curl = (dv_dx - du_dy)[valid] / diagonal
    features = np.asarray(
        [
            np.median(selected_magnitude),
            np.quantile(selected_magnitude, 0.9),
            mean_vector[0],
            mean_vector[1],
            coherence,
            np.std(selected_magnitude),
            np.quantile(np.abs(divergence), 0.9),
            np.quantile(np.abs(curl), 0.9),
        ],
        dtype=np.float32,
    )
    return np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)


def summarize_residual_flow(flow: np.ndarray, reference_image: np.ndarray | None = None) -> np.ndarray:
    """Summarize flow after subtracting a robust dominant-motion estimate.

    The per-component median is used as a background-motion proxy. Unlike the
    ROI-conditioned diagnostic, this representation requires no object mask
    and is therefore suitable for a deployable-input ablation.
    """
    values = np.asarray(flow, dtype=np.float32)
    if values.ndim != 3 or values.shape[-1] != 2 or min(values.shape[:2]) < 2:
        raise ValueError("Expected flow shaped [H,W,2].")
    finite = np.isfinite(values).all(axis=-1)
    if not np.any(finite):
        return np.zeros(len(FLOW_FEATURE_NAMES), dtype=np.float32)
    dominant = np.median(values[finite], axis=0)
    residual = values - dominant.reshape(1, 1, 2)
    return summarize_flow(residual, reference_image)


def summarize_tiled_flow(
    flow: np.ndarray,
    reference_image: np.ndarray | None = None,
    *,
    rows: int = 2,
    cols: int = 2,
) -> np.ndarray:
    """Summarize causal flow in fixed spatial tiles without an ROI mask."""
    values = np.asarray(flow, dtype=np.float32)
    if values.ndim != 3 or values.shape[-1] != 2 or rows < 1 or cols < 1:
        raise ValueError("Expected flow [H,W,2] and positive tile dimensions.")
    height, width = values.shape[:2]
    outputs: list[np.ndarray] = []
    for row in range(rows):
        y0, y1 = height * row // rows, height * (row + 1) // rows
        for col in range(cols):
            x0, x1 = width * col // cols, width * (col + 1) // cols
            tile = values[y0:y1, x0:x1]
            image_tile = None if reference_image is None else np.asarray(reference_image)[y0:y1, x0:x1]
            outputs.append(summarize_flow(tile, image_tile))
    return np.concatenate(outputs).astype(np.float32)


def summarize_motion_clusters(
    flow: np.ndarray,
    *,
    clusters: int = 3,
    iterations: int = 12,
) -> np.ndarray:
    """Summarize a dense flow field as deterministic local motion clusters.

    This is a mask-free diagnostic. It clusters finite flow vectors with a
    deterministic magnitude-quantile initialization, then sorts clusters by
    area so feature order is stable. The output describes motion mixtures and
    spatial support; it does not use object masks or future frames.
    """
    values = np.asarray(flow, dtype=np.float32)
    if values.ndim != 3 or values.shape[-1] != 2 or clusters < 1 or iterations < 1:
        raise ValueError("Expected flow [H,W,2] and positive clusters/iterations.")
    height, width = values.shape[:2]
    finite = np.isfinite(values).all(axis=-1)
    if not np.any(finite):
        return np.zeros(clusters * 9, dtype=np.float32)
    diagonal = max(float(np.hypot(height, width)), 1.0)
    vectors = values[finite].astype(np.float64) / diagonal
    magnitudes = np.linalg.norm(vectors, axis=1)
    order = np.argsort(magnitudes, kind="mergesort")
    positions = np.argwhere(finite).astype(np.float64)
    positions[:, 0] /= max(height - 1, 1)
    positions[:, 1] /= max(width - 1, 1)
    quantiles = np.linspace(0.0, 1.0, clusters)
    centers = vectors[order[np.asarray(np.round(quantiles * max(len(order) - 1, 0)), dtype=np.int64)]].copy()
    labels = np.zeros(len(vectors), dtype=np.int64)
    for _ in range(iterations):
        distances = np.sum((vectors[:, None, :] - centers[None, :, :]) ** 2, axis=-1)
        labels = np.argmin(distances, axis=1)
        updated = centers.copy()
        for index in range(clusters):
            selected = vectors[labels == index]
            if len(selected):
                updated[index] = selected.mean(axis=0)
        if np.allclose(updated, centers, atol=1e-8, rtol=0.0):
            centers = updated
            break
        centers = updated
    areas = np.asarray([(labels == index).sum() for index in range(clusters)], dtype=np.int64)
    cluster_order = np.argsort(-areas, kind="mergesort")
    outputs: list[float] = []
    for index in cluster_order:
        selected = labels == index
        cluster_vectors = vectors[selected]
        if not len(cluster_vectors):
            outputs.extend((0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.5, 0.0, 0.0))
            continue
        cluster_magnitudes = np.linalg.norm(cluster_vectors, axis=1)
        cluster_positions = positions[selected]
        directions = cluster_vectors / np.maximum(cluster_magnitudes[:, None], 1e-8)
        outputs.extend(
            (
                float(cluster_vectors[:, 0].mean()),
                float(cluster_vectors[:, 1].mean()),
                float(cluster_magnitudes.mean()),
                float(np.quantile(cluster_magnitudes, 0.9)),
                float(np.mean(selected)),
                float(cluster_positions[:, 1].mean()),
                float(cluster_positions[:, 0].mean()),
                float(np.linalg.norm(directions.mean(axis=0))),
                float(cluster_magnitudes.std()),
            )
        )
    return np.nan_to_num(np.asarray(outputs, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)


def summarize_privileged_roi_flow(
    flow: np.ndarray,
    object_mask: np.ndarray,
    reference_image: np.ndarray | None = None,
) -> np.ndarray:
    """Concatenate object and background flow summaries using a GT mask.

    This is an observability upper bound. The mask is privileged training or
    diagnostic supervision and must not be treated as a deployable input.
    Output layout is ``PRIVILEGED_ROI_FEATURE_NAMES`` with shape ``[16]``.
    """
    region = np.asarray(object_mask, dtype=bool)
    if region.ndim != 2:
        raise ValueError("object_mask must be a 2-D boolean array matching the flow map.")
    object_features = summarize_flow(flow, reference_image, region)
    background_features = summarize_flow(flow, reference_image, ~region)
    return np.concatenate((object_features, background_features)).astype(np.float32)


def privileged_roi_sequence(frames: np.ndarray, masks: np.ndarray) -> np.ndarray:
    """Causal privileged ROI-flow features for each control window.

    ``frames`` is ``[T,H,W]`` luminance or ``[T,H,W,3]`` RGB. ``masks`` is
    ``[T,H,W]``. Row 0 is zero because window 0 has no completed previous
    pair; row ``i`` uses frames ``[i-1, i]`` and mask ``i-1``.
    """
    images = np.asarray(frames)
    regions = np.asarray(masks, dtype=bool)
    if images.ndim not in (3, 4) or regions.ndim != 3:
        raise ValueError("Expected frames [T,H,W] or [T,H,W,3] and masks [T,H,W].")
    if len(images) != len(regions) or len(images) < 2:
        raise ValueError("frames and masks must contain at least two aligned samples.")
    if images.shape[:3] != regions.shape:
        raise ValueError("Mask spatial shape must match the frame height and width.")
    windows = len(images) - 1
    values = np.zeros((windows, PRIVILEGED_ROI_FEATURE_DIM), dtype=np.float32)
    for index in range(1, windows):
        flow = estimate_dense_flow(images[index - 1], images[index])
        values[index] = summarize_privileged_roi_flow(flow, regions[index - 1], images[index - 1])
    return values


def estimate_flow_features(frames: np.ndarray, timestamps: np.ndarray | None = None) -> np.ndarray:
    """Estimate one flow-summary vector per adjacent pre-prism RGB pair.

    When timestamps are supplied, displacement-like components are converted
    to normalized image velocity per second. Direction coherence and magnitude
    dispersion remain dimensionless.
    """
    values = np.asarray(frames)
    if values.ndim != 4 or values.shape[-1] < 3 or len(values) < 2:
        raise ValueError("Expected at least two RGB frames shaped [T,H,W,3].")
    features = np.stack(
        [summarize_flow(estimate_dense_flow(values[index], values[index + 1]), values[index]) for index in range(len(values) - 1)]
    )
    if timestamps is not None:
        times = np.asarray(timestamps, dtype=np.float64).reshape(-1)
        if len(times) != len(values) or not np.all(np.isfinite(times)) or not np.all(np.diff(times) > 0.0):
            raise ValueError("timestamps must match frames and be finite and strictly increasing.")
        features[:, [0, 1, 2, 3, 6, 7]] /= np.diff(times)[:, None]
    return features.astype(np.float32)
