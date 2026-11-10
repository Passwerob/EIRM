"""Metrics for separating useful events from redundant event bursts."""

from __future__ import annotations

import numpy as np

from .types import EventStream


QUALITY_PROTOCOL_VERSION = "ami_event_quality_v2"
TARGET_OBSERVABILITY_VERSION = "ami_teo_v1"


def compute_reference_edge_mask(reference_intensity: np.ndarray, *, edge_quantile: float = 0.75) -> np.ndarray:
    """Compute a frozen edge mask from clean, unmodulated reference intensity."""
    if not 0.0 < edge_quantile < 1.0:
        raise ValueError("edge_quantile must be strictly between zero and one.")
    image = np.asarray(reference_intensity, dtype=np.float64)
    if image.ndim != 2:
        raise ValueError("reference_intensity must have shape [H, W].")
    log_image = np.log(np.clip(image, 1e-6, None))
    gy, gx = np.gradient(log_image)
    magnitude = np.hypot(gx, gy)
    positive_gradient = magnitude[magnitude > 1e-12]
    if not len(positive_gradient):
        return np.zeros_like(magnitude, dtype=bool)
    threshold = float(np.quantile(positive_gradient, edge_quantile))
    return (magnitude >= threshold) & (magnitude > 1e-12)


def event_quality_metrics(
    events: EventStream,
    *,
    reference_intensity: np.ndarray | None = None,
    reference_edge_mask: np.ndarray | None = None,
    previous_active_pixels: np.ndarray | None = None,
    edge_quantile: float = 0.75,
    image_shape: tuple[int, int] | None = None,
) -> dict[str, float | int]:
    """Return interpretable event quality statistics for one control window.

    Event count is reported, but never used as a proxy for information.  When
    an intensity image is supplied, ``edge_coverage`` measures the fraction of
    reference edges touched by at least one event and ``structural_ratio`` is
    the fraction of event pixels lying on those edges.  ``temporal_overlap``
    compares this window's active pixels with the preceding window.
    """
    if not 0.0 < edge_quantile < 1.0:
        raise ValueError("edge_quantile must be strictly between zero and one.")
    if image_shape is not None:
        height, width = image_shape
        if height <= 0 or width <= 0:
            raise ValueError("image_shape dimensions must be positive.")
    elif reference_intensity is not None:
        image = np.asarray(reference_intensity)
        if image.ndim != 2:
            raise ValueError("reference_intensity must have shape [H, W].")
        height, width = image.shape
    elif reference_edge_mask is not None:
        mask = np.asarray(reference_edge_mask)
        if mask.ndim != 2:
            raise ValueError("reference_edge_mask must have shape [H, W].")
        height, width = mask.shape
    elif len(events):
        height = int(events.y.max()) + 1
        width = int(events.x.max()) + 1
    else:
        height = width = 1

    valid = (events.x >= 0) & (events.x < width) & (events.y >= 0) & (events.y < height)
    pixels = events.y[valid].astype(np.int64) * width + events.x[valid].astype(np.int64)
    active = np.unique(pixels)
    active_count = int(len(active))
    event_count = int(np.count_nonzero(valid))
    image_area = float(height * width)

    overlap = 0.0
    if previous_active_pixels is not None:
        previous = np.unique(np.asarray(previous_active_pixels, dtype=np.int64))
        if len(previous) or active_count:
            overlap = float(len(np.intersect1d(active, previous)) / max(len(np.union1d(active, previous)), 1))

    edge_coverage = 0.0
    edge_precision = 0.0
    edge_f1 = 0.0
    structural_ratio = 0.0
    edge_pixels = None
    if reference_edge_mask is not None:
        edge_pixels = np.asarray(reference_edge_mask, dtype=bool)
        if edge_pixels.shape != (height, width):
            raise ValueError("reference_edge_mask shape must match image_shape.")
    elif reference_intensity is not None:
        image = np.asarray(reference_intensity, dtype=np.float64)
        if image.shape != (height, width):
            raise ValueError("reference_intensity shape must match image_shape.")
        edge_pixels = compute_reference_edge_mask(image, edge_quantile=edge_quantile)
    if edge_pixels is not None:
        edge_count = int(np.count_nonzero(edge_pixels))
        if edge_count:
            event_edge_pixels = np.unique(
                events.y[valid][edge_pixels[events.y[valid], events.x[valid]]].astype(np.int64) * width
                + events.x[valid][edge_pixels[events.y[valid], events.x[valid]]].astype(np.int64)
            )
            edge_coverage = float(len(event_edge_pixels) / edge_count)
            edge_precision = float(len(event_edge_pixels) / active_count) if active_count else 0.0
            if edge_coverage + edge_precision > 0.0:
                edge_f1 = float(2.0 * edge_coverage * edge_precision / (edge_coverage + edge_precision))
        if event_count:
            structural_ratio = float(np.count_nonzero(edge_pixels[events.y[valid], events.x[valid]]) / event_count)

    polarity_balance = 0.0
    if event_count:
        polarity_balance = float(abs(np.mean(events.p[valid] > 0) - 0.5) * 2.0)
    event_efficiency = float(active_count / event_count) if event_count else 0.0
    # Global spatial novelty is not target information: in co-motion, sparse
    # background events can yield a high value while the target has no events.
    # Report an edge-conditioned version separately for target-support audits.
    edge_event_efficiency = 0.0
    edge_event_count = 0
    if edge_pixels is not None and event_count:
        on_edge = edge_pixels[events.y[valid], events.x[valid]]
        edge_event_count = int(np.count_nonzero(on_edge))
        if edge_event_count:
            edge_pixels_active = np.unique(
                events.y[valid][on_edge].astype(np.int64) * width
                + events.x[valid][on_edge].astype(np.int64)
            )
            edge_event_efficiency = float(len(edge_pixels_active) / edge_event_count)
    # Target-support coverage is the scientific quantity of interest in the
    # co-motion case; multiplying by unique-event efficiency prevents repeated
    # firing from looking like additional useful support.
    coverage_efficiency = float(edge_coverage * event_efficiency)
    return {
        "event_count": event_count,
        "event_density": float(event_count / image_area),
        "active_pixel_count": active_count,
        "unique_pixel_ratio": float(active_count / image_area),
        "event_efficiency": event_efficiency,
        "repeat_event_fraction": float(1.0 - event_efficiency) if event_count else 0.0,
        "edge_event_count": edge_event_count,
        "edge_event_efficiency": edge_event_efficiency,
        "edge_repeat_event_fraction": float(1.0 - edge_event_efficiency) if edge_event_count else 0.0,
        "edge_coverage": edge_coverage,
        "edge_precision": edge_precision,
        "edge_f1": edge_f1,
        "coverage_efficiency": coverage_efficiency,
        "structural_ratio": structural_ratio,
        "temporal_overlap": overlap,
        "polarity_imbalance": polarity_balance,
    }


def target_observability_metrics(
    events: EventStream,
    *,
    target_mask: np.ndarray,
    target_edge_mask: np.ndarray,
    image_shape: tuple[int, int] | None = None,
    time_range: tuple[float, float] | None = None,
    temporal_bins: int = 4,
) -> dict[str, float | int | str]:
    """Score target coverage and on-target event usefulness.

    Image-wide edge F1 treats every background event as a false positive. Active
    optical excitation modulates the whole field of view, so those background
    events are a side effect, not evidence that the target was unobserved.

    This metric evaluates three quantities on the declared target support
    ``Ω`` and its edge set ``E ⊆ Ω``:

    * coverage ``C = |A ∩ E| / |E|``: fraction of target structure observed;
    * purity ``Π = |A ∩ E| / |A ∩ Ω|``: of the pixels that fired on the
      target, how many sit on structure rather than interior fill;
    * temporal support ``τ``: fraction of the control window that contains at
      least one target-edge event.

    The scalar **TEO** (Target Event Observability) is ``F1(C, Π) · τ``.
    Background events never enter ``A ∩ Ω``, so they cannot lower TEO. A
    method that never hits the target (passive co-motion) still scores 0.
    """
    if temporal_bins < 1:
        raise ValueError("temporal_bins must be positive.")
    target = np.asarray(target_mask, dtype=bool)
    edges = np.asarray(target_edge_mask, dtype=bool) & target
    if target.ndim != 2 or edges.shape != target.shape:
        raise ValueError("target_mask and target_edge_mask must be 2-D and aligned.")
    if image_shape is None:
        height, width = target.shape
    else:
        height, width = image_shape
        if (height, width) != target.shape:
            raise ValueError("image_shape must match target_mask.")

    x = np.asarray(events.x)
    y = np.asarray(events.y)
    t = np.asarray(events.t, dtype=np.float64)
    valid = (x >= 0) & (x < width) & (y >= 0) & (y < height)
    if time_range is not None:
        start, end = (float(time_range[0]), float(time_range[1]))
        if not np.isfinite([start, end]).all() or end <= start:
            raise ValueError("time_range must be a finite increasing pair.")
        valid = valid & (t > start) & (t <= end)
    else:
        start = float(t[valid].min()) if np.any(valid) else 0.0
        end = float(t[valid].max()) if np.any(valid) else 1.0
        if end <= start:
            end = start + 1e-12

    yi = y[valid].astype(np.int64)
    xi = x[valid].astype(np.int64)
    ti = t[valid]
    on_target = target[yi, xi]
    on_edge = edges[yi, xi]
    active_target = np.unique(yi[on_target] * width + xi[on_target])
    active_edge = np.unique(yi[on_edge] * width + xi[on_edge])
    edge_count = int(np.count_nonzero(edges))
    coverage = float(len(active_edge) / edge_count) if edge_count else 0.0
    purity = float(len(active_edge) / len(active_target)) if len(active_target) else 0.0
    target_f1 = float(2.0 * coverage * purity / (coverage + purity)) if coverage + purity else 0.0

    temporal_support = 0.0
    if np.any(on_edge):
        span = end - start
        bins = np.minimum(((ti[on_edge] - start) / span * temporal_bins).astype(np.int64), temporal_bins - 1)
        temporal_support = float(len(np.unique(bins)) / temporal_bins)

    teo = float(target_f1 * temporal_support)
    return {
        "protocol": TARGET_OBSERVABILITY_VERSION,
        "target_coverage": coverage,
        "target_purity": purity,
        "target_f1": target_f1,
        "target_temporal_support": temporal_support,
        "teo": teo,
        "target_active_pixels": int(len(active_target)),
        "target_edge_active_pixels": int(len(active_edge)),
        "target_edge_pixels": edge_count,
        "target_event_count": int(np.count_nonzero(on_target)),
        "target_edge_event_count": int(np.count_nonzero(on_edge)),
        "background_event_count": int(np.count_nonzero(~on_target)),
    }


def event_quality_score(
    metrics: dict[str, float | int],
    *,
    target_density: float | None = None,
    overlap_weight: float = 0.5,
    density_weight: float = 0.1,
    efficiency_weight: float = 0.5,
    coverage_efficiency_weight: float = 0.0,
) -> float:
    """Score useful excitation while discounting repeated events.

    ``event_efficiency`` is the number of unique active pixels divided by the
    number of events. It distinguishes new spatial support from repeatedly
    firing the same pixels, which image-area coverage alone cannot do.
    """
    if min(overlap_weight, density_weight, efficiency_weight, coverage_efficiency_weight) < 0.0:
        raise ValueError("Event-quality weights must be non-negative.")
    # F1 is the primary observability term: coverage alone can be increased by
    # flooding the sensor with off-structure events. A smaller recall term
    # still rewards revealing more of the target when precision is comparable.
    score = float(metrics["edge_f1"]) + 0.25 * float(metrics["edge_coverage"])
    score += 0.15 * float(metrics["unique_pixel_ratio"])
    score += efficiency_weight * float(metrics.get("event_efficiency", 0.0))
    score += coverage_efficiency_weight * float(metrics.get("coverage_efficiency", 0.0))
    score -= overlap_weight * float(metrics["temporal_overlap"])
    score -= 0.05 * float(metrics["polarity_imbalance"])
    if target_density is not None:
        if target_density <= 0.0:
            raise ValueError("target_density must be positive.")
        density = float(metrics["event_density"])
        score -= density_weight * float((np.log1p(density) - np.log1p(target_density)) ** 2)
    return float(score)


def event_signal_fidelity_metrics(
    observed: EventStream,
    reference: EventStream,
    *,
    image_shape: tuple[int, int],
    time_range: tuple[float, float],
    temporal_bins: int = 5,
    spatial_tolerance: int = 1,
) -> dict[str, float | int]:
    """Compare observed and clean-reference event occupancy.

    Events are converted into binary ``[time, polarity, y, x]`` occupancy so
    exact duplicates cannot inflate agreement. Spatial tolerance handles
    small localization errors while time-bin and polarity agreement remain
    explicit. This metric requires a clean synthetic or calibrated reference;
    it is not available for unconstrained real data.
    """
    height, width = image_shape
    start, end = map(float, time_range)
    if height <= 0 or width <= 0 or not np.isfinite([start, end]).all() or end <= start:
        raise ValueError("image_shape and time_range must be valid.")
    if temporal_bins < 1 or spatial_tolerance < 0:
        raise ValueError("temporal_bins must be positive and spatial_tolerance non-negative.")

    def occupancy(events: EventStream) -> np.ndarray:
        grid = np.zeros((temporal_bins, 2, height, width), dtype=bool)
        valid = (
            (events.x >= 0)
            & (events.x < width)
            & (events.y >= 0)
            & (events.y < height)
            & (events.t >= start)
            & (events.t <= end)
        )
        if np.any(valid):
            bins = np.minimum(((events.t[valid] - start) / (end - start) * temporal_bins).astype(np.int64), temporal_bins - 1)
            polarity = (events.p[valid] > 0).astype(np.int64)
            grid[bins, polarity, events.y[valid], events.x[valid]] = True
        return grid

    def dilate_spatial(grid: np.ndarray) -> np.ndarray:
        if spatial_tolerance == 0:
            return grid
        padded = np.pad(grid, ((0, 0), (0, 0), (spatial_tolerance, spatial_tolerance), (spatial_tolerance, spatial_tolerance)))
        output = np.zeros_like(grid)
        for dy in range(2 * spatial_tolerance + 1):
            for dx in range(2 * spatial_tolerance + 1):
                output |= padded[:, :, dy : dy + height, dx : dx + width]
        return output

    observed_grid = occupancy(observed)
    reference_grid = occupancy(reference)
    observed_count = int(np.count_nonzero(observed_grid))
    reference_count = int(np.count_nonzero(reference_grid))
    matched_observed = int(np.count_nonzero(observed_grid & dilate_spatial(reference_grid)))
    matched_reference = int(np.count_nonzero(reference_grid & dilate_spatial(observed_grid)))
    precision = float(matched_observed / observed_count) if observed_count else float(reference_count == 0)
    recall = float(matched_reference / reference_count) if reference_count else 1.0
    f1 = float(2.0 * precision * recall / (precision + recall)) if precision + recall else 0.0
    return {
        "observed_occupied_bins": observed_count,
        "reference_occupied_bins": reference_count,
        "matched_observed_bins": matched_observed,
        "matched_reference_bins": matched_reference,
        "spatiotemporal_precision": precision,
        "spatiotemporal_recall": recall,
        "spatiotemporal_f1": f1,
        "temporal_bins": temporal_bins,
        "spatial_tolerance_px": spatial_tolerance,
    }


def active_pixel_ids(events: EventStream, *, image_shape: tuple[int, int]) -> np.ndarray:
    """Return sorted unique flattened pixel IDs for use across adjacent windows."""
    height, width = image_shape
    valid = (events.x >= 0) & (events.x < width) & (events.y >= 0) & (events.y < height)
    return np.unique(events.y[valid].astype(np.int64) * width + events.x[valid].astype(np.int64))
