"""Frozen event representation and feature-matching protocol."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import cv2
import numpy as np

from ami_simulator.types import EventStream


@dataclass(frozen=True)
class FeatureMatchingConfig:
    nfeatures: int = 500
    scale_factor: float = 1.2
    levels: int = 8
    fast_threshold: int = 10
    correspondence_tolerance_px: float = 3.0
    ransac_reprojection_threshold_px: float = 3.0
    minimum_matches: int = 8

    def __post_init__(self) -> None:
        if min(self.nfeatures, self.levels, self.minimum_matches) <= 0 or self.scale_factor <= 1.0:
            raise ValueError("ORB feature counts/levels must be positive and scale_factor must exceed one.")
        if self.fast_threshold < 0 or self.correspondence_tolerance_px <= 0.0 or self.ransac_reprojection_threshold_px <= 0.0:
            raise ValueError("Feature thresholds and tolerances must be valid.")

    def metadata(self) -> dict[str, object]:
        return {"detector": "opencv_orb", "matcher": "hamming_cross_check", **asdict(self)}


def _validate_window(time_range: tuple[float, float]) -> tuple[float, float]:
    start, end = map(float, time_range)
    if not np.isfinite([start, end]).all() or end <= start:
        raise ValueError("time_range must be finite and strictly increasing.")
    return start, end


def event_time_surface(
    events: EventStream,
    *,
    image_shape: tuple[int, int],
    time_range: tuple[float, float],
    decay_seconds: float,
) -> np.ndarray:
    """Return a polarity-aware latest-event surface shaped ``[2,H,W]``.

    Channel zero is negative polarity and channel one is positive polarity.
    Each active value is ``exp(-(window_end - last_event_time) / decay)``.
    Exact duplicate events cannot increase the representation.
    """
    height, width = image_shape
    start, end = _validate_window(time_range)
    if height <= 0 or width <= 0 or not np.isfinite(decay_seconds) or decay_seconds <= 0.0:
        raise ValueError("image_shape and decay_seconds must be positive.")
    valid = (
        (events.x >= 0)
        & (events.x < width)
        & (events.y >= 0)
        & (events.y < height)
        & (events.t >= start)
        & (events.t <= end)
    )
    latest = np.full(2 * height * width, -np.inf, dtype=np.float64)
    if np.any(valid):
        polarity = (events.p[valid] > 0).astype(np.int64)
        flat = polarity * height * width + events.y[valid].astype(np.int64) * width + events.x[valid].astype(np.int64)
        np.maximum.at(latest, flat, events.t[valid])
    active = np.isfinite(latest)
    surface = np.zeros_like(latest, dtype=np.float32)
    surface[active] = np.exp(-(end - latest[active]) / decay_seconds).astype(np.float32)
    return surface.reshape(2, height, width)


def time_surface_feature_image(surface: np.ndarray) -> np.ndarray:
    """Convert a polarity-aware surface into a stable uint8 ORB input."""
    values = np.asarray(surface, dtype=np.float32)
    if values.ndim != 3 or values.shape[0] != 2 or not np.all(np.isfinite(values)):
        raise ValueError("surface must be finite with shape [2,H,W].")
    if values.min(initial=0.0) < 0.0 or values.max(initial=0.0) > 1.0 + 1e-6:
        raise ValueError("surface values must lie in [0,1].")
    activity = np.max(values, axis=0)
    return np.rint(np.clip(activity, 0.0, 1.0) * 255.0).astype(np.uint8)


def _feature_image(representation: np.ndarray) -> np.ndarray:
    values = np.asarray(representation)
    if values.ndim == 3:
        return time_surface_feature_image(values)
    if values.ndim != 2:
        raise ValueError("Feature representation must have shape [H,W] or [2,H,W].")
    if values.dtype == np.uint8:
        return values
    if not np.all(np.isfinite(values)):
        raise ValueError("Feature representation must contain finite values.")
    if values.min(initial=0.0) < 0.0 or values.max(initial=0.0) > 1.0:
        raise ValueError("Floating-point feature images must lie in [0,1].")
    return np.rint(values.astype(np.float32) * 255.0).astype(np.uint8)


def _warp_points(points_xy: np.ndarray, homography: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    points = np.asarray(points_xy, dtype=np.float64).reshape(-1, 2)
    matrix = np.asarray(homography, dtype=np.float64)
    if matrix.shape != (3, 3) or not np.all(np.isfinite(matrix)):
        raise ValueError("homography must be a finite [3,3] matrix.")
    homogeneous = np.column_stack((points, np.ones(len(points)))) @ matrix.T
    valid = np.abs(homogeneous[:, 2]) > 1e-12
    output = np.full((len(points), 2), np.nan, dtype=np.float64)
    output[valid] = homogeneous[valid, :2] / homogeneous[valid, 2:3]
    return output, valid


def feature_matching_metrics(
    reference_representation: np.ndarray,
    candidate_representation: np.ndarray,
    homography_reference_to_candidate: np.ndarray,
    *,
    config: FeatureMatchingConfig | None = None,
) -> dict[str, float | int]:
    """Evaluate ORB detections/matches against a known geometric mapping."""
    cfg = config or FeatureMatchingConfig()
    reference = _feature_image(reference_representation)
    candidate = _feature_image(candidate_representation)
    if reference.shape != candidate.shape:
        raise ValueError("Feature image shapes must match.")
    orb = cv2.ORB_create(
        nfeatures=cfg.nfeatures,
        scaleFactor=cfg.scale_factor,
        nlevels=cfg.levels,
        fastThreshold=cfg.fast_threshold,
    )
    reference_keypoints, reference_descriptors = orb.detectAndCompute(reference, None)
    candidate_keypoints, candidate_descriptors = orb.detectAndCompute(candidate, None)
    reference_xy = np.asarray([keypoint.pt for keypoint in reference_keypoints], dtype=np.float64).reshape(-1, 2)
    candidate_xy = np.asarray([keypoint.pt for keypoint in candidate_keypoints], dtype=np.float64).reshape(-1, 2)

    warped_reference, warp_valid = _warp_points(reference_xy, homography_reference_to_candidate)
    height, width = candidate.shape
    visible = warp_valid & (warped_reference[:, 0] >= 0.0) & (warped_reference[:, 0] <= width - 1.0) & (warped_reference[:, 1] >= 0.0) & (warped_reference[:, 1] <= height - 1.0)
    visible_count = int(np.count_nonzero(visible))
    repeated_count = 0
    if visible_count and len(candidate_xy):
        distances = np.linalg.norm(warped_reference[visible, None, :] - candidate_xy[None, :, :], axis=-1)
        repeated_count = int(np.count_nonzero(np.min(distances, axis=1) <= cfg.correspondence_tolerance_px))
    repeatability = float(repeated_count / visible_count) if visible_count else 0.0

    matches = []
    if reference_descriptors is not None and candidate_descriptors is not None and len(reference_descriptors) and len(candidate_descriptors):
        matches = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True).match(reference_descriptors, candidate_descriptors)
    errors: list[float] = []
    accepted_reference: list[tuple[float, float]] = []
    accepted_candidate: list[tuple[float, float]] = []
    correct = 0
    for match in matches:
        query = match.queryIdx
        predicted = warped_reference[query]
        if visible[query]:
            error = float(np.linalg.norm(predicted - candidate_xy[match.trainIdx]))
            errors.append(error)
            if error <= cfg.correspondence_tolerance_px:
                correct += 1
        accepted_reference.append(tuple(reference_xy[query]))
        accepted_candidate.append(tuple(candidate_xy[match.trainIdx]))

    match_count = len(matches)
    precision = float(correct / match_count) if match_count else 0.0
    recall = float(correct / visible_count) if visible_count else 0.0
    ransac_inlier_ratio = 0.0
    if match_count >= 4:
        _, mask = cv2.findHomography(
            np.asarray(accepted_reference, dtype=np.float32),
            np.asarray(accepted_candidate, dtype=np.float32),
            cv2.RANSAC,
            cfg.ransac_reprojection_threshold_px,
        )
        if mask is not None:
            ransac_inlier_ratio = float(np.mean(mask.astype(bool)))
    failed = match_count < cfg.minimum_matches
    return {
        "reference_keypoints": len(reference_keypoints),
        "candidate_keypoints": len(candidate_keypoints),
        "visible_reference_keypoints": visible_count,
        "repeated_keypoints": repeated_count,
        "repeatability": repeatability,
        "accepted_matches": match_count,
        "correct_matches": correct,
        "matching_precision": precision,
        "matching_recall": recall,
        "ransac_inlier_ratio": ransac_inlier_ratio,
        "median_localization_error_px": float(np.median(errors)) if errors else None,
        "matching_failed": int(failed),
    }
