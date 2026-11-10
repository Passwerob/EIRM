"""Inverse-speed optical excitation: slow motion → faster prism, fast motion → slower prism."""

from __future__ import annotations

import numpy as np

# Low / mid / high natural-motion gears. Faster prism is used only when the
# scene is not already generating a dense event stream. Class order is
# 360, 180, 0 so logit 0 remains the upshift, matching the frozen m0 net.
INVERSE_RPM = (360.0, 180.0, 0.0)
THREE_GEAR_RPM = INVERSE_RPM
QUALITY_INVERSE_RPM = (360.0, 180.0)
QUALITY_FLOOR_RPM = 180.0
# Class 0 is the upshift, matching prism INVERSE_RPM. Slowest camera is last.
CAMERA_RATES = (1.5, 1.0, 0.5)
CAMERA_HOLD_RATE = 1.0
CAMERA_FLOOR_RATE = 0.5
CAMERA_STORM_OCCUPANCY = 0.50
CAMERA_DEFICIT_OCCUPANCY = 0.35
CAMERA_UPSHIFT_SLACK = 0.025


def motion_thresholds(values: np.ndarray, quantiles: tuple[float, float] = (1.0 / 3.0, 2.0 / 3.0)) -> tuple[float, float]:
    """Fit train-only cut points that split motion into slow / mid / fast."""
    samples = np.asarray(values, dtype=np.float64).reshape(-1)
    if samples.size < 3 or not np.all(np.isfinite(samples)):
        raise ValueError("Motion samples must be a finite array with at least three values.")
    low, high = (float(value) for value in np.quantile(samples, quantiles))
    if not low <= high:
        raise ValueError("Motion quantiles must be non-decreasing.")
    return low, high


def inverse_rpm(
    motion: float,
    thresholds: tuple[float, float],
    *,
    previous_density: float = 0.0,
    previous_overlap: float = 0.0,
    storm_density: float = 0.75,
    storm_overlap: float = 0.65,
    floor_rpm: float = 0.0,
) -> float:
    """Map one causal motion score to a slower prism speed as motion rises.

    Event-storm brakes can only downshift. They never increase RPM. ``floor_rpm``
    keeps a minimum excitation so downstream structure is not turned off.
    """
    if not np.isfinite(motion) or not np.isfinite(previous_density) or not np.isfinite(previous_overlap):
        raise ValueError("Motion and storm feedback must be finite.")
    if storm_density < 0.0 or not 0.0 <= storm_overlap <= 1.0:
        raise ValueError("Storm thresholds must be valid.")
    if not np.isfinite(floor_rpm) or floor_rpm < 0.0:
        raise ValueError("floor_rpm must be finite and non-negative.")
    low, high = thresholds
    if motion <= low:
        rpm = INVERSE_RPM[0]
    elif motion <= high:
        rpm = INVERSE_RPM[1]
    else:
        rpm = INVERSE_RPM[2]
    if previous_density >= storm_density or previous_overlap >= storm_overlap:
        rpm = min(rpm, INVERSE_RPM[1])
    if previous_density >= 1.5 * storm_density:
        rpm = INVERSE_RPM[2]
    return float(max(rpm, floor_rpm))


def f1_argmax_rpm(
    f1_fast: float,
    f1_anchor: float,
    *,
    previous_density: float = 0.0,
    previous_overlap: float = 0.0,
    upshift_margin: float = 0.0,
    storm_density: float = 0.75,
    storm_overlap: float = 0.65,
) -> float:
    """Choose 360 RPM only when teacher edge F1 is at least as good as 180 RPM.

    This is the downstream-quality teacher. It does not invert global visual
    motion. Storm brakes can only refuse the upshift.
    """
    if not np.isfinite(f1_fast) or not np.isfinite(f1_anchor) or not np.isfinite(previous_density) or not np.isfinite(previous_overlap) or not np.isfinite(upshift_margin):
        raise ValueError("Teacher F1 values, storm feedback, and upshift_margin must be finite.")
    if storm_density < 0.0 or not 0.0 <= storm_overlap <= 1.0 or upshift_margin < 0.0:
        raise ValueError("Storm thresholds and upshift_margin must be valid.")
    if previous_density >= storm_density or previous_overlap >= storm_overlap:
        return QUALITY_FLOOR_RPM
    if float(f1_fast) + 1e-12 < float(f1_anchor) + float(upshift_margin):
        return QUALITY_FLOOR_RPM
    return 360.0


def f1_argmax_three_rpm(
    f1_off: float,
    f1_anchor: float,
    f1_fast: float,
    *,
    previous_density: float = 0.0,
    previous_overlap: float = 0.0,
    upshift_margin: float = 0.0,
    storm_density: float = 0.75,
    storm_overlap: float = 0.65,
) -> float:
    """Choose among {0, 180, 360} by teacher edge F1. Ties take the lower RPM.

    Storm occupancy refuses 360. A severe storm may turn the prism off. This
    is not the EVEYES qualitative 3-gear teacher: image-wide F1 on that pack
    almost never prefers 0 RPM over 180 RPM.
    """
    scores = (float(f1_off), float(f1_anchor), float(f1_fast))
    if not all(np.isfinite(value) for value in scores) or not np.isfinite(previous_density) or not np.isfinite(previous_overlap) or not np.isfinite(upshift_margin):
        raise ValueError("Teacher F1 values, storm feedback, and upshift_margin must be finite.")
    if storm_density < 0.0 or not 0.0 <= storm_overlap <= 1.0 or upshift_margin < 0.0:
        raise ValueError("Storm thresholds and upshift_margin must be valid.")
    ranked = ((0.0, scores[0]), (180.0, scores[1]), (360.0, scores[2] - float(upshift_margin)))
    if previous_density >= 1.5 * storm_density:
        return 0.0
    if previous_density >= storm_density or previous_overlap >= storm_overlap:
        ranked = ranked[:2]
    best_rpm, best_score = ranked[0]
    for rpm, score in ranked[1:]:
        if score > best_score + 1e-12:
            best_rpm, best_score = rpm, score
    return float(best_rpm)


def f1_argmax_camera_rate(
    f1_floor: float,
    f1_hold: float,
    f1_fast: float,
    *,
    previous_occupancy: float = 0.0,
    upshift_margin: float = 0.0,
    storm_occupancy: float = CAMERA_STORM_OCCUPANCY,
) -> float:
    """Choose among {0.5, 1.0, 1.5}× by teacher edge F1. Ties take the slower camera.

    Storm occupancy refuses 1.5×. This is the camera-path analogue of
    ``f1_argmax_three_rpm``: the student never sees F1 at deploy time.
    """
    scores = (float(f1_floor), float(f1_hold), float(f1_fast))
    if not all(np.isfinite(value) for value in scores) or not np.isfinite(previous_occupancy) or not np.isfinite(upshift_margin):
        raise ValueError("Teacher F1 values, occupancy, and upshift_margin must be finite.")
    if not np.isfinite(storm_occupancy) or storm_occupancy < 0.0 or upshift_margin < 0.0:
        raise ValueError("storm_occupancy and upshift_margin must be finite and non-negative.")
    ranked = (
        (CAMERA_FLOOR_RATE, scores[0]),
        (CAMERA_HOLD_RATE, scores[1]),
        (CAMERA_RATES[0], scores[2] - float(upshift_margin)),
    )
    if previous_occupancy >= storm_occupancy:
        ranked = ranked[:2]
    best_rate, best_score = ranked[0]
    for rate, score in ranked[1:]:
        if score > best_score + 1e-12:
            best_rate, best_score = rate, score
    return float(best_rate)


def deficit_camera_rate(
    f1_floor: float,
    f1_hold: float,
    f1_fast: float,
    *,
    previous_occupancy: float = 0.0,
    deficit_occupancy: float = CAMERA_DEFICIT_OCCUPANCY,
    storm_occupancy: float = CAMERA_STORM_OCCUPANCY,
    upshift_slack: float = CAMERA_UPSHIFT_SLACK,
    upshift_margin: float = 0.0,
) -> float:
    """Low occupancy may upshift to 1.5× if F1 does not collapse versus 1.0×.

    Pure F1-argmax never selects 1.5×: faster path always raises occupancy and
    slightly lowers edge F1. This is the camera analogue of prism excitation
    deficit. Storm occupancy still refuses 1.5×. Mid occupancy uses F1-argmax;
    ties take the slower camera.
    """
    scores = (float(f1_floor), float(f1_hold), float(f1_fast))
    if not all(np.isfinite(value) for value in scores) or not np.isfinite(previous_occupancy):
        raise ValueError("Teacher F1 values and occupancy must be finite.")
    if not np.isfinite(deficit_occupancy) or not np.isfinite(storm_occupancy) or not np.isfinite(upshift_slack) or not np.isfinite(upshift_margin):
        raise ValueError("Deficit, storm, slack, and margin must be finite.")
    if deficit_occupancy < 0.0 or storm_occupancy < 0.0 or upshift_slack < 0.0 or upshift_margin < 0.0:
        raise ValueError("Deficit, storm, slack, and margin must be non-negative.")
    if previous_occupancy >= storm_occupancy:
        return f1_argmax_camera_rate(
            scores[0],
            scores[1],
            scores[2],
            previous_occupancy=previous_occupancy,
            upshift_margin=upshift_margin,
            storm_occupancy=storm_occupancy,
        )
    if previous_occupancy <= deficit_occupancy:
        if scores[2] + float(upshift_slack) + 1e-12 >= scores[1]:
            return CAMERA_RATES[0]
        return f1_argmax_camera_rate(
            scores[0],
            scores[1],
            scores[2],
            previous_occupancy=storm_occupancy,
            upshift_margin=upshift_margin,
            storm_occupancy=storm_occupancy,
        )
    return f1_argmax_camera_rate(
        scores[0],
        scores[1],
        scores[2],
        previous_occupancy=previous_occupancy,
        upshift_margin=upshift_margin,
        storm_occupancy=storm_occupancy,
    )


def three_gear_rpm(
    motion: float,
    thresholds: tuple[float, float],
    *,
    previous_density: float = 0.0,
    previous_overlap: float = 0.0,
    f1_fast: float | None = None,
    f1_anchor: float | None = None,
    upshift_margin: float = 0.0,
    storm_density: float = 0.75,
    storm_overlap: float = 0.65,
) -> float:
    """Causal 0/180/360 law: too little motion may upshift, too much may turn off.

    360 RPM still needs teacher F1 not to regress versus 180 RPM. Fast visual
    motion may select 0 RPM. Storm brakes can only downshift.
    """
    rpm = inverse_rpm(
        motion,
        thresholds,
        previous_density=previous_density,
        previous_overlap=previous_overlap,
        storm_density=storm_density,
        storm_overlap=storm_overlap,
        floor_rpm=0.0,
    )
    if rpm < 360.0:
        return float(rpm)
    if f1_fast is None or f1_anchor is None:
        return 360.0
    if not np.isfinite(f1_fast) or not np.isfinite(f1_anchor) or not np.isfinite(upshift_margin):
        raise ValueError("Teacher F1 values and upshift_margin must be finite.")
    if float(f1_fast) + 1e-12 < float(f1_anchor) + float(upshift_margin):
        return 180.0
    return 360.0


def excitation_deficit_rpm(
    imu_motion: float,
    previous_density: float,
    previous_overlap: float = 0.0,
    *,
    imu_threshold: float,
    density_threshold: float,
    storm_density: float = 0.75,
    storm_overlap: float = 0.65,
) -> float:
    """Speed the prism up when the camera is moving but events are still scarce.

    That is the deployable stand-in for co-motion: ego-motion is present, yet
    the previous window did not produce a dense event field. Object-only
    scenes have ~0 IMU and stay on the 180 RPM floor.
    """
    if not np.isfinite(imu_motion) or not np.isfinite(previous_density) or not np.isfinite(previous_overlap):
        raise ValueError("IMU motion and storm feedback must be finite.")
    if not np.isfinite(imu_threshold) or not np.isfinite(density_threshold) or density_threshold < 0.0:
        raise ValueError("Deficit thresholds must be finite; density_threshold must be non-negative.")
    if storm_density < 0.0 or not 0.0 <= storm_overlap <= 1.0:
        raise ValueError("Storm thresholds must be valid.")
    if previous_density >= storm_density or previous_overlap >= storm_overlap:
        return QUALITY_FLOOR_RPM
    if float(imu_motion) >= float(imu_threshold) and float(previous_density) <= float(density_threshold):
        return 360.0
    return QUALITY_FLOOR_RPM


def quality_inverse_rpm(
    motion: float,
    thresholds: tuple[float, float],
    *,
    previous_density: float = 0.0,
    previous_overlap: float = 0.0,
    f1_fast: float | None = None,
    f1_anchor: float | None = None,
    upshift_margin: float = 0.0,
    storm_density: float = 0.75,
    storm_overlap: float = 0.65,
) -> float:
    """Inverse schedule that never drops below 180 RPM.

    360 RPM is used only in the slow-motion bin, and only when a teacher F1
    score says the upshift is not worse than the 180 RPM anchor.
    """
    rpm = inverse_rpm(
        motion,
        thresholds,
        previous_density=previous_density,
        previous_overlap=previous_overlap,
        storm_density=storm_density,
        storm_overlap=storm_overlap,
        floor_rpm=QUALITY_FLOOR_RPM,
    )
    if rpm <= QUALITY_FLOOR_RPM:
        return QUALITY_FLOOR_RPM
    if f1_fast is None or f1_anchor is None:
        return float(rpm)
    if not np.isfinite(f1_fast) or not np.isfinite(f1_anchor) or not np.isfinite(upshift_margin):
        raise ValueError("Teacher F1 values and upshift_margin must be finite.")
    if float(f1_fast) + 1e-12 < float(f1_anchor) + float(upshift_margin):
        return QUALITY_FLOOR_RPM
    return float(rpm)


def inverse_rpm_index(rpm: float, gears: tuple[float, ...] = INVERSE_RPM) -> int:
    matches = [index for index, value in enumerate(gears) if value == float(rpm)]
    if len(matches) != 1:
        raise ValueError(f"rpm must be one of {gears}.")
    return matches[0]


def spearman_rho(left: np.ndarray, right: np.ndarray) -> float:
    """Return Spearman correlation of two equal-length finite vectors."""
    x = np.asarray(left, dtype=np.float64).reshape(-1)
    y = np.asarray(right, dtype=np.float64).reshape(-1)
    if x.shape != y.shape or len(x) < 3 or not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
        raise ValueError("Spearman inputs must be finite, equal-length, and have at least three samples.")

    def _average_rank(values: np.ndarray) -> np.ndarray:
        order = np.argsort(values, kind="mergesort")
        ranks = np.empty(len(values), dtype=np.float64)
        sorted_values = values[order]
        start = 0
        while start < len(values):
            end = start + 1
            while end < len(values) and sorted_values[end] == sorted_values[start]:
                end += 1
            ranks[order[start:end]] = 0.5 * (start + end - 1) + 1.0
            start = end
        return ranks

    x_center = _average_rank(x) - float(len(x) + 1) / 2.0
    y_center = _average_rank(y) - float(len(y) + 1) / 2.0
    denom = float(np.linalg.norm(x_center) * np.linalg.norm(y_center))
    if denom < 1e-12:
        return 0.0
    return float(np.dot(x_center, y_center) / denom)
