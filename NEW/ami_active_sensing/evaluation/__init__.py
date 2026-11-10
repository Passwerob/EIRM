"""Event representations, perception metrics, and trajectory sweeps."""

from .perception_metrics import (
    FeatureMatchingConfig,
    event_time_surface,
    feature_matching_metrics,
    time_surface_feature_image,
)
from .trajectory_sweep import SweepConfig, evaluate_trajectory
from .reports import matching_task_utility, select_winners, useful_event_utility, winner_distribution
from .adaptive_rollout import StreamingThresholdState, rollout_adaptive_controller

__all__ = [
    "FeatureMatchingConfig",
    "SweepConfig",
    "evaluate_trajectory",
    "event_time_surface",
    "feature_matching_metrics",
    "time_surface_feature_image",
    "matching_task_utility",
    "select_winners",
    "useful_event_utility",
    "winner_distribution",
    "StreamingThresholdState",
    "rollout_adaptive_controller",
]
