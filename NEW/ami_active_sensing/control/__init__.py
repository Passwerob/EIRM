"""Trajectory generation and adaptive optical-excitation control."""

from .trajectory import (
    ActuatorConstrainedTrajectory,
    OpticalTrajectory,
    PiecewiseTrajectory,
    TrajectoryLimits,
    TrajectoryParameters,
    make_trajectory,
)
from .adaptive_controller import RuleBasedAdaptiveController, TrajectoryDecision

__all__ = [
    "ActuatorConstrainedTrajectory",
    "OpticalTrajectory",
    "PiecewiseTrajectory",
    "TrajectoryLimits",
    "TrajectoryParameters",
    "make_trajectory",
    "RuleBasedAdaptiveController",
    "TrajectoryDecision",
]
