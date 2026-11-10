from __future__ import annotations

import unittest

import numpy as np

from ami_simulator.types import ImuStream
from ami_active_sensing.control.adaptive_controller import RuleBasedAdaptiveController
from ami_active_sensing.control.trajectory import TrajectoryLimits, TrajectoryParameters
from ami_active_sensing.motion.motion_features import (
    EVENT_FEATURE_NAMES,
    build_state_features,
    extract_event_features,
    extract_flow_features,
    extract_imu_features,
    state_vector,
)
from ami_active_sensing.motion.optical_flow import FarnebackFlowBackend, OpticalFlowResult


def candidates() -> list[TrajectoryParameters]:
    return [
        TrajectoryParameters("constant", angular_speed_rad_s=0.0, duration_s=0.1),
        TrajectoryParameters("constant", angular_speed_rad_s=2.0, duration_s=0.1),
        TrajectoryParameters("constant", angular_speed_rad_s=8.0, duration_s=0.1),
    ]


class MotionFeatureTests(unittest.TestCase):
    def test_constant_translation_features(self) -> None:
        flow = np.zeros((8, 10, 2), dtype=np.float32)
        flow[..., 0] = 2.0
        result = OpticalFlowResult(flow, np.ones((8, 10), dtype=bool), 0.5, {})
        features = extract_flow_features(result, low_motion_threshold_px_s=1.0, magnitude_histogram_max_px_s=10.0)
        self.assertAlmostEqual(features["flow_mean_px_s"], 4.0)
        self.assertAlmostEqual(features["flow_low_motion_fraction"], 0.0)
        self.assertAlmostEqual(features["flow_spatial_consistency"], 1.0)
        self.assertAlmostEqual(features["flow_direction_entropy"], 0.0)

    def test_zero_flow_is_low_motion(self) -> None:
        result = OpticalFlowResult(np.zeros((4, 5, 2), np.float32), np.ones((4, 5), bool), 0.1, {})
        features = extract_flow_features(result, low_motion_threshold_px_s=2.0, magnitude_histogram_max_px_s=100.0)
        self.assertEqual(features["flow_low_motion_fraction"], 1.0)
        self.assertEqual(features["flow_spatial_consistency"], 1.0)

    def test_farneback_detects_synthetic_translation(self) -> None:
        first = np.zeros((64, 64, 3), dtype=np.uint8)
        first[20:44, 18:38] = 255
        second = np.zeros_like(first)
        second[20:44, 22:42] = 255
        result = FarnebackFlowBackend().estimate(first, second, dt_seconds=0.1)
        center_dx = float(np.median(result.flow_xy_px[22:42, 22:38, 0]))
        self.assertGreater(center_dx, 2.5)
        self.assertEqual(result.flow_xy_px.shape, (64, 64, 2))

    def test_imu_is_optional_and_has_explicit_availability(self) -> None:
        self.assertEqual(extract_imu_features(None)["imu_available"], 0.0)
        imu = ImuStream(
            np.asarray([0.0, 0.5, 1.0]),
            np.tile([0.0, 0.0, 9.81], (3, 1)),
            np.tile([0.0, 0.0, 2.0], (3, 1)),
        )
        features = extract_imu_features(imu)
        self.assertEqual(features["imu_available"], 1.0)
        self.assertAlmostEqual(features["imu_rotational_energy_rad2_s"], 4.0)
        self.assertEqual(features["imu_dominant_axis_z"], 1.0)

    def test_state_vector_schema_is_stable_for_flow_ablation(self) -> None:
        flow_result = OpticalFlowResult(np.zeros((2, 2, 2), np.float32), np.ones((2, 2), bool), 0.1, {})
        flow = extract_flow_features(flow_result, low_motion_threshold_px_s=1.0, magnitude_histogram_max_px_s=10.0)
        event = extract_event_features({name: 0.0 for name in EVENT_FEATURE_NAMES})
        previous = candidates()[0]
        enabled = build_state_features(flow, event, previous, use_flow=True)
        ablated = build_state_features(flow, event, previous, use_flow=False)
        enabled_vector, enabled_names = state_vector(enabled)
        ablated_vector, ablated_names = state_vector(ablated)
        self.assertEqual(enabled_names, ablated_names)
        self.assertEqual(enabled_vector.shape, ablated_vector.shape)
        self.assertEqual(ablated["flow_low_motion_fraction"], 1.0)
        self.assertLessEqual(enabled["flow_low_motion_fraction"], ablated["flow_low_motion_fraction"])


class ControllerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.limits = TrajectoryLimits(10.0, 1000.0, min_duration_s=0.05, max_duration_s=0.2)

    def test_low_information_deficit_selects_fast_excitation(self) -> None:
        controller = RuleBasedAdaptiveController(candidates(), self.limits, minimum_dwell_s=0.0, effort_weight=0.0, switch_weight=0.0)
        state = {
            "flow_low_motion_fraction": 1.0,
            "edge_coverage": 0.0,
            "event_efficiency": 0.0,
            "temporal_overlap": 1.0,
            "invalid_border_fraction": 0.0,
        }
        selected = controller.select(state, timestamp=0.0)
        self.assertEqual(selected.angular_speed_rad_s, 8.0)

    def test_useful_speed_cap_excludes_event_flooding_candidates(self) -> None:
        controller = RuleBasedAdaptiveController(candidates(), self.limits, minimum_dwell_s=0.0, effort_weight=0.0, switch_weight=0.0, useful_speed_cap_rad_s=5.0)
        state = {
            "flow_low_motion_fraction": 1.0,
            "edge_coverage": 0.0,
            "event_efficiency": 0.0,
            "temporal_overlap": 1.0,
            "invalid_border_fraction": 0.0,
        }
        selected = controller.select(state, timestamp=0.0)
        self.assertEqual(selected.angular_speed_rad_s, 2.0)

    def test_high_natural_motion_and_quality_selects_no_excitation(self) -> None:
        controller = RuleBasedAdaptiveController(candidates(), self.limits, initial_index=2, minimum_dwell_s=0.0, effort_weight=0.0, switch_weight=0.0)
        state = {
            "flow_low_motion_fraction": 0.0,
            "edge_coverage": 1.0,
            "event_efficiency": 1.0,
            "temporal_overlap": 0.0,
            "invalid_border_fraction": 0.0,
        }
        selected = controller.select(state, timestamp=0.0)
        self.assertEqual(selected.angular_speed_rad_s, 0.0)

    def test_event_deficit_is_gated_by_low_natural_motion(self) -> None:
        controller = RuleBasedAdaptiveController(candidates(), self.limits, initial_index=2, minimum_dwell_s=0.0, effort_weight=0.0, switch_weight=0.0)
        state = {
            "flow_low_motion_fraction": 0.0,
            "edge_coverage": 0.0,
            "event_efficiency": 0.0,
            "temporal_overlap": 1.0,
            "invalid_border_fraction": 0.0,
        }
        selected = controller.select(state, timestamp=0.0)
        self.assertEqual(selected.angular_speed_rad_s, 0.0)

    def test_imu_prior_reduces_deficit_without_direct_command(self) -> None:
        controller = RuleBasedAdaptiveController(candidates(), self.limits, minimum_dwell_s=0.0, effort_weight=0.0, switch_weight=0.0)
        state = {
            "flow_low_motion_fraction": 1.0,
            "edge_coverage": 0.0,
            "event_efficiency": 0.0,
            "temporal_overlap": 1.0,
            "invalid_border_fraction": 0.0,
        }
        controller.select(state, timestamp=0.0)
        no_imu_deficit = controller.last_decision.information_deficit
        controller.reset(timestamp=0.0)
        controller.select({**state, "imu_available": 1.0, "imu_gyro_p90_rad_s": 0.005}, timestamp=0.0)
        imu_deficit = controller.last_decision.information_deficit
        self.assertLess(imu_deficit, no_imu_deficit)
        self.assertNotEqual(controller.last_decision.selected_index, -1)

    def test_default_switch_penalty_does_not_hold_unneeded_excitation(self) -> None:
        controller = RuleBasedAdaptiveController(candidates(), self.limits, initial_index=1, minimum_dwell_s=0.0)
        state = {
            "flow_low_motion_fraction": 0.0,
            "edge_coverage": 1.0,
            "event_efficiency": 1.0,
            "temporal_overlap": 0.0,
            "invalid_border_fraction": 0.0,
        }
        selected = controller.select(state, timestamp=0.0)
        self.assertEqual(selected.angular_speed_rad_s, 0.0)

    def test_useful_objective_ignores_event_count(self) -> None:
        controller = RuleBasedAdaptiveController(candidates(), self.limits, minimum_dwell_s=0.0, effort_weight=0.0, switch_weight=0.0)
        base = {
            "flow_low_motion_fraction": 0.5,
            "edge_coverage": 0.5,
            "event_efficiency": 0.5,
            "temporal_overlap": 0.5,
            "invalid_border_fraction": 0.0,
        }
        first = controller.select({**base, "event_count": 1.0}, timestamp=0.0)
        controller.reset(timestamp=0.0)
        second = controller.select({**base, "event_count": 1_000_000.0}, timestamp=0.0)
        self.assertEqual(first, second)

    def test_count_objective_selects_maximum_excitation(self) -> None:
        controller = RuleBasedAdaptiveController(candidates(), self.limits, objective="event_count", minimum_dwell_s=0.0, effort_weight=0.0, switch_weight=0.0)
        selected = controller.select({}, timestamp=0.0)
        self.assertEqual(selected.angular_speed_rad_s, 8.0)

    def test_minimum_dwell_blocks_immediate_switch(self) -> None:
        controller = RuleBasedAdaptiveController(candidates(), self.limits, minimum_dwell_s=0.1, effort_weight=0.0, switch_weight=0.0)
        controller.reset(timestamp=0.0)
        state = {
            "flow_low_motion_fraction": 1.0,
            "edge_coverage": 0.0,
            "event_efficiency": 0.0,
            "temporal_overlap": 1.0,
            "invalid_border_fraction": 0.0,
        }
        self.assertEqual(controller.select(state, timestamp=0.05).angular_speed_rad_s, 0.0)
        self.assertEqual(controller.select(state, timestamp=0.10).angular_speed_rad_s, 8.0)


if __name__ == "__main__":
    unittest.main()
