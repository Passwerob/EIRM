from __future__ import annotations

import tempfile
import sys
import types
import unittest
from pathlib import Path

import numpy as np
import torch
import h5py

from ami_simulator.events import CompiledCudaThresholdEventSimulator, CudaThresholdEventSimulator, ThresholdEventSimulator, read_events_h5
from ami_simulator.contrast_maximization import contrast_maximization
from ami_simulator.event_quality import active_pixel_ids, compute_reference_edge_mask, event_quality_metrics, event_quality_score, event_signal_fidelity_metrics, target_observability_metrics
from ami_simulator.control import CausalGearSelector, NeuralOpticalModulationController, OpticalModulationController
from ami_simulator.geometry import RotatingWedgePrism
from ami_simulator.dsec import read_dsec_events_h5, read_imu_csv
from ami_simulator.pipeline import SimulationConfig, rgb_to_linear_luminance, run_simulation, temporal_resample_linear
from ami_simulator.rotation import ConstantRotation, ImuGearRotation, ImuRotation, ScheduledRotation, motion_adaptive_hysteresis, motion_quantile_thresholds
from ami_simulator.motion_net import CausalAnchorQGRUPolicy, CausalAnchorQualityResidualPolicy, CausalAnchorResidualGRUPolicy, CausalAnchorResidualPolicy, CausalFlowFusionGearPolicy, CausalFlowFusionQPolicy, CausalFlowFusionRPMPolicy, CausalGatedFlowQPolicy, CausalRPMPolicy, TinyCausalGearPolicy
from ami_simulator.flow import estimate_flow_features, privileged_roi_sequence, summarize_flow, summarize_motion_clusters, summarize_privileged_roi_flow, summarize_residual_flow, summarize_tiled_flow
from ami_simulator.inverse_motion import CAMERA_RATES, INVERSE_RPM, deficit_camera_rate, excitation_deficit_rpm, f1_argmax_camera_rate, f1_argmax_rpm, f1_argmax_three_rpm, inverse_rpm, inverse_rpm_index, motion_thresholds, quality_inverse_rpm, spearman_rho as inverse_spearman_rho, three_gear_rpm
from ami_simulator.types import CameraIntrinsics, EventStream, ImuStream
from ami_simulator.warp import compensate_events
from ami_simulator.vfi import DebugLinearVFIBackend, PracticalRIFEBackend, resample_with_vfi
from scripts.validate_score_task_correlation import normalized_selection_regret, spearman_rho
from scripts.evaluate_anchor_feature_matching import _feature_metrics
from scripts.select_phase_continuous_checkpoint import robust_rollout_metrics
from scripts.train_causal_q_policy import _counterfactual_sequence, checkpoint_epochs
from scripts.evaluate_phase_continuous_control import _summary


def intrinsics() -> CameraIntrinsics:
    return CameraIntrinsics(64, 48, 55.0, 55.0, 31.5, 23.5)


class GeometryTests(unittest.TestCase):
    def test_intrinsics_resize_preserves_pixel_center_mapping(self) -> None:
        resized = CameraIntrinsics(100, 50, 80.0, 60.0, 49.5, 24.5).resized(200, 100)
        self.assertEqual((resized.width, resized.height), (200, 100))
        self.assertTrue(np.isclose(resized.fx, 160.0))
        self.assertTrue(np.isclose(resized.fy, 120.0))
        self.assertTrue(np.isclose(resized.cx, 99.5))
        self.assertTrue(np.isclose(resized.cy, 49.5))

    def test_forward_inverse_round_trip(self) -> None:
        prism = RotatingWedgePrism(intrinsics(), 0.5)
        points = torch.tensor([[2.0, 4.0], [31.5, 23.5], [60.0, 43.0]])
        theta = torch.tensor([0.0, 0.7, 2.0])
        self.assertLess(torch.max(torch.abs(prism.inverse_points(prism.forward_points(points, theta), theta) - points)).item(), 1e-4)

    def test_constant_rotation_period(self) -> None:
        rotation = ConstantRotation(720.0)
        self.assertTrue(np.isclose(rotation.theta_values(np.array([0.0, 1.0 / 12.0]))[-1], 2.0 * np.pi))

    def test_constant_rotation_phase_is_continuous_across_windows(self) -> None:
        rotation = ConstantRotation(720.0)
        first_end = rotation.theta_values(np.array([0.0, 0.01]), reference_time=0.0)[-1]
        second_start = rotation.theta_values(np.array([0.01]), reference_time=0.0)[0]
        self.assertTrue(np.isclose(first_end, second_start))

    def test_scheduled_rotation_has_continuous_phase(self) -> None:
        rotation = ScheduledRotation(np.array([0.0, 0.5, 1.0]), np.array([60.0, 120.0, 60.0]))
        angles = rotation.theta_values(np.array([0.0, 0.5, 1.0]))
        self.assertTrue(np.all(np.diff(angles) >= 0.0))
        self.assertTrue(np.isclose(rotation.omega_values(np.array([0.5]))[0], 120.0 * 2.0 * np.pi / 60.0))

    def test_imu_rotation_integrates_gyro(self) -> None:
        imu = ImuStream(np.array([0.0, 0.5, 1.0]), np.zeros((3, 3)), np.tile([0.0, 0.0, 2.0], (3, 1)))
        rotation = ImuRotation(imu, axis=2)
        self.assertTrue(np.isclose(rotation.theta_values(np.array([0.25, 0.75, 1.0]))[-1], 2.0))

    def test_imu_gear_rotation_uses_hysteresis_and_ramp(self) -> None:
        times = np.arange(0.0, 1.0, 0.1)
        gyro = np.zeros((len(times), 3), dtype=np.float64)
        gyro[:, 2] = [0.0, 0.0, 0.2, 0.2, 0.2, 0.0, 0.0, 0.0, 0.8, 0.8]
        imu = ImuStream(times, np.zeros_like(gyro), gyro)
        rotation = ImuGearRotation(
            imu,
            base_rpm=60.0,
            motion_axis=2,
            speed_thresholds_rad_s=(0.1, 0.5),
            gear_scales=(1.0, 0.5, 0.0),
            smoothing_seconds=0.0,
            hysteresis_rad_s=0.0,
            max_accel_rad_s2=100.0,
        )
        self.assertEqual(rotation.metadata()["policy"], "imu_gear_modulated")
        self.assertEqual(rotation.metadata()["gear_counts"], [5, 3, 2])
        angles = rotation.theta_values(times)
        self.assertTrue(np.all(np.diff(angles) >= 0.0))
        omega = rotation.omega_values(times)
        self.assertEqual(omega.shape, times.shape)
        self.assertTrue(np.all(omega >= 0.0))

    def test_motion_quantile_thresholds_are_scene_specific(self) -> None:
        imu = ImuStream(
            t=np.arange(10, dtype=np.float64),
            gyro=np.column_stack((np.arange(10, dtype=np.float64), np.zeros((10, 2)))),
            accel=np.zeros((10, 3), dtype=np.float64),
        )
        thresholds = motion_quantile_thresholds(imu)
        self.assertEqual(len(thresholds), 4)
        self.assertTrue(all(left < right for left, right in zip(thresholds[:-1], thresholds[1:])))
        self.assertGreaterEqual(motion_adaptive_hysteresis(imu), 1e-3)

    def test_per_event_compensation(self) -> None:
        prism = RotatingWedgePrism(intrinsics(), 0.5)
        rotation = ConstantRotation(720.0)
        reference = np.array([[15.0, 12.0], [31.5, 23.5], [50.0, 35.0]], dtype=np.float32)
        timestamps = np.array([0.001, 0.02, 0.05])
        theta = rotation.theta_values(timestamps).astype(np.float32)
        sensor = prism.forward_points(torch.from_numpy(reference), torch.from_numpy(theta)).numpy()
        events = EventStream(np.rint(sensor[:, 0]), np.rint(sensor[:, 1]), timestamps, np.array([-1, 1, -1], dtype=np.int8))
        restored = compensate_events(events, prism, rotation, theta0=0.0, reference_time=0.0)
        recovered = np.column_stack((restored.extras["x_reference"], restored.extras["y_reference"]))
        self.assertLess(np.max(np.linalg.norm(recovered - reference, axis=1)), 0.8)


class PipelineTests(unittest.TestCase):

    def test_flow_summary_captures_translation_and_is_scale_normalized(self) -> None:
        flow = np.zeros((16, 20, 2), dtype=np.float32)
        flow[..., 0] = 2.0
        features = summarize_flow(flow)
        self.assertEqual(features.shape, (8,))
        self.assertGreater(features[0], 0.0)
        self.assertGreater(features[4], 0.99)
        self.assertTrue(np.all(np.isfinite(features)))

    def test_flow_summary_region_mask_excludes_outside_motion(self) -> None:
        flow = np.zeros((8, 8, 2), dtype=np.float32)
        flow[:, :4, 0] = 1.0
        flow[:, 4:, 0] = 5.0
        mask = np.zeros((8, 8), dtype=bool)
        mask[:, :4] = True
        features = summarize_flow(flow, region_mask=mask)
        self.assertTrue(np.isclose(features[0], 1.0 / np.hypot(8.0, 8.0), atol=1e-6))

    def test_residual_flow_removes_dominant_translation(self) -> None:
        flow = np.zeros((8, 8, 2), dtype=np.float32)
        flow[..., 0] = 3.0
        flow[2:4, 2:4, 0] = 0.0
        features = summarize_residual_flow(flow)
        self.assertLess(float(features[0]), 0.2)
        self.assertGreater(float(features[5]), 0.0)

    def test_tiled_flow_preserves_local_motion_mixture_without_mask(self) -> None:
        flow = np.zeros((8, 8, 2), dtype=np.float32)
        flow[:, :4, 0] = 1.0
        flow[:, 4:, 0] = 5.0
        features = summarize_tiled_flow(flow, rows=2, cols=2)
        self.assertEqual(features.shape, (32,))
        self.assertLess(float(features[0]), float(features[8]))
        self.assertTrue(np.all(np.isfinite(features)))

    def test_motion_cluster_flow_is_fixed_size_and_finite(self) -> None:
        flow = np.zeros((8, 8, 2), dtype=np.float32)
        flow[:, :4, 0] = 1.0
        flow[:, 4:, 0] = 5.0
        features = summarize_motion_clusters(flow, clusters=3)
        self.assertEqual(features.shape, (27,))
        self.assertTrue(np.all(np.isfinite(features)))

    def test_privileged_roi_flow_splits_object_and_background(self) -> None:
        flow = np.zeros((8, 8, 2), dtype=np.float32)
        flow[:, :4, 0] = 1.0
        flow[:, 4:, 0] = 5.0
        mask = np.zeros((8, 8), dtype=bool)
        mask[:, :4] = True
        features = summarize_privileged_roi_flow(flow, mask)
        self.assertEqual(features.shape, (16,))
        self.assertTrue(np.isclose(features[0], 1.0 / np.hypot(8.0, 8.0), atol=1e-6))
        self.assertTrue(np.isclose(features[8], 5.0 / np.hypot(8.0, 8.0), atol=1e-6))

    def test_privileged_roi_sequence_is_causal_and_zero_on_first_window(self) -> None:
        frames = np.zeros((4, 8, 8), dtype=np.float32)
        frames[1, :, :4] = 1.0
        frames[2, :, :4] = 0.4
        frames[3, :, :4] = 0.1
        masks = np.zeros((4, 8, 8), dtype=bool)
        masks[:, :, :4] = True
        sequence = privileged_roi_sequence(frames, masks)
        self.assertEqual(sequence.shape, (3, 16))
        self.assertTrue(np.all(sequence[0] == 0.0))
        self.assertGreater(float(sequence[1, 0]), 0.0)
        self.assertTrue(np.all(np.isfinite(sequence)))


class InverseMotionTests(unittest.TestCase):
    def test_inverse_rpm_slows_down_as_motion_rises(self) -> None:
        thresholds = motion_thresholds(np.asarray([0.0, 1.0, 2.0, 3.0, 4.0, 5.0]))
        slow = inverse_rpm(0.0, thresholds)
        mid = inverse_rpm(float(np.mean(thresholds)), thresholds)
        fast = inverse_rpm(10.0, thresholds)
        self.assertEqual(slow, INVERSE_RPM[0])
        self.assertEqual(mid, INVERSE_RPM[1])
        self.assertEqual(fast, INVERSE_RPM[2])
        self.assertGreater(slow, fast)

    def test_quality_floor_never_turns_the_prism_off(self) -> None:
        thresholds = (1.0, 2.0)
        self.assertEqual(inverse_rpm(10.0, thresholds, floor_rpm=180.0), 180.0)
        self.assertEqual(quality_inverse_rpm(0.0, thresholds, f1_fast=0.40, f1_anchor=0.50), 180.0)
        self.assertEqual(quality_inverse_rpm(0.0, thresholds, f1_fast=0.55, f1_anchor=0.50), 360.0)
        self.assertEqual(quality_inverse_rpm(10.0, thresholds, f1_fast=0.90, f1_anchor=0.50), 180.0)

    def test_storm_brake_can_only_downshift(self) -> None:
        thresholds = (1.0, 2.0)
        self.assertEqual(inverse_rpm(0.0, thresholds, previous_density=0.8), INVERSE_RPM[1])
        self.assertEqual(inverse_rpm(0.0, thresholds, previous_density=1.2), INVERSE_RPM[2])
        self.assertEqual(inverse_rpm(10.0, thresholds, previous_density=0.8), INVERSE_RPM[2])

    def test_f1_argmax_upshifts_only_when_teacher_improves(self) -> None:
        self.assertEqual(f1_argmax_rpm(0.40, 0.50), 180.0)
        self.assertEqual(f1_argmax_rpm(0.55, 0.50), 360.0)
        self.assertEqual(f1_argmax_rpm(0.90, 0.50, previous_density=0.8), 180.0)

    def test_f1_argmax_three_picks_lowest_rpm_on_ties(self) -> None:
        self.assertEqual(f1_argmax_three_rpm(0.50, 0.50, 0.50), 0.0)
        self.assertEqual(f1_argmax_three_rpm(0.10, 0.50, 0.55), 360.0)
        self.assertEqual(f1_argmax_three_rpm(0.10, 0.50, 0.40), 180.0)
        self.assertEqual(f1_argmax_three_rpm(0.90, 0.50, 0.80, previous_density=0.8), 0.0)
        self.assertEqual(f1_argmax_three_rpm(0.10, 0.50, 0.90, previous_density=1.2), 0.0)

    def test_f1_argmax_camera_rate_picks_slowest_on_ties(self) -> None:
        self.assertEqual(f1_argmax_camera_rate(0.50, 0.50, 0.50), 0.5)
        self.assertEqual(f1_argmax_camera_rate(0.40, 0.50, 0.55), 1.5)
        self.assertEqual(f1_argmax_camera_rate(0.40, 0.50, 0.45), 1.0)
        self.assertEqual(f1_argmax_camera_rate(0.40, 0.50, 0.90, previous_occupancy=0.50), 1.0)
        self.assertEqual(f1_argmax_camera_rate(0.60, 0.50, 0.90, previous_occupancy=0.80), 0.5)
        self.assertEqual(tuple(CAMERA_RATES), (1.5, 1.0, 0.5))
        self.assertEqual(inverse_rpm_index(1.5, CAMERA_RATES), 0)

    def test_deficit_camera_rate_upshifts_when_occupancy_is_scarce(self) -> None:
        self.assertEqual(deficit_camera_rate(0.58, 0.56, 0.54, previous_occupancy=0.20, upshift_slack=0.025), 1.5)
        self.assertEqual(deficit_camera_rate(0.58, 0.56, 0.50, previous_occupancy=0.20, upshift_slack=0.025), 0.5)
        self.assertEqual(deficit_camera_rate(0.40, 0.50, 0.55, previous_occupancy=0.40), 1.5)
        self.assertEqual(deficit_camera_rate(0.40, 0.50, 0.54, previous_occupancy=0.60), 1.0)
        self.assertEqual(deficit_camera_rate(0.58, 0.56, 0.54, previous_occupancy=0.20, deficit_occupancy=0.10, upshift_slack=0.025), 0.5)

    def test_three_gear_turns_off_when_natural_motion_is_fast(self) -> None:
        thresholds = (1.0, 2.0)
        self.assertEqual(three_gear_rpm(0.0, thresholds, f1_fast=0.40, f1_anchor=0.50), 180.0)
        self.assertEqual(three_gear_rpm(0.0, thresholds, f1_fast=0.55, f1_anchor=0.50), 360.0)
        self.assertEqual(three_gear_rpm(1.5, thresholds, f1_fast=0.90, f1_anchor=0.50), 180.0)
        self.assertEqual(three_gear_rpm(10.0, thresholds, f1_fast=0.90, f1_anchor=0.50), 0.0)
        self.assertEqual(three_gear_rpm(0.0, thresholds, previous_density=0.8, f1_fast=0.90, f1_anchor=0.50), 180.0)
        self.assertEqual(three_gear_rpm(0.0, thresholds, previous_density=1.2, f1_fast=0.90, f1_anchor=0.50), 0.0)

    def test_excitation_deficit_needs_imu_and_low_density(self) -> None:
        self.assertEqual(excitation_deficit_rpm(0.0, 0.1, imu_threshold=0.3, density_threshold=0.3), 180.0)
        self.assertEqual(excitation_deficit_rpm(0.6, 0.1, imu_threshold=0.3, density_threshold=0.3), 360.0)
        self.assertEqual(excitation_deficit_rpm(0.6, 0.5, imu_threshold=0.3, density_threshold=0.3), 180.0)
        self.assertEqual(excitation_deficit_rpm(0.6, 0.1, previous_overlap=0.9, imu_threshold=0.3, density_threshold=0.3), 180.0)

    def test_spearman_detects_inverse_schedule(self) -> None:
        motion = np.asarray([0.0, 1.0, 2.0, 3.0])
        rpm = np.asarray([360.0, 360.0, 180.0, 0.0])
        self.assertLess(inverse_spearman_rho(motion, rpm), 0.0)
        self.assertEqual(inverse_spearman_rho(motion, np.full(4, 180.0)), 0.0)


class PipelineMoreTests(unittest.TestCase):
    def test_dense_flow_feature_sequence_is_finite(self) -> None:
        frames = np.zeros((3, 32, 32, 3), dtype=np.uint8)
        frames[0, 8:20, 6:18] = 255
        frames[1, 8:20, 8:20] = 255
        frames[2, 8:20, 10:22] = 255
        features = estimate_flow_features(frames, np.array([0.0, 0.1, 0.2]))
        self.assertEqual(features.shape, (2, 8))
        self.assertTrue(np.all(np.isfinite(features)))
        self.assertGreater(float(features[:, 1].mean()), 0.0)

    def test_flow_fusion_lstm_is_bounded(self) -> None:
        policy = CausalFlowFusionRPMPolicy(feature_dim=17, hidden_dim=16, min_rpm=0.0, max_rpm=1080.0)
        prediction = policy(torch.randn(4, 8, 17))
        self.assertEqual(tuple(prediction.shape), (4,))
        self.assertTrue(bool(torch.all((prediction >= 0.0) & (prediction <= 1080.0))))

    def test_anchor_residual_gru_output_shapes(self) -> None:
        policy = CausalAnchorResidualGRUPolicy(feature_dim=20, hidden_dim=16, residual_actions=2)
        delta, switch = policy(torch.randn(3, 8, 20))
        self.assertEqual(tuple(delta.shape), (3, 2))
        self.assertEqual(tuple(switch.shape), (3,))

    def test_flow_fusion_gear_lstm_shape(self) -> None:
        policy = CausalFlowFusionGearPolicy(feature_dim=17, hidden_dim=16, num_gears=5)
        logits = policy(torch.randn(4, 8, 17))
        self.assertEqual(tuple(logits.shape), (4, 5))

    def test_flow_fusion_q_lstm_shape(self) -> None:
        policy = CausalFlowFusionQPolicy(feature_dim=17, hidden_dim=16, num_gears=5)
        utilities = policy(torch.randn(4, 8, 17))
        self.assertEqual(tuple(utilities.shape), (4, 5))
        self.assertEqual(policy.num_actions, 5)

    def test_gated_flow_q_starts_as_exact_base_policy(self) -> None:
        base = CausalFlowFusionQPolicy(feature_dim=11, hidden_dim=16, num_gears=5)
        gated = CausalGatedFlowQPolicy(feature_dim=19, hidden_dim=16, num_gears=5)
        gated.base_policy.load_state_dict(base.state_dict())
        values = torch.randn(4, 8, 19)
        base_values = torch.cat((values[..., :9], values[..., 17:]), dim=-1)
        self.assertTrue(torch.allclose(gated(values), base(base_values), atol=1e-7))
        self.assertLess(float(gated.gate_values(values).mean()), 0.13)

    def test_phase_q_feature_layout_keeps_flow_and_phase_separate(self) -> None:
        count, gears = 2, 5
        matrix = np.full((count, gears), 0.5, dtype=np.float32)
        oracle = {
            "prism_phase_start": np.array([0.0, np.pi / 2]),
            "imu_motion": np.array([0.2, 0.3]),
            "active_masks_packed": np.zeros((count, gears, 1), dtype=np.uint8),
            "active_mask_pixel_count": np.array(8),
            "edge_f1": matrix,
            "coverage": matrix,
            "unique_ratio": matrix,
            "event_efficiency": matrix,
            "polarity_balance": np.zeros_like(matrix),
            "densities": matrix,
            "target_density": np.ones(count, dtype=np.float32),
            "valid_fraction": np.ones_like(matrix),
            "structural_ratio": matrix,
            "flow_features": np.tile(np.arange(8, dtype=np.float32), (count, 1)),
        }
        objective = {"event_efficiency_weight": 0.5, "overlap_weight": 0.5, "density_weight": 0.1, "invalid_fraction_weight": 0.25, "rpm_effort_weight": 0.05, "rpm_change_weight": 0.05}
        features, utilities = _counterfactual_sequence(oracle, np.array([1, 2]), use_flow=True, motion_delta=False, rpm_values=np.array([1080.0, 720.0, 360.0, 180.0, 0.0]), objective=objective)
        self.assertEqual(features.shape, (count, 19))
        self.assertEqual(utilities.shape, (count, gears))
        self.assertTrue(np.array_equal(features[1, 9:17], np.arange(8, dtype=np.float32)))
        self.assertTrue(np.allclose(features[:, -2:], [[0.0, 1.0], [1.0, 0.0]], atol=1e-6))

    def test_motion_delta_is_delayed_and_does_not_use_current_window(self) -> None:
        count, gears = 3, 5
        matrix = np.full((count, gears), 0.5, dtype=np.float32)
        oracle = {
            "prism_phase_start": np.zeros(count),
            "imu_motion": np.array([0.2, 0.5, 0.1]),
            "active_masks_packed": np.zeros((count, gears, 1), dtype=np.uint8),
            "active_mask_pixel_count": np.array(8),
            "edge_f1": matrix,
            "coverage": matrix,
            "unique_ratio": matrix,
            "event_efficiency": matrix,
            "polarity_balance": np.zeros_like(matrix),
            "densities": matrix,
            "target_density": np.ones(count, dtype=np.float32),
            "valid_fraction": np.ones_like(matrix),
            "structural_ratio": matrix,
            "flow_features": np.zeros((count, 8), dtype=np.float32),
        }
        objective = {"event_efficiency_weight": 0.5, "overlap_weight": 0.5, "density_weight": 0.1, "invalid_fraction_weight": 0.25, "rpm_effort_weight": 0.05, "rpm_change_weight": 0.05}
        features, _ = _counterfactual_sequence(oracle, np.array([1, 2, 3]), use_flow=True, motion_delta=True, rpm_values=np.array([1080.0, 720.0, 360.0, 180.0, 0.0]), objective=objective)
        self.assertEqual(features.shape, (count, 20))
        self.assertTrue(np.allclose(features[:, 9], [0.0, 0.0, 0.3]))
        self.assertTrue(np.array_equal(features[1, 10:18], np.zeros(8, dtype=np.float32)))

    def test_edge_event_efficiency_is_zero_when_only_background_events_exist(self) -> None:
        events = EventStream(
            np.array([0, 1, 2]), np.array([0, 1, 2]), np.array([0.1, 0.2, 0.3]), np.ones(3, dtype=np.int8)
        )
        edge_mask = np.zeros((4, 4), dtype=bool)
        edge_mask[3, 3] = True
        metrics = event_quality_metrics(events, reference_edge_mask=edge_mask, image_shape=(4, 4))
        self.assertGreater(metrics["event_efficiency"], 0.0)
        self.assertEqual(metrics["edge_event_efficiency"], 0.0)

    def test_direct_rollout_candidate_epochs_are_predeclared_and_bounded(self) -> None:
        self.assertEqual(checkpoint_epochs(500), [0, 10, 20, 50, 100, 200, 500])
        self.assertEqual(checkpoint_epochs(15), [0, 10, 15])
        self.assertEqual(checkpoint_epochs(20, [20, 5, 5, -1, 30]), [5, 20])

    def test_robust_rollout_selection_balances_mean_and_worst_group(self) -> None:
        rows = [
            {"condition": condition, "policy": "learned_causal", "windows": 2, "mean_objective": objective, "mean_edge_f1": f1, "mean_rpm": 180.0}
            for condition, objective, f1 in zip(
                ("camera_only", "object_only", "co_motion", "counter_motion"),
                (0.9, 0.8, 0.5, 0.7),
                (0.6, 0.5, 0.4, 0.5),
            )
        ]
        metrics = robust_rollout_metrics({"rows": rows})
        self.assertTrue(np.isclose(metrics["balanced_objective_mean"], 0.725))
        self.assertEqual(metrics["worst_condition_objective"], 0.5)
        self.assertTrue(np.isclose(metrics["robust_selection_score"], 0.6125))

    def test_direct_summary_reports_fixed_motion_speed_bins(self) -> None:
        records = [{"utility": float(i), "event_density": 0.1, "event_efficiency": 0.8, "edge_precision": 0.5, "edge_coverage": 0.6, "edge_f1": 0.55, "temporal_overlap": 0.2, "coverage_efficiency": 0.48} for i in range(4)]
        report = _summary(records, [180.0, 180.0, 360.0, 360.0], [False, True, False], np.array([0.1, 0.3, 0.6, 0.9]))
        self.assertEqual(set(report["motion_bins"]), {"0.00-0.25", "0.25-0.50", "0.50-0.75", "0.75-1.00"})
        self.assertEqual(report["motion_bin_worst_objective"], 0.0)
        self.assertEqual(report["motion_bin_objective_range"], 3.0)

    def test_direct_summary_detects_speed_transitions_and_recovery(self) -> None:
        records = [{"utility": value, "event_density": 0.1, "event_efficiency": 0.8, "edge_precision": 0.5, "edge_coverage": 0.6, "edge_f1": 0.55, "temporal_overlap": 0.2, "coverage_efficiency": 0.48} for value in (1.0, 1.0, 1.0, 0.2, 0.95, 0.95, 0.95)]
        report = _summary(records, [180.0] * len(records), [False] * len(records), np.array([0.1, 0.1, 0.1, 0.8, 0.8, 0.8, 0.8]))
        self.assertEqual(len(report["transitions"]), 1)
        self.assertTrue(np.isclose(report["transitions"][0]["retention_ratio"], 0.2))
        self.assertEqual(report["transitions"][0]["recovery_latency_windows"], 1)

    def test_score_task_correlation_statistics(self) -> None:
        values = np.array([0.1, 0.2, 0.3, 0.4])
        self.assertTrue(np.isclose(spearman_rho(values, values), 1.0))
        self.assertTrue(np.isclose(spearman_rho(values, values[::-1]), -1.0))
        regret, agreement = normalized_selection_regret(
            np.array([0.1, 0.9, 0.2, 0.8]),
            np.array([0.0, 1.0, 0.1, 0.7]),
            np.array(["a", "a", "b", "b"]),
        )
        self.assertEqual(regret, 0.0)
        self.assertEqual(agreement, 1.0)

    def test_vfi_resampling_preserves_endpoints_and_timestamps(self) -> None:
        frames = np.stack((np.zeros((2, 3, 3), np.float32), np.ones((2, 3, 3), np.float32)))
        values, times = resample_with_vfi(frames, np.array([0.0, 0.1]), target_fps=40.0, backend=DebugLinearVFIBackend())
        self.assertTrue(np.allclose(times, [0.0, 0.025, 0.05, 0.075, 0.1]))
        self.assertTrue(np.allclose(values[:, 0, 0, 0], [0.0, 0.25, 0.5, 0.75, 1.0]))

    def test_practical_rife_uses_direct_arbitrary_timestep(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            repository = Path(temporary) / "Practical-RIFE"
            checkpoint = repository / "train_log"
            checkpoint.mkdir(parents=True)
            captured: dict[str, object] = {}

            class FakeModel:
                def __init__(self, arbitrary: bool = False) -> None:
                    captured["arbitrary"] = arbitrary

                def load_model(self, path: str, rank: int) -> None:
                    captured["checkpoint"] = (path, rank)

                def eval(self) -> None:
                    pass

                def inference(self, first, second, timestep: float = 0.5):
                    captured.setdefault("timesteps", []).append(timestep)
                    return first * (1.0 - timestep) + second * timestep

            model_package = types.ModuleType("model")
            rife_module = types.ModuleType("model.RIFE")
            rife_module.Model = FakeModel
            previous_model = sys.modules.get("model")
            previous_rife = sys.modules.get("model.RIFE")
            sys.modules["model"] = model_package
            sys.modules["model.RIFE"] = rife_module
            try:
                backend = PracticalRIFEBackend(repository, device="cpu")
                output = backend.interpolate(np.zeros((32, 32, 3), np.float32), np.ones((32, 32, 3), np.float32), np.array([0.25, 0.75]))
            finally:
                if previous_model is None:
                    sys.modules.pop("model", None)
                else:
                    sys.modules["model"] = previous_model
                if previous_rife is None:
                    sys.modules.pop("model.RIFE", None)
                else:
                    sys.modules["model.RIFE"] = previous_rife
            self.assertTrue(captured["arbitrary"])
            self.assertEqual(captured["timesteps"], [0.25, 0.75])
            self.assertTrue(np.allclose(output[:, 0, 0, 0], [0.25, 0.75]))

    def test_event_quality_penalizes_repeated_pixels(self) -> None:
        image = np.zeros((8, 8), dtype=np.float32)
        image[:, 4:] = 1.0
        first = EventStream(np.array([3, 4]), np.array([2, 2]), np.array([0.01, 0.02]), np.array([1, -1], dtype=np.int8))
        repeated = EventStream(np.array([3, 4, 3, 4]), np.array([2, 2, 2, 2]), np.arange(4) * 0.01, np.array([1, -1, 1, -1], dtype=np.int8))
        first_metrics = event_quality_metrics(first, reference_intensity=image, image_shape=image.shape)
        repeated_metrics = event_quality_metrics(repeated, reference_intensity=image, previous_active_pixels=active_pixel_ids(first, image_shape=image.shape), image_shape=image.shape)
        self.assertGreater(first_metrics["edge_coverage"], 0.0)
        self.assertGreater(repeated_metrics["temporal_overlap"], 0.0)
        self.assertEqual(first_metrics["event_efficiency"], 1.0)
        self.assertEqual(repeated_metrics["event_efficiency"], 0.5)
        self.assertLess(event_quality_score(repeated_metrics), event_quality_score(first_metrics))

    def test_event_efficiency_penalizes_within_window_repetition(self) -> None:
        image = np.zeros((8, 8), dtype=np.float32)
        image[:, 4:] = 1.0
        efficient = EventStream(np.array([3, 4]), np.array([2, 2]), np.array([0.01, 0.02]), np.array([1, -1], dtype=np.int8))
        repeated = EventStream(np.array([3, 4, 3, 4]), np.array([2, 2, 2, 2]), np.arange(4) * 0.01, np.array([1, -1, 1, -1], dtype=np.int8))
        efficient_metrics = event_quality_metrics(efficient, reference_intensity=image, image_shape=image.shape)
        repeated_metrics = event_quality_metrics(repeated, reference_intensity=image, image_shape=image.shape)
        self.assertLess(repeated_metrics["event_efficiency"], efficient_metrics["event_efficiency"])
        self.assertLess(event_quality_score(repeated_metrics, density_weight=0.0), event_quality_score(efficient_metrics, density_weight=0.0))

    def test_coverage_efficiency_rewards_unique_target_support(self) -> None:
        image = np.zeros((8, 8), dtype=np.float32)
        image[:, 4:] = 1.0
        events = EventStream(np.array([3, 3, 4]), np.array([2, 2, 2]), np.arange(3) * 0.01, np.ones(3, dtype=np.int8))
        metrics = event_quality_metrics(events, reference_intensity=image, image_shape=image.shape)
        self.assertTrue(np.isclose(metrics["coverage_efficiency"], metrics["edge_coverage"] * metrics["event_efficiency"]))
        self.assertLess(metrics["coverage_efficiency"], metrics["edge_coverage"])

    def test_event_quality_does_not_call_flat_image_an_edge(self) -> None:
        image = np.full((4, 4), 0.25, dtype=np.float32)
        events = EventStream(np.array([1]), np.array([1]), np.array([0.01]), np.array([1], dtype=np.int8))
        metrics = event_quality_metrics(events, reference_intensity=image, image_shape=image.shape)
        self.assertEqual(metrics["edge_coverage"], 0.0)
        self.assertEqual(metrics["structural_ratio"], 0.0)

    def test_edge_f1_prefers_structural_events_at_equal_count(self) -> None:
        image = np.zeros((8, 8), dtype=np.float32)
        image[:, 4:] = 1.0
        structured = EventStream(np.full(8, 3), np.arange(8), np.linspace(0.0, 0.07, 8), np.ones(8, dtype=np.int8))
        off_edge = EventStream(np.zeros(8), np.arange(8), np.linspace(0.0, 0.07, 8), np.ones(8, dtype=np.int8))
        structured_metrics = event_quality_metrics(structured, reference_intensity=image, image_shape=image.shape)
        off_edge_metrics = event_quality_metrics(off_edge, reference_intensity=image, image_shape=image.shape)
        self.assertEqual(len(structured), len(off_edge))
        self.assertGreater(structured_metrics["edge_precision"], off_edge_metrics["edge_precision"])
        self.assertGreater(structured_metrics["edge_f1"], off_edge_metrics["edge_f1"])

    def test_quality_score_rejects_coverage_from_event_flooding(self) -> None:
        common = {
            "structural_ratio": 0.3,
            "unique_pixel_ratio": 0.2,
            "event_efficiency": 0.5,
            "temporal_overlap": 0.2,
            "polarity_imbalance": 0.0,
            "event_density": 1.0,
        }
        flooded = {**common, "edge_coverage": 0.95, "edge_f1": 0.15}
        informative = {**common, "edge_coverage": 0.70, "edge_f1": 0.60}
        self.assertGreater(event_quality_score(informative), event_quality_score(flooded))

    def test_coverage_efficiency_can_be_explicitly_weighted(self) -> None:
        metrics = {"edge_f1": 0.2, "edge_coverage": 0.8, "unique_pixel_ratio": 0.1, "event_efficiency": 0.5, "coverage_efficiency": 0.4, "temporal_overlap": 0.0, "polarity_imbalance": 0.0, "event_density": 1.0}
        base = event_quality_score(metrics, density_weight=0.0, efficiency_weight=0.0)
        weighted = event_quality_score(metrics, density_weight=0.0, efficiency_weight=0.0, coverage_efficiency_weight=1.0)
        self.assertTrue(np.isclose(weighted - base, 0.4))

    def test_frozen_clean_edge_mask_is_independent_of_noisy_reference(self) -> None:
        clean = np.zeros((8, 8), dtype=np.float32)
        clean[:, 4:] = 1.0
        noisy = np.random.default_rng(3).uniform(0.01, 1.0, clean.shape).astype(np.float32)
        edge_mask = compute_reference_edge_mask(clean)
        events = EventStream(np.full(8, 3), np.arange(8), np.linspace(0.0, 0.07, 8), np.ones(8, dtype=np.int8))
        clean_metrics = event_quality_metrics(events, reference_intensity=clean, reference_edge_mask=edge_mask, image_shape=clean.shape)
        noisy_metrics = event_quality_metrics(events, reference_intensity=noisy, reference_edge_mask=edge_mask, image_shape=clean.shape)
        self.assertEqual(clean_metrics["edge_f1"], noisy_metrics["edge_f1"])

    def test_teo_ignores_background_events_and_zeros_empty_targets(self) -> None:
        target = np.zeros((8, 8), dtype=bool)
        target[2:6, 2:6] = True
        edges = np.zeros((8, 8), dtype=bool)
        edges[2:6, 2] = True
        on_edge = EventStream(np.full(4, 2), np.arange(2, 6), np.linspace(0.01, 0.04, 4), np.ones(4, dtype=np.int8))
        with_background = EventStream(
            np.concatenate([on_edge.x, np.zeros(8)]),
            np.concatenate([on_edge.y, np.arange(8)]),
            np.concatenate([on_edge.t, np.linspace(0.011, 0.039, 8)]),
            np.ones(12, dtype=np.int8),
        )
        background_only = EventStream(np.zeros(8), np.arange(8), np.linspace(0.01, 0.04, 8), np.ones(8, dtype=np.int8))
        interior = EventStream(np.full(4, 4), np.arange(2, 6), np.linspace(0.01, 0.04, 4), np.ones(4, dtype=np.int8))
        burst = EventStream(np.full(4, 2), np.arange(2, 6), np.full(4, 0.011), np.ones(4, dtype=np.int8))
        window = (0.0, 0.05)
        clean = target_observability_metrics(on_edge, target_mask=target, target_edge_mask=edges, time_range=window)
        cluttered = target_observability_metrics(with_background, target_mask=target, target_edge_mask=edges, time_range=window)
        empty = target_observability_metrics(background_only, target_mask=target, target_edge_mask=edges, time_range=window)
        filled = target_observability_metrics(interior, target_mask=target, target_edge_mask=edges, time_range=window)
        brief = target_observability_metrics(burst, target_mask=target, target_edge_mask=edges, time_range=window)
        self.assertEqual(clean["target_coverage"], cluttered["target_coverage"])
        self.assertEqual(clean["target_purity"], cluttered["target_purity"])
        self.assertEqual(clean["teo"], cluttered["teo"])
        self.assertEqual(empty["teo"], 0.0)
        self.assertEqual(empty["target_coverage"], 0.0)
        self.assertEqual(filled["target_purity"], 0.0)
        self.assertEqual(filled["teo"], 0.0)
        self.assertGreater(clean["target_temporal_support"], brief["target_temporal_support"])
        self.assertGreater(clean["teo"], brief["teo"])


    def test_signal_fidelity_rejects_noise_and_ignores_duplicates(self) -> None:
        reference = EventStream(np.array([3, 3, 3, 3]), np.array([1, 2, 3, 4]), np.array([0.01, 0.02, 0.03, 0.04]), np.array([1, -1, 1, -1], dtype=np.int8))
        duplicated = EventStream(np.repeat(reference.x, 2), np.repeat(reference.y, 2), np.repeat(reference.t, 2), np.repeat(reference.p, 2))
        shifted = EventStream(reference.x + 1, reference.y, reference.t, reference.p)
        noise = EventStream(np.zeros(4), reference.y, reference.t, reference.p)
        duplicate_metrics = event_signal_fidelity_metrics(duplicated, reference, image_shape=(8, 8), time_range=(0.0, 0.05), spatial_tolerance=1)
        shifted_metrics = event_signal_fidelity_metrics(shifted, reference, image_shape=(8, 8), time_range=(0.0, 0.05), spatial_tolerance=1)
        noise_metrics = event_signal_fidelity_metrics(noise, reference, image_shape=(8, 8), time_range=(0.0, 0.05), spatial_tolerance=1)
        self.assertEqual(duplicate_metrics["observed_occupied_bins"], duplicate_metrics["reference_occupied_bins"])
        self.assertEqual(duplicate_metrics["spatiotemporal_f1"], 1.0)
        self.assertEqual(shifted_metrics["spatiotemporal_f1"], 1.0)
        self.assertLess(noise_metrics["spatiotemporal_f1"], shifted_metrics["spatiotemporal_f1"])

    def test_quality_controller_overrides_high_motion_prior_when_coverage_is_low(self) -> None:
        controller = OpticalModulationController(initial_rpm=720.0, step_rpm=120.0)
        next_rpm = controller.update(
            {"edge_coverage": 0.1, "temporal_overlap": 0.8, "structural_ratio": 0.1},
            dt_seconds=0.05,
            natural_motion_ratio=1.0,
        )
        self.assertGreater(next_rpm, 720.0)

    def test_quality_controller_reduces_redundant_speed(self) -> None:
        controller = OpticalModulationController(initial_rpm=720.0, step_rpm=120.0)
        next_rpm = controller.update(
            {"edge_coverage": 0.9, "temporal_overlap": 0.9, "structural_ratio": 0.8},
            dt_seconds=0.05,
        )
        self.assertLess(next_rpm, 720.0)

    def test_causal_gear_selector_enforces_dwell_and_margin(self) -> None:
        selector = CausalGearSelector(np.array([1080.0, 720.0, 360.0]), initial_rpm=720.0, initial_time=0.0, minimum_dwell_seconds=0.05, switch_margin=0.1)
        selected, unconstrained = selector.select(np.array([0.0, 0.2, 0.3]), timestamp=0.01)
        self.assertEqual((selected, unconstrained), (0, 0))
        selected, unconstrained = selector.select(np.array([0.3, 0.2, 0.0]), timestamp=0.02)
        self.assertEqual((selected, unconstrained), (0, 2))
        selected, unconstrained = selector.select(np.array([0.3, 0.2, 0.0]), timestamp=0.07)
        self.assertEqual((selected, unconstrained), (2, 2))
        selected, unconstrained = selector.select(np.array([0.25, 0.2, 0.21]), timestamp=0.13)
        self.assertEqual((selected, unconstrained), (2, 1))

    def test_neural_controller_rate_limits_policy_output(self) -> None:
        policy = CausalRPMPolicy(feature_dim=14, hidden_dim=16, min_rpm=0.0, max_rpm=1440.0)
        controller = NeuralOpticalModulationController(policy, initial_rpm=720.0, max_accel_rpm_s=100.0)
        next_rpm = controller.update(np.zeros((4, 14), dtype=np.float32), dt_seconds=0.1)
        self.assertLessEqual(abs(next_rpm - 720.0), 10.0)

    def test_tiny_gear_policy_shape(self) -> None:
        policy = TinyCausalGearPolicy(feature_dim=8, hidden_dim=16, num_gears=5)
        logits = policy(torch.zeros((3, 12, 8)))
        self.assertEqual(tuple(logits.shape), (3, 5))
        self.assertTrue(torch.allclose(policy.probabilities(torch.zeros((3, 12, 8))).sum(dim=-1), torch.ones(3)))

    def test_learned_camera_nfb_holds_until_history_and_storm_blocks_upshift(self) -> None:
        from ami_simulator.camera_nfb import FEATURE_DIM, WINDOW, camera_feature, choose_learned_camera_rate, sample_path

        volume = np.arange(4, dtype=np.float64)[:, None, None] * np.ones((4, 2, 2))
        self.assertTrue(np.allclose(sample_path(volume, 1.5)[0, 0], 1.5))
        model = TinyCausalGearPolicy(feature_dim=FEATURE_DIM, hidden_dim=8, num_gears=3)
        checkpoint = {
            "candidate_rates": (1.5, 1.0, 0.5),
            "window": WINDOW,
            "feature_mean": [0.0, 0.0, 0.0, 0.0],
            "feature_std": [1.0, 1.0, 1.0, 1.0],
            "feature_clip": 5.0,
            "storm_occupancy": 0.50,
        }
        short = [camera_feature(0.0, 0.0, 0.0, 1.0)]
        self.assertEqual(choose_learned_camera_rate(short, model, checkpoint, previous_occupancy=0.0), 1.0)
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.zero_()
            model.head[-1].bias.zero_()
            model.head[-1].bias[0] = 5.0
        history = [camera_feature(0.1, 0.2, 0.2, 1.0) for _ in range(WINDOW)]
        self.assertEqual(choose_learned_camera_rate(history, model, checkpoint, previous_occupancy=0.2), 1.5)
        self.assertEqual(choose_learned_camera_rate(history, model, checkpoint, previous_occupancy=0.6), 1.0)
        checkpoint["deficit_occupancy"] = 0.35
        self.assertEqual(choose_learned_camera_rate(history, model, checkpoint, previous_occupancy=0.20), 1.5)
        self.assertEqual(choose_learned_camera_rate(history, model, checkpoint, previous_occupancy=0.40), 1.0)

    def test_anchor_residual_policy_outputs_gains_and_confidence(self) -> None:
        policy = CausalAnchorResidualPolicy(feature_dim=20, hidden_dim=16, residual_actions=2)
        delta, switch = policy(torch.zeros((3, 8, 20)))
        self.assertEqual(tuple(delta.shape), (3, 2))
        self.assertEqual(tuple(switch.shape), (3,))
        self.assertTrue(torch.all(torch.isfinite(delta)))
        self.assertTrue(torch.all(torch.isfinite(switch)))

    def test_anchor_quality_residual_policy_outputs_efficiency_delta(self) -> None:
        policy = CausalAnchorQualityResidualPolicy(feature_dim=20, hidden_dim=16, residual_actions=2)
        delta, switch, efficiency = policy(torch.zeros((3, 8, 20)))
        self.assertEqual(tuple(delta.shape), (3, 2))
        self.assertEqual(tuple(switch.shape), (3,))
        self.assertEqual(tuple(efficiency.shape), (3, 2))
        self.assertTrue(torch.all(torch.isfinite(efficiency)))

    def test_feature_matching_protocol_reports_finite_metrics(self) -> None:
        from ami_simulator.types import EventStream

        image = np.zeros((48, 64), dtype=np.float32)
        image[12:36, 16:48] = 1.0
        events = EventStream(
            np.asarray([16, 47, 16, 47], dtype=np.int32),
            np.asarray([12, 12, 35, 35], dtype=np.int32),
            np.linspace(0.0, 0.003, 4),
            np.asarray([1, -1, 1, -1], dtype=np.int8),
        )
        metrics = _feature_metrics(image, events)
        self.assertIn("spatial_f1", metrics)
        self.assertTrue(np.all(np.isfinite([float(value) for value in metrics.values()])))

    def test_causal_rpm_policy_is_bounded(self) -> None:
        policy = CausalRPMPolicy(feature_dim=14, hidden_dim=16, min_rpm=180.0, max_rpm=1440.0)
        values = policy(torch.zeros((3, 12, 14)))
        self.assertEqual(tuple(values.shape), (3,))
        self.assertTrue(torch.all(values >= 180.0))
        self.assertTrue(torch.all(values <= 1440.0))

    def test_dsec_event_offset_and_polarity(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "events.h5"
            with h5py.File(path, "w") as handle:
                group = handle.create_group("events")
                group["x"] = [1, 2]
                group["y"] = [3, 4]
                group["t"] = [10, 20]
                group["p"] = [0, 1]
                handle["t_offset"] = 1_000_000
            events = read_dsec_events_h5(path)
            self.assertTrue(np.allclose(events.t, [1.00001, 1.00002]))
            self.assertTrue(np.array_equal(events.p, [-1, 1]))

    def test_imu_csv_units(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "imu.csv"
            path.write_text("timestamp_us,ax,ay,az,gx,gy,gz\n0,0,0,1,0,0,180\n1000000,0,0,1,0,0,180\n", encoding="utf-8")
            imu = read_imu_csv(path, timestamp_unit="us", gyro_unit="deg/s")
            self.assertTrue(np.isclose(imu.t[-1], 1.0))
            self.assertTrue(np.isclose(imu.gyro[0, 2], np.pi))

    def test_imu_csv_unit_annotated_columns(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "eveyes_imu.csv"
            path.write_text(
                "timestamp_s,timestamp_us,gyro_x_rad_s,gyro_y_rad_s,gyro_z_rad_s,gyro_norm_rad_s,accel_x_m_s2,accel_y_m_s2,accel_z_m_s2\n"
                "0,0,0.1,0.2,0.3,0.374,0,0.5,9.8\n"
                "0.001,1000,0.2,0.3,0.4,0.539,0,0.5,9.8\n",
                encoding="utf-8",
            )
            imu = read_imu_csv(path, timestamp_unit="s", gyro_unit="rad/s")
            self.assertTrue(np.allclose(imu.gyro[0], [0.1, 0.2, 0.3]))
            self.assertTrue(np.allclose(imu.accel[1], [0.0, 0.5, 9.8]))
    def test_luminance_is_not_normalized_per_frame(self) -> None:
        frames = np.zeros((2, 2, 2, 3), dtype=np.float32)
        frames[0] = 0.1
        frames[1] = 0.2
        luminance = rgb_to_linear_luminance(frames)
        self.assertTrue(np.all(luminance[1] > luminance[0]))

    def test_resampling_obeys_timestamps(self) -> None:
        frames = np.array([np.zeros((1, 1, 3)), np.ones((1, 1, 3))], dtype=np.float32)
        output, times = temporal_resample_linear(frames, np.array([0.0, 0.01]), 400.0)
        self.assertTrue(np.allclose(times, [0.0, 0.0025, 0.005, 0.0075, 0.01]))
        self.assertTrue(np.allclose(output[:, 0, 0, 0], [0.0, 0.25, 0.5, 0.75, 1.0]))

    def test_pipeline_writes_compensated_h5(self) -> None:
        frames = np.zeros((2, 16, 20, 3), dtype=np.float32)
        frames[1, 4:12, 5:15] = 1.0
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "output"
            summary = run_simulation(frames, np.array([0.0, 0.01]), CameraIntrinsics(20, 16, 20.0, 20.0, 9.5, 7.5), output, config=SimulationConfig(prism_angle_deg=0.2, rpm=720.0, contrast_threshold=0.2))
            events = read_events_h5(output / "events_compensated.h5")
            self.assertEqual(len(events), summary["compensated_event_count"])
            self.assertEqual(set(events.extras), {"x_reference", "y_reference"})

    def test_pipeline_requires_explicit_vfi_backend(self) -> None:
        frames = np.zeros((2, 4, 4, 3), dtype=np.float32)
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaisesRegex(ValueError, "explicit VFI backend"):
                run_simulation(frames, np.array([0.0, 0.1]), CameraIntrinsics(4, 4, 4.0, 4.0, 1.5, 1.5), temporary, config=SimulationConfig(interpolation_fps=40.0))

    def test_pipeline_accepts_explicit_debug_vfi_backend(self) -> None:
        frames = np.zeros((2, 4, 4, 3), dtype=np.float32)
        frames[1, :, 2:] = 1.0
        with tempfile.TemporaryDirectory() as temporary:
            summary = run_simulation(
                frames,
                np.array([0.0, 0.1]),
                CameraIntrinsics(4, 4, 4.0, 4.0, 1.5, 1.5),
                temporary,
                config=SimulationConfig(interpolation_fps=40.0),
                vfi_backend=DebugLinearVFIBackend(),
            )
            self.assertEqual(summary["simulated_frame_count"], 5)
            self.assertEqual(summary["vfi"]["vfi_backend"], "debug_linear")

    def test_threshold_event_timing_and_polarity(self) -> None:
        frames = np.stack([np.full((2, 2), 0.5), np.full((2, 2), 0.5 * np.exp(0.45))])
        events = ThresholdEventSimulator(0.2).simulate(frames, np.array([2.0, 2.01]))
        self.assertGreater(len(events), 0)
        self.assertGreaterEqual(events.t.min(), 2.0)
        self.assertLessEqual(events.t.max(), 2.01)
        self.assertTrue(np.all(events.p == 1))

    def test_threshold_state_matches_single_sequence_across_windows(self) -> None:
        values = np.exp(np.asarray([0.0, 0.13, 0.26, 0.39], dtype=np.float64))[:, None, None] * 0.4
        timestamps = np.asarray([0.0, 0.01, 0.02, 0.03])
        simulator = ThresholdEventSimulator(0.2)
        complete = simulator.simulate(values, timestamps)
        first, state = simulator.simulate_with_state(values[:3], timestamps[:3])
        second, _ = simulator.simulate_with_state(values[2:], timestamps[2:], reference_log=state)
        self.assertEqual(len(complete), len(first) + len(second))
        self.assertTrue(np.array_equal(complete.p, np.concatenate((first.p, second.p))))
        self.assertTrue(np.allclose(complete.t, np.concatenate((first.t, second.t))))

    def test_threshold_state_changes_window_reset_result(self) -> None:
        values = np.exp(np.asarray([0.0, 0.12, 0.24], dtype=np.float64))[:, None, None] * 0.5
        timestamps = np.asarray([0.0, 0.01, 0.02])
        simulator = ThresholdEventSimulator(0.2)
        _, state = simulator.simulate_with_state(values[:2], timestamps[:2])
        continued, _ = simulator.simulate_with_state(values[1:], timestamps[1:], reference_log=state)
        reset = simulator.simulate(values[1:], timestamps[1:])
        self.assertGreater(len(continued), len(reset))

    def test_horizon2_q_policy_returns_finite_action_values(self) -> None:
        policy = CausalAnchorQGRUPolicy(feature_dim=47, hidden_dim=16, num_actions=3)
        values = policy(torch.zeros(2, 8, 47))
        self.assertEqual(tuple(values.shape), (2, 3))
        self.assertTrue(torch.isfinite(values).all())

    def test_contrast_maximization_selects_candidate(self) -> None:
        events = EventStream(
            np.asarray([2, 3, 4, 5], dtype=np.int32),
            np.asarray([4, 4, 4, 4], dtype=np.int32),
            np.asarray([0.0, 0.1, 0.2, 0.3], dtype=np.float64),
            np.ones(4, dtype=np.int8),
        )
        result = contrast_maximization(events, (8, 12), [(-10.0, 0.0), (-5.0, 0.0), (0.0, 0.0), (5.0, 0.0)])
        self.assertIn(result["velocity"], [(-10.0, 0.0), (-5.0, 0.0), (0.0, 0.0), (5.0, 0.0)])
        self.assertGreaterEqual(float(result["contrast"]), 0.0)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is not available in this test environment")
    def test_cuda_threshold_matches_cpu_event_statistics(self) -> None:
        v, u = np.meshgrid(np.arange(12, dtype=np.float32), np.arange(16, dtype=np.float32), indexing="ij")
        frames = np.stack([0.5 + 0.1 * np.sin((u + index) / 3.0) + 0.1 * np.cos((v - index) / 4.0) for index in range(4)]).astype(np.float32)
        timestamps = np.array([0.0, 0.002, 0.004, 0.006])
        cpu = ThresholdEventSimulator(0.1).simulate(frames, timestamps)
        cuda = CudaThresholdEventSimulator(0.1).simulate(torch.from_numpy(frames).cuda(), timestamps)
        self.assertEqual(len(cuda), len(cpu))
        self.assertEqual(int(np.count_nonzero(cuda.p > 0)), int(np.count_nonzero(cpu.p > 0)))
        self.assertEqual(int(np.count_nonzero(cuda.p < 0)), int(np.count_nonzero(cpu.p < 0)))
        self.assertTrue(np.all(np.diff(cuda.t) >= 0.0))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is not available in this test environment")
    def test_compiled_cuda_threshold_matches_cpu_event_statistics(self) -> None:
        v, u = np.meshgrid(np.arange(12, dtype=np.float32), np.arange(16, dtype=np.float32), indexing="ij")
        frames = np.stack([0.5 + 0.1 * np.sin((u + index) / 3.0) + 0.1 * np.cos((v - index) / 4.0) for index in range(4)]).astype(np.float32)
        timestamps = np.array([0.0, 0.002, 0.004, 0.006])
        cpu = ThresholdEventSimulator(0.1).simulate(frames, timestamps)
        cuda = CompiledCudaThresholdEventSimulator(0.1, max_events_per_pixel=8).simulate(torch.from_numpy(frames).cuda(), timestamps)
        self.assertEqual(len(cuda), len(cpu))
        self.assertEqual(int(np.count_nonzero(cuda.p > 0)), int(np.count_nonzero(cpu.p > 0)))
        self.assertEqual(int(np.count_nonzero(cuda.p < 0)), int(np.count_nonzero(cpu.p < 0)))
        self.assertTrue(np.all(np.diff(cuda.t) >= 0.0))


if __name__ == "__main__":
    unittest.main()
