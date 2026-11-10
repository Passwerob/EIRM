from __future__ import annotations

import unittest

import numpy as np

from ami_simulator.types import CameraIntrinsics, ImuStream
from ami_active_sensing.control.trajectory import TrajectoryLimits, TrajectoryParameters
from ami_active_sensing.evaluation.adaptive_rollout import StreamingThresholdState, rollout_adaptive_controller
from ami_active_sensing.evaluation.trajectory_sweep import SweepConfig


class AdaptiveRolloutTests(unittest.TestCase):
    def test_streaming_threshold_matches_one_shot(self) -> None:
        from ami_simulator.events import ThresholdEventSimulator

        rng = np.random.default_rng(5)
        frames = np.clip(rng.uniform(0.1, 0.9, size=(5, 6, 7)), 1e-4, 1.0).astype(np.float32)
        times = np.linspace(0.0, 0.2, len(frames))
        one_shot = ThresholdEventSimulator(0.1).simulate(frames, times)
        streaming = StreamingThresholdState(0.1)
        first = streaming.simulate_interval(frames[:3], times[:3])
        second = streaming.simulate_interval(frames[2:], times[2:])
        combined = first.__class__(np.concatenate([first.x, second.x]), np.concatenate([first.y, second.y]), np.concatenate([first.t, second.t]), np.concatenate([first.p, second.p])).sorted_by_time()
        self.assertEqual(len(one_shot), len(combined))
        self.assertTrue(np.array_equal(one_shot.x, combined.x))
        self.assertTrue(np.allclose(one_shot.t, combined.t))
        self.assertTrue(np.array_equal(one_shot.p, combined.p))

    def test_rollout_is_causal_and_keeps_action_log(self) -> None:
        size = 64
        frame = np.zeros((size, size, 3), dtype=np.uint8)
        frame[12:52:6, :] = 255
        frame[:, 12:52:6] = 255
        frames = np.repeat(frame[None], 25, axis=0)
        times = np.linspace(0.0, 0.2, len(frames))
        edges = np.asarray([0.0, 0.1, 0.2])
        intrinsics = CameraIntrinsics(size, size, size, size, (size - 1) / 2, (size - 1) / 2)
        candidates = [
            TrajectoryParameters("constant", angular_speed_rad_s=0.0, duration_s=0.1),
            TrajectoryParameters("constant", angular_speed_rad_s=12.0, duration_s=0.1),
        ]
        report = rollout_adaptive_controller(
            frames,
            times,
            intrinsics,
            edges,
            candidates,
            TrajectoryLimits(20.0, 500.0, min_duration_s=0.1, max_duration_s=0.1),
            config=SweepConfig(prism_angle_deg=1.0, contrast_threshold=0.15, event_window_seconds=0.1),
        )
        self.assertEqual(report["window_count"], 2)
        self.assertEqual(report["state_count"], 1)
        self.assertEqual(len(report["decisions"]), 2)
        self.assertEqual(len(report["trajectory_segments"]), 2)
        self.assertEqual(len(report["states"]), 1)
        self.assertIn("flow_mean_px_s", report["states"][0])
        self.assertIn("event_density", report["states"][0])
        self.assertTrue(all("selected_parameters" in row for row in report["windows"]))

    def test_rollout_exposes_optional_imu_state(self) -> None:
        size = 64
        frame = np.zeros((size, size, 3), dtype=np.uint8)
        frame[::8, :] = 255
        frames = np.repeat(frame[None], 25, axis=0)
        times = np.linspace(0.0, 0.2, len(frames))
        imu_times = np.linspace(0.0, 0.2, 201)
        imu = ImuStream(imu_times, np.tile([0.0, 0.0, 9.8], (len(imu_times), 1)), np.tile([0.0, 0.0, 0.01], (len(imu_times), 1)))
        intrinsics = CameraIntrinsics(size, size, size, size, (size - 1) / 2, (size - 1) / 2)
        candidates = [TrajectoryParameters("constant", angular_speed_rad_s=0.0, duration_s=0.1)]
        report = rollout_adaptive_controller(frames, times, intrinsics, [0.0, 0.1, 0.2], candidates, TrajectoryLimits(20.0, 500.0, min_duration_s=0.1, max_duration_s=0.1), config=SweepConfig(event_window_seconds=0.1), imu_stream=imu)
        self.assertEqual(report["states"][0]["imu_available"], 1.0)
        self.assertGreater(report["states"][0]["imu_gyro_mean_rad_s"], 0.0)


if __name__ == "__main__":
    unittest.main()
