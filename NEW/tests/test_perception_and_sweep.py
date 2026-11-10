from __future__ import annotations

import unittest

import cv2
import numpy as np

from ami_simulator.types import CameraIntrinsics, EventStream
from ami_active_sensing.control.trajectory import TrajectoryParameters, make_trajectory
from ami_active_sensing.evaluation.perception_metrics import event_time_surface, feature_matching_metrics
from ami_active_sensing.evaluation.trajectory_sweep import SweepConfig, evaluate_trajectory


class PerceptionMetricTests(unittest.TestCase):
    def test_time_surface_is_duplicate_invariant(self) -> None:
        original = EventStream(np.asarray([2, 4]), np.asarray([3, 5]), np.asarray([0.02, 0.08]), np.asarray([1, -1], np.int8))
        duplicated = EventStream(np.repeat(original.x, 3), np.repeat(original.y, 3), np.repeat(original.t, 3), np.repeat(original.p, 3))
        first = event_time_surface(original, image_shape=(8, 8), time_range=(0.0, 0.1), decay_seconds=0.05)
        second = event_time_surface(duplicated, image_shape=(8, 8), time_range=(0.0, 0.1), decay_seconds=0.05)
        self.assertTrue(np.array_equal(first, second))
        self.assertGreater(first[1, 3, 2], 0.0)
        self.assertGreater(first[0, 5, 4], 0.0)

    def test_identical_textured_images_match(self) -> None:
        image = np.zeros((160, 160), dtype=np.uint8)
        rng = np.random.default_rng(4)
        for x, y in rng.integers(12, 148, size=(80, 2)):
            cv2.circle(image, (int(x), int(y)), 3, 255, -1)
        metrics = feature_matching_metrics(image, image.copy(), np.eye(3))
        self.assertGreater(metrics["reference_keypoints"], 20)
        self.assertGreater(metrics["repeatability"], 0.95)
        self.assertGreater(metrics["matching_precision"], 0.95)
        self.assertGreater(metrics["ransac_inlier_ratio"], 0.95)

    def test_empty_images_report_failure(self) -> None:
        metrics = feature_matching_metrics(np.zeros((64, 64), np.uint8), np.zeros((64, 64), np.uint8), np.eye(3))
        self.assertEqual(metrics["accepted_matches"], 0)
        self.assertEqual(metrics["matching_failed"], 1)


class SweepIntegrationTests(unittest.TestCase):
    def test_existing_simulator_accepts_new_trajectory(self) -> None:
        height = width = 64
        base = np.zeros((height, width, 3), dtype=np.uint8)
        base[8:56:8, :] = 255
        base[:, 8:56:8] = 255
        frames = np.repeat(base[None], 11, axis=0)
        timestamps = np.linspace(0.0, 0.1, len(frames))
        intrinsics = CameraIntrinsics(width, height, 64.0, 64.0, 31.5, 31.5)
        trajectory = make_trajectory(TrajectoryParameters("constant", angular_speed_rad_s=20.0, duration_s=0.1))
        report = evaluate_trajectory(
            "fixed_fast",
            frames,
            timestamps,
            intrinsics,
            trajectory,
            [0.0, 0.05, 0.1],
            config=SweepConfig(prism_angle_deg=1.0, event_window_seconds=0.05),
        )
        self.assertEqual(report["window_count"], 2)
        self.assertGreater(report["raw_event_count"], 0)
        self.assertEqual(report["trajectory"]["single_dof"], True)

    def test_no_prism_static_scene_has_no_events(self) -> None:
        frames = np.full((3, 16, 20, 3), 128, dtype=np.uint8)
        times = np.asarray([0.0, 0.05, 0.1])
        intrinsics = CameraIntrinsics(20, 16, 20.0, 20.0, 9.5, 7.5)
        trajectory = make_trajectory(TrajectoryParameters("constant", angular_speed_rad_s=0.0, duration_s=0.1))
        report = evaluate_trajectory("no_prism", frames, times, intrinsics, trajectory, [0.0, 0.05, 0.1])
        self.assertEqual(report["raw_event_count"], 0)

    def test_sweep_validates_per_window_homographies(self) -> None:
        frames = np.full((5, 16, 20, 3), 128, dtype=np.uint8)
        times = np.linspace(0.0, 0.2, 5)
        intrinsics = CameraIntrinsics(20, 16, 20.0, 20.0, 9.5, 7.5)
        trajectory = make_trajectory(TrajectoryParameters("constant", angular_speed_rad_s=0.0, duration_s=0.2))
        with self.assertRaisesRegex(ValueError, "shape"):
            evaluate_trajectory(
                "bad_geometry",
                frames,
                times,
                intrinsics,
                trajectory,
                [0.0, 0.1, 0.2],
                homographies_reference_to_candidate=np.eye(3)[None, None],
            )


if __name__ == "__main__":
    unittest.main()
