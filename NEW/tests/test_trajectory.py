from __future__ import annotations

import unittest

import numpy as np

from ami_simulator.rotation import ConstantRotation
from ami_active_sensing.control.trajectory import (
    ActuatorConstrainedTrajectory,
    PiecewiseTrajectory,
    TrajectoryLimits,
    TrajectoryParameters,
    make_trajectory,
    validate_trajectory_limits,
)


class TrajectoryTests(unittest.TestCase):
    def test_constant_matches_existing_rotation(self) -> None:
        rpm = 120.0
        omega = rpm * 2.0 * np.pi / 60.0
        params = TrajectoryParameters("constant", angular_speed_rad_s=omega, duration_s=1.0)
        new = make_trajectory(params, start_time=2.0)
        existing = ConstantRotation(rpm)
        times = np.linspace(2.0, 3.0, 17)
        self.assertTrue(np.allclose(new.theta_values(times, reference_time=2.0), existing.theta_values(times, reference_time=2.0)))

    def test_sinusoidal_derivative_matches_finite_difference(self) -> None:
        params = TrajectoryParameters("sinusoid", amplitude_rad=0.2, frequency_hz=2.0, phase_rad=0.3, duration_s=1.0)
        trajectory = make_trajectory(params)
        times = np.linspace(0.01, 0.99, 1001)
        theta = trajectory.theta_values(times)
        numerical = np.gradient(theta, times)
        self.assertLess(float(np.max(np.abs(numerical[2:-2] - trajectory.omega_values(times)[2:-2]))), 1e-4)

    def test_direction_reverses_angle_and_speed(self) -> None:
        positive = make_trajectory(TrajectoryParameters("triangle", amplitude_rad=0.1, frequency_hz=1.0, direction=1, duration_s=1.0))
        negative = make_trajectory(TrajectoryParameters("triangle", amplitude_rad=0.1, frequency_hz=1.0, direction=-1, duration_s=1.0))
        times = np.linspace(0.0, 1.0, 19)
        self.assertTrue(np.allclose(positive.theta_values(times), -negative.theta_values(times)))
        self.assertTrue(np.allclose(positive.omega_values(times), -negative.omega_values(times)))

    def test_piecewise_angle_is_continuous(self) -> None:
        first = TrajectoryParameters("constant", angular_speed_rad_s=2.0, duration_s=0.5)
        second = TrajectoryParameters("sinusoid", amplitude_rad=0.2, frequency_hz=1.0, duration_s=0.5)
        trajectory = PiecewiseTrajectory([(0.0, first), (0.5, second)])
        around = trajectory.theta_values(np.asarray([0.5 - 1e-9, 0.5, 0.5 + 1e-9]))
        self.assertLess(float(np.max(np.abs(np.diff(around)))), 1e-7)
        self.assertAlmostEqual(float(around[1]), 1.0)

    def test_limits_reject_excess_speed(self) -> None:
        params = TrajectoryParameters("constant", angular_speed_rad_s=11.0, duration_s=0.2)
        with self.assertRaisesRegex(ValueError, "max_speed"):
            validate_trajectory_limits(params, TrajectoryLimits(10.0, 100.0))

    def test_actuator_wrapper_limits_speed_and_acceleration(self) -> None:
        times = np.linspace(0.0, 0.2, 201)
        desired = make_trajectory(TrajectoryParameters("constant", angular_speed_rad_s=20.0, duration_s=0.2))
        limits = TrajectoryLimits(10.0, 50.0)
        constrained = ActuatorConstrainedTrajectory(desired, times, limits)
        omega = constrained.omega_values(times)
        acceleration = np.diff(omega) / np.diff(times)
        self.assertLessEqual(float(np.max(np.abs(omega))), 10.0 + 1e-9)
        self.assertLessEqual(float(np.max(np.abs(acceleration))), 50.0 + 1e-9)
        self.assertTrue(np.all(np.diff(constrained.theta_values(times)) >= 0.0))

    def test_invalid_waveform_parameters_fail_early(self) -> None:
        with self.assertRaisesRegex(ValueError, "requires positive"):
            TrajectoryParameters("sinusoid", amplitude_rad=0.0, frequency_hz=1.0)


if __name__ == "__main__":
    unittest.main()
