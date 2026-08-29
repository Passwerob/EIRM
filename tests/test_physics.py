from __future__ import annotations

import numpy as np
import torch
import imageio.v3 as iio
import sys
import types

from ami_sim.events import DebugThresholdEventSimulator, read_events_h5, write_events_h5
from ami_sim.analysis.validation import event_density_in_mask, filter_events_to_mask
from ami_sim.pipeline import interpolate_sequence_to_count, rgb_to_luminance
from ami_sim.datasets import DSECRGBSequence
from ami_sim.metrics import edge_coverage, event_spatial_variance
from ami_sim.vfi import PracticalRIFEBackend
from ami_sim.prism import (
    ConstantRotationPolicy,
    GlobalCircularApproximation,
    PaperSimplifiedPrismModel,
    PrismWarper,
    compensate_events,
)
from ami_sim.types import CameraIntrinsics, EventStream


def intrinsics() -> CameraIntrinsics:
    return CameraIntrinsics(64, 48, 55.0, 55.0, 31.5, 23.5)


def test_global_circle_has_circular_trajectory() -> None:
    model = GlobalCircularApproximation(intrinsics(), amplitude_px=5.0)
    point = torch.tensor([[31.5, 23.5]])
    positions = torch.cat([
        model.forward_points(point, torch.tensor(theta)) for theta in np.linspace(0.0, 2.0 * np.pi, 9)[:-1]
    ])
    radius = torch.linalg.vector_norm(positions - point, dim=-1)
    assert torch.allclose(radius, torch.full_like(radius, 5.0), atol=1e-5)


def test_constant_720_rpm_period() -> None:
    policy = ConstantRotationPolicy(720.0)
    period = 1.0 / 12.0
    state = policy.states_at(np.array([0.0, period]))[-1]
    assert np.isclose(state.omega, 24.0 * np.pi)
    assert np.isclose(state.theta, 2.0 * np.pi)


def test_constant_rotation_closed_form_matches_state_integration() -> None:
    policy = ConstantRotationPolicy(720.0)
    times = np.linspace(0.25, 0.35, 17)
    integrated = np.asarray([state.theta for state in policy.states_at(times, theta0=0.3, reference_time=0.2)])
    closed_form = policy.theta_values(times, theta0=0.3, reference_time=0.2)
    assert np.allclose(closed_form, integrated)


def test_paper_ray_warp_forward_inverse() -> None:
    model = PaperSimplifiedPrismModel(intrinsics(), prism_angle_deg=0.5)
    points = torch.tensor([[2.0, 4.0], [31.5, 23.5], [60.0, 43.0]], dtype=torch.float32)
    theta = torch.tensor([0.0, 0.7, 2.0], dtype=torch.float32)
    warped = model.forward_points(points, theta)
    restored = model.inverse_points(warped, theta)
    assert torch.max(torch.abs(restored - points)).item() < 1e-4


def test_rpm_zero_warp_is_fixed_in_time() -> None:
    model = GlobalCircularApproximation(intrinsics(), amplitude_px=3.0)
    policy = ConstantRotationPolicy(0.0)
    warper = PrismWarper(model, policy, 48, 64, reference_time=10.0)
    frame = torch.rand((1, 48, 64))
    frames = frame.unsqueeze(0).repeat(3, 1, 1, 1)
    warped, valid = warper.warp(frames, np.array([10.0, 10.01, 10.02]))
    assert torch.allclose(warped[0], warped[1])
    assert torch.equal(valid[0], valid[1])


def test_dewarp_is_inverse_image_sampling_direction() -> None:
    model = PaperSimplifiedPrismModel(intrinsics(), prism_angle_deg=0.0)
    warper = PrismWarper(model, ConstantRotationPolicy(720.0), 48, 64, reference_time=0.0)
    frame = torch.rand((2, 1, 48, 64))
    timestamps = np.array([0.0, 0.01])
    warped, _ = warper.warp(frame, timestamps)
    restored, valid = warper.dewarp(warped, timestamps)
    assert torch.allclose(restored, frame, atol=1e-5)
    assert bool(valid.all())


def test_compensation_is_per_event_and_preserves_polarity() -> None:
    model = GlobalCircularApproximation(intrinsics(), amplitude_px=7.0)
    policy = ConstantRotationPolicy(720.0)
    reference = np.array([[15.0, 12.0], [31.5, 23.5], [50.0, 35.0]], dtype=np.float32)
    times = np.array([0.001, 0.02, 0.05])
    theta = np.asarray([policy.theta_at(t, reference_time=0.0) for t in times])
    sensor = model.forward_points(torch.from_numpy(reference), torch.from_numpy(theta.astype(np.float32))).numpy()
    events = EventStream(np.rint(sensor[:, 0]), np.rint(sensor[:, 1]), times, np.array([-1, 1, -1], dtype=np.int8))
    compensated = compensate_events(events, model, policy, reference_time=0.0)
    # Integer sensor coordinates introduce at most half a pixel before inverse mapping.
    assert np.max(np.linalg.norm(np.stack((compensated.extras["x_float"], compensated.extras["y_float"]), axis=-1) - reference, axis=-1)) < 0.8
    assert np.array_equal(compensated.p, events.p)


def test_compensation_restores_original_order_for_unsorted_events() -> None:
    model = GlobalCircularApproximation(intrinsics(), amplitude_px=4.0)
    policy = ConstantRotationPolicy(720.0)
    times = np.array([0.04, 0.01, 0.03, 0.02])
    events = EventStream(
        np.array([10, 20, 30, 40]),
        np.array([11, 21, 31, 41]),
        times,
        np.array([1, -1, 1, -1], dtype=np.int8),
    )
    compensated = compensate_events(events, model, policy, reference_time=0.0)
    assert np.array_equal(compensated.t, times)
    assert np.array_equal(compensated.x, events.x)
    assert np.array_equal(compensated.p, events.p)
    assert len(compensated.extras["x_float"]) == len(events)


def test_event_timestamps_and_polarity() -> None:
    frames = np.stack([np.full((4, 4), 0.5), np.full((4, 4), 0.5 * np.exp(0.45))]).astype(np.float32)
    timestamps = np.array([2.0, 2.01])
    events = DebugThresholdEventSimulator(0.2).simulate(frames, timestamps)
    assert len(events) > 0
    assert events.t.min() >= timestamps[0]
    assert events.t.max() <= timestamps[1]
    assert np.all(events.p == 1)


def test_h5_round_trip_preserves_compensation_fields(tmp_path) -> None:
    events = EventStream(np.array([2, 1]), np.array([3, 4]), np.array([0.2, 0.1]), np.array([1, -1], dtype=np.int8))
    events = events.with_extras(x_float=np.array([2.1, 1.1]), y_float=np.array([3.1, 4.1]))
    path = tmp_path / "events.h5"
    write_events_h5(path, events)
    restored = read_events_h5(path)
    assert np.all(np.diff(restored.t) >= 0)
    assert set(restored.extras) == {"x_float", "y_float"}
    assert np.array_equal(restored.p, np.array([-1, 1], dtype=np.int8))


def test_event_density_respects_region_and_float_compensation_coordinates() -> None:
    events = EventStream(np.array([0, 3]), np.array([0, 3]), np.array([0.0, 0.01]), np.array([1, -1], dtype=np.int8))
    events = events.with_extras(x_float=np.array([0.1, 1.1]), y_float=np.array([0.1, 1.1]))
    mask = np.zeros((4, 4), dtype=bool)
    mask[:2, :2] = True
    assert np.isclose(event_density_in_mask(events, mask, (4, 4)), 0.25)
    assert np.isclose(event_density_in_mask(events, mask, (4, 4), compensated=True), 0.5)


def test_valid_mask_filter_uses_float_compensated_coordinates() -> None:
    events = EventStream(np.array([0, 3]), np.array([0, 3]), np.array([0.0, 0.01]), np.array([1, -1], dtype=np.int8))
    events = events.with_extras(x_float=np.array([0.1, 1.1]), y_float=np.array([0.1, 1.1]))
    mask = np.zeros((4, 4), dtype=bool)
    mask[1, 1] = True
    integer_filtered = filter_events_to_mask(events, mask, (4, 4))
    float_filtered = filter_events_to_mask(events, mask, (4, 4), compensated=True)
    assert len(integer_filtered) == 0
    assert len(float_filtered) == 1
    assert float_filtered.p[0] == -1


def test_luminance_has_no_per_frame_normalisation() -> None:
    rgb = np.zeros((2, 2, 2, 3), dtype=np.float32)
    rgb[0] = 0.1
    rgb[1] = 0.2
    intensity = rgb_to_luminance(rgb)
    assert np.all(intensity[1] > intensity[0])
    ratio = float(intensity[1, 0, 0] / intensity[0, 0, 0])
    assert 3.0 < ratio < 4.0


def test_fixed_count_interpolation_normalises_mixed_rgb_ranges(tmp_path) -> None:
    rgb_dir = tmp_path / "rgb"
    rgb_dir.mkdir()
    for index, value in enumerate((0, 255)):
        iio.imwrite(rgb_dir / f"{index:06d}.png", np.full((8, 8, 3), value, dtype=np.uint8))
    timestamps = tmp_path / "timestamps.csv"
    timestamps.write_text("0,0.0\n1,1.0\n", encoding="utf-8")
    sequence = DSECRGBSequence.from_paths(rgb_dir, timestamps, timestamp_unit="s")
    output, target_times, _ = interpolate_sequence_to_count(
        sequence,
        type("Linear", (), {"interpolate": lambda self, a, b, targets: np.stack([np.full_like(a, tau, dtype=np.float32) for tau in targets])})(),
        3,
    )
    assert output.shape == (3, 8, 8, 3)
    assert np.allclose(output[:, 0, 0, 0], [0.0, 0.5, 1.0])
    assert np.allclose(target_times, [0.0, 0.5, 1.0])


def test_compensated_synthetic_edge_is_sharper() -> None:
    model = GlobalCircularApproximation(intrinsics(), amplitude_px=8.0)
    policy = ConstantRotationPolicy(720.0)
    reference = np.repeat([[32.0, 24.0]], 24, axis=0).astype(np.float32)
    times = np.linspace(0.0, 1.0 / 24.0, len(reference), endpoint=False)
    theta = np.asarray([policy.theta_at(value, reference_time=0.0) for value in times], dtype=np.float32)
    raw_xy = model.forward_points(torch.from_numpy(reference), torch.from_numpy(theta)).numpy()
    raw = EventStream(np.rint(raw_xy[:, 0]), np.rint(raw_xy[:, 1]), times, np.ones(len(times), dtype=np.int8))
    compensated = compensate_events(raw, model, policy, reference_time=0.0)
    raw_variance = event_spatial_variance(raw)["radial_variance"]
    compensated_variance = event_spatial_variance(compensated, compensated=True)["radial_variance"]
    assert compensated_variance < raw_variance


def test_forward_inverse_disabled_prism_is_identity() -> None:
    points = torch.tensor([[0.0, 0.0], [31.5, 23.5], [63.0, 47.0]])
    global_model = GlobalCircularApproximation(intrinsics(), amplitude_px=0.0)
    paper_model = PaperSimplifiedPrismModel(intrinsics(), prism_angle_deg=0.0)
    assert torch.allclose(global_model.forward_points(points, 1.2), points)
    assert torch.allclose(paper_model.forward_points(points, 1.2), points, atol=1e-6)


def test_fps_samples_follow_same_physical_rotation_trajectory() -> None:
    policy = ConstantRotationPolicy(720.0)
    times_240 = np.array([0.0, 1.0 / 240.0, 2.0 / 240.0])
    times_480 = np.array([0.0, 1.0 / 480.0, 2.0 / 480.0])
    theta_240 = np.array([state.theta for state in policy.states_at(times_240)])
    theta_480 = np.array([state.theta for state in policy.states_at(times_480)])
    assert np.isclose(theta_240[1], theta_480[2])


def test_dsec_header_timestamps_and_resolution(tmp_path) -> None:
    rgb_dir = tmp_path / "rgb"
    rgb_dir.mkdir()
    for index in range(2):
        iio.imwrite(rgb_dir / f"{index:06d}.png", np.zeros((6, 8, 3), dtype=np.uint8))
    timestamps = tmp_path / "timestamps.csv"
    timestamps.write_text("index,timestamp\n0,1000000\n1,1050000\n", encoding="utf-8")
    sequence = DSECRGBSequence.from_paths(rgb_dir, timestamps, timestamp_unit="us")
    assert sequence.resolution == (8, 6)
    assert np.allclose(sequence.timestamps, [1.0, 1.05])
    assert np.array_equal(sequence.raw_timestamps, [1000000, 1050000])


def test_practical_rife_uses_arbitrary_direct_timestep_api(tmp_path, monkeypatch) -> None:
    repository = tmp_path / "Practical-RIFE"
    checkpoint = repository / "train_log"
    checkpoint.mkdir(parents=True)
    captured: dict[str, object] = {}

    class FakeModel:
        def __init__(self, arbitrary: bool = False) -> None:
            captured["arbitrary"] = arbitrary

        def load_model(self, path: str, rank: int) -> None:
            captured["checkpoint"] = (path, rank)

        def eval(self) -> None:
            captured["evaluated"] = True

        def inference(self, first, second, scale: float = 1.0, timestep: float = 0.5):
            captured.setdefault("timesteps", []).append(timestep)
            return first * (1.0 - timestep) + second * timestep

    model_module = types.ModuleType("model")
    rife_module = types.ModuleType("model.RIFE")
    rife_module.Model = FakeModel
    monkeypatch.setitem(sys.modules, "model", model_module)
    monkeypatch.setitem(sys.modules, "model.RIFE", rife_module)
    backend = PracticalRIFEBackend(repository, device="cpu")
    first = np.zeros((32, 32, 3), dtype=np.float32)
    second = np.ones((32, 32, 3), dtype=np.float32)
    output = backend.interpolate(first, second, np.array([0.25, 0.75]))
    assert captured["arbitrary"] is True
    assert captured["timesteps"] == [0.25, 0.75]
    assert np.allclose(output[:, 0, 0, 0], [0.25, 0.75])
    assert backend.metadata()["recursive_interpolation"] is False


def test_practical_rife_loads_official_train_log_layout(tmp_path, monkeypatch) -> None:
    repository = tmp_path / "Practical-RIFE"
    checkpoint = repository / "train_log"
    checkpoint.mkdir(parents=True)
    (checkpoint / "flownet.pkl").write_bytes(b"weights")
    captured: dict[str, object] = {}

    class FakeModel:
        def load_model(self, path: str, rank: int) -> None:
            captured["checkpoint"] = (path, rank)

        def eval(self) -> None:
            captured["evaluated"] = True

        def inference(self, first, second, timestep: float = 0.5, scale: float = 1.0):
            captured.setdefault("timesteps", []).append(timestep)
            return first * (1.0 - timestep) + second * timestep

    train_log_module = types.ModuleType("train_log")
    rife_module = types.ModuleType("train_log.RIFE_HDv3")
    rife_module.Model = FakeModel
    monkeypatch.delitem(sys.modules, "model.RIFE", raising=False)
    monkeypatch.setitem(sys.modules, "train_log", train_log_module)
    monkeypatch.setitem(sys.modules, "train_log.RIFE_HDv3", rife_module)
    backend = PracticalRIFEBackend(repository, device="cpu")
    first = np.zeros((32, 32, 3), dtype=np.float32)
    second = np.ones((32, 32, 3), dtype=np.float32)
    output = backend.interpolate(first, second, np.array([0.25, 0.75]))
    assert captured["checkpoint"] == (str(checkpoint), -1)
    assert captured["timesteps"] == [0.25, 0.75]
    assert np.allclose(output[:, 0, 0, 0], [0.25, 0.75])
    assert backend.metadata()["model_source"] == "train_log.RIFE_HDv3"
