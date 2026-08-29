from pathlib import Path

import numpy as np

from ami_sim.events import DVSVoltmeterTorchBackend

try:
    import torch
    from ami_sim.events import DVSVoltmeterCudaFastBackend
except ImportError:  # pragma: no cover - torch is a project dependency
    torch = None
    DVSVoltmeterCudaFastBackend = None
import pytest


DVS_ROOT = Path(__file__).resolve().parents[1] / "DVS-Voltmeter"
DVS_AVAILABLE = (DVS_ROOT / "src" / "simulator.py").is_file()


@pytest.mark.skipif(not DVS_AVAILABLE, reason="external DVS-Voltmeter checkout is unavailable")
def test_inprocess_dvs_backend_returns_sorted_canonical_events() -> None:
    frames = np.zeros((5, 16, 16), dtype=np.float32)
    frames[1:, 4:12, 4:12] = np.linspace(0.1, 1.0, 4, dtype=np.float32)[:, None, None]
    timestamps = np.linspace(0.0, 0.04, len(frames), dtype=np.float64)
    backend = DVSVoltmeterTorchBackend(
        DVS_ROOT,
        device="cpu",
        transfer_chunk_frames=2,
        seed=0,
    )
    events = backend.simulate(frames, timestamps)
    assert backend.supports_torch is True
    assert len(events) > 0
    assert np.all(np.diff(events.t) >= 0)
    assert np.all(np.isin(events.p, (-1, 1)))


@pytest.mark.skipif(not DVS_AVAILABLE, reason="external DVS-Voltmeter checkout is unavailable")
def test_inprocess_dvs_precision_modes_are_finite_and_metadata_is_explicit() -> None:
    frames = np.zeros((6, 16, 16), dtype=np.float32)
    frames[1:] = np.linspace(0.1, 1.0, 5, dtype=np.float32)[:, None, None]
    timestamps = np.linspace(0.0, 0.05, len(frames), dtype=np.float64)
    root = DVS_ROOT

    results = {}
    for precision in ("float64", "float32"):
        backend = DVSVoltmeterTorchBackend(
            root,
            device="cpu",
            transfer_chunk_frames=2,
            seed=0,
            precision=precision,
        )
        events = backend.simulate(frames, timestamps)
        results[precision] = events
        assert backend.metadata()["precision"] == precision
        assert len(events) > 0
        assert np.all(np.isfinite(events.t))
        assert np.all(np.diff(events.t) >= 0)
        assert np.all(np.isin(events.p, (-1, 1)))

    # Precision changes floating-point rounding, but should not change the
    # scale of the stochastic output for this deterministic smoke sequence.
    relative_count_error = abs(len(results["float64"]) - len(results["float32"])) / len(results["float64"])
    assert relative_count_error < 0.01


@pytest.mark.skipif(
    not DVS_AVAILABLE or torch is None or not torch.cuda.is_available(),
    reason="external DVS-Voltmeter checkout or CUDA is unavailable",
)
def test_custom_cuda_backend_matches_reference_event_scale() -> None:
    frames = torch.rand(
        (6, 32, 32),
        device="cuda",
        generator=torch.Generator(device="cuda").manual_seed(11),
    )
    timestamps = np.linspace(0.0, 5.0 / 480.0, len(frames), dtype=np.float64)
    root = DVS_ROOT
    reference = DVSVoltmeterTorchBackend(root, device="cuda", transfer_chunk_frames=6, seed=3).simulate(frames, timestamps)
    fast = DVSVoltmeterCudaFastBackend(
        root,
        device="cuda",
        transfer_chunk_frames=6,
        seed=3,
        max_events_per_pixel=32,
    ).simulate(frames, timestamps)
    assert len(fast) > 0
    assert np.all(np.diff(fast.t) >= 0)
    assert np.all(np.isin(fast.p, (-1, 1)))
    assert abs(len(fast) - len(reference)) / len(reference) < 0.02
    assert abs(float((fast.p > 0).mean()) - float((reference.p > 0).mean())) < 0.02
    assert np.allclose(fast.t * 1e6, np.round(fast.t * 1e6), atol=1e-5)
