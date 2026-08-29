"""Command line entry point for the EIRM AMI simulator."""

from __future__ import annotations

import argparse
from pathlib import Path

from ami_sim.datasets import DSECRGBSequence
from ami_sim.events import (
    DebugThresholdEventSimulator,
    DVSVoltmeterBackend,
    DVSVoltmeterCudaFastBackend,
    DVSVoltmeterTorchBackend,
    V2EBackend,
)
from ami_sim.pipeline import SimulationOptions, _default_intrinsics, run_simulation
from ami_sim.prism import ConstantRotationPolicy, GlobalCircularApproximation, MotionMagnitudeAdaptiveRotation, PaperSimplifiedPrismModel
from ami_sim.vfi import DebugLinearVFIBackend, GIMMVFIBackend, PracticalRIFEBackend


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="EIRM: simulate a rotating wedge-prism AMI event camera from RGB.")
    parser.add_argument("--dataset", choices=["dsec"], default="dsec")
    parser.add_argument("--rgb-dir", required=True, type=Path)
    parser.add_argument("--timestamps", required=True, type=Path)
    parser.add_argument("--sequence-name", "--sequence", dest="sequence_name", default=None)
    parser.add_argument("--camera", choices=["left", "right"], default="left")
    parser.add_argument("--calibration", type=Path, default=None)
    parser.add_argument("--timestamp-unit", choices=["auto", "s", "ms", "us", "ns"], default="auto")
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--vfi", choices=["practical_rife", "gimm_vfi", "debug_linear"], default="practical_rife")
    parser.add_argument("--practical-rife-root", type=Path, default=None)
    parser.add_argument("--rife-checkpoint", type=Path, default=None)
    parser.add_argument("--rife-fp16", action="store_true", help="Run Practical-RIFE in fp16 Tensor Core mode on CUDA.")
    parser.add_argument("--rife-inference-batch-size", type=int, default=8, help="Number of arbitrary RIFE timesteps evaluated per GPU batch.")
    parser.add_argument("--gimm-adapter", default=None, help="package.module:function callable for direct GIMM timestep queries")
    parser.add_argument("--device", default=None)
    parser.add_argument("--interp-fps", type=float, choices=[240, 480, 960], default=480)
    parser.add_argument("--output-frame-count", type=int, default=None, help="Interpolate to exactly this many output frames.")
    parser.add_argument("--prism-model", choices=["global-circle", "global_circle", "paper-simplified", "paper_simplified"], default="paper_simplified")
    parser.add_argument("--prism-angle-deg", type=float, default=0.5)
    parser.add_argument("--global-amplitude-px", type=float, default=4.0)
    parser.add_argument("--rotation-policy", choices=["constant", "motion-adaptive"], default="constant")
    parser.add_argument("--rpm", type=float, default=720.0)
    parser.add_argument("--theta0", type=float, default=0.0)
    parser.add_argument("--padding-mode", choices=["border", "reflection", "zeros"], default="border")
    parser.add_argument(
        "--event-simulator",
        choices=["dvs_voltmeter", "dvs_voltmeter_cuda", "dvs_voltmeter_cuda_fast", "v2e", "debug_threshold"],
        default="dvs_voltmeter",
    )
    parser.add_argument("--event-command", default=None)
    parser.add_argument("--dvs-root", type=Path, default=Path(__file__).resolve().parents[1] / "DVS-Voltmeter")
    parser.add_argument("--dvs-camera-type", choices=["DVS346", "DVS240"], default="DVS346")
    parser.add_argument(
        "--dvs-precision",
        choices=["float32", "float64"],
        default="float64",
        help="DVS-Voltmeter arithmetic precision. float64 is the reproducible reference; float32 is an experimental alternative mode.",
    )
    parser.add_argument("--event-transfer-chunk-frames", type=int, default=16)
    parser.add_argument("--dvs-max-events-per-pixel", type=int, default=32)
    parser.add_argument("--event-output-time-unit", choices=["s", "ms", "us", "ns"], default="s")
    parser.add_argument("--contrast-threshold", type=float, default=0.2)
    parser.add_argument("--event-window-ms", type=float, default=33.0)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--compensate", action="store_true", default=True, help="Keep per-event inverse compensation (enabled by the pipeline).")
    parser.add_argument("--save-raw-events", action="store_true", default=True, help="Keep raw AMI events (enabled by the pipeline).")
    parser.add_argument("--prism-disabled", action="store_true", help="Set the prism deflection to zero for the vanilla consistency test.")
    parser.add_argument("--allow-default-intrinsics", action="store_true")
    parser.add_argument("--save-debug-video", action="store_true")
    parser.add_argument("--valid-orientation-samples", type=int, default=360)
    parser.add_argument("--warp-batch-size", type=int, default=8)
    parser.add_argument("--compensation-chunk-size", type=int, default=1_000_000)
    parser.add_argument("--cuda-fast-math", action="store_true", help="Enable TF32/cuDNN autotuning on CUDA.")
    parser.add_argument("--seed", type=int, default=0)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    sequence = DSECRGBSequence.from_paths(
        args.rgb_dir,
        args.timestamps,
        sequence_name=args.sequence_name,
        camera=args.camera,
        timestamp_unit=args.timestamp_unit,
        calibration_path=args.calibration,
        max_frames=args.max_frames,
    )
    if args.vfi == "practical_rife":
        if args.practical_rife_root is None:
            raise SystemExit("--practical-rife-root is required for the practical_rife backend.")
        vfi = PracticalRIFEBackend(
            args.practical_rife_root,
            args.rife_checkpoint,
            device=args.device,
            fp16=args.rife_fp16,
            inference_batch_size=args.rife_inference_batch_size,
        )
    elif args.vfi == "gimm_vfi":
        if not args.gimm_adapter:
            raise SystemExit("--gimm-adapter is required for the gimm_vfi backend.")
        vfi = GIMMVFIBackend(args.gimm_adapter)
    else:
        vfi = DebugLinearVFIBackend()
    if args.event_simulator == "dvs_voltmeter" and args.device and args.device.startswith("cuda") and args.event_command is None:
        # CUDA is an explicit performance request; use the in-process official
        # implementation unless the caller supplied an external command.
        event_backend = DVSVoltmeterTorchBackend(
            args.dvs_root,
            camera_type=args.dvs_camera_type,
            device=args.device,
            transfer_chunk_frames=args.event_transfer_chunk_frames,
            seed=args.seed,
            precision=args.dvs_precision,
        )
    elif args.event_simulator == "dvs_voltmeter":
        event_backend = DVSVoltmeterBackend(args.event_command, output_time_unit=args.event_output_time_unit)
    elif args.event_simulator == "dvs_voltmeter_cuda":
        event_backend = DVSVoltmeterTorchBackend(
            args.dvs_root,
            camera_type=args.dvs_camera_type,
            device=args.device,
            transfer_chunk_frames=args.event_transfer_chunk_frames,
            seed=args.seed,
            precision=args.dvs_precision,
        )
    elif args.event_simulator == "dvs_voltmeter_cuda_fast":
        event_backend = DVSVoltmeterCudaFastBackend(
            args.dvs_root,
            camera_type=args.dvs_camera_type,
            device=args.device,
            transfer_chunk_frames=args.event_transfer_chunk_frames,
            seed=args.seed,
            precision=args.dvs_precision,
            max_events_per_pixel=args.dvs_max_events_per_pixel,
        )
    elif args.event_simulator == "v2e":
        event_backend = V2EBackend(args.event_command, output_time_unit=args.event_output_time_unit)
    else:
        event_backend = DebugThresholdEventSimulator(args.contrast_threshold)
    height, width = sequence.resolution[1], sequence.resolution[0]
    intrinsics = sequence.calibration.intrinsics if sequence.calibration else _default_intrinsics(width, height)
    prism_model = args.prism_model.replace("_", "-")
    if prism_model == "global-circle":
        model = GlobalCircularApproximation(intrinsics, 0.0 if args.prism_disabled else args.global_amplitude_px)
    else:
        model = PaperSimplifiedPrismModel(intrinsics, 0.0 if args.prism_disabled else args.prism_angle_deg)
    if args.rotation_policy == "constant":
        policy = ConstantRotationPolicy(args.rpm)
    else:
        policy = MotionMagnitudeAdaptiveRotation()
    options = SimulationOptions(
        interp_fps=args.interp_fps,
        output_frame_count=args.output_frame_count,
        prism_model="global_circle" if prism_model == "global-circle" else "paper_simplified",
        prism_angle_deg=0.0 if args.prism_disabled else args.prism_angle_deg,
        rotation_rpm=args.rpm,
        theta0=args.theta0,
        padding_mode=args.padding_mode,
        event_window_ms=args.event_window_ms,
        valid_orientation_samples=args.valid_orientation_samples,
        save_debug_video=args.save_debug_video,
        allow_default_intrinsics=args.allow_default_intrinsics,
        random_seed=args.seed,
        device=args.device,
        warp_batch_size=args.warp_batch_size,
        compensation_chunk_size=args.compensation_chunk_size,
        cuda_fast_math=args.cuda_fast_math,
    )
    run_simulation(
        sequence,
        vfi,
        event_backend,
        model,
        policy,
        args.output or Path("outputs") / sequence.sequence_name,
        options=options,
        config=vars(args),
    )
    print(f"EIRM simulation written to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
