"""Core API for the AMI rotating-prism event simulator."""

from .events import CompiledCudaThresholdEventSimulator, CudaThresholdEventSimulator, EventStream, ThresholdEventSimulator, read_events_h5, write_events_h5
from .contrast_maximization import contrast_maximization
from .geometry import CameraIntrinsics, RotatingWedgePrism
from .pipeline import SimulationConfig, run_simulation
from .rotation import ConstantRotation, ImuGearRotation, ImuRotation, ScheduledRotation, motion_adaptive_hysteresis, motion_quantile_thresholds
from .types import ImuStream
from .dsec import read_dsec_events_h5, read_imu_csv
from .motion_net import CausalAnchorQualityResidualPolicy, CausalAnchorResidualPolicy, CausalFlowFusionGearPolicy, CausalFlowFusionQPolicy, CausalFlowFusionRPMPolicy, CausalGatedFlowQPolicy, CausalRPMPolicy, TinyCausalGearPolicy
from .event_quality import QUALITY_PROTOCOL_VERSION, TARGET_OBSERVABILITY_VERSION, active_pixel_ids, compute_reference_edge_mask, event_quality_metrics, event_quality_score, event_signal_fidelity_metrics, target_observability_metrics
from .control import CausalGearSelector, NeuralOpticalModulationController, OpticalModulationController
from .vfi import DebugLinearVFIBackend, PracticalRIFEBackend, resample_with_vfi
from .flow import CAUSAL_BASE_FEATURE_DIM, FLOW_FEATURE_NAMES, PRIVILEGED_ROI_FEATURE_DIM, PRIVILEGED_ROI_FEATURE_NAMES, PRIVILEGED_ROI_Q_FEATURE_DIM, estimate_dense_flow, estimate_flow_features, privileged_roi_sequence, summarize_flow, summarize_privileged_roi_flow, summarize_residual_flow
from .inverse_motion import CAMERA_DEFICIT_OCCUPANCY, CAMERA_FLOOR_RATE, CAMERA_HOLD_RATE, CAMERA_RATES, CAMERA_STORM_OCCUPANCY, CAMERA_UPSHIFT_SLACK, INVERSE_RPM, QUALITY_FLOOR_RPM, QUALITY_INVERSE_RPM, THREE_GEAR_RPM, deficit_camera_rate, excitation_deficit_rpm, f1_argmax_camera_rate, f1_argmax_rpm, f1_argmax_three_rpm, inverse_rpm, inverse_rpm_index, motion_thresholds, quality_inverse_rpm, spearman_rho, three_gear_rpm

__all__ = [
    "CameraIntrinsics",
    "QUALITY_PROTOCOL_VERSION",
    "TARGET_OBSERVABILITY_VERSION",
    "CausalGearSelector",
    "CompiledCudaThresholdEventSimulator",
    "CudaThresholdEventSimulator",
    "ConstantRotation",
    "ImuGearRotation",
    "ImuRotation",
    "ScheduledRotation",
    "motion_quantile_thresholds",
    "motion_adaptive_hysteresis",
    "ImuStream",
    "TinyCausalGearPolicy",
    "CausalAnchorResidualPolicy",
    "CausalAnchorQualityResidualPolicy",
    "CausalRPMPolicy",
    "CausalFlowFusionRPMPolicy",
    "CausalFlowFusionGearPolicy",
    "CausalFlowFusionQPolicy",
    "CausalGatedFlowQPolicy",
    "EventStream",
    "RotatingWedgePrism",
    "SimulationConfig",
    "ThresholdEventSimulator",
    "contrast_maximization",
    "read_events_h5",
    "read_dsec_events_h5",
    "read_imu_csv",
    "run_simulation",
    "write_events_h5",
    "active_pixel_ids",
    "compute_reference_edge_mask",
    "event_quality_metrics",
    "event_quality_score",
    "event_signal_fidelity_metrics",
    "target_observability_metrics",
    "OpticalModulationController",
    "NeuralOpticalModulationController",
    "DebugLinearVFIBackend",
    "PracticalRIFEBackend",
    "resample_with_vfi",
    "CAUSAL_BASE_FEATURE_DIM",
    "FLOW_FEATURE_NAMES",
    "PRIVILEGED_ROI_FEATURE_DIM",
    "PRIVILEGED_ROI_FEATURE_NAMES",
    "PRIVILEGED_ROI_Q_FEATURE_DIM",
    "estimate_dense_flow",
    "estimate_flow_features",
    "privileged_roi_sequence",
    "summarize_flow",
    "summarize_privileged_roi_flow",
    "summarize_residual_flow",
    "CAMERA_DEFICIT_OCCUPANCY",
    "CAMERA_FLOOR_RATE",
    "CAMERA_HOLD_RATE",
    "CAMERA_RATES",
    "CAMERA_STORM_OCCUPANCY",
    "CAMERA_UPSHIFT_SLACK",
    "INVERSE_RPM",
    "QUALITY_FLOOR_RPM",
    "QUALITY_INVERSE_RPM",
    "THREE_GEAR_RPM",
    "deficit_camera_rate",
    "excitation_deficit_rpm",
    "f1_argmax_camera_rate",
    "f1_argmax_rpm",
    "f1_argmax_three_rpm",
    "inverse_rpm",
    "inverse_rpm_index",
    "motion_thresholds",
    "quality_inverse_rpm",
    "spearman_rho",
    "three_gear_rpm",
]
