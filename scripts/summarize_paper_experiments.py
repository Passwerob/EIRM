#!/usr/bin/env python3
"""Build one machine-readable summary for the current paper experiments."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("results/ami_next/paper_experiment_results.json"))
    args = parser.parse_args()
    main_f1 = load(Path("results/ami_next/main_f1_test_current.json"))
    main_teo = load(Path("results/ami_next/main_teo_test_current.json"))
    downstream = load(Path("results/ami_next/downstream_dvs_v2e_f1_ours_test.json"))
    controller = load(Path("results/ami_next/controller_ablation_test_current.json"))
    cm = load(Path("results/ami_next/cm_metrics_summary.json"))
    compensation = load(Path("results/ami_next/compensation_ablation_camera_only.json"))
    dl3dv_compensation = load(Path("results/ami_next/dl3dv_compensation_ablation.json"))
    cm_edge = load(Path("results/ami_next/cm_edge_reconstruction_all_regimes.json"))
    adaptive_cm = load(Path("results/ami_next/controlled_adaptive_cm_iwe.json"))
    sweep = load(Path("results/ami_next/motion_excitation_sweep/analysis_summary.json"))
    constructed = load(Path("results/ami_next/constructed_eval_camera_passby_complete.json"))
    report = {
        "protocol": "ami_paper_experiment_summary_v1",
        "main_downstream": {
            "source": "results/ami_next/downstream_dvs_v2e_f1_ours_test.json",
            "aggregate": downstream["aggregate"],
            "decision": downstream.get("decision", {}),
        },
        "target_observability": {
            "source": "results/ami_next/main_teo_test_current.json",
            "aggregate": main_teo["aggregate"],
        },
        "motion_observability": {
            "source": "results/ami_next/motion_excitation_sweep/analysis_summary.json",
            "summary": sweep,
        },
        "controller_ablation": {
            "source": "results/ami_next/controller_ablation_test_current.json",
            "aggregate": controller["aggregate"],
            "interpretation": "The causal policy is a small positive aggregate gain over fixed 180, but below the predeclared +0.02 co-motion gate.",
        },
        "cm_iwe": {
            "source": "results/ami_next/cm_metrics_summary.json",
            "aggregate": cm["aggregate"],
            "role": "downstream representation diagnostic; not an event-generator quality metric by itself",
        },
        "cm_iwe_edge_reconstruction": {
            "source": "results/ami_next/cm_edge_reconstruction_all_regimes.json",
            "aggregate": cm_edge["aggregate"],
            "per_regime": cm_edge["per_regime"],
            "interpretation": "Fixed internal AMI speeds and external DVS/v2e are compared using the same CM-IWE grid and RGB edge reference.",
        },
        "adaptive_cm_iwe": {
            "source": "results/ami_next/controlled_adaptive_cm_iwe.json",
            "aggregate": adaptive_cm["aggregate"],
            "per_regime": adaptive_cm["per_regime"],
            "interpretation": "True causal EIRM event streams on all four held-out controlled regimes, passed through the same CM-IWE velocity grid and RGB edge scorer.",
        },
        "compensation_ablation": {
            "source": "results/ami_next/compensation_ablation_camera_only.json",
            "summary": compensation["summary"],
            "scope": "EVEYES Scene_005 camera-only pilot, 120 windows",
            "interpretation": "Per-event compensation improves alignment over interval-level and no compensation on the same generated event stream.",
        },
        "dl3dv_compensation_ablation": {
            "source": "results/ami_next/dl3dv_compensation_ablation.json",
            "summary": dl3dv_compensation["summary"],
            "scope": "five DL3DV scenes, 24 frames, 160-pixel evaluation width",
            "interpretation": "Per-event timestamp compensation improves edge alignment over interval-level and no compensation on identical AMI event streams.",
        },
        "complete_mechanism_scene": {
            "source": "results/ami_next/constructed_eval_camera_passby_complete.json",
            "verdict": constructed["verdict"],
            "two_sided_ok": constructed["two_sided_ok"],
            "scope": "mechanism figure; storm segment has five windows",
        },
        "controller_status": {
            "frozen_claim": "active optical modulation improves controlled event observability",
            "not_yet_supported": "a large causal-controller gain over fixed 180 across all regimes",
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "sections": list(report)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
