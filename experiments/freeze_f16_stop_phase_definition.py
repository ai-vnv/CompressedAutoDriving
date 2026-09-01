#!/usr/bin/env python3
"""Freeze the F16 stop-control sub-phase definition BEFORE any statistic is computed.

The repository already owns an authoritative, mutually exclusive public phase taxonomy
(`duckie_pomdp.explain.development_protocol.public_phase`): combined_pedestrian_stop,
stop_required, stop_satisfied, pedestrian_relevant, lane_curve, nominal. Its thresholds
come from the frozen F12 config.

F16 subdivides only the stop-control part of that taxonomy. The subdivision introduces
**no new numeric threshold**: it uses the official label plus the *sign* of the public
velocity derivative. Every rule below is deterministic from public fields, uses no future
failure label, and was fixed before any error statistic was inspected.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))

from run_f15_cross_curriculum_recovery import (  # noqa: E402
    artifact_root,
    frozen_paths,
    load_config,
    phase_thresholds,
    provenance,
    write_json,
)

CONFIG = ROOT / "configs/f16_sequence_int8_recovery_v1.toml"


def main() -> None:
    config = load_config(CONFIG)
    root = artifact_root(config, CONFIG)
    target = root / "protocol/stop_phase_definition.json"
    if target.exists():
        raise RuntimeError(f"refusing to overwrite the frozen stop-phase definition: {target}")

    paths = frozen_paths(config, CONFIG)
    thresholds = phase_thresholds(paths["f12_config"])

    payload = {
        **provenance(config, CONFIG),
        "classification": "FROZEN",
        "written_before_any_statistic_computed": True,
        "authoritative_source": "duckie_pomdp.explain.development_protocol.public_phase",
        "inherited_thresholds": {
            "pedestrian_existence": thresholds.pedestrian_existence,
            "pedestrian_max_range_m": thresholds.pedestrian_max_range_m,
            "lane_curve_min_abs_curvature_inv_m": thresholds.lane_curve_min_abs_curvature_inv_m,
            "stop_satisfied_vicinity_m": thresholds.stop_satisfied_vicinity_m,
        },
        "new_numeric_thresholds_introduced": [],
        "derived_quantity": {
            "name": "delta_v",
            "definition": "actual_linear_velocity_mps[t] - actual_linear_velocity_mps[t-1]",
            "source_field": "actual_linear_velocity_mps (public 29D index 5)",
            "first_step_rule": "delta_v[0] := 0.0 (treated as non-negative)",
            "note": "only the SIGN of delta_v is used; no magnitude threshold is applied",
        },
        "sub_phases": {
            "NOMINAL": {
                "rule": "official phase not in {stop_required, combined_pedestrian_stop, stop_satisfied}",
                "meaning": "not inside a stop-control episode",
            },
            "APPROACH": {
                "rule": "official phase in {stop_required, combined_pedestrian_stop} AND delta_v >= 0",
                "meaning": "stop is required and the vehicle is not yet slowing",
            },
            "DECELERATION": {
                "rule": "official phase in {stop_required, combined_pedestrian_stop} AND delta_v < 0",
                "meaning": "stop is required and speed is decreasing",
            },
            "STOP_HOLD": {
                "rule": "official phase == stop_satisfied AND delta_v <= 0",
                "meaning": "stop condition satisfied and speed not increasing",
            },
            "RESTART": {
                "rule": "official phase == stop_satisfied AND delta_v > 0",
                "meaning": "stop condition satisfied and speed increasing again",
            },
        },
        "properties": {
            "mutually_exclusive": True,
            "deterministic_from_public_fields_only": True,
            "uses_future_or_failure_labels": False,
            "uses_privileged_simulator_truth": False,
            "chosen_after_seeing_error_distribution": False,
        },
        "populations": {
            "A_teacher_states": {
                "source": "artifacts/f15_cross_curriculum_recovery_v1/recovery/datasets/multicurriculum_public_states.npz",
                "question": "on the same, teacher-visited states, does INT8 already change the action mapping?",
                "caveat": "states come from successful Original trajectories and are not where INT8 fails",
            },
            "B_self_visited": {
                "source": "artifacts/f15_cross_curriculum_recovery_v1/telemetry/selection_{fp32,ptq,qat}_w64",
                "question": "which phases does each actor actually occupy while failure develops?",
                "comparison_rule": (
                    "compare phase occupancy and per-actor behavior only; do NOT align step t of one "
                    "trajectory against step t of another, because the trajectories have diverged"
                ),
            },
            "C_same_state_cross_policy": {
                "source": "exact public 29D rows recorded along PTQ/QAT closed-loop trajectories",
                "question": (
                    "on the very states that carried the INT8 actor toward failure, does the FP32 "
                    "mapping already differ from the INT8 mapping?"
                ),
                "discriminates": [
                    "policy mapping already differs on the same state",
                    "versus trajectory drifted off-distribution and failure amplified",
                ],
            },
        },
        "metrics_per_phase": [
            "v_mae_mps", "omega_mae_rad_s", "omega_pearson", "omega_spearman",
            "omega_sign_disagreement", "delta_v_mae", "delta_omega_mae",
            "saturation_disagreement", "state_count", "state_fraction",
            "mean_stop_line_distance_m", "mean_actual_velocity_mps",
        ],
        "claim_limit": (
            "This is diagnostic localization, not causal explanation. A phase-specific "
            "degradation identifies WHERE in the control sequence the mapping diverges; it does "
            "not identify quantization granularity, scale resolution, or any other mechanism. "
            "See docs/F16_QUANTIZATION_SCOPE_LIMITATION.md."
        ),
    }
    target.parent.mkdir(parents=True, exist_ok=True)
    write_json(target, payload)
    print(json.dumps({
        "frozen": str(target),
        "sub_phases": list(payload["sub_phases"]),
        "new_numeric_thresholds_introduced": payload["new_numeric_thresholds_introduced"],
        "inherited_thresholds": payload["inherited_thresholds"],
    }, indent=2))


if __name__ == "__main__":
    main()
