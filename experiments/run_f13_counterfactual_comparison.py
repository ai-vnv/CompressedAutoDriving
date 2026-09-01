#!/usr/bin/env python3
"""Compare frozen Original and deployed A7 semantic counterfactual responses."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib

from duckie_pomdp.control.ppo_protocol import load_ppo_curriculum_protocol
from duckie_pomdp.explain.compressed_policy_analysis import (
    actor_physical,
    classification_from_counterfactual,
    file_sha256,
    paired_effect_metrics,
    require_quantized_linear_graph,
    verify_hash,
)
from duckie_pomdp.explain.development_protocol import apply_semantic_intervention
from duckie_pomdp.optimization.actor_compression import extract_original_actor


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs/f13_explain_compressed_v1.toml"
PRIMARY = {
    "pedestrian_absent": "pedestrian_relevant",
    "stop_absent": "stop_required",
    "lane_centered": "lane_curve",
    "sham": "nominal",
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    print(json.dumps(run(args.config.resolve()), indent=2))


def run(config_path: Path) -> dict[str, Any]:
    config = load_config(config_path)
    root = resolve(config_path, config["artifacts"]["directory"])
    output = root / "counterfactual"
    if output.exists():
        raise FileExistsError("F13 counterfactual output already exists")
    replay = read_json(root / "integrity/replay_integrity.json")
    if replay["classification"] != "PASS":
        raise RuntimeError("F13 replay calibration did not pass")
    original_path = resolve(config_path, config["frozen"]["original"]["checkpoint"])
    a7_path = resolve(config_path, config["frozen"]["a7"]["checkpoint"])
    verify_hash(original_path, config["frozen"]["original"]["sha256"])
    verify_hash(a7_path, config["frozen"]["a7"]["sha256"])
    original, _, _ = extract_original_actor(
        original_path, expected_sha256=config["frozen"]["original"]["sha256"]
    )
    a7 = torch.jit.load(str(a7_path), map_location="cpu").eval()
    require_quantized_linear_graph(a7)
    policy_path = resolve(config_path, config["frozen"]["contract"]["policy_config"])
    protocol = load_ppo_curriculum_protocol(policy_path)
    trace_path = resolve(config_path, config["frozen"]["f11"]["r004_trace"])
    index_path = resolve(config_path, config["frozen"]["f11"]["r004_final_mean_attribution"])
    verify_hash(trace_path, config["frozen"]["f11"]["r004_trace_sha256"])
    verify_hash(index_path, config["frozen"]["f11"]["r004_final_mean_attribution_sha256"])
    with np.load(trace_path, allow_pickle=False) as archive:
        trace = {key: archive[key] for key in archive.files}
    with np.load(index_path, allow_pickle=False) as archive:
        sample_index = np.asarray(archive["sample_index"], dtype=np.int64)
    observations = np.asarray(trace["observation"][sample_index], dtype=np.float32)
    physical = np.asarray(trace["physical_observation"][sample_index], dtype=np.float32)
    stored_action = np.asarray(trace["physical_action"][sample_index], dtype=np.float32)
    phases = np.asarray(trace["public_phase"][sample_index], dtype="U40")
    seeds = np.asarray(trace["seed"][sample_index], dtype=np.int64)
    steps = np.asarray(trace["step"][sample_index], dtype=np.int32)
    interventions = tuple(str(value) for value in config["counterfactual"]["interventions"])
    r003 = read_json(resolve(config_path, config["frozen"]["f11"]["r003_validation"]))
    r003_config_path = resolve(config_path, config["frozen"]["f11"]["r003_config"])
    verify_hash(r003_config_path, config["frozen"]["f11"]["r003_config_sha256"])
    with r003_config_path.open("rb") as stream:
        r003_config = tomllib.load(stream)
    intervention_config = r003_config["r003"]
    if list(interventions) != list(r003["interventions"]):
        raise RuntimeError("F13 interventions differ from frozen R003")

    original_factual = actor_physical(original, observations)
    a7_factual = actor_physical(a7, observations)
    original_replay_error = float(np.max(np.abs(original_factual - stored_action)))
    if original_replay_error > float(replay["frozen_original_replay_tolerance"]):
        raise RuntimeError("Original factual replay exceeds frozen F13 tolerance")
    a7_repeat_error = float(np.max(np.abs(a7_factual - actor_physical(a7, observations))))
    if a7_repeat_error > float(replay["frozen_a7_repeat_tolerance"]):
        raise RuntimeError("A7 repeatability exceeds frozen F13 tolerance")

    counterfactual = np.empty((len(interventions), len(observations), 29), dtype=np.float32)
    changed_count = np.empty((len(interventions), len(observations)), dtype=np.int16)
    intended_fields: dict[str, list[str]] = {}
    for intervention_index, name in enumerate(interventions):
        registered: tuple[str, ...] | None = None
        for row_index, values in enumerate(physical):
            if name == "sham":
                changed, intended = observations[row_index].copy(), ()
            else:
                changed, intended = apply_semantic_intervention(
                    values,
                    name,
                    protocol,
                    lane_low_confidence_validity=float(intervention_config["lane_low_confidence_validity"]),
                    lane_low_confidence_min_lateral_std_m=float(intervention_config["lane_low_confidence_min_lateral_std_m"]),
                    lane_low_confidence_min_heading_std_rad=float(intervention_config["lane_low_confidence_min_heading_std_rad"]),
                    lane_low_confidence_min_curvature_std_inv_m=float(intervention_config["lane_low_confidence_min_curvature_std_inv_m"]),
                )
            if registered is None:
                registered = intended
            elif registered != intended:
                raise RuntimeError("registered intervention fields changed")
            counterfactual[intervention_index, row_index] = changed
            changed_count[intervention_index, row_index] = np.count_nonzero(changed != observations[row_index])
        intended_fields[name] = list(registered or ())
    maximum = float(config["counterfactual"]["maximum_normalized_absolute_value"])
    if not np.isfinite(counterfactual).all() or float(np.max(np.abs(counterfactual))) > maximum:
        raise RuntimeError("counterfactual vectors violate public normalized bounds")

    original_cf = np.stack([actor_physical(original, values) for values in counterfactual])
    a7_cf = np.stack([actor_physical(a7, values) for values in counterfactual])
    original_delta = original_cf - original_factual[None, :, :]
    a7_delta = a7_cf - a7_factual[None, :, :]
    drift = a7_delta - original_delta
    rows: list[dict[str, Any]] = []
    summary: dict[str, Any] = {}
    ranges = (0.4, 8.0)
    deadbands = (
        float(replay["frozen_original_replay_tolerance"]),
        float(replay["frozen_original_replay_tolerance"]),
    )
    for intervention_index, intervention in enumerate(interventions):
        summary[intervention] = {}
        for phase in config["attribution"]["phases"]:
            mask = phases == phase
            if not np.any(mask):
                continue
            phase_summary: dict[str, Any] = {"count": int(np.sum(mask))}
            for action_index, action_name in enumerate(("v_cmd_mps", "omega_cmd_rad_s")):
                metrics = paired_effect_metrics(
                    original_delta[intervention_index, mask, action_index],
                    a7_delta[intervention_index, mask, action_index],
                    action_range=ranges[action_index],
                    direction_deadband=deadbands[action_index],
                )
                phase_summary[action_name] = metrics
                rows.append({
                    "intervention": intervention,
                    "phase": phase,
                    "action": action_name,
                    **metrics,
                })
            summary[intervention][phase] = phase_summary

    original_tol = float(replay["frozen_original_replay_tolerance"])
    a7_tol = float(replay["frozen_a7_repeat_tolerance"])
    sham_index = interventions.index("sham")
    sham_original_max = float(np.max(np.abs(original_delta[sham_index])))
    sham_a7_max = float(np.max(np.abs(a7_delta[sham_index])))
    sham_pass = sham_original_max <= original_tol and sham_a7_max <= a7_tol
    mean_limit = float(config["counterfactual"]["maximum_normalized_mean_effect_drift"])
    p95_limit = float(config["counterfactual"]["maximum_normalized_p95_effect_drift"])
    direction_limit = float(config["counterfactual"]["minimum_direction_agreement"])
    primary_checks: dict[str, bool] = {}
    for intervention, phase in PRIMARY.items():
        if intervention == "sham":
            primary_checks["sham"] = sham_pass
            continue
        action = "omega_cmd_rad_s" if intervention == "lane_centered" else "v_cmd_mps"
        metrics = summary[intervention][phase][action]
        magnitude_pass = (
            metrics["normalized_mean_effect_drift"] <= mean_limit
            and metrics["normalized_p95_effect_drift"] <= p95_limit
        )
        if intervention in ("pedestrian_absent", "stop_absent"):
            direction_pass = (
                metrics["original_mean"] > 0.0
                and metrics["compressed_mean"] > 0.0
                and metrics["paired_direction_agreement"] >= direction_limit
            )
        else:
            original_abs = metrics["original_mean_absolute"]
            compressed_abs = metrics["compressed_mean_absolute"]
            normalized_abs_drift = abs(compressed_abs - original_abs) / 8.0
            metrics["normalized_mean_absolute_effect_drift"] = normalized_abs_drift
            direction_pass = normalized_abs_drift <= mean_limit
        primary_checks[f"{intervention}:{phase}:{action}"] = bool(magnitude_pass and direction_pass)

    representatives = representative_rows(physical, phases, seeds, steps, protocol.observation_order)
    output.mkdir(parents=True)
    np.savez_compressed(
        output / "original_vs_a7_counterfactual.npz",
        sample_index=sample_index,
        seed=seeds,
        step=steps,
        public_phase=phases,
        intervention_names=np.asarray(interventions, dtype="U40"),
        factual_observation=observations,
        counterfactual_observation=counterfactual,
        changed_feature_count=changed_count,
        original_factual_action=original_factual,
        a7_factual_action=a7_factual,
        original_counterfactual_action=original_cf,
        a7_counterfactual_action=a7_cf,
        original_delta=original_delta,
        a7_delta=a7_delta,
        functional_drift=drift,
    )
    write_csv(output / "counterfactual_summary.csv", rows)
    write_json(output / "representative_states.json", representatives)
    result = {
        "schema_version": 1,
        "classification": classification_from_counterfactual(primary_checks, sham_pass),
        "sample_count": int(len(observations)),
        "source_trace_sha256": file_sha256(trace_path),
        "original_sha256": file_sha256(original_path),
        "a7_sha256": file_sha256(a7_path),
        "original_replay_maximum_absolute_error": original_replay_error,
        "a7_repeat_maximum_absolute_error": a7_repeat_error,
        "sham": {
            "original_maximum_absolute_effect": sham_original_max,
            "a7_maximum_absolute_effect": sham_a7_max,
            "pass": sham_pass,
        },
        "primary_checks": primary_checks,
        "summary": summary,
        "intended_fields": intended_fields,
        "stored_privileged_truth": False,
        "r006_modified_or_recovered": False,
        "models_modified": False,
    }
    write_json(output / "counterfactual_metrics.json", result)
    return result


def representative_rows(
    physical: np.ndarray,
    phases: np.ndarray,
    seeds: np.ndarray,
    steps: np.ndarray,
    order: tuple[str, ...],
) -> dict[str, Any]:
    field_for_phase = {
        "pedestrian_relevant": "pedestrian_range_mean_m",
        "stop_required": "stop_line_distance_m",
        "lane_curve": "lane_curvature_mean_inv_m",
    }
    result: dict[str, Any] = {}
    for phase, field in field_for_phase.items():
        indexes = np.flatnonzero(phases == phase)
        values = physical[indexes, order.index(field)]
        target = float(np.median(values))
        local = int(np.argmin(np.abs(values - target)))
        row = int(indexes[local])
        result[phase] = {
            "row_index": row,
            "seed": int(seeds[row]),
            "step": int(steps[row]),
            "selection_rule": f"nearest public {field} to its within-phase median",
            "selection_value": float(values[local]),
            "selection_target": target,
        }
    return result


def load_config(path: Path) -> dict[str, Any]:
    with path.open("rb") as stream:
        return tomllib.load(stream)


def resolve(config_path: Path, value: str) -> Path:
    return (config_path.parent / value).resolve()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
