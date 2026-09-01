#!/usr/bin/env python3
"""Run the preregistered once-only R006 semantic action intervention study."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib

from duckie_pomdp.control.ppo import PPOAgent
from duckie_pomdp.explain.confirmatory_intervention import (
    intervention_gate_results,
    paired_effect_summary,
)
from duckie_pomdp.explain.development_protocol import apply_semantic_intervention
from duckie_pomdp.explain.ppo_integrated_gradients import PPOActionLimits, target_values
try:
    from run_f11_r004_once import _load as load_r004
except ModuleNotFoundError:  # imported as experiments.run_f11_r006_once in tests
    from experiments.run_f11_r004_once import _load as load_r004


ROOT = Path(__file__).resolve().parents[1]
FORBIDDEN_SCHEMA_TOKENS = (
    "privileged",
    "evaluation_gt",
    "world_pose",
    "true_",
    "gt_",
    "bbox",
    "iou",
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs" / "f11_ppo_explanation_r006_v1.toml",
    )
    parser.add_argument("--mode", choices=("preflight", "once"), required=True)
    args = parser.parse_args()
    if args.mode == "preflight":
        preflight(args.config.resolve())
    else:
        run_once(args.config.resolve())


def preflight(config_path: Path) -> None:
    loaded = _load(config_path, verify_locked_data=False)
    output = _outputs(config_path, loaded[0])
    if output["directory"].exists():
        raise FileExistsError("R006 output already exists; once-only run is closed")
    _, _, _, checkpoint, protocol, _ = loaded[1:]
    agent, payload = PPOAgent.load(checkpoint, device=_device())
    probe = torch.zeros((2, len(protocol.observation_order)), device=_device())
    with torch.no_grad():
        actor = agent.model.actor(probe)
        critic = agent.model.value(probe)
    if actor.shape != (2, 2) or critic.shape != (2,):
        raise RuntimeError("frozen actor/critic preflight failed")
    print(
        json.dumps(
            {
                "classification": "PASS",
                "mode": "preflight",
                "r004_trace_reused_without_render": True,
                "r006_output_exists": False,
                "checkpoint_sha256": sha256(checkpoint),
                "checkpoint_global_step": int(payload["global_step"]),
                "observation_dimension": len(protocol.observation_order),
                "device": _device(),
            },
            indent=2,
        )
    )


def run_once(config_path: Path) -> None:
    # Load only metadata/code inputs before the claim. The locked trace and the
    # R004 sample-index artifact are neither opened nor hashed at this point.
    config, r004, r001, source_config, checkpoint, protocol, r003 = _load(
        config_path, verify_locked_data=False
    )
    output = _outputs(config_path, config)
    if output["directory"].exists():
        raise FileExistsError("refusing to reopen once-only R006")
    output["directory"].mkdir(parents=True)
    _write_json(
        output["launch_claim"],
        {
            "schema_version": 1,
            "run_id": "R006",
            "once_only": True,
            "launched_at_utc": datetime.now(timezone.utc).isoformat(),
            "config_sha256": sha256(config_path),
            "source_trace_sha256": config["frozen"]["r004_trace_sha256"],
            "locked_seeds": list(config["data"]["locked_evaluation_seeds"]),
            "trace_access_begins_after_this_claim": True,
            "simulator_rerender_permitted": False,
            "rerun_permitted": False,
        },
    )
    try:
        # The claim now exists. Only after it is durable may locked data be
        # opened, including the byte reads performed by SHA256 verification.
        config, r004, r001, source_config, checkpoint, protocol, r003 = _load(
            config_path, verify_locked_data=True
        )
        _execute(
            config_path,
            config,
            r004,
            r001,
            source_config,
            checkpoint,
            protocol,
            r003,
            output,
        )
    except Exception as error:
        _write_json(
            output["failure_marker"],
            {
                "classification": "FAILED",
                "error_type": type(error).__name__,
                "error": str(error),
                "rerun_permitted": False,
            },
        )
        raise


def _execute(
    config_path: Path,
    config: dict[str, Any],
    r004: dict[str, Any],
    r001: dict[str, Any],
    source_config: Path,
    checkpoint: Path,
    protocol: Any,
    r003: dict[str, Any],
    output: dict[str, Path],
) -> None:
    trace_path = _resolve(config_path, str(config["frozen"]["r004_trace"]))
    attribution_path = _resolve(
        config_path, str(config["frozen"]["r004_final_attribution"])
    )
    with np.load(trace_path, allow_pickle=False) as archive:
        trace = {name: archive[name] for name in archive.files}
    with np.load(attribution_path, allow_pickle=False) as archive:
        sample_index = np.asarray(archive["sample_index"], dtype=np.int64)
        attribution_seed = np.asarray(archive["seed"], dtype=np.int64)
        attribution_step = np.asarray(archive["step"], dtype=np.int32)
        attribution_phase = np.asarray(archive["public_phase"], dtype="U40")
    _validate_trace(config, trace, sample_index, attribution_seed, attribution_step, attribution_phase)

    observations = np.asarray(trace["observation"][sample_index], dtype=np.float32)
    physical = np.asarray(trace["physical_observation"][sample_index], dtype=np.float32)
    seeds = np.asarray(trace["seed"][sample_index], dtype=np.int64)
    steps = np.asarray(trace["step"][sample_index], dtype=np.int32)
    phases = np.asarray(trace["public_phase"][sample_index], dtype="U40")
    interventions = tuple(str(value) for value in config["intervention"]["names"])
    if interventions != tuple(str(value) for value in r003["r003"]["interventions"]):
        raise RuntimeError("R006 intervention set differs from frozen R003")

    counterfactual = np.empty((len(interventions), len(observations), 29), dtype=np.float32)
    changed_count = np.empty((len(interventions), len(observations)), dtype=np.int16)
    intended_fields: dict[str, list[str]] = {}
    for intervention_index, name in enumerate(interventions):
        registered: tuple[str, ...] | None = None
        for row_index, values in enumerate(physical):
            if name == "sham":
                changed = observations[row_index].copy()
                intended: tuple[str, ...] = ()
            else:
                changed, intended = apply_semantic_intervention(
                    values,
                    name,
                    protocol,
                    lane_low_confidence_validity=float(
                        r003["r003"]["lane_low_confidence_validity"]
                    ),
                    lane_low_confidence_min_lateral_std_m=float(
                        r003["r003"]["lane_low_confidence_min_lateral_std_m"]
                    ),
                    lane_low_confidence_min_heading_std_rad=float(
                        r003["r003"]["lane_low_confidence_min_heading_std_rad"]
                    ),
                    lane_low_confidence_min_curvature_std_inv_m=float(
                        r003["r003"]["lane_low_confidence_min_curvature_std_inv_m"]
                    ),
                )
            if registered is None:
                registered = intended
            elif registered != intended:
                raise RuntimeError("registered intervention fields changed across rows")
            counterfactual[intervention_index, row_index] = changed
            changed_count[intervention_index, row_index] = int(
                np.count_nonzero(~np.isclose(changed, observations[row_index]))
            )
        intended_fields[name] = list(registered or ())
    maximum = float(config["intervention"]["maximum_normalized_absolute_value"])
    if not np.isfinite(counterfactual).all() or np.max(np.abs(counterfactual)) > maximum:
        raise RuntimeError("counterfactual observation is non-finite or out of bounds")

    agent, payload = PPOAgent.load(checkpoint, device=_device())
    checkpoint_before = sha256(checkpoint)
    model_before = _model_hash(agent)
    limits = PPOActionLimits(
        float(r001["frozen_policy"]["maximum_linear_velocity_mps"]),
        float(r001["frozen_policy"]["maximum_angular_velocity_rad_s"]),
    )
    factual_v, factual_omega, factual_value = _actor_values(agent, observations, limits)
    stored_action = np.asarray(trace["physical_action"][sample_index], dtype=np.float32)
    replay_error = float(
        max(
            np.max(np.abs(factual_v - stored_action[:, 0])),
            np.max(np.abs(factual_omega - stored_action[:, 1])),
        )
    )
    if replay_error > float(config["gate"]["factual_action_replay_tolerance"]):
        raise RuntimeError("factual action replay differs from frozen R004 trace")

    cf_v = np.empty((len(interventions), len(observations)), dtype=np.float32)
    cf_omega = np.empty_like(cf_v)
    cf_value = np.empty_like(cf_v)
    for index in range(len(interventions)):
        cf_v[index], cf_omega[index], cf_value[index] = _actor_values(
            agent, counterfactual[index], limits
        )
    if sha256(checkpoint) != checkpoint_before or _model_hash(agent) != model_before:
        raise RuntimeError("frozen PPO changed during R006")
    delta_v = cf_v - factual_v[None, :]
    delta_omega = cf_omega - factual_omega[None, :]
    delta_value = cf_value - factual_value[None, :]

    np.savez_compressed(
        output["interventions"],
        sample_index=sample_index,
        seed=seeds,
        step=steps,
        public_phase=phases,
        factual_observation=observations,
        counterfactual_observation=counterfactual,
        intervention_names=np.asarray(interventions, dtype="U40"),
        feature_names=np.asarray(protocol.observation_order, dtype="U64"),
        changed_feature_count=changed_count,
        factual_v_cmd_mps=factual_v,
        factual_omega_cmd_rad_s=factual_omega,
        factual_critic_value=factual_value,
        counterfactual_v_cmd_mps=cf_v,
        counterfactual_omega_cmd_rad_s=cf_omega,
        counterfactual_critic_value=cf_value,
        delta_v_cmd_mps=delta_v,
        delta_omega_cmd_rad_s=delta_omega,
        delta_critic_value=delta_value,
    )
    paired_rows = _paired_rows(
        interventions, seeds, steps, phases, factual_v, factual_omega, factual_value,
        cf_v, cf_omega, cf_value, delta_v, delta_omega, delta_value
    )
    _write_csv(output["paired_effects"], paired_rows)
    summary_rows = _summary_rows(config, interventions, seeds, phases, delta_v, delta_omega, delta_value)
    _write_csv(output["summary"], summary_rows)
    bearing_rows = _bearing_rows(
        config, protocol, interventions, physical, seeds, phases, delta_omega
    )
    _write_csv(output["bearing_summary"], bearing_rows)

    name_index = {name: interventions.index(name) for name in interventions}
    pedestrian_mask = phases == "pedestrian_relevant"
    pedestrian_irrelevant = phases != "pedestrian_relevant"
    stop_mask = phases == "stop_required"
    stop_control = phases == str(config["gate"]["stop_negative_control_phase"])
    lane_mask = phases == "lane_curve"
    lane_control = phases == str(config["gate"]["lane_negative_control_phase"])
    sham_index = name_index["sham"]
    sham_max = float(
        max(
            np.max(np.abs(delta_v[sham_index])),
            np.max(np.abs(delta_omega[sham_index])),
            np.max(np.abs(delta_value[sham_index])),
        )
    )
    criteria, diagnostics = intervention_gate_results(
        pedestrian_delta_v=delta_v[name_index["pedestrian_absent"], pedestrian_mask],
        pedestrian_seeds=seeds[pedestrian_mask],
        pedestrian_irrelevant_delta_v=delta_v[
            name_index["pedestrian_absent"], pedestrian_irrelevant
        ],
        stop_delta_v=delta_v[name_index["stop_absent"], stop_mask],
        stop_seeds=seeds[stop_mask],
        stop_control_delta_v=delta_v[name_index["stop_absent"], stop_control],
        lane_delta_omega=delta_omega[name_index["lane_centered"], lane_mask],
        lane_seeds=seeds[lane_mask],
        lane_control_delta_omega=delta_omega[
            name_index["lane_centered"], lane_control
        ],
        sham_maximum_absolute_effect=sham_max,
        gate=config["gate"],
        bootstrap=config["bootstrap"],
    )
    structural = {
        "r004_classification_pass": True,
        "r003_operator_gate_pass": True,
        "same_frozen_locked_trace_reused": True,
        "simulator_rerendered": False,
        "all_locked_seeds_exact": set(int(value) for value in np.unique(seeds))
        == set(int(value) for value in config["data"]["locked_evaluation_seeds"]),
        "all_required_phases_present": set(str(value) for value in np.unique(phases))
        == set(str(value) for value in config["phases"]["required"]),
        "factual_action_replay_exact": replay_error
        <= float(config["gate"]["factual_action_replay_tolerance"]),
        "all_counterfactuals_finite_and_bounded": True,
        "only_registered_fields_changed": True,
        "no_privileged_truth_stored": True,
        "model_immutable": True,
    }
    if not all(structural.values()):
        classification = "FAILED"
    elif all(criteria.values()):
        classification = "PASS"
    else:
        classification = "LIMITED"
    metrics = {
        "schema_version": 1,
        "run_id": "R006",
        "classification": classification,
        "allowed_wording": "confirmatory counterfactual policy dependence",
        "not_a_real_world_causal_claim": True,
        "sample_count": int(len(observations)),
        "seeds": [int(value) for value in np.unique(seeds)],
        "phase_counts": {
            str(phase): int(np.sum(phases == phase)) for phase in np.unique(phases)
        },
        "interventions": list(interventions),
        "intended_fields": intended_fields,
        "criteria": criteria,
        "structural_integrity": structural,
        "confirmatory_diagnostics": diagnostics,
        "factual_action_replay_max_abs_error": replay_error,
        "checkpoint_global_step": int(payload["global_step"]),
        "checkpoint_sha256": checkpoint_before,
        "model_state_sha256_before_after": model_before,
        "config_sha256": sha256(config_path),
        "r004_trace_sha256": sha256(trace_path),
        "intervention_artifact_sha256": sha256(output["interventions"]),
        "paired_effects_sha256": sha256(output["paired_effects"]),
        "summary_sha256": sha256(output["summary"]),
        "bearing_summary_sha256": sha256(output["bearing_summary"]),
        "r007_started": False,
    }
    _write_json(output["metrics"], metrics)
    manifest = {
        "schema_version": 1,
        "classification": classification,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "config_sha256": sha256(config_path),
        "protocol_sha256": str(config["frozen"]["r006_protocol_sha256"]),
        "checkpoint_sha256": checkpoint_before,
        "source_config_sha256": sha256(source_config),
        "r004_trace_sha256": sha256(trace_path),
        "files": {
            name: {"path": str(path.relative_to(ROOT)), "sha256": sha256(path)}
            for name, path in output.items()
            if name not in {"directory", "failure_marker", "manifest"} and path.exists()
        },
        "r007_started": False,
    }
    _write_json(output["manifest"], manifest)
    print(json.dumps(metrics, indent=2))


def _load(
    config_path: Path, *, verify_locked_data: bool
) -> tuple[Any, ...]:
    with config_path.open("rb") as stream:
        config = tomllib.load(stream)
    metadata_inputs = (
        "r004_config", "r004_protocol", "r004_report", "r004_metrics",
        "r004_manifest", "r004_trace_manifest", "r004_launch_claim", "r003_config",
        "r003_metrics", "r003_operator_source", "r004_runner", "r006_protocol",
        "r006_runner", "r006_statistics_source",
    )
    locked_data_inputs = ("r004_trace", "r004_final_attribution")
    for name in metadata_inputs + (locked_data_inputs if verify_locked_data else ()):
        path = _resolve(config_path, str(config["frozen"][name]))
        if sha256(path) != str(config["frozen"][f"{name}_sha256"]):
            raise ValueError(f"frozen {name} hash mismatch")
    r004_config_path = _resolve(config_path, str(config["frozen"]["r004_config"]))
    r004, r001, source_config, checkpoint, protocol, groups = load_r004(r004_config_path)
    r004_metrics = json.loads(
        _resolve(config_path, str(config["frozen"]["r004_metrics"])).read_text()
    )
    r004_manifest = json.loads(
        _resolve(config_path, str(config["frozen"]["r004_manifest"])).read_text()
    )
    r003_metrics = json.loads(
        _resolve(config_path, str(config["frozen"]["r003_metrics"])).read_text()
    )
    if r004_metrics["classification"] != "PASS" or r004_manifest["classification"] != "PASS":
        raise ValueError("R004 must remain PASS")
    if r003_metrics["classification"] != "PASS":
        raise ValueError("R003 operator gate must remain PASS")
    if list(r004_metrics["seeds"]) != list(config["data"]["locked_evaluation_seeds"]):
        raise ValueError("R006 locked seeds differ from R004")
    r003_config_path = _resolve(config_path, str(config["frozen"]["r003_config"]))
    with r003_config_path.open("rb") as stream:
        r003 = tomllib.load(stream)
    return config, r004, r001, source_config, checkpoint, protocol, r003


def _validate_trace(
    config: dict[str, Any],
    trace: dict[str, np.ndarray],
    sample_index: np.ndarray,
    attribution_seed: np.ndarray,
    attribution_step: np.ndarray,
    attribution_phase: np.ndarray,
) -> None:
    lowered = tuple(str(name).lower() for name in trace)
    if any(token in name for name in lowered for token in FORBIDDEN_SCHEMA_TOKENS):
        raise RuntimeError("R004 trace contains a forbidden evaluation schema")
    if trace["observation"].shape != (17600, 29):
        raise RuntimeError("R004 public trace shape changed")
    if len(sample_index) != int(config["data"]["expected_sample_count"]):
        raise RuntimeError("R006 sample count differs from preregistration")
    if not np.array_equal(trace["seed"][sample_index], attribution_seed):
        raise RuntimeError("R004 attribution seed alignment failed")
    if not np.array_equal(trace["step"][sample_index], attribution_step):
        raise RuntimeError("R004 attribution step alignment failed")
    if not np.array_equal(trace["public_phase"][sample_index], attribution_phase):
        raise RuntimeError("R004 attribution phase alignment failed")


def _actor_values(agent: PPOAgent, observations: np.ndarray, limits: PPOActionLimits) -> tuple[np.ndarray, ...]:
    tensor = torch.as_tensor(observations, dtype=torch.float32, device=_device())
    with torch.no_grad():
        v = target_values(agent.model, tensor, target="v_cmd_mps", action_limits=limits)
        omega = target_values(agent.model, tensor, target="omega_cmd_rad_s", action_limits=limits)
        value = agent.model.value(tensor)
    return (
        v.cpu().numpy().astype(np.float32),
        omega.cpu().numpy().astype(np.float32),
        value.cpu().numpy().astype(np.float32),
    )


def _summary_rows(config: dict[str, Any], interventions: tuple[str, ...], seeds: np.ndarray, phases: np.ndarray, delta_v: np.ndarray, delta_omega: np.ndarray, delta_value: np.ndarray) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for i, name in enumerate(interventions):
        for phase in np.unique(phases):
            mask = phases == phase
            row: dict[str, object] = {"intervention": name, "public_phase": str(phase)}
            for label, values, offset in (
                ("delta_v_cmd_mps", delta_v[i, mask], 11),
                ("delta_omega_cmd_rad_s", delta_omega[i, mask], 22),
                ("delta_critic_value", delta_value[i, mask], 33),
            ):
                summary = paired_effect_summary(
                    values,
                    seeds[mask],
                    bootstrap_replicates=int(config["bootstrap"]["replicates"]),
                    bootstrap_seed=int(config["bootstrap"]["seed"]),
                    confidence_level=float(config["bootstrap"]["confidence_level"]),
                    direction_tolerance=float(config["gate"]["direction_tolerance"]),
                )
                for key, value in summary.items():
                    row[f"{label}_{key}"] = value
            rows.append(row)
    return rows


def _paired_rows(interventions: tuple[str, ...], seeds: np.ndarray, steps: np.ndarray, phases: np.ndarray, factual_v: np.ndarray, factual_omega: np.ndarray, factual_value: np.ndarray, cf_v: np.ndarray, cf_omega: np.ndarray, cf_value: np.ndarray, delta_v: np.ndarray, delta_omega: np.ndarray, delta_value: np.ndarray) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for i, name in enumerate(interventions):
        for row in range(len(seeds)):
            rows.append({
                "seed": int(seeds[row]), "step": int(steps[row]),
                "public_phase": str(phases[row]), "intervention": name,
                "factual_v_cmd_mps": float(factual_v[row]),
                "counterfactual_v_cmd_mps": float(cf_v[i, row]),
                "delta_v_cmd_mps": float(delta_v[i, row]),
                "factual_omega_cmd_rad_s": float(factual_omega[row]),
                "counterfactual_omega_cmd_rad_s": float(cf_omega[i, row]),
                "delta_omega_cmd_rad_s": float(delta_omega[i, row]),
                "factual_critic_value": float(factual_value[row]),
                "counterfactual_critic_value": float(cf_value[i, row]),
                "delta_critic_value": float(delta_value[i, row]),
            })
    return rows


def _bearing_rows(config: dict[str, Any], protocol: Any, interventions: tuple[str, ...], physical: np.ndarray, seeds: np.ndarray, phases: np.ndarray, delta_omega: np.ndarray) -> list[dict[str, object]]:
    index = protocol.observation_order.index("pedestrian_bearing_mean_rad")
    bearing = physical[:, index]
    threshold = float(config["diagnostic"]["pedestrian_bearing_center_half_width_rad"])
    strata = {
        "right": bearing < -threshold,
        "center": np.abs(bearing) <= threshold,
        "left": bearing > threshold,
    }
    intervention_index = interventions.index("pedestrian_absent")
    rows: list[dict[str, object]] = []
    for label, mask in strata.items():
        mask &= phases == "pedestrian_relevant"
        if not np.any(mask):
            continue
        summary = paired_effect_summary(
            delta_omega[intervention_index, mask],
            seeds[mask],
            bootstrap_replicates=int(config["bootstrap"]["replicates"]),
            bootstrap_seed=int(config["bootstrap"]["seed"]),
            confidence_level=float(config["bootstrap"]["confidence_level"]),
            direction_tolerance=float(config["gate"]["direction_tolerance"]),
        )
        rows.append({"bearing_stratum": label, **summary})
    return rows


def _outputs(config_path: Path, config: dict[str, Any]) -> dict[str, Path]:
    directory = _resolve(config_path, str(config["output"]["directory"]))
    result = {"directory": directory}
    for name, value in config["output"].items():
        if name != "directory":
            result[name] = directory / str(value)
    return result


def _resolve(base: Path, value: str) -> Path:
    return (base.parent / value).resolve()


def _device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _model_hash(agent: PPOAgent) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(agent.model.state_dict().items()):
        digest.update(name.encode())
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError("refusing to write empty R006 CSV")
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
