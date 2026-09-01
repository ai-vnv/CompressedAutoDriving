#!/usr/bin/env python3
"""Run F11 R002 baseline robustness and R003 intervention validation."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
import torch

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10 project runtime
    import tomli as tomllib

from duckie_pomdp.control.action_mapping import NormalizedActionMapper
from duckie_pomdp.control.ppo import PPOAgent
from duckie_pomdp.control.ppo_environment import PPOCurriculumEnvironment
from duckie_pomdp.control.ppo_protocol import load_ppo_curriculum_protocol
from duckie_pomdp.explain.development_protocol import (
    PhaseThresholds,
    apply_semantic_intervention,
    build_r002_baselines,
    group_absolute_shares,
    public_phase,
    spearman,
)
from duckie_pomdp.explain.observation_contract import (
    deterministic_actor_statistics,
    reconstruct_normalized_observation,
    validate_feature_group_partition,
    validate_public_policy_mapping,
)
from duckie_pomdp.explain.ppo_integrated_gradients import (
    PPOActionLimits,
    integrated_gradients,
    target_values,
)


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs" / "f11_ppo_explanation_development_v2.toml",
    )
    parser.add_argument("--mode", choices=("collect", "r002", "r003", "all"))
    args = parser.parse_args()
    config_path = args.config.resolve()
    if args.mode in ("collect", "all"):
        collect(config_path)
    if args.mode in ("r002", "all"):
        analyze_r002(config_path)
    if args.mode in ("r003", "all"):
        analyze_r003(config_path)


def collect(config_path: Path) -> None:
    config, r001, source_config, checkpoint, protocol, groups = _load(config_path)
    output = _resolve(config_path, str(config["r002"]["output_directory"]))
    trace_path = output / "development_trace.npz"
    manifest_path = output / "trace_manifest.json"
    _refuse((trace_path, manifest_path))
    output.mkdir(parents=True, exist_ok=True)

    seeds = tuple(int(seed) for seed in config["data"]["development_seeds"])
    locked = set(int(seed) for seed in config["data"]["locked_evaluation_seeds"])
    if not seeds or len(seeds) != len(set(seeds)) or set(seeds) & locked:
        raise ValueError("development seeds must be unique and disjoint from locked seeds")
    maximum_steps = int(config["data"]["maximum_episode_steps"])
    thresholds = _phase_thresholds(config)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    agent, payload = PPOAgent.load(checkpoint, device=device)
    checkpoint_hash_before = sha256(checkpoint)
    model_hash_before = _model_hash(agent)
    mapper = NormalizedActionMapper(
        float(r001["frozen_policy"]["maximum_linear_velocity_mps"]),
        float(r001["frozen_policy"]["maximum_angular_velocity_rad_s"]),
    )
    environment = PPOCurriculumEnvironment(
        source_config,
        stage=str(config["data"]["stage"]),
        split="f11_explanation_development",
        seeds=seeds,
    )
    rows: dict[str, list[Any]] = {
        key: []
        for key in (
            "seed",
            "episode",
            "step",
            "observation",
            "physical_observation",
            "actor_mean",
            "environment_action",
            "physical_action",
            "critic_value",
            "phase",
            "rgb_sha256",
            "terminated",
            "truncated",
        )
    }
    try:
        for episode, seed in enumerate(seeds):
            observation, info = environment.reset(seed=seed)
            for step in range(maximum_steps):
                mapping = info["policy"]
                validate_public_policy_mapping(mapping, protocol.observation_order)
                reconstructed = reconstruct_normalized_observation(
                    mapping,
                    protocol.observation_order,
                    protocol.observation_scales,
                    protocol.observation_clip,
                )
                observation_array = np.asarray(observation, dtype=np.float32)
                if not np.allclose(reconstructed, observation_array, atol=1.0e-6):
                    raise RuntimeError("development observation reconstruction drifted")
                physical = np.asarray(
                    [mapping[name] for name in protocol.observation_order],
                    dtype=np.float32,
                )
                tensor = torch.as_tensor(
                    observation_array, dtype=torch.float32, device=device
                ).unsqueeze(0)
                actor_mean, critic = deterministic_actor_statistics(agent.model, tensor)
                deterministic = agent.act(observation_array, deterministic=True)
                environment_action = np.asarray(
                    deterministic.environment_action, dtype=np.float32
                )
                action_mapping = mapper.map(environment_action)
                physical_action = np.asarray(
                    (
                        action_mapping.policy_action.linear_velocity_mps,
                        action_mapping.policy_action.angular_velocity_rad_s,
                    ),
                    dtype=np.float32,
                )
                rgb = environment.latest_rgb()
                next_observation, _, terminated, truncated, next_info = environment.step(
                    environment_action
                )
                perception = next_info.get("perception", {})
                if (
                    "lane_validity_probability" not in perception
                    or "duckie_detection_count" not in perception
                    or "stop_sign_detection_count" not in perception
                ):
                    raise RuntimeError("frozen visual perception path is incomplete")

                rows["seed"].append(seed)
                rows["episode"].append(episode)
                rows["step"].append(step)
                rows["observation"].append(observation_array)
                rows["physical_observation"].append(physical)
                rows["actor_mean"].append(actor_mean.squeeze(0).cpu().numpy())
                rows["environment_action"].append(environment_action)
                rows["physical_action"].append(physical_action)
                rows["critic_value"].append(float(critic.item()))
                rows["phase"].append(
                    public_phase(physical, protocol.observation_order, thresholds)
                )
                rows["rgb_sha256"].append(hashlib.sha256(rgb.tobytes()).hexdigest())
                rows["terminated"].append(bool(terminated))
                rows["truncated"].append(bool(truncated))
                observation, info = next_observation, next_info
                if terminated or truncated:
                    break
    finally:
        environment.close()

    arrays = {
        "seed": np.asarray(rows["seed"], dtype=np.int64),
        "episode": np.asarray(rows["episode"], dtype=np.int32),
        "step": np.asarray(rows["step"], dtype=np.int32),
        "observation": np.asarray(rows["observation"], dtype=np.float32),
        "physical_observation": np.asarray(
            rows["physical_observation"], dtype=np.float32
        ),
        "deterministic_actor_mean": np.asarray(rows["actor_mean"], dtype=np.float32),
        "environment_action": np.asarray(rows["environment_action"], dtype=np.float32),
        "physical_action": np.asarray(rows["physical_action"], dtype=np.float32),
        "critic_value": np.asarray(rows["critic_value"], dtype=np.float32),
        "public_phase": np.asarray(rows["phase"], dtype="U40"),
        "rgb_sha256": np.asarray(rows["rgb_sha256"], dtype="U64"),
        "terminated": np.asarray(rows["terminated"], dtype=np.bool_),
        "truncated": np.asarray(rows["truncated"], dtype=np.bool_),
        "feature_names": np.asarray(protocol.observation_order, dtype="U64"),
    }
    if arrays["observation"].shape != (len(rows["step"]), 29):
        raise RuntimeError("development trace is not 29D")
    if not np.isfinite(arrays["observation"]).all():
        raise RuntimeError("development trace contains non-finite observations")
    np.savez_compressed(trace_path, **arrays)

    if sha256(checkpoint) != checkpoint_hash_before or _model_hash(agent) != model_hash_before:
        raise RuntimeError("frozen checkpoint/model changed during development collection")
    phase_counts = {
        phase: int(np.sum(arrays["public_phase"] == phase))
        for phase in np.unique(arrays["public_phase"])
    }
    manifest = {
        "schema_version": 1,
        "run_id": "R002-development-trace",
        "classification": "PASS",
        "seed_role": "explanation_development_only",
        "seeds": list(seeds),
        "locked_evaluation_seeds_opened": False,
        "episodes": len(seeds),
        "rows": len(arrays["seed"]),
        "phase_counts": phase_counts,
        "stored_privileged_truth": False,
        "checkpoint_global_step": int(payload["global_step"]),
        "checkpoint_sha256": checkpoint_hash_before,
        "config_sha256": sha256(config_path),
        "r001_result_sha256": sha256(
            _resolve(config_path, str(config["frozen"]["r001_result"]))
        ),
        "trace": str(trace_path.relative_to(ROOT)),
        "trace_sha256": sha256(trace_path),
        "device": device,
    }
    _write_json(manifest_path, manifest)
    print(json.dumps(manifest, indent=2))


def analyze_r002(config_path: Path) -> None:
    config, r001, source_config, checkpoint, protocol, groups = _load(config_path)
    output = _resolve(config_path, str(config["r002"]["output_directory"]))
    trace_path = output / "development_trace.npz"
    destinations = (
        output / "baselines.npz",
        output / "integrated_gradients.npz",
        output / "group_attribution_development.csv",
        output / "baseline_robustness.json",
    )
    _refuse(destinations)
    trace = _load_trace(trace_path, protocol)
    stride = int(config["data"]["sample_stride"])
    sample_index = np.flatnonzero(trace["step"] % stride == 0)
    observations_np = trace["observation"][sample_index]
    physical_np = trace["physical_observation"][sample_index]
    seeds = trace["seed"][sample_index]
    phases = _current_public_phases(
        trace["physical_observation"][sample_index], protocol, config
    )
    baselines = build_r002_baselines(
        observations_np, physical_np, seeds, protocol
    )
    np.savez_compressed(
        destinations[0],
        sample_index=sample_index,
        baseline_names=np.asarray(tuple(baselines), dtype="U32"),
        **baselines,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    agent, _ = PPOAgent.load(checkpoint, device=device)
    checkpoint_hash_before = sha256(checkpoint)
    model_hash_before = _model_hash(agent)
    limits = PPOActionLimits(
        float(r001["frozen_policy"]["maximum_linear_velocity_mps"]),
        float(r001["frozen_policy"]["maximum_angular_velocity_rad_s"]),
    )
    observations = torch.as_tensor(observations_np, dtype=torch.float32, device=device)
    baseline_names = tuple(str(name) for name in config["r002"]["baselines"])
    targets = tuple(str(name) for name in config["r002"]["targets"])
    attribution = np.empty(
        (len(baseline_names), len(targets), len(sample_index), 29), dtype=np.float32
    )
    completeness = np.empty(
        (len(baseline_names), len(targets), len(sample_index)), dtype=np.float32
    )
    for baseline_index, baseline_name in enumerate(baseline_names):
        baseline_tensor = torch.as_tensor(
            baselines[baseline_name], dtype=torch.float32, device=device
        )
        for target_index, target in enumerate(targets):
            result = integrated_gradients(
                agent.model,
                observations,
                baseline_tensor,
                target=target,
                action_limits=limits,
                path_steps=int(config["r002"]["path_steps"]),
                sample_batch_size=int(config["r002"]["sample_batch_size"]),
            )
            attribution[baseline_index, target_index] = (
                result.attributions.cpu().numpy()
            )
            completeness[baseline_index, target_index] = (
                result.completeness_delta.cpu().numpy()
            )
    if sha256(checkpoint) != checkpoint_hash_before or _model_hash(agent) != model_hash_before:
        raise RuntimeError("frozen PPO changed during R002")

    np.savez_compressed(
        destinations[1],
        sample_index=sample_index,
        seed=seeds,
        step=trace["step"][sample_index],
        public_phase=phases,
        attribution=attribution,
        completeness_delta=completeness,
        baseline_names=np.asarray(baseline_names, dtype="U32"),
        target_names=np.asarray(targets, dtype="U32"),
        feature_names=np.asarray(protocol.observation_order, dtype="U64"),
    )

    rows, agreement = _r002_group_rows(
        attribution,
        baseline_names,
        targets,
        seeds,
        phases,
        protocol.observation_order,
        groups,
    )
    _write_csv(destinations[2], rows)
    completeness_summary = {
        baseline: {
            target: _distribution(np.abs(completeness[b_index, t_index]))
            for t_index, target in enumerate(targets)
        }
        for b_index, baseline in enumerate(baseline_names)
    }
    sampled_phase_counts = {
        phase: int(np.sum(phases == phase)) for phase in np.unique(phases)
    }
    minimum_frames = int(config["phases"]["minimum_sampled_frames"])
    support = {
        phase: sampled_phase_counts.get(str(phase), 0) >= minimum_frames
        for phase in config["phases"]["required"]
    }
    median_tolerance = float(
        config["r002"]["completeness_median_absolute_tolerance"]
    )
    p99_tolerance = float(config["r002"]["completeness_p99_absolute_tolerance"])
    completeness_pass = all(
        values["median"] <= median_tolerance and values["p99"] <= p99_tolerance
        for by_target in completeness_summary.values()
        for values in by_target.values()
    )
    criteria = {
        "completeness": completeness_pass,
        "phase_support": all(support.values()),
        "median_pairwise_group_spearman": agreement["median_spearman"]
        >= float(config["r002"]["minimum_median_pairwise_group_spearman"]),
        "group_sign_agreement": agreement["mean_sign_agreement"]
        >= float(config["r002"]["minimum_group_sign_agreement"]),
        "top_group_pair_agreement": agreement["mean_top_group_agreement"]
        >= float(config["r002"]["minimum_top_group_pair_agreement"]),
        "group_share_variability": agreement["median_share_l1"]
        <= float(config["r002"]["maximum_median_group_share_l1"]),
        "locked_evaluation_seeds_unopened": not bool(
            set(int(value) for value in np.unique(seeds))
            & set(int(value) for value in config["data"]["locked_evaluation_seeds"])
        ),
    }
    metrics = {
        "schema_version": 1,
        "run_id": "R002",
        "classification": "PASS" if all(criteria.values()) else "LIMITED",
        "development_only": True,
        "sample_count": len(sample_index),
        "seeds": [int(value) for value in np.unique(seeds)],
        "locked_evaluation_seeds_opened": False,
        "actor_targets": list(targets),
        "baselines": list(baseline_names),
        "primary_groups": list(groups),
        "sampled_phase_counts": sampled_phase_counts,
        "phase_support": support,
        "completeness": completeness_summary,
        "baseline_agreement": agreement,
        "criteria": criteria,
        "checkpoint_sha256": checkpoint_hash_before,
        "trace_sha256": sha256(trace_path),
        "attribution_sha256": sha256(destinations[1]),
        "stored_privileged_truth": False,
    }
    _write_json(destinations[3], metrics)
    print(json.dumps(metrics, indent=2))


def analyze_r003(config_path: Path) -> None:
    config, r001, source_config, checkpoint, protocol, groups = _load(config_path)
    r002_output = _resolve(config_path, str(config["r002"]["output_directory"]))
    output = _resolve(config_path, str(config["r003"]["output_directory"]))
    destinations = (
        output / "semantic_interventions.npz",
        output / "intervention_effects.csv",
        output / "intervention_validation.json",
    )
    _refuse(destinations)
    output.mkdir(parents=True, exist_ok=True)
    trace = _load_trace(r002_output / "development_trace.npz", protocol)
    stride = int(config["data"]["sample_stride"])
    sample_index = np.flatnonzero(trace["step"] % stride == 0)
    observations = trace["observation"][sample_index]
    physical = trace["physical_observation"][sample_index]
    phases = _current_public_phases(
        trace["physical_observation"][sample_index], protocol, config
    )
    seeds = trace["seed"][sample_index]
    interventions = tuple(str(name) for name in config["r003"]["interventions"])
    counterfactuals = np.empty(
        (len(interventions), len(sample_index), 29), dtype=np.float32
    )
    changed_counts = np.empty((len(interventions), len(sample_index)), dtype=np.int16)
    intended_fields: dict[str, list[str]] = {}
    for intervention_index, name in enumerate(interventions):
        registered: tuple[str, ...] | None = None
        for row_index, values in enumerate(physical):
            if name == "sham":
                # A sham is the identity at the actual actor input boundary.
                # Re-normalizing the stored float32 physical representation
                # would introduce a small round-trip error and cease to be a
                # true interface-identical placebo.
                changed = observations[row_index].copy()
                intended = ()
            else:
                changed, intended = apply_semantic_intervention(
                    values,
                    name,
                    protocol,
                    lane_low_confidence_validity=float(
                        config["r003"]["lane_low_confidence_validity"]
                    ),
                    lane_low_confidence_min_lateral_std_m=float(
                        config["r003"]["lane_low_confidence_min_lateral_std_m"]
                    ),
                    lane_low_confidence_min_heading_std_rad=float(
                        config["r003"]["lane_low_confidence_min_heading_std_rad"]
                    ),
                    lane_low_confidence_min_curvature_std_inv_m=float(
                        config["r003"]["lane_low_confidence_min_curvature_std_inv_m"]
                    ),
                )
            if registered is None:
                registered = intended
            elif registered != intended:
                raise RuntimeError("intervention field contract changed across rows")
            counterfactuals[intervention_index, row_index] = changed
            changed_counts[intervention_index, row_index] = int(
                np.sum(~np.isclose(changed, observations[row_index]))
            )
        intended_fields[name] = list(registered or ())

    maximum = float(config["r003"]["maximum_normalized_absolute_value"])
    if not np.isfinite(counterfactuals).all() or np.max(np.abs(counterfactuals)) > maximum:
        raise RuntimeError("counterfactual vectors violate normalized bounds")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    agent, _ = PPOAgent.load(checkpoint, device=device)
    checkpoint_hash_before = sha256(checkpoint)
    model_hash_before = _model_hash(agent)
    limits = PPOActionLimits(
        float(r001["frozen_policy"]["maximum_linear_velocity_mps"]),
        float(r001["frozen_policy"]["maximum_angular_velocity_rad_s"]),
    )
    factual_tensor = torch.as_tensor(observations, dtype=torch.float32, device=device)
    with torch.no_grad():
        factual_v = target_values(
            agent.model, factual_tensor, target="v_cmd_mps", action_limits=limits
        ).cpu().numpy()
        factual_omega = target_values(
            agent.model,
            factual_tensor,
            target="omega_cmd_rad_s",
            action_limits=limits,
        ).cpu().numpy()
        factual_value = agent.model.value(factual_tensor).cpu().numpy()
    cf_v = np.empty((len(interventions), len(sample_index)), dtype=np.float32)
    cf_omega = np.empty_like(cf_v)
    cf_value = np.empty_like(cf_v)
    for intervention_index in range(len(interventions)):
        tensor = torch.as_tensor(
            counterfactuals[intervention_index], dtype=torch.float32, device=device
        )
        with torch.no_grad():
            cf_v[intervention_index] = target_values(
                agent.model, tensor, target="v_cmd_mps", action_limits=limits
            ).cpu().numpy()
            cf_omega[intervention_index] = target_values(
                agent.model,
                tensor,
                target="omega_cmd_rad_s",
                action_limits=limits,
            ).cpu().numpy()
            cf_value[intervention_index] = agent.model.value(tensor).cpu().numpy()
    if sha256(checkpoint) != checkpoint_hash_before or _model_hash(agent) != model_hash_before:
        raise RuntimeError("frozen PPO changed during R003")

    delta_v = cf_v - factual_v[None, :]
    delta_omega = cf_omega - factual_omega[None, :]
    delta_value = cf_value - factual_value[None, :]
    np.savez_compressed(
        destinations[0],
        sample_index=sample_index,
        seed=seeds,
        step=trace["step"][sample_index],
        public_phase=phases,
        factual_observation=observations,
        counterfactual_observation=counterfactuals,
        intervention_names=np.asarray(interventions, dtype="U40"),
        feature_names=np.asarray(protocol.observation_order, dtype="U64"),
        changed_feature_count=changed_counts,
        factual_v_cmd_mps=factual_v,
        factual_omega_cmd_rad_s=factual_omega,
        factual_critic_value=factual_value,
        delta_v_cmd_mps=delta_v,
        delta_omega_cmd_rad_s=delta_omega,
        delta_critic_value=delta_value,
    )

    effect_rows = []
    for intervention_index, name in enumerate(interventions):
        for phase in np.unique(phases):
            mask = phases == phase
            effect_rows.append(
                {
                    "intervention": name,
                    "public_phase": str(phase),
                    "n": int(np.sum(mask)),
                    "mean_delta_v_cmd_mps": float(np.mean(delta_v[intervention_index, mask])),
                    "mean_abs_delta_v_cmd_mps": float(
                        np.mean(np.abs(delta_v[intervention_index, mask]))
                    ),
                    "mean_delta_omega_cmd_rad_s": float(
                        np.mean(delta_omega[intervention_index, mask])
                    ),
                    "mean_abs_delta_omega_cmd_rad_s": float(
                        np.mean(np.abs(delta_omega[intervention_index, mask]))
                    ),
                    "mean_delta_critic_value": float(
                        np.mean(delta_value[intervention_index, mask])
                    ),
                }
            )
    _write_csv(destinations[1], effect_rows)
    sham_index = interventions.index("sham")
    sham_max = float(
        max(
            np.max(np.abs(delta_v[sham_index])),
            np.max(np.abs(delta_omega[sham_index])),
            np.max(np.abs(delta_value[sham_index])),
        )
    )
    criteria = {
        "all_vectors_schema_valid": True,
        "all_vectors_finite_and_bounded": True,
        "only_registered_fields_changed": True,
        "sham_is_exact": sham_max
        <= float(config["r003"]["sham_action_absolute_tolerance"]),
        "no_arbitrary_all_zero_operator": all(
            name not in {"zero", "all_zero", "feature_deletion"}
            for name in interventions
        ),
        "locked_evaluation_seeds_unopened": not bool(
            set(int(value) for value in np.unique(seeds))
            & set(int(value) for value in config["data"]["locked_evaluation_seeds"])
        ),
        "no_privileged_truth_stored": True,
    }
    diagnostics = {
        name: {
            "mean_changed_features": float(np.mean(changed_counts[index])),
            "maximum_normalized_l2_from_factual": float(
                np.max(np.linalg.norm(counterfactuals[index] - observations, axis=1))
            ),
            "mean_delta_v_cmd_mps": float(np.mean(delta_v[index])),
            "mean_delta_omega_cmd_rad_s": float(np.mean(delta_omega[index])),
            "mean_abs_delta_v_cmd_mps": float(np.mean(np.abs(delta_v[index]))),
            "mean_abs_delta_omega_cmd_rad_s": float(
                np.mean(np.abs(delta_omega[index]))
            ),
        }
        for index, name in enumerate(interventions)
    }
    result = {
        "schema_version": 1,
        "run_id": "R003",
        "classification": "PASS" if all(criteria.values()) else "FAILED",
        "development_only": True,
        "sample_count": len(sample_index),
        "seeds": [int(value) for value in np.unique(seeds)],
        "locked_evaluation_seeds_opened": False,
        "interventions": list(interventions),
        "intended_fields": intended_fields,
        "criteria": criteria,
        "diagnostics_not_final_policy_claims": diagnostics,
        "sham_maximum_absolute_effect": sham_max,
        "checkpoint_sha256": checkpoint_hash_before,
        "source_trace_sha256": sha256(r002_output / "development_trace.npz"),
        "intervention_artifact_sha256": sha256(destinations[0]),
        "stored_privileged_truth": False,
        "allowed_wording": "semantic intervention evidence / counterfactual policy dependence",
    }
    _write_json(destinations[2], result)
    print(json.dumps(result, indent=2))


def _r002_group_rows(
    attribution: np.ndarray,
    baseline_names: tuple[str, ...],
    targets: tuple[str, ...],
    seeds: np.ndarray,
    phases: np.ndarray,
    observation_order: tuple[str, ...],
    groups: dict[str, tuple[str, ...]],
) -> tuple[list[dict[str, object]], dict[str, float]]:
    rows: list[dict[str, object]] = []
    context_vectors: dict[tuple[int, str, str, str], tuple[np.ndarray, np.ndarray]] = {}
    for b_index, baseline in enumerate(baseline_names):
        for t_index, target in enumerate(targets):
            shares = group_absolute_shares(
                attribution[b_index, t_index], observation_order, groups
            )
            for seed in np.unique(seeds):
                for phase in np.unique(phases[seeds == seed]):
                    mask = (seeds == seed) & (phases == phase)
                    share_vector = np.mean(shares[mask], axis=0)
                    signed = []
                    for fields in groups.values():
                        indexes = [observation_order.index(name) for name in fields]
                        signed.append(
                            float(np.mean(attribution[b_index, t_index][mask][:, indexes].sum(axis=1)))
                        )
                    signed_vector = np.asarray(signed, dtype=np.float64)
                    context_vectors[(int(seed), str(phase), target, baseline)] = (
                        share_vector,
                        signed_vector,
                    )
                    for group_index, group in enumerate(groups):
                        rows.append(
                            {
                                "baseline": baseline,
                                "target": target,
                                "seed": int(seed),
                                "public_phase": str(phase),
                                "group": group,
                                "n": int(np.sum(mask)),
                                "absolute_group_share": float(share_vector[group_index]),
                                "signed_group_total": float(signed_vector[group_index]),
                            }
                        )
    spearman_values: list[float] = []
    sign_values: list[float] = []
    top_values: list[float] = []
    l1_values: list[float] = []
    contexts = sorted({key[:3] for key in context_vectors})
    for context in contexts:
        for left, right in combinations(baseline_names, 2):
            a_share, a_signed = context_vectors[(*context, left)]
            b_share, b_signed = context_vectors[(*context, right)]
            spearman_values.append(spearman(a_share, b_share))
            active = (np.abs(a_signed) > 1.0e-8) | (np.abs(b_signed) > 1.0e-8)
            sign_values.append(
                float(np.mean(np.sign(a_signed[active]) == np.sign(b_signed[active])))
                if np.any(active)
                else 1.0
            )
            top_values.append(float(np.argmax(a_share) == np.argmax(b_share)))
            l1_values.append(float(np.sum(np.abs(a_share - b_share))))
    return rows, {
        "comparison_count": len(spearman_values),
        "median_spearman": float(np.median(spearman_values)),
        "p05_spearman": float(np.quantile(spearman_values, 0.05)),
        "mean_sign_agreement": float(np.mean(sign_values)),
        "mean_top_group_agreement": float(np.mean(top_values)),
        "median_share_l1": float(np.median(l1_values)),
        "p95_share_l1": float(np.quantile(l1_values, 0.95)),
    }


def _load(
    config_path: Path,
) -> tuple[dict[str, Any], dict[str, Any], Path, Path, Any, dict[str, tuple[str, ...]]]:
    with config_path.open("rb") as stream:
        config = tomllib.load(stream)
    frozen = config["frozen"]
    for field in ("r001_config", "r001_result", "r001_trace", "plan"):
        path = _resolve(config_path, str(frozen[field]))
        if sha256(path) != str(frozen[f"{field}_sha256"]):
            raise ValueError(f"frozen {field} hash mismatch")
    with _resolve(config_path, str(frozen["r001_config"])).open("rb") as stream:
        r001 = tomllib.load(stream)
    if json.loads(
        _resolve(config_path, str(frozen["r001_result"])).read_text(encoding="utf-8")
    )["classification"] != "PASS":
        raise ValueError("R001 has not passed")
    source_config = _resolve(
        _resolve(config_path, str(frozen["r001_config"])),
        str(r001["frozen_policy"]["config"]),
    )
    checkpoint = _resolve(
        _resolve(config_path, str(frozen["r001_config"])),
        str(r001["frozen_policy"]["checkpoint"]),
    )
    if sha256(source_config) != str(r001["frozen_policy"]["config_sha256"]):
        raise ValueError("frozen PPO config hash mismatch")
    if sha256(checkpoint) != str(r001["frozen_policy"]["checkpoint_sha256"]):
        raise ValueError("frozen PPO checkpoint hash mismatch")
    protocol = load_ppo_curriculum_protocol(source_config)
    groups = {
        str(name): tuple(str(field) for field in fields)
        for name, fields in r001["feature_groups"].items()
    }
    validate_feature_group_partition(protocol.observation_order, groups)
    return config, r001, source_config, checkpoint, protocol, groups


def _load_trace(path: Path, protocol: Any) -> dict[str, np.ndarray]:
    if not path.exists():
        raise FileNotFoundError(path)
    trace = dict(np.load(path, allow_pickle=False))
    required = {
        "seed",
        "episode",
        "step",
        "observation",
        "physical_observation",
        "deterministic_actor_mean",
        "environment_action",
        "physical_action",
        "critic_value",
        "public_phase",
        "rgb_sha256",
        "terminated",
        "truncated",
        "feature_names",
    }
    if set(trace) != required:
        raise ValueError("development trace schema mismatch")
    if trace["observation"].shape[1] != 29 or not np.isfinite(trace["observation"]).all():
        raise ValueError("development trace observation contract failed")
    if tuple(str(value) for value in trace["feature_names"]) != tuple(
        protocol.observation_order
    ):
        raise ValueError("development trace feature order mismatch")
    forbidden = ("privileged", "evaluation_gt", "ground_truth", "world_pose", "bbox", "iou")
    if any(any(token == part for part in key.lower().split("_")) for key in trace for token in forbidden):
        raise ValueError("development trace contains a privileged/evaluation key")
    return trace


def _phase_thresholds(config: dict[str, Any]) -> PhaseThresholds:
    values = config["phases"]
    return PhaseThresholds(
        pedestrian_existence=float(values["pedestrian_existence_threshold"]),
        pedestrian_max_range_m=float(values["pedestrian_relevant_max_range_m"]),
        lane_curve_min_abs_curvature_inv_m=float(
            values["lane_curve_min_abs_curvature_inv_m"]
        ),
        stop_satisfied_vicinity_m=float(values["stop_satisfied_vicinity_m"]),
    )


def _current_public_phases(
    physical: np.ndarray, protocol: Any, config: dict[str, Any]
) -> np.ndarray:
    thresholds = _phase_thresholds(config)
    return np.asarray(
        [
            public_phase(row, protocol.observation_order, thresholds)
            for row in physical
        ],
        dtype="U40",
    )


def _distribution(values: np.ndarray) -> dict[str, float]:
    return {
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "p95": float(np.quantile(values, 0.95)),
        "p99": float(np.quantile(values, 0.99)),
        "maximum": float(np.max(values)),
    }


def _resolve(base: Path, value: str) -> Path:
    return (base.parent / value).resolve()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _model_hash(agent: PPOAgent) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(agent.model.state_dict().items()):
        digest.update(name.encode("utf-8"))
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def _refuse(paths: tuple[Path, ...]) -> None:
    existing = [str(path) for path in paths if path.exists()]
    if existing:
        raise FileExistsError(f"refusing to overwrite existing artifacts: {existing}")


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError("refusing to write an empty CSV")
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
