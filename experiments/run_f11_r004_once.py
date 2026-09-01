#!/usr/bin/env python3
"""Run the frozen once-only F11 R004 locked actor attribution."""

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

from duckie_pomdp.control.action_mapping import NormalizedActionMapper
from duckie_pomdp.control.ppo import PPOAgent
from duckie_pomdp.control.ppo_environment import PPOCurriculumEnvironment
from duckie_pomdp.control.ppo_protocol import load_ppo_curriculum_protocol
from duckie_pomdp.explain.development_protocol import (
    PhaseThresholds,
    group_absolute_shares,
    public_phase,
)
from duckie_pomdp.explain.final_attribution import (
    InsufficientReferenceSupport,
    draw_locked_same_phase_distinct_seed_references,
    locked_phase_seed_support,
    mean_all_reference_attributions,
)
from duckie_pomdp.explain.observation_contract import (
    deterministic_actor_statistics,
    reconstruct_normalized_observation,
    validate_feature_group_partition,
    validate_public_policy_mapping,
)
from duckie_pomdp.explain.ppo_integrated_gradients import (
    PPOActionLimits,
    distributional_integrated_gradients,
)
from run_f11_r002b_expected_gradients import (
    _distribution,
    _group_analysis,
    _seed_cluster_bootstrap,
)


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs" / "f11_ppo_explanation_r004_v1.toml",
    )
    parser.add_argument("--mode", choices=("preflight", "once"), required=True)
    args = parser.parse_args()
    config_path = args.config.resolve()
    if args.mode == "preflight":
        preflight(config_path)
    else:
        run_once(config_path)


def preflight(config_path: Path) -> None:
    config, r001, source_config, checkpoint, protocol, groups = _load(config_path)
    output = _output_paths(config_path, config)
    if output["directory"].exists():
        raise FileExistsError("R004 output already exists; once-only run is closed")
    agent, payload = PPOAgent.load(
        checkpoint, device="cuda" if torch.cuda.is_available() else "cpu"
    )
    device = next(agent.model.parameters()).device
    probe = torch.zeros((2, len(protocol.observation_order)), device=device)
    actor, critic = deterministic_actor_statistics(agent.model, probe)
    if actor.shape != (2, 2) or critic.shape != (2,):
        raise RuntimeError("frozen actor/critic technical probe failed")
    print(
        json.dumps(
            {
                "classification": "PASS",
                "mode": "preflight",
                "locked_seeds_opened": False,
                "r004_output_exists": False,
                "checkpoint_sha256": sha256(checkpoint),
                "checkpoint_global_step": int(payload["global_step"]),
                "observation_dimension": len(protocol.observation_order),
                "groups": list(groups),
                "device": str(device),
            },
            indent=2,
        )
    )


def run_once(config_path: Path) -> None:
    config, r001, source_config, checkpoint, protocol, groups = _load(config_path)
    output = _output_paths(config_path, config)
    if output["directory"].exists():
        raise FileExistsError("refusing to reopen once-only R004")
    output["directory"].mkdir(parents=True)
    _write_json(
        output["launch_claim"],
        {
            "schema_version": 1,
            "run_id": "R004",
            "once_only": True,
            "launched_at_utc": datetime.now(timezone.utc).isoformat(),
            "config_sha256": sha256(config_path),
            "locked_seeds": [int(value) for value in config["data"]["locked_evaluation_seeds"]],
            "locked_seed_access_begins_after_this_claim": True,
            "rerun_permitted": False,
        },
    )
    try:
        trace, trace_manifest = _collect_locked(
            config_path, config, r001, source_config, checkpoint, protocol
        )
        np.savez_compressed(output["trace"], **trace)
        trace_manifest["trace_sha256"] = sha256(output["trace"])
        _write_json(output["trace_manifest"], trace_manifest)
        _analyze_locked(
            config_path,
            config,
            r001,
            checkpoint,
            protocol,
            groups,
            trace,
            output,
        )
    except Exception as error:
        _write_json(
            output["directory"] / "once_only_failure.json",
            {
                "classification": "FAILED",
                "error_type": type(error).__name__,
                "error": str(error),
                "rerun_permitted": False,
            },
        )
        raise


def _collect_locked(
    config_path: Path,
    config: dict[str, Any],
    r001: dict[str, Any],
    source_config: Path,
    checkpoint: Path,
    protocol: Any,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    seeds = tuple(int(value) for value in config["data"]["locked_evaluation_seeds"])
    forbidden = set(int(value) for value in config["data"]["forbidden_development_seeds"])
    if len(seeds) != len(set(seeds)) or set(seeds) & forbidden:
        raise ValueError("locked seed protocol is invalid")
    agent, payload = PPOAgent.load(
        checkpoint, device="cuda" if torch.cuda.is_available() else "cpu"
    )
    device = next(agent.model.parameters()).device
    checkpoint_hash_before = sha256(checkpoint)
    model_hash_before = _model_hash(agent)
    mapper = NormalizedActionMapper(
        float(r001["frozen_policy"]["maximum_linear_velocity_mps"]),
        float(r001["frozen_policy"]["maximum_angular_velocity_rad_s"]),
    )
    environment = PPOCurriculumEnvironment(
        source_config,
        stage=str(config["data"]["stage"]),
        split="f11_r004_once_only_locked",
        seeds=seeds,
    )
    thresholds = _phase_thresholds(config)
    rows: dict[str, list[Any]] = {
        name: []
        for name in (
            "seed", "episode", "step", "observation", "physical_observation",
            "actor_mean", "environment_action", "physical_action", "critic_value",
            "phase", "rgb_sha256", "terminated", "truncated",
        )
    }
    try:
        for episode, seed in enumerate(seeds):
            observation, info = environment.reset(seed=seed)
            for step in range(int(config["data"]["maximum_episode_steps"])):
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
                    raise RuntimeError("locked observation reconstruction drifted")
                physical = np.asarray(
                    [mapping[name] for name in protocol.observation_order], dtype=np.float32
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
    if sha256(checkpoint) != checkpoint_hash_before or _model_hash(agent) != model_hash_before:
        raise RuntimeError("frozen PPO changed during R004 collection")
    trace = {
        "seed": np.asarray(rows["seed"], dtype=np.int64),
        "episode": np.asarray(rows["episode"], dtype=np.int32),
        "step": np.asarray(rows["step"], dtype=np.int32),
        "observation": np.asarray(rows["observation"], dtype=np.float32),
        "physical_observation": np.asarray(rows["physical_observation"], dtype=np.float32),
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
    if trace["observation"].ndim != 2 or trace["observation"].shape[1] != 29:
        raise RuntimeError("locked trace is not 29D")
    if not np.isfinite(trace["observation"]).all():
        raise RuntimeError("locked trace contains non-finite actor inputs")
    phase_counts = {
        phase: int(np.sum(trace["public_phase"] == phase))
        for phase in np.unique(trace["public_phase"])
    }
    manifest = {
        "schema_version": 1,
        "run_id": "R004-locked-trace",
        "seed_role": "once_only_locked_evaluation",
        "seeds": list(seeds),
        "episodes": len(seeds),
        "rows": len(trace["seed"]),
        "phase_counts": phase_counts,
        "stored_privileged_truth": False,
        "checkpoint_global_step": int(payload["global_step"]),
        "checkpoint_sha256": checkpoint_hash_before,
        "config_sha256": sha256(config_path),
        "device": str(device),
    }
    return trace, manifest


def _analyze_locked(
    config_path: Path,
    config: dict[str, Any],
    r001: dict[str, Any],
    checkpoint: Path,
    protocol: Any,
    groups: dict[str, tuple[str, ...]],
    trace: dict[str, np.ndarray],
    output: dict[str, Path],
) -> None:
    stride = int(config["data"]["sample_stride"])
    sample_index = np.flatnonzero(trace["step"] % stride == 0)
    observations_np = trace["observation"][sample_index]
    physical_np = trace["physical_observation"][sample_index]
    seeds = trace["seed"][sample_index]
    phases = np.asarray(
        [
            public_phase(row, protocol.observation_order, _phase_thresholds(config))
            for row in physical_np
        ],
        dtype="U40",
    )
    support = locked_phase_seed_support(phases, seeds)
    required_phases = tuple(str(value) for value in config["phases"]["required"])
    minimum_other = int(config["reference_distribution"]["minimum_other_seed_support"])
    phase_seed_support = {
        phase: {
            "seed_count": len(support.get(phase, ())),
            "seeds": list(support.get(phase, ())),
            "sufficient": len(support.get(phase, ())) >= minimum_other + 1,
        }
        for phase in required_phases
    }
    if not all(value["sufficient"] for value in phase_seed_support.values()):
        _write_json(
            output["metrics"],
            {
                "schema_version": 1,
                "run_id": "R004",
                "classification": "LIMITED",
                "reason": "insufficient same-phase distinct-cross-seed reference support",
                "phase_seed_support": phase_seed_support,
                "fallback_used": False,
                "stored_privileged_truth": False,
                "rerun_permitted": False,
            },
        )
        return

    reference_config = config["reference_distribution"]
    draw_seeds = tuple(int(value) for value in reference_config["draw_seeds"])
    reference_count = int(reference_config["references_per_input"])
    references = np.empty(
        (len(draw_seeds), reference_count, len(sample_index), 29), dtype=np.float32
    )
    reference_indexes = np.empty(
        (len(draw_seeds), reference_count, len(sample_index)), dtype=np.int64
    )
    try:
        for draw_index, draw_seed in enumerate(draw_seeds):
            references[draw_index], reference_indexes[draw_index] = (
                draw_locked_same_phase_distinct_seed_references(
                    observations_np,
                    phases,
                    seeds,
                    draw_seed=draw_seed,
                    references_per_input=reference_count,
                    minimum_other_seed_support=minimum_other,
                )
            )
    except InsufficientReferenceSupport as error:  # fail closed, no fallback
        _write_json(
            output["metrics"],
            {
                "schema_version": 1,
                "run_id": "R004",
                "classification": "LIMITED",
                "reason": str(error),
                "phase_seed_support": phase_seed_support,
                "fallback_used": False,
                "stored_privileged_truth": False,
                "rerun_permitted": False,
            },
        )
        return
    reference_seeds = seeds[reference_indexes]
    reference_phases = phases[reference_indexes]
    same_phase = bool(np.all(reference_phases == phases[None, None, :]))
    cross_seed = bool(np.all(reference_seeds != seeds[None, None, :]))
    distinct_seed = bool(
        all(
            len(np.unique(reference_seeds[draw, :, row])) == reference_count
            for draw in range(len(draw_seeds))
            for row in range(len(sample_index))
        )
    )
    if not same_phase or not cross_seed or not distinct_seed:
        raise RuntimeError("locked reference invariant failed")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    agent, _ = PPOAgent.load(checkpoint, device=device)
    checkpoint_hash_before = sha256(checkpoint)
    model_hash_before = _model_hash(agent)
    observations = torch.as_tensor(observations_np, dtype=torch.float32, device=device)
    limits = PPOActionLimits(
        float(r001["frozen_policy"]["maximum_linear_velocity_mps"]),
        float(r001["frozen_policy"]["maximum_angular_velocity_rad_s"]),
    )
    targets = tuple(str(value) for value in config["attribution"]["targets"])
    attributions = np.empty(
        (len(draw_seeds), len(targets), len(sample_index), 29), dtype=np.float32
    )
    completeness = np.empty(
        (len(draw_seeds), len(targets), len(sample_index)), dtype=np.float32
    )
    reference_values = np.empty_like(completeness)
    for draw_index in range(len(draw_seeds)):
        reference_tensor = torch.as_tensor(
            references[draw_index], dtype=torch.float32, device=device
        )
        for target_index, target in enumerate(targets):
            result = distributional_integrated_gradients(
                agent.model,
                observations,
                reference_tensor,
                target=target,
                action_limits=limits,
                path_steps=int(config["attribution"]["path_steps"]),
                sample_batch_size=int(config["attribution"]["sample_batch_size"]),
            )
            attributions[draw_index, target_index] = result.attributions.cpu().numpy()
            completeness[draw_index, target_index] = result.completeness_delta.cpu().numpy()
            reference_values[draw_index, target_index] = result.mean_reference_values.cpu().numpy()
    if sha256(checkpoint) != checkpoint_hash_before or _model_hash(agent) != model_hash_before:
        raise RuntimeError("frozen PPO changed during R004 attribution")
    final_attribution = mean_all_reference_attributions(attributions)
    final_completeness = np.asarray(np.mean(completeness, axis=0), dtype=np.float32)
    draw_names = tuple(f"draw_{index + 1:02d}" for index in range(len(draw_seeds)))
    _, agreement, per_seed = _group_analysis(
        attributions,
        draw_names,
        targets,
        seeds,
        phases,
        protocol.observation_order,
        groups,
    )
    agreement_bootstrap = _seed_cluster_bootstrap(
        per_seed,
        replicates=int(config["bootstrap"]["replicates"]),
        random_seed=int(config["bootstrap"]["seed"]),
        confidence=float(config["bootstrap"]["confidence_level"]),
    )
    group_rows, group_summary, group_overall = _final_group_outputs(
        final_attribution,
        targets,
        seeds,
        trace["step"][sample_index],
        phases,
        protocol.observation_order,
        groups,
        bootstrap_replicates=int(config["bootstrap"]["replicates"]),
        bootstrap_seed=int(config["bootstrap"]["seed"]) + 1,
        confidence=float(config["bootstrap"]["confidence_level"]),
    )
    phase_counts = {phase: int(np.sum(phases == phase)) for phase in np.unique(phases)}
    phase_support = {
        phase: phase_counts.get(phase, 0) >= int(config["phases"]["minimum_sampled_frames"])
        for phase in required_phases
    }
    completeness_summary = {
        target: _distribution(np.abs(final_completeness[target_index]))
        for target_index, target in enumerate(targets)
    }
    criteria = {
        "final_completeness": all(
            values["median"] <= float(config["gate"]["completeness_median_absolute_tolerance"])
            and values["p99"] <= float(config["gate"]["completeness_p99_absolute_tolerance"])
            for values in completeness_summary.values()
        ),
        "public_phase_support": all(phase_support.values()),
        "same_phase_references": same_phase,
        "distinct_cross_seed_references": cross_seed and distinct_seed,
        "median_pairwise_group_spearman": agreement["median_spearman"]
        >= float(config["gate"]["minimum_median_pairwise_group_spearman"]),
        "group_sign_agreement": agreement["mean_sign_agreement"]
        >= float(config["gate"]["minimum_group_sign_agreement"]),
        "top_group_pair_agreement": agreement["mean_top_group_agreement"]
        >= float(config["gate"]["minimum_top_group_pair_agreement"]),
        "group_share_variability": agreement["median_share_l1"]
        <= float(config["gate"]["maximum_median_group_share_l1"]),
        "no_privileged_truth_stored": True,
        "all_24_references_equally_aggregated": int(config["attribution"]["effective_reference_count"])
        == len(draw_seeds) * reference_count,
    }
    classification = "PASS" if all(criteria.values()) else "LIMITED"

    np.savez_compressed(
        output["references"],
        sample_index=sample_index,
        seed=seeds,
        step=trace["step"][sample_index],
        public_phase=phases,
        observation=observations_np,
        draw_seeds=np.asarray(draw_seeds, dtype=np.int64),
        reference_index=reference_indexes,
        reference_seed=reference_seeds,
        reference_observation=references,
        feature_names=np.asarray(protocol.observation_order, dtype="U64"),
    )
    np.savez_compressed(
        output["draw_attributions"],
        sample_index=sample_index,
        seed=seeds,
        step=trace["step"][sample_index],
        public_phase=phases,
        attribution=attributions,
        completeness_delta=completeness,
        mean_reference_value=reference_values,
        draw_names=np.asarray(draw_names, dtype="U16"),
        target_names=np.asarray(targets, dtype="U32"),
        feature_names=np.asarray(protocol.observation_order, dtype="U64"),
    )
    np.savez_compressed(
        output["final_attributions"],
        sample_index=sample_index,
        seed=seeds,
        step=trace["step"][sample_index],
        public_phase=phases,
        attribution=final_attribution,
        completeness_delta=final_completeness,
        effective_reference_count=np.asarray(24, dtype=np.int32),
        target_names=np.asarray(targets, dtype="U32"),
        feature_names=np.asarray(protocol.observation_order, dtype="U64"),
    )
    _write_csv(output["group_rows"], group_rows)
    _write_csv(output["group_summary"], group_summary)
    metrics = {
        "schema_version": 1,
        "run_id": "R004",
        "classification": classification,
        "once_only_locked_evaluation": True,
        "sample_count": len(sample_index),
        "seeds": [int(value) for value in np.unique(seeds)],
        "reference_protocol": {
            "draw_count": len(draw_seeds),
            "references_per_draw": reference_count,
            "effective_reference_count": len(draw_seeds) * reference_count,
            "same_phase": same_phase,
            "cross_seed": cross_seed,
            "distinct_seed_within_draw": distinct_seed,
            "fallback_used": False,
        },
        "final_estimator": "equal mean of all six draw means / 24 references",
        "phase_counts": phase_counts,
        "phase_seed_support": phase_seed_support,
        "phase_support": phase_support,
        "completeness": completeness_summary,
        "draw_agreement": agreement,
        "draw_agreement_seed_bootstrap_95pct": agreement_bootstrap,
        "final_group_attribution_overall": group_overall,
        "criteria": criteria,
        "checkpoint_sha256": checkpoint_hash_before,
        "config_sha256": sha256(config_path),
        "trace_sha256": sha256(output["trace"]),
        "references_sha256": sha256(output["references"]),
        "draw_attribution_sha256": sha256(output["draw_attributions"]),
        "final_attribution_sha256": sha256(output["final_attributions"]),
        "stored_privileged_truth": False,
        "rerun_permitted": False,
        "r006_r007_started": False,
    }
    _write_json(output["metrics"], metrics)
    print(json.dumps(metrics, indent=2))


def _final_group_outputs(
    attribution: np.ndarray,
    targets: tuple[str, ...],
    seeds: np.ndarray,
    steps: np.ndarray,
    phases: np.ndarray,
    observation_order: tuple[str, ...],
    groups: dict[str, tuple[str, ...]],
    *,
    bootstrap_replicates: int,
    bootstrap_seed: int,
    confidence: float,
) -> tuple[list[dict[str, object]], list[dict[str, object]], dict[str, Any]]:
    rows: list[dict[str, object]] = []
    summary: list[dict[str, object]] = []
    overall: dict[str, Any] = {}
    rng = np.random.default_rng(bootstrap_seed)
    tail = (1.0 - confidence) / 2.0
    for target_index, target in enumerate(targets):
        shares = group_absolute_shares(attribution[target_index], observation_order, groups)
        signed = np.stack(
            [
                attribution[target_index][
                    :, [observation_order.index(name) for name in fields]
                ].sum(axis=1)
                for fields in groups.values()
            ],
            axis=1,
        )
        for row_index in range(len(seeds)):
            for group_index, group in enumerate(groups):
                rows.append(
                    {
                        "sample_index": row_index,
                        "seed": int(seeds[row_index]),
                        "step": int(steps[row_index]),
                        "public_phase": str(phases[row_index]),
                        "target": target,
                        "group": group,
                        "absolute_group_share": float(shares[row_index, group_index]),
                        "signed_group_total": float(signed[row_index, group_index]),
                    }
                )
        overall[target] = {}
        for phase in ("all", *tuple(str(value) for value in np.unique(phases))):
            mask = np.ones(len(seeds), dtype=bool) if phase == "all" else phases == phase
            available_seeds = np.unique(seeds[mask])
            for group_index, group in enumerate(groups):
                share_values = shares[mask, group_index]
                signed_values = signed[mask, group_index]
                boot_share = np.empty(bootstrap_replicates, dtype=np.float64)
                boot_signed = np.empty(bootstrap_replicates, dtype=np.float64)
                phase_indexes = np.flatnonzero(mask)
                by_seed = {
                    int(seed): phase_indexes[seeds[phase_indexes] == seed]
                    for seed in available_seeds
                }
                for replicate in range(bootstrap_replicates):
                    selected_seeds = rng.choice(
                        available_seeds, size=len(available_seeds), replace=True
                    )
                    selected_indexes = np.concatenate(
                        [by_seed[int(seed)] for seed in selected_seeds]
                    )
                    boot_share[replicate] = float(
                        np.mean(shares[selected_indexes, group_index])
                    )
                    boot_signed[replicate] = float(
                        np.mean(signed[selected_indexes, group_index])
                    )
                item = {
                    "target": target,
                    "public_phase": phase,
                    "group": group,
                    "n": int(np.sum(mask)),
                    "seed_count": len(available_seeds),
                    "mean_absolute_group_share": float(np.mean(share_values)),
                    "share_ci_low": float(np.quantile(boot_share, tail)),
                    "share_ci_high": float(np.quantile(boot_share, 1.0 - tail)),
                    "mean_signed_group_total": float(np.mean(signed_values)),
                    "signed_ci_low": float(np.quantile(boot_signed, tail)),
                    "signed_ci_high": float(np.quantile(boot_signed, 1.0 - tail)),
                }
                summary.append(item)
                if phase == "all":
                    overall[target][group] = {
                        "mean_absolute_group_share": item["mean_absolute_group_share"],
                        "share_ci_low": item["share_ci_low"],
                        "share_ci_high": item["share_ci_high"],
                        "mean_signed_group_total": item["mean_signed_group_total"],
                    }
        overall[target] = dict(
            sorted(
                overall[target].items(),
                key=lambda item: item[1]["mean_absolute_group_share"],
                reverse=True,
            )
        )
    return rows, summary, overall


def _load(config_path: Path) -> tuple[dict[str, Any], dict[str, Any], Path, Path, Any, dict[str, tuple[str, ...]]]:
    with config_path.open("rb") as stream:
        config = tomllib.load(stream)
    for field in (
        "r002b_config", "r002b_metrics", "r002b_manifest", "r002b_protocol",
        "r003_metrics", "r001_result", "r002b_runner",
    ):
        path = _resolve(config_path, str(config["frozen"][field]))
        if sha256(path) != str(config["frozen"][f"{field}_sha256"]):
            raise ValueError(f"frozen {field} hash mismatch")
    r002b_metrics = json.loads(
        _resolve(config_path, str(config["frozen"]["r002b_metrics"])).read_text()
    )
    r003_metrics = json.loads(
        _resolve(config_path, str(config["frozen"]["r003_metrics"])).read_text()
    )
    if r002b_metrics["classification"] != "PASS" or not r002b_metrics["r004_unlocked"]:
        raise ValueError("R002b has not unlocked R004")
    if r003_metrics["classification"] != "PASS":
        raise ValueError("R003 must remain PASS")
    r002b_config_path = _resolve(config_path, str(config["frozen"]["r002b_config"]))
    with r002b_config_path.open("rb") as stream:
        r002b = tomllib.load(stream)
    development_path = _resolve(
        r002b_config_path, str(r002b["frozen"]["r002_r003_config"])
    )
    with development_path.open("rb") as stream:
        development = tomllib.load(stream)
    r001_config_path = _resolve(
        development_path, str(development["frozen"]["r001_config"])
    )
    with r001_config_path.open("rb") as stream:
        r001 = tomllib.load(stream)
    source_config = _resolve(r001_config_path, str(r001["frozen_policy"]["config"]))
    checkpoint = _resolve(r001_config_path, str(r001["frozen_policy"]["checkpoint"]))
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


def _output_paths(config_path: Path, config: dict[str, Any]) -> dict[str, Path]:
    directory = _resolve(config_path, str(config["output"]["directory"]))
    values = {"directory": directory}
    for name, value in config["output"].items():
        if name != "directory":
            values[name] = directory / str(value)
    return values


def _phase_thresholds(config: dict[str, Any]) -> PhaseThresholds:
    values = config["phases"]
    return PhaseThresholds(
        pedestrian_existence=float(values["pedestrian_existence_threshold"]),
        pedestrian_max_range_m=float(values["pedestrian_relevant_max_range_m"]),
        lane_curve_min_abs_curvature_inv_m=float(values["lane_curve_min_abs_curvature_inv_m"]),
        stop_satisfied_vicinity_m=float(values["stop_satisfied_vicinity_m"]),
    )


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
        digest.update(name.encode())
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError("refusing to write empty R004 CSV")
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()

