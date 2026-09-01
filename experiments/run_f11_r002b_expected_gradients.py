#!/usr/bin/env python3
"""Run the single preregistered R002b distributional-IG development study."""

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
except ModuleNotFoundError:  # pragma: no cover - Python 3.10 runtime
    import tomli as tomllib

from duckie_pomdp.control.ppo import PPOAgent
from duckie_pomdp.control.ppo_protocol import load_ppo_curriculum_protocol
from duckie_pomdp.explain.development_protocol import (
    PhaseThresholds,
    draw_phase_conditioned_references,
    group_absolute_shares,
    public_phase,
    spearman,
)
from duckie_pomdp.explain.observation_contract import (
    validate_feature_group_partition,
)
from duckie_pomdp.explain.ppo_integrated_gradients import (
    PPOActionLimits,
    distributional_integrated_gradients,
)


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs" / "f11_ppo_explanation_r002b_v1.toml",
    )
    args = parser.parse_args()
    run(args.config.resolve())


def run(config_path: Path) -> None:
    config, development, r001, checkpoint, protocol, groups = _load(config_path)
    output = _resolve(config_path, str(config["output"]["directory"]))
    destinations = {
        key: output / str(config["output"][key])
        for key in ("references", "attributions", "group_rows", "metrics")
    }
    _refuse(tuple(destinations.values()))

    trace_path = _resolve(config_path, str(config["frozen"]["development_trace"]))
    trace = _load_trace(trace_path, protocol)
    stride = int(config["data"]["sample_stride"])
    sample_index = np.flatnonzero(trace["step"] % stride == 0)
    observations_np = np.asarray(trace["observation"][sample_index], dtype=np.float32)
    physical_np = np.asarray(
        trace["physical_observation"][sample_index], dtype=np.float32
    )
    seeds = np.asarray(trace["seed"][sample_index], dtype=np.int64)
    phases = _public_phases(physical_np, protocol, development)

    expected_development = tuple(int(value) for value in config["data"]["development_seeds"])
    locked = set(int(value) for value in config["data"]["locked_evaluation_seeds"])
    observed_seeds = tuple(int(value) for value in np.unique(seeds))
    if observed_seeds != expected_development:
        raise ValueError("R002b trace does not contain exactly the frozen development seeds")
    if set(observed_seeds) & locked:
        raise ValueError("locked evaluation seed appeared in R002b input")

    reference_config = config["reference_distribution"]
    draw_seeds = tuple(int(value) for value in reference_config["draw_seeds"])
    if len(draw_seeds) < 2 or len(draw_seeds) != len(set(draw_seeds)):
        raise ValueError("R002b needs at least two unique deterministic draws")
    reference_count = int(reference_config["references_per_input"])
    exclude_same_seed = bool(reference_config["exclude_same_seed"])
    if not bool(reference_config["condition_on_public_phase"]):
        raise ValueError("R002b must remain phase-conditioned")
    if not bool(reference_config["sample_without_replacement"]):
        raise ValueError("R002b reference protocol is frozen without replacement")

    references = np.empty(
        (len(draw_seeds), reference_count, len(sample_index), 29), dtype=np.float32
    )
    reference_indexes = np.empty(
        (len(draw_seeds), reference_count, len(sample_index)), dtype=np.int64
    )
    for draw_index, draw_seed in enumerate(draw_seeds):
        drawn, indexes = draw_phase_conditioned_references(
            observations_np,
            phases,
            seeds,
            draw_seed=draw_seed,
            references_per_input=reference_count,
            exclude_same_seed=exclude_same_seed,
        )
        references[draw_index] = drawn
        reference_indexes[draw_index] = indexes

    reference_seeds = seeds[reference_indexes]
    reference_phases = phases[reference_indexes]
    same_phase = bool(np.all(reference_phases == phases[None, None, :]))
    cross_seed = bool(np.all(reference_seeds != seeds[None, None, :]))
    if not same_phase or not cross_seed:
        raise RuntimeError("phase-conditioned cross-seed reference invariant failed")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    agent, _ = PPOAgent.load(checkpoint, device=device)
    checkpoint_hash_before = sha256(checkpoint)
    model_hash_before = _model_hash(agent)
    limits = PPOActionLimits(
        float(r001["frozen_policy"]["maximum_linear_velocity_mps"]),
        float(r001["frozen_policy"]["maximum_angular_velocity_rad_s"]),
    )
    observations = torch.as_tensor(observations_np, dtype=torch.float32, device=device)
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
            completeness[draw_index, target_index] = (
                result.completeness_delta.cpu().numpy()
            )
            reference_values[draw_index, target_index] = (
                result.mean_reference_values.cpu().numpy()
            )
    if sha256(checkpoint) != checkpoint_hash_before or _model_hash(agent) != model_hash_before:
        raise RuntimeError("frozen PPO changed during R002b")

    draw_names = tuple(f"draw_{index + 1:02d}" for index in range(len(draw_seeds)))
    rows, agreement, per_seed = _group_analysis(
        attributions,
        draw_names,
        targets,
        seeds,
        phases,
        protocol.observation_order,
        groups,
    )
    bootstrap = _seed_cluster_bootstrap(
        per_seed,
        replicates=int(config["bootstrap"]["replicates"]),
        random_seed=int(config["bootstrap"]["seed"]),
        confidence=float(config["bootstrap"]["confidence_level"]),
    )
    completeness_summary = {
        draw_names[draw_index]: {
            target: _distribution(np.abs(completeness[draw_index, target_index]))
            for target_index, target in enumerate(targets)
        }
        for draw_index in range(len(draw_seeds))
    }
    median_tolerance = float(
        config["gate"]["completeness_median_absolute_tolerance"]
    )
    p99_tolerance = float(config["gate"]["completeness_p99_absolute_tolerance"])
    completeness_pass = all(
        values["median"] <= median_tolerance and values["p99"] <= p99_tolerance
        for by_target in completeness_summary.values()
        for values in by_target.values()
    )
    required_phases = tuple(str(value) for value in development["phases"]["required"])
    phase_counts = {phase: int(np.sum(phases == phase)) for phase in np.unique(phases)}
    phase_support = {
        phase: phase_counts.get(phase, 0)
        >= int(development["phases"]["minimum_sampled_frames"])
        for phase in required_phases
    }
    criteria = {
        "completeness": completeness_pass,
        "public_phase_support": all(phase_support.values()),
        "phase_conditioned_references": same_phase,
        "cross_seed_references": cross_seed,
        "median_pairwise_group_spearman": agreement["median_spearman"]
        >= float(config["gate"]["minimum_median_pairwise_group_spearman"]),
        "group_sign_agreement": agreement["mean_sign_agreement"]
        >= float(config["gate"]["minimum_group_sign_agreement"]),
        "top_group_pair_agreement": agreement["mean_top_group_agreement"]
        >= float(config["gate"]["minimum_top_group_pair_agreement"]),
        "group_share_variability": agreement["median_share_l1"]
        <= float(config["gate"]["maximum_median_group_share_l1"]),
        "cross_seed_bootstrap_reported": all(
            "ci_low" in values and "ci_high" in values
            for values in bootstrap.values()
        ),
        "locked_evaluation_seeds_unopened": not bool(set(observed_seeds) & locked),
        "no_privileged_truth_stored": True,
    }
    classification = "PASS" if all(criteria.values()) else "LIMITED"

    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        destinations["references"],
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
        destinations["attributions"],
        sample_index=sample_index,
        seed=seeds,
        step=trace["step"][sample_index],
        public_phase=phases,
        attribution=attributions,
        completeness_delta=completeness,
        mean_reference_value=reference_values,
        draw_names=np.asarray(draw_names, dtype="U16"),
        draw_seeds=np.asarray(draw_seeds, dtype=np.int64),
        target_names=np.asarray(targets, dtype="U32"),
        feature_names=np.asarray(protocol.observation_order, dtype="U64"),
    )
    _write_csv(destinations["group_rows"], rows)
    metrics = {
        "schema_version": 1,
        "run_id": "R002b",
        "classification": classification,
        "development_only": True,
        "method": "phase-conditioned distributional multi-reference integrated gradients",
        "sample_count": len(sample_index),
        "seeds": list(observed_seeds),
        "locked_evaluation_seeds_opened": False,
        "reference_protocol": {
            "kind": str(reference_config["kind"]),
            "draw_count": len(draw_seeds),
            "draw_seeds": list(draw_seeds),
            "references_per_input": reference_count,
            "exclude_same_seed": exclude_same_seed,
            "sample_without_replacement": True,
        },
        "actor_targets": list(targets),
        "primary_groups": list(groups),
        "sampled_phase_counts": phase_counts,
        "phase_support": phase_support,
        "completeness": completeness_summary,
        "draw_agreement": agreement,
        "cross_seed_bootstrap_95pct": bootstrap,
        "criteria": criteria,
        "r004_unlocked": classification == "PASS",
        "r002c_permitted": False,
        "checkpoint_sha256": checkpoint_hash_before,
        "config_sha256": sha256(config_path),
        "source_trace_sha256": sha256(trace_path),
        "references_sha256": sha256(destinations["references"]),
        "attribution_sha256": sha256(destinations["attributions"]),
        "stored_privileged_truth": False,
    }
    _write_json(destinations["metrics"], metrics)
    print(json.dumps(metrics, indent=2))


def _group_analysis(
    attribution: np.ndarray,
    draw_names: tuple[str, ...],
    targets: tuple[str, ...],
    seeds: np.ndarray,
    phases: np.ndarray,
    observation_order: tuple[str, ...],
    groups: dict[str, tuple[str, ...]],
) -> tuple[list[dict[str, object]], dict[str, float], dict[int, dict[str, list[float]]]]:
    rows: list[dict[str, object]] = []
    contexts: dict[tuple[int, str, str, str], tuple[np.ndarray, np.ndarray]] = {}
    for draw_index, draw in enumerate(draw_names):
        for target_index, target in enumerate(targets):
            shares = group_absolute_shares(
                attribution[draw_index, target_index], observation_order, groups
            )
            for seed in np.unique(seeds):
                for phase in np.unique(phases[seeds == seed]):
                    mask = (seeds == seed) & (phases == phase)
                    share_vector = np.mean(shares[mask], axis=0)
                    signed_vector = np.asarray(
                        [
                            float(
                                np.mean(
                                    attribution[draw_index, target_index][mask][
                                        :, [observation_order.index(name) for name in fields]
                                    ].sum(axis=1)
                                )
                            )
                            for fields in groups.values()
                        ],
                        dtype=np.float64,
                    )
                    contexts[(int(seed), str(phase), target, draw)] = (
                        share_vector,
                        signed_vector,
                    )
                    for group_index, group in enumerate(groups):
                        rows.append(
                            {
                                "draw": draw,
                                "target": target,
                                "seed": int(seed),
                                "public_phase": str(phase),
                                "group": group,
                                "n": int(np.sum(mask)),
                                "absolute_group_share": float(share_vector[group_index]),
                                "signed_group_total": float(signed_vector[group_index]),
                            }
                        )

    per_seed = {
        int(seed): {name: [] for name in ("spearman", "sign", "top", "l1")}
        for seed in np.unique(seeds)
    }
    for seed, phase, target in sorted({key[:3] for key in contexts}):
        for left, right in combinations(draw_names, 2):
            left_share, left_signed = contexts[(seed, phase, target, left)]
            right_share, right_signed = contexts[(seed, phase, target, right)]
            active = (np.abs(left_signed) > 1.0e-8) | (np.abs(right_signed) > 1.0e-8)
            per_seed[seed]["spearman"].append(spearman(left_share, right_share))
            per_seed[seed]["sign"].append(
                float(np.mean(np.sign(left_signed[active]) == np.sign(right_signed[active])))
                if np.any(active)
                else 1.0
            )
            per_seed[seed]["top"].append(
                float(np.argmax(left_share) == np.argmax(right_share))
            )
            per_seed[seed]["l1"].append(float(np.sum(np.abs(left_share - right_share))))
    agreement = _aggregate_agreement(per_seed, tuple(sorted(per_seed)))
    return rows, agreement, per_seed


def _aggregate_agreement(
    per_seed: dict[int, dict[str, list[float]]], selected_seeds: tuple[int, ...]
) -> dict[str, float]:
    values = {
        name: np.concatenate(
            [np.asarray(per_seed[seed][name], dtype=np.float64) for seed in selected_seeds]
        )
        for name in ("spearman", "sign", "top", "l1")
    }
    return {
        "comparison_count": int(len(values["spearman"])),
        "median_spearman": float(np.median(values["spearman"])),
        "p05_spearman": float(np.quantile(values["spearman"], 0.05)),
        "mean_sign_agreement": float(np.mean(values["sign"])),
        "mean_top_group_agreement": float(np.mean(values["top"])),
        "median_share_l1": float(np.median(values["l1"])),
        "p95_share_l1": float(np.quantile(values["l1"], 0.95)),
    }


def _seed_cluster_bootstrap(
    per_seed: dict[int, dict[str, list[float]]],
    *,
    replicates: int,
    random_seed: int,
    confidence: float,
) -> dict[str, dict[str, float]]:
    if replicates <= 0 or not 0.0 < confidence < 1.0:
        raise ValueError("invalid bootstrap configuration")
    seeds = np.asarray(sorted(per_seed), dtype=np.int64)
    rng = np.random.default_rng(random_seed)
    metric_names = (
        "median_spearman",
        "mean_sign_agreement",
        "mean_top_group_agreement",
        "median_share_l1",
    )
    estimates = {name: np.empty(replicates, dtype=np.float64) for name in metric_names}
    for replicate in range(replicates):
        selected = tuple(int(value) for value in rng.choice(seeds, size=len(seeds), replace=True))
        summary = _aggregate_agreement(per_seed, selected)
        for name in metric_names:
            estimates[name][replicate] = summary[name]
    tail = (1.0 - confidence) / 2.0
    return {
        name: {
            "mean": float(np.mean(values)),
            "ci_low": float(np.quantile(values, tail)),
            "ci_high": float(np.quantile(values, 1.0 - tail)),
        }
        for name, values in estimates.items()
    }


def _load(config_path: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], Path, Any, dict[str, tuple[str, ...]]]:
    with config_path.open("rb") as stream:
        config = tomllib.load(stream)
    for field in (
        "r002_r003_config",
        "r002_metrics",
        "r003_metrics",
        "development_trace",
        "r001_result",
    ):
        path = _resolve(config_path, str(config["frozen"][field]))
        if sha256(path) != str(config["frozen"][f"{field}_sha256"]):
            raise ValueError(f"frozen {field} hash mismatch")
    development_path = _resolve(
        config_path, str(config["frozen"]["r002_r003_config"])
    )
    with development_path.open("rb") as stream:
        development = tomllib.load(stream)
    r002 = json.loads(
        _resolve(config_path, str(config["frozen"]["r002_metrics"])).read_text()
    )
    r003 = json.loads(
        _resolve(config_path, str(config["frozen"]["r003_metrics"])).read_text()
    )
    if r002["classification"] != "LIMITED" or r003["classification"] != "PASS":
        raise ValueError("R002b requires frozen R002 LIMITED and R003 PASS")
    r001_path = _resolve(config_path, str(config["frozen"]["r001_result"]))
    r001_result = json.loads(r001_path.read_text())
    if r001_result["classification"] != "PASS":
        raise ValueError("R001 must remain PASS")
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
    return config, development, r001, checkpoint, protocol, groups


def _load_trace(path: Path, protocol: Any) -> dict[str, np.ndarray]:
    trace = dict(np.load(path, allow_pickle=False))
    if trace["observation"].shape[1] != 29:
        raise ValueError("development trace is not 29D")
    if tuple(str(value) for value in trace["feature_names"]) != tuple(protocol.observation_order):
        raise ValueError("development trace feature order mismatch")
    forbidden = ("privileged", "evaluation_gt", "ground_truth", "world_pose", "bbox", "iou")
    if any(
        any(token == part for part in key.lower().split("_"))
        for key in trace
        for token in forbidden
    ):
        raise ValueError("development trace contains privileged/evaluation schema")
    return trace


def _public_phases(physical: np.ndarray, protocol: Any, development: dict[str, Any]) -> np.ndarray:
    values = development["phases"]
    thresholds = PhaseThresholds(
        pedestrian_existence=float(values["pedestrian_existence_threshold"]),
        pedestrian_max_range_m=float(values["pedestrian_relevant_max_range_m"]),
        lane_curve_min_abs_curvature_inv_m=float(values["lane_curve_min_abs_curvature_inv_m"]),
        stop_satisfied_vicinity_m=float(values["stop_satisfied_vicinity_m"]),
    )
    return np.asarray(
        [public_phase(row, protocol.observation_order, thresholds) for row in physical],
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
        digest.update(name.encode())
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def _refuse(paths: tuple[Path, ...]) -> None:
    existing = [str(path) for path in paths if path.exists()]
    if existing:
        raise FileExistsError(f"refusing to overwrite R002b artifacts: {existing}")


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError("refusing to write an empty CSV")
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()

