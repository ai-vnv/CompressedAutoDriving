"""Collect and explain frozen C4 PPO decisions with Integrated Gradients."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10 in the frozen project environment.
    import tomli as tomllib

from duckie_pomdp.control.ppo import PPOAgent
from duckie_pomdp.control.ppo_environment import PPOCurriculumEnvironment
from duckie_pomdp.control.ppo_protocol import load_ppo_curriculum_protocol
from duckie_pomdp.explain.ppo_integrated_gradients import (
    PPOActionLimits,
    integrated_gradients,
    target_values,
)


ROOT = Path(__file__).resolve().parents[1]
FORBIDDEN_TRAJECTORY_TOKENS = (
    "evaluation_gt",
    "privileged",
    "true_",
    "world_pose",
    "gt_bbox",
    "collision",
    "reward",
)

SHORT_LABELS = (
    "lane valid", "lane d", "lane sigma d", "lane phi", "lane sigma phi",
    "v actual", "omega actual", "curvature", "sigma curvature", "stop-line dist",
    "P(ped)", "ped range", "ped sigma r", "ped bearing", "ped sigma beta",
    "ped rdot", "ped sigma rdot", "ped betadot", "ped sigma betadot",
    "P(sign)", "sign range", "sign sigma r", "sign bearing", "sign sigma beta",
    "stop NONE", "stop REQUIRED", "stop SATISFIED", "prev v cmd", "prev omega cmd",
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs" / "f11_ppo_integrated_gradients_v1.toml",
    )
    parser.add_argument(
        "--mode", choices=("collect", "analyze", "all", "verify"), default="all"
    )
    args = parser.parse_args()
    config = load_config(args.config)
    output = resolve_path(args.config, config["artifacts"]["directory"])
    trajectory_path = output / "policy_trajectory.npz"
    if args.mode in {"collect", "all"}:
        if trajectory_path.exists():
            raise FileExistsError(f"refusing to overwrite {trajectory_path}")
        output.mkdir(parents=True, exist_ok=True)
        collect(config, args.config, trajectory_path)
    if args.mode in {"analyze", "all"}:
        analyze(config, args.config, trajectory_path, output)
    if args.mode == "verify":
        verify_artifacts(config, args.config, output)


def load_config(path: Path) -> dict[str, Any]:
    with path.open("rb") as handle:
        config = tomllib.load(handle)
    if config.get("method") != "integrated_gradients":
        raise ValueError("F11 config must select integrated_gradients")
    if config["recorded_second_idea"]["status"] != "recorded_not_executed":
        raise ValueError("the second explanation idea must remain unexecuted")
    return config


def collect(config: dict[str, Any], config_path: Path, destination: Path) -> None:
    frozen = config["frozen_policy"]
    source_config = resolve_path(config_path, frozen["config"])
    checkpoint = resolve_path(config_path, frozen["checkpoint"])
    verify_hash(source_config, frozen["config_sha256"])
    checkpoint_hash_before = verify_hash(checkpoint, frozen["checkpoint_sha256"])
    protocol = load_ppo_curriculum_protocol(source_config)
    order = tuple(protocol.observation_order)
    if len(order) != int(frozen["observation_dimension"]):
        raise ValueError("frozen policy observation dimension mismatch")
    seeds = tuple(int(seed) for seed in config["data"]["seeds"])
    if len(seeds) != len(set(seeds)):
        raise ValueError("explanation seeds must be unique")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    agent, payload = PPOAgent.load(checkpoint, device=device)
    if str(payload["stage"]) != str(frozen["stage"]):
        raise ValueError("checkpoint stage mismatch")
    environment = PPOCurriculumEnvironment(
        source_config,
        stage=str(config["data"]["stage"]),
        split="explanation",
        seeds=seeds,
    )
    rows: dict[str, list[Any]] = {
        "seed": [], "episode": [], "step": [], "scenario": [],
        "pedestrian_mode": [], "observation": [], "baseline": [],
        "physical_observation": [], "raw_action": [], "environment_action": [],
        "physical_action": [], "critic_value": [], "terminated": [],
        "truncated": [],
    }
    try:
        for episode_index, seed in enumerate(seeds):
            observation, current_info = environment.reset(seed=seed)
            baseline = np.asarray(observation, dtype=np.float32).copy()
            _validate_policy_vector(observation, baseline, order)
            scenario = str(current_info["scenario"])
            pedestrian_mode = str(current_info.get("pedestrian_mode") or "none")
            for step in range(int(config["data"]["maximum_episode_steps"])):
                action = agent.act(np.asarray(observation, dtype=np.float32), deterministic=True)
                environment_action = np.asarray(action.environment_action, dtype=np.float32)
                physical = np.asarray(
                    (
                        0.5 * (environment_action[0] + 1.0)
                        * float(frozen["maximum_linear_velocity_mps"]),
                        environment_action[1]
                        * float(frozen["maximum_angular_velocity_rad_s"]),
                    ),
                    dtype=np.float32,
                )
                policy_mapping = current_info["policy"]
                physical_observation = np.asarray(
                    [float(policy_mapping[name]) for name in order], dtype=np.float32
                )
                next_observation, _, terminated, truncated, next_info = environment.step(
                    environment_action
                )
                if not np.isclose(float(next_info["v_cmd"]), physical[0], atol=1.0e-6):
                    raise RuntimeError("physical linear action mapping drifted")
                if not np.isclose(float(next_info["omega_cmd"]), physical[1], atol=1.0e-6):
                    raise RuntimeError("physical angular action mapping drifted")
                rows["seed"].append(seed)
                rows["episode"].append(episode_index)
                rows["step"].append(step)
                rows["scenario"].append(scenario)
                rows["pedestrian_mode"].append(pedestrian_mode)
                rows["observation"].append(np.asarray(observation, dtype=np.float32))
                rows["baseline"].append(baseline)
                rows["physical_observation"].append(physical_observation)
                rows["raw_action"].append(np.asarray(action.raw_action, dtype=np.float32))
                rows["environment_action"].append(environment_action)
                rows["physical_action"].append(physical)
                rows["critic_value"].append(float(action.value))
                rows["terminated"].append(bool(terminated))
                rows["truncated"].append(bool(truncated))
                observation = next_observation
                current_info = next_info
                if terminated or truncated:
                    break
    finally:
        environment.close()

    arrays = {
        "seed": np.asarray(rows["seed"], dtype=np.int64),
        "episode": np.asarray(rows["episode"], dtype=np.int32),
        "step": np.asarray(rows["step"], dtype=np.int32),
        "scenario": np.asarray(rows["scenario"], dtype="U64"),
        "pedestrian_mode": np.asarray(rows["pedestrian_mode"], dtype="U32"),
        "observation": np.asarray(rows["observation"], dtype=np.float32),
        "baseline": np.asarray(rows["baseline"], dtype=np.float32),
        "physical_observation": np.asarray(rows["physical_observation"], dtype=np.float32),
        "raw_action": np.asarray(rows["raw_action"], dtype=np.float32),
        "environment_action": np.asarray(rows["environment_action"], dtype=np.float32),
        "physical_action": np.asarray(rows["physical_action"], dtype=np.float32),
        "critic_value": np.asarray(rows["critic_value"], dtype=np.float32),
        "terminated": np.asarray(rows["terminated"], dtype=np.bool_),
        "truncated": np.asarray(rows["truncated"], dtype=np.bool_),
        "feature_names": np.asarray(order, dtype="U64"),
    }
    validate_trajectory_schema(arrays)
    np.savez_compressed(destination, **arrays)
    checkpoint_hash_after = sha256(checkpoint)
    if checkpoint_hash_after != checkpoint_hash_before:
        raise RuntimeError("frozen checkpoint changed during trajectory collection")
    metadata = {
        "schema_version": 1,
        "method": "integrated_gradients",
        "seed_role": config["data"]["seed_role"],
        "seeds": list(seeds),
        "rows": len(arrays["seed"]),
        "episodes": len(seeds),
        "observation_dimension": arrays["observation"].shape[1],
        "policy_config": str(source_config.relative_to(ROOT)),
        "policy_config_sha256": sha256(source_config),
        "checkpoint": str(checkpoint.relative_to(ROOT)),
        "checkpoint_sha256_before": checkpoint_hash_before,
        "checkpoint_sha256_after": checkpoint_hash_after,
        "trajectory": destination.name,
        "trajectory_sha256": sha256(destination),
        "stored_privileged_truth": False,
        "device": device,
        "torch_version": torch.__version__,
        "numpy_version": np.__version__,
    }
    write_json(destination.parent / "trajectory_manifest.json", metadata)


def analyze(
    config: dict[str, Any], config_path: Path, trajectory_path: Path, output: Path
) -> None:
    if not trajectory_path.exists():
        raise FileNotFoundError(trajectory_path)
    frozen = config["frozen_policy"]
    source_config = resolve_path(config_path, frozen["config"])
    checkpoint = resolve_path(config_path, frozen["checkpoint"])
    verify_hash(source_config, frozen["config_sha256"])
    checkpoint_hash_before = verify_hash(checkpoint, frozen["checkpoint_sha256"])
    protocol = load_ppo_curriculum_protocol(source_config)
    groups = resolve_groups(config, tuple(protocol.observation_order))
    trajectory = dict(np.load(trajectory_path, allow_pickle=False))
    validate_trajectory_schema(trajectory)
    stride = int(config["data"]["sample_stride"])
    sample_mask = trajectory["step"] % stride == 0
    sample_index = np.flatnonzero(sample_mask)
    if sample_index.size == 0:
        raise RuntimeError("explanation stride selected no frames")
    observations_np = trajectory["observation"][sample_index]
    baselines_np = trajectory["baseline"][sample_index]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    agent, payload = PPOAgent.load(checkpoint, device=device)
    del payload
    model = agent.model
    parameter_hash_before = model_parameter_sha256(model)
    limits = PPOActionLimits(
        float(frozen["maximum_linear_velocity_mps"]),
        float(frozen["maximum_angular_velocity_rad_s"]),
    )
    observations = torch.as_tensor(observations_np, dtype=torch.float32, device=device)
    baselines = torch.as_tensor(baselines_np, dtype=torch.float32, device=device)
    target_names = tuple(config["integrated_gradients"]["targets"])
    attribution_rows: list[np.ndarray] = []
    target_value_rows: list[np.ndarray] = []
    baseline_value_rows: list[np.ndarray] = []
    completeness_rows: list[np.ndarray] = []
    for target in target_names:
        result = integrated_gradients(
            model,
            observations,
            baselines,
            target=target,
            action_limits=limits,
            path_steps=int(config["integrated_gradients"]["path_steps"]),
            sample_batch_size=int(config["integrated_gradients"]["sample_batch_size"]),
        )
        attribution_rows.append(result.attributions.cpu().numpy())
        target_value_rows.append(result.input_values.cpu().numpy())
        baseline_value_rows.append(result.baseline_values.cpu().numpy())
        completeness_rows.append(result.completeness_delta.cpu().numpy())
    attributions = np.asarray(attribution_rows, dtype=np.float32)
    target_values_np = np.asarray(target_value_rows, dtype=np.float32)
    baseline_values_np = np.asarray(baseline_value_rows, dtype=np.float32)
    completeness = np.asarray(completeness_rows, dtype=np.float32)
    parameter_hash_after = model_parameter_sha256(model)
    checkpoint_hash_after = sha256(checkpoint)
    if parameter_hash_after != parameter_hash_before:
        raise RuntimeError("PPO parameters changed during explanation")
    if checkpoint_hash_after != checkpoint_hash_before:
        raise RuntimeError("frozen checkpoint changed during explanation")

    np.savez_compressed(
        output / "integrated_gradients.npz",
        sample_index=sample_index.astype(np.int64),
        seed=trajectory["seed"][sample_index],
        episode=trajectory["episode"][sample_index],
        step=trajectory["step"][sample_index],
        observations=observations_np,
        baselines=baselines_np,
        attributions=attributions,
        target_values=target_values_np,
        baseline_values=baseline_values_np,
        completeness_delta=completeness,
        target_names=np.asarray(target_names, dtype="U32"),
        feature_names=trajectory["feature_names"],
    )
    feature_rows = feature_summary(
        attributions, target_names, tuple(protocol.observation_order), groups
    )
    write_csv(output / "feature_attribution_summary.csv", feature_rows)
    group_rows = group_summary(feature_rows)
    write_csv(output / "group_attribution_summary.csv", group_rows)
    stability = stability_summary(
        attributions,
        trajectory["seed"][sample_index],
        trajectory["step"][sample_index],
        target_names,
    )
    faithfulness_rows, faithfulness_metrics = faithfulness_summary(
        model,
        observations,
        baselines,
        attributions,
        config,
        limits,
    )
    write_csv(output / "faithfulness.csv", faithfulness_rows)

    completeness_metrics = {
        target: distribution_summary(np.abs(completeness[index]))
        for index, target in enumerate(target_names)
    }
    tolerance_median = float(
        config["integrated_gradients"]["completeness_median_absolute_tolerance"]
    )
    tolerance_p99 = float(
        config["integrated_gradients"]["completeness_p99_absolute_tolerance"]
    )
    completeness_pass = all(
        values["median"] <= tolerance_median and values["p99"] <= tolerance_p99
        for values in completeness_metrics.values()
    )
    actor_targets = tuple(config["faithfulness"]["targets"])
    faithfulness_pass = all(
        faithfulness_metrics[target]["top_auc"]
        > faithfulness_metrics[target]["random_auc"]
        for target in actor_targets
    )
    classification = "PASS" if completeness_pass and faithfulness_pass else "LIMITED"
    metrics = {
        "schema_version": 1,
        "classification": classification,
        "method": "integrated_gradients",
        "second_method_executed": False,
        "checkpoint_sha256_before": checkpoint_hash_before,
        "checkpoint_sha256_after": checkpoint_hash_after,
        "model_parameter_sha256_before": parameter_hash_before,
        "model_parameter_sha256_after": parameter_hash_after,
        "trajectory_sha256": sha256(trajectory_path),
        "sample_count": int(sample_index.size),
        "episode_count": int(np.unique(trajectory["episode"]).size),
        "seeds": [int(value) for value in np.unique(trajectory["seed"])],
        "sample_stride": stride,
        "path_steps": int(config["integrated_gradients"]["path_steps"]),
        "baseline": config["integrated_gradients"]["baseline"],
        "stored_privileged_truth": False,
        "completeness": completeness_metrics,
        "completeness_pass": completeness_pass,
        "stability": stability,
        "faithfulness": faithfulness_metrics,
        "faithfulness_pass": faithfulness_pass,
        "action_saturation": {
            "v_fraction": float(np.mean(np.abs(trajectory["raw_action"][:, 0]) >= 1.0)),
            "omega_fraction": float(np.mean(np.abs(trajectory["raw_action"][:, 1]) >= 1.0)),
        },
        "top_features": top_features_by_target(feature_rows, count=8),
        "top_groups": top_groups_by_target(group_rows),
    }
    write_json(output / "integrated_gradients_metrics.json", metrics)
    make_plots(
        output,
        config,
        trajectory,
        sample_index,
        attributions,
        target_names,
        feature_rows,
        group_rows,
        faithfulness_rows,
    )
    write_report(output, config, metrics)
    verify_artifacts(config, config_path, output)


def faithfulness_summary(
    model: torch.nn.Module,
    observations: torch.Tensor,
    baselines: torch.Tensor,
    attributions: np.ndarray,
    config: dict[str, Any],
    limits: PPOActionLimits,
) -> tuple[list[dict[str, object]], dict[str, dict[str, float]]]:
    fractions = np.asarray(config["faithfulness"]["fractions"], dtype=float)
    repetitions = int(config["faithfulness"]["random_repetitions"])
    rng = np.random.default_rng(int(config["faithfulness"]["random_seed"]))
    dimensions = observations.shape[1]
    rows: list[dict[str, object]] = []
    metrics: dict[str, dict[str, float]] = {}
    spans = {"v_cmd_mps": limits.maximum_linear_velocity_mps, "omega_cmd_rad_s": 2.0 * limits.maximum_angular_velocity_rad_s}
    target_to_index = {
        target: index
        for index, target in enumerate(config["integrated_gradients"]["targets"])
    }
    for target in config["faithfulness"]["targets"]:
        target_index = target_to_index[target]
        importance = np.abs(attributions[target_index])
        top_order = np.argsort(-importance, axis=1, kind="stable")
        bottom_order = np.argsort(importance, axis=1, kind="stable")
        with torch.no_grad():
            original = target_values(
                model, observations, target=target, action_limits=limits
            ).cpu().numpy()
        curves: dict[str, list[float]] = {"top": [], "bottom": [], "random": []}
        for fraction in fractions:
            count = min(dimensions, max(1, int(np.ceil(float(fraction) * dimensions))))
            for strategy, order in (("top", top_order), ("bottom", bottom_order)):
                deviation = masked_deviation(
                    model, observations, baselines, original, order[:, :count], target, limits
                ) / spans[target]
                curves[strategy].append(float(np.mean(deviation)))
                rows.append(summary_row(target, strategy, fraction, deviation, repetitions=1))
            random_values = []
            for _ in range(repetitions):
                random_order = np.argsort(
                    rng.random((observations.shape[0], dimensions)), axis=1
                )[:, :count]
                random_values.append(
                    masked_deviation(
                        model,
                        observations,
                        baselines,
                        original,
                        random_order,
                        target,
                        limits,
                    )
                    / spans[target]
                )
            random_deviation = np.mean(np.asarray(random_values), axis=0)
            curves["random"].append(float(np.mean(random_deviation)))
            rows.append(
                summary_row(
                    target, "random", fraction, random_deviation, repetitions=repetitions
                )
            )
        x = np.concatenate(([0.0], fractions))
        aucs = {
            strategy: float(np.trapz(np.concatenate(([0.0], values)), x))
            for strategy, values in curves.items()
        }
        metrics[target] = {
            "top_auc": aucs["top"],
            "random_auc": aucs["random"],
            "bottom_auc": aucs["bottom"],
            "top_to_random_ratio": aucs["top"] / max(aucs["random"], 1.0e-12),
        }
    return rows, metrics


def masked_deviation(
    model: torch.nn.Module,
    observations: torch.Tensor,
    baselines: torch.Tensor,
    original: np.ndarray,
    indices: np.ndarray,
    target: str,
    limits: PPOActionLimits,
) -> np.ndarray:
    masked = observations.clone()
    row_index = torch.arange(masked.shape[0], device=masked.device).unsqueeze(1)
    column_index = torch.as_tensor(indices, dtype=torch.long, device=masked.device)
    masked[row_index, column_index] = baselines[row_index, column_index]
    with torch.no_grad():
        value = target_values(
            model, masked, target=target, action_limits=limits
        ).cpu().numpy()
    return np.abs(value - original)


def feature_summary(
    attributions: np.ndarray,
    target_names: tuple[str, ...],
    feature_names: tuple[str, ...],
    groups: dict[str, tuple[int, ...]],
) -> list[dict[str, object]]:
    group_for_index = {
        index: group for group, indices in groups.items() for index in indices
    }
    rows: list[dict[str, object]] = []
    for target_index, target in enumerate(target_names):
        values = attributions[target_index]
        absolute = np.abs(values)
        total = float(np.sum(np.mean(absolute, axis=0)))
        top_five = np.argsort(-absolute, axis=1)[:, :5]
        for feature_index, feature in enumerate(feature_names):
            rows.append(
                {
                    "target": target,
                    "feature_index": feature_index,
                    "feature": feature,
                    "group": group_for_index[feature_index],
                    "mean_signed_ig": float(np.mean(values[:, feature_index])),
                    "mean_abs_ig": float(np.mean(absolute[:, feature_index])),
                    "median_abs_ig": float(np.median(absolute[:, feature_index])),
                    "attribution_share": float(np.mean(absolute[:, feature_index]) / max(total, 1.0e-12)),
                    "top5_frequency": float(np.mean(np.any(top_five == feature_index, axis=1))),
                }
            )
    return rows


def group_summary(feature_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str], list[dict[str, object]]] = {}
    for row in feature_rows:
        grouped.setdefault((str(row["target"]), str(row["group"])), []).append(row)
    rows = []
    for (target, group), members in grouped.items():
        rows.append(
            {
                "target": target,
                "group": group,
                "mean_abs_ig": float(sum(float(row["mean_abs_ig"]) for row in members)),
                "attribution_share": float(sum(float(row["attribution_share"]) for row in members)),
                "feature_count": len(members),
            }
        )
    return rows


def stability_summary(
    attributions: np.ndarray,
    seeds: np.ndarray,
    steps: np.ndarray,
    target_names: tuple[str, ...],
) -> dict[str, dict[str, float]]:
    result: dict[str, dict[str, float]] = {}
    for target_index, target in enumerate(target_names):
        correlations = []
        for index in range(len(seeds) - 1):
            if seeds[index] != seeds[index + 1] or steps[index + 1] <= steps[index]:
                continue
            correlations.append(
                spearman(np.abs(attributions[target_index, index]), np.abs(attributions[target_index, index + 1]))
            )
        result[target] = distribution_summary(np.asarray(correlations, dtype=float))
    return result


def resolve_groups(
    config: dict[str, Any], feature_names: tuple[str, ...]
) -> dict[str, tuple[int, ...]]:
    groups = {
        group: tuple(feature_names.index(name) for name in names)
        for group, names in config["feature_groups"].items()
    }
    flattened = [index for indices in groups.values() for index in indices]
    if sorted(flattened) != list(range(len(feature_names))):
        raise ValueError("feature groups must cover every feature exactly once")
    return groups


def make_plots(
    output: Path,
    config: dict[str, Any],
    trajectory: dict[str, np.ndarray],
    sample_index: np.ndarray,
    attributions: np.ndarray,
    target_names: tuple[str, ...],
    feature_rows: list[dict[str, object]],
    group_rows: list[dict[str, object]],
    faithfulness_rows: list[dict[str, object]],
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans"],
            "font.size": 9,
            "axes.labelsize": 10,
            "axes.titlesize": 11,
            "legend.fontsize": 8,
        }
    )
    qualitative_seed = int(config["data"]["qualitative_seed"])
    seed_mask = trajectory["seed"][sample_index] == qualitative_seed
    if not np.any(seed_mask):
        raise RuntimeError("pre-registered qualitative seed is absent")
    steps = trajectory["step"][sample_index][seed_mask]
    selected_rows = sample_index[seed_mask]
    actions = trajectory["physical_action"][selected_rows]
    physical = trajectory["physical_observation"][selected_rows]
    name_to_index = {
        str(name): index for index, name in enumerate(trajectory["feature_names"])
    }

    figure, axes = plt.subplots(
        4, 1, figsize=(14.0, 13.0), sharex=True,
        gridspec_kw={"height_ratios": [1.0, 4.0, 4.0, 1.2]},
    )
    axes[0].plot(steps, actions[:, 0], color="#0072B2", label="v_cmd (m/s)")
    action_axis = axes[0].twinx()
    action_axis.plot(steps, actions[:, 1], color="#D55E00", label="omega_cmd (rad/s)")
    axes[0].set_ylabel("v_cmd")
    action_axis.set_ylabel("omega_cmd")
    handles = axes[0].lines + action_axis.lines
    axes[0].legend(handles, [line.get_label() for line in handles], loc="upper right")

    for axis, target in zip(axes[1:3], ("v_cmd_mps", "omega_cmd_rad_s"), strict=True):
        target_index = target_names.index(target)
        matrix = attributions[target_index, seed_mask].T
        limit = float(np.quantile(np.abs(matrix), 0.99))
        limit = max(limit, 1.0e-8)
        image = axis.imshow(
            matrix,
            aspect="auto",
            origin="upper",
            cmap="RdBu_r",
            vmin=-limit,
            vmax=limit,
            extent=(float(steps[0]), float(steps[-1]), len(SHORT_LABELS) - 0.5, -0.5),
        )
        axis.set_yticks(np.arange(len(SHORT_LABELS)))
        axis.set_yticklabels(SHORT_LABELS, fontsize=7)
        axis.set_ylabel("29D belief feature")
        axis.set_title(f"Signed Integrated Gradients for {target}")
        figure.colorbar(image, ax=axis, pad=0.01, label="attribution")

    axes[3].plot(
        steps,
        physical[:, name_to_index["pedestrian_existence_probability"]],
        color="#009E73",
        label="P(pedestrian exists)",
    )
    axes[3].plot(
        steps,
        physical[:, name_to_index["stop_sign_existence_probability"]],
        color="#CC79A7",
        label="P(stop sign exists)",
    )
    axes[3].plot(
        steps,
        physical[:, name_to_index["stop_mode_required"]],
        color="#E69F00",
        linestyle="--",
        label="stop REQUIRED",
    )
    axes[3].plot(
        steps,
        physical[:, name_to_index["stop_mode_satisfied"]],
        color="#56B4E9",
        linestyle=":",
        label="stop SATISFIED",
    )
    axes[3].set_ylim(-0.05, 1.05)
    axes[3].set_ylabel("public belief")
    axes[3].set_xlabel("simulator step")
    axes[3].legend(ncol=4, loc="upper center")
    figure.suptitle(
        f"Frozen C4 PPO Integrated Gradients — pre-registered seed {qualitative_seed}"
    )
    figure.tight_layout()
    save_figure(figure, output / "ig_qualitative_timeline")

    group_names = list(config["feature_groups"])
    colors = ["#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7", "#56B4E9"]
    figure, axes = plt.subplots(1, 3, figsize=(14.0, 4.6), sharey=True)
    for axis, target in zip(axes, target_names, strict=True):
        values = [
            next(
                float(row["attribution_share"])
                for row in group_rows
                if row["target"] == target and row["group"] == group
            )
            for group in group_names
        ]
        axis.barh(group_names, values, color=colors[: len(group_names)])
        axis.invert_yaxis()
        axis.set_title(target)
        axis.set_xlabel("mean |IG| share")
        axis.set_xlim(0.0, max(0.5, max(values) * 1.15))
    axes[0].set_ylabel("semantic feature group")
    figure.suptitle("Global PPO attribution share by semantic group")
    figure.tight_layout()
    save_figure(figure, output / "ig_group_importance")

    figure, axes = plt.subplots(1, 2, figsize=(10.5, 4.3), sharey=True)
    for axis, target in zip(axes, ("v_cmd_mps", "omega_cmd_rad_s"), strict=True):
        for strategy, color, linestyle in (
            ("top", "#0072B2", "-"),
            ("random", "#E69F00", "--"),
            ("bottom", "#009E73", ":"),
        ):
            selected = sorted(
                (
                    row for row in faithfulness_rows
                    if row["target"] == target and row["strategy"] == strategy
                ),
                key=lambda row: float(row["fraction"]),
            )
            axis.plot(
                [float(row["fraction"]) for row in selected],
                [float(row["mean_normalized_action_change"]) for row in selected],
                marker="o",
                color=color,
                linestyle=linestyle,
                label=strategy,
            )
        axis.set_title(target)
        axis.set_xlabel("fraction of features reset to baseline")
        axis.grid(alpha=0.25)
    axes[0].set_ylabel("mean absolute action change / output span")
    axes[1].legend(title="replacement ranking")
    figure.suptitle("Integrated Gradients deletion faithfulness")
    figure.tight_layout()
    save_figure(figure, output / "ig_faithfulness")


def write_report(output: Path, config: dict[str, Any], metrics: dict[str, Any]) -> None:
    test_summary = parse_test_summary(output / "full_tests.log")
    lines = [
        "# F11 Integrated Gradients Report for Review",
        "",
        f"Classification: **{metrics['classification']}**",
        "",
        "This report explains the frozen C4 PPO checkpoint only. No policy,",
        "perception, belief, or normalization parameter was updated.",
        "",
        "## Provenance",
        "",
        f"- Checkpoint SHA256 before/after: `{metrics['checkpoint_sha256_before']}` / `{metrics['checkpoint_sha256_after']}`.",
        f"- Explanation seeds: `{metrics['seeds']}` (`{metrics['episode_count']}` episodes).",
        f"- Sampled frames: `{metrics['sample_count']}` at stride `{metrics['sample_stride']}`.",
        f"- IG path intervals: `{metrics['path_steps']}`.",
        f"- Baseline: `{metrics['baseline']}`.",
        "- Privileged truth stored: `false`.",
        "",
        "## Quantitative results",
        "",
    ]
    for target, values in metrics["completeness"].items():
        lines.append(
            f"- `{target}` completeness |delta|: median `{values['median']:.3e}`, P99 `{values['p99']:.3e}`."
        )
    lines.extend(["", "Faithfulness AUC (larger action change is better for top-ranked deletion):", ""])
    for target, values in metrics["faithfulness"].items():
        lines.append(
            f"- `{target}`: top `{values['top_auc']:.6f}`, random `{values['random_auc']:.6f}`, bottom `{values['bottom_auc']:.6f}`, top/random `{values['top_to_random_ratio']:.3f}`."
        )
    lines.extend(["", "Top features by mean absolute IG:", ""])
    for target, rows in metrics["top_features"].items():
        rendered = ", ".join(
            f"{row['feature']} ({row['attribution_share']:.1%})" for row in rows[:5]
        )
        lines.append(f"- `{target}`: {rendered}.")
    lines.extend(["", "Top semantic groups by mean absolute IG:", ""])
    for target, rows in metrics["top_groups"].items():
        rendered = ", ".join(
            f"{row['group']} ({row['attribution_share']:.1%})" for row in rows[:3]
        )
        lines.append(f"- `{target}`: {rendered}.")
    lines.extend(["", "Adjacent sampled-frame attribution-rank stability:", ""])
    for target, values in metrics["stability"].items():
        lines.append(
            f"- `{target}`: median Spearman `{values['median']:.4f}`, mean `{values['mean']:.4f}`."
        )
    if metrics["classification"] == "LIMITED":
        lines.extend(
            [
                "",
                "## Why the result is LIMITED",
                "",
                "IG-ranked deletion improves velocity action faithfulness over the",
                "random control, but it does not improve yaw-rate faithfulness over",
                "random on the aggregate AUC. Steering attributions are therefore",
                "useful descriptive sensitivities, not a validated causal ranking.",
            ]
        )
    lines.extend(
        [
            "",
            "## Interpretation boundary",
            "",
            "Integrated Gradients explains sensitivity of the frozen MLP relative",
            "to each episode's public reset observation. It does not by itself",
            "establish closed-loop causality. The separately registered semantic",
            "counterfactual belief-ablation method remains unexecuted.",
            "",
            "## Tests",
            "",
            f"- Full active suite: `{test_summary}`.",
            "",
        ]
    )
    (ROOT / "docs" / "F11_IG_REPORT_FOR_REVIEW.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )


def parse_test_summary(path: Path) -> str:
    if not path.is_file():
        return "not run"
    text = path.read_text(encoding="utf-8", errors="replace")
    matches = re.findall(r"(\d+) passed(?:, (\d+) failed)?(?:, (\d+) skipped)?", text)
    if not matches:
        return "unparseable"
    passed, failed, skipped = matches[-1]
    return f"{passed} passed, {failed or '0'} failed, {skipped or '0'} skipped"


def verify_artifacts(config: dict[str, Any], config_path: Path, output: Path) -> None:
    frozen = config["frozen_policy"]
    verify_hash(resolve_path(config_path, frozen["config"]), frozen["config_sha256"])
    verify_hash(resolve_path(config_path, frozen["checkpoint"]), frozen["checkpoint_sha256"])
    required = (
        "policy_trajectory.npz", "trajectory_manifest.json",
        "integrated_gradients.npz", "integrated_gradients_metrics.json",
        "feature_attribution_summary.csv", "group_attribution_summary.csv",
        "faithfulness.csv", "ig_qualitative_timeline.png",
        "ig_qualitative_timeline.pdf", "ig_group_importance.png",
        "ig_group_importance.pdf", "ig_faithfulness.png", "ig_faithfulness.pdf",
    )
    missing = [name for name in required if not (output / name).is_file()]
    if missing:
        raise FileNotFoundError(f"missing F11 artifacts: {missing}")
    trajectory = dict(np.load(output / "policy_trajectory.npz", allow_pickle=False))
    validate_trajectory_schema(trajectory)
    metrics = json.loads((output / "integrated_gradients_metrics.json").read_text())
    if metrics["checkpoint_sha256_before"] != metrics["checkpoint_sha256_after"]:
        raise RuntimeError("checkpoint hash changed across explanation")
    if metrics["model_parameter_sha256_before"] != metrics["model_parameter_sha256_after"]:
        raise RuntimeError("model parameters changed across explanation")
    print(json.dumps({"classification": metrics["classification"], "artifacts_verified": len(required)}))


def validate_trajectory_schema(arrays: dict[str, np.ndarray]) -> None:
    keys = tuple(arrays)
    for key in keys:
        lowered = key.lower()
        if any(token in lowered for token in FORBIDDEN_TRAJECTORY_TOKENS):
            raise ValueError(f"privileged/evaluation field is forbidden: {key}")
    required = {
        "seed", "episode", "step", "scenario", "pedestrian_mode", "observation",
        "baseline", "physical_observation", "raw_action", "environment_action",
        "physical_action", "critic_value", "terminated", "truncated", "feature_names",
    }
    if set(keys) != required:
        raise ValueError(f"trajectory schema mismatch: {sorted(set(keys) ^ required)}")
    rows = arrays["observation"].shape[0]
    if arrays["observation"].ndim != 2 or arrays["observation"].shape[1] != 29:
        raise ValueError("trajectory observations must have shape (N, 29)")
    if arrays["baseline"].shape != arrays["observation"].shape:
        raise ValueError("trajectory baselines must match observations")
    for key, value in arrays.items():
        if key == "feature_names":
            if value.shape != (29,):
                raise ValueError("feature_names must have shape (29,)")
            continue
        if value.shape[0] != rows:
            raise ValueError(f"trajectory row count mismatch for {key}")
        if np.issubdtype(value.dtype, np.number) and not np.all(np.isfinite(value)):
            raise ValueError(f"trajectory contains non-finite values in {key}")


def _validate_policy_vector(
    observation: np.ndarray, baseline: np.ndarray, order: tuple[str, ...]
) -> None:
    if np.asarray(observation).shape != (len(order),):
        raise ValueError("policy observation has the wrong dimension")
    if not np.all(np.isfinite(observation)) or not np.all(np.isfinite(baseline)):
        raise ValueError("policy observation must be finite")


def summary_row(
    target: str,
    strategy: str,
    fraction: float,
    values: np.ndarray,
    *,
    repetitions: int,
) -> dict[str, object]:
    return {
        "target": target,
        "strategy": strategy,
        "fraction": float(fraction),
        "samples": int(values.size),
        "random_repetitions": repetitions,
        "mean_normalized_action_change": float(np.mean(values)),
        "std_normalized_action_change": float(np.std(values, ddof=1)) if values.size > 1 else 0.0,
        "median_normalized_action_change": float(np.median(values)),
    }


def distribution_summary(values: np.ndarray) -> dict[str, float | int]:
    array = np.asarray(values, dtype=float)
    if array.size == 0:
        return {"n": 0, "mean": float("nan"), "std": float("nan"), "median": float("nan"), "p01": float("nan"), "p99": float("nan")}
    return {
        "n": int(array.size),
        "mean": float(np.mean(array)),
        "std": float(np.std(array, ddof=1)) if array.size > 1 else 0.0,
        "median": float(np.median(array)),
        "p01": float(np.quantile(array, 0.01)),
        "p99": float(np.quantile(array, 0.99)),
    }


def spearman(left: np.ndarray, right: np.ndarray) -> float:
    left_rank = average_ranks(np.asarray(left, dtype=float))
    right_rank = average_ranks(np.asarray(right, dtype=float))
    if np.std(left_rank) <= 1.0e-12 or np.std(right_rank) <= 1.0e-12:
        return 0.0
    return float(np.corrcoef(left_rank, right_rank)[0, 1])


def average_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    ranks = np.empty(values.size, dtype=float)
    start = 0
    while start < values.size:
        stop = start + 1
        while stop < values.size and sorted_values[stop] == sorted_values[start]:
            stop += 1
        ranks[order[start:stop]] = 0.5 * (start + stop - 1)
        start = stop
    return ranks


def top_features_by_target(
    rows: list[dict[str, object]], *, count: int
) -> dict[str, list[dict[str, object]]]:
    targets = sorted({str(row["target"]) for row in rows})
    return {
        target: sorted(
            (row for row in rows if row["target"] == target),
            key=lambda row: float(row["mean_abs_ig"]),
            reverse=True,
        )[:count]
        for target in targets
    }


def top_groups_by_target(rows: list[dict[str, object]]) -> dict[str, list[dict[str, object]]]:
    targets = sorted({str(row["target"]) for row in rows})
    return {
        target: sorted(
            (row for row in rows if row["target"] == target),
            key=lambda row: float(row["attribution_share"]),
            reverse=True,
        )
        for target in targets
    }


def model_parameter_sha256(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        array = tensor.detach().cpu().contiguous().numpy()
        digest.update(name.encode("utf-8"))
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
        digest.update(array.tobytes())
    return digest.hexdigest()


def save_figure(figure: Any, stem: Path) -> None:
    figure.savefig(stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
    figure.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    import matplotlib.pyplot as plt

    plt.close(figure)


def write_csv(path: Path, rows: Iterable[dict[str, object]]) -> None:
    materialized = list(rows)
    if not materialized:
        raise ValueError(f"cannot write empty CSV: {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(materialized[0]))
        writer.writeheader()
        writer.writerows(materialized)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def resolve_path(config_path: Path, value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path.resolve()
    return (config_path.parent / path).resolve()


def verify_hash(path: Path, expected: str) -> str:
    actual = sha256(path)
    if actual.lower() != str(expected).lower():
        raise RuntimeError(f"SHA256 mismatch for {path}: {actual} != {expected}")
    return actual


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


if __name__ == "__main__":
    main()
