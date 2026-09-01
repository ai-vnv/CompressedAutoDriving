"""Train one frozen stage of the belief-conditioned PPO curriculum."""

from __future__ import annotations

import argparse
import csv
import json
import platform
from dataclasses import asdict, replace
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import numpy as np
import torch

from duckie_pomdp.control import (
    PPOAgent,
    PPOConfig,
    PPOCurriculumEnvironment,
    PPORolloutBuffer,
    load_ppo_curriculum_protocol,
    protocol_artifact_root,
    require_curriculum_transition,
    require_pretraining_gate,
    require_stage_in_protocol_scope,
)
from duckie_pomdp.control.f10_protocol import file_sha256


ROOT = Path(__file__).resolve().parents[1]


def _version(name: str) -> str | None:
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def _ppo_config(protocol, stage_key: str, *, smoke: bool) -> PPOConfig:
    settings = protocol.ppo
    config = PPOConfig(
        observation_dimension=len(protocol.observation_order),
        action_dimension=2,
        hidden_sizes=settings.hidden_sizes,
        learning_rate=settings.learning_rate,
        n_steps=settings.n_steps,
        batch_size=settings.batch_size,
        n_epochs=settings.n_epochs,
        gamma=settings.gamma,
        gae_lambda=settings.gae_lambda,
        clip_range=settings.clip_range,
        entropy_coefficient=settings.entropy_coefficient,
        value_function_coefficient=settings.value_function_coefficient,
        max_gradient_norm=settings.max_gradient_norm,
        initial_log_std=settings.initial_log_std,
        seed=settings.training_seed,
        device=settings.device,
    )
    overrides = dict(
        protocol.raw.get("ppo_stage_overrides", {}).get(stage_key, {})
    )
    allowed = {
        "learning_rate",
        "n_epochs",
        "clip_range",
        "entropy_coefficient",
        "max_gradient_norm",
        "target_kl",
    }
    unknown = set(overrides) - allowed
    if unknown:
        raise ValueError(f"unsupported PPO stage overrides: {sorted(unknown)}")
    if overrides:
        typed = {
            name: (
                int(value)
                if name == "n_epochs"
                else None
                if name == "target_kl" and value is None
                else float(value)
            )
            for name, value in overrides.items()
        }
        config = replace(config, **typed)
    if config.target_kl is not None and config.target_kl <= 0.0:
        raise ValueError("target_kl must be positive when configured")
    return replace(config, n_steps=64, batch_size=32, n_epochs=2) if smoke else config


def _new_episode(info: dict[str, Any]) -> dict[str, Any]:
    return {
        "seed": info["seed"],
        "scenario": info["scenario"],
        "pedestrian_mode": info.get("pedestrian_mode"),
        "pedestrian_speed_mps": info.get("pedestrian_speed_mps"),
        "pedestrian_training_phase": info.get("pedestrian_training_phase"),
        "pedestrian_start_delay_s": info.get("pedestrian_start_delay_s"),
        "c2_rehearsal_no_pedestrian": bool(
            info.get("c2_rehearsal_no_pedestrian", False)
        ),
        "length": 0,
        "return": 0.0,
        "reward_progress": 0.0,
        "reward_lane": 0.0,
        "reward_pedestrian": 0.0,
        "reward_stop": 0.0,
        "reward_smoothness": 0.0,
        "reward_terminal": 0.0,
        "progress_m": 0.0,
        "completed": False,
        "collision": False,
        "unsafe_events": 0,
        "minimum_clearance_m": None,
        "stop_completed": False,
        "stop_violation": False,
        "lane_failure": False,
        "yellow_contact_steps": 0,
        "yellow_recovery_events": 0,
        "yellow_recovery_successes": 0,
        "yellow_recovery_failures": 0,
        "invalid_pose": False,
        "sum_v_cmd": 0.0,
        "sum_abs_omega_cmd": 0.0,
        "sum_action_change": 0.0,
    }


def _accumulate(episode: dict[str, Any], reward: float, info: dict[str, Any], action_change: float) -> None:
    episode["length"] += 1
    episode["return"] += reward
    for name in (
        "reward_progress", "reward_lane", "reward_pedestrian",
        "reward_stop", "reward_smoothness", "reward_terminal",
    ):
        episode[name] += float(info[name])
    episode["progress_m"] = float(info["progress_m"])
    for name in ("completed", "collision", "stop_completed", "stop_violation", "lane_failure", "invalid_pose"):
        episode[name] = episode[name] or bool(info[name])
    episode["yellow_contact_steps"] += int(bool(info["yellow_contact"]))
    episode["yellow_recovery_events"] += int(bool(info["yellow_recovery_started"]))
    episode["yellow_recovery_successes"] += int(bool(info["yellow_recovered"]))
    episode["yellow_recovery_failures"] += int(
        info["termination_reason"] == "yellow_recovery_failed"
    )
    episode["unsafe_events"] += int(bool(info["unsafe_proximity"]))
    clearance = info["pedestrian_clearance_m"]
    if clearance is not None:
        episode["minimum_clearance_m"] = (
            float(clearance)
            if episode["minimum_clearance_m"] is None
            else min(float(clearance), episode["minimum_clearance_m"])
        )
    episode["sum_v_cmd"] += float(info["v_cmd"])
    episode["sum_abs_omega_cmd"] += abs(float(info["omega_cmd"]))
    episode["sum_action_change"] += action_change


def _start_wandb(protocol, stage_key: str, manifest: dict, output: Path, *, smoke: bool, mode: str):
    if mode == "disabled":
        return None
    import wandb

    settings = protocol.raw["wandb"]
    return wandb.init(
        entity=str(settings["entity"]),
        project=str(settings["project"]),
        group=str(settings["group"]),
        job_type="smoke" if smoke else f"train-{stage_key}",
        name=f"f10-ppo-{stage_key}-{'smoke' if smoke else 'train'}-{protocol.ppo.training_seed}",
        config=manifest,
        mode=mode,
        dir=str(output),
        reinit="finish_previous",
    )


def _behavior_warm_start(agent: PPOAgent, protocol, stage_key: str) -> dict[str, Any] | None:
    settings = protocol.raw.get("behavior_warm_start", {}).get(stage_key)
    if not settings or not bool(settings.get("enabled", False)):
        return None
    precomputed_value = settings.get("precomputed_checkpoint")
    if precomputed_value is not None:
        precomputed_path = (
            protocol.config_path.parent / str(precomputed_value)
        ).resolve()
        precomputed_sha = str(settings["precomputed_checkpoint_sha256"])
        if (
            not precomputed_path.is_file()
            or file_sha256(precomputed_path) != precomputed_sha
        ):
            raise RuntimeError("precomputed behavior checkpoint hash mismatch")
        precomputed_agent, precomputed_payload = PPOAgent.load(
            precomputed_path, device=str(agent.device)
        )
        if (
            precomputed_agent.config.observation_dimension
            != agent.config.observation_dimension
            or precomputed_agent.config.action_dimension
            != agent.config.action_dimension
            or precomputed_agent.config.hidden_sizes != agent.config.hidden_sizes
        ):
            raise RuntimeError("precomputed behavior checkpoint architecture mismatch")
        if (
            precomputed_payload.get("stage") != stage_key
            or int(precomputed_payload.get("global_step", -1)) != 0
        ):
            raise RuntimeError("precomputed behavior checkpoint must be stage step zero")
        agent.model.load_state_dict(precomputed_agent.model.state_dict())
        return {
            "mode": "precomputed_teacher_checkpoint",
            "checkpoint": str(precomputed_path),
            "checkpoint_sha256": precomputed_sha,
            "checkpoint_step": 0,
            "student_observation_uses_privileged_truth": False,
            "optimizer_state_retained": False,
        }
    initialization: dict[str, Any] | None = None
    if settings.get("initialize_from_checkpoint") is not None:
        initialization_path = (
            protocol.config_path.parent / str(settings["initialize_from_checkpoint"])
        ).resolve()
        initialization_sha = str(settings["initialize_from_checkpoint_sha256"])
        if (
            not initialization_path.is_file()
            or file_sha256(initialization_path) != initialization_sha
        ):
            raise RuntimeError("behavior initialization checkpoint hash mismatch")
        initialization_agent, initialization_payload = PPOAgent.load(
            initialization_path, device=str(agent.device)
        )
        if (
            initialization_agent.config.observation_dimension
            != agent.config.observation_dimension
            or initialization_agent.config.action_dimension
            != agent.config.action_dimension
        ):
            raise RuntimeError("behavior initialization architecture mismatch")
        agent.model.load_state_dict(initialization_agent.model.state_dict())
        initialization = {
            "path": str(initialization_path),
            "sha256": initialization_sha,
            "stage": initialization_payload["stage"],
            "global_step": int(initialization_payload["global_step"]),
            "optimizer_state_retained": False,
        }
    dataset = (protocol.config_path.parent / str(settings["dataset"])).resolve()
    expected = str(settings["dataset_sha256"])
    if not dataset.is_file() or file_sha256(dataset) != expected:
        raise RuntimeError("behavior warm-start dataset hash mismatch")
    with np.load(dataset) as data:
        observations = np.asarray(data["observations"], dtype=np.float32)
        actions = np.asarray(data["actions"], dtype=np.float32)
        weights = np.asarray(data["weights"], dtype=np.float32)
        value_targets = (
            np.asarray(data["value_targets"], dtype=np.float32)
            if "value_targets" in data
            else None
        )
        value_weights = (
            np.asarray(data["value_weights"], dtype=np.float32)
            if "value_weights" in data
            else None
        )
    if observations.ndim != 2 or observations.shape[1] != len(protocol.observation_order):
        raise ValueError("behavior warm-start observation shape mismatch")
    if actions.shape != (len(observations), 2) or weights.shape != (len(observations),):
        raise ValueError("behavior warm-start action/weight shape mismatch")
    if not all(np.all(np.isfinite(value)) for value in (observations, actions, weights)):
        raise ValueError("behavior warm-start dataset must be finite")
    if np.any(actions < -1.0) or np.any(actions > 1.0) or np.any(weights <= 0.0):
        raise ValueError("behavior warm-start targets/weights are invalid")
    critic_enabled = bool(settings.get("critic_enabled", False))
    if critic_enabled:
        if value_targets is None or value_weights is None:
            raise ValueError("critic warm start requires value targets and weights")
        if value_targets.shape != (len(observations),) or value_weights.shape != (
            len(observations),
        ):
            raise ValueError("critic warm-start target/weight shape mismatch")
        if not np.all(np.isfinite(value_targets)) or not np.all(np.isfinite(value_weights)):
            raise ValueError("critic warm-start targets and weights must be finite")
        if np.any(value_weights < 0.0) or not np.any(value_weights > 0.0):
            raise ValueError("critic warm start requires positive supervised weight")

    device = agent.device
    x = torch.as_tensor(observations, dtype=torch.float32, device=device)
    y = torch.as_tensor(actions, dtype=torch.float32, device=device)
    w = torch.as_tensor(weights, dtype=torch.float32, device=device)
    with torch.no_grad():
        before = float(torch.mean((agent.model.actor(x) - y) ** 2).item())
    optimizer = torch.optim.Adam(
        agent.model.actor.parameters(),
        lr=float(settings["learning_rate"]),
        eps=1.0e-5,
    )
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(settings["seed"]))
    batch_size = int(settings["batch_size"])
    epochs = int(settings["epochs"])
    for _ in range(epochs):
        order = torch.randperm(len(x), generator=generator)
        for start in range(0, len(x), batch_size):
            indices = order[start : start + batch_size].to(device)
            prediction = agent.model.actor(x[indices])
            loss = torch.mean(w[indices, None] * (prediction - y[indices]) ** 2)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(agent.model.actor.parameters(), 1.0)
            optimizer.step()
    with torch.no_grad():
        after = float(torch.mean((agent.model.actor(x) - y) ** 2).item())
    if not np.isfinite(after) or after >= before:
        raise RuntimeError("behavior warm start did not improve actor imitation")
    critic_metrics: dict[str, Any] = {
        "enabled": False,
        "supervised_rows": 0,
        "mse_before": None,
        "mse_after": None,
    }
    if critic_enabled:
        assert value_targets is not None and value_weights is not None
        supervised = value_weights > 0.0
        value_x = torch.as_tensor(
            observations[supervised], dtype=torch.float32, device=device
        )
        value_y = torch.as_tensor(
            value_targets[supervised], dtype=torch.float32, device=device
        )
        value_w = torch.as_tensor(
            value_weights[supervised], dtype=torch.float32, device=device
        )
        value_w = value_w / value_w.mean()
        with torch.no_grad():
            critic_before = float(
                torch.mean((agent.model.value(value_x) - value_y) ** 2).item()
            )
        critic_optimizer = torch.optim.Adam(
            agent.model.critic.parameters(),
            lr=float(settings["critic_learning_rate"]),
            eps=1.0e-5,
        )
        critic_epochs = int(settings["critic_epochs"])
        critic_batch_size = int(settings.get("critic_batch_size", batch_size))
        critic_generator = torch.Generator(device="cpu")
        critic_generator.manual_seed(int(settings["seed"]) + 1)
        for _ in range(critic_epochs):
            order = torch.randperm(len(value_x), generator=critic_generator)
            for start in range(0, len(value_x), critic_batch_size):
                indices = order[start : start + critic_batch_size].to(device)
                prediction = agent.model.value(value_x[indices])
                critic_loss = torch.mean(
                    value_w[indices] * (prediction - value_y[indices]) ** 2
                )
                critic_optimizer.zero_grad(set_to_none=True)
                critic_loss.backward()
                torch.nn.utils.clip_grad_norm_(agent.model.critic.parameters(), 1.0)
                critic_optimizer.step()
        with torch.no_grad():
            critic_after = float(
                torch.mean((agent.model.value(value_x) - value_y) ** 2).item()
            )
        if not np.isfinite(critic_after) or critic_after >= critic_before:
            raise RuntimeError("behavior warm start did not improve critic fit")
        critic_metrics = {
            "enabled": True,
            "supervised_rows": int(np.sum(supervised)),
            "epochs": critic_epochs,
            "batch_size": critic_batch_size,
            "learning_rate": float(settings["critic_learning_rate"]),
            "mse_before": critic_before,
            "mse_after": critic_after,
        }
    return {
        "dataset": str(dataset),
        "dataset_sha256": expected,
        "rows": int(len(x)),
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": float(settings["learning_rate"]),
        "seed": int(settings["seed"]),
        "mse_before": before,
        "mse_after": after,
        "teacher_uses_privileged_truth": bool(
            settings.get("teacher_uses_privileged_truth", False)
        ),
        "student_observation_uses_privileged_truth": False,
        "critic": critic_metrics,
        "initialization": initialization,
    }


def train(
    config_path: Path,
    stage_key: str,
    output_dir: Path,
    *,
    smoke: bool,
    wandb_mode: str,
    source_checkpoint: Path | None,
) -> dict[str, Any]:
    protocol = load_ppo_curriculum_protocol(config_path)
    require_stage_in_protocol_scope(protocol, stage_key)
    stage = protocol.stage(stage_key)
    ppo_config = _ppo_config(protocol, stage_key, smoke=smoke)
    planned_steps = 128 if smoke else stage.training_steps
    if planned_steps % ppo_config.n_steps:
        raise ValueError("training budget must contain whole PPO rollouts")
    if stage_key == "c0" and source_checkpoint is not None:
        raise ValueError("C0 must start from random initialization")
    if stage_key != "c0" and source_checkpoint is None:
        raise ValueError(f"{stage_key} requires the selected previous-stage checkpoint")
    if not smoke:
        require_pretraining_gate(
            protocol, protocol_artifact_root(protocol) / "pretraining_gate.json"
        )
    transition_gate = None
    if stage_key != "c0" and not smoke:
        assert source_checkpoint is not None
        transition_gate = require_curriculum_transition(
            protocol,
            stage_key,
            source_checkpoint,
            protocol_artifact_root(protocol),
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    guarded = ("training_metrics.csv", "episode_metrics.csv", "training_run_manifest.json")
    existing = [output_dir / name for name in guarded if (output_dir / name).exists()]
    if existing:
        raise FileExistsError(f"refusing to overwrite PPO stage artifacts: {existing}")

    source = None
    if source_checkpoint is None:
        agent = PPOAgent(ppo_config)
    else:
        agent, payload = PPOAgent.load(
            source_checkpoint,
            device=ppo_config.device,
            learning_rate=ppo_config.learning_rate,
        )
        if agent.config.observation_dimension != ppo_config.observation_dimension:
            raise RuntimeError("curriculum checkpoint observation dimension changed")
        transition_settings = protocol.raw.get("curriculum_transition", {}).get(
            stage_key, {}
        )
        optimizer_state_retained = not bool(
            transition_settings.get("reset_optimizer", False)
        )
        reset_log_std = transition_settings.get("reset_log_std")
        if not optimizer_state_retained:
            agent.config = ppo_config
            agent.optimizer = torch.optim.Adam(
                agent.model.parameters(), lr=ppo_config.learning_rate, eps=1.0e-5
            )
        if reset_log_std is not None:
            with torch.no_grad():
                agent.model.log_std.fill_(float(reset_log_std))
        source = {
            "path": str(source_checkpoint.resolve()),
            "sha256": file_sha256(source_checkpoint),
            "stage": payload["stage"],
            "global_step": int(payload["global_step"]),
            "optimizer_state_retained": optimizer_state_retained,
            "reset_log_std": None if reset_log_std is None else float(reset_log_std),
        }

    behavior_metrics = _behavior_warm_start(agent, protocol, stage_key)

    config_sha = file_sha256(config_path)
    manifest = {
        "schema_version": 1,
        "stage": stage_key,
        "smoke": smoke,
        "config": str(config_path.resolve()),
        "config_sha256": config_sha,
        "observation_dimension": len(protocol.observation_order),
        "observation_order": list(protocol.observation_order),
        "ppo": asdict(ppo_config),
        "planned_environment_steps": planned_steps,
        "seed_split": {
            "training": list(stage.training_seeds),
            "development": list(stage.development_seeds),
            "stage_final": list(stage.stage_final_seeds),
        },
        "source_checkpoint": source,
        "behavior_warm_start": behavior_metrics,
        "curriculum_transition_gate": transition_gate,
        "upstream": {
            "yolo_sha256": protocol.detector_checkpoint_sha256,
            "belief_config_sha256": protocol.belief_config_sha256,
            "action_config_sha256": protocol.action_config_sha256,
        },
        "versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "gym": _version("gym"),
            "ultralytics": _version("ultralytics"),
            "wandb": _version("wandb"),
            "stable_baselines3": _version("stable-baselines3"),
        },
    }
    (output_dir / "config_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    (output_dir / "normalization.json").write_text(
        json.dumps(
            {
                "type": "fixed_physical_scales",
                "ordering": list(protocol.observation_order),
                "scales": list(protocol.observation_scales),
                "clip": protocol.observation_clip,
                "mutable": False,
            }, indent=2,
        ) + "\n", encoding="utf-8"
    )
    run = _start_wandb(protocol, stage_key, manifest, output_dir, smoke=smoke, mode=wandb_mode)
    env = PPOCurriculumEnvironment(config_path, stage=stage_key, split="training")
    checkpoints = output_dir / "checkpoints"
    checkpoints.mkdir(parents=True, exist_ok=False)
    behavior_checkpoint = None
    if behavior_metrics is not None:
        behavior_checkpoint = checkpoints / f"ppo_{stage_key}_step_{0:07d}.pt"
        agent.save(
            behavior_checkpoint,
            global_step=0,
            stage=stage_key,
            metadata={
                "config_sha256": config_sha,
                "source_checkpoint_sha256": None if source is None else source["sha256"],
                "observation_order": list(protocol.observation_order),
                "smoke": smoke,
                "behavior_warm_start": behavior_metrics,
            },
        )
    checkpoint_interval = planned_steps if smoke else stage.checkpoint_interval_steps
    step_fields = [
        "stage", "global_step", "episode", "seed", "scenario",
        "pedestrian_mode", "pedestrian_speed_mps", "pedestrian_training_phase",
        "pedestrian_start_delay_s", "c2_rehearsal_no_pedestrian", "reward",
        "reward_progress", "reward_lane", "reward_pedestrian", "reward_stop",
        "reward_smoothness", "reward_terminal", "terminated", "truncated",
        "completed", "collision", "unsafe_proximity", "stop_completed",
        "stop_violation", "lane_failure", "yellow_contact",
        "yellow_recovery_started", "yellow_recovery_active", "yellow_recovered",
        "invalid_pose", "v_cmd", "omega_cmd",
        "v_actual", "omega_actual", "action_change", "policy_loss", "value_loss",
        "entropy", "approximate_kl", "clip_fraction", "gradient_norm",
        "explained_variance", "mean_log_std", "update_count",
        "optimization_steps", "early_stopped",
    ] + [f"policy.{name}" for name in protocol.observation_order] + [
        "evaluation_gt.lane_lateral_error_m",
        "evaluation_gt.lane_heading_error_rad",
        "evaluation_gt.road_curvature_inv_m",
        "evaluation_gt.pedestrian_range_m", "evaluation_gt.pedestrian_bearing_rad",
        "evaluation_gt.stop_line_distance_m",
    ]
    episode_fields = [
        "stage", "episode", "global_step", "seed", "scenario",
        "pedestrian_mode", "pedestrian_speed_mps", "pedestrian_training_phase",
        "pedestrian_start_delay_s", "c2_rehearsal_no_pedestrian",
        "length", "return",
        "reward_progress", "reward_lane", "reward_pedestrian", "reward_stop",
        "reward_smoothness", "reward_terminal", "progress_m", "completed",
        "collision", "unsafe_events", "minimum_clearance_m", "stop_completed",
        "stop_violation", "lane_failure", "yellow_contact_steps",
        "yellow_recovery_events", "yellow_recovery_successes",
        "yellow_recovery_failures", "invalid_pose", "timeout",
        "termination_reason", "truncation_reason", "mean_v_cmd",
        "mean_abs_omega_cmd", "mean_action_change",
    ]
    training_path = output_dir / "training_metrics.csv"
    episode_path = output_dir / "episode_metrics.csv"
    observation, reset_info = env.reset()
    episode = _new_episode(reset_info)
    episode_index = 0
    completed_episodes = 0
    previous_physical = np.zeros(2, dtype=np.float32)
    checkpoint_paths: list[Path] = (
        [] if behavior_checkpoint is None else [behavior_checkpoint]
    )
    last_metrics = {name: np.nan for name in (
        "policy_loss", "value_loss", "entropy", "approximate_kl", "clip_fraction",
        "gradient_norm", "explained_variance", "mean_log_std", "update_count",
        "optimization_steps", "early_stopped",
    )}
    try:
        with training_path.open("w", newline="", encoding="utf-8") as step_stream, episode_path.open("w", newline="", encoding="utf-8") as episode_stream:
            step_writer = csv.DictWriter(step_stream, fieldnames=step_fields)
            episode_writer = csv.DictWriter(episode_stream, fieldnames=episode_fields)
            step_writer.writeheader()
            episode_writer.writeheader()
            global_step = 0
            while global_step < planned_steps:
                rollout = PPORolloutBuffer(
                    ppo_config.n_steps,
                    ppo_config.observation_dimension,
                    ppo_config.action_dimension,
                )
                pending_rows: list[dict[str, Any]] = []
                for _ in range(ppo_config.n_steps):
                    global_step += 1
                    sampled = agent.act(observation, deterministic=False)
                    next_observation, reward, terminated, truncated, info = env.step(sampled.environment_action)
                    next_value = agent.value(next_observation)
                    done = terminated or truncated
                    rollout.add(
                        observation,
                        sampled.raw_action,
                        sampled.log_probability,
                        sampled.value,
                        reward,
                        next_value,
                        terminated=terminated,
                        episode_done=done,
                    )
                    physical = np.asarray((info["v_cmd"], info["omega_cmd"]), dtype=np.float32)
                    change = float(np.linalg.norm(np.asarray(
                        ((physical[0] - previous_physical[0]) / 0.4, (physical[1] - previous_physical[1]) / 4.0),
                        dtype=np.float32,
                    )))
                    _accumulate(episode, reward, info, change)
                    row = {
                        "stage": stage_key, "global_step": global_step, "episode": episode_index,
                        "seed": episode["seed"], "scenario": episode["scenario"],
                        "pedestrian_mode": episode["pedestrian_mode"],
                        "pedestrian_speed_mps": episode["pedestrian_speed_mps"],
                        "pedestrian_training_phase": episode["pedestrian_training_phase"],
                        "pedestrian_start_delay_s": episode["pedestrian_start_delay_s"],
                        "c2_rehearsal_no_pedestrian": episode["c2_rehearsal_no_pedestrian"],
                        "reward": reward,
                        **{name: info[name] for name in (
                            "reward_progress", "reward_lane", "reward_pedestrian", "reward_stop",
                            "reward_smoothness", "reward_terminal", "completed", "collision",
                            "unsafe_proximity", "stop_completed", "stop_violation", "lane_failure",
                            "yellow_contact", "yellow_recovery_started",
                            "yellow_recovery_active", "yellow_recovered", "invalid_pose",
                            "v_cmd", "omega_cmd", "v_actual", "omega_actual",
                        )},
                        "terminated": terminated, "truncated": truncated, "action_change": change,
                    }
                    row.update({f"policy.{name}": info["policy"][name] for name in protocol.observation_order})
                    row.update({f"evaluation_gt.{name}": value for name, value in info["evaluation_gt"].items()})
                    pending_rows.append(row)
                    if done:
                        length = episode["length"]
                        episode_writer.writerow({
                            "stage": stage_key, "episode": episode_index, "global_step": global_step,
                            "pedestrian_mode": episode["pedestrian_mode"],
                            "pedestrian_speed_mps": episode["pedestrian_speed_mps"],
                            "pedestrian_training_phase": episode["pedestrian_training_phase"],
                            "pedestrian_start_delay_s": episode["pedestrian_start_delay_s"],
                            "c2_rehearsal_no_pedestrian": episode["c2_rehearsal_no_pedestrian"],
                            **{name: episode[name] for name in (
                                "seed", "scenario", "length", "return", "reward_progress", "reward_lane",
                                "reward_pedestrian", "reward_stop", "reward_smoothness", "reward_terminal",
                                "progress_m", "completed", "collision", "unsafe_events", "minimum_clearance_m",
                                "stop_completed", "stop_violation", "lane_failure", "invalid_pose",
                                "yellow_contact_steps", "yellow_recovery_events",
                                "yellow_recovery_successes", "yellow_recovery_failures",
                            )},
                            "timeout": truncated,
                            "termination_reason": info["termination_reason"],
                            "truncation_reason": info["truncation_reason"],
                            "mean_v_cmd": episode["sum_v_cmd"] / length,
                            "mean_abs_omega_cmd": episode["sum_abs_omega_cmd"] / length,
                            "mean_action_change": episode["sum_action_change"] / length,
                        })
                        if run is not None:
                            run.log({
                                "episode/return": episode["return"],
                                "episode/progress_m": episode["progress_m"],
                                "episode/completed": int(episode["completed"]),
                                "episode/collision": int(episode["collision"]),
                                "episode/stop_completed": int(episode["stop_completed"]),
                                "episode/stop_violation": int(episode["stop_violation"]),
                                "episode/lane_failure": int(episode["lane_failure"]),
                                "episode/yellow_contact_steps": episode["yellow_contact_steps"],
                                "episode/yellow_recovery_events": episode["yellow_recovery_events"],
                                "episode/yellow_recovery_successes": episode["yellow_recovery_successes"],
                                "episode/yellow_recovery_failures": episode["yellow_recovery_failures"],
                            }, step=global_step)
                        completed_episodes += 1
                        episode_index += 1
                        observation, reset_info = env.reset()
                        episode = _new_episode(reset_info)
                        previous_physical = np.zeros(2, dtype=np.float32)
                    else:
                        observation = next_observation
                        previous_physical = physical

                last_metrics = agent.update(rollout)
                for row in pending_rows:
                    row.update(last_metrics)
                    step_writer.writerow(row)
                if run is not None:
                    run.log({
                        "train/reward_step": pending_rows[-1]["reward"],
                        "train/episodes": completed_episodes,
                        **{f"ppo/{name}": value for name, value in last_metrics.items()},
                    }, step=global_step)
                if global_step % checkpoint_interval == 0:
                    checkpoint = checkpoints / f"ppo_{stage_key}_step_{global_step:07d}.pt"
                    agent.save(
                        checkpoint,
                        global_step=global_step,
                        stage=stage_key,
                        metadata={
                            "config_sha256": config_sha,
                            "source_checkpoint_sha256": None if source is None else source["sha256"],
                            "observation_order": list(protocol.observation_order),
                            "smoke": smoke,
                        },
                    )
                    checkpoint_paths.append(checkpoint)
                step_stream.flush()
                episode_stream.flush()
                print(json.dumps({
                    "stage": stage_key, "global_step": global_step,
                    "episodes": completed_episodes, "updates": agent.update_count,
                    **last_metrics,
                }), flush=True)
    finally:
        env.close()
        if run is not None:
            run.finish()
    if not checkpoint_paths:
        raise RuntimeError("PPO training produced no checkpoint")
    probe = np.zeros(ppo_config.observation_dimension, dtype=np.float32)
    before = agent.act(probe, deterministic=True).environment_action
    loaded, payload = PPOAgent.load(checkpoint_paths[-1], device=ppo_config.device)
    after = loaded.act(probe, deterministic=True).environment_action
    reload_verified = bool(np.array_equal(before, after))
    if not reload_verified or agent.update_count <= 0:
        raise RuntimeError("PPO gradient/checkpoint smoke invariant failed")
    result = {
        "schema_version": 1,
        "stage": stage_key,
        "smoke": smoke,
        "config_sha256": config_sha,
        "environment_steps": planned_steps,
        "ppo_updates_total": agent.update_count,
        "completed_episodes": completed_episodes,
        "checkpoint_reload_verified": reload_verified,
        "checkpoints": [
            {"path": str(path.resolve()), "sha256": file_sha256(path), "global_step": int(PPOAgent.load(path, device="cpu")[1]["global_step"])}
            for path in checkpoint_paths
        ],
        "source_checkpoint": source,
        "last_update_metrics": last_metrics,
        "training_metrics_sha256": file_sha256(training_path),
        "episode_metrics_sha256": file_sha256(episode_path),
    }
    (output_dir / "training_run_manifest.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("c0", "c1", "c2", "c3", "c4"))
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "f10_ppo_v1.toml")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--source-checkpoint", type=Path)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--wandb-mode", choices=("online", "offline", "disabled"), default="online")
    args = parser.parse_args()
    protocol = load_ppo_curriculum_protocol(args.config.resolve())
    output = args.output_dir or protocol_artifact_root(protocol) / args.stage / (
        "smoke" if args.smoke else "training"
    )
    print(json.dumps(train(
        args.config.resolve(), args.stage, output.resolve(), smoke=args.smoke,
        wandb_mode=args.wandb_mode,
        source_checkpoint=None if args.source_checkpoint is None else args.source_checkpoint.resolve(),
    ), indent=2))


if __name__ == "__main__":
    main()
