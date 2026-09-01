"""Train the single predeclared F10 SAC baseline on real Gym-Duckietown."""

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
    F10GymEnvironment,
    ReplayBuffer,
    SACAgent,
    SACConfig,
    load_f10_protocol,
)
from duckie_pomdp.control.f10_protocol import file_sha256


ROOT = Path(__file__).resolve().parents[1]


def _package_version(name: str) -> str | None:
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def _sac_config(protocol, *, smoke: bool) -> SACConfig:
    settings = protocol.sac
    config = SACConfig(
        observation_dimension=len(protocol.observation_order),
        action_dimension=2,
        hidden_sizes=settings.hidden_sizes,
        learning_rate=settings.learning_rate,
        gamma=settings.gamma,
        tau=settings.tau,
        batch_size=settings.batch_size,
        replay_buffer_size=settings.replay_buffer_size,
        learning_starts=settings.learning_starts,
        train_frequency=settings.train_frequency,
        gradient_steps=settings.gradient_steps,
        initial_entropy_coefficient=settings.initial_entropy_coefficient,
        target_entropy=settings.target_entropy,
        seed=settings.training_seed,
        device=settings.device,
    )
    if smoke:
        return replace(
            config,
            batch_size=32,
            replay_buffer_size=1000,
            learning_starts=32,
        )
    return config


def _manifest(protocol, sac_config: SACConfig, *, smoke: bool, steps: int) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "stage": "F10_SAC_SMOKE" if smoke else "F10_SAC_BASELINE_TRAINING",
        "f10_config": str(protocol.config_path),
        "f10_config_sha256": file_sha256(protocol.config_path),
        "upstream": {
            "yolo_checkpoint": str(protocol.detector_checkpoint_path),
            "yolo_checkpoint_sha256": protocol.detector_checkpoint_sha256,
            "belief_config": str(protocol.belief_config_path),
            "belief_config_sha256": protocol.belief_config_sha256,
            "action_config_sha256": protocol.action_config_sha256,
        },
        "seed_split": asdict(protocol.seeds),
        "sac": asdict(sac_config),
        "environment_steps": steps,
        "smoke_overrides": (
            {"batch_size": 32, "replay_buffer_size": 1000, "learning_starts": 32}
            if smoke
            else None
        ),
        "versions": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "cuda_available": torch.cuda.is_available(),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "gym": _package_version("gym"),
            "gym_duckietown_source": "6.2.0",
            "ultralytics": _package_version("ultralytics"),
            "numpy": np.__version__,
        },
    }


def _new_episode(reset_info: dict[str, Any]) -> dict[str, Any]:
    return {
        "seed": reset_info["seed"],
        "scenario": reset_info["scenario"],
        "episode_length": 0,
        "total_reward": 0.0,
        "reward_progress": 0.0,
        "reward_lane": 0.0,
        "reward_stop": 0.0,
        "reward_pedestrian": 0.0,
        "reward_comfort": 0.0,
        "reward_terminal": 0.0,
        "progress_m": 0.0,
        "collision": False,
        "unsafe_proximity_events": 0,
        "stop_completed": False,
        "stop_violation": False,
        "lane_departure_events": 0,
        "invalid_pose": False,
        "mean_v_cmd": 0.0,
        "mean_abs_omega_cmd": 0.0,
        "mean_action_change": 0.0,
    }


def _accumulate(episode: dict[str, Any], reward: float, info: dict[str, Any], action_change: float) -> None:
    episode["episode_length"] += 1
    episode["total_reward"] += reward
    for name in (
        "reward_progress",
        "reward_lane",
        "reward_stop",
        "reward_pedestrian",
        "reward_comfort",
        "reward_terminal",
    ):
        episode[name] += info[name]
    episode["progress_m"] = info["progress_m"]
    episode["collision"] = episode["collision"] or info["collision"]
    episode["unsafe_proximity_events"] += int(info["unsafe_proximity"])
    episode["stop_completed"] = episode["stop_completed"] or info["stop_completed"]
    episode["stop_violation"] = episode["stop_violation"] or info["stop_violation"]
    episode["lane_departure_events"] += int(info["lane_departure"])
    episode["invalid_pose"] = episode["invalid_pose"] or info["invalid_pose"]
    length = episode["episode_length"]
    for name, value in (
        ("mean_v_cmd", info["v_cmd"]),
        ("mean_abs_omega_cmd", abs(info["omega_cmd"])),
        ("mean_action_change", action_change),
    ):
        episode[name] += (value - episode[name]) / length


def _start_wandb(protocol, manifest: dict[str, Any], output_dir: Path, *, smoke: bool, mode: str):
    if mode == "disabled":
        return None
    try:
        import wandb
    except ModuleNotFoundError as error:
        raise RuntimeError("W&B telemetry requested but wandb is not installed") from error
    settings = protocol.raw["wandb"]
    return wandb.init(
        entity=str(settings["entity"]),
        project=str(settings["project"]),
        group=str(settings["group"]),
        job_type="smoke" if smoke else str(settings["job_type"]),
        name=(
            f"f10-sac-smoke-{protocol.sac.training_seed}"
            if smoke
            else f"f10-sac-baseline-{protocol.sac.training_seed}"
        ),
        config=manifest,
        mode=mode,
        dir=str(output_dir),
        reinit="finish_previous",
    )


def _require_pretraining_gate(protocol, output_dir: Path) -> dict[str, Any]:
    gate_path = output_dir / "pretraining_gate.json"
    if not gate_path.is_file():
        raise RuntimeError(
            "full F10 training requires artifacts/f10/pretraining_gate.json"
        )
    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    if gate.get("ready_for_full_training") is not True:
        raise RuntimeError("F10 pre-training gate is not READY")
    current_config_sha = file_sha256(protocol.config_path)
    if gate.get("f10_config_sha256") != current_config_sha:
        raise RuntimeError("F10 config changed after the pre-training gate")
    expected_source_sha = gate.get("source_sha256", {}).get(
        "experiments/train_f10_sac.py"
    )
    if expected_source_sha != file_sha256(Path(__file__)):
        raise RuntimeError("F10 training implementation changed after the gate")
    return gate


def train(
    config_path: Path,
    output_dir: Path,
    *,
    smoke: bool,
    wandb_mode: str,
) -> dict[str, Any]:
    protocol = load_f10_protocol(config_path)
    planned_steps = 96 if smoke else protocol.sac.training_steps
    sac_config = _sac_config(protocol, smoke=smoke)
    output_dir.mkdir(parents=True, exist_ok=True)
    pretraining_gate = None if smoke else _require_pretraining_gate(protocol, output_dir)
    artifact_names = (
        "training_metrics.csv",
        "episode_metrics.csv",
        "config_manifest.json",
        "normalization.json",
        "training_run_manifest.json",
    )
    existing = [output_dir / name for name in artifact_names if (output_dir / name).exists()]
    if existing:
        raise FileExistsError(f"refusing to overwrite F10 training artifacts: {existing}")

    manifest = _manifest(protocol, sac_config, smoke=smoke, steps=planned_steps)
    if pretraining_gate is not None:
        manifest["pretraining_gate"] = {
            "path": str((output_dir / "pretraining_gate.json").resolve()),
            "sha256": file_sha256(output_dir / "pretraining_gate.json"),
            "verified_at_utc": pretraining_gate["verified_at_utc"],
        }
    (output_dir / "config_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )
    (output_dir / "normalization.json").write_text(
        json.dumps(
            {
                "type": "fixed_physical_scales",
                "ordering": list(protocol.observation_order),
                "scales": list(protocol.observation_scales),
                "clip": protocol.observation_clip,
                "mutable": False,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    wandb_run = _start_wandb(
        protocol,
        manifest,
        output_dir,
        smoke=smoke,
        mode=wandb_mode,
    )
    environment = F10GymEnvironment(config_path, split="training")
    agent = SACAgent(sac_config)
    replay = ReplayBuffer(
        sac_config.replay_buffer_size,
        sac_config.observation_dimension,
        sac_config.action_dimension,
        seed=sac_config.seed + 1,
    )
    action_rng = np.random.default_rng(sac_config.seed + 2)
    training_path = output_dir / "training_metrics.csv"
    episodes_path = output_dir / "episode_metrics.csv"
    checkpoints = output_dir / "checkpoints"
    checkpoints.mkdir(parents=True, exist_ok=False)
    checkpoint_interval = planned_steps if smoke else protocol.sac.checkpoint_interval_steps
    training_fields = [
        "global_step",
        "episode",
        "seed",
        "scenario",
        "reward",
        "reward_progress",
        "reward_lane",
        "reward_stop",
        "reward_pedestrian",
        "reward_comfort",
        "reward_terminal",
        "terminated",
        "truncated",
        "collision",
        "unsafe_proximity",
        "lane_departure",
        "stop_completed",
        "stop_violation",
        "invalid_pose",
        "termination_reason",
        "truncation_reason",
        "v_cmd",
        "omega_cmd",
        "v_actual",
        "omega_actual",
        "action_change",
        "buffer_size",
        "update_count",
        "q1_loss",
        "q2_loss",
        "actor_loss",
        "alpha_loss",
        "entropy_coefficient",
        "mean_log_probability",
        "mean_q",
    ] + [f"observation_normalized_{name}" for name in protocol.observation_order]
    episode_fields = [
        "episode",
        "global_step",
        "seed",
        "scenario",
        "episode_length",
        "total_reward",
        "reward_progress",
        "reward_lane",
        "reward_stop",
        "reward_pedestrian",
        "reward_comfort",
        "reward_terminal",
        "progress_m",
        "collision",
        "unsafe_proximity_events",
        "stop_completed",
        "stop_violation",
        "lane_departure_events",
        "invalid_pose",
        "timeout",
        "termination_reason",
        "truncation_reason",
        "mean_v_cmd",
        "mean_abs_omega_cmd",
        "mean_action_change",
    ]

    episode_index = 0
    completed_episodes = 0
    checkpoint_paths: list[Path] = []
    gradient_updates = 0
    try:
        observation, reset_info = environment.reset()
        episode = _new_episode(reset_info)
        previous_physical_action = np.zeros(2, dtype=np.float32)
        with training_path.open("w", newline="", encoding="utf-8") as training_stream, episodes_path.open("w", newline="", encoding="utf-8") as episode_stream:
            training_writer = csv.DictWriter(training_stream, fieldnames=training_fields)
            episode_writer = csv.DictWriter(episode_stream, fieldnames=episode_fields)
            training_writer.writeheader()
            episode_writer.writeheader()
            for global_step in range(1, planned_steps + 1):
                if global_step <= sac_config.learning_starts:
                    normalized_action = action_rng.uniform(-1.0, 1.0, size=2).astype(np.float32)
                else:
                    normalized_action = agent.act(observation, deterministic=False)
                next_observation, reward, terminated, truncated, info = environment.step(normalized_action)
                replay.add(observation, normalized_action, reward, next_observation, terminated)
                physical_action = np.array([info["v_cmd"], info["omega_cmd"]], dtype=np.float32)
                normalized_change = np.array(
                    [
                        (physical_action[0] - previous_physical_action[0]) / 0.4,
                        (physical_action[1] - previous_physical_action[1]) / 4.0,
                    ],
                    dtype=np.float32,
                )
                action_change = float(np.linalg.norm(normalized_change))
                _accumulate(episode, reward, info, action_change)

                update_metrics = {name: np.nan for name in (
                    "q1_loss", "q2_loss", "actor_loss", "alpha_loss",
                    "entropy_coefficient", "mean_log_probability", "mean_q",
                )}
                if (
                    global_step >= sac_config.learning_starts
                    and len(replay) >= sac_config.batch_size
                    and global_step % sac_config.train_frequency == 0
                ):
                    for _ in range(sac_config.gradient_steps):
                        update_metrics = agent.update(replay)
                        gradient_updates += 1
                row = {
                    "global_step": global_step,
                    "episode": episode_index,
                    "seed": episode["seed"],
                    "scenario": episode["scenario"],
                    "reward": reward,
                    "reward_progress": info["reward_progress"],
                    "reward_lane": info["reward_lane"],
                    "reward_stop": info["reward_stop"],
                    "reward_pedestrian": info["reward_pedestrian"],
                    "reward_comfort": info["reward_comfort"],
                    "reward_terminal": info["reward_terminal"],
                    "terminated": terminated,
                    "truncated": truncated,
                    "collision": info["collision"],
                    "unsafe_proximity": info["unsafe_proximity"],
                    "lane_departure": info["lane_departure"],
                    "stop_completed": info["stop_completed"],
                    "stop_violation": info["stop_violation"],
                    "invalid_pose": info["invalid_pose"],
                    "termination_reason": info["termination_reason"],
                    "truncation_reason": info["truncation_reason"],
                    "v_cmd": info["v_cmd"],
                    "omega_cmd": info["omega_cmd"],
                    "v_actual": info["v_actual"],
                    "omega_actual": info["omega_actual"],
                    "action_change": action_change,
                    "buffer_size": len(replay),
                    "update_count": agent.update_count,
                    **update_metrics,
                }
                row.update(
                    {
                        f"observation_normalized_{name}": float(observation[index])
                        for index, name in enumerate(protocol.observation_order)
                    }
                )
                training_writer.writerow(row)
                if (
                    wandb_run is not None
                    and global_step % int(protocol.raw["wandb"]["log_interval_steps"]) == 0
                ):
                    wandb_run.log(
                        {
                            "train/reward_step": reward,
                            "train/reward_progress": info["reward_progress"],
                            "train/reward_lane": info["reward_lane"],
                            "train/reward_stop": info["reward_stop"],
                            "train/reward_pedestrian": info["reward_pedestrian"],
                            "train/reward_comfort": info["reward_comfort"],
                            "train/reward_terminal": info["reward_terminal"],
                            "train/v_cmd": info["v_cmd"],
                            "train/abs_omega_cmd": abs(info["omega_cmd"]),
                            "train/v_actual": info["v_actual"],
                            "train/abs_omega_actual": abs(info["omega_actual"]),
                            "train/buffer_size": len(replay),
                            "train/update_count": agent.update_count,
                            "loss/q1": update_metrics["q1_loss"],
                            "loss/q2": update_metrics["q2_loss"],
                            "loss/actor": update_metrics["actor_loss"],
                            "loss/alpha": update_metrics["alpha_loss"],
                            "train/entropy_coefficient": update_metrics["entropy_coefficient"],
                        },
                        step=global_step,
                    )

                done = terminated or truncated
                if done:
                    episode_writer.writerow(
                        {
                            "episode": episode_index,
                            "global_step": global_step,
                            **episode,
                            "timeout": truncated,
                            "termination_reason": info["termination_reason"],
                            "truncation_reason": info["truncation_reason"],
                        }
                    )
                    completed_episodes += 1
                    if wandb_run is not None:
                        wandb_run.log(
                            {
                                "episode/return": episode["total_reward"],
                                "episode/length": episode["episode_length"],
                                "episode/progress_m": episode["progress_m"],
                                "episode/collision": int(episode["collision"]),
                                "episode/stop_completed": int(episode["stop_completed"]),
                                "episode/stop_violation": int(episode["stop_violation"]),
                                "episode/invalid_pose": int(episode["invalid_pose"]),
                                "episode/mean_v_cmd": episode["mean_v_cmd"],
                                "episode/mean_abs_omega_cmd": episode["mean_abs_omega_cmd"],
                                "episode/mean_action_change": episode["mean_action_change"],
                            },
                            step=global_step,
                        )
                    episode_index += 1
                    observation, reset_info = environment.reset()
                    episode = _new_episode(reset_info)
                    previous_physical_action = np.zeros(2, dtype=np.float32)
                else:
                    observation = next_observation
                    previous_physical_action = physical_action

                if global_step % checkpoint_interval == 0:
                    checkpoint_path = checkpoints / f"sac_step_{global_step:07d}.pt"
                    agent.save(
                        checkpoint_path,
                        global_step=global_step,
                        metadata={
                            "f10_config_sha256": manifest["f10_config_sha256"],
                            "smoke": smoke,
                        },
                    )
                    checkpoint_paths.append(checkpoint_path)
                report_interval = 32 if smoke else 1000
                if global_step % report_interval == 0:
                    training_stream.flush()
                    episode_stream.flush()
                    print(
                        json.dumps(
                            {
                                "global_step": global_step,
                                "episodes": completed_episodes,
                                "buffer": len(replay),
                                "updates": agent.update_count,
                                "alpha": agent.entropy_coefficient,
                                "last_reward": reward,
                            }
                        ),
                        flush=True,
                    )
    finally:
        environment.close()

    if not checkpoint_paths:
        raise RuntimeError("training produced no checkpoint")
    final_checkpoint = checkpoint_paths[-1]
    deterministic_before = agent.act(observation, deterministic=True)
    loaded, payload = SACAgent.load(final_checkpoint, device=sac_config.device)
    deterministic_after = loaded.act(observation, deterministic=True)
    load_verified = bool(np.array_equal(deterministic_before, deterministic_after))
    if not load_verified or gradient_updates <= 0:
        raise RuntimeError("SAC smoke invariant failed: gradients/checkpoint reload")
    run_manifest = {
        **manifest,
        "completed_episodes": completed_episodes,
        "gradient_updates": gradient_updates,
        "checkpoint_paths": [str(path.resolve()) for path in checkpoint_paths],
        "final_checkpoint": str(final_checkpoint.resolve()),
        "final_checkpoint_sha256": file_sha256(final_checkpoint),
        "checkpoint_global_step": int(payload["global_step"]),
        "checkpoint_reload_verified": load_verified,
        "training_metrics_rows": planned_steps,
        "wandb": {
            "mode": wandb_mode,
            "run_id": None if wandb_run is None else wandb_run.id,
            "run_url": None if wandb_run is None else wandb_run.url,
        },
    }
    (output_dir / "training_run_manifest.json").write_text(
        json.dumps(run_manifest, indent=2) + "\n",
        encoding="utf-8",
    )
    if wandb_run is not None:
        wandb_run.summary.update(
            {
                "completed_episodes": completed_episodes,
                "gradient_updates": gradient_updates,
                "final_checkpoint_sha256": run_manifest["final_checkpoint_sha256"],
                "checkpoint_reload_verified": load_verified,
            }
        )
        wandb_run.save(str(output_dir / "training_run_manifest.json"), policy="now")
        wandb_run.finish()
    return run_manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "f10_sac_v1.toml")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument(
        "--wandb-mode",
        choices=("online", "offline", "disabled"),
        default=None,
    )
    args = parser.parse_args()
    output = args.output_dir or (
        ROOT / "artifacts" / "f10" / "smoke"
        if args.smoke
        else ROOT / "artifacts" / "f10"
    )
    wandb_mode = args.wandb_mode or ("offline" if args.smoke else "online")
    result = train(
        args.config.resolve(),
        output.resolve(),
        smoke=args.smoke,
        wandb_mode=wandb_mode,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
