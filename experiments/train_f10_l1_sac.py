"""Train the predeclared F10-L1 SAC lane curriculum on real small_loop."""

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
    LaneCurriculumEnvironment,
    ReplayBuffer,
    SACAgent,
    SACConfig,
    load_lane_protocol,
)
from duckie_pomdp.control.f10_protocol import file_sha256


ROOT = Path(__file__).resolve().parents[1]
SOURCE_FILES = (
    "experiments/train_f10_l1_sac.py",
    "src/duckie_pomdp/control/lane_environment.py",
    "src/duckie_pomdp/control/lane_policy_observation.py",
    "src/duckie_pomdp/control/lane_protocol.py",
    "src/duckie_pomdp/control/lane_reward.py",
    "src/duckie_pomdp/control/sac.py",
)
REWARD_FIELDS = (
    "reward_progress",
    "reward_lane",
    "reward_yellow",
    "reward_comfort",
    "reward_living",
    "reward_terminal",
)


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


def _manifest(protocol, config: SACConfig, *, smoke: bool, steps: int) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "stage": "F10_L1_SMOKE" if smoke else "F10_L1_TRAINING",
        "config": str(protocol.config_path),
        "config_sha256": file_sha256(protocol.config_path),
        "source_sha256": {
            item: file_sha256(ROOT / item) for item in SOURCE_FILES
        },
        "upstream": {
            "action_config": str(protocol.action_config_path),
            "action_config_sha256": protocol.action_config_sha256,
            "environment_spec": str(protocol.environment_spec_path),
            "environment_spec_sha256": protocol.environment_spec_sha256,
            "small_loop_map": str(protocol.map_path),
            "small_loop_map_sha256": protocol.map_sha256,
        },
        "seed_split": asdict(protocol.seeds),
        "sac": asdict(config),
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
            "wandb": _package_version("wandb"),
            "numpy": np.__version__,
        },
    }


def _start_wandb(protocol, manifest: dict[str, Any], output: Path, *, smoke: bool, mode: str):
    if mode == "disabled":
        return None
    try:
        import wandb
    except ModuleNotFoundError as error:
        raise RuntimeError("W&B telemetry requested but wandb is unavailable") from error
    settings = protocol.raw["wandb"]
    return wandb.init(
        entity=str(settings["entity"]),
        project=str(settings["project"]),
        group=str(settings["group"]),
        job_type="smoke" if smoke else str(settings["job_type"]),
        name=(
            f"f10-l1-smoke-{protocol.sac.training_seed}"
            if smoke
            else f"f10-l1-lane-sac-{protocol.sac.training_seed}"
        ),
        config=manifest,
        mode=mode,
        dir=str(output),
        reinit="finish_previous",
    )


def _require_gate(protocol, output_dir: Path) -> dict[str, Any]:
    gate_path = output_dir.parent / "pretraining_gate.json"
    if not gate_path.is_file():
        raise RuntimeError("full F10-L1 training requires pretraining_gate.json")
    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    if gate.get("ready_for_training") is not True:
        raise RuntimeError("F10-L1 pre-training gate is not READY")
    if gate.get("config_sha256") != file_sha256(protocol.config_path):
        raise RuntimeError("F10-L1 config changed after the pre-training gate")
    current_sources = {item: file_sha256(ROOT / item) for item in SOURCE_FILES}
    if gate.get("source_sha256") != current_sources:
        raise RuntimeError("F10-L1 source changed after the pre-training gate")
    return gate


def _new_episode(reset_info: dict[str, Any]) -> dict[str, Any]:
    return {
        "seed": int(reset_info["seed"]),
        "episode_length": 0,
        "total_return": 0.0,
        **{name: 0.0 for name in REWARD_FIELDS},
        "path_length_m": 0.0,
        "lap_completed": False,
        "invalid_pose": False,
        "yellow_crossing": False,
        "lane_departure": False,
        "minimum_yellow_clearance_m": float("inf"),
        "mean_abs_lateral_error_m": 0.0,
        "mean_abs_heading_error_rad": 0.0,
        "mean_actual_velocity_mps": 0.0,
        "mean_v_cmd_mps": 0.0,
        "mean_abs_omega_cmd_rad_s": 0.0,
        "mean_action_change": 0.0,
    }


def _running_mean(current: float, value: float, count: int) -> float:
    return current + (value - current) / count


def _accumulate_episode(
    episode: dict[str, Any],
    reward: float,
    info: dict[str, Any],
    action_change: float,
) -> None:
    episode["episode_length"] += 1
    count = int(episode["episode_length"])
    episode["total_return"] += reward
    for name in REWARD_FIELDS:
        episode[name] += float(info[name])
    episode["path_length_m"] = float(info["path_length_m"])
    for name in ("lap_completed", "invalid_pose", "yellow_crossing", "lane_departure"):
        episode[name] = bool(episode[name] or info[name])
    episode["minimum_yellow_clearance_m"] = min(
        float(episode["minimum_yellow_clearance_m"]),
        float(info["yellow_clearance_m"]),
    )
    for name, value in (
        ("mean_abs_lateral_error_m", abs(float(info["lateral_error_m"]))),
        ("mean_abs_heading_error_rad", abs(float(info["heading_error_rad"]))),
        ("mean_actual_velocity_mps", float(info["v_actual"])),
        ("mean_v_cmd_mps", float(info["v_cmd"])),
        ("mean_abs_omega_cmd_rad_s", abs(float(info["omega_cmd"]))),
        ("mean_action_change", action_change),
    ):
        episode[name] = _running_mean(float(episode[name]), value, count)


def _require_finite_update(metrics: dict[str, float], *, step: int) -> None:
    invalid = {name: value for name, value in metrics.items() if not np.isfinite(value)}
    if invalid:
        raise FloatingPointError(f"non-finite SAC metrics at step {step}: {invalid}")


def train(
    config_path: Path,
    output_dir: Path,
    *,
    smoke: bool,
    wandb_mode: str,
) -> dict[str, Any]:
    protocol = load_lane_protocol(config_path)
    planned_steps = 128 if smoke else protocol.sac.training_steps
    sac_config = _sac_config(protocol, smoke=smoke)
    output_dir.mkdir(parents=True, exist_ok=True)
    gate = None if smoke else _require_gate(protocol, output_dir)
    protected = (
        "training_metrics.csv",
        "episode_metrics.csv",
        "config_manifest.json",
        "normalization.json",
        "training_run_manifest.json",
    )
    existing = [output_dir / name for name in protected if (output_dir / name).exists()]
    if existing:
        raise FileExistsError(f"refusing to overwrite F10-L1 artifacts: {existing}")

    manifest = _manifest(protocol, sac_config, smoke=smoke, steps=planned_steps)
    if gate is not None:
        gate_path = output_dir.parent / "pretraining_gate.json"
        manifest["pretraining_gate"] = {
            "path": str(gate_path.resolve()),
            "sha256": file_sha256(gate_path),
            "verified_at_utc": gate["verified_at_utc"],
        }
    (output_dir / "config_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
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
        protocol, manifest, output_dir, smoke=smoke, mode=wandb_mode
    )
    environment = LaneCurriculumEnvironment(config_path, split="training")
    agent = SACAgent(sac_config)
    replay = ReplayBuffer(
        sac_config.replay_buffer_size,
        sac_config.observation_dimension,
        sac_config.action_dimension,
        seed=sac_config.seed + 1,
    )
    action_rng = np.random.default_rng(sac_config.seed + 2)
    checkpoints_dir = output_dir / "checkpoints"
    checkpoints_dir.mkdir(parents=True, exist_ok=False)
    checkpoint_interval = planned_steps if smoke else protocol.sac.checkpoint_interval_steps
    training_path = output_dir / "training_metrics.csv"
    episodes_path = output_dir / "episode_metrics.csv"
    observation_fields = [
        f"observation_normalized_{name}" for name in protocol.observation_order
    ]
    training_fields = [
        "global_step", "episode", "seed", "reward", *REWARD_FIELDS,
        "terminated", "truncated", "lap_completed", "invalid_pose",
        "yellow_crossing", "lane_departure", "termination_reason",
        "truncation_reason", "path_length_m", "yellow_clearance_m",
        "lateral_error_m", "heading_error_rad", "v_cmd", "omega_cmd",
        "v_actual", "omega_actual", "action_change", "buffer_size",
        "update_count", "q1_loss", "q2_loss", "actor_loss", "alpha_loss",
        "entropy_coefficient", "mean_log_probability", "mean_q",
        *observation_fields,
    ]
    episode_fields = [
        "episode", "global_step", "seed", "episode_length", "total_return",
        *REWARD_FIELDS, "path_length_m", "lap_completed", "invalid_pose",
        "yellow_crossing", "lane_departure", "minimum_yellow_clearance_m",
        "mean_abs_lateral_error_m", "mean_abs_heading_error_rad",
        "mean_actual_velocity_mps", "mean_v_cmd_mps",
        "mean_abs_omega_cmd_rad_s", "mean_action_change", "timeout",
        "termination_reason", "truncation_reason",
    ]
    checkpoint_paths: list[Path] = []
    completed_episodes = 0
    episode_index = 0
    gradient_updates = 0
    observation: np.ndarray | None = None
    try:
        observation, reset_info = environment.reset()
        episode = _new_episode(reset_info)
        previous_physical = np.zeros(2, dtype=np.float32)
        with training_path.open("w", newline="", encoding="utf-8") as train_stream, episodes_path.open("w", newline="", encoding="utf-8") as episode_stream:
            train_writer = csv.DictWriter(train_stream, fieldnames=training_fields)
            episode_writer = csv.DictWriter(episode_stream, fieldnames=episode_fields)
            train_writer.writeheader()
            episode_writer.writeheader()
            for global_step in range(1, planned_steps + 1):
                if global_step <= sac_config.learning_starts:
                    action = action_rng.uniform(-1.0, 1.0, 2).astype(np.float32)
                else:
                    action = agent.act(observation, deterministic=False)
                next_observation, reward, terminated, truncated, info = environment.step(action)
                if not np.isfinite(reward) or not np.all(np.isfinite(next_observation)):
                    raise FloatingPointError(f"non-finite transition at step {global_step}")
                replay.add(observation, action, reward, next_observation, terminated)
                physical = np.asarray((info["v_cmd"], info["omega_cmd"]), dtype=np.float32)
                action_change = float(
                    np.linalg.norm(
                        (physical - previous_physical)
                        / np.asarray((0.4, 4.0), dtype=np.float32)
                    )
                )
                _accumulate_episode(episode, reward, info, action_change)
                update_metrics = {
                    name: float("nan")
                    for name in (
                        "q1_loss", "q2_loss", "actor_loss", "alpha_loss",
                        "entropy_coefficient", "mean_log_probability", "mean_q",
                    )
                }
                if (
                    global_step >= sac_config.learning_starts
                    and len(replay) >= sac_config.batch_size
                    and global_step % sac_config.train_frequency == 0
                ):
                    for _ in range(sac_config.gradient_steps):
                        update_metrics = agent.update(replay)
                        _require_finite_update(update_metrics, step=global_step)
                        gradient_updates += 1
                row = {
                    "global_step": global_step,
                    "episode": episode_index,
                    "seed": episode["seed"],
                    "reward": reward,
                    **{name: info[name] for name in REWARD_FIELDS},
                    "terminated": terminated,
                    "truncated": truncated,
                    "lap_completed": info["lap_completed"],
                    "invalid_pose": info["invalid_pose"],
                    "yellow_crossing": info["yellow_crossing"],
                    "lane_departure": info["lane_departure"],
                    "termination_reason": info["termination_reason"],
                    "truncation_reason": info["truncation_reason"],
                    "path_length_m": info["path_length_m"],
                    "yellow_clearance_m": info["yellow_clearance_m"],
                    "lateral_error_m": info["lateral_error_m"],
                    "heading_error_rad": info["heading_error_rad"],
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
                        name: float(observation[index])
                        for index, name in enumerate(observation_fields)
                    }
                )
                train_writer.writerow(row)
                if (
                    wandb_run is not None
                    and global_step % int(protocol.raw["wandb"]["log_interval_steps"]) == 0
                ):
                    wandb_run.log(
                        {
                            "train/reward": reward,
                            **{
                                f"train/{name}": float(info[name])
                                for name in REWARD_FIELDS
                            },
                            "train/path_length_m": float(info["path_length_m"]),
                            "train/yellow_clearance_m": float(info["yellow_clearance_m"]),
                            "train/abs_lateral_error_m": abs(float(info["lateral_error_m"])),
                            "train/abs_heading_error_rad": abs(float(info["heading_error_rad"])),
                            "train/v_cmd_mps": float(info["v_cmd"]),
                            "train/abs_omega_cmd_rad_s": abs(float(info["omega_cmd"])),
                            "train/v_actual_mps": float(info["v_actual"]),
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

                if terminated or truncated:
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
                                "episode/return": episode["total_return"],
                                "episode/length": episode["episode_length"],
                                "episode/path_length_m": episode["path_length_m"],
                                "episode/lap_completed": int(episode["lap_completed"]),
                                "episode/invalid_pose": int(episode["invalid_pose"]),
                                "episode/yellow_crossing": int(episode["yellow_crossing"]),
                                "episode/lane_departure": int(episode["lane_departure"]),
                                "episode/mean_abs_lateral_error_m": episode["mean_abs_lateral_error_m"],
                                "episode/mean_abs_heading_error_rad": episode["mean_abs_heading_error_rad"],
                                "episode/mean_actual_velocity_mps": episode["mean_actual_velocity_mps"],
                            },
                            step=global_step,
                        )
                    episode_index += 1
                    observation, reset_info = environment.reset()
                    episode = _new_episode(reset_info)
                    previous_physical = np.zeros(2, dtype=np.float32)
                else:
                    observation = next_observation
                    previous_physical = physical

                if global_step % checkpoint_interval == 0:
                    checkpoint = checkpoints_dir / f"sac_step_{global_step:07d}.pt"
                    agent.save(
                        checkpoint,
                        global_step=global_step,
                        metadata={
                            "stage": "F10-L1",
                            "config_sha256": manifest["config_sha256"],
                            "smoke": smoke,
                        },
                    )
                    checkpoint_paths.append(checkpoint)
                report_interval = 32 if smoke else 1000
                if global_step % report_interval == 0:
                    train_stream.flush()
                    episode_stream.flush()
                    print(
                        json.dumps(
                            {
                                "global_step": global_step,
                                "episodes": completed_episodes,
                                "updates": agent.update_count,
                                "buffer": len(replay),
                                "alpha": agent.entropy_coefficient,
                                "last_reward": reward,
                            }
                        ),
                        flush=True,
                    )
    finally:
        environment.close()

    if observation is None or not checkpoint_paths:
        raise RuntimeError("F10-L1 training produced no checkpoint")
    final_checkpoint = checkpoint_paths[-1]
    before = agent.act(observation, deterministic=True)
    loaded, payload = SACAgent.load(final_checkpoint, device=sac_config.device)
    after = loaded.act(observation, deterministic=True)
    reload_verified = bool(np.array_equal(before, after))
    if not reload_verified or gradient_updates <= 0:
        raise RuntimeError("F10-L1 gradient/checkpoint smoke invariant failed")
    run_manifest = {
        **manifest,
        "completed_episodes": completed_episodes,
        "gradient_updates": gradient_updates,
        "checkpoint_paths": [str(path.resolve()) for path in checkpoint_paths],
        "final_checkpoint": str(final_checkpoint.resolve()),
        "final_checkpoint_sha256": file_sha256(final_checkpoint),
        "checkpoint_global_step": int(payload["global_step"]),
        "checkpoint_reload_verified": reload_verified,
        "training_metrics_rows": planned_steps,
        "wandb": {
            "mode": wandb_mode,
            "run_id": None if wandb_run is None else wandb_run.id,
            "run_url": None if wandb_run is None else wandb_run.url,
        },
    }
    manifest_path = output_dir / "training_run_manifest.json"
    manifest_path.write_text(
        json.dumps(run_manifest, indent=2) + "\n", encoding="utf-8"
    )
    if wandb_run is not None:
        wandb_run.summary.update(
            {
                "completed_episodes": completed_episodes,
                "gradient_updates": gradient_updates,
                "checkpoint_reload_verified": reload_verified,
                "final_checkpoint_sha256": run_manifest["final_checkpoint_sha256"],
            }
        )
        wandb_run.save(str(manifest_path), policy="now")
        wandb_run.finish()
    return run_manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=Path, default=ROOT / "configs" / "f10_l1_lane_v1.toml"
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument(
        "--wandb-mode", choices=("online", "offline", "disabled"), default=None
    )
    args = parser.parse_args()
    output = args.output_dir or (
        ROOT / "artifacts" / "f10_l1" / "smoke"
        if args.smoke
        else ROOT / "artifacts" / "f10_l1" / "training"
    )
    wandb_settings = load_lane_protocol(args.config.resolve()).raw["wandb"]
    mode = args.wandb_mode or str(
        wandb_settings["smoke_mode"] if args.smoke else wandb_settings["training_mode"]
    )
    result = train(
        args.config.resolve(), output.resolve(), smoke=args.smoke, wandb_mode=mode
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
