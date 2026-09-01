"""Warm-start SAC on the frozen F10-L2 ``experiment_loop`` curriculum."""

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
    LaneTransferEnvironment,
    ReplayBuffer,
    SACAgent,
    SACConfig,
    load_lane_transfer_protocol,
)
from duckie_pomdp.control.f10_protocol import file_sha256


ROOT = Path(__file__).resolve().parents[1]
SOURCE_FILES = (
    "experiments/train_f10_l2_sac.py",
    "src/duckie_pomdp/control/lane_environment.py",
    "src/duckie_pomdp/control/lane_policy_observation.py",
    "src/duckie_pomdp/control/lane_reward.py",
    "src/duckie_pomdp/control/lane_transfer_environment.py",
    "src/duckie_pomdp/control/lane_transfer_protocol.py",
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
LOSS_FIELDS = (
    "q1_loss",
    "q2_loss",
    "actor_loss",
    "alpha_loss",
    "entropy_coefficient",
    "mean_log_probability",
    "mean_q",
)


def _package_version(name: str) -> str | None:
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def _target_sac_config(protocol, *, smoke: bool) -> SACConfig:
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


def load_transfer_agent(protocol, *, smoke: bool) -> tuple[SACAgent, dict[str, Any]]:
    target = _target_sac_config(protocol, smoke=smoke)
    agent, payload = SACAgent.load(
        protocol.transfer_checkpoint_path,
        device=target.device,
    )
    if int(payload["global_step"]) != protocol.source_global_step:
        raise RuntimeError("F10-L2 source checkpoint step mismatch")
    source = asdict(agent.config)
    expected = asdict(target)
    transferable = set(expected) - {"seed", "batch_size", "replay_buffer_size", "learning_starts"}
    mismatches = {
        name: (source[name], expected[name])
        for name in transferable
        if source[name] != expected[name]
    }
    if mismatches:
        raise RuntimeError(f"F10-L2 SAC architecture/hyperparameter mismatch: {mismatches}")
    agent.config = target
    torch.manual_seed(target.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(target.seed)
    return agent, payload


def _new_episode(seed: int) -> dict[str, Any]:
    return {
        "seed": seed,
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


def _mean_update(current: float, value: float, count: int) -> float:
    return current + (value - current) / count


def _accumulate(
    episode: dict[str, Any], reward: float, info: dict[str, Any], action_change: float
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
    samples = (
        ("mean_abs_lateral_error_m", abs(float(info["lateral_error_m"]))),
        ("mean_abs_heading_error_rad", abs(float(info["heading_error_rad"]))),
        ("mean_actual_velocity_mps", float(info["v_actual"])),
        ("mean_v_cmd_mps", float(info["v_cmd"])),
        ("mean_abs_omega_cmd_rad_s", abs(float(info["omega_cmd"]))),
        ("mean_action_change", action_change),
    )
    for name, value in samples:
        episode[name] = _mean_update(float(episode[name]), value, count)


def _manifest(protocol, agent: SACAgent, *, smoke: bool, steps: int) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "stage": "F10_L2_SMOKE" if smoke else "F10_L2_TRANSFER_TRAINING",
        "config": str(protocol.config_path),
        "config_sha256": file_sha256(protocol.config_path),
        "source_sha256": {name: file_sha256(ROOT / name) for name in SOURCE_FILES},
        "upstream": {
            "action_config": str(protocol.action_config_path),
            "action_config_sha256": protocol.action_config_sha256,
            "environment_spec": str(protocol.environment_spec_path),
            "environment_spec_sha256": protocol.environment_spec_sha256,
            "map": str(protocol.map_path),
            "map_sha256": protocol.map_sha256,
            "transfer_checkpoint": str(protocol.transfer_checkpoint_path),
            "transfer_checkpoint_sha256": protocol.transfer_checkpoint_sha256,
            "source_global_step": protocol.source_global_step,
        },
        "seed_split": asdict(protocol.seeds),
        "sac": asdict(agent.config),
        "environment_steps": steps,
        "buffer_fill_policy": "warm_start_policy_stochastic",
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
    import wandb

    settings = protocol.raw["wandb"]
    return wandb.init(
        entity=str(settings["entity"]),
        project=str(settings["project"]),
        group=str(settings["group"]),
        job_type="smoke" if smoke else str(settings["job_type"]),
        name=(
            f"f10-l2-smoke-{protocol.sac.training_seed}"
            if smoke
            else f"f10-l2-transfer-sac-{protocol.sac.training_seed}"
        ),
        config=manifest,
        mode=mode,
        dir=str(output),
        reinit="finish_previous",
    )


def _require_gate(protocol, output: Path) -> dict[str, Any]:
    gate_path = output.parent / "pretraining_gate.json"
    if not gate_path.is_file():
        raise RuntimeError("full F10-L2 training requires pretraining_gate.json")
    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    if gate.get("ready_for_training") is not True:
        raise RuntimeError("F10-L2 pre-training gate is not READY")
    if gate.get("config_sha256") != file_sha256(protocol.config_path):
        raise RuntimeError("F10-L2 config changed after its pre-training gate")
    current = {name: file_sha256(ROOT / name) for name in SOURCE_FILES}
    if gate.get("source_sha256") != current:
        raise RuntimeError("F10-L2 source changed after its pre-training gate")
    return gate


def train(
    config_path: Path,
    output: Path,
    *,
    smoke: bool,
    wandb_mode: str,
) -> dict[str, Any]:
    protocol = load_lane_transfer_protocol(config_path)
    steps = 128 if smoke else protocol.sac.training_steps
    output.mkdir(parents=True, exist_ok=True)
    protected = ("training_metrics.csv", "episode_metrics.csv", "training_run_manifest.json")
    existing = [output / name for name in protected if (output / name).exists()]
    if existing:
        raise FileExistsError(f"refusing to overwrite F10-L2 artifacts: {existing}")
    gate = None if smoke else _require_gate(protocol, output)
    agent, source_payload = load_transfer_agent(protocol, smoke=smoke)
    initial_update_count = agent.update_count
    manifest = _manifest(protocol, agent, smoke=smoke, steps=steps)
    if gate is not None:
        gate_path = output.parent / "pretraining_gate.json"
        manifest["pretraining_gate"] = {
            "path": str(gate_path.resolve()),
            "sha256": file_sha256(gate_path),
        }
    (output / "config_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    (output / "normalization.json").write_text(
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
    run = _start_wandb(protocol, manifest, output, smoke=smoke, mode=wandb_mode)
    environment = LaneTransferEnvironment(config_path, split="training")
    replay = ReplayBuffer(
        agent.config.replay_buffer_size,
        agent.config.observation_dimension,
        agent.config.action_dimension,
        seed=protocol.sac.training_seed + 1,
    )
    checkpoint_dir = output / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=False)
    checkpoint_interval = steps if smoke else protocol.sac.checkpoint_interval_steps
    observation_fields = [
        f"observation_normalized_{name}" for name in protocol.observation_order
    ]
    step_fields = [
        "global_step", "episode", "seed", "reward", *REWARD_FIELDS,
        "terminated", "truncated", "lap_completed", "invalid_pose",
        "yellow_crossing", "lane_departure", "termination_reason",
        "truncation_reason", "path_length_m", "yellow_clearance_m",
        "lateral_error_m", "heading_error_rad", "v_cmd", "omega_cmd",
        "v_actual", "omega_actual", "action_change", "buffer_size",
        "source_update_count", "transfer_update_count", *LOSS_FIELDS,
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
    checkpoints: list[Path] = []
    completed_episodes = 0
    transfer_updates = 0
    episode_index = 0
    observation: np.ndarray | None = None
    try:
        observation, reset_info = environment.reset()
        episode = _new_episode(int(reset_info["seed"]))
        previous_physical = np.zeros(2, dtype=np.float32)
        with (output / "training_metrics.csv").open(
            "w", newline="", encoding="utf-8"
        ) as step_stream, (output / "episode_metrics.csv").open(
            "w", newline="", encoding="utf-8"
        ) as episode_stream:
            step_writer = csv.DictWriter(step_stream, fieldnames=step_fields)
            episode_writer = csv.DictWriter(episode_stream, fieldnames=episode_fields)
            step_writer.writeheader()
            episode_writer.writeheader()
            for global_step in range(1, steps + 1):
                # Transfer-specific invariant: new-map buffer fill comes from
                # the source policy, never unrelated uniform random actions.
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
                _accumulate(episode, reward, info, action_change)
                losses = {name: float("nan") for name in LOSS_FIELDS}
                if (
                    global_step >= agent.config.learning_starts
                    and len(replay) >= agent.config.batch_size
                    and global_step % agent.config.train_frequency == 0
                ):
                    for _ in range(agent.config.gradient_steps):
                        losses = agent.update(replay)
                        if not all(np.isfinite(value) for value in losses.values()):
                            raise FloatingPointError(
                                f"non-finite F10-L2 loss at step {global_step}: {losses}"
                            )
                        transfer_updates += 1
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
                    "source_update_count": initial_update_count,
                    "transfer_update_count": transfer_updates,
                    **losses,
                }
                row.update(
                    {
                        name: float(observation[index])
                        for index, name in enumerate(observation_fields)
                    }
                )
                step_writer.writerow(row)
                if run is not None and global_step % int(protocol.raw["wandb"]["log_interval_steps"]) == 0:
                    run.log(
                        {
                            "train/reward": reward,
                            **{f"train/{name}": float(info[name]) for name in REWARD_FIELDS},
                            "train/path_length_m": float(info["path_length_m"]),
                            "train/yellow_clearance_m": float(info["yellow_clearance_m"]),
                            "train/abs_lateral_error_m": abs(float(info["lateral_error_m"])),
                            "train/abs_heading_error_rad": abs(float(info["heading_error_rad"])),
                            "train/v_actual_mps": float(info["v_actual"]),
                            "train/transfer_update_count": transfer_updates,
                            **{f"loss/{name}": value for name, value in losses.items()},
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
                    if run is not None:
                        run.log(
                            {
                                "episode/return": episode["total_return"],
                                "episode/length": episode["episode_length"],
                                "episode/path_length_m": episode["path_length_m"],
                                "episode/lap_completed": int(episode["lap_completed"]),
                                "episode/invalid_pose": int(episode["invalid_pose"]),
                                "episode/yellow_crossing": int(episode["yellow_crossing"]),
                                "episode/lane_departure": int(episode["lane_departure"]),
                                "episode/mean_abs_lateral_error_m": episode["mean_abs_lateral_error_m"],
                            },
                            step=global_step,
                        )
                    episode_index += 1
                    observation, reset_info = environment.reset()
                    episode = _new_episode(int(reset_info["seed"]))
                    previous_physical = np.zeros(2, dtype=np.float32)
                else:
                    observation = next_observation
                    previous_physical = physical
                if global_step % checkpoint_interval == 0:
                    checkpoint = checkpoint_dir / f"sac_transfer_step_{global_step:07d}.pt"
                    agent.save(
                        checkpoint,
                        global_step=global_step,
                        metadata={
                            "stage": "F10-L2",
                            "config_sha256": manifest["config_sha256"],
                            "source_checkpoint_sha256": protocol.transfer_checkpoint_sha256,
                            "source_global_step": protocol.source_global_step,
                            "transfer_step": global_step,
                            "smoke": smoke,
                        },
                    )
                    checkpoints.append(checkpoint)
                report_interval = 32 if smoke else 1000
                if global_step % report_interval == 0:
                    step_stream.flush()
                    episode_stream.flush()
                    print(
                        json.dumps(
                            {
                                "global_step": global_step,
                                "episodes": completed_episodes,
                                "transfer_updates": transfer_updates,
                                "buffer": len(replay),
                                "alpha": agent.entropy_coefficient,
                                "last_reward": reward,
                            }
                        ),
                        flush=True,
                    )
    finally:
        environment.close()
    if observation is None or not checkpoints or transfer_updates <= 0:
        raise RuntimeError("F10-L2 did not produce a trained checkpoint")
    final_checkpoint = checkpoints[-1]
    before = agent.act(observation, deterministic=True)
    loaded, payload = SACAgent.load(final_checkpoint, device=agent.config.device)
    after = loaded.act(observation, deterministic=True)
    if not np.array_equal(before, after):
        raise RuntimeError("F10-L2 checkpoint reload changed deterministic action")
    result = {
        **manifest,
        "source_checkpoint_payload_step": int(source_payload["global_step"]),
        "source_update_count": initial_update_count,
        "completed_episodes": completed_episodes,
        "transfer_gradient_updates": transfer_updates,
        "total_agent_update_count": agent.update_count,
        "checkpoint_paths": [str(path.resolve()) for path in checkpoints],
        "final_checkpoint": str(final_checkpoint.resolve()),
        "final_checkpoint_sha256": file_sha256(final_checkpoint),
        "checkpoint_global_step": int(payload["global_step"]),
        "checkpoint_reload_verified": True,
        "training_metrics_rows": steps,
        "wandb": {
            "mode": wandb_mode,
            "run_id": None if run is None else run.id,
            "run_url": None if run is None else run.url,
        },
    }
    manifest_path = output / "training_run_manifest.json"
    manifest_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    if run is not None:
        run.summary.update(
            {
                "completed_episodes": completed_episodes,
                "transfer_gradient_updates": transfer_updates,
                "checkpoint_reload_verified": True,
                "final_checkpoint_sha256": result["final_checkpoint_sha256"],
            }
        )
        run.save(str(manifest_path), policy="now")
        run.finish()
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=Path, default=ROOT / "configs" / "f10_l2_transfer_v1.toml"
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument(
        "--wandb-mode", choices=("online", "offline", "disabled"), default=None
    )
    args = parser.parse_args()
    protocol = load_lane_transfer_protocol(args.config.resolve())
    output = args.output_dir or (
        ROOT / "artifacts" / "f10_l2" / ("smoke" if args.smoke else "training")
    )
    settings = protocol.raw["wandb"]
    mode = args.wandb_mode or str(
        settings["smoke_mode"] if args.smoke else settings["training_mode"]
    )
    print(
        json.dumps(
            train(
                args.config.resolve(),
                output.resolve(),
                smoke=args.smoke,
                wandb_mode=mode,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
