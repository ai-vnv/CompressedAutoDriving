#!/usr/bin/env python3
"""R001: audit the deployed RGB -> belief -> 29D -> frozen PPO boundary."""

from __future__ import annotations

import argparse
import hashlib
import json
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
from duckie_pomdp.explain.observation_contract import (
    assert_policy_vector_precedes_privileged_read,
    deterministic_actor_statistics,
    reconstruct_normalized_observation,
    validate_feature_group_partition,
    validate_public_policy_mapping,
)


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs" / "f11_ppo_explanation_v2.toml",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    run_audit(args.config.resolve(), overwrite=bool(args.overwrite))


def run_audit(config_path: Path, *, overwrite: bool = False) -> dict[str, Any]:
    raw = _load_toml(config_path)
    r001 = raw["r001"]
    output = _resolve(config_path, str(r001["output_directory"]))
    result_path = output / "contract_audit.json"
    trace_path = output / "public_trace.npz"
    manifest_path = output / "trace_manifest.json"
    if not overwrite and any(path.exists() for path in (result_path, trace_path, manifest_path)):
        raise FileExistsError("R001 artifacts already exist; refusing to overwrite")
    output.mkdir(parents=True, exist_ok=True)

    frozen = raw["frozen_policy"]
    perception = raw["frozen_perception"]
    plan = raw["protocol"]
    source_config = _resolve(config_path, str(frozen["config"]))
    checkpoint = _resolve(config_path, str(frozen["checkpoint"]))
    lane_checkpoint = _resolve(config_path, str(perception["lane_checkpoint"]))
    yolo_checkpoint = _resolve(config_path, str(perception["yolo_checkpoint"]))
    plan_path = _resolve(config_path, str(plan["plan"]))
    verified_hashes = {
        "explanation_config": sha256(config_path),
        "policy_config": _verify_hash(source_config, str(frozen["config_sha256"])),
        "policy_checkpoint": _verify_hash(checkpoint, str(frozen["checkpoint_sha256"])),
        "lane_checkpoint": _verify_hash(
            lane_checkpoint, str(perception["lane_checkpoint_sha256"])
        ),
        "yolo_checkpoint": _verify_hash(
            yolo_checkpoint, str(perception["yolo_checkpoint_sha256"])
        ),
        "plan": _verify_hash(plan_path, str(plan["plan_sha256"])),
    }

    protocol = load_ppo_curriculum_protocol(source_config)
    order = tuple(protocol.observation_order)
    expected_dimension = int(frozen["observation_dimension"])
    if len(order) != expected_dimension:
        raise ValueError("frozen observation dimension mismatch")
    groups = {
        str(name): tuple(str(field) for field in fields)
        for name, fields in raw["feature_groups"].items()
    }
    expected_groups = ("Lane", "Ego", "StopLine", "Pedestrian", "Stop", "PreviousAction")
    if tuple(groups) != expected_groups:
        raise ValueError("primary feature-group order differs from frozen plan")
    validate_feature_group_partition(order, groups)
    assert_policy_vector_precedes_privileged_read(PPOCurriculumEnvironment.reset)
    assert_policy_vector_precedes_privileged_read(PPOCurriculumEnvironment.step)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    agent, payload = PPOAgent.load(checkpoint, device=device)
    if str(payload["stage"]) != str(frozen["stage"]):
        raise ValueError("checkpoint stage mismatch")
    if int(agent.config.observation_dimension) != expected_dimension:
        raise ValueError("checkpoint observation dimension mismatch")
    if tuple(agent.config.hidden_sizes) != tuple(int(v) for v in frozen["hidden_sizes"]):
        raise ValueError("checkpoint hidden architecture mismatch")
    model_hash_before = model_state_sha256(agent.model)

    pilot_seed = int(raw["seeds"]["r001_pilot"])
    maximum_steps = int(r001["maximum_episode_steps"])
    tolerance = float(r001["replay_absolute_tolerance"])
    mapper = NormalizedActionMapper(
        float(frozen["maximum_linear_velocity_mps"]),
        float(frozen["maximum_angular_velocity_rad_s"]),
    )
    environment = PPOCurriculumEnvironment(
        source_config,
        stage=str(frozen["stage"]),
        split="explanation_contract_audit",
        seeds=(pilot_seed,),
    )
    rows: dict[str, list[Any]] = {
        "step": [],
        "observation": [],
        "physical_observation": [],
        "actor_mean": [],
        "environment_action": [],
        "physical_action": [],
        "critic_value": [],
        "rgb_sha256": [],
        "terminated": [],
        "truncated": [],
    }
    max_normalization_error = 0.0
    max_actor_api_error = 0.0
    max_action_mapping_error = 0.0
    perception_frames = 0
    lane_runtime_frames = 0
    yolo_runtime_frames = 0
    try:
        observation, info = environment.reset(seed=pilot_seed)
        for step in range(maximum_steps):
            public_mapping = info["policy"]
            validate_public_policy_mapping(public_mapping, order)
            reconstructed = reconstruct_normalized_observation(
                public_mapping,
                order,
                protocol.observation_scales,
                protocol.observation_clip,
            )
            observation_array = np.asarray(observation, dtype=np.float32)
            max_normalization_error = max(
                max_normalization_error,
                float(np.max(np.abs(reconstructed - observation_array))),
            )

            observation_tensor = torch.as_tensor(
                observation_array, dtype=torch.float32, device=device
            ).unsqueeze(0)
            direct_mean, direct_value = deterministic_actor_statistics(
                agent.model, observation_tensor
            )
            deterministic = agent.act(observation_array, deterministic=True)
            mean_array = direct_mean.squeeze(0).cpu().numpy().astype(np.float32)
            max_actor_api_error = max(
                max_actor_api_error,
                float(np.max(np.abs(mean_array - deterministic.raw_action))),
                abs(float(direct_value.item()) - float(deterministic.value)),
            )
            environment_action = np.asarray(
                deterministic.environment_action, dtype=np.float32
            )
            mapping = mapper.map(environment_action)
            physical_action = np.asarray(
                [
                    mapping.policy_action.linear_velocity_mps,
                    mapping.policy_action.angular_velocity_rad_s,
                ],
                dtype=np.float32,
            )
            rgb = environment.latest_rgb()
            if rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.dtype != np.uint8:
                raise ValueError("runtime front RGB contract is invalid")

            next_observation, _, terminated, truncated, next_info = environment.step(
                environment_action
            )
            max_action_mapping_error = max(
                max_action_mapping_error,
                abs(float(next_info["v_cmd"]) - float(physical_action[0])),
                abs(float(next_info["omega_cmd"]) - float(physical_action[1])),
            )
            perception_info = next_info.get("perception", {})
            if perception_info:
                perception_frames += 1
            if "lane_validity_probability" in perception_info:
                lane_runtime_frames += 1
            if (
                "duckie_detection_count" in perception_info
                and "stop_sign_detection_count" in perception_info
            ):
                yolo_runtime_frames += 1

            rows["step"].append(step)
            rows["observation"].append(observation_array)
            rows["physical_observation"].append(
                np.asarray([public_mapping[name] for name in order], dtype=np.float32)
            )
            rows["actor_mean"].append(mean_array)
            rows["environment_action"].append(environment_action)
            rows["physical_action"].append(physical_action)
            rows["critic_value"].append(float(direct_value.item()))
            rows["rgb_sha256"].append(hashlib.sha256(rgb.tobytes()).hexdigest())
            rows["terminated"].append(bool(terminated))
            rows["truncated"].append(bool(truncated))
            observation, info = next_observation, next_info
            if terminated or truncated:
                break
    finally:
        environment.close()

    arrays = {
        "step": np.asarray(rows["step"], dtype=np.int32),
        "observation": np.asarray(rows["observation"], dtype=np.float32),
        "physical_observation": np.asarray(
            rows["physical_observation"], dtype=np.float32
        ),
        "deterministic_actor_mean": np.asarray(rows["actor_mean"], dtype=np.float32),
        "environment_action": np.asarray(rows["environment_action"], dtype=np.float32),
        "physical_action": np.asarray(rows["physical_action"], dtype=np.float32),
        "critic_value": np.asarray(rows["critic_value"], dtype=np.float32),
        "rgb_sha256": np.asarray(rows["rgb_sha256"], dtype="U64"),
        "terminated": np.asarray(rows["terminated"], dtype=np.bool_),
        "truncated": np.asarray(rows["truncated"], dtype=np.bool_),
        "feature_names": np.asarray(order, dtype="U64"),
    }
    if arrays["observation"].shape != (len(rows["step"]), expected_dimension):
        raise ValueError("collected observation matrix has invalid shape")
    if len(rows["step"]) == 0 or not np.all(np.isfinite(arrays["observation"])):
        raise ValueError("R001 did not collect finite public observations")
    if lane_runtime_frames != len(rows["step"]):
        raise RuntimeError("visual lane runtime was not present on every audited step")
    if yolo_runtime_frames != len(rows["step"]):
        raise RuntimeError("YOLO runtime was not present on every audited step")
    if max(max_normalization_error, max_actor_api_error, max_action_mapping_error) > tolerance:
        raise RuntimeError("online observation/action contract exceeded replay tolerance")

    np.savez_compressed(trace_path, **arrays)
    replay_agent, replay_payload = PPOAgent.load(checkpoint, device=device)
    replay_observations = torch.as_tensor(
        arrays["observation"], dtype=torch.float32, device=device
    )
    replay_mean, replay_value = deterministic_actor_statistics(
        replay_agent.model, replay_observations
    )
    replay_mean_array = replay_mean.cpu().numpy().astype(np.float32)
    replay_value_array = replay_value.cpu().numpy().astype(np.float32)
    replay_action_error = float(
        np.max(np.abs(replay_mean_array - arrays["deterministic_actor_mean"]))
    )
    replay_value_error = float(
        np.max(np.abs(replay_value_array - arrays["critic_value"]))
    )
    if max(replay_action_error, replay_value_error) > tolerance:
        raise RuntimeError("fresh checkpoint replay exceeded tolerance")
    if str(replay_payload["stage"]) != str(frozen["stage"]):
        raise ValueError("reloaded checkpoint stage mismatch")

    model_hash_after = model_state_sha256(agent.model)
    checkpoint_hash_after = sha256(checkpoint)
    if model_hash_after != model_hash_before:
        raise RuntimeError("in-memory frozen PPO parameters changed during R001")
    if checkpoint_hash_after != verified_hashes["policy_checkpoint"]:
        raise RuntimeError("frozen PPO checkpoint changed during R001")

    checks = {
        "frozen_hashes": True,
        "observation_dimension_29": True,
        "group_partition_exact": True,
        "policy_vector_before_privileged_read_reset": True,
        "policy_vector_before_privileged_read_step": True,
        "public_mapping_has_no_privileged_fields": True,
        "normalization_exact": max_normalization_error <= tolerance,
        "deterministic_actor_mean_not_sample": max_actor_api_error <= tolerance,
        "physical_action_mapping_exact": max_action_mapping_error <= tolerance,
        "real_rgb_present": len(set(rows["rgb_sha256"])) > 1,
        "lane_runtime_present": lane_runtime_frames == len(rows["step"]),
        "yolo_runtime_present": yolo_runtime_frames == len(rows["step"]),
        "fresh_checkpoint_replay_exact": max(replay_action_error, replay_value_error)
        <= tolerance,
        "model_unchanged": model_hash_after == model_hash_before,
        "checkpoint_unchanged": checkpoint_hash_after
        == verified_hashes["policy_checkpoint"],
        "trace_stores_no_privileged_truth": not any(
            any(token in key.lower() for token in ("gt", "privileged", "world_pose"))
            for key in arrays
        ),
    }
    classification = "PASS" if all(checks.values()) else "FAILED"
    result = {
        "schema_version": 1,
        "run_id": "R001",
        "classification": classification,
        "seed_role": "explanation_contract_pilot",
        "seed": pilot_seed,
        "steps": len(rows["step"]),
        "device": device,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "verified_hashes": verified_hashes,
        "checkpoint_global_step": int(payload["global_step"]),
        "checkpoint_stage": str(payload["stage"]),
        "model_state_sha256_before": model_hash_before,
        "model_state_sha256_after": model_hash_after,
        "observation_order": list(order),
        "observation_scales": [float(value) for value in protocol.observation_scales],
        "observation_clip": float(protocol.observation_clip),
        "primary_feature_groups": {name: list(values) for name, values in groups.items()},
        "actor_target": "deterministic_distribution_mean_before_sampling",
        "errors": {
            "maximum_normalization_absolute": max_normalization_error,
            "maximum_actor_api_absolute": max_actor_api_error,
            "maximum_action_mapping_absolute": max_action_mapping_error,
            "maximum_replay_actor_mean_absolute": replay_action_error,
            "maximum_replay_critic_value_absolute": replay_value_error,
        },
        "runtime_frames": {
            "perception": perception_frames,
            "lane": lane_runtime_frames,
            "yolo": yolo_runtime_frames,
            "unique_rgb": len(set(rows["rgb_sha256"])),
        },
        "checks": checks,
        "trace": str(trace_path.relative_to(ROOT)),
        "trace_sha256": sha256(trace_path),
        "stored_privileged_truth": False,
    }
    _write_json(result_path, result)
    manifest = {
        "schema_version": 1,
        "run_id": "R001",
        "classification": classification,
        "config": str(config_path.relative_to(ROOT)),
        "config_sha256": verified_hashes["explanation_config"],
        "result": str(result_path.relative_to(ROOT)),
        "result_sha256": sha256(result_path),
        "trace": str(trace_path.relative_to(ROOT)),
        "trace_sha256": sha256(trace_path),
        "stored_privileged_truth": False,
    }
    _write_json(manifest_path, manifest)
    print(json.dumps(result, indent=2))
    if classification != "PASS":
        raise RuntimeError("R001 contract audit failed")
    return result


def model_state_sha256(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        array = tensor.detach().cpu().contiguous().numpy()
        digest.update(name.encode("utf-8"))
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(str(array.shape).encode("ascii"))
        digest.update(array.tobytes())
    return digest.hexdigest()


def _load_toml(path: Path) -> dict[str, Any]:
    with path.open("rb") as stream:
        return tomllib.load(stream)


def _resolve(config_path: Path, value: str) -> Path:
    return (config_path.parent / value).resolve()


def _verify_hash(path: Path, expected: str) -> str:
    actual = sha256(path)
    if actual != expected.lower():
        raise RuntimeError(f"SHA256 mismatch for {path}: {actual} != {expected}")
    return actual


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

