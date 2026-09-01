"""Frozen protocol loader for the F10-L2 lane-transfer experiment."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from duckie_pomdp.control.f10_protocol import file_sha256
from duckie_pomdp.control.lane_protocol import (
    LaneProtocol,
    LaneSACSettings,
    LaneSeedSplit,
)

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib


@dataclass(frozen=True)
class LaneTransferProtocol(LaneProtocol):
    transfer_checkpoint_path: Path
    transfer_checkpoint_sha256: str
    source_global_step: int


def _installed_map_path(map_name: str) -> Path:
    import duckietown_world

    package_root = Path(duckietown_world.__file__).resolve().parent
    return package_root / "data" / "gd1" / "maps" / f"{map_name}.yaml"


def load_lane_transfer_protocol(
    path: str | Path,
    *,
    require_frozen: bool = True,
) -> LaneTransferProtocol:
    config_path = Path(path).resolve()
    with config_path.open("rb") as stream:
        data: dict[str, Any] = tomllib.load(stream)
    if int(data.get("schema_version", 0)) != 1 or data.get("stage") != "F10-L2":
        raise ValueError("lane transfer config must declare F10-L2 schema version 1")

    provenance = data["provenance"]
    simulator = data["simulator"]
    if provenance["map_name"] != simulator["map"]:
        raise ValueError("F10-L2 provenance and simulator map must match")
    if simulator["map"] != "experiment_loop":
        raise ValueError("F10-L2 is frozen to experiment_loop")
    if simulator["direction"] != "closed_loop_mixed_turns":
        raise ValueError("F10-L2 must declare the mixed-turn closed route")

    split = data["seed_split"]
    seeds = LaneSeedSplit(
        training=tuple(int(value) for value in split["training"]),
        development=tuple(int(value) for value in split["development"]),
        final_evaluation=tuple(int(value) for value in split["final_evaluation"]),
        historical_evaluation=tuple(
            int(value) for value in split["historical_evaluation"]
        ),
    )
    seeds.validate()

    observation = data["observation"]
    order = tuple(str(value) for value in observation["ordering"])
    scales_table = observation["scales"]
    if set(order) != set(scales_table):
        raise ValueError("F10-L2 observation ordering and scales must match")
    scales = tuple(float(scales_table[name]) for name in order)
    if any(value <= 0.0 for value in scales):
        raise ValueError("F10-L2 observation scales must be positive")

    action = data["action"]
    action_bounds = (
        float(action["minimum_linear_velocity_mps"]),
        float(action["maximum_linear_velocity_mps"]),
        float(action["minimum_angular_velocity_rad_s"]),
        float(action["maximum_angular_velocity_rad_s"]),
    )
    if action_bounds != (0.0, 0.4, -4.0, 4.0):
        raise ValueError("F10-L2 must reuse the validated F2 action envelope")

    sac_data = data["sac"]
    sac = LaneSACSettings(
        hidden_sizes=tuple(int(value) for value in sac_data["hidden_sizes"]),
        learning_rate=float(sac_data["learning_rate"]),
        gamma=float(sac_data["gamma"]),
        tau=float(sac_data["tau"]),
        batch_size=int(sac_data["batch_size"]),
        replay_buffer_size=int(sac_data["replay_buffer_size"]),
        learning_starts=int(sac_data["learning_starts"]),
        train_frequency=int(sac_data["train_frequency"]),
        gradient_steps=int(sac_data["gradient_steps"]),
        initial_entropy_coefficient=float(sac_data["initial_entropy_coefficient"]),
        target_entropy=float(sac_data["target_entropy"]),
        training_steps=int(sac_data["training_steps"]),
        checkpoint_interval_steps=int(sac_data["checkpoint_interval_steps"]),
        training_seed=int(sac_data["training_seed"]),
        device=str(sac_data["device"]),
    )
    if sac.training_steps <= sac.learning_starts:
        raise ValueError("F10-L2 training budget must exceed buffer fill")
    if sac.checkpoint_interval_steps <= 0 or (
        sac.training_steps % sac.checkpoint_interval_steps
    ):
        raise ValueError("F10-L2 checkpoint interval must divide its budget")

    def relative(value: str) -> Path:
        return (config_path.parent / value).resolve()

    transfer = data["transfer"]
    if transfer["replay_buffer"] != "new_empty":
        raise ValueError("F10-L2 source replay buffer is unavailable")
    if transfer["buffer_fill_policy"] != "warm_start_policy_stochastic":
        raise ValueError("F10-L2 buffer must be filled by the warm-start policy")
    restored = (
        "load_actor",
        "load_critics",
        "load_entropy",
        "load_optimizer_states",
    )
    if not all(bool(transfer[name]) for name in restored):
        raise ValueError("F10-L2 must restore the complete SAC training state")

    protocol = LaneTransferProtocol(
        config_path=config_path,
        action_config_path=relative(str(provenance["action_config"])),
        action_config_sha256=str(provenance["action_config_sha256"]),
        environment_spec_path=relative(str(provenance["environment_spec"])),
        environment_spec_sha256=str(provenance["environment_spec_sha256"]),
        map_path=_installed_map_path(str(provenance["map_name"])),
        map_sha256=str(provenance["map_sha256"]),
        seeds=seeds,
        observation_order=order,
        observation_scales=scales,
        observation_clip=float(observation["clip_normalized"]),
        action_bounds=action_bounds,
        sac=sac,
        raw=data,
        transfer_checkpoint_path=relative(str(transfer["checkpoint"])),
        transfer_checkpoint_sha256=str(transfer["checkpoint_sha256"]),
        source_global_step=int(transfer["source_global_step"]),
    )
    if require_frozen:
        dependencies = (
            (protocol.action_config_path, protocol.action_config_sha256, "action config"),
            (
                protocol.environment_spec_path,
                protocol.environment_spec_sha256,
                "environment spec",
            ),
            (protocol.map_path, protocol.map_sha256, "experiment_loop map"),
            (
                protocol.transfer_checkpoint_path,
                protocol.transfer_checkpoint_sha256,
                "source checkpoint",
            ),
        )
        for dependency, expected, label in dependencies:
            if not dependency.is_file():
                raise FileNotFoundError(f"F10-L2 {label} is missing: {dependency}")
            actual = file_sha256(dependency)
            if actual != expected:
                raise RuntimeError(
                    f"frozen F10-L2 {label} hash mismatch: expected {expected}, got {actual}"
                )
    return protocol

