"""Frozen, validated protocol loader for the F10 SAC experiment."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256 as _sha256
from pathlib import Path
from typing import Any

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib


@dataclass(frozen=True)
class SeedSplit:
    training: tuple[int, ...]
    development: tuple[int, ...]
    final_evaluation: tuple[int, ...]
    historical_evaluation: tuple[int, ...]

    def validate(self) -> None:
        groups = {
            "training": set(self.training),
            "development": set(self.development),
            "final_evaluation": set(self.final_evaluation),
        }
        if any(not values for values in groups.values()):
            raise ValueError("every F10 seed split must be non-empty")
        if any(len(values) != len(getattr(self, name)) for name, values in groups.items()):
            raise ValueError("seed splits cannot contain duplicates")
        pairs = (("training", "development"), ("training", "final_evaluation"), ("development", "final_evaluation"))
        for left, right in pairs:
            overlap = groups[left] & groups[right]
            if overlap:
                raise ValueError(f"F10 seed split leakage between {left} and {right}: {sorted(overlap)}")
        historical = set(self.historical_evaluation)
        for name, values in groups.items():
            overlap = values & historical
            if overlap:
                raise ValueError(f"F10 {name} seeds reuse historical evaluation seeds: {sorted(overlap)}")


@dataclass(frozen=True)
class SACSettings:
    hidden_sizes: tuple[int, ...]
    learning_rate: float
    gamma: float
    tau: float
    batch_size: int
    replay_buffer_size: int
    learning_starts: int
    train_frequency: int
    gradient_steps: int
    initial_entropy_coefficient: float
    target_entropy: float
    training_steps: int
    checkpoint_interval_steps: int
    training_seed: int
    device: str


@dataclass(frozen=True)
class F10Protocol:
    config_path: Path
    scenario_path: Path
    action_config_path: Path
    action_config_sha256: str
    belief_config_path: Path
    belief_config_sha256: str
    detector_checkpoint_path: Path
    detector_checkpoint_sha256: str
    seeds: SeedSplit
    observation_order: tuple[str, ...]
    observation_scales: tuple[float, ...]
    observation_clip: float
    action_bounds: tuple[float, float, float, float]
    sac: SACSettings
    raw: dict[str, Any]


def file_sha256(path: str | Path) -> str:
    digest = _sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_f10_protocol(path: str | Path, *, require_frozen: bool = True) -> F10Protocol:
    config_path = Path(path).resolve()
    with config_path.open("rb") as stream:
        data: dict[str, Any] = tomllib.load(stream)
    if int(data.get("schema_version", 0)) != 1:
        raise ValueError("unsupported F10 schema_version")

    def relative(value: str) -> Path:
        return (config_path.parent / value).resolve()

    provenance = data["provenance"]
    split = data["seed_split"]
    observation = data["observation"]
    order = tuple(str(value) for value in observation["ordering"])
    scales_table = observation["scales"]
    if set(order) != set(scales_table):
        raise ValueError("observation ordering and scale keys must match exactly")
    scales = tuple(float(scales_table[name]) for name in order)
    if any(value <= 0.0 for value in scales):
        raise ValueError("observation scales must be positive")

    seeds = SeedSplit(
        training=tuple(int(value) for value in split["training"]),
        development=tuple(int(value) for value in split["development"]),
        final_evaluation=tuple(int(value) for value in split["final_evaluation"]),
        historical_evaluation=tuple(int(value) for value in split["historical_evaluation"]),
    )
    seeds.validate()

    action = data["action"]
    action_bounds = (
        float(action["minimum_linear_velocity_mps"]),
        float(action["maximum_linear_velocity_mps"]),
        float(action["minimum_angular_velocity_rad_s"]),
        float(action["maximum_angular_velocity_rad_s"]),
    )
    if action_bounds != (0.0, 0.4, -4.0, 4.0):
        raise ValueError("F10 action bounds must reuse the F2-validated envelope")

    sac_data = data["sac"]
    sac = SACSettings(
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
        raise ValueError("training_steps must exceed learning_starts")
    if sac.checkpoint_interval_steps <= 0 or sac.training_steps % sac.checkpoint_interval_steps:
        raise ValueError("checkpoint interval must divide the training budget")

    protocol = F10Protocol(
        config_path=config_path,
        scenario_path=relative(str(provenance["scenario"])),
        action_config_path=relative(str(provenance["action_config"])),
        action_config_sha256=str(provenance["action_config_sha256"]),
        belief_config_path=relative(str(provenance["frozen_belief_config"])),
        belief_config_sha256=str(provenance["frozen_belief_config_sha256"]),
        detector_checkpoint_path=relative(str(provenance["detector_checkpoint"])),
        detector_checkpoint_sha256=str(provenance["detector_checkpoint_sha256"]),
        seeds=seeds,
        observation_order=order,
        observation_scales=scales,
        observation_clip=float(observation["clip_normalized"]),
        action_bounds=action_bounds,
        sac=sac,
        raw=data,
    )
    _validate_paths(protocol, require_frozen=require_frozen)
    return protocol


def _validate_paths(protocol: F10Protocol, *, require_frozen: bool) -> None:
    for path in (protocol.scenario_path, protocol.action_config_path, protocol.belief_config_path, protocol.detector_checkpoint_path):
        if not path.is_file():
            raise FileNotFoundError(f"F10 dependency does not exist: {path}")
    if not require_frozen:
        return
    checks = (
        (protocol.action_config_path, protocol.action_config_sha256, "action config"),
        (protocol.belief_config_path, protocol.belief_config_sha256, "belief config"),
        (protocol.detector_checkpoint_path, protocol.detector_checkpoint_sha256, "YOLO checkpoint"),
    )
    for path, expected, label in checks:
        actual = file_sha256(path)
        if actual != expected:
            raise RuntimeError(f"frozen F10 {label} hash mismatch: expected {expected}, got {actual}")

