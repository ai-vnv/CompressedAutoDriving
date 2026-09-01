"""Frozen protocol loader for the F10-L1 lane curriculum experiment."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from duckie_pomdp.control.f10_protocol import file_sha256

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib


@dataclass(frozen=True)
class LaneSeedSplit:
    training: tuple[int, ...]
    development: tuple[int, ...]
    final_evaluation: tuple[int, ...]
    historical_evaluation: tuple[int, ...]

    def validate(self) -> None:
        active = {
            "training": set(self.training),
            "development": set(self.development),
            "final_evaluation": set(self.final_evaluation),
        }
        if any(not values for values in active.values()):
            raise ValueError("every F10-L1 seed split must be non-empty")
        for name, values in active.items():
            if len(values) != len(getattr(self, name)):
                raise ValueError(f"F10-L1 {name} contains duplicate seeds")
        names = tuple(active)
        for index, left in enumerate(names):
            for right in names[index + 1 :]:
                overlap = active[left] & active[right]
                if overlap:
                    raise ValueError(
                        f"F10-L1 seed leakage between {left} and {right}: "
                        f"{sorted(overlap)}"
                    )
        historical = set(self.historical_evaluation)
        for name, values in active.items():
            overlap = values & historical
            if overlap:
                raise ValueError(
                    f"F10-L1 {name} reuses historical seeds: {sorted(overlap)}"
                )


@dataclass(frozen=True)
class LaneSACSettings:
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
class LaneProtocol:
    config_path: Path
    action_config_path: Path
    action_config_sha256: str
    environment_spec_path: Path
    environment_spec_sha256: str
    map_path: Path
    map_sha256: str
    seeds: LaneSeedSplit
    observation_order: tuple[str, ...]
    observation_scales: tuple[float, ...]
    observation_clip: float
    action_bounds: tuple[float, float, float, float]
    sac: LaneSACSettings
    raw: dict[str, Any]


def _installed_map_path(map_name: str) -> Path:
    import duckietown_world

    package_root = Path(duckietown_world.__file__).resolve().parent
    return package_root / "data" / "gd1" / "maps" / f"{map_name}.yaml"


def load_lane_protocol(
    path: str | Path,
    *,
    require_frozen: bool = True,
) -> LaneProtocol:
    config_path = Path(path).resolve()
    with config_path.open("rb") as stream:
        data: dict[str, Any] = tomllib.load(stream)
    if int(data.get("schema_version", 0)) != 1:
        raise ValueError("unsupported F10-L1 schema_version")
    if data.get("stage") != "F10-L1":
        raise ValueError("lane curriculum config must declare stage F10-L1")

    provenance = data["provenance"]
    split = data["seed_split"]
    observation = data["observation"]
    order = tuple(str(value) for value in observation["ordering"])
    scales_table = observation["scales"]
    if set(order) != set(scales_table):
        raise ValueError("lane observation ordering and scales must match")
    scales = tuple(float(scales_table[name]) for name in order)
    if any(value <= 0.0 for value in scales):
        raise ValueError("lane observation scales must be positive")

    seeds = LaneSeedSplit(
        training=tuple(int(value) for value in split["training"]),
        development=tuple(int(value) for value in split["development"]),
        final_evaluation=tuple(int(value) for value in split["final_evaluation"]),
        historical_evaluation=tuple(
            int(value) for value in split["historical_evaluation"]
        ),
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
        raise ValueError("F10-L1 must reuse the F2 action envelope")
    simulator = data["simulator"]
    if simulator["map"] != "small_loop":
        raise ValueError("F10-L1 is restricted to small_loop")
    if simulator["direction"] != "counterclockwise":
        raise ValueError("F10-L1 direction must be counterclockwise")

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
        initial_entropy_coefficient=float(
            sac_data["initial_entropy_coefficient"]
        ),
        target_entropy=float(sac_data["target_entropy"]),
        training_steps=int(sac_data["training_steps"]),
        checkpoint_interval_steps=int(sac_data["checkpoint_interval_steps"]),
        training_seed=int(sac_data["training_seed"]),
        device=str(sac_data["device"]),
    )
    if sac.training_steps <= sac.learning_starts:
        raise ValueError("training steps must exceed learning starts")
    if (
        sac.checkpoint_interval_steps <= 0
        or sac.training_steps % sac.checkpoint_interval_steps
    ):
        raise ValueError("checkpoint interval must divide the training budget")

    def relative(value: str) -> Path:
        return (config_path.parent / value).resolve()

    map_path = _installed_map_path(str(provenance["map_name"]))
    protocol = LaneProtocol(
        config_path=config_path,
        action_config_path=relative(str(provenance["action_config"])),
        action_config_sha256=str(provenance["action_config_sha256"]),
        environment_spec_path=relative(str(provenance["environment_spec"])),
        environment_spec_sha256=str(provenance["environment_spec_sha256"]),
        map_path=map_path,
        map_sha256=str(provenance["map_sha256"]),
        seeds=seeds,
        observation_order=order,
        observation_scales=scales,
        observation_clip=float(observation["clip_normalized"]),
        action_bounds=action_bounds,
        sac=sac,
        raw=data,
    )
    if require_frozen:
        checks = (
            (
                protocol.action_config_path,
                protocol.action_config_sha256,
                "action config",
            ),
            (
                protocol.environment_spec_path,
                protocol.environment_spec_sha256,
                "environment spec",
            ),
            (protocol.map_path, protocol.map_sha256, "small_loop map"),
        )
        for dependency, expected, label in checks:
            if not dependency.is_file():
                raise FileNotFoundError(f"F10-L1 {label} is missing: {dependency}")
            actual = file_sha256(dependency)
            if actual != expected:
                raise RuntimeError(
                    f"frozen F10-L1 {label} hash mismatch: "
                    f"expected {expected}, got {actual}"
                )
    return protocol

