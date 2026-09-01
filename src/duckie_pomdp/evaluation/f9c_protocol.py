"""Configuration and freeze-boundary guards for the F9c robust-belief runs.

F9c reuses the F7-frozen EKF dynamics and existence-track algorithm and only
replaces the observation model (bias, covariance, and conditional detection
probability). This module makes it impossible to load an F9c config that
silently edits the frozen F7 boundary, and it keeps the F9c seed split
disjoint from every earlier F7/F8/F9/F9b split.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10
    import tomli as tomllib

from duckie_pomdp.evaluation.f9_protocol import F9ScenarioSpec, sha256
from duckie_pomdp.scenario import PedestrianMode
from duckie_pomdp.domain.action import PolicyAction


@dataclass(frozen=True)
class RobustObservationSwitches:
    bias_refit: bool
    innovation_gate: bool
    temporal_association: bool
    covariance_calibration: bool
    conditional_detection: bool


@dataclass(frozen=True)
class AcceptanceBands:
    coverage_68_low: float
    coverage_68_high: float
    coverage_95_low: float
    coverage_95_high: float
    max_std_over_rmse: float
    max_rmse_ratio_vs_baseline: float


@dataclass(frozen=True)
class F9cProtocol:
    config_path: Path
    checkpoint_path: Path
    checkpoint_sha256: str
    frozen_f7_config_path: Path
    calibration_seeds: tuple[int, ...]
    final_evaluation_seeds: tuple[int, ...]
    forbidden_seeds: tuple[int, ...]
    scenarios: tuple[F9ScenarioSpec, ...]
    robust: RobustObservationSwitches
    acceptance: AcceptanceBands
    minimum_support: dict[str, int]
    artifacts: dict[str, Path]

    @property
    def config_sha256(self) -> str:
        return sha256(self.config_path)


def load_f9c_protocol(
    path: str | Path,
    *,
    require_frozen: bool = False,
) -> F9cProtocol:
    config_path = Path(path).resolve()
    with config_path.open("rb") as stream:
        data: dict[str, Any] = tomllib.load(stream)

    def relative(value: str) -> Path:
        return (config_path.parent / value).resolve()

    provenance = data["provenance"]
    split = data["split"]
    robust_observation = data["robust_observation"]
    acceptance = data["acceptance"]
    protocol = F9cProtocol(
        config_path=config_path,
        checkpoint_path=relative(str(provenance["checkpoint"])),
        checkpoint_sha256=str(provenance["checkpoint_sha256"]),
        frozen_f7_config_path=relative(str(provenance["frozen_f7_config"])),
        calibration_seeds=tuple(int(value) for value in split["calibration_seeds"]),
        final_evaluation_seeds=tuple(
            int(value) for value in split["final_evaluation_seeds"]
        ),
        forbidden_seeds=tuple(int(value) for value in split["forbidden_seeds"]),
        scenarios=tuple(_scenario_spec(item) for item in data["scenario_matrix"]),
        robust=RobustObservationSwitches(
            bias_refit=bool(robust_observation["bias_refit"]),
            innovation_gate=bool(robust_observation["innovation_gate"]),
            temporal_association=bool(robust_observation["temporal_association"]),
            covariance_calibration=bool(
                robust_observation["covariance_calibration"]
            ),
            conditional_detection=bool(
                robust_observation["conditional_detection"]
            ),
        ),
        acceptance=AcceptanceBands(
            coverage_68_low=float(acceptance["coverage_68_low"]),
            coverage_68_high=float(acceptance["coverage_68_high"]),
            coverage_95_low=float(acceptance["coverage_95_low"]),
            coverage_95_high=float(acceptance["coverage_95_high"]),
            max_std_over_rmse=float(acceptance["max_std_over_rmse"]),
            max_rmse_ratio_vs_baseline=float(
                acceptance["max_rmse_ratio_vs_baseline"]
            ),
        ),
        minimum_support={
            name: int(value) for name, value in data["minimum_support"].items()
        },
        artifacts={
            name: relative(str(value)) for name, value in data["artifacts"].items()
        },
    )
    _validate(protocol, data, require_frozen=require_frozen)
    return protocol


def _scenario_spec(item: dict[str, Any]) -> F9ScenarioSpec:
    steps = int(item["steps"])
    if steps <= 0:
        raise ValueError("F9c scenario steps must be positive")
    return F9ScenarioSpec(
        name=str(item["name"]),
        pedestrian_mode=PedestrianMode(str(item["pedestrian_mode"])),
        action=PolicyAction(
            float(item["linear_velocity_mps"]),
            float(item["angular_velocity_rad_s"]),
        ),
        steps=steps,
        ego_start_x_offset_m=float(item.get("ego_start_x_offset_m", 0.0)),
        ego_heading_offset_rad=float(item.get("ego_heading_offset_rad", 0.0)),
        use_for_calibration=bool(item.get("use_for_calibration", True)),
        use_for_final_evaluation=bool(
            item.get("use_for_final_evaluation", True)
        ),
    )


def _validate(
    protocol: F9cProtocol,
    data: dict[str, Any],
    *,
    require_frozen: bool,
) -> None:
    if sha256(protocol.checkpoint_path) != protocol.checkpoint_sha256:
        raise ValueError("frozen YOLO checkpoint hash mismatch")

    with protocol.frozen_f7_config_path.open("rb") as stream:
        frozen_f7 = tomllib.load(stream)

    if data["ekf"] != frozen_f7["ekf"]:
        raise ValueError("F9c changes frozen F7 dynamics")
    for key in ("prior_probability", "survival_probability", "birth_probability"):
        if data["existence"][key] != frozen_f7["existence"][key]:
            raise ValueError(f"F9c changes frozen F7 existence parameter {key}")
    # F9c deliberately does NOT validate existence.detection_probability
    # against the frozen F7 config: that parameter is intentionally unfrozen
    # for F9c's conditional-detection calibration work.

    calibration = set(protocol.calibration_seeds)
    final = set(protocol.final_evaluation_seeds)
    forbidden = set(protocol.forbidden_seeds)
    if not calibration or not final or calibration & final:
        raise ValueError(
            "F9c calibration and final evaluation seeds must be nonempty and disjoint"
        )
    if (calibration | final) & forbidden:
        raise ValueError("F9c seeds overlap a forbidden earlier-split seed")

    provenance = data["provenance"]
    detector_manifest_path = (
        protocol.config_path.parent / str(provenance["detector_dataset_manifest"])
    ).resolve()
    detector_manifest = json.loads(
        detector_manifest_path.read_text(encoding="utf-8")
    )
    detector_seeds = {
        int(seed)
        for seeds in detector_manifest["split_seeds"].values()
        for seed in seeds
    }
    if (calibration | final) & detector_seeds:
        raise ValueError("F9c seeds overlap detector train/validation/test data")

    if require_frozen:
        measurement_frozen = bool(data["measurement_model"]["parameters_frozen"])
        covariance_frozen = bool(
            data["covariance_calibration"]["parameters_frozen"]
        )
        conditional_frozen = bool(
            data["conditional_detection"]["parameters_frozen"]
        )
        if not (measurement_frozen and covariance_frozen and conditional_frozen):
            raise ValueError("F9c measurement/covariance/detection parameters are not frozen")
        frozen_config_path = protocol.artifacts["frozen_config_json"]
        if not frozen_config_path.exists():
            raise ValueError("frozen F9c config artifact is missing")
        frozen_config = json.loads(frozen_config_path.read_text(encoding="utf-8"))
        if frozen_config.get("config_sha256") != protocol.config_sha256:
            raise ValueError("frozen F9c config artifact hash mismatch")
