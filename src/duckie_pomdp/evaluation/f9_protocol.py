"""Configuration and split guards shared by the F9 calibration/evaluation runs."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10
    import tomli as tomllib

from duckie_pomdp.domain.action import PolicyAction
from duckie_pomdp.scenario import (
    MinimalPOMDPScenario,
    PedestrianMode,
    load_scenario,
)


@dataclass(frozen=True)
class F9ScenarioSpec:
    name: str
    pedestrian_mode: PedestrianMode
    action: PolicyAction
    steps: int
    ego_start_x_offset_m: float
    ego_heading_offset_rad: float
    use_for_calibration: bool
    use_for_final_evaluation: bool


@dataclass(frozen=True)
class F9DetectorConfig:
    confidence_threshold: float
    nms_iou_threshold: float
    image_size: int
    device: str | int
    max_detections: int


@dataclass(frozen=True)
class F9Protocol:
    config_path: Path
    checkpoint_path: Path
    checkpoint_sha256: str
    detector_dataset_manifest_path: Path
    scenario_path: Path
    annotation_config_path: Path
    frozen_f7_config_path: Path
    calibration_seeds: tuple[int, ...]
    final_evaluation_seeds: tuple[int, ...]
    detector: F9DetectorConfig
    image_width_px: int
    image_height_px: int
    domain_randomization: bool
    dynamics_randomization: bool
    calibration_capture_every_steps: int
    negative_capture_every_steps: int
    start_x_range_m: tuple[float, float]
    lateral_offset_range_m: tuple[float, float]
    heading_range_rad: tuple[float, float]
    scenarios: tuple[F9ScenarioSpec, ...]
    matching_iou_threshold: float
    minimum_samples_per_range_bin: int
    heteroscedastic_sigma_ratio_threshold: float
    detection_beta_prior_alpha: float
    detection_beta_prior_beta: float
    near_max_m: float
    medium_max_m: float
    parameters_frozen: bool
    calibration_artifact_path: Path
    calibration_artifact_sha256: str
    range_bias_m: float
    bearing_bias_rad: float
    active_belief_probability_threshold: float
    chi_square_2d_95_threshold: float
    error_cases_per_category: int
    artifacts: dict[str, Path]

    @property
    def config_sha256(self) -> str:
        return sha256(self.config_path)

    def scenario_for(
        self,
        seed: int,
        scenario_index: int,
    ) -> MinimalPOMDPScenario:
        spec = self.scenarios[scenario_index]
        base = load_scenario(self.scenario_path).with_pedestrian_mode(
            spec.pedestrian_mode
        )
        rng = np.random.default_rng(seed + 100_003 * scenario_index)
        base_pose = base.ego_start_pose_m
        return replace(
            base,
            seed=seed,
            ego_start_pose_m=(
                float(rng.uniform(*self.start_x_range_m))
                + spec.ego_start_x_offset_m,
                base_pose[1],
                base_pose[2] + float(rng.uniform(*self.lateral_offset_range_m)),
            ),
            ego_heading_rad=(
                float(rng.uniform(*self.heading_range_rad))
                + spec.ego_heading_offset_rad
            ),
        )

    def distance_bin(self, range_m: float) -> str:
        if range_m < self.near_max_m:
            return "near"
        if range_m < self.medium_max_m:
            return "medium"
        return "far"


def load_f9_protocol(
    path: str | Path,
    *,
    require_frozen: bool = False,
) -> F9Protocol:
    config_path = Path(path).resolve()
    with config_path.open("rb") as stream:
        data: dict[str, Any] = tomllib.load(stream)

    def relative(value: str) -> Path:
        return (config_path.parent / value).resolve()

    provenance = data["provenance"]
    split = data["split"]
    detector = data["detector"]
    simulator = data["simulator"]
    perturbation = data["trajectory_perturbation"]
    calibration = data["calibration_protocol"]
    measurement = data["measurement_model"]
    evaluation = data["evaluation"]
    protocol = F9Protocol(
        config_path=config_path,
        checkpoint_path=relative(str(provenance["checkpoint"])),
        checkpoint_sha256=str(provenance["checkpoint_sha256"]),
        detector_dataset_manifest_path=relative(
            str(provenance["detector_dataset_manifest"])
        ),
        scenario_path=relative(str(provenance["scenario"])),
        annotation_config_path=relative(str(provenance["annotation_rules"])),
        frozen_f7_config_path=relative(str(provenance["frozen_f7_config"])),
        calibration_seeds=tuple(int(value) for value in split["calibration_seeds"]),
        final_evaluation_seeds=tuple(
            int(value) for value in split["final_evaluation_seeds"]
        ),
        detector=F9DetectorConfig(
            confidence_threshold=float(detector["confidence_threshold"]),
            nms_iou_threshold=float(detector["nms_iou_threshold"]),
            image_size=int(detector["image_size"]),
            device=detector["device"],
            max_detections=int(detector["max_detections"]),
        ),
        image_width_px=int(simulator["image_width_px"]),
        image_height_px=int(simulator["image_height_px"]),
        domain_randomization=bool(simulator["domain_randomization"]),
        dynamics_randomization=bool(simulator["dynamics_randomization"]),
        calibration_capture_every_steps=int(
            simulator["calibration_capture_every_steps"]
        ),
        negative_capture_every_steps=int(simulator["negative_capture_every_steps"]),
        start_x_range_m=_bounds(perturbation["start_x_range_m"]),
        lateral_offset_range_m=_bounds(perturbation["lateral_offset_range_m"]),
        heading_range_rad=_bounds(perturbation["heading_range_rad"]),
        scenarios=tuple(_scenario_spec(item) for item in data["scenario_matrix"]),
        matching_iou_threshold=float(calibration["matching_iou_threshold"]),
        minimum_samples_per_range_bin=int(
            calibration["minimum_samples_per_range_bin"]
        ),
        heteroscedastic_sigma_ratio_threshold=float(
            calibration["heteroscedastic_sigma_ratio_threshold"]
        ),
        detection_beta_prior_alpha=float(
            calibration["detection_beta_prior_alpha"]
        ),
        detection_beta_prior_beta=float(calibration["detection_beta_prior_beta"]),
        near_max_m=float(calibration["distance_bin_near_max_m"]),
        medium_max_m=float(calibration["distance_bin_medium_max_m"]),
        parameters_frozen=bool(measurement["parameters_frozen"]),
        calibration_artifact_path=relative(str(measurement["calibration_artifact"])),
        calibration_artifact_sha256=str(
            measurement["calibration_artifact_sha256"]
        ),
        range_bias_m=float(measurement["range_bias_m"]),
        bearing_bias_rad=float(measurement["bearing_bias_rad"]),
        active_belief_probability_threshold=float(
            evaluation["active_belief_probability_threshold"]
        ),
        chi_square_2d_95_threshold=float(
            evaluation["chi_square_2d_95_threshold"]
        ),
        error_cases_per_category=int(evaluation["error_cases_per_category"]),
        artifacts={
            name: relative(str(value)) for name, value in data["artifacts"].items()
        },
    )
    _validate(protocol, data, require_frozen=require_frozen)
    return protocol


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _scenario_spec(item: dict[str, Any]) -> F9ScenarioSpec:
    steps = int(item["steps"])
    if steps <= 0:
        raise ValueError("F9 scenario steps must be positive")
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


def _bounds(values: list[float]) -> tuple[float, float]:
    bounds = tuple(float(value) for value in values)
    if len(bounds) != 2 or bounds[0] > bounds[1]:
        raise ValueError("F9 perturbation bounds must be ordered pairs")
    return bounds


def _validate(
    protocol: F9Protocol,
    data: dict[str, Any],
    *,
    require_frozen: bool,
) -> None:
    if sha256(protocol.checkpoint_path) != protocol.checkpoint_sha256:
        raise ValueError("frozen YOLO checkpoint hash mismatch")
    calibration = set(protocol.calibration_seeds)
    final = set(protocol.final_evaluation_seeds)
    if not calibration or not final or calibration & final:
        raise ValueError("F9 calibration and final seeds must be nonempty and disjoint")
    detector_manifest = json.loads(
        protocol.detector_dataset_manifest_path.read_text(encoding="utf-8")
    )
    detector_seeds = {
        int(seed)
        for seeds in detector_manifest["split_seeds"].values()
        for seed in seeds
    }
    if (calibration | final) & detector_seeds:
        raise ValueError("F9 seeds overlap detector train/validation/test data")
    if not 0.0 < protocol.matching_iou_threshold <= 1.0:
        raise ValueError("matching IoU threshold must be within (0, 1]")
    if protocol.negative_capture_every_steps <= 0:
        raise ValueError("negative capture interval must be positive")
    if protocol.near_max_m <= 0.0 or protocol.medium_max_m <= protocol.near_max_m:
        raise ValueError("F9 distance thresholds are invalid")
    if len({spec.name for spec in protocol.scenarios}) != len(protocol.scenarios):
        raise ValueError("F9 scenario names must be unique")
    if not any(spec.use_for_calibration for spec in protocol.scenarios):
        raise ValueError("F9 requires at least one calibration scenario")
    if not any(spec.use_for_final_evaluation for spec in protocol.scenarios):
        raise ValueError("F9 requires at least one final-evaluation scenario")

    with protocol.frozen_f7_config_path.open("rb") as stream:
        frozen_f7 = tomllib.load(stream)
    if data["ekf"] != frozen_f7["ekf"]:
        raise ValueError("F9 changes the frozen F7 EKF configuration")
    for key in ("prior_probability", "survival_probability", "birth_probability"):
        if data["existence"][key] != frozen_f7["existence"][key]:
            raise ValueError(f"F9 changes frozen existence parameter {key}")

    if require_frozen:
        if not protocol.parameters_frozen:
            raise ValueError("F9 measurement parameters are not frozen")
        if not protocol.calibration_artifact_sha256:
            raise ValueError("frozen F9 config lacks calibration artifact hash")
        if sha256(protocol.calibration_artifact_path) != (
            protocol.calibration_artifact_sha256
        ):
            raise ValueError("F9 calibration artifact hash mismatch")
