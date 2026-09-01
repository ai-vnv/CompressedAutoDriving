"""Frozen protocol contracts for the staged F10-PPO curriculum."""

from __future__ import annotations

import json
from dataclasses import dataclass
from importlib.util import find_spec
from pathlib import Path
from typing import Any
from xml.etree import ElementTree

from duckie_pomdp.scenario import load_scenario

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib

from .f10_protocol import file_sha256


STAGE_NAMES = ("c0", "c1", "c2", "c3", "c4")
SPLIT_NAMES = ("training", "development", "stage_final")
PRETRAINING_SOURCE_PATHS = (
    "configs/f10_ppo_v1.toml",
    "docs/F10_PPO_FORMULATION.md",
    "docs/F10_PPO_CURRICULUM.md",
    "src/duckie_pomdp/adapters/gym_duckietown.py",
    "src/duckie_pomdp/control/action_mapping.py",
    "src/duckie_pomdp/control/belief_runtime.py",
    "src/duckie_pomdp/control/ppo.py",
    "src/duckie_pomdp/control/ppo_environment.py",
    "src/duckie_pomdp/control/ppo_observation.py",
    "src/duckie_pomdp/control/ppo_protocol.py",
    "src/duckie_pomdp/control/ppo_reward.py",
    "src/duckie_pomdp/control/reward.py",
    "src/duckie_pomdp/control/stop_belief.py",
    "src/duckie_pomdp/evaluation/f10_ppo_policy.py",
    "experiments/audit_f10_ppo_reward.py",
    "experiments/evaluate_f10_ppo.py",
    "experiments/train_f10_ppo.py",
)
PRETRAINING_EVIDENCE_PATHS = (
    "artifacts/f10_ppo/c0/reward_audit_final.json",
    "artifacts/f10_ppo/c0/smoke_final/training_run_manifest.json",
    "artifacts/f10_ppo/pretraining_full_tests.log",
    "artifacts/f10_ppo/pretraining_tests.xml",
    "artifacts/f10_ppo/wandb_preflight.json",
    "artifacts/f10_ppo/environment_profile.json",
    "artifacts/f10_ppo/agent_follows_doc_pass.json",
)

VISUAL_PRETRAINING_SOURCE_PATHS = (
    "configs/f10_ppo_env_v1.json",
    "configs/action_v1.toml",
    "configs/f9c_robust_belief_v1.toml",
    "configs/scenario_pomdp_v1.toml",
    # lane_belief_config is intentionally NOT hardcoded here: it is derived
    # per-protocol in pretraining_source_paths() from
    # protocol.lane_belief_config_path, because v4 protocols may point it at
    # lane_belief_v2.toml (an *extension* config chained to v1 via
    # source_config/source_config_sha256 -- see
    # lane_belief_runtime._resolve_filter_config_path) instead of v1
    # directly. Deriving it keeps a v3 protocol's frozen-source inventory
    # (and any already-recorded pretraining gate evidence for it) exactly
    # what it was before v4 existed, while a v4 protocol picks up both the
    # extension config and, through the chain, v1 underneath it.
    "src/duckie_pomdp/adapters/gym_duckietown.py",
    "src/duckie_pomdp/belief/lane_ekf.py",
    "src/duckie_pomdp/control/action_mapping.py",
    "src/duckie_pomdp/control/belief_runtime.py",
    "src/duckie_pomdp/control/lane_belief_runtime.py",
    "src/duckie_pomdp/control/ppo.py",
    "src/duckie_pomdp/control/ppo_environment.py",
    "src/duckie_pomdp/control/ppo_observation.py",
    "src/duckie_pomdp/control/ppo_protocol.py",
    "src/duckie_pomdp/control/ppo_reward.py",
    "src/duckie_pomdp/control/reward.py",
    "src/duckie_pomdp/control/stop_belief.py",
    "src/duckie_pomdp/domain/belief.py",
    "src/duckie_pomdp/domain/measurement.py",
    "src/duckie_pomdp/evaluation/f10_ppo_policy.py",
    "src/duckie_pomdp/perception/lane_measurement.py",
    "experiments/audit_f10_ppo_reward.py",
    "experiments/build_f10_ppo_pretraining_gate.py",
    "experiments/evaluate_f10_ppo.py",
    "experiments/preflight_f10_ppo_wandb.py",
    "experiments/profile_f10_ppo_environment.py",
    "experiments/audit_f10_ppo_reset_memory.py",
    "experiments/train_f10_ppo.py",
)

VISUAL_MAXIMUM_STAGE = "c1"


def _uses_separated_object_curriculum(protocol: "PPOCurriculumProtocol") -> bool:
    settings = protocol.raw.get("object_curriculum", {})
    return bool(
        settings.get("objects_separated") is True
        and settings.get("duckie_path_must_intersect_ego_route") is True
    )


@dataclass(frozen=True)
class PPOSettings:
    hidden_sizes: tuple[int, ...]
    learning_rate: float
    n_steps: int
    batch_size: int
    n_epochs: int
    gamma: float
    gae_lambda: float
    clip_range: float
    entropy_coefficient: float
    value_function_coefficient: float
    max_gradient_norm: float
    initial_log_std: float
    training_seed: int
    device: str


@dataclass(frozen=True)
class CurriculumStage:
    key: str
    name: str
    map_name: str
    scenario_config_path: Path | None
    pedestrian_active: bool
    stop_active: bool
    domain_randomization: bool
    pedestrian_modes: tuple[str, ...]
    episode_horizon_steps: int
    training_steps: int
    checkpoint_interval_steps: int
    training_seeds: tuple[int, ...]
    development_seeds: tuple[int, ...]
    stage_final_seeds: tuple[int, ...]

    def seeds_for(self, split: str) -> tuple[int, ...]:
        if split not in SPLIT_NAMES:
            raise ValueError(f"invalid curriculum split: {split}")
        return tuple(getattr(self, f"{split}_seeds"))


@dataclass(frozen=True)
class PPOCurriculumProtocol:
    config_path: Path
    scenario_path: Path
    pomdp_map_path: Path
    action_config_path: Path
    action_config_sha256: str
    belief_config_path: Path
    belief_config_sha256: str
    lane_belief_config_path: Path | None
    lane_belief_config_sha256: str | None
    detector_checkpoint_path: Path
    detector_checkpoint_sha256: str
    observation_order: tuple[str, ...]
    observation_scales: tuple[float, ...]
    observation_clip: float
    action_bounds: tuple[float, float, float, float]
    ppo: PPOSettings
    stages: dict[str, CurriculumStage]
    global_final: dict[str, tuple[int, ...]]
    historical_seeds: tuple[int, ...]
    raw: dict[str, Any]

    def stage(self, key: str) -> CurriculumStage:
        try:
            return self.stages[key.lower()]
        except KeyError as error:
            raise ValueError(f"unknown curriculum stage: {key}") from error


def protocol_artifact_root(protocol: PPOCurriculumProtocol) -> Path:
    """Resolve the experiment-specific artifact root from the frozen config."""

    value = str(protocol.raw["artifacts"]["directory"])
    return (protocol.config_path.parent / value).resolve()


def _lane_belief_config_chain_sources(
    lane_belief_config_path: Path, project_root: Path
) -> tuple[str, ...]:
    """The lane-belief config plus, if it is a v2-style extension, its source.

    ``lane_belief_v1.toml`` is a complete config and is returned alone (this
    is the only case v3 ever hits, so a v3 protocol's frozen-source
    inventory is unchanged from before v4 existed). A v2-style *extension*
    config additionally names a ``source_config`` -- e.g.
    ``lane_belief_v2.toml`` pointing back at ``lane_belief_v1.toml`` -- which
    is also a frozen source (it is what
    ``lane_belief_runtime._resolve_filter_config_path`` loads the point
    extractor / bias calibration / EKF process noise from at runtime) and is
    included here too.
    """

    with lane_belief_config_path.open("rb") as handle:
        raw = tomllib.load(handle)
    sources = (lane_belief_config_path.relative_to(project_root).as_posix(),)
    source_config = raw.get("source_config")
    if source_config is None:
        return sources
    source_path = (lane_belief_config_path.parent / str(source_config)).resolve()
    return sources + (source_path.relative_to(project_root).as_posix(),)


def pretraining_source_paths(protocol: PPOCurriculumProtocol) -> tuple[str, ...]:
    representation = str(
        protocol.raw["observation"].get("representation", "legacy_state_v1")
    )
    if representation == "legacy_state_v1":
        return PRETRAINING_SOURCE_PATHS
    if representation == "visual_lane_belief_v2":
        # Bind the complete runtime package instead of maintaining a fragile
        # hand-written import closure. Any package edit or addition then
        # invalidates the C0/C1 launch gate.
        project_root = protocol.config_path.parent.parent
        formulation_value = str(
            protocol.raw["provenance"].get(
                "formulation_doc", "../docs/F10_PPO_VISUAL_FORMULATION.md"
            )
        )
        formulation_path = (protocol.config_path.parent / formulation_value).resolve()
        try:
            protocol_sources = (
                protocol.config_path.relative_to(project_root).as_posix(),
                formulation_path.relative_to(project_root).as_posix(),
            )
            parent_name = protocol.raw.get("extends")
            if parent_name is not None:
                parent_path = (
                    protocol.config_path.parent / str(parent_name)
                ).resolve()
                protocol_sources = protocol_sources + (
                    parent_path.relative_to(project_root).as_posix(),
                )
            if protocol.lane_belief_config_path is not None:
                protocol_sources = protocol_sources + _lane_belief_config_chain_sources(
                    protocol.lane_belief_config_path, project_root
                )
        except ValueError as error:
            raise ValueError(
                "visual PPO config/formulation must stay inside the project"
            ) from error
        if not formulation_path.is_file():
            raise FileNotFoundError(formulation_path)
        package_sources = tuple(
            sorted(
                path.relative_to(project_root).as_posix()
                for path in (project_root / "src" / "duckie_pomdp").rglob("*.py")
            )
        )
        scenario_sources: list[str] = []
        for stage in protocol.stages.values():
            scenario_path = stage.scenario_config_path
            if scenario_path is None:
                continue
            scenario = load_scenario(scenario_path)
            scenario_sources.extend(
                (
                    scenario_path.relative_to(project_root).as_posix(),
                    scenario.map_path.relative_to(project_root).as_posix(),
                )
            )
        codex_lane_generators = ()
        if (
            protocol.lane_belief_config_path is not None
            and protocol.lane_belief_config_path.name == "lane_belief_v3_codex.toml"
        ):
            # These programs created the calibration and once-only held-out
            # evidence consumed by the Codex visual-lane gate. Bind their
            # sources too, without invalidating earlier visual-v2/v3 gates.
            codex_lane_generators = (
                "experiments/calibrate_visual_lane_codex_v4.py",
                "experiments/validate_visual_lane_codex_v4.py",
            )
        elif (
            protocol.lane_belief_config_path is not None
            and protocol.lane_belief_config_path.name
            == "lane_belief_v4_transfer.toml"
        ):
            codex_lane_generators = (
                "experiments/calibrate_lane_transfer_v5.py",
                "experiments/validate_lane_transfer_v5.py",
                "experiments/diagnose_c1_lane_transfer.py",
            )
        elif (
            protocol.lane_belief_config_path is not None
            and protocol.lane_belief_config_path.name
            == "lane_belief_v8_competence_rgb.toml"
        ):
            codex_lane_generators = (
                "configs/lane_rgb_train_v3_competence.toml",
                "experiments/generate_lane_rgb_competence_v9.py",
                "experiments/build_lane_rgb_combined_v9.py",
                "experiments/train_lane_rgb_v7.py",
                "experiments/validate_lane_rgb_v7.py",
                "experiments/validate_lane_rgb_closed_loop_v7.py",
                "experiments/diagnose_lane_rgb_closed_loop_v8.py",
            )
        object_generators = ()
        if _uses_separated_object_curriculum(protocol):
            object_generators = (
                "experiments/validate_experiment_loop_objects_v10.py",
                "experiments/audit_f10_ppo_object_reset_memory.py",
            )
            if protocol.raw.get("runtime_detection"):
                object_generators += (
                    "experiments/validate_c4_duckie_confidence_filter.py",
                )
                if protocol.raw["runtime_detection"].get(
                    "duckie_maximum_bottom_y_px"
                ) is not None:
                    object_generators += (
                        "experiments/analyze_c4_duckie_false_positives.py",
                        "experiments/validate_c4_duckie_image_domain.py",
                    )
            if protocol.raw.get("behavior_warm_start"):
                object_generators += (
                    "experiments/build_ppo_behavior_warm_start.py",
                    "experiments/freeze_behavior_source_prefix.py",
                )
                warm = protocol.raw.get("behavior_warm_start", {}).get("c2", {})
                if warm.get("multitask_distillation") is True:
                    object_generators += (
                        "experiments/freeze_ppo_multitask_behavior_source.py",
                        "experiments/build_ppo_multitask_behavior_warm_start.py",
                        "experiments/validate_ppo_multitask_distillation.py",
                    )
            warm_c3 = protocol.raw.get("behavior_warm_start", {}).get("c3", {})
            if warm_c3.get("enabled"):
                object_generators += (
                    "experiments/build_ppo_c3_behavior_warm_start.py",
                )
                if warm_c3.get("dagger") is True:
                    object_generators += (
                        "experiments/build_ppo_c3_dagger_warm_start.py",
                        "experiments/validate_ppo_c3_dagger_distillation.py",
                    )
            warm_c4 = protocol.raw.get("behavior_warm_start", {}).get("c4", {})
            if warm_c4.get("enabled") and warm_c4.get("dagger") is True:
                if warm_c4.get("precomputed_conditional_distillation") is True:
                    object_generators += (
                        "experiments/build_ppo_conditional_rehearsal_v27.py",
                        "experiments/distill_ppo_conditional_actor_v27.py",
                        "experiments/screen_ppo_actor_interpolation_v26.py",
                    )
                if warm_c4.get("precomputed_existence_gated_distillation") is True:
                    object_generators += (
                        "experiments/build_ppo_belief_gated_rehearsal_v28.py",
                        "experiments/distill_ppo_pedestrian_adapter_v29.py",
                        "experiments/diagnose_c4_policy_divergence.py",
                        "experiments/screen_ppo_actor_interpolation_v26.py",
                    )
                if warm_c4.get("cumulative_policy_rehearsal") is True:
                    object_generators += (
                        (
                            "experiments/build_ppo_c4_privileged_guidance_v24.py"
                            if warm_c4.get("privileged_teacher_guidance") is True
                            else "experiments/build_ppo_c4_cumulative_dagger_v23.py"
                        ),
                        "experiments/validate_ppo_c4_cumulative_dagger_v23.py",
                        "experiments/visualize_ppo_behavior_dataset.py",
                    )
                else:
                    object_generators += (
                        "experiments/build_ppo_c4_dagger_warm_start.py",
                        "experiments/validate_ppo_c4_dagger_distillation.py",
                    )
        return tuple(
            dict.fromkeys(
                protocol_sources
                + VISUAL_PRETRAINING_SOURCE_PATHS
                + codex_lane_generators
                + object_generators
                + tuple(scenario_sources)
                + package_sources
            )
        )
    raise ValueError(f"unsupported PPO observation representation: {representation}")


def require_stage_in_protocol_scope(
    protocol: PPOCurriculumProtocol,
    stage_key: str,
) -> None:
    """Fail closed when an experiment-specific protocol reaches its stop gate."""

    key = stage_key.lower()
    if key not in STAGE_NAMES:
        raise ValueError(f"unknown curriculum stage: {stage_key}")
    representation = str(
        protocol.raw["observation"].get("representation", "legacy_state_v1")
    )
    configured_maximum = str(
        protocol.raw.get("protocol_scope", {}).get(
            "maximum_stage", VISUAL_MAXIMUM_STAGE
        )
    ).lower()
    if configured_maximum not in STAGE_NAMES:
        raise ValueError(f"invalid protocol_scope.maximum_stage: {configured_maximum}")
    if (
        representation == "visual_lane_belief_v2"
        and STAGE_NAMES.index(key) > STAGE_NAMES.index(configured_maximum)
    ):
        raise RuntimeError(
            f"visual-lane v2 experiment scope stops after {configured_maximum.upper()}; "
            f"{key.upper()} requires a separately reviewed gate"
        )


def pretraining_evidence_paths(protocol: PPOCurriculumProtocol) -> tuple[str, ...]:
    project_root = protocol.config_path.parent.parent
    artifact_root = protocol_artifact_root(protocol)
    try:
        prefix = artifact_root.relative_to(project_root).as_posix()
    except ValueError as error:
        raise ValueError("PPO artifact directory must stay inside the project") from error
    if _uses_separated_object_curriculum(protocol):
        maximum_stage = str(
            protocol.raw.get("protocol_scope", {}).get(
                "maximum_stage", VISUAL_MAXIMUM_STAGE
            )
        ).lower()
        includes_c4 = STAGE_NAMES.index(maximum_stage) >= STAGE_NAMES.index("c4")
        warm_c3 = protocol.raw.get("behavior_warm_start", {}).get("c3", {})
        smoke_stage = (
            "c4" if includes_c4 else ("c3" if warm_c3.get("enabled") else "c2")
        )
        reward_paths = (
            f"{prefix}/c2/reward_audit.json",
            f"{prefix}/c3/reward_audit.json",
        ) + ((f"{prefix}/c4/reward_audit.json",) if includes_c4 else ())
        paths = reward_paths + (
            f"{prefix}/{smoke_stage}/smoke/training_run_manifest.json",
            f"{prefix}/pretraining_tests.log",
            f"{prefix}/pretraining_tests.xml",
            f"{prefix}/wandb_preflight.json",
            f"{prefix}/environment_profile.json",
            f"{prefix}/object_scenario_gate.json",
            f"{prefix}/object_reset_memory_audit.json",
        )
        if protocol.raw.get("runtime_detection"):
            paths += (f"{prefix}/duckie_confidence_gate.json",)
            if protocol.raw["runtime_detection"].get(
                "duckie_maximum_bottom_y_px"
            ) is not None:
                paths += (
                    f"{prefix}/duckie_bbox_audit.csv",
                    f"{prefix}/duckie_image_domain_gate.json",
                )
        warm = protocol.raw.get("behavior_warm_start", {}).get("c2", {})
        if warm.get("enabled"):
            paths += (
                f"{prefix}/c2/behavior_warm_start.npz",
                f"{prefix}/c2/behavior_warm_start_manifest.json",
                f"{prefix}/c2/behavior_warm_start_source.csv",
            )
            if warm.get("multitask_distillation") is True:
                paths += (f"{prefix}/c2/multitask_distillation_gate.json",)
        if warm_c3.get("enabled"):
            for key in ("dataset", "manifest", "source_csv"):
                configured = (
                    protocol.config_path.parent / str(warm_c3[key])
                ).resolve()
                try:
                    relative = configured.relative_to(project_root).as_posix()
                except ValueError as error:
                    raise ValueError(
                        "C3 behavior evidence must stay inside the project"
                    ) from error
                paths += (relative,)
            if warm_c3.get("dagger") is True:
                learner = (
                    protocol.config_path.parent
                    / str(warm_c3["learner_checkpoint"])
                ).resolve()
                try:
                    paths += (learner.relative_to(project_root).as_posix(),)
                except ValueError as error:
                    raise ValueError(
                        "C3 DAgger learner checkpoint must stay inside the project"
                    ) from error
                paths += (f"{prefix}/c3/dagger_distillation_gate.json",)
        warm_c4 = protocol.raw.get("behavior_warm_start", {}).get("c4", {})
        if warm_c4.get("enabled"):
            for key in ("dataset", "manifest", "source_csv"):
                configured = (
                    protocol.config_path.parent / str(warm_c4[key])
                ).resolve()
                try:
                    paths += (configured.relative_to(project_root).as_posix(),)
                except ValueError as error:
                    raise ValueError(
                        "C4 behavior evidence must stay inside the project"
                    ) from error
            if warm_c4.get("dagger") is True:
                learner = (
                    protocol.config_path.parent
                    / str(warm_c4["learner_checkpoint"])
                ).resolve()
                try:
                    paths += (learner.relative_to(project_root).as_posix(),)
                except ValueError as error:
                    raise ValueError(
                        "C4 DAgger learner checkpoint must stay inside the project"
                    ) from error
                paths += (f"{prefix}/c4/dagger_distillation_gate.json",)
            if (
                warm_c4.get("precomputed_conditional_distillation") is True
                or warm_c4.get("precomputed_existence_gated_distillation") is True
            ):
                for key in (
                    "precomputed_checkpoint",
                    "precomputed_manifest",
                    "conditional_dataset",
                    "conditional_dataset_manifest",
                ):
                    configured = (
                        protocol.config_path.parent / str(warm_c4[key])
                    ).resolve()
                    try:
                        paths += (configured.relative_to(project_root).as_posix(),)
                    except ValueError as error:
                        raise ValueError(
                            "conditional distillation evidence must stay inside the project"
                        ) from error
        return paths
    paths = (
        f"{prefix}/c0/reward_audit_memory_final.json",
        f"{prefix}/c0/smoke_memory_final/training_run_manifest.json",
        f"{prefix}/pretraining_active_tests_memory_final.log",
        f"{prefix}/pretraining_active_tests_memory_final.xml",
        f"{prefix}/wandb_preflight.json",
        f"{prefix}/environment_profile.json",
        f"{prefix}/agent_follows_doc_memory_pass.json",
    )
    representation = str(
        protocol.raw["observation"].get("representation", "legacy_state_v1")
    )
    if representation == "visual_lane_belief_v2":
        lane_path = protocol.lane_belief_config_path
        if lane_path is not None and lane_path.name == "lane_belief_v3_codex.toml":
            lane_evidence = (
                f"{prefix}/lane_calibration/lane_measurement_calibration_metrics.json",
                f"{prefix}/lane_belief_gate/final_metrics.json",
            )
        elif (
            lane_path is not None
            and lane_path.name == "lane_belief_v4_transfer.toml"
        ):
            lane_evidence = (
                f"{prefix}/lane_calibration/lane_transfer_calibration_metrics.json",
                f"{prefix}/lane_belief_gate/final_metrics.json",
            )
        elif (
            lane_path is not None
            and lane_path.name == "lane_belief_v8_competence_rgb.toml"
        ):
            lane_evidence = (
                f"{prefix}/lane_rgb_model/model_manifest.json",
                f"{prefix}/lane_rgb_model/best.pt",
                "datasets/lane_rgb_competence_v9/manifest.json",
                "datasets/lane_rgb_combined_v9/manifest.json",
                f"{prefix}/lane_rgb_final/final_metrics.json",
                f"{prefix}/lane_closed_loop_gate/final_metrics.json",
            )
        else:
            lane_evidence = (
                "artifacts/visual_lane/lane_belief_final_validation_metrics.json",
            )
        return paths + lane_evidence + (f"{prefix}/c0/reset_memory_audit.json",)
    return paths


def load_ppo_curriculum_protocol(
    path: str | Path,
    *,
    require_frozen: bool = True,
) -> PPOCurriculumProtocol:
    config_path = Path(path).resolve()
    data = _load_protocol_data(config_path)
    if int(data.get("schema_version", 0)) != 1:
        raise ValueError("unsupported F10-PPO schema version")

    def relative(value: str) -> Path:
        return (config_path.parent / value).resolve()

    provenance = data["provenance"]
    observation = data["observation"]
    representation = str(observation.get("representation", "legacy_state_v1"))
    ordering = tuple(str(value) for value in observation["ordering"])
    scales_table = observation["scales"]
    if len(ordering) != len(set(ordering)):
        raise ValueError("PPO observation ordering contains duplicates")
    if set(ordering) != set(scales_table):
        raise ValueError("PPO observation ordering/scales mismatch")
    scales = tuple(float(scales_table[name]) for name in ordering)
    if any(value <= 0.0 for value in scales):
        raise ValueError("PPO observation scales must be positive")

    action = data["action"]
    action_bounds = (
        float(action["minimum_linear_velocity_mps"]),
        float(action["maximum_linear_velocity_mps"]),
        float(action["minimum_angular_velocity_rad_s"]),
        float(action["maximum_angular_velocity_rad_s"]),
    )
    if action_bounds != (0.0, 0.4, -4.0, 4.0):
        raise ValueError("PPO must reuse the F2 action envelope")

    ppo_raw = data["ppo"]
    ppo = PPOSettings(
        hidden_sizes=tuple(int(value) for value in ppo_raw["hidden_sizes"]),
        learning_rate=float(ppo_raw["learning_rate"]),
        n_steps=int(ppo_raw["n_steps"]),
        batch_size=int(ppo_raw["batch_size"]),
        n_epochs=int(ppo_raw["n_epochs"]),
        gamma=float(ppo_raw["gamma"]),
        gae_lambda=float(ppo_raw["gae_lambda"]),
        clip_range=float(ppo_raw["clip_range"]),
        entropy_coefficient=float(ppo_raw["entropy_coefficient"]),
        value_function_coefficient=float(ppo_raw["value_function_coefficient"]),
        max_gradient_norm=float(ppo_raw["max_gradient_norm"]),
        initial_log_std=float(ppo_raw["initial_log_std"]),
        training_seed=int(ppo_raw["training_seed"]),
        device=str(ppo_raw["device"]),
    )
    if ppo.n_steps <= 0 or ppo.batch_size <= 0 or ppo.n_steps % ppo.batch_size:
        raise ValueError("PPO n_steps must be positive and divisible by batch_size")

    stages: dict[str, CurriculumStage] = {}
    for key in STAGE_NAMES:
        raw_stage = data["curriculum"][key]
        stage = CurriculumStage(
            key=key,
            name=str(raw_stage["name"]),
            map_name=str(raw_stage["map"]),
            scenario_config_path=(
                relative(str(raw_stage["scenario_config"]))
                if "scenario_config" in raw_stage
                else None
            ),
            pedestrian_active=bool(raw_stage["pedestrian_active"]),
            stop_active=bool(raw_stage["stop_active"]),
            domain_randomization=bool(raw_stage["domain_randomization"]),
            pedestrian_modes=tuple(str(value) for value in raw_stage.get("pedestrian_modes", ())),
            episode_horizon_steps=int(raw_stage["episode_horizon_steps"]),
            training_steps=int(raw_stage["training_steps"]),
            checkpoint_interval_steps=int(raw_stage["checkpoint_interval_steps"]),
            training_seeds=tuple(int(value) for value in raw_stage["training_seeds"]),
            development_seeds=tuple(int(value) for value in raw_stage["development_seeds"]),
            stage_final_seeds=tuple(int(value) for value in raw_stage["stage_final_seeds"]),
        )
        if stage.training_steps % stage.checkpoint_interval_steps:
            raise ValueError(f"{key} checkpoint interval must divide training steps")
        if stage.training_steps % ppo.n_steps or stage.checkpoint_interval_steps % ppo.n_steps:
            raise ValueError(f"{key} training/checkpoint steps must align to PPO rollouts")
        if (
            stage.map_name == "pomdp_v1" or stage.scenario_config_path is not None
        ) and not stage.pedestrian_modes:
            raise ValueError(f"{key} needs at least one pedestrian mode")
        if (
            stage.scenario_config_path is not None
            and not stage.scenario_config_path.is_file()
        ):
            raise FileNotFoundError(stage.scenario_config_path)
        stages[key] = stage

    global_final = {
        str(name): tuple(int(value) for value in values)
        for name, values in data["global_final"].items()
    }
    explicit_exclusions = tuple(
        int(value)
        for value in data.get("seed_protocol", {}).get(
            "excluded_historical_seeds", ()
        )
    )
    range_exclusions: list[int] = []
    for bounds in data.get("seed_protocol", {}).get(
        "excluded_historical_seed_ranges", ()
    ):
        if len(bounds) != 2:
            raise ValueError("excluded historical seed ranges must be [start, end]")
        start, end = (int(value) for value in bounds)
        if start > end:
            raise ValueError("excluded historical seed range start exceeds end")
        range_exclusions.extend(range(start, end + 1))
    historical = tuple(
        sorted(
            set(_historical_seed_inventory(config_path.parent))
            | set(explicit_exclusions)
            | set(range_exclusions)
        )
    )
    _validate_seed_isolation(stages, global_final, historical)

    protocol = PPOCurriculumProtocol(
        config_path=config_path,
        scenario_path=relative(str(provenance["scenario"])),
        pomdp_map_path=relative(str(provenance["pomdp_map"])),
        action_config_path=relative(str(provenance["action_config"])),
        action_config_sha256=str(provenance["action_config_sha256"]),
        belief_config_path=relative(str(provenance["frozen_belief_config"])),
        belief_config_sha256=str(provenance["frozen_belief_config_sha256"]),
        lane_belief_config_path=(
            relative(str(provenance["lane_belief_config"]))
            if "lane_belief_config" in provenance
            else None
        ),
        lane_belief_config_sha256=(
            str(provenance["lane_belief_config_sha256"])
            if "lane_belief_config_sha256" in provenance
            else None
        ),
        detector_checkpoint_path=relative(str(provenance["detector_checkpoint"])),
        detector_checkpoint_sha256=str(provenance["detector_checkpoint_sha256"]),
        observation_order=ordering,
        observation_scales=scales,
        observation_clip=float(observation["clip_normalized"]),
        action_bounds=action_bounds,
        ppo=ppo,
        stages=stages,
        global_final=global_final,
        historical_seeds=historical,
        raw=data,
    )
    if representation == "visual_lane_belief_v2" and protocol.lane_belief_config_path is None:
        raise ValueError("visual-lane observation requires frozen lane_belief_config")
    _validate_provenance(protocol, provenance, require_frozen=require_frozen)
    return protocol


def _load_protocol_data(
    config_path: Path,
    *,
    ancestry: tuple[Path, ...] = (),
) -> dict[str, Any]:
    """Load an optional hash-pinned protocol extension.

    Small remediation protocols may override only the fields that changed
    while retaining a source-bound parent. This avoids copying hundreds of
    unchanged TOML lines and preserves an auditable parent hash.
    """

    resolved = config_path.resolve()
    if resolved in ancestry:
        raise ValueError("cyclic PPO protocol extension")
    with resolved.open("rb") as stream:
        child: dict[str, Any] = tomllib.load(stream)
    parent_name = child.get("extends")
    if parent_name is None:
        return child
    parent_path = (resolved.parent / str(parent_name)).resolve()
    if not parent_path.is_file():
        raise FileNotFoundError(parent_path)
    expected = str(child.get("extends_sha256", ""))
    if not expected or file_sha256(parent_path) != expected:
        raise RuntimeError("PPO protocol parent hash mismatch")
    parent = _load_protocol_data(parent_path, ancestry=ancestry + (resolved,))
    return _deep_merge(parent, child)


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged: dict[str, Any] = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(dict(merged[key]), value)
        else:
            merged[key] = value
    return merged


def require_pretraining_gate(
    protocol: PPOCurriculumProtocol,
    gate_path: str | Path,
) -> dict[str, Any]:
    """Verify that substantive PPO training is bound to audited evidence."""

    path = Path(gate_path).resolve()
    if not path.is_file():
        raise RuntimeError(f"full PPO training requires pretraining gate: {path}")
    gate = json.loads(path.read_text(encoding="utf-8"))
    ready = gate.get("ready_for_training") is True or gate.get(
        "ready_for_c0_training"
    ) is True
    if gate.get("schema_version") != 1 or not ready:
        raise RuntimeError("F10-PPO pretraining gate is not ready")
    if gate.get("config_sha256") != file_sha256(protocol.config_path):
        raise RuntimeError("F10-PPO pretraining gate/config hash mismatch")
    project_root = protocol.config_path.parent.parent
    frozen_sources = gate.get("frozen_sources")
    evidence = gate.get("evidence")
    source_paths = pretraining_source_paths(protocol)
    evidence_paths = pretraining_evidence_paths(protocol)
    if set(frozen_sources or ()) != set(source_paths):
        raise RuntimeError("F10-PPO pretraining gate source inventory mismatch")
    if set(evidence or ()) != set(evidence_paths):
        raise RuntimeError("F10-PPO pretraining gate evidence inventory mismatch")
    for section in ("frozen_sources", "evidence"):
        entries = gate.get(section)
        if not isinstance(entries, dict) or not entries:
            raise RuntimeError(f"F10-PPO pretraining gate has no {section}")
        for recorded_path, expected_sha in entries.items():
            candidate = Path(str(recorded_path))
            candidate = candidate if candidate.is_absolute() else project_root / candidate
            candidate = candidate.resolve()
            if not candidate.is_file() or file_sha256(candidate) != str(expected_sha):
                raise RuntimeError(f"F10-PPO pretraining gate invalidated by {recorded_path}")
    _validate_pretraining_evidence(project_root, protocol)
    return gate


def classify_curriculum_stage(
    *,
    safety_pass: bool,
    skill_pass: bool,
    retention_pass: bool,
) -> tuple[str, bool]:
    """Return the frozen classification and whether progression is permitted."""

    if safety_pass and skill_pass and retention_pass:
        return "PASS", True
    if safety_pass:
        return "LIMITED", False
    return "FAILED", False


def evaluate_retention_change(
    protocol: PPOCurriculumProtocol,
    stage_key: str,
    baseline_summary: dict[str, Any],
    current_summary: dict[str, Any],
) -> tuple[dict[str, Any], bool]:
    """Evaluate a pre-registered C1-C3 catastrophic-forgetting limit."""

    definitions = {
        "c1": ("completion_rate", "maximum_small_loop_completion_drop", "drop"),
        "c2": ("completion_rate", "maximum_c1_completion_drop", "drop"),
        "c3": ("collision_rate", "maximum_c2_collision_rate_increase", "increase"),
    }
    try:
        metric, threshold_name, direction = definitions[stage_key]
    except KeyError as error:
        raise ValueError("retention change is defined only for C1-C3") from error
    baseline = float(baseline_summary[metric])
    current = float(current_summary[metric])
    change = baseline - current if direction == "drop" else current - baseline
    threshold = float(protocol.raw["acceptance"][stage_key][threshold_name])
    passed = change <= threshold
    return {
        "metric": metric,
        "baseline": baseline,
        "current": current,
        direction: change,
        "maximum_allowed": threshold,
        "passed": passed,
    }, passed


def load_retention_reference(
    protocol: PPOCurriculumProtocol,
    target_stage: str,
    artifact_root: str | Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Load a hash-bound retention baseline, including imported predecessors."""

    target = target_stage.lower()
    if target not in STAGE_NAMES:
        raise ValueError(f"unknown retention target: {target_stage}")
    imported = protocol.raw.get("curriculum_import", {}).get(target)
    if imported is None:
        reference_path = Path(artifact_root).resolve() / target / "retention_metrics.json"
        expected_config_sha = file_sha256(protocol.config_path)
        source_protocol_sha = expected_config_sha
        imported_reference = False
    else:
        imported = dict(imported)
        source_protocol_path = (
            protocol.config_path.parent / str(imported["source_protocol"])
        ).resolve()
        source_protocol_sha = str(imported["source_protocol_sha256"])
        if (
            not source_protocol_path.is_file()
            or file_sha256(source_protocol_path) != source_protocol_sha
        ):
            raise RuntimeError("imported retention source protocol hash mismatch")
        source_root = (
            protocol.config_path.parent / str(imported["artifact_root"])
        ).resolve()
        reference_path = source_root / target / "retention_metrics.json"
        expected_config_sha = source_protocol_sha
        imported_reference = True

    reference = _read_json(reference_path)
    if (
        reference.get("config_sha256") != expected_config_sha
        or reference.get("stage") != target
        or reference.get("retention_pass") is not True
        or target not in reference.get("summaries", {})
    ):
        raise RuntimeError("retention reference provenance or status mismatch")
    return reference["summaries"][target], {
        "reference_checkpoint_stage": target,
        "reference_imported": imported_reference,
        "reference_protocol_sha256": source_protocol_sha,
        "reference_metrics_sha256": file_sha256(reference_path),
    }


def require_curriculum_transition(
    protocol: PPOCurriculumProtocol,
    stage_key: str,
    source_checkpoint: str | Path,
    artifact_root: str | Path,
) -> dict[str, Any]:
    """Permit C1-C4 only from the selected passing previous-stage policy."""

    key = stage_key.lower()
    require_stage_in_protocol_scope(protocol, key)
    index = STAGE_NAMES.index(key)
    if index == 0:
        raise ValueError("C0 has no curriculum predecessor")
    previous = STAGE_NAMES[index - 1]
    imported = protocol.raw.get("curriculum_import", {}).get(previous)
    if imported is not None:
        return _require_imported_curriculum_transition(
            protocol,
            previous,
            source_checkpoint,
            dict(imported),
        )
    previous_dir = Path(artifact_root).resolve() / previous
    checkpoint_manifest = _read_json(previous_dir / "checkpoint_manifest.json")
    final_metrics = _read_json(previous_dir / "stage_final_metrics.json")
    retention = _read_json(previous_dir / "retention_metrics.json")
    config_sha = file_sha256(protocol.config_path)
    if any(value.get("config_sha256") != config_sha for value in (checkpoint_manifest, final_metrics, retention)):
        raise RuntimeError("previous-stage/config provenance mismatch")
    if checkpoint_manifest.get("stage") != previous or final_metrics.get("stage") != previous:
        raise RuntimeError("previous-stage identity mismatch")
    if checkpoint_manifest.get("selected_is_gate_eligible") is not True:
        raise RuntimeError("previous-stage selected checkpoint was not development-gate eligible")
    if final_metrics.get("classification") != "PASS" or final_metrics.get("progression_permitted") is not True:
        raise RuntimeError(f"curriculum STOP: {previous} did not PASS")
    if retention.get("retention_pass") is not True:
        raise RuntimeError(f"curriculum STOP: {previous} retention failed")
    source = Path(source_checkpoint).resolve()
    expected = checkpoint_manifest["artifacts"]["selected"]["sha256"]
    if not source.is_file() or file_sha256(source) != expected:
        raise RuntimeError("source checkpoint is not the selected previous-stage checkpoint")
    _, payload = _load_checkpoint_metadata(source)
    if payload.get("stage") != previous:
        raise RuntimeError("source checkpoint payload stage mismatch")
    if payload.get("metadata", {}).get("config_sha256") != config_sha:
        raise RuntimeError("source checkpoint payload config mismatch")
    if tuple(payload.get("metadata", {}).get("observation_order", ())) != protocol.observation_order:
        raise RuntimeError("source checkpoint observation contract mismatch")
    checkpoint_config = payload.get("config", {})
    expected_contract = {
        "observation_dimension": len(protocol.observation_order),
        "action_dimension": 2,
        "hidden_sizes": protocol.ppo.hidden_sizes,
        "learning_rate": protocol.ppo.learning_rate,
        "n_steps": protocol.ppo.n_steps,
        "batch_size": protocol.ppo.batch_size,
        "n_epochs": protocol.ppo.n_epochs,
        "gamma": protocol.ppo.gamma,
        "gae_lambda": protocol.ppo.gae_lambda,
        "clip_range": protocol.ppo.clip_range,
        "entropy_coefficient": protocol.ppo.entropy_coefficient,
        "value_function_coefficient": protocol.ppo.value_function_coefficient,
        "max_gradient_norm": protocol.ppo.max_gradient_norm,
        "initial_log_std": protocol.ppo.initial_log_std,
        "seed": protocol.ppo.training_seed,
    }
    for name, expected_value in expected_contract.items():
        actual_value = checkpoint_config.get(name)
        if name == "hidden_sizes":
            actual_value = tuple(actual_value or ())
        if actual_value != expected_value:
            raise RuntimeError(f"source checkpoint PPO contract mismatch: {name}")
    return {
        "previous_stage": previous,
        "source_checkpoint_sha256": expected,
        "stage_final_metrics_sha256": file_sha256(previous_dir / "stage_final_metrics.json"),
        "retention_metrics_sha256": file_sha256(previous_dir / "retention_metrics.json"),
    }


def _require_imported_curriculum_transition(
    protocol: PPOCurriculumProtocol,
    previous: str,
    source_checkpoint: str | Path,
    imported: dict[str, Any],
) -> dict[str, Any]:
    """Verify a passed predecessor imported by a hash-pinned remediation."""

    source_protocol_path = (
        protocol.config_path.parent / str(imported["source_protocol"])
    ).resolve()
    source_protocol_sha = str(imported["source_protocol_sha256"])
    if (
        not source_protocol_path.is_file()
        or file_sha256(source_protocol_path) != source_protocol_sha
    ):
        raise RuntimeError("imported predecessor protocol hash mismatch")
    source_protocol = load_ppo_curriculum_protocol(source_protocol_path)
    if (
        source_protocol.observation_order != protocol.observation_order
        or source_protocol.action_bounds != protocol.action_bounds
        or source_protocol.ppo != protocol.ppo
    ):
        raise RuntimeError("imported predecessor policy contract mismatch")

    source_root = (
        protocol.config_path.parent / str(imported["artifact_root"])
    ).resolve()
    previous_dir = source_root / previous
    checkpoint_manifest = _read_json(previous_dir / "checkpoint_manifest.json")
    final_metrics = _read_json(previous_dir / "stage_final_metrics.json")
    retention = _read_json(previous_dir / "retention_metrics.json")
    if any(
        value.get("config_sha256") != source_protocol_sha
        for value in (checkpoint_manifest, final_metrics, retention)
    ):
        raise RuntimeError("imported predecessor evidence provenance mismatch")
    if (
        checkpoint_manifest.get("stage") != previous
        or final_metrics.get("stage") != previous
        or checkpoint_manifest.get("selected_is_gate_eligible") is not True
        or final_metrics.get("classification") != "PASS"
        or final_metrics.get("progression_permitted") is not True
        or retention.get("retention_pass") is not True
    ):
        raise RuntimeError("imported predecessor did not pass its frozen gates")

    configured_checkpoint = (
        protocol.config_path.parent / str(imported["selected_checkpoint"])
    ).resolve()
    expected = str(imported["selected_checkpoint_sha256"])
    source = Path(source_checkpoint).resolve()
    if source != configured_checkpoint:
        raise RuntimeError("source is not the configured imported checkpoint")
    if not source.is_file() or file_sha256(source) != expected:
        raise RuntimeError("imported predecessor checkpoint hash mismatch")
    if checkpoint_manifest["artifacts"]["selected"]["sha256"] != expected:
        raise RuntimeError("imported predecessor selection hash mismatch")

    _, payload = _load_checkpoint_metadata(source)
    if (
        payload.get("stage") != previous
        or payload.get("metadata", {}).get("config_sha256") != source_protocol_sha
        or tuple(payload.get("metadata", {}).get("observation_order", ()))
        != protocol.observation_order
    ):
        raise RuntimeError("imported predecessor checkpoint metadata mismatch")
    checkpoint_config = dict(payload.get("config", {}))
    if (
        int(checkpoint_config.get("observation_dimension", -1))
        != len(protocol.observation_order)
        or int(checkpoint_config.get("action_dimension", -1)) != 2
        or tuple(checkpoint_config.get("hidden_sizes", ()))
        != protocol.ppo.hidden_sizes
    ):
        raise RuntimeError("imported predecessor checkpoint architecture mismatch")
    return {
        "previous_stage": previous,
        "imported": True,
        "source_protocol_sha256": source_protocol_sha,
        "source_checkpoint_sha256": expected,
        "stage_final_metrics_sha256": file_sha256(
            previous_dir / "stage_final_metrics.json"
        ),
        "retention_metrics_sha256": file_sha256(
            previous_dir / "retention_metrics.json"
        ),
    }


def _validate_pretraining_evidence(project_root: Path, protocol: PPOCurriculumProtocol) -> None:
    evidence_paths = pretraining_evidence_paths(protocol)
    config_sha = file_sha256(protocol.config_path)
    if _uses_separated_object_curriculum(protocol):
        _validate_object_pretraining_evidence(
            project_root, protocol, evidence_paths, config_sha
        )
        return
    reward = _read_json(project_root / evidence_paths[0])
    if reward.get("passed") is not True or reward.get("config_sha256") != config_sha:
        raise RuntimeError("reward audit evidence is not a passing current-config run")
    smoke = _read_json(project_root / evidence_paths[1])
    if not (
        smoke.get("smoke") is True
        and smoke.get("config_sha256") == config_sha
        and int(smoke.get("environment_steps", 0)) == 128
        and int(smoke.get("ppo_updates_total", 0)) >= 2
        and smoke.get("checkpoint_reload_verified") is True
    ):
        raise RuntimeError("PPO smoke evidence is invalid")
    counts = junit_counts(project_root / evidence_paths[3])
    if not (
        counts["tests"] >= 436
        and counts["failures"] == 0
        and counts["errors"] == 0
        and counts["skipped"] == 0
    ):
        raise RuntimeError(f"full-suite evidence mismatch: {counts}")
    wandb = _read_json(project_root / evidence_paths[4])
    if not (
        wandb.get("entity") == "vnv"
        and wandb.get("project") == "DuckiePOMDP"
        and wandb.get("state") == "finished"
        and wandb.get("config_sha256") == config_sha
    ):
        raise RuntimeError("W&B online destination evidence is invalid")
    environment = _read_json(project_root / evidence_paths[5])
    if environment.get("environment_id") != "e11c4b9d" or not environment.get("cuda_witness", {}).get("gpu"):
        raise RuntimeError("environment/CUDA evidence is invalid")
    audit = _read_json(project_root / evidence_paths[6])
    if audit.get("verdict") != "PASS" or audit.get("config_sha256") != config_sha:
        raise RuntimeError("fresh agent-follows-doc evidence is not PASS")
    representation = str(
        protocol.raw["observation"].get("representation", "legacy_state_v1")
    )
    if representation == "visual_lane_belief_v2":
        _validate_visual_pretraining_evidence(project_root, protocol, evidence_paths)


def _validate_object_pretraining_evidence(
    project_root: Path,
    protocol: PPOCurriculumProtocol,
    evidence_paths: tuple[str, ...],
    config_sha: str,
) -> None:
    """Validate the C2/C3 experiment-loop gate without replaying frozen C0/C1."""

    def artifact(suffix: str) -> dict[str, Any]:
        return _read_json(
            project_root / _unique_evidence_path(evidence_paths, suffix=suffix)
        )

    maximum_stage = str(
        protocol.raw.get("protocol_scope", {}).get(
            "maximum_stage", VISUAL_MAXIMUM_STAGE
        )
    ).lower()
    includes_c4 = STAGE_NAMES.index(maximum_stage) >= STAGE_NAMES.index("c4")
    reward_stages = ("c2", "c3") + (("c4",) if includes_c4 else ())
    for stage_key in reward_stages:
        reward = artifact(f"/{stage_key}/reward_audit.json")
        stage = protocol.stage(stage_key)
        scenario_provenance = reward.get("scenario_provenance") or {}
        scenario = (
            None
            if stage.scenario_config_path is None
            else load_scenario(stage.scenario_config_path)
        )
        if not (
            reward.get("passed") is True
            and reward.get("config_sha256") == config_sha
            and reward.get("source_sha256")
            == file_sha256(project_root / "experiments" / "audit_f10_ppo_reward.py")
            and reward.get("reference_policy_sha256")
            == file_sha256(
                project_root
                / "src"
                / "duckie_pomdp"
                / "evaluation"
                / "f10_ppo_policy.py"
            )
            and tuple(int(value) for value in reward.get("seeds", ()))
            == stage.training_seeds[:2]
            and scenario is not None
            and scenario_provenance.get("config_sha256")
            == file_sha256(stage.scenario_config_path)
            and scenario_provenance.get("map_sha256")
            == file_sha256(scenario.map_path)
        ):
            raise RuntimeError(f"{stage_key.upper()} reward audit is not PASS")
        if stage.stop_active:
            checks = reward.get("checks", {})
            if not (
                checks.get("simple_stops") is True
                and checks.get("simple_does_not_violate_stop") is True
                and checks.get("simple_restarts_after_stop") is True
                and checks.get("simple_no_collision") is True
                and checks.get("simple_no_invalid_pose") is True
            ):
                raise RuntimeError("C3 reference controller geometry is unsafe")

    warm_c3 = protocol.raw.get("behavior_warm_start", {}).get("c3", {})
    smoke_stage = (
        "c4" if includes_c4 else ("c3" if warm_c3.get("enabled") else "c2")
    )
    smoke = artifact(f"/{smoke_stage}/smoke/training_run_manifest.json")
    imported_stage = {"c2": "c1", "c3": "c2", "c4": "c3"}[smoke_stage]
    imported = dict(protocol.raw["curriculum_import"][imported_stage])
    if not (
        smoke.get("stage") == smoke_stage
        and smoke.get("smoke") is True
        and smoke.get("config_sha256") == config_sha
        and int(smoke.get("environment_steps", 0)) == 128
        and int(smoke.get("ppo_updates_total", 0)) >= 2
        and smoke.get("checkpoint_reload_verified") is True
        and smoke.get("source_checkpoint", {}).get("sha256")
        == imported["selected_checkpoint_sha256"]
    ):
        raise RuntimeError(
            f"{smoke_stage.upper()} imported-checkpoint PPO smoke evidence is invalid"
        )

    counts = junit_counts(
        project_root / _unique_evidence_path(evidence_paths, suffix="/pretraining_tests.xml")
    )
    if not (
        counts["tests"] >= 436
        and counts["failures"] == 0
        and counts["errors"] == 0
        and counts["skipped"] == 0
    ):
        raise RuntimeError(f"object-curriculum full-suite mismatch: {counts}")

    wandb = artifact("/wandb_preflight.json")
    if not (
        wandb.get("entity") == "vnv"
        and wandb.get("project") == "DuckiePOMDP"
        and wandb.get("group") == protocol.raw["wandb"]["group"]
        and wandb.get("state") == "finished"
        and wandb.get("config_sha256") == config_sha
    ):
        raise RuntimeError("object-curriculum W&B destination evidence is invalid")

    environment = artifact("/environment_profile.json")
    if (
        environment.get("environment_id") != "e11c4b9d"
        or not environment.get("cuda_witness", {}).get("gpu")
    ):
        raise RuntimeError("object-curriculum environment/CUDA evidence is invalid")

    scenario = artifact("/object_scenario_gate.json")
    geometry = scenario.get("geometry", {})
    isolation = scenario.get("stage_isolation", {})
    conflicts = tuple(scenario.get("temporal_conflict", ()))
    if not (
        scenario.get("passed") is True
        and scenario.get("config_sha256") == config_sha
        and geometry.get("pedestrian_crosses_route") is True
        and float(geometry.get("forward_route_separation_m", 0.0)) > 0.75
        and isolation.get("c2", {}).get("stop_sign_physically_absent") is True
        and isolation.get("c3", {}).get("pedestrian_physically_absent") is True
        and len(conflicts) == 2
        and all(int(item.get("duckie_detection_frames", 0)) > 0 for item in conflicts)
        and all(float(item.get("minimum_clearance_m", 99.0)) <= 0.25 for item in conflicts)
    ):
        raise RuntimeError("real experiment-loop object scenario gate is invalid")

    if includes_c4:
        combined_isolation = isolation.get("c4", {})
        combined = tuple(scenario.get("combined_temporal") or ())
        if not (
            combined_isolation.get("pedestrian_physically_absent") is False
            and combined_isolation.get("stop_sign_physically_absent") is False
            and len(combined) == 2
            and all(int(item.get("duckie_detection_frames", 0)) > 0 for item in combined)
            and all(int(item.get("stop_sign_detection_frames", 0)) > 0 for item in combined)
            and all(float(item.get("minimum_clearance_m", 99.0)) <= 0.65 for item in combined)
            and all(
                float(item.get("pedestrian_speed_at_minimum_clearance_mps", 0.0))
                >= 0.15
                for item in combined
            )
            and all(
                item.get("pedestrian_exited_after_crossing") is True
                for item in combined
            )
            and all(item.get("stop_completed") is True for item in combined)
            and all(item.get("restarted") is True for item in combined)
            and all(item.get("collision") is False for item in combined)
            and all(item.get("stop_violation") is False for item in combined)
            and all(item.get("completed") is True for item in combined)
        ):
            raise RuntimeError("real combined C4 scenario gate is invalid")

    runtime_detection = protocol.raw.get("runtime_detection")
    if runtime_detection:
        confidence = artifact("/duckie_confidence_gate.json")
        threshold = float(runtime_detection["duckie_minimum_confidence"])
        calibration_path = project_root / "artifacts" / "f9_yolo_measurement_calibration.csv"
        if not (
            confidence.get("passed") is True
            and confidence.get("config_sha256") == config_sha
            and float(confidence.get("threshold", -1.0)) == threshold
            and confidence.get("calibration_csv_sha256")
            == file_sha256(calibration_path)
            and int(confidence.get("correct_detection_count", 0)) > 0
            and int(confidence.get("incorrect_detection_count", 0)) > 0
            and float(confidence.get("minimum_correct_confidence", 0.0))
            >= threshold
            and float(confidence.get("maximum_incorrect_confidence", 1.0))
            < threshold
        ):
            raise RuntimeError("Duckie confidence-filter evidence is invalid")
        maximum_bottom = runtime_detection.get("duckie_maximum_bottom_y_px")
        if maximum_bottom is not None:
            domain = artifact("/duckie_image_domain_gate.json")
            audit_path = project_root / _unique_evidence_path(
                evidence_paths, suffix="/duckie_bbox_audit.csv"
            )
            if not (
                domain.get("passed") is True
                and domain.get("config_sha256") == config_sha
                and float(domain.get("maximum_bottom_y_px", -1.0))
                == float(maximum_bottom)
                and domain.get("uses_privileged_truth_for_runtime_filter") is False
                and domain.get("c4_audit_csv_sha256") == file_sha256(audit_path)
                and int(domain.get("f9_correct_rows_retained", 0)) >= 1000
                and int(domain.get("c4_visible_rows_retained", 0)) > 0
                and int(domain.get("c4_absent_false_rows_rejected", 0)) > 0
                and int(domain.get("c4_absent_false_rows_accepted", -1)) <= 1
            ):
                raise RuntimeError("Duckie image-domain plausibility evidence is invalid")

    memory = artifact("/object_reset_memory_audit.json")
    if not (
        memory.get("passed") is True
        and memory.get("config_sha256") == config_sha
        and tuple(memory.get("stages", ()))
        == (("c2", "c3", "c4") if includes_c4 else ("c2", "c3"))
        and int(memory.get("resets_per_stage", 0)) >= 12
        and memory.get("integration_reused_within_stage") is True
        and memory.get("stage_isolation_preserved") is True
    ):
        raise RuntimeError("object scenario reset/memory audit is invalid")

    warm = protocol.raw.get("behavior_warm_start", {}).get("c2", {})
    if warm.get("enabled"):
        dataset_path = (
            protocol.config_path.parent / str(warm["dataset"])
        ).resolve()
        manifest = artifact("/c2/behavior_warm_start_manifest.json")
        snapshot_path = project_root / _unique_evidence_path(
            evidence_paths, suffix="/c2/behavior_warm_start_source.csv"
        )
        if not (
            manifest.get("uses_evaluation_gt") is False
            and int(manifest.get("rows", 0)) > 0
            and int(manifest.get("observation_dimension", 0))
            == len(protocol.observation_order)
            and manifest.get("dataset_sha256") == str(warm["dataset_sha256"])
            and dataset_path.is_file()
            and file_sha256(dataset_path) == str(warm["dataset_sha256"])
            and snapshot_path.is_file()
            and Path(str(manifest.get("source_csv", ""))).resolve()
            == snapshot_path.resolve()
            and manifest.get("source_csv_sha256") == file_sha256(snapshot_path)
            and int(manifest.get("source_snapshot_rows", 0))
            == int(manifest.get("rows", -1))
        ):
            raise RuntimeError("behavior warm-start evidence is invalid")
        if warm.get("multitask_distillation") is True:
            imported = dict(protocol.raw["curriculum_import"]["c1"])
            if not (
                manifest.get("source_role")
                == "balanced_c1_anchor_and_c2_hazard_policy_observations"
                and int(manifest.get("anchor_rows", 0)) > 0
                and int(manifest.get("anchor_rows", 0))
                == int(manifest.get("hazard_rows", -1))
                and abs(
                    float(manifest.get("anchor_weight_mass", 0.0))
                    - float(manifest.get("hazard_weight_mass", -1.0))
                ) <= 0.1
                and manifest.get("source_checkpoint_sha256")
                == imported["selected_checkpoint_sha256"]
                and manifest.get("source_checkpoint_stage") == "c1"
            ):
                raise RuntimeError("multitask behavior distillation evidence is invalid")
            distillation = artifact("/c2/multitask_distillation_gate.json")
            if not (
                distillation.get("passed") is True
                and distillation.get("config_sha256") == config_sha
                and int(distillation.get("checkpoint_step", -1)) == 0
                and distillation.get("c1_retention_pass") is True
                and all(distillation.get("c2_checks", {}).values())
            ):
                raise RuntimeError("multitask distillation real-runtime gate is invalid")

    warm_c3 = protocol.raw.get("behavior_warm_start", {}).get("c3", {})
    if warm_c3.get("enabled"):
        configured_paths = {
            key: (protocol.config_path.parent / str(warm_c3[key])).resolve()
            for key in ("dataset", "manifest", "source_csv")
        }
        evidence_resolved = {
            (project_root / path).resolve() for path in evidence_paths
        }
        if not set(configured_paths.values()).issubset(evidence_resolved):
            raise RuntimeError("C3 behavior artifacts are absent from frozen evidence")
        manifest = _read_json(configured_paths["manifest"])
        imported = dict(protocol.raw["curriculum_import"]["c2"])
        source_protocol = (
            protocol.config_path.parent / str(warm_c3["source_protocol"])
        ).resolve()
        episodes = tuple(manifest.get("source_episodes", ()))
        common_valid = (
            configured_paths["dataset"].is_file()
            and configured_paths["source_csv"].is_file()
            and file_sha256(configured_paths["dataset"])
            == str(warm_c3["dataset_sha256"])
            and file_sha256(configured_paths["manifest"])
            == str(warm_c3["manifest_sha256"])
            and file_sha256(configured_paths["source_csv"])
            == str(warm_c3["source_csv_sha256"])
            and source_protocol.is_file()
            and file_sha256(source_protocol)
            == str(warm_c3["source_protocol_sha256"])
            and manifest.get("uses_evaluation_gt") is False
            and manifest.get("source_config_sha256")
            == str(warm_c3["source_protocol_sha256"])
            and manifest.get("dataset_sha256")
            == str(warm_c3["dataset_sha256"])
            and manifest.get("source_csv_sha256")
            == str(warm_c3["source_csv_sha256"])
            and manifest.get("anchor_dataset_sha256")
            == str(warm_c3["anchor_dataset_sha256"])
            and int(manifest.get("observation_dimension", 0))
            == len(protocol.observation_order)
            and int(manifest.get("anchor_rows", 0)) > 0
            and int(manifest.get("stop_action_rows", 0)) > 0
            and int(manifest.get("satisfied_observation_rows", 0)) > 0
            and int(manifest.get("detected_steps", 0)) > 0
            and episodes
            and all(
                episode.get("completed") is True
                and episode.get("stop_completed") is True
                and episode.get("stop_violation") is False
                and episode.get("restarted") is True
                and episode.get("collision") is False
                and episode.get("invalid_pose") is False
                for episode in episodes
            )
            and imported["selected_checkpoint_sha256"]
            == str(warm_c3["source_checkpoint_sha256"])
        )
        if warm_c3.get("dagger") is True:
            learner_checkpoint = (
                protocol.config_path.parent
                / str(warm_c3["learner_checkpoint"])
            ).resolve()
            evidence_resolved = {
                (project_root / path).resolve() for path in evidence_paths
            }
            distillation = artifact("/c3/dagger_distillation_gate.json")
            role_valid = (
                manifest.get("source_role")
                == "balanced_c2_retention_c3_teacher_and_dagger_v1"
                and manifest.get("builder_sha256")
                == file_sha256(
                    project_root / "experiments/build_ppo_c3_dagger_warm_start.py"
                )
                and int(manifest.get("teacher_rows", 0)) > 0
                and int(manifest.get("dagger_rows", 0)) > 0
                and int(manifest.get("rows", 0))
                == int(manifest.get("anchor_rows", -1))
                + int(manifest.get("teacher_rows", -1))
                + int(manifest.get("dagger_rows", -1))
                and int(manifest.get("dagger_stalled_observation_rows", 0)) > 0
                and int(manifest.get("dagger_teacher_drive_rows", 0)) > 0
                and set(manifest.get("training_seeds_only", ())).issubset(
                    set(protocol.historical_seeds)
                )
                and abs(
                    float(manifest.get("anchor_weight_mass", 0.0))
                    - float(manifest.get("teacher_weight_mass", -1.0))
                ) <= 0.1
                and abs(
                    float(manifest.get("anchor_weight_mass", 0.0))
                    - float(manifest.get("dagger_weight_mass", -1.0))
                ) <= 0.1
                and learner_checkpoint in evidence_resolved
                and learner_checkpoint.is_file()
                and file_sha256(learner_checkpoint)
                == str(warm_c3["learner_checkpoint_sha256"])
                and manifest.get("learner_checkpoint_sha256")
                == str(warm_c3["learner_checkpoint_sha256"])
                and manifest.get("learner_checkpoint_stage") == "c3"
                and int(manifest.get("learner_checkpoint_step", -1)) == 0
                and distillation.get("passed") is True
                and distillation.get("config_sha256") == config_sha
                and distillation.get("checkpoint_step") == 0
                and distillation.get("c2_retention_pass") is True
                and all(distillation.get("c3_checks", {}).values())
            )
        else:
            role_valid = (
                manifest.get("source_role")
                == "balanced_c2_retention_and_c3_public_belief_teacher"
                and int(manifest.get("anchor_rows", 0))
                == int(manifest.get("stop_rows", -1))
                and abs(
                    float(manifest.get("anchor_weight_mass", 0.0))
                    - float(manifest.get("stop_weight_mass", -1.0))
                ) <= 0.1
            )
        if not (common_valid and role_valid):
            raise RuntimeError("C3 public-belief behavior warm-start evidence is invalid")

    warm_c4 = protocol.raw.get("behavior_warm_start", {}).get("c4", {})
    if warm_c4.get("enabled"):
        configured_paths = {
            key: (protocol.config_path.parent / str(warm_c4[key])).resolve()
            for key in ("dataset", "manifest", "source_csv")
        }
        evidence_resolved = {
            (project_root / path).resolve() for path in evidence_paths
        }
        learner_checkpoint = (
            protocol.config_path.parent / str(warm_c4["learner_checkpoint"])
        ).resolve()
        source_protocol = (
            protocol.config_path.parent / str(warm_c4["source_protocol"])
        ).resolve()
        manifest = _read_json(configured_paths["manifest"])
        distillation = artifact("/c4/dagger_distillation_gate.json")
        imported = dict(protocol.raw["curriculum_import"]["c3"])
        cumulative = warm_c4.get("cumulative_policy_rehearsal") is True
        privileged_guidance = warm_c4.get("privileged_teacher_guidance") is True
        conditional_precomputed = (
            warm_c4.get("precomputed_conditional_distillation") is True
        )
        existence_gated_precomputed = (
            warm_c4.get("precomputed_existence_gated_distillation") is True
        )
        if conditional_precomputed and existence_gated_precomputed:
            raise RuntimeError("C4 precomputed distillation mode must be unique")
        precomputed = conditional_precomputed or existence_gated_precomputed
        precomputed_paths: dict[str, Path] = {}
        precomputed_manifest: dict[str, Any] = {}
        conditional_manifest: dict[str, Any] = {}
        if precomputed:
            precomputed_paths = {
                key: (protocol.config_path.parent / str(warm_c4[key])).resolve()
                for key in (
                    "precomputed_checkpoint",
                    "precomputed_manifest",
                    "conditional_dataset",
                    "conditional_dataset_manifest",
                )
            }
            precomputed_manifest = _read_json(
                precomputed_paths["precomputed_manifest"]
            )
            conditional_manifest = _read_json(
                precomputed_paths["conditional_dataset_manifest"]
            )
        builder = project_root / (
            "experiments/build_ppo_c4_privileged_guidance_v24.py"
            if privileged_guidance
            else "experiments/build_ppo_c4_cumulative_dagger_v23.py"
            if cumulative
            else "experiments/build_ppo_c4_dagger_warm_start.py"
        )
        truth_boundary_valid = (
            manifest.get("teacher_uses_evaluation_gt") is True
            and manifest.get("student_observations_use_evaluation_gt") is False
            and manifest.get("privileged_truth_stored_in_npz") is False
            if privileged_guidance
            else manifest.get("uses_evaluation_gt") is False
        )
        common_valid = (
            warm_c4.get("dagger") is True
            and set(configured_paths.values()).issubset(evidence_resolved)
            and learner_checkpoint in evidence_resolved
            and all(path.is_file() for path in configured_paths.values())
            and learner_checkpoint.is_file()
            and source_protocol.is_file()
            and file_sha256(configured_paths["dataset"])
            == str(warm_c4["dataset_sha256"])
            and file_sha256(configured_paths["manifest"])
            == str(warm_c4["manifest_sha256"])
            and file_sha256(configured_paths["source_csv"])
            == str(warm_c4["source_csv_sha256"])
            and file_sha256(learner_checkpoint)
            == str(warm_c4["learner_checkpoint_sha256"])
            and file_sha256(source_protocol)
            == str(warm_c4["source_protocol_sha256"])
            and truth_boundary_valid
            and manifest.get("source_config_sha256")
            == str(warm_c4["source_protocol_sha256"])
            and manifest.get("builder_sha256") == file_sha256(builder)
            and manifest.get("dataset_sha256") == str(warm_c4["dataset_sha256"])
            and manifest.get("source_csv_sha256")
            == str(warm_c4["source_csv_sha256"])
            and manifest.get("learner_checkpoint_sha256")
            == str(warm_c4["learner_checkpoint_sha256"])
            and manifest.get("learner_checkpoint_stage") == "c4"
            and int(manifest.get("learner_checkpoint_step", -1)) >= 1024
            and int(manifest.get("observation_dimension", 0))
            == len(protocol.observation_order)
            and (
                (
                    int(manifest.get("duckie_detected_steps", 0)) > 0
                    and int(manifest.get("stop_detected_steps", 0)) > 0
                )
                or (
                    privileged_guidance
                    and all(
                        int(item.get("duckie_detection_frames", 0)) > 0
                        and int(item.get("stop_sign_detection_frames", 0)) > 0
                        for item in scenario.get("combined_temporal", ())
                    )
                )
            )
            and imported["selected_checkpoint_sha256"]
            == str(warm_c4["source_checkpoint_sha256"])
            and distillation.get("passed") is True
            and distillation.get("config_sha256") == config_sha
            and int(distillation.get("checkpoint_step", -1)) == 0
            and (
                not precomputed
                or (
                    set(precomputed_paths.values()).issubset(evidence_resolved)
                    and all(path.is_file() for path in precomputed_paths.values())
                    and file_sha256(precomputed_paths["precomputed_checkpoint"])
                    == str(warm_c4["precomputed_checkpoint_sha256"])
                    and file_sha256(precomputed_paths["precomputed_manifest"])
                    == str(warm_c4["precomputed_manifest_sha256"])
                    and file_sha256(precomputed_paths["conditional_dataset"])
                    == str(warm_c4["conditional_dataset_sha256"])
                    and file_sha256(
                        precomputed_paths["conditional_dataset_manifest"]
                    ) == str(warm_c4["conditional_dataset_manifest_sha256"])
                    and precomputed_manifest.get("output_sha256")
                    == str(warm_c4["precomputed_checkpoint_sha256"])
                    and precomputed_manifest.get("dataset_sha256")
                    == str(warm_c4["conditional_dataset_sha256"])
                    and precomputed_manifest.get(
                        "student_observation_uses_privileged_truth"
                    ) is False
                    and conditional_manifest.get("output_sha256")
                    == str(warm_c4["conditional_dataset_sha256"])
                    and conditional_manifest.get(
                        "student_observation_uses_privileged_truth"
                    ) is False
                    and (
                        (
                            conditional_precomputed
                            and set(conditional_manifest.get("task_counts", {}))
                            == {"c2_correction", "c3_retention", "c4_retention"}
                            and len(
                                conditional_manifest.get("task_weight_mass", {})
                            ) == 3
                            and min(
                                float(value)
                                for value in conditional_manifest.get(
                                    "task_weight_mass", {}
                                ).values()
                            ) > 0.0
                        )
                        or (
                            existence_gated_precomputed
                            and set(conditional_manifest.get("group_counts", {}))
                            == {
                                "c2_hazard_correction",
                                "c2_neutral_counterfactual",
                                "c2_nonhazard_retention",
                                "c3_retention",
                                "c4_retention",
                            }
                            and len(
                                conditional_manifest.get("group_weight_mass", {})
                            ) == 5
                            and min(
                                float(value)
                                for value in conditional_manifest.get(
                                    "group_weight_mass", {}
                                ).values()
                            ) > 0.0
                            and float(
                                conditional_manifest.get(
                                    "minimum_existence_probability", -1.0
                                )
                            )
                            == float(
                                protocol.raw["observation"][
                                    "pedestrian_kinematics_min_existence_probability"
                                ]
                            )
                            and float(
                                precomputed_manifest.get("runtime_semantics", {}).get(
                                    "pedestrian_kinematics_min_existence_probability",
                                    -1.0,
                                )
                            )
                            == float(
                                protocol.raw["observation"][
                                    "pedestrian_kinematics_min_existence_probability"
                                ]
                            )
                        )
                    )
                    and distillation.get("checkpoint_sha256")
                    == str(warm_c4["precomputed_checkpoint_sha256"])
                )
            )
        )
        if cumulative:
            role_rows = dict(manifest.get("role_rows", {}))
            role_masses = dict(manifest.get("role_weight_masses", {}))
            required_roles = {
                "c2_source_policy_rehearsal",
                "c3_source_policy_rehearsal",
                "c4_teacher_trajectory",
                "c4_dagger_learner_state",
            }
            if privileged_guidance:
                required_roles.add("c4_privileged_guided_episode")
            base = float(role_masses.get("c4_teacher_trajectory", -1.0))
            role_valid = (
                manifest.get("source_role")
                == (
                    "v23_cumulative_plus_one_privileged_c4_guided_episode_v1"
                    if privileged_guidance
                    else "c2_c3_policy_rehearsal_c4_teacher_dagger_v2"
                )
                and set(role_rows) == required_roles
                and set(role_masses) == required_roles
                and int(manifest.get("rows", 0)) == sum(int(value) for value in role_rows.values())
                and base > 0.0
                and abs(float(role_masses["c4_dagger_learner_state"]) - base) <= 0.2
                and abs(
                    float(role_masses["c2_source_policy_rehearsal"])
                    - (4.0 if privileged_guidance else 2.0) * base
                ) <= 0.2
                and abs(
                    float(role_masses["c3_source_policy_rehearsal"])
                    - (4.0 if privileged_guidance else 2.0) * base
                ) <= 0.2
                and (
                    not privileged_guidance
                    or (
                        abs(
                            float(role_masses["c4_privileged_guided_episode"])
                            - base
                        ) <= 0.2
                        and int(manifest.get("guided_episode_count", 0)) == 1
                        and int(manifest.get("critic_supervised_rows", 0)) > 0
                        and manifest.get("guided_episode", {}).get("completed") is True
                        and manifest.get("guided_episode", {}).get("stop_completed") is True
                        and manifest.get("guided_episode", {}).get("stop_violation") is False
                        and manifest.get("guided_episode", {}).get("collision") is False
                        and manifest.get("guided_episode", {}).get("unsafe") is False
                    )
                )
                and all(
                    episode.get("completed") is True
                    and episode.get("collision") is False
                    and episode.get("unsafe") is False
                    and episode.get("invalid_pose") is False
                    for episode in manifest.get("c2_source_episodes", ())
                )
                and all(
                    episode.get("completed") is True
                    and episode.get("stop_completed") is True
                    and episode.get("restarted") is True
                    and episode.get("stop_violation") is False
                    and episode.get("collision") is False
                    and episode.get("invalid_pose") is False
                    for key in ("c3_source_episodes", "c4_teacher_episodes")
                    for episode in manifest.get(key, ())
                )
                and all(distillation.get("c2_checks", {}).values())
                and all(distillation.get("c3_checks", {}).values())
                and all(distillation.get("c4_checks", {}).values())
            )
        else:
            episodes = tuple(manifest.get("source_episodes", ()))
            anchor_mass = float(manifest.get("anchor_weight_mass", -1.0))
            teacher_mass = float(manifest.get("teacher_weight_mass", -1.0))
            dagger_mass = float(manifest.get("dagger_weight_mass", -1.0))
            anchor_multiplier = float(warm_c4["anchor_weight_multiplier"])
            role_valid = (
                manifest.get("source_role")
                == "balanced_cumulative_c3_c4_teacher_and_dagger_v1"
                and manifest.get("anchor_dataset_sha256")
                == str(warm_c4["anchor_dataset_sha256"])
                and int(manifest.get("rows", 0))
                == int(manifest.get("anchor_rows", -1))
                + int(manifest.get("teacher_rows", -1))
                + int(manifest.get("dagger_rows", -1))
                and min(anchor_mass, teacher_mass, dagger_mass) > 0.0
                and abs(teacher_mass - dagger_mass) <= 0.1
                and abs(anchor_mass - anchor_multiplier * teacher_mass) <= 0.2
                and float(manifest.get("anchor_weight_multiplier", -1.0)) == anchor_multiplier
                and int(manifest.get("stop_action_rows", 0)) > 0
                and int(manifest.get("dagger_teacher_drive_rows", 0)) > 0
                and episodes
                and all(
                    episode.get("completed") is True
                    and episode.get("stop_completed") is True
                    and episode.get("restarted") is True
                    and episode.get("stop_violation") is False
                    and episode.get("collision") is False
                    and episode.get("invalid_pose") is False
                    for episode in episodes
                )
                and distillation.get("c3_retention_pass") is True
                and all(distillation.get("c4_checks", {}).values())
            )
        valid = common_valid and role_valid
        if not valid:
            raise RuntimeError("C4 public-belief DAgger warm-start evidence is invalid")


def _unique_evidence_path(
    evidence_paths: tuple[str, ...],
    *,
    suffix: str,
) -> str:
    """Resolve one evidence artifact by meaning instead of tuple position."""

    matches = tuple(path for path in evidence_paths if path.endswith(suffix))
    if len(matches) != 1:
        raise RuntimeError(
            f"expected exactly one pretraining evidence path ending {suffix!r}; "
            f"found {matches}"
        )
    return matches[0]


def _validate_visual_pretraining_evidence(
    project_root: Path,
    protocol: PPOCurriculumProtocol,
    evidence_paths: tuple[str, ...],
) -> None:
    """Validate visual-lane artifacts without relying on inventory ordering."""

    rgb_model_evidence = tuple(
        path
        for path in evidence_paths
        if path.endswith("/lane_rgb_model/model_manifest.json")
    )
    codex_calibration = tuple(
        path
        for path in evidence_paths
        if path.endswith("/lane_calibration/lane_measurement_calibration_metrics.json")
        or path.endswith("/lane_calibration/lane_transfer_calibration_metrics.json")
    )
    if rgb_model_evidence:
        if len(rgb_model_evidence) != 1:
            raise RuntimeError("ambiguous lane RGB model evidence")
        _validate_lane_rgb_competence_evidence(
            project_root, protocol, evidence_paths, rgb_model_evidence[0]
        )
        lane_path = None
        expected_seed_role = None
    elif codex_calibration:
        if len(codex_calibration) != 1:
            raise RuntimeError("ambiguous visual-lane calibration evidence")
        calibration = _read_json(project_root / codex_calibration[0])
        if "splits" in calibration:
            calibration_seeds = {
                int(value)
                for values in calibration["splits"]["calibration"].values()
                for value in values
            }
            development_seeds = {
                int(value)
                for values in calibration["splits"]["development"].values()
                for value in values
            }
            calibration_valid = (
                calibration.get("gate_pass") is True
                and calibration.get("split_unit") == "seed/trajectory"
                and calibration.get("seed_overlap") == []
                and calibration.get("runtime_inputs")
                == ["front_rgb", "fixed_camera_calibration"]
            )
        else:
            calibration_seeds = set(
                int(value) for value in calibration.get("calibration_seeds", ())
            )
            development_seeds = set(
                int(value) for value in calibration.get("development_seeds", ())
            )
            calibration_valid = (
                calibration.get("gate_pass") is True
                and calibration.get("direction") == "counter-clockwise"
                and calibration.get("split_unit") == "seed/trajectory"
                and calibration.get("seed_overlap") == []
                and calibration.get("runtime_inputs")
                == ["front_rgb", "fixed_camera_calibration"]
            )
        if not (
            calibration_valid
            and calibration_seeds
            and development_seeds
            and not calibration_seeds.intersection(development_seeds)
        ):
            raise RuntimeError("visual-lane calibration evidence is not PASS")

        lane_path = _unique_evidence_path(
            evidence_paths,
            suffix="/lane_belief_gate/final_metrics.json",
        )
        expected_seed_role = "once-only held-out gate"
    else:
        lane_path = _unique_evidence_path(
            evidence_paths,
            suffix="/lane_belief_final_validation_metrics.json",
        )
        expected_seed_role = "once-only final visual-lane gate"

    if lane_path is not None:
        lane = _read_json(project_root / lane_path)
        if not (
            lane.get("gate_pass") is True
            and lane.get("seed_role") == expected_seed_role
            and lane.get("direction") == "counter-clockwise"
        ):
            raise RuntimeError("visual-lane final validation evidence is not PASS")
        if codex_calibration and lane.get("config_sha256") != protocol.lane_belief_config_sha256:
            raise RuntimeError("visual-lane final validation/config provenance mismatch")

    memory_path = _unique_evidence_path(
        evidence_paths,
        suffix="/c0/reset_memory_audit.json",
    )
    memory = _read_json(project_root / memory_path)
    config_sha = file_sha256(protocol.config_path)
    if not (
        memory.get("passed") is True
        and memory.get("config_sha256") == config_sha
        and int(memory.get("unique_integration_count", 0)) == 1
        and int(memory.get("unique_simulator_count", 0)) == 1
        and int(memory.get("resets", 0)) >= 36
    ):
        raise RuntimeError("native reset memory audit evidence is not PASS")


def _validate_lane_rgb_competence_evidence(
    project_root: Path,
    protocol: PPOCurriculumProtocol,
    evidence_paths: tuple[str, ...],
    model_manifest_path: str,
) -> None:
    """Validate the camera-only competence model and both held-out lane gates."""

    model = _read_json(project_root / model_manifest_path)
    checkpoint_path = _unique_evidence_path(
        evidence_paths, suffix="/lane_rgb_model/best.pt"
    )
    checkpoint_sha = file_sha256(project_root / checkpoint_path)
    if not (
        model.get("gate_pass") is True
        and model.get("runtime_input") == "front_rgb_only"
        and model.get("best_checkpoint_sha256") == checkpoint_sha
        and model.get("final_split_consumed") is False
        and float(model.get("preprocessing", {}).get("crop_top_fraction", -1.0))
        == 0.25
        and float(
            model.get("preprocessing", {}).get(
                "horizontal_flip_probability", -1.0
            )
        )
        == 0.0
    ):
        raise RuntimeError("lane RGB model evidence is invalid")
    if protocol.lane_belief_config_path is None:
        raise RuntimeError("lane RGB evidence requires a lane belief config")
    with protocol.lane_belief_config_path.open("rb") as handle:
        lane_config = tomllib.load(handle)
    learned = lane_config.get("lane_rgb_model", {})
    if (
        learned.get("enabled") is not True
        or str(learned.get("checkpoint_sha256")) != checkpoint_sha
        or float(learned.get("crop_top_fraction", -1.0)) != 0.25
    ):
        raise RuntimeError("lane RGB runtime/config provenance mismatch")

    competence_path = _unique_evidence_path(
        evidence_paths, suffix="/lane_rgb_competence_v9/manifest.json"
    )
    competence = _read_json(project_root / competence_path)
    if not (
        competence.get("runtime_input") == "front_rgb_only"
        and competence.get("direction") == "counter-clockwise"
        and all(not values for values in competence.get("seed_overlaps", {}).values())
    ):
        raise RuntimeError("lane RGB competence dataset evidence is invalid")
    combined_path = _unique_evidence_path(
        evidence_paths, suffix="/lane_rgb_combined_v9/manifest.json"
    )
    combined = _read_json(project_root / combined_path)
    logical_counts = tuple(
        int(value)
        for value in combined.get("logical_training_counts_by_turn", {}).values()
    )
    if not (
        combined.get("runtime_input") == "front_rgb_only"
        and combined.get("horizontal_flip_forbidden") is True
        and combined.get("dynamic_final_excluded") is True
        and logical_counts
        and len(set(logical_counts)) == 1
        and all(not values for values in combined.get("split_overlaps", {}).values())
    ):
        raise RuntimeError("combined lane RGB dataset evidence is invalid")

    frame_path = _unique_evidence_path(
        evidence_paths, suffix="/lane_rgb_final/final_metrics.json"
    )
    frame = _read_json(project_root / frame_path)
    if not (
        frame.get("gate_pass") is True
        and frame.get("seed_role") == "once-only held-out lane RGB final"
        and frame.get("runtime_input") == "front_rgb_only"
        and frame.get("checkpoint_sha256") == checkpoint_sha
    ):
        raise RuntimeError("lane RGB once-only frame gate is invalid")
    closed_loop_path = _unique_evidence_path(
        evidence_paths, suffix="/lane_closed_loop_gate/final_metrics.json"
    )
    closed_loop = _read_json(project_root / closed_loop_path)
    if not (
        closed_loop.get("gate_pass") is True
        and closed_loop.get("seed_role") == "once-only held-out final"
        and closed_loop.get("direction") == "counter-clockwise"
        and closed_loop.get("config_sha256") == file_sha256(protocol.config_path)
        and str(closed_loop.get("runtime_chain", "")).startswith("front_rgb")
    ):
        raise RuntimeError("lane RGB once-only closed-loop gate is invalid")


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise RuntimeError(f"required evidence is missing: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def junit_counts(path: str | Path) -> dict[str, int]:
    """Read pytest JUnit counts from either testsuite or testsuites roots."""

    root = ElementTree.parse(Path(path)).getroot()
    suites = [root] if root.tag.endswith("testsuite") else list(root.findall("testsuite"))
    if not suites:
        raise RuntimeError("JUnit evidence contains no testsuite")
    return {
        name: sum(int(suite.attrib.get(name, 0)) for suite in suites)
        for name in ("tests", "failures", "errors", "skipped")
    }


def _load_checkpoint_metadata(path: Path) -> tuple[object, dict[str, Any]]:
    # Local import keeps protocol parsing independent of Torch until a stage
    # transition is actually authorized.
    from .ppo import PPOAgent

    return PPOAgent.load(path, device="cpu")


def _validate_seed_isolation(
    stages: dict[str, CurriculumStage],
    global_final: dict[str, tuple[int, ...]],
    historical: tuple[int, ...],
) -> None:
    groups: dict[str, set[int]] = {}
    for key, stage in stages.items():
        for split in SPLIT_NAMES:
            values = stage.seeds_for(split)
            if not values or len(values) != len(set(values)):
                raise ValueError(f"{key}/{split} seeds are empty or duplicated")
            groups[f"{key}/{split}"] = set(values)
    for name, values in global_final.items():
        if not values or len(values) != len(set(values)):
            raise ValueError(f"global final {name} seeds are empty or duplicated")
        groups[f"global/{name}"] = set(values)
    names = tuple(groups)
    for index, left in enumerate(names):
        for right in names[index + 1 :]:
            overlap = groups[left] & groups[right]
            if overlap:
                raise ValueError(f"seed leakage {left} vs {right}: {sorted(overlap)}")
    historical_set = set(historical)
    for name, values in groups.items():
        overlap = values & historical_set
        if overlap:
            raise ValueError(f"{name} reuses historical seeds: {sorted(overlap)}")


def _historical_seed_inventory(config_dir: Path) -> tuple[int, ...]:
    # Existing experiments are all below 20,000. This explicit closed interval
    # is intentionally conservative and prevents accidental reuse even when an
    # old config represented seeds as ranges rather than arrays.
    return tuple(range(0, 20_000))


def _validate_provenance(
    protocol: PPOCurriculumProtocol,
    raw: dict[str, Any],
    *,
    require_frozen: bool,
) -> None:
    paths = (
        protocol.scenario_path,
        protocol.pomdp_map_path,
        protocol.action_config_path,
        protocol.belief_config_path,
        protocol.detector_checkpoint_path,
    ) + (() if protocol.lane_belief_config_path is None else (protocol.lane_belief_config_path,))
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(path)
    if not require_frozen:
        return
    checks = (
        (protocol.pomdp_map_path, str(raw["pomdp_map_sha256"]), "POMDP map"),
        (protocol.action_config_path, protocol.action_config_sha256, "action config"),
        (protocol.belief_config_path, protocol.belief_config_sha256, "belief config"),
        (protocol.detector_checkpoint_path, protocol.detector_checkpoint_sha256, "YOLO checkpoint"),
        (_duckietown_map_path("small_loop"), str(raw["small_loop_sha256"]), "small_loop map"),
        (_duckietown_map_path("experiment_loop"), str(raw["experiment_loop_sha256"]), "experiment_loop map"),
    ) + (
        ()
        if protocol.lane_belief_config_path is None
        else (
            (
                protocol.lane_belief_config_path,
                str(protocol.lane_belief_config_sha256),
                "visual lane belief config",
            ),
        )
    )
    for path, expected, label in checks:
        actual = file_sha256(path)
        if actual != expected:
            raise RuntimeError(f"frozen {label} hash mismatch: {actual} != {expected}")


def _duckietown_map_path(name: str) -> Path:
    spec = find_spec("duckietown_world")
    if spec is None or spec.origin is None:
        raise FileNotFoundError("duckietown_world package")
    path = Path(spec.origin).resolve().parent / "data" / "gd1" / "maps" / f"{name}.yaml"
    if not path.is_file():
        raise FileNotFoundError(path)
    return path
