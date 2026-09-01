from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from duckie_pomdp.control.ppo import PPOActorCritic, PPOConfig
from duckie_pomdp.control.ppo_environment import PPOCurriculumEnvironment
from duckie_pomdp.control.ppo_protocol import load_ppo_curriculum_protocol
from duckie_pomdp.explain.observation_contract import (
    assert_policy_vector_precedes_privileged_read,
    deterministic_actor_statistics,
    reconstruct_normalized_observation,
    validate_feature_group_partition,
    validate_public_policy_mapping,
)

from experiments.verify_f11_r001_contract import EXPECTED_TRACE_KEYS


ROOT = Path(__file__).resolve().parents[1]


def test_r001_config_defines_exact_primary_group_partition() -> None:
    try:
        import tomllib
    except ModuleNotFoundError:  # pragma: no cover - Python 3.10 project runtime
        import tomli as tomllib

    config_path = ROOT / "configs" / "f11_ppo_explanation_v2.toml"
    with config_path.open("rb") as stream:
        raw = tomllib.load(stream)
    protocol = load_ppo_curriculum_protocol(
        ROOT / "configs" / raw["frozen_policy"]["config"]
    )
    groups = raw["feature_groups"]
    assert tuple(groups) == (
        "Lane",
        "Ego",
        "StopLine",
        "Pedestrian",
        "Stop",
        "PreviousAction",
    )
    validate_feature_group_partition(protocol.observation_order, groups)
    assert len(protocol.observation_order) == 29


def test_reconstruction_matches_fixed_physical_normalization() -> None:
    order = ("a", "b", "c")
    mapping = {"a": 1.0, "b": -4.0, "c": 100.0}
    reconstructed = reconstruct_normalized_observation(
        mapping, order, (2.0, 2.0, 10.0), 3.0
    )
    np.testing.assert_array_equal(
        reconstructed, np.asarray([0.5, -2.0, 3.0], dtype=np.float32)
    )


def test_public_mapping_rejects_privileged_fields() -> None:
    with pytest.raises(ValueError, match="privileged/evaluation"):
        validate_public_policy_mapping(
            {"lane": 0.0, "evaluation_gt_range": 1.0},
            ("lane", "evaluation_gt_range"),
        )


def test_public_mapping_allows_previous_action_fields() -> None:
    order = (
        "previous_linear_velocity_cmd_mps",
        "previous_angular_velocity_cmd_rad_s",
    )
    validate_public_policy_mapping(
        {
            "previous_linear_velocity_cmd_mps": 0.2,
            "previous_angular_velocity_cmd_rad_s": -0.1,
        },
        order,
    )


def test_actor_explanation_target_is_distribution_mean_not_sample() -> None:
    model = PPOActorCritic(
        PPOConfig(
            observation_dimension=3,
            action_dimension=2,
            hidden_sizes=(4,),
            learning_rate=3.0e-4,
            n_steps=8,
            batch_size=4,
            n_epochs=1,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            entropy_coefficient=0.0,
            value_function_coefficient=0.5,
            max_gradient_norm=0.5,
            initial_log_std=-0.5,
            seed=7,
            device="cpu",
        )
    )
    observations = torch.zeros((5, 3), dtype=torch.float32)
    actor_mean, value = deterministic_actor_statistics(model, observations)
    expected_mean = model.distribution(observations).mean
    torch.testing.assert_close(actor_mean, expected_mean)
    assert actor_mean.shape == (5, 2)
    assert value.shape == (5,)


def test_policy_vector_is_built_before_privileged_read() -> None:
    assert_policy_vector_precedes_privileged_read(PPOCurriculumEnvironment.reset)
    assert_policy_vector_precedes_privileged_read(PPOCurriculumEnvironment.step)


def test_r001_public_trace_schema_contains_no_privileged_fields() -> None:
    assert len(EXPECTED_TRACE_KEYS) == len(set(EXPECTED_TRACE_KEYS))
    assert not any(
        token in field.lower()
        for field in EXPECTED_TRACE_KEYS
        for token in ("privileged", "evaluation_gt", "ground_truth", "bbox", "iou")
    )
