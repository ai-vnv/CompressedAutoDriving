from __future__ import annotations

from inspect import signature
from pathlib import Path

import pytest
import torch

from duckie_pomdp.control.lane_belief_runtime import VisualLaneBeliefRuntime
from duckie_pomdp.control.ppo_protocol import load_ppo_curriculum_protocol
from duckie_pomdp.perception.lane_rgb_model import (
    LanePoseMobileNet,
    LaneRGBMeasurementEstimator,
    LaneRGBModelConfig,
)


ROOT = Path(__file__).resolve().parents[1]


def test_lane_rgb_network_has_fixed_three_value_contract() -> None:
    model = LanePoseMobileNet().eval()
    with torch.inference_mode():
        output = model(torch.zeros((1, 3, 64, 64), dtype=torch.float32))
    assert output.shape == (1, 3)


def test_lane_rgb_runtime_api_accepts_image_only() -> None:
    parameters = tuple(signature(LaneRGBMeasurementEstimator.estimate).parameters)
    assert parameters == ("self", "front_rgb")
    runtime_parameters = tuple(signature(VisualLaneBeliefRuntime.update).parameters)
    assert "privileged_state" not in runtime_parameters
    assert "true_lane_pose" not in runtime_parameters


def test_lane_rgb_config_rejects_invalid_uncertainty() -> None:
    with pytest.raises(ValueError):
        LaneRGBModelConfig(
            checkpoint=Path("missing.pt"),
            checkpoint_sha256="0" * 64,
            device="cpu",
            image_size_px=224,
            target_scales=(0.12, 0.70, 15.0),
            residual_sigmas=(0.01, 0.0, 1.0),
        )


def test_lane_rgb_config_rejects_invalid_top_crop() -> None:
    with pytest.raises(ValueError):
        LaneRGBModelConfig(
            checkpoint=Path("missing.pt"),
            checkpoint_sha256="0" * 64,
            device="cpu",
            image_size_px=224,
            target_scales=(0.12, 0.70, 15.0),
            residual_sigmas=(0.01, 0.05, 1.0),
            crop_top_fraction=0.8,
        )


def test_v7_dataset_seed_ranges_are_excluded_from_curriculum() -> None:
    protocol = load_ppo_curriculum_protocol(ROOT / "configs" / "f10_ppo_visual_v7.toml")
    curriculum = {
        seed
        for stage in protocol.stages.values()
        for split in (stage.training_seeds, stage.development_seeds, stage.stage_final_seeds)
        for seed in split
    }
    assert not curriculum & set(range(80_000, 84_000))
    assert not curriculum & set(range(85_000, 86_000))
