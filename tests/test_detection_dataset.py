from __future__ import annotations

from inspect import signature
from pathlib import Path

import numpy as np
import pytest

from duckie_pomdp.adapters.gym_duckietown import (
    GymDuckietownConfig,
    create_gym_duckietown,
)
from duckie_pomdp.dataset.annotations import (
    SilhouetteRules,
    YoloBox,
    assess_silhouette,
)
from duckie_pomdp.dataset.config import load_dataset_config
from duckie_pomdp.dataset.split import assert_no_split_leakage
from duckie_pomdp.domain.detection import (
    BoundingBox,
    ObjectClass,
    YOLO_V1_CLASS_NAMES,
    object_class_from_yolo_id,
    yolo_class_id,
)
from duckie_pomdp.ports.detector import ObjectDetector
from duckie_pomdp.scenario import PedestrianMode, load_scenario


def test_canonical_yolo_class_mapping_is_exactly_two_classes() -> None:
    assert YOLO_V1_CLASS_NAMES == ("stop_sign", "duckie")
    assert yolo_class_id(ObjectClass.STOP_SIGN) == 0
    assert yolo_class_id(ObjectClass.DUCKIE) == 1
    assert object_class_from_yolo_id(0) is ObjectClass.STOP_SIGN
    assert object_class_from_yolo_id(1) is ObjectClass.DUCKIE
    with pytest.raises(ValueError):
        object_class_from_yolo_id(-1)
    with pytest.raises(ValueError):
        object_class_from_yolo_id(2)


def test_silhouette_bbox_normalizes_to_standard_yolo_label() -> None:
    rules = SilhouetteRules(20, 5.0, 5.0, 1, 30.0)
    decision = assess_silhouette(
        class_id=0,
        box=BoundingBox(64.0, 48.0, 192.0, 144.0),
        visible_pixel_count=500,
        image_width_px=640,
        image_height_px=480,
        rules=rules,
    )
    assert decision.accepted
    assert decision.yolo_box == YoloBox(0, 0.2, 0.2, 0.2, 0.2)
    assert YoloBox.parse(decision.yolo_box.to_line()) == decision.yolo_box


def test_tiny_or_heavily_truncated_silhouettes_are_rejected() -> None:
    rules = SilhouetteRules(20, 5.0, 5.0, 1, 30.0)
    tiny = assess_silhouette(
        class_id=1,
        box=BoundingBox(10.0, 10.0, 12.0, 12.0),
        visible_pixel_count=4,
        image_width_px=640,
        image_height_px=480,
        rules=rules,
    )
    corner = assess_silhouette(
        class_id=1,
        box=BoundingBox(0.0, 0.0, 30.0, 30.0),
        visible_pixel_count=300,
        image_width_px=640,
        image_height_px=480,
        rules=rules,
    )
    assert tiny.reason == "too_few_visible_pixels"
    assert corner.reason == "too_heavily_truncated"


def test_dataset_config_has_disjoint_episode_seed_splits() -> None:
    config = load_dataset_config(Path("configs/detection_dataset_v1.toml"))
    rows = [
        {"split": split, "seed": seed, "episode_id": f"{split}_{seed}_{mode.value}"}
        for split, seeds in config.split_seeds.items()
        for seed in seeds
        for mode in config.pedestrian_modes
    ]
    assert_no_split_leakage(rows)


def test_split_leakage_is_rejected() -> None:
    with pytest.raises(ValueError, match="split leakage"):
        assert_no_split_leakage(
            (
                {"split": "train", "seed": 5, "episode_id": "a"},
                {"split": "val", "seed": 5, "episode_id": "b"},
            )
        )


def test_detector_runtime_port_accepts_only_rgb() -> None:
    parameters = list(signature(ObjectDetector.detect).parameters)
    assert parameters == ["self", "rgb"]


def test_real_segmentation_silhouettes_produce_two_visible_boxes() -> None:
    scenario = load_scenario(
        Path("configs/scenario_pomdp_v1.toml")
    ).with_pedestrian_mode(PedestrianMode.STATIONARY)
    integration = create_gym_duckietown(
        GymDuckietownConfig(
            scenario=scenario,
            camera_width=320,
            camera_height=240,
        )
    )
    try:
        observation = integration.agent.reset(seed=scenario.seed)
        samples = integration.projection_validation.sample_object_silhouettes(
            ("sign_stop", "duckie")
        )
        assert observation.front_rgb.shape == (240, 320, 3)
        assert {sample.object_kind for sample in samples} == {"sign_stop", "duckie"}
        for sample in samples:
            box = sample.bounding_box
            assert sample.visible_pixel_count > 20
            assert 0.0 <= box.x_min_px < box.x_max_px <= 320.0
            assert 0.0 <= box.y_min_px < box.y_max_px <= 240.0
        sign_box = next(
            sample.bounding_box
            for sample in samples
            if sample.object_kind == "sign_stop"
        )
        sign_crop = observation.front_rgb[
            int(sign_box.y_min_px) : int(sign_box.y_max_px),
            int(sign_box.x_min_px) : int(sign_box.x_max_px),
        ]
        red_pixels = np.count_nonzero(
            (sign_crop[..., 0] > 80)
            & (sign_crop[..., 0] > 1.25 * sign_crop[..., 1])
            & (sign_crop[..., 0] > 1.25 * sign_crop[..., 2])
        )
        assert red_pixels > 30, "stop sign must expose its red face to the ego camera"
    finally:
        integration.close()
