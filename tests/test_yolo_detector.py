from __future__ import annotations

import inspect
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from duckie_pomdp.domain.detection import ObjectClass
from duckie_pomdp.perception.yolo_detector import (
    YoloObjectDetector,
    rgb_to_pil,
    validate_v1_class_mapping,
)


class FakeModel:
    names = {0: "stop_sign", 1: "duckie"}

    def __init__(self) -> None:
        self.received = None

    def predict(self, *, source, **kwargs):
        self.received = np.asarray(source)
        boxes = SimpleNamespace(
            xyxy=np.asarray([[1.0, 2.0, 11.0, 22.0]]),
            conf=np.asarray([0.75]),
            cls=np.asarray([1.0]),
        )
        return [SimpleNamespace(boxes=boxes)]


def test_rgb_channel_order_is_preserved() -> None:
    rgb = np.zeros((2, 3, 3), dtype=np.uint8)
    rgb[0, 0] = [251, 17, 83]
    converted = np.asarray(rgb_to_pil(rgb))
    assert converted[0, 0].tolist() == [251, 17, 83]


def test_detector_accepts_rgb_only_and_normalizes_output() -> None:
    signature = inspect.signature(YoloObjectDetector.detect)
    assert list(signature.parameters) == ["self", "rgb"]
    model = FakeModel()
    detector = YoloObjectDetector(Path("unused.pt"), model=model)
    rgb = np.zeros((24, 32, 3), dtype=np.uint8)
    rgb[0, 0] = [255, 0, 0]
    detections = detector.detect(rgb)
    assert model.received[0, 0].tolist() == [255, 0, 0]
    assert len(detections) == 1
    assert detections[0].object_class is ObjectClass.DUCKIE
    assert detections[0].confidence == pytest.approx(0.75)
    assert detections[0].bottom_center.x_px == pytest.approx(6.0)
    assert detections[0].bottom_center.y_px == pytest.approx(22.0)
    assert model.received.shape == (24, 32, 3)


def test_detector_rejects_non_rgb_and_non_uint8() -> None:
    with pytest.raises(ValueError):
        rgb_to_pil(np.zeros((4, 4), dtype=np.uint8))
    with pytest.raises(ValueError):
        rgb_to_pil(np.zeros((4, 4, 3), dtype=np.float32))


def test_checkpoint_mapping_is_exact() -> None:
    assert validate_v1_class_mapping({0: "stop_sign", 1: "duckie"}) == (
        "stop_sign",
        "duckie",
    )
    with pytest.raises(ValueError):
        validate_v1_class_mapping({0: "duckie", 1: "stop_sign"})
    with pytest.raises(ValueError):
        validate_v1_class_mapping({0: "stop_sign", 1: "duckie", 2: "car"})
