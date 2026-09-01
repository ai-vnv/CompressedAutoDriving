import numpy as np
import pytest

from duckie_pomdp.domain.detection import BoundingBox, Detection, ObjectClass
from duckie_pomdp.perception.confidence_filter import (
    ClassConfidenceFilter,
    DuckieImageDomainFilter,
)


class _Detector:
    def __init__(self, detections):
        self.detections = tuple(detections)
        self.received = None

    def detect(self, rgb):
        self.received = rgb
        return self.detections


def _detection(object_class: ObjectClass, confidence: float) -> Detection:
    return Detection(object_class, confidence, BoundingBox(1.0, 2.0, 3.0, 4.0))


def test_class_filter_uses_rgb_only_and_does_not_filter_other_classes() -> None:
    base = _Detector(
        (
            _detection(ObjectClass.DUCKIE, 0.39),
            _detection(ObjectClass.DUCKIE, 0.40),
            _detection(ObjectClass.STOP_SIGN, 0.10),
        )
    )
    filtered = ClassConfidenceFilter(base, {ObjectClass.DUCKIE: 0.40})
    rgb = np.zeros((8, 8, 3), dtype=np.uint8)

    result = filtered.detect(rgb)

    assert base.received is rgb
    assert [(item.object_class, item.confidence) for item in result] == [
        (ObjectClass.DUCKIE, 0.40),
        (ObjectClass.STOP_SIGN, 0.10),
    ]


@pytest.mark.parametrize("threshold", [-0.01, 1.01, float("nan")])
def test_class_filter_rejects_invalid_thresholds(threshold: float) -> None:
    with pytest.raises(ValueError, match="within"):
        ClassConfidenceFilter(_Detector(()), {ObjectClass.DUCKIE: threshold})


def test_duckie_image_domain_filter_rejects_low_image_false_positive_only() -> None:
    accepted_duckie = Detection(
        ObjectClass.DUCKIE, 0.9, BoundingBox(10.0, 120.0, 60.0, 232.0)
    )
    yellow_line_false_positive = Detection(
        ObjectClass.DUCKIE, 0.8, BoundingBox(20.0, 250.0, 90.0, 300.0)
    )
    stop_sign = Detection(
        ObjectClass.STOP_SIGN, 0.8, BoundingBox(20.0, 250.0, 90.0, 300.0)
    )
    base = _Detector((accepted_duckie, yellow_line_false_positive, stop_sign))
    filtered = DuckieImageDomainFilter(base, maximum_bottom_y_px=240.0)
    rgb = np.zeros((480, 640, 3), dtype=np.uint8)

    assert filtered.detect(rgb) == (accepted_duckie, stop_sign)
    assert base.received is rgb


@pytest.mark.parametrize("maximum", [0.0, -1.0, float("nan")])
def test_duckie_image_domain_filter_rejects_invalid_maximum(maximum: float) -> None:
    with pytest.raises(ValueError, match="positive"):
        DuckieImageDomainFilter(_Detector(()), maximum_bottom_y_px=maximum)
