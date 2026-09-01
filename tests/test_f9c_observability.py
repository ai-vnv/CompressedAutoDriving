# tests/test_f9c_observability.py
import numpy as np
import pytest

from duckie_pomdp.belief.observability import (
    EffectiveDetectionModel,
    ObservabilityClass,
    PredictedObservabilityModel,
)
from duckie_pomdp.perception.camera_geometry import (
    CalibratedGroundProjector,
    CameraCalibration,
)


@pytest.fixture
def model():
    projector = CalibratedGroundProjector(
        CameraCalibration(
            image_width_px=640,
            image_height_px=480,
            vertical_fov_deg=75.0,
            camera_height_m=0.108,
            camera_pitch_deg=19.15,
            camera_forward_offset_m=0.066,
        )
    )
    return PredictedObservabilityModel(projector, image_width_px=640)


def test_pedestrian_predicted_straight_ahead_is_center(model):
    predicted = model.classify(np.array([0.0, 0.85, 0.0, 0.0]))
    assert predicted.observability_class is ObservabilityClass.CENTER
    assert predicted.normalized_horizontal_offset == pytest.approx(0.0, abs=1e-6)


def test_pedestrian_predicted_far_to_the_side_is_edge_fov(model):
    predicted = model.classify(np.array([0.45, 0.60, 0.0, 0.0]))
    assert predicted.observability_class is ObservabilityClass.EDGE_FOV


def test_pedestrian_predicted_between_centre_and_edge_is_mid_fov(model):
    predicted = model.classify(np.array([0.30, 0.85, 0.0, 0.0]))
    assert predicted.observability_class is ObservabilityClass.MID_FOV


def test_pedestrian_predicted_beyond_the_image_is_outside_domain(model):
    predicted = model.classify(np.array([3.0, 0.40, 0.0, 0.0]))
    assert predicted.observability_class is ObservabilityClass.OUTSIDE_DOMAIN


def test_pedestrian_predicted_behind_the_camera_is_outside_domain(model):
    predicted = model.classify(np.array([0.0, -0.50, 0.0, 0.0]))
    assert predicted.observability_class is ObservabilityClass.OUTSIDE_DOMAIN
    assert predicted.normalized_horizontal_offset is None


def test_classification_uses_no_privileged_input(model):
    import inspect

    parameters = set(inspect.signature(PredictedObservabilityModel.classify).parameters)
    assert parameters == {"self", "predicted_state"}


def _detection_model(policy="prediction_only"):
    return EffectiveDetectionModel(
        {
            ObservabilityClass.CENTER: 0.99,
            ObservabilityClass.MID_FOV: 0.97,
            ObservabilityClass.EDGE_FOV: 0.72,
            ObservabilityClass.OUTSIDE_DOMAIN: 0.05,
        },
        outside_domain_miss_policy=policy,
    )


def test_effective_detection_probability_is_lower_at_the_edge_of_the_field_of_view():
    from duckie_pomdp.belief.observability import PredictedObservability

    model = _detection_model()
    center = PredictedObservability(ObservabilityClass.CENTER, 0.0, 0.85)
    edge = PredictedObservability(ObservabilityClass.EDGE_FOV, 0.8, 0.85)
    assert model.probability(center) == 0.99
    assert model.probability(edge) == 0.72


def test_an_outside_domain_miss_is_declared_uninformative():
    from duckie_pomdp.belief.observability import PredictedObservability

    model = _detection_model()
    outside = PredictedObservability(ObservabilityClass.OUTSIDE_DOMAIN, None, 0.85)
    edge = PredictedObservability(ObservabilityClass.EDGE_FOV, 0.8, 0.85)
    assert not model.miss_is_informative(outside)
    assert model.miss_is_informative(edge)
    # The probability is still reported, it is simply never applied to a miss.
    assert model.probability(outside) == 0.05


def test_effective_detection_model_rejects_a_missing_class():
    with pytest.raises(ValueError, match="every observability class"):
        EffectiveDetectionModel(
            {ObservabilityClass.CENTER: 0.99},
            outside_domain_miss_policy="prediction_only",
        )


def test_effective_detection_model_rejects_an_unknown_outside_domain_policy():
    """Invariant I3's guard rail needs its own guard: an unrecognised policy
    string must fail loudly at construction, not silently make outside-domain
    misses informative again."""
    with pytest.raises(ValueError, match="prediction_only"):
        EffectiveDetectionModel(
            {
                ObservabilityClass.CENTER: 0.99,
                ObservabilityClass.MID_FOV: 0.97,
                ObservabilityClass.EDGE_FOV: 0.72,
                ObservabilityClass.OUTSIDE_DOMAIN: 0.05,
            },
            outside_domain_miss_policy="prediction-only",
        )
