# tests/test_f9c_bias_correction.py
import pytest

from duckie_pomdp.belief.bias_correction import FrozenBiasCorrection
from duckie_pomdp.domain.detection import ObjectClass
from duckie_pomdp.domain.measurement import ObjectMeasurement


def _measurement(range_m, bearing_rad):
    from math import cos, sin

    return ObjectMeasurement(
        object_class=ObjectClass.DUCKIE,
        detected=True,
        confidence=0.8,
        x_left_m=range_m * sin(bearing_rad),
        y_forward_m=range_m * cos(bearing_rad),
        range_m=range_m,
        bearing_rad=bearing_rad,
    )


def test_global_additive_subtracts_the_frozen_bias():
    correction = FrozenBiasCorrection(
        model="global_additive",
        range_bias_m=-0.0459,
        bearing_bias_rad=0.0041,
        range_bin_bias_m=None,
        near_max_m=0.55,
        medium_max_m=0.80,
    )
    corrected = correction.correct(_measurement(0.900, 0.050))
    assert corrected.range_m == pytest.approx(0.900 + 0.0459)
    assert corrected.bearing_rad == pytest.approx(0.050 - 0.0041)


def test_cartesian_fields_stay_consistent_with_the_corrected_polar_pair():
    from math import cos, sin

    correction = FrozenBiasCorrection(
        "global_additive", -0.0459, 0.0041, None, 0.55, 0.80
    )
    corrected = correction.correct(_measurement(0.900, 0.050))
    assert corrected.x_left_m == pytest.approx(
        corrected.range_m * sin(corrected.bearing_rad)
    )
    assert corrected.y_forward_m == pytest.approx(
        corrected.range_m * cos(corrected.bearing_rad)
    )


def test_per_range_bin_selects_the_bin_from_the_measured_range():
    correction = FrozenBiasCorrection(
        model="per_range_bin",
        range_bias_m=0.0,
        bearing_bias_rad=0.0041,
        range_bin_bias_m={"near": -0.0255, "medium": -0.0417, "far": -0.0479},
        near_max_m=0.55,
        medium_max_m=0.80,
    )
    assert correction.correct(_measurement(0.50, 0.0)).range_m == pytest.approx(
        0.50 + 0.0255
    )
    assert correction.correct(_measurement(0.70, 0.0)).range_m == pytest.approx(
        0.70 + 0.0417
    )
    assert correction.correct(_measurement(0.95, 0.0)).range_m == pytest.approx(
        0.95 + 0.0479
    )


def test_identity_correction_is_a_no_op():
    correction = FrozenBiasCorrection.identity()
    original = _measurement(0.900, 0.050)
    corrected = correction.correct(original)
    assert corrected.range_m == pytest.approx(0.900)
    assert corrected.bearing_rad == pytest.approx(0.050)


def test_correction_leaves_a_missing_measurement_untouched():
    correction = FrozenBiasCorrection("global_additive", -0.0459, 0.0041, None, 0.55, 0.80)
    missing = ObjectMeasurement.missing(ObjectClass.DUCKIE)
    assert correction.correct(missing) is missing


def test_corrected_range_is_clamped_at_zero():
    correction = FrozenBiasCorrection("global_additive", 0.50, 0.0, None, 0.55, 0.80)
    assert correction.correct(_measurement(0.20, 0.0)).range_m == 0.0


def test_bearing_correction_wraps_across_pi():
    from math import pi

    correction = FrozenBiasCorrection("global_additive", 0.0, -0.02, None, 0.55, 0.80)
    corrected = correction.correct(_measurement(0.90, pi - 0.01))
    assert corrected.bearing_rad == pytest.approx(-pi + 0.01)


def test_per_range_bin_requires_all_three_bins():
    with pytest.raises(ValueError, match="near, medium, far"):
        FrozenBiasCorrection(
            "per_range_bin", 0.0, 0.0, {"near": -0.02}, 0.55, 0.80
        )


def test_correction_never_receives_ground_truth():
    import inspect

    assert set(inspect.signature(FrozenBiasCorrection.correct).parameters) == {
        "self",
        "measurement",
    }


def test_identity_model_is_a_no_op_even_with_nonzero_bias_fields():
    """`identity` is a semantic guarantee, not a consequence of zeroed fields.

    `from_config` can build model="identity" carrying stray bias values without
    passing through the identity() classmethod, so the no-op must hold
    regardless of what the fields happen to contain.
    """
    correction = FrozenBiasCorrection(
        model="identity",
        range_bias_m=-0.0459,
        bearing_bias_rad=0.0041,
        range_bin_bias_m={"near": -0.0255, "medium": -0.0417, "far": -0.0479},
        near_max_m=0.55,
        medium_max_m=0.80,
    )
    original = _measurement(0.900, 0.050)
    corrected = correction.correct(original)
    assert corrected.range_m == pytest.approx(0.900)
    assert corrected.bearing_rad == pytest.approx(0.050)
    assert corrected.x_left_m == pytest.approx(original.x_left_m)
    assert corrected.y_forward_m == pytest.approx(original.y_forward_m)
