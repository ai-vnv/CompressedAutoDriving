# tests/test_f9c_association.py
import numpy as np
import pytest

from duckie_pomdp.belief.measurement_association import (
    AssociationConfig,
    CandidateMeasurement,
    MeasurementAssociator,
)
from duckie_pomdp.domain.detection import ObjectClass
from duckie_pomdp.domain.measurement import ObjectMeasurement

GATE = 9.21034037197618


def _candidate(range_m, bearing_rad, confidence, bbox=(0, 0, 10, 10)):
    from math import cos, sin

    return CandidateMeasurement(
        measurement=ObjectMeasurement(
            object_class=ObjectClass.DUCKIE,
            detected=True,
            confidence=confidence,
            x_left_m=range_m * sin(bearing_rad),
            y_forward_m=range_m * cos(bearing_rad),
            range_m=range_m,
            bearing_rad=bearing_rad,
        ),
        confidence=confidence,
        bbox_key=bbox,
    )


def _covariance_for(_range_m):
    return np.diag([4.0e-4, 1.6e-4])


def _associator():
    return MeasurementAssociator(
        AssociationConfig(
            chi_square_gate=GATE,
            initialization_rule="highest_confidence_then_bbox_lexicographic",
        )
    )


def test_without_an_active_track_the_highest_confidence_candidate_initializes():
    result = _associator().associate(
        [_candidate(0.90, 0.01, 0.42), _candidate(0.70, 0.05, 0.81)],
        predicted_measurement=None,
        innovation_covariance_for=_covariance_for,
    )
    assert result.mode == "initialization"
    assert result.selected_index == 1


def test_initialization_breaks_exact_confidence_ties_deterministically():
    result = _associator().associate(
        [
            _candidate(0.90, 0.01, 0.50, bbox=(30, 0, 40, 10)),
            _candidate(0.70, 0.05, 0.50, bbox=(10, 0, 20, 10)),
        ],
        predicted_measurement=None,
        innovation_covariance_for=_covariance_for,
    )
    assert result.selected_index == 1, "lexicographically smallest bbox wins ties"


def test_with_an_active_track_the_most_consistent_candidate_wins_over_confidence():
    result = _associator().associate(
        [_candidate(0.62, 0.30, 0.95), _candidate(0.90, 0.02, 0.31)],
        predicted_measurement=np.array([0.90, 0.02]),
        innovation_covariance_for=_covariance_for,
    )
    assert result.mode == "temporal"
    assert result.selected_index == 1
    assert result.highest_confidence_index == 0
    assert result.differed_from_highest_confidence


def test_association_wraps_bearing_across_pi():
    from math import pi

    result = _associator().associate(
        [_candidate(0.90, -pi + 0.01, 0.40)],
        predicted_measurement=np.array([0.90, pi - 0.01]),
        innovation_covariance_for=_covariance_for,
    )
    assert result.mode == "temporal"
    assert result.selected_index == 0
    assert result.candidate_nis[0] == pytest.approx(0.02**2 / 1.6e-4, rel=1e-6)


def test_every_candidate_outside_the_gate_yields_no_selection():
    result = _associator().associate(
        [_candidate(1.80, 0.50, 0.90)],
        predicted_measurement=np.array([0.90, 0.02]),
        innovation_covariance_for=_covariance_for,
    )
    assert result.mode == "all_gated_out"
    assert result.selected is None


def test_no_candidates_is_reported_distinctly_from_all_gated_out():
    result = _associator().associate(
        [],
        predicted_measurement=np.array([0.90, 0.02]),
        innovation_covariance_for=_covariance_for,
    )
    assert result.mode == "no_candidate"
    assert result.selected is None


def test_associate_signature_admits_no_privileged_argument():
    import inspect

    parameters = set(inspect.signature(MeasurementAssociator.associate).parameters)
    assert parameters == {
        "self",
        "candidates",
        "predicted_measurement",
        "innovation_covariance_for",
    }
