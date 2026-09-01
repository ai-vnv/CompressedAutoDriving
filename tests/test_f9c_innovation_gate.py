# tests/test_f9c_innovation_gate.py
import numpy as np
import pytest

from duckie_pomdp.belief.innovation_gate import (
    InnovationGate,
    InnovationGateConfig,
    normalized_innovation_squared,
)

THRESHOLD = 9.21034037197618


def _gate():
    return InnovationGate(InnovationGateConfig(chi_square_threshold=THRESHOLD))


def test_nis_matches_the_closed_form_for_a_diagonal_covariance():
    innovation = np.array([0.03, 0.01])
    covariance = np.diag([0.0004, 0.0001])
    expected = 0.03**2 / 0.0004 + 0.01**2 / 0.0001
    assert normalized_innovation_squared(innovation, covariance) == pytest.approx(expected)


def test_nis_uses_the_full_matrix_not_only_the_diagonal():
    innovation = np.array([0.02, -0.02])
    covariance = np.array([[4.0e-4, 3.0e-4], [3.0e-4, 4.0e-4]])
    expected = float(innovation @ np.linalg.solve(covariance, innovation))
    assert normalized_innovation_squared(innovation, covariance) == pytest.approx(expected)
    assert expected != pytest.approx(0.02**2 / 4.0e-4 + 0.02**2 / 4.0e-4)


def test_consistent_measurement_is_accepted():
    decision = _gate().evaluate(np.array([0.005, 0.002]), np.diag([0.0004, 0.0001]))
    assert decision.accepted
    assert decision.nis < THRESHOLD


def test_gross_outlier_is_rejected():
    decision = _gate().evaluate(np.array([0.30, 0.0]), np.diag([0.0004, 0.0001]))
    assert not decision.accepted
    assert decision.nis > THRESHOLD


def test_the_gate_exposes_no_covariance_scaling_knob():
    """Invariant I1: the gate must not be able to hand the correction a
    different R than association and the gate itself thresholded against."""
    decision = _gate().evaluate(np.array([0.30, 0.0]), np.diag([0.0004, 0.0001]))
    assert not hasattr(decision, "covariance_scale")
    assert not hasattr(decision, "downweighted")
    import duckie_pomdp.belief.innovation_gate as module

    assert not hasattr(module, "GateMode")


def test_gate_is_exactly_inclusive_at_the_threshold():
    # Build the boundary exactly rather than round-tripping through sqrt:
    # (THRESHOLD**0.5)**2 lands ~2e-15 above THRESHOLD, so an inclusive gate
    # would appear to reject its own boundary for purely floating-point reasons.
    covariance = np.diag([1.0, 1.0])
    innovation = np.array([3.0, 0.0])
    nis = normalized_innovation_squared(innovation, covariance)
    assert nis == 9.0
    gate = InnovationGate(InnovationGateConfig(chi_square_threshold=nis))
    decision = gate.evaluate(innovation, covariance)
    assert decision.nis == nis
    assert decision.accepted, "NIS exactly at the threshold must be accepted"
    just_above = gate.evaluate(np.array([3.0 + 1.0e-6, 0.0]), covariance)
    assert not just_above.accepted, "the boundary must still be a boundary"


def test_gate_rejects_a_non_positive_definite_innovation_covariance():
    with pytest.raises(ValueError, match="positive definite"):
        _gate().evaluate(np.array([0.01, 0.01]), np.diag([0.0004, -1.0e-9]))


def test_gate_signature_cannot_receive_ground_truth():
    import inspect

    parameters = set(inspect.signature(InnovationGate.evaluate).parameters)
    assert parameters == {"self", "innovation", "innovation_covariance"}
