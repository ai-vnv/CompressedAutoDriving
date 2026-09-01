## Task 4: Temporal measurement association

**Files:**
- Create: `src/duckie_pomdp/belief/measurement_association.py`
- Test: `tests/test_f9c_association.py`

**Interfaces:**
- Consumes: `duckie_pomdp.domain.measurement.ObjectMeasurement`, `normalized_innovation_squared` (Task 3).
- Produces:
  - `AssociationConfig(chi_square_gate: float, initialization_rule: str)`.
  - `CandidateMeasurement(measurement: ObjectMeasurement, confidence: float, bbox_key: tuple[int, int, int, int])`.
  - `AssociationResult(selected_index: int | None, selected: CandidateMeasurement | None, mode: str, candidate_nis: tuple[float | None, ...], highest_confidence_index: int | None, differed_from_highest_confidence: bool)` where `mode ∈ {"initialization", "temporal", "no_candidate", "all_gated_out"}`.
  - `MeasurementAssociator.associate(candidates, *, predicted_measurement, innovation_covariance_for) -> AssociationResult`, where `predicted_measurement: NDArray | None` is `h(x̂⁻)` (None means no active track) and `innovation_covariance_for: Callable[[float], NDArray]` returns `S` for a candidate range.

Association lives in the belief layer, not in perception — perception only produces candidates. This keeps the hexagonal boundary intact.

**Invariant I1:** `innovation_covariance_for` is injected, not constructed here, precisely so that the coordinator can hand the *same* provider to the associator, the gate, and the EKF correction. The associator must never build an `S` of its own. Document this in the module docstring: `associate` thresholds against exactly the covariance its caller will later correct with.

- [ ] **Step 1: Write the failing tests**

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `$PY -m pytest tests/test_f9c_association.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'duckie_pomdp.belief.measurement_association'`

- [ ] **Step 3: Write the implementation**

Key points: the innovation is `[z_r − ẑ_r, wrap_angle(z_β − ẑ_β)]` using `duckie_pomdp.perception.measurement_calibration.wrap_angle`; `S` comes from `innovation_covariance_for(candidate_range)`; selection is `argmin` over NIS subject to `nis <= chi_square_gate`; `highest_confidence_index` is computed with the same `(-confidence, bbox_key)` ordering as `select_single_duckie` so the diagnostic comparison is apples-to-apples. When `predicted_measurement is None`, skip NIS entirely and return `mode="initialization"` with `candidate_nis` all `None`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `$PY -m pytest tests/test_f9c_association.py -q`
Expected: PASS (7 tests)

- [ ] **Step 5: Checkpoint**

Run: `$PY -m pytest tests -q` → expect 139 passed.

---

