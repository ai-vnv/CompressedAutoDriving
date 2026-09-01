## Task 3: Innovation gate

**Files:**
- Create: `src/duckie_pomdp/belief/innovation_gate.py`
- Test: `tests/test_f9c_innovation_gate.py`

**Interfaces:**
- Consumes: nothing from earlier tasks except NumPy.
- Produces:
  - `InnovationGateConfig(chi_square_threshold: float)`.
  - `GateDecision(accepted: bool, nis: float, threshold: float)`.
  - `normalized_innovation_squared(innovation: NDArray, innovation_covariance: NDArray) -> float`.
  - `InnovationGate.evaluate(innovation: NDArray, innovation_covariance: NDArray) -> GateDecision`.

The gate consumes only the innovation and `S`. It never sees a measurement object, a confidence, or ground truth — that is enforced by the signature.

**Why there is no downweight mode.** An earlier draft offered `GateMode.DOWNWEIGHT`, accepting an inconsistent measurement with `covariance_scale = 25`. That is incompatible with invariant I1. Association and the gate would threshold against `S = HP⁻Hᵀ + λR`, while the correction would then use `S = HP⁻Hᵀ + 25λR` — so `S_gate ≠ S_correction`, and the `λ` fitted to make the NIS median match χ²₂ would no longer describe the covariance the decision boundary was calibrated on. Since the frozen config selects `hard_reject` anyway, F9c v1 implements **only** hard rejection:

```text
NIS ≤ threshold  → accept
NIS >  threshold → reject, EKF runs prediction only
```

A soft-downweighting variant remains a legitimate follow-up experiment if hard rejection turns out to be too aggressive — but it needs its own gate design that keeps one `S`, and it is out of scope here. Do not add a `mode` field "for future flexibility"; an unused branch that violates a stated invariant is worse than no branch.

- [ ] **Step 1: Write the failing tests**

```python
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
    covariance = np.diag([1.0, 1.0])
    innovation = np.array([THRESHOLD**0.5, 0.0])
    decision = _gate().evaluate(innovation, covariance)
    assert decision.nis == pytest.approx(THRESHOLD)
    assert decision.accepted, "NIS exactly at the threshold must be accepted"


def test_gate_rejects_a_non_positive_definite_innovation_covariance():
    with pytest.raises(ValueError, match="positive definite"):
        _gate().evaluate(np.array([0.01, 0.01]), np.diag([0.0004, -1.0e-9]))


def test_gate_signature_cannot_receive_ground_truth():
    import inspect

    parameters = set(inspect.signature(InnovationGate.evaluate).parameters)
    assert parameters == {"self", "innovation", "innovation_covariance"}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `$PY -m pytest tests/test_f9c_innovation_gate.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'duckie_pomdp.belief.innovation_gate'`

- [ ] **Step 3: Write the implementation**

```python
"""Innovation-consistency gate. Consumes only filter statistics, never truth.

Hard rejection only. A soft-downweight branch would make the EKF correction use
a different measurement covariance than the one association and this gate
thresholded against, breaking the single-innovation-covariance invariant.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class InnovationGateConfig:
    chi_square_threshold: float

    def __post_init__(self) -> None:
        if not isfinite(self.chi_square_threshold) or self.chi_square_threshold <= 0.0:
            raise ValueError("chi-square threshold must be finite and positive")


@dataclass(frozen=True)
class GateDecision:
    accepted: bool
    nis: float
    threshold: float


def normalized_innovation_squared(
    innovation: NDArray[np.float64],
    innovation_covariance: NDArray[np.float64],
) -> float:
    vector = np.asarray(innovation, dtype=float)
    matrix = np.asarray(innovation_covariance, dtype=float)
    if vector.shape != (2,) or matrix.shape != (2, 2):
        raise ValueError("innovation gate expects a 2D polar innovation")
    if not np.all(np.isfinite(vector)) or not np.all(np.isfinite(matrix)):
        raise ValueError("innovation statistics must be finite")
    symmetric = 0.5 * (matrix + matrix.T)
    eigenvalues = np.linalg.eigvalsh(symmetric)
    if float(eigenvalues.min()) <= 0.0:
        raise ValueError("innovation covariance must be positive definite")
    return float(vector @ np.linalg.solve(symmetric, vector))


class InnovationGate:
    def __init__(self, config: InnovationGateConfig) -> None:
        self.config = config

    def evaluate(
        self,
        innovation: NDArray[np.float64],
        innovation_covariance: NDArray[np.float64],
    ) -> GateDecision:
        nis = normalized_innovation_squared(innovation, innovation_covariance)
        threshold = self.config.chi_square_threshold
        return GateDecision(nis <= threshold, nis, threshold)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `$PY -m pytest tests/test_f9c_innovation_gate.py -q`
Expected: PASS (8 tests)

- [ ] **Step 5: Checkpoint**

Run: `$PY -m pytest tests -q` → expect 123 passed.

---

## Task 3b: Frozen F9c bias correction as an explicit runtime stage

*(Numbered 3b so the later task numbers referenced throughout this document stay stable.)*

**Files:**
- Create: `src/duckie_pomdp/belief/bias_correction.py`
- Test: `tests/test_f9c_bias_correction.py`

**Interfaces:**
- Consumes: `ObjectMeasurement`, `wrap_angle`.
- Produces:
  - `FrozenBiasCorrection(model: str, range_bias_m: float, bearing_bias_rad: float, range_bin_bias_m: Mapping[str, float] | None, near_max_m: float, medium_max_m: float)` with `correct(measurement: ObjectMeasurement) -> ObjectMeasurement` and classmethods `identity()` and `from_config(data: Mapping) -> FrozenBiasCorrection`.
  - `model ∈ {"identity", "global_additive", "per_range_bin"}`.

**Why this task exists.** An earlier draft fitted `b_r`, `b_β` in Task 9 but never wired them into the Robust-B runtime — the coordinator went straight from candidate to association. Estimating a correction and not applying it is the quietest possible failure: every artifact would look right and the correction would do nothing. The bias stage is therefore a named, separately tested runtime component.

**Locked runtime order:**

```text
z_raw → FROZEN F9c BIAS CORRECTION → z_corr → association → gate → EKF
```

Bias correction runs **before** association, not after, because association thresholds candidates on innovation against `h(x̂⁻)` — comparing an uncorrected measurement against a corrected prediction would inject the full bias into every NIS and would systematically mis-rank candidates in duplicate frames.

**Baseline A keeps the F9b class.** `AdditiveMeasurementBias` in `perception/f9_pipeline.py` stays untouched and is what Baseline A uses, with the F9b constants `b_r = −0.045904804710162034`, `b_β = +0.00414567890700929`. Robust B uses `FrozenBiasCorrection` with the F9c-fitted values. Keeping them as two classes is what makes `bias_refit = false` mean *exactly* "Baseline A's bias", by construction rather than by coincidence.

- [ ] **Step 1: Write the failing tests**

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `$PY -m pytest tests/test_f9c_bias_correction.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'duckie_pomdp.belief.bias_correction'`

- [ ] **Step 3: Implement**

Follow the shape of `AdditiveMeasurementBias.correct` in `perception/f9_pipeline.py:42-61` — same clamping (`max(0.0, r − b)`), same `wrap_angle`, same rebuild of the Cartesian fields from the corrected polar pair. The only additions are the `per_range_bin` lookup and `identity()`. Do **not** modify `AdditiveMeasurementBias`; Baseline A depends on it being exactly what F9b ran.

Bin selection uses the **measured** range, never a predicted or true range — a bias correction that consumed the filter's own prediction would be a feedback loop.

- [ ] **Step 4: Run tests to verify they pass**

Run: `$PY -m pytest tests/test_f9c_bias_correction.py -q`
Expected: PASS (9 tests)

- [ ] **Step 5: Checkpoint**

Run: `$PY -m pytest tests -q` → expect 132 passed.

---

