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

