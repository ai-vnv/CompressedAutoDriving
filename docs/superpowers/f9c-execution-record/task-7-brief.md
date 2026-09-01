## Task 7: Predicted observability and effective detection probability

**Files:**
- Create: `src/duckie_pomdp/belief/observability.py`
- Modify: `src/duckie_pomdp/belief/existence_filter.py`
- Test: `tests/test_f9c_observability.py`, `tests/test_f9c_existence.py`

**Interfaces:**
- Consumes: `duckie_pomdp.perception.camera_geometry` (read it first to reuse the existing intrinsics/extrinsics types — do not re-derive the projection), the frozen `PedestrianEKF` predicted state.
- Produces:
  - `ObservabilityClass` str-enum: `CENTER`, `MID_FOV`, `EDGE_FOV`, `OUTSIDE_DOMAIN`.
  - `PredictedObservability(observability_class: ObservabilityClass, normalized_horizontal_offset: float | None, predicted_range_m: float)`.
  - `PredictedObservabilityModel(projector, image_width_px: int).classify(predicted_state: NDArray) -> PredictedObservability`.
  - `EffectiveDetectionModel(probability_by_class: Mapping[ObservabilityClass, float], *, outside_domain_miss_policy: str)` with two methods: `probability(observability) -> float` and `miss_is_informative(observability) -> bool`, the latter returning `False` for `OUTSIDE_DOMAIN` when the policy is `"prediction_only"` (invariant I3).
  - `ExistenceFilterConfig` gains `detection_probability` as a *default*; `ExistenceFilter.update(detected: bool, *, detection_probability: float | None = None, observation_informative: bool = True) -> float`. With `observation_informative=False` the filter runs the `P_S`/`P_birth` prediction step and returns without applying any likelihood. Defaults (`None`, `True`) reproduce F9b behaviour exactly.

The `probability` / `miss_is_informative` split is what keeps invariant I3 honest: `P_D^eff(OUTSIDE_DOMAIN)` is still *estimated* and reported as a diagnostic, but it is never *applied* to a miss. That prevents the number from silently becoming a tuning knob.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_f9c_observability.py
import numpy as np
import pytest

from duckie_pomdp.belief.observability import (
    EffectiveDetectionModel,
    ObservabilityClass,
    PredictedObservabilityModel,
)


def test_pedestrian_predicted_straight_ahead_is_center(model):
    predicted = model.classify(np.array([0.0, 0.85, 0.0, 0.0]))
    assert predicted.observability_class is ObservabilityClass.CENTER
    assert predicted.normalized_horizontal_offset == pytest.approx(0.0, abs=1e-6)


def test_pedestrian_predicted_far_to_the_side_is_edge_fov(model):
    predicted = model.classify(np.array([0.45, 0.60, 0.0, 0.0]))
    assert predicted.observability_class is ObservabilityClass.EDGE_FOV


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
```

```python
# tests/test_f9c_existence.py
import pytest

from duckie_pomdp.belief.existence_filter import ExistenceFilter, ExistenceFilterConfig

CONFIG = ExistenceFilterConfig(
    prior_probability=0.50,
    detection_probability=0.9766775777414075,
    false_positive_probability=0.00078003120124805,
    survival_probability=0.995,
    birth_probability=0.005,
)


def test_default_update_reproduces_the_frozen_f9b_collapse():
    existence = ExistenceFilter(CONFIG)
    existence.probability = 0.99
    assert existence.update(False) == pytest.approx(0.614, abs=0.01)
    assert existence.update(False) == pytest.approx(0.036, abs=0.01)


def test_a_low_effective_detection_probability_preserves_belief_through_misses():
    existence = ExistenceFilter(CONFIG)
    existence.probability = 0.99
    for _ in range(5):
        existence.update(False, detection_probability=0.10)
    assert existence.probability > 0.60


def test_existence_still_decays_monotonically_under_repeated_misses():
    existence = ExistenceFilter(CONFIG)
    existence.probability = 0.99
    values = [existence.update(False, detection_probability=0.30) for _ in range(20)]
    assert all(later <= earlier for earlier, later in zip(values, values[1:]))
    assert values[-1] < 0.10


def test_belief_recovers_rapidly_after_re_detection():
    existence = ExistenceFilter(CONFIG)
    existence.probability = 0.99
    for _ in range(5):
        existence.update(False, detection_probability=0.30)
    recovered = existence.update(True, detection_probability=0.98)
    assert recovered > 0.95


def test_detection_probability_override_must_exceed_the_false_alarm_rate():
    existence = ExistenceFilter(CONFIG)
    with pytest.raises(ValueError, match="false-positive"):
        existence.update(False, detection_probability=0.0001)


def test_an_uninformative_observation_applies_no_likelihood_at_all():
    """Invariant I3. An outside-domain miss must move P(e) by the survival
    prediction only, never by the miss likelihood ratio."""
    existence = ExistenceFilter(CONFIG)
    existence.probability = 0.90
    expected = 0.995 * 0.90 + 0.005 * (1.0 - 0.90)
    assert existence.update(False, observation_informative=False) == pytest.approx(
        expected
    )


def test_outside_domain_miss_decays_only_through_survival():
    """Invariant I3, over a long absence: 40 uninformative misses must leave
    P(e) far above the in-domain collapse, decaying at the P_S half-life."""
    existence = ExistenceFilter(CONFIG)
    existence.probability = 0.99
    for _ in range(40):
        existence.update(False, observation_informative=False)
    assert existence.probability > 0.80
    informative = ExistenceFilter(CONFIG)
    informative.probability = 0.99
    for _ in range(40):
        informative.update(False, detection_probability=0.97)
    assert informative.probability < 0.01


def test_an_uninformative_observation_ignores_any_detection_probability_passed():
    existence = ExistenceFilter(CONFIG)
    existence.probability = 0.90
    quiet = existence.update(
        False, detection_probability=0.0001, observation_informative=False
    )
    assert quiet == pytest.approx(0.995 * 0.90 + 0.005 * 0.10)


def test_a_detection_still_counts_when_the_belief_predicted_outside_domain():
    """A detection is always evidence, even from a region we predicted was
    unobservable -- that is exactly the signal the prediction was wrong."""
    existence = ExistenceFilter(CONFIG)
    existence.probability = 0.20
    updated = existence.update(True, detection_probability=0.97)
    assert updated > 0.90


def test_ps_is_untouched_by_f9c():
    assert CONFIG.survival_probability == 0.995
```

Build the `model` fixture from the real calibration in `configs/scenario_pomdp_v1.toml` via `CalibratedGroundProjector`, mirroring how `tests/test_camera_geometry.py` constructs one — read that file first.

- [ ] **Step 2: Run tests to verify they fail**

Run: `$PY -m pytest tests/test_f9c_observability.py tests/test_f9c_existence.py -q`
Expected: FAIL — `ModuleNotFoundError` for observability; `TypeError: update() got an unexpected keyword argument 'detection_probability'` for existence.

- [ ] **Step 3: Implement**

`PredictedObservabilityModel.classify` takes the *predicted* EKF state `[x_left, y_forward, …]`, converts to an image column with the existing calibrated projection, and bins `|u − W/2| / (W/2)` with the same thresholds already used by `_fov_region` in `experiments/validate_f9_yolo_ekf.py:62-71` (`<1/3` center, `<2/3` mid, else edge). Return `OUTSIDE_DOMAIN` when `y_forward <= 0` (behind camera) or the projected column falls outside `[0, W)`. Reuse the thresholds by importing or duplicating them with a comment pointing at the source — the evaluation binning and the runtime binning must agree or the calibrated `P_D^eff` will not apply.

In `ExistenceFilter.update`:

```python
def update(
    self,
    detected: bool,
    *,
    detection_probability: float | None = None,
    observation_informative: bool = True,
) -> float:
    config = self.config
    predicted = (
        config.survival_probability * self.probability
        + config.birth_probability * (1.0 - self.probability)
    )
    if not observation_informative:
        # Invariant I3: the camera cannot inform us about a region the belief
        # predicts is unobservable. Prediction only, no likelihood applied.
        self.probability = min(1.0, max(0.0, predicted))
        return self.probability
    probability = (
        config.detection_probability
        if detection_probability is None
        else detection_probability
    )
    if not 0.0 <= probability <= 1.0 or probability <= config.false_positive_probability:
        raise ValueError(
            "effective detection probability must exceed the false-positive rate"
        )
    ...  # existing numerator/denominator arithmetic, unchanged
```

Note the ordering: the validation lives *after* the `observation_informative` early return, so `test_an_uninformative_observation_ignores_any_detection_probability_passed` passes. Do **not** touch the `P_S`/`P_birth` prediction step in either branch.

- [ ] **Step 4: Run tests to verify they pass**

Run: `$PY -m pytest tests/test_f9c_observability.py tests/test_f9c_existence.py -q`
Expected: PASS (18 tests)

- [ ] **Step 5: Checkpoint**

Run: `$PY -m pytest tests -q` → expect 170 passed. The existing `tests/test_pedestrian_ekf.py` existence tests must still pass unmodified — that is the proof the default path is unchanged.

---

