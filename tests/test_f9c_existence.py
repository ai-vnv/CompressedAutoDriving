# tests/test_f9c_existence.py
from dataclasses import replace

import pytest

from duckie_pomdp.belief.existence_filter import ExistenceFilter, ExistenceFilterConfig

CONFIG = ExistenceFilterConfig(
    prior_probability=0.50,
    detection_probability=0.9766775777414075,
    false_positive_probability=0.00078003120124805,
    survival_probability=0.995,
    birth_probability=0.005,
)

_LR_NOMINAL = (1.0 - CONFIG.detection_probability) / (1.0 - CONFIG.false_positive_probability)


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


# ---------------------------------------------------------------------------
# Invariant I8: the miss-likelihood floor (Task 9 fix round 1).
# ---------------------------------------------------------------------------


def test_the_default_floor_is_a_no_op():
    assert CONFIG.miss_likelihood_floor == 0.0


def test_the_default_no_op_floor_reproduces_todays_behaviour_exactly():
    """The default config's miss branch must be bit-identical to the
    pre-I8 formula: LR_used = max(LR_nominal, 0.0) == LR_nominal always,
    since LR_nominal > 0 for any valid P_D > P_FA."""

    baseline = ExistenceFilter(CONFIG)
    baseline.probability = 0.99
    assert baseline.update(False) == pytest.approx(0.614, abs=0.01)
    assert baseline.update(False) == pytest.approx(0.036, abs=0.01)


def test_the_floor_bounds_a_single_miss():
    """A miss whose instantaneous detection_probability implies an
    LR far below the floor must not collapse belief as far as the
    unfloored update would."""

    floor = _LR_NOMINAL ** (1.0 / 7.0)
    high_confidence_detection = 0.999  # instantaneous LR ~ 0.001, << floor

    floored = ExistenceFilter(replace(CONFIG, miss_likelihood_floor=floor))
    floored.probability = 0.90
    floored.update(False, detection_probability=high_confidence_detection)

    unfloored = ExistenceFilter(CONFIG)
    unfloored.probability = 0.90
    unfloored.update(False, detection_probability=high_confidence_detection)

    assert floored.probability > unfloored.probability


def test_a_run_of_l_mean_misses_reproduces_the_nominal_single_miss_evidence():
    """With survival=1/birth=0 (no prediction drift), each miss update
    multiplies the existence ODDS by LR_used exactly (derived algebraically
    in the module docstring). L_mean floored misses therefore multiply the
    odds by floor**L_mean == LR_nominal by construction of the floor -- the
    SAME odds shift as one single UNFLOORED miss from the same prior."""

    l_mean = 7
    floor = _LR_NOMINAL ** (1.0 / l_mean)
    static_config = replace(CONFIG, survival_probability=1.0, birth_probability=0.0)
    floored_config = replace(static_config, miss_likelihood_floor=floor)

    burst = ExistenceFilter(floored_config)
    burst.probability = 0.90
    for _ in range(l_mean):
        burst.update(False)

    single = ExistenceFilter(static_config)  # floor=0.0 default -> LR_used=LR_nominal
    single.probability = 0.90
    single.update(False)

    assert burst.probability == pytest.approx(single.probability, rel=1e-9)


def test_a_much_longer_run_still_decays_below_the_active_threshold():
    """A run many times longer than L_mean must still drive existence below
    the active threshold -- the floor bounds a SINGLE miss's evidence, it
    does not prevent a genuinely long absence from being believed."""

    l_mean = 7
    floor = _LR_NOMINAL ** (1.0 / l_mean)
    floored = ExistenceFilter(replace(CONFIG, miss_likelihood_floor=floor))
    floored.probability = 0.99
    for _ in range(20 * l_mean):
        floored.update(False)
    assert floored.probability < 0.50


def test_a_detection_is_unaffected_by_the_floor():
    floor = 0.9  # deliberately large, would matter a lot on a miss

    floored = ExistenceFilter(replace(CONFIG, miss_likelihood_floor=floor))
    floored.probability = 0.50
    floored_result = floored.update(True, detection_probability=0.97)

    unfloored = ExistenceFilter(CONFIG)
    unfloored.probability = 0.50
    unfloored_result = unfloored.update(True, detection_probability=0.97)

    assert floored_result == pytest.approx(unfloored_result, abs=1e-12)
