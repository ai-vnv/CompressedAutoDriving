"""Scalar Bayesian existence filter kept separate from EKF kinematics.

**Invariant I8 (Task 9 fix round 1): the miss-likelihood floor.** F9c's own
calibration (seeds 6101-6108) found the detector genuinely reliable -- the
per-class effective detection probabilities are all >= 0.949 -- yet belief
still collapses fast on a miss, because a single independent-Bernoulli miss
carries strong evidence against existence (``LR_miss = (1-P_D)/(1-P_FA)`` is
small). The real mis-specification is that misses are *bursty*: a
correlated run of several consecutive misses (motion blur, brief occlusion,
an unlucky frame) is not evidence as strong as that many *independent*
misses would be, but the plain Bayesian update below treats every miss as
independent. ``miss_likelihood_floor`` bounds how far a single miss's
likelihood ratio can push the posterior:

``LR_used = max(LR_miss, miss_likelihood_floor)``

The floor is fit offline as ``LR_floor = LR_nominal ** (1 / L_mean)``, where
``L_mean`` is the MEASURED mean length of consecutive genuine-miss runs on
calibration data and ``LR_nominal`` is the global (not per-class) fitted
miss likelihood ratio -- see
``evaluation.f9c_calibration.mean_genuine_miss_run_length`` and
``fit_global_effective_detection``. This makes a run of exactly ``L_mean``
floored misses reproduce ``LR_floor ** L_mean == LR_nominal`` exactly: the
same cumulative evidence as ONE unfloored miss, i.e. a typical-length burst
is treated as roughly one independent miss rather than ``L_mean`` of them. A
run materially longer than ``L_mean`` still drives existence down (the floor
bounds a single miss's evidence; it does not cap a genuinely long absence).
Never hand-picked, never tuned to hit a retention target -- it is a measured
quantity, the same species of correction as the posterior-variance floor in
``covariance_calibration.py``: bound the influence of a model that is
over-confident about its own reliability.

The default ``miss_likelihood_floor = 0.0`` is a strict no-op: since
``LR_nominal = (1 - P_D) / (1 - P_FA) > 0`` for any valid config (validated
below, ``detection_probability > false_positive_probability``),
``max(LR_nominal, 0.0) == LR_nominal`` always, reproducing every existing
caller's behaviour bit-for-bit.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite


@dataclass(frozen=True)
class ExistenceFilterConfig:
    prior_probability: float
    detection_probability: float
    false_positive_probability: float
    survival_probability: float
    birth_probability: float
    miss_likelihood_floor: float = 0.0

    def __post_init__(self) -> None:
        for name, value in self.__dict__.items():
            if not isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be within [0, 1]")
        if self.detection_probability <= self.false_positive_probability:
            raise ValueError("detection probability must exceed false-positive probability")


class ExistenceFilter:
    def __init__(self, config: ExistenceFilterConfig) -> None:
        self.config = config
        self.probability = config.prior_probability

    def reset(self) -> None:
        self.probability = self.config.prior_probability

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
            # Invariant I3: the camera cannot inform us about a region the
            # belief predicts is unobservable. Prediction only, no
            # likelihood applied -- and no validation of a detection
            # probability that will never be used.
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

        if detected:
            numerator = probability * predicted
            denominator = numerator + config.false_positive_probability * (
                1.0 - predicted
            )
        else:
            # Invariant I8: floor the miss likelihood ratio. Algebraically
            # equivalent to the un-floored form when
            # lr_used == lr_nominal -- multiplying numerator and denominator
            # by 1/(1 - false_positive_probability) turns
            # `(1-P_D)*predicted / [(1-P_D)*predicted + (1-P_FA)*(1-predicted)]`
            # into `lr_nominal*predicted / [lr_nominal*predicted + (1-predicted)]`,
            # the same ratio -- so this is a strict generalisation, not a
            # different formula that happens to agree at the default.
            lr_nominal = (1.0 - probability) / (
                1.0 - config.false_positive_probability
            )
            lr_used = max(lr_nominal, config.miss_likelihood_floor)
            numerator = lr_used * predicted
            denominator = numerator + (1.0 - predicted)
        self.probability = (
            predicted if denominator <= 0.0 else numerator / denominator
        )
        self.probability = min(1.0, max(0.0, self.probability))
        return self.probability

