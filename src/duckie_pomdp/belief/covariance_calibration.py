"""Covariance calibration: R inflation and a posterior variance floor.

Gate F9c found that the pedestrian belief's *point estimates* were acceptable
but its *reported uncertainty* was severely overconfident: a filter range
standard deviation of ~0.0058 m against an actual RMSE of ~0.028 m, with
nominal 68%/95% intervals covering only ~15%/26% of the truth. A variance
decomposition attributed roughly 78% of the range error to a per-episode or
per-seed *offset* rather than per-frame noise. Because the EKF's process
noise is frozen and small, a constant offset within an episode is never
averaged away by filtering, and the posterior variance keeps shrinking with
more frames regardless of how the measurement noise ``R`` is inflated.

This module therefore provides two independent mechanisms with two
independent jobs:

- **R inflation (``range_scale``, ``bearing_scale``)** scales the measurement
  covariance so the innovation covariance ``S`` is honest, which calibrates
  gate/association thresholds. It does *not* fix long-episode coverage,
  because inflating R still lets the posterior variance shrink with more
  frames.
- **Posterior variance floor (``range_posterior_floor_m``,
  ``bearing_posterior_floor_rad``)** is added, in quadrature, to the
  filter's *reported* standard deviation at the belief-reporting boundary.
  It makes the reported uncertainty honest by adding back the offset
  variance that the filter structurally cannot see: the posterior floor
  represents observation bias that varies slowly within an episode and is
  therefore not averaged away by the EKF. Version 1 does not augment the
  frozen EKF state with a bias term; the floor is a documented approximation
  of that missing state, calibrated from measured between-group variance.

**Methodological honesty.** ``estimate_variance_components`` and
``estimate_nested_variance_components`` are an *approximate nested
variance-component estimator*: the one-level ANOVA moment estimator applied
twice (once to seeds, once to seed-centred residuals grouped by episode).
This is **not** a REML mixed-effects fit and must not be described as one.
It reports no standard errors on the fitted components and can be biased
under strong group-size imbalance. That is acceptable because the estimate
feeds an uncertainty *floor*, which only needs to be right to within a
modest factor to restore coverage, and because the project is constrained
to NumPy only, with no new dependencies (e.g. no ``statsmodels`` mixed-model
solver).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class CovarianceCalibration:
    range_scale: float
    bearing_scale: float
    range_posterior_floor_m: float
    bearing_posterior_floor_rad: float

    def __post_init__(self) -> None:
        if self.range_scale < 1.0:
            raise ValueError("range_scale must be at least 1")
        if self.bearing_scale < 1.0:
            raise ValueError("bearing_scale must be at least 1")
        if self.range_posterior_floor_m < 0.0:
            raise ValueError(
                "range_posterior_floor_m must be non-negative "
                f"(got {self.range_posterior_floor_m!r}); a negative floor "
                "would shrink reported uncertainty in "
                "floor_polar_standard_deviation's quadrature sum instead of "
                "inflating it, silently defeating the calibration"
            )
        if self.bearing_posterior_floor_rad < 0.0:
            raise ValueError(
                "bearing_posterior_floor_rad must be non-negative "
                f"(got {self.bearing_posterior_floor_rad!r}); a negative "
                "floor would shrink reported uncertainty in "
                "floor_polar_standard_deviation's quadrature sum instead of "
                "inflating it, silently defeating the calibration"
            )

    def inflate(
        self, measurement_covariance: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        """Scale the range and bearing variance independently.

        Returns a new matrix; ``measurement_covariance`` is never mutated.
        """

        inflated = np.array(measurement_covariance, dtype=float, copy=True)
        inflated[0, 0] *= self.range_scale
        inflated[1, 1] *= self.bearing_scale
        return inflated

    def floor_polar_standard_deviation(
        self, range_std_m: float, bearing_std_rad: float
    ) -> tuple[float, float]:
        """Add the posterior floor in quadrature; never shrinks the input."""

        floored_range = (range_std_m**2 + self.range_posterior_floor_m**2) ** 0.5
        floored_bearing = (
            bearing_std_rad**2 + self.bearing_posterior_floor_rad**2
        ) ** 0.5
        return floored_range, floored_bearing


@dataclass(frozen=True)
class VarianceComponents:
    between_group_variance: float
    within_group_variance: float
    group_count: int
    sample_count: int


def estimate_variance_components(
    residuals_by_group: Mapping[str, Sequence[float]],
) -> VarianceComponents:
    """One-level unbalanced ANOVA moment estimator (Henderson's method).

    ``MS_within = sum_i sum_j (x_ij - xbar_i)^2 / (N - k)``
    ``MS_between = sum_i n_i (xbar_i - xbar)^2 / (k - 1)``
    ``n_effective = (N - sum_i n_i^2 / N) / (k - 1)``
    ``tau^2 = max(0, (MS_between - MS_within) / n_effective)``

    ``tau^2`` is clamped at zero: a moment estimator can produce a negative
    estimate when the true between-group variance is near zero, which is not
    a valid variance.
    """

    groups = [np.asarray(values, dtype=float) for values in residuals_by_group.values()]
    k = len(groups)
    if k < 2:
        raise ValueError("at least two groups are required to estimate variance components")

    group_sizes = np.array([group.size for group in groups], dtype=float)
    if np.any(group_sizes < 1):
        raise ValueError("every group must contain at least one sample")

    n_total = int(group_sizes.sum())
    group_means = np.array([group.mean() for group in groups], dtype=float)
    grand_mean = np.concatenate(groups).mean()

    within_ss = sum(
        float(np.sum((group - mean) ** 2))
        for group, mean in zip(groups, group_means)
    )
    ms_within = within_ss / (n_total - k)

    between_ss = float(np.sum(group_sizes * (group_means - grand_mean) ** 2))
    ms_between = between_ss / (k - 1)

    n_effective = (n_total - float(np.sum(group_sizes**2)) / n_total) / (k - 1)
    between_group_variance = max(0.0, (ms_between - ms_within) / n_effective)

    return VarianceComponents(
        between_group_variance=between_group_variance,
        within_group_variance=ms_within,
        group_count=k,
        sample_count=n_total,
    )


@dataclass(frozen=True)
class NestedVarianceComponents:
    seed_variance: float
    episode_variance: float
    within_variance: float
    seed_count: int
    episode_count: int
    sample_count: int


def estimate_nested_variance_components(
    residuals_by_seed_episode: Mapping[tuple[str, str], Sequence[float]],
) -> NestedVarianceComponents:
    """Apply the one-level moment estimator twice: seeds, then episodes.

    Step 1 groups all samples by seed (pooling across episodes within a
    seed) and fits ``estimate_variance_components`` to get ``seed_variance``.

    Step 2 subtracts each seed's own mean from its samples (seed-centred
    residuals) and re-fits ``estimate_variance_components`` grouped by
    episode, to get ``episode_variance`` and ``within_variance``.

    This is a moment-estimator approximation of a two-level nested random
    effects model, not a REML fit -- see the module docstring.
    """

    by_seed: dict[str, list[float]] = {}
    for (seed, _episode), values in residuals_by_seed_episode.items():
        by_seed.setdefault(seed, []).extend(float(v) for v in values)

    seed_components = estimate_variance_components(by_seed)
    seed_variance = seed_components.between_group_variance

    seed_means = {seed: float(np.mean(values)) for seed, values in by_seed.items()}

    by_episode_centred: dict[tuple[str, str], list[float]] = {}
    for (seed, episode), values in residuals_by_seed_episode.items():
        mean = seed_means[seed]
        by_episode_centred[(seed, episode)] = [float(v) - mean for v in values]

    episode_components = estimate_variance_components(
        {str(key): values for key, values in by_episode_centred.items()}
    )

    return NestedVarianceComponents(
        seed_variance=seed_variance,
        episode_variance=episode_components.between_group_variance,
        within_variance=episode_components.within_group_variance,
        seed_count=seed_components.group_count,
        episode_count=episode_components.group_count,
        sample_count=episode_components.sample_count,
    )


def posterior_floor_from_components(components: NestedVarianceComponents) -> float:
    """sqrt(tau_seed^2 + tau_episode^2 + tau_seed^2/n_seeds + tau_episode^2/n_episodes).

    The last two terms are the standard error of the fitted bias at each
    level. The seed component must be divided by the *seed* count, not the
    episode count: episodes inside one seed are not independent draws of
    the seed-level offset, so dividing by the episode count would
    understate the floor roughly 3x (see the nested-vs-flat test).
    """

    variance = (
        components.seed_variance
        + components.episode_variance
        + components.seed_variance / components.seed_count
        + components.episode_variance / components.episode_count
    )
    return variance**0.5
