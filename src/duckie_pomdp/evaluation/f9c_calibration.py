"""Gate F9c calibration-only fitting: bias, covariance scales, P_D^eff.

This module is pure statistics over calibration-only rows (seeds 6101-6108).
It never reads final-evaluation or frozen-test-split data, and every fitting
set below is selected by an EXPLICIT, pre-specified, ground-truth-based rule
-- never by the runtime gate/association decision -- because the gate's own
decision depends on the parameters being fitted here (invariant I6).

**Nested variance-component estimator.** ``fit_covariance_scales`` and the
experiment script that calls this module use
``belief.covariance_calibration.estimate_nested_variance_components``
(grouping residuals by ``(seed, episode)``), never the one-level
``estimate_variance_components`` directly -- see that module's docstring for
why the flat form understates the floor.

**Lambda fitting and the invariant-I1 provider.** ``fit_covariance_scales``
solves for ``(lambda_r, lambda_beta)`` by matching each axis's OWN marginal
NIS to the chi-square(1) median -- NOT by jointly targeting the chi-square(2)
median with a two-parameter search, which is underdetermined (one scalar
equation, two unknowns) and was shown, by this module's own TDD, to make the
fit depend on where a coordinate-descent search started. See
``fit_covariance_scales``'s docstring for the full argument; ``joint_nis_median``
reports the resulting *joint* 2-DOF statistic as a diagnostic, not a fitting
target.

The ``S = H P^- H^T + R`` each row's NIS needs is decomposed, once per
fitting-set row, into ``H P^- H^T`` and the base (pre-inflation) ``R`` using
the EXACT matrix ``RobustPedestrianBeliefUpdater._innovation_covariance``
produced at render time (lambda_r = lambda_beta = 1, since that is what the
as-checked-in config carries before this calibration runs) -- see
``lambda_fit_ingredients``. This is an EXACT decomposition, not an
approximation, because at lambda = 1, ``CovarianceCalibration.inflate`` is
the identity map on R.

Re-scoring a row's NIS at a *different* candidate bias (the bias this module
selects, which generally differs slightly from the identity bias used to
collect ``H P^- H^T``/base-R) is where an approximation is unavoidable
without a second full render pass: the innovation is shifted additively by
the new bias, holding ``H P^- H^T`` fixed (see ``_row_ingredients``). This
is a small-signal approximation -- correct to first order, and exact when
the fitted bias is zero -- disclosed here and in the calibration metrics
artifact, in the same spirit as ``covariance_calibration.py``'s own
disclosed moment-estimator approximation.

**Ground-truth fitting-set selection (invariant I6).** Both
``fit_covariance_scales`` and the lambda-ingredient capture in the
experiment script select rows by
``eligible_visible AND valid_projected_measurement AND correct_class AND
selected_correct_iou50`` -- all four are ground-truth-derived -- never by
``kinematic_measurement_accepted`` or any NIS threshold.

**Detection evidence, never localization acceptance (invariant I2).**
``fit_effective_detection`` conditions on ``detector_detected``, never on
``kinematic_measurement_accepted``.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import cos, sin
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10
    import tomli as tomllib

from duckie_pomdp.belief.bias_correction import FrozenBiasCorrection
from duckie_pomdp.belief.covariance_calibration import CovarianceCalibration
from duckie_pomdp.belief.innovation_gate import (
    InnovationGateConfig,
    normalized_innovation_squared,
)
from duckie_pomdp.belief.measurement_association import AssociationConfig
from duckie_pomdp.belief.observability import EffectiveDetectionModel, ObservabilityClass
from duckie_pomdp.belief.pedestrian_ekf import EKFStepDiagnostics
from duckie_pomdp.belief.robust_updater import (
    RobustObservationConfig,
    RobustObservationSwitches,
)
from duckie_pomdp.domain.detection import ObjectClass
from duckie_pomdp.domain.measurement import ObjectMeasurement
from duckie_pomdp.perception.measurement_calibration import wrap_angle
from duckie_pomdp.perception.measurement_noise import PolarMeasurementNoiseModel

#: median of the chi-square distribution with 2 degrees of freedom
#: (solves 1 - exp(-x/2) = 0.5, i.e. x = 2 ln 2).
CHI2_2DOF_MEDIAN = 1.3862943611198906

#: median of the chi-square distribution with 1 degree of freedom (solves
#: erf(sqrt(x/2)) = 0.5). Used to fit each axis's own MARGINAL scale -- see
#: fit_covariance_scales for why the joint 2-DOF median cannot be used
#: directly to identify two independent scales.
CHI2_1DOF_MEDIAN = 0.4549364231195548

#: decision-rule constants for select_bias_model, pre-specified -- do not
#: relax them (see module docstring and the task-9 brief).
MIN_SAMPLES_PER_BIN = 100
MIN_SCENARIOS_PER_BIN = 3
LOSO_IMPROVEMENT_THRESHOLD = 0.10

#: Beta prior for fit_effective_detection's posterior mean.
DETECTION_BETA_ALPHA = 0.5
DETECTION_BETA_BETA = 0.5

OBSERVABILITY_CLASS_NAMES: tuple[str, ...] = (
    "center",
    "mid_fov",
    "edge_fov",
    "outside_domain",
)

_RANGE_BINS = ("near", "medium", "far")


# ---------------------------------------------------------------------------
# Bias fitting.
# ---------------------------------------------------------------------------


def _is_matched_row(row: Mapping[str, Any]) -> bool:
    if not bool(row.get("selected_correct_iou50")):
        return False
    range_error = row.get("range_error_m")
    bearing_error = row.get("bearing_error_rad")
    return range_error not in (None, "") and bearing_error not in (None, "")


def fit_bias(rows: Sequence[Mapping[str, Any]], *, model: str) -> dict[str, Any]:
    """Fit a bias-correction model from ground-truth-matched calibration rows.

    ``model="global_additive"`` fits one range offset and one bearing offset
    over every matched row. ``model="per_range_bin"`` fits a separate range
    offset per distance bin (bearing stays global -- the plan's structural
    prediction is that bearing offset is episode-carried, not range-carried).
    """

    matched = [row for row in rows if _is_matched_row(row)]
    if len(matched) < 2:
        raise ValueError("fit_bias requires at least two matched rows")

    bearing_errors = np.asarray(
        [float(row["bearing_error_rad"]) for row in matched], dtype=float
    )
    bearing_bias_rad = float(np.mean(bearing_errors))

    if model == "global_additive":
        range_errors = np.asarray(
            [float(row["range_error_m"]) for row in matched], dtype=float
        )
        return {
            "model": "global_additive",
            "range_bias_m": float(np.mean(range_errors)),
            "bearing_bias_rad": bearing_bias_rad,
            "sample_count": len(matched),
        }

    if model == "per_range_bin":
        range_bin_bias_m: dict[str, float] = {}
        range_bin_sample_count: dict[str, int] = {}
        for name in _RANGE_BINS:
            selected = [row for row in matched if row.get("distance_bin") == name]
            range_bin_sample_count[name] = len(selected)
            range_bin_bias_m[name] = (
                float(
                    np.mean([float(row["range_error_m"]) for row in selected])
                )
                if selected
                else 0.0
            )
        return {
            "model": "per_range_bin",
            "range_bin_bias_m": range_bin_bias_m,
            "range_bin_sample_count": range_bin_sample_count,
            "bearing_bias_rad": bearing_bias_rad,
            "sample_count": len(matched),
        }

    raise ValueError(f"unknown bias model: {model!r}")


def range_bias_for_row(fit: Mapping[str, Any], row: Mapping[str, Any]) -> float:
    if fit["model"] == "global_additive":
        return float(fit["range_bias_m"])
    if fit["model"] == "per_range_bin":
        return float(fit["range_bin_bias_m"][row["distance_bin"]])
    raise ValueError(f"unknown bias model: {fit['model']!r}")


def leave_one_seed_out_range_rmse(
    rows: Sequence[Mapping[str, Any]], *, model: str
) -> float:
    """LOSO-CV range RMSE: hold out whole SEEDS, never frames.

    Frames within a seed share the seed's own offset, so a frame-level split
    would let the training fold see (most of) the same offset it is being
    scored against, reporting a falsely low error.
    """

    matched = [row for row in rows if _is_matched_row(row)]
    seeds = sorted({row["seed"] for row in matched})
    if len(seeds) < 2:
        raise ValueError("leave-one-seed-out requires at least two distinct seeds")

    pooled_residuals: list[float] = []
    for held_out_seed in seeds:
        train = [row for row in matched if row["seed"] != held_out_seed]
        test = [row for row in matched if row["seed"] == held_out_seed]
        if not train or not test:
            continue
        fit = fit_bias(train, model=model)
        for row in test:
            error = float(row["range_error_m"])
            bias = range_bias_for_row(fit, row)
            pooled_residuals.append(error - bias)

    if not pooled_residuals:
        raise ValueError("leave-one-seed-out produced no held-out residuals")
    residuals = np.asarray(pooled_residuals, dtype=float)
    return float(np.sqrt(np.mean(residuals**2)))


def _per_bin_support_sufficient(matched: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    per_bin: dict[str, dict[str, int]] = {}
    sufficient = True
    for name in _RANGE_BINS:
        selected = [row for row in matched if row.get("distance_bin") == name]
        scenario_count = len({row.get("scenario") for row in selected})
        per_bin[name] = {
            "sample_count": len(selected),
            "scenario_count": scenario_count,
        }
        if len(selected) < MIN_SAMPLES_PER_BIN or scenario_count < MIN_SCENARIOS_PER_BIN:
            sufficient = False
    return {"sufficient": sufficient, "by_bin": per_bin}


def select_bias_model(
    rows: Sequence[Mapping[str, Any]],
) -> tuple[str, dict[str, Any]]:
    """Pre-specified decision rule -- do not relax it.

    ``per_range_bin`` wins only if BOTH hold:
      1. every bin has >= 100 matched samples drawn from >= 3 distinct
         scenarios, AND
      2. leave-one-seed-out held-out range RMSE improves by >= 10% relative
         to global additive.
    Otherwise ``global_additive``.
    """

    matched = [row for row in rows if _is_matched_row(row)]
    support = _per_bin_support_sufficient(matched)
    loso_global = leave_one_seed_out_range_rmse(rows, model="global_additive")
    evidence: dict[str, Any] = {
        "loso_rmse_global_additive": loso_global,
        "per_bin_support": support,
    }

    chosen = "global_additive"
    if support["sufficient"]:
        loso_per_bin = leave_one_seed_out_range_rmse(rows, model="per_range_bin")
        improvement = (
            (loso_global - loso_per_bin) / loso_global if loso_global > 0.0 else 0.0
        )
        evidence["loso_rmse_per_range_bin"] = loso_per_bin
        evidence["per_range_bin_relative_improvement"] = improvement
        if improvement >= LOSO_IMPROVEMENT_THRESHOLD:
            chosen = "per_range_bin"

    fit = fit_bias(rows, model=chosen)
    fit["selection_evidence"] = evidence
    return chosen, fit


def frozen_bias_correction_from_fit(
    fit: Mapping[str, Any],
    *,
    near_max_m: float,
    medium_max_m: float,
) -> FrozenBiasCorrection:
    """Build the belief-layer ``FrozenBiasCorrection`` from a fit_bias/
    select_bias_model result dict."""

    model = fit["model"]
    if model == "global_additive":
        return FrozenBiasCorrection(
            model="global_additive",
            range_bias_m=float(fit["range_bias_m"]),
            bearing_bias_rad=float(fit["bearing_bias_rad"]),
            range_bin_bias_m=None,
            near_max_m=near_max_m,
            medium_max_m=medium_max_m,
        )
    if model == "per_range_bin":
        return FrozenBiasCorrection(
            model="per_range_bin",
            range_bias_m=0.0,
            bearing_bias_rad=float(fit["bearing_bias_rad"]),
            range_bin_bias_m={
                name: float(value)
                for name, value in fit["range_bin_bias_m"].items()
            },
            near_max_m=near_max_m,
            medium_max_m=medium_max_m,
        )
    raise ValueError(f"unknown bias model: {model!r}")


# ---------------------------------------------------------------------------
# Covariance-scale (lambda) fitting -- invariant I1 and I6.
# ---------------------------------------------------------------------------


def lambda_fit_ingredients(
    diagnostics: EKFStepDiagnostics | None,
    measurement_noise: PolarMeasurementNoiseModel,
    range_m: float,
) -> dict[str, Any] | None:
    """Decompose the invariant-I1 ``S`` (captured at lambda = 1) into
    ``H P^- H^T`` and the base (pre-inflation) ``R``, from a REAL
    ``EKFStepDiagnostics`` produced by ``PedestrianEKF.correct``. Exact,
    because ``CovarianceCalibration.inflate`` is the identity map at
    lambda = 1.

    Returns ``None`` when no genuine innovation/S was computed this frame
    (uninitialized track, or the special init-only correction branch).
    """

    if (
        diagnostics is None
        or diagnostics.innovation_covariance is None
        or diagnostics.nis is None
        or diagnostics.innovation_range_m is None
        or diagnostics.innovation_bearing_rad is None
    ):
        return None
    base_r = measurement_noise.covariance(range_m)
    s = np.array(diagnostics.innovation_covariance, dtype=float)
    return {
        "lambda_fit_eligible": True,
        "hph_rr": float(s[0, 0] - base_r[0, 0]),
        "hph_rb": float(s[0, 1]),
        "hph_bb": float(s[1, 1] - base_r[1, 1]),
        "base_range_variance_m2": float(base_r[0, 0]),
        "base_bearing_variance_rad2": float(base_r[1, 1]),
        "range_innovation_m": float(diagnostics.innovation_range_m),
        "bearing_innovation_rad": float(diagnostics.innovation_bearing_rad),
    }


def _is_lambda_fitting_row(row: Mapping[str, Any]) -> bool:
    return (
        bool(row.get("eligible_visible"))
        and bool(
            row.get(
                "valid_projected_measurement",
                row.get("raw_range_m") not in (None, ""),
            )
        )
        and bool(row.get("correct_class", True))
        and bool(row.get("selected_correct_iou50"))
        and bool(row.get("lambda_fit_eligible"))
    )


def _lambda_fitting_rows(
    rows: Sequence[Mapping[str, Any]]
) -> list[Mapping[str, Any]]:
    return [row for row in rows if _is_lambda_fitting_row(row)]


@dataclass(frozen=True)
class _RowIngredients:
    innovation_r: float
    innovation_b: float
    hph_rr: float
    hph_rb: float
    hph_bb: float
    base_r: float
    base_b: float


def _row_ingredients(
    row: Mapping[str, Any], bias: FrozenBiasCorrection
) -> _RowIngredients:
    """Bias-shift the render-time (bias = identity) innovation additively.

    See the module docstring: this is a documented small-signal
    approximation. ``H P^- H^T`` and the base R are unaffected by bias (they
    depend only on the EKF's own predicted state / measured range at render
    time), so only the innovation is adjusted here.
    """

    raw_range_m = float(row["raw_range_m"])
    raw_bearing_rad = float(row["raw_bearing_rad"])
    raw_measurement = ObjectMeasurement(
        object_class=ObjectClass.DUCKIE,
        detected=True,
        confidence=1.0,
        x_left_m=raw_range_m * sin(raw_bearing_rad),
        y_forward_m=raw_range_m * cos(raw_bearing_rad),
        range_m=raw_range_m,
        bearing_rad=raw_bearing_rad,
    )
    corrected = bias.correct(raw_measurement)
    range_shift = corrected.range_m - raw_range_m
    bearing_shift = wrap_angle(corrected.bearing_rad - raw_bearing_rad)

    return _RowIngredients(
        innovation_r=float(row["range_innovation_m"]) + range_shift,
        innovation_b=wrap_angle(float(row["bearing_innovation_rad"]) + bearing_shift),
        hph_rr=float(row["hph_rr"]),
        hph_rb=float(row["hph_rb"]),
        hph_bb=float(row["hph_bb"]),
        base_r=float(row["base_range_variance_m2"]),
        base_b=float(row["base_bearing_variance_rad2"]),
    )


def _nis_for_ingredient(
    ingredient: _RowIngredients, lambda_r: float, lambda_beta: float
) -> float:
    s = np.array(
        [
            [
                ingredient.hph_rr + lambda_r * ingredient.base_r,
                ingredient.hph_rb,
            ],
            [
                ingredient.hph_rb,
                ingredient.hph_bb + lambda_beta * ingredient.base_b,
            ],
        ],
        dtype=float,
    )
    innovation = np.array([ingredient.innovation_r, ingredient.innovation_b], dtype=float)
    return normalized_innovation_squared(innovation, s)


def _bisect_lambda(
    scalar_fn,
    *,
    target: float,
    lower: float = 1.0,
    tolerance: float = 1e-9,
    max_iterations: int = 200,
) -> float:
    """Find lambda >= lower with scalar_fn(lambda) == target.

    scalar_fn is (assumed) non-increasing in lambda: inflating R only grows
    the corresponding diagonal entry of S, which shrinks NIS. lower = 1.0
    matches CovarianceCalibration's own floor (range_scale/bearing_scale >=
    1). Deterministic: a single monotonic bisection over a fixed bracket has
    one root -- the result cannot depend on any starting guess.
    """

    f_lower = scalar_fn(lower)
    if f_lower <= target:
        return lower

    upper = lower
    f_upper = f_lower
    while f_upper > target:
        upper *= 2.0
        f_upper = scalar_fn(upper)
        if upper > 1.0e9:
            raise RuntimeError("lambda search exceeded its bracket")

    for _ in range(max_iterations):
        mid = 0.5 * (lower + upper)
        f_mid = scalar_fn(mid)
        if abs(f_mid - target) < tolerance:
            return mid
        if f_mid > target:
            lower = mid
        else:
            upper = mid
    return 0.5 * (lower + upper)


def fit_covariance_scales(
    rows: Sequence[Mapping[str, Any]],
    *,
    bias: FrozenBiasCorrection,
) -> tuple[float, float]:
    """Fit (lambda_r, lambda_beta), the R-inflation scales invariant I1's
    ``S = H P^- H^T + lambda R`` uses.

    **Why per-axis marginals, not a joint 2-parameter fit against the joint
    2-DOF median.** The brief's target is "the calibration NIS median
    matches the chi-square(2) median" -- but that is ONE scalar equation in
    TWO unknowns (lambda_r, lambda_beta), which is underdetermined: many
    (lambda_r, lambda_beta) pairs can drive the joint median to the same
    value, and a coordinate-descent search over that underdetermined system
    converges to a *different point on that solution curve* depending on
    where it starts -- i.e. it is NOT seed-independent, which contradicts
    invariant I6's requirement that the fit not depend on anything but the
    (ground-truth-selected) data. An earlier version of this function did
    exactly that coordinate descent; a seed-independence test written
    against it (this module's own TDD) caught the fixed-point-depends-on-
    the-seed bug directly.

    Instead, each axis is solved independently by matching its OWN marginal
    NIS -- ``innovation_r^2 / (hph_rr + lambda_r * base_range_variance)``
    and the bearing analogue -- to the chi-square(1) median
    (``CHI2_1DOF_MEDIAN``), ignoring the H P^- H^T cross term in the
    marginal denominator. Each is a single monotonic bisection: well-posed,
    exactly seed-independent (there is no starting-guess-dependent search),
    and exact whenever the cross term is zero (as in a diagonal S). The
    resulting JOINT 2-DOF NIS median (using the FULL S, cross term
    included) is reported by ``build_calibration_metrics`` as a diagnostic,
    not as the fitting target.

    The fitting set is selected by ground truth (invariant I6):
    ``eligible_visible AND valid_projected_measurement AND correct_class AND
    selected_correct_iou50``.
    """

    fitting_rows = _lambda_fitting_rows(rows)
    if len(fitting_rows) < 8:
        raise ValueError(
            "fit_covariance_scales requires at least 8 ground-truth-selected "
            f"fitting rows with lambda_fit_eligible; got {len(fitting_rows)}"
        )

    ingredients = [_row_ingredients(row, bias) for row in fitting_rows]

    def median_range_nis(lambda_r: float) -> float:
        values = np.array(
            [
                ing.innovation_r**2 / (ing.hph_rr + lambda_r * ing.base_r)
                for ing in ingredients
            ],
            dtype=float,
        )
        return float(np.median(values))

    def median_bearing_nis(lambda_beta: float) -> float:
        values = np.array(
            [
                ing.innovation_b**2 / (ing.hph_bb + lambda_beta * ing.base_b)
                for ing in ingredients
            ],
            dtype=float,
        )
        return float(np.median(values))

    lambda_r = _bisect_lambda(median_range_nis, target=CHI2_1DOF_MEDIAN)
    lambda_beta = _bisect_lambda(median_bearing_nis, target=CHI2_1DOF_MEDIAN)
    return lambda_r, lambda_beta


def joint_nis_median(
    rows: Sequence[Mapping[str, Any]],
    *,
    bias: FrozenBiasCorrection,
    lambda_r: float,
    lambda_beta: float,
) -> float:
    """Diagnostic: the JOINT 2-DOF NIS median at the fitted (lambda_r,
    lambda_beta), using the FULL S (including the H P^- H^T cross term).
    Expected to land near CHI2_2DOF_MEDIAN when cross-correlation is modest
    -- reported, not targeted (see fit_covariance_scales)."""

    fitting_rows = _lambda_fitting_rows(rows)
    ingredients = [_row_ingredients(row, bias) for row in fitting_rows]
    values = np.array(
        [_nis_for_ingredient(ing, lambda_r, lambda_beta) for ing in ingredients],
        dtype=float,
    )
    return float(np.median(values))


def lambda_excluded_sample_report(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Report the count and predicted_nis distribution of eligible/correct
    frames EXCLUDED from the lambda fitting set (e.g. no track yet), so the
    fit's blind spot is visible rather than implied."""

    eligible_correct = [
        row
        for row in rows
        if bool(row.get("eligible_visible"))
        and bool(
            row.get(
                "valid_projected_measurement",
                row.get("raw_range_m") not in (None, ""),
            )
        )
        and bool(row.get("correct_class", True))
        and bool(row.get("selected_correct_iou50"))
    ]
    excluded = [row for row in eligible_correct if not _is_lambda_fitting_row(row)]
    nis_values = [
        float(row["predicted_nis"])
        for row in excluded
        if row.get("predicted_nis") not in (None, "")
    ]
    return {
        "eligible_correct_count": len(eligible_correct),
        "included_in_lambda_fit_count": len(eligible_correct) - len(excluded),
        "excluded_from_lambda_fit_count": len(excluded),
        "excluded_predicted_nis_sample_count": len(nis_values),
        "excluded_predicted_nis_median": (
            float(np.median(nis_values)) if nis_values else None
        ),
        "excluded_predicted_nis_p95": (
            float(np.quantile(nis_values, 0.95)) if nis_values else None
        ),
    }


def predicted_nis_diagnostic(
    rows: Sequence[Mapping[str, Any]], *, gate_threshold: float
) -> dict[str, Any]:
    """Distribution of the runtime's OWN as-checked-in-config
    ``predicted_nis`` (lambda=1, bias 0/0) across every row that has one,
    plus the fraction exceeding the innovation-gate threshold. Reported
    verbatim -- the median not matching the chi-square(2) target does not
    mean the gate's tail behaviour is unusable; the two are checked
    separately here on purpose."""

    values = [
        float(row["predicted_nis"])
        for row in rows
        if row.get("predicted_nis") not in (None, "")
    ]
    if not values:
        return {
            "sample_count": 0,
            "median": None,
            "p95": None,
            "fraction_above_gate_threshold": None,
            "gate_threshold": gate_threshold,
        }
    array = np.asarray(values, dtype=float)
    return {
        "sample_count": len(array),
        "median": float(np.median(array)),
        "p95": float(np.quantile(array, 0.95)),
        "fraction_above_gate_threshold": float(np.mean(array > gate_threshold)),
        "gate_threshold": gate_threshold,
    }


# ---------------------------------------------------------------------------
# Effective detection probability -- invariant I2.
# ---------------------------------------------------------------------------


def fit_effective_detection(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Beta(0.5, 0.5) posterior mean of
    ``P(detector_detected | pedestrian exists, predicted_observability_class = c)``
    for each of the four observability classes.

    Conditions on ``detector_detected`` (invariant I2), never on
    ``kinematic_measurement_accepted``. Existence comes from offline ground
    truth (``gt_pedestrian_exists``, defaulting to True when absent, since
    every calibration scenario places exactly one pedestrian in the world).
    """

    trials = {name: 0 for name in OBSERVABILITY_CLASS_NAMES}
    successes = {name: 0 for name in OBSERVABILITY_CLASS_NAMES}
    for row in rows:
        if not bool(row.get("gt_pedestrian_exists", True)):
            continue
        observability = row.get("predicted_observability_class")
        name = (
            observability.value
            if isinstance(observability, ObservabilityClass)
            else str(observability)
        )
        if name not in trials:
            continue
        trials[name] += 1
        if bool(row.get("detector_detected")):
            successes[name] += 1

    result: dict[str, Any] = {}
    for name in OBSERVABILITY_CLASS_NAMES:
        n = trials[name]
        k = successes[name]
        mean = (k + DETECTION_BETA_ALPHA) / (n + DETECTION_BETA_ALPHA + DETECTION_BETA_BETA)
        result[f"detection_probability_{name}"] = float(mean)
        result[f"trials_{name}"] = n
        result[f"successes_{name}"] = k
    result["outside_domain_trial_count_below_30"] = trials["outside_domain"] < 30
    return result


# ---------------------------------------------------------------------------
# Invariant I8 (Task 9 fix round 1): the miss-likelihood floor.
#
# ``ExistenceFilterConfig.miss_likelihood_floor`` bounds a single miss's
# likelihood ratio at ``LR_floor = LR_nominal ** (1 / L_mean)``, where
# ``L_mean`` is the MEASURED mean length of consecutive genuine-miss runs
# and ``LR_nominal`` is the GLOBAL (not per-class) fitted miss likelihood
# ratio -- never hand-picked, never tuned to hit a retention target. See
# ``belief.existence_filter``'s module docstring for the full rationale.
# ---------------------------------------------------------------------------


def mean_genuine_miss_run_length(
    rows: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    """Mean length of consecutive GENUINE-MISS runs: frames where
    ``eligible_visible`` (GT) is True but ``detector_detected`` is False,
    grouped by episode and ordered by frame. A frame the GT calls
    ineligible/invisible is not a genuine miss and splits a run rather than
    extending it -- it is simply not evidence of the detector failing to
    see something that was there to see.
    """

    by_episode: dict[str, list[tuple[int, bool]]] = {}
    for row in rows:
        episode = str(row["episode"])
        frame = int(row["frame"])
        is_genuine_miss = bool(row.get("eligible_visible")) and not bool(
            row.get("detector_detected")
        )
        by_episode.setdefault(episode, []).append((frame, is_genuine_miss))

    run_lengths: list[int] = []
    for entries in by_episode.values():
        entries.sort(key=lambda item: item[0])
        current = 0
        for _, is_miss in entries:
            if is_miss:
                current += 1
            else:
                if current:
                    run_lengths.append(current)
                current = 0
        if current:
            run_lengths.append(current)

    if not run_lengths:
        return {
            "run_count": 0,
            "mean_run_length": None,
            "median_run_length": None,
            "max_run_length": None,
            "run_lengths": [],
        }
    return {
        "run_count": len(run_lengths),
        "mean_run_length": float(np.mean(run_lengths)),
        "median_run_length": float(np.median(run_lengths)),
        "max_run_length": int(max(run_lengths)),
        "run_lengths": run_lengths,
    }


def fit_global_effective_detection(
    rows: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    """Beta(0.5, 0.5) posterior mean of ``P(detector_detected |
    eligible_visible)``, pooled across every observability class -- the
    single GLOBAL detection probability ``LR_nominal`` needs, as distinct
    from ``fit_effective_detection``'s per-class values."""

    eligible = [row for row in rows if bool(row.get("eligible_visible"))]
    trials = len(eligible)
    successes = sum(1 for row in eligible if bool(row.get("detector_detected")))
    mean = (successes + DETECTION_BETA_ALPHA) / (
        trials + DETECTION_BETA_ALPHA + DETECTION_BETA_BETA
    )
    return {
        "detection_probability_global": float(mean),
        "trials_global": trials,
        "successes_global": successes,
    }


def fit_false_alarm_probability(
    rows: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    """Beta(0.5, 0.5) posterior mean of ``P(detector_detected | NOT
    eligible_visible)`` over naturally-ineligible frames in the calibration
    data (this run did not collect F9a-style explicit counterfactual
    hidden-pedestrian negatives)."""

    negative = [row for row in rows if not bool(row.get("eligible_visible"))]
    trials = len(negative)
    successes = sum(1 for row in negative if bool(row.get("detector_detected")))
    mean = (successes + DETECTION_BETA_ALPHA) / (
        trials + DETECTION_BETA_ALPHA + DETECTION_BETA_BETA
    )
    return {
        "false_positive_probability": float(mean),
        "trials": trials,
        "successes": successes,
    }


def fit_miss_likelihood_floor(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Fit invariant I8's ``LR_floor = LR_nominal ** (1 / L_mean)`` from
    calibration data. Raises if there is no measured miss run (L_mean
    undefined) rather than silently falling back to a default."""

    detection = fit_global_effective_detection(rows)
    false_alarm = fit_false_alarm_probability(rows)
    run_report = mean_genuine_miss_run_length(rows)

    p_d = detection["detection_probability_global"]
    p_fa = false_alarm["false_positive_probability"]
    lr_nominal = (1.0 - p_d) / (1.0 - p_fa)

    l_mean = run_report["mean_run_length"]
    if l_mean is None or l_mean <= 0.0:
        raise ValueError(
            "cannot fit a miss-likelihood floor: no genuine-miss run was "
            "observed in the calibration data"
        )
    lr_floor = lr_nominal ** (1.0 / l_mean)

    return {
        "detection_probability_global": p_d,
        "false_positive_probability": p_fa,
        "lr_nominal": lr_nominal,
        "l_mean": l_mean,
        "run_length_report": run_report,
        "lr_floor": lr_floor,
        "detection_trial_report": detection,
        "false_alarm_trial_report": false_alarm,
    }


def adjust_miss_likelihood_floor_for_false_positive_rate(
    fit: Mapping[str, Any], *, false_positive_probability: float
) -> dict[str, Any]:
    """Recompute ``LR_nominal``/``LR_floor`` from a ``fit_miss_likelihood_floor``
    result, substituting a different ``false_positive_probability``.

    ``fit_false_alarm_probability``'s own trial set ("naturally ineligible"
    frames, i.e. frames the GT silhouette rule marked not visible) is
    contaminated by borderline-visibility frames the detector can still
    correctly fire on -- this run did not collect F9a-style counterfactual
    hidden-pedestrian renders, so "not eligible_visible" is not the same
    population as "pedestrian genuinely absent from the image." That
    inflates the self-fit false-positive rate well above what a genuine
    false-alarm rate should be. This function keeps the self-fit numbers
    visible for comparison (never silently discarded) while letting the
    caller substitute a more carefully estimated rate -- e.g. F9b's own
    counterfactual-based frozen constant -- for the value actually used.
    """

    p_d = float(fit["detection_probability_global"])
    l_mean = float(fit["l_mean"])
    lr_nominal = (1.0 - p_d) / (1.0 - false_positive_probability)
    lr_floor = lr_nominal ** (1.0 / l_mean)
    return {
        "false_positive_probability_used": false_positive_probability,
        "detection_probability_global": p_d,
        "lr_nominal": lr_nominal,
        "l_mean": l_mean,
        "lr_floor": lr_floor,
    }


def detection_by_range_and_fov(
    rows: Sequence[Mapping[str, Any]]
) -> dict[str, dict[str, dict[str, Any]]]:
    """``P(detector_detected | eligible_visible)`` cross-tabulated by GT
    ``distance_bin`` x GT ``fov_region`` (both ground truth, not the
    runtime prediction) -- the diagnostic that showed the
    EDGE_FOV > CENTER effective-detection inversion is a range confound,
    not a genuine FOV effect: within any one distance bin, detection rate
    is close to flat across FOV position."""

    buckets: dict[tuple[str, str], list[bool]] = {}
    for row in rows:
        if not bool(row.get("eligible_visible")):
            continue
        distance_bin = str(row.get("distance_bin"))
        fov_region = str(row.get("fov_region"))
        buckets.setdefault((distance_bin, fov_region), []).append(
            bool(row.get("detector_detected"))
        )

    result: dict[str, dict[str, dict[str, Any]]] = {}
    for (distance_bin, fov_region), detections in buckets.items():
        n = len(detections)
        k = sum(detections)
        mean = (k + DETECTION_BETA_ALPHA) / (n + DETECTION_BETA_ALPHA + DETECTION_BETA_BETA)
        result.setdefault(fov_region, {})[distance_bin] = {
            "detection_probability": float(mean),
            "trials": n,
            "successes": k,
        }
    return result


# ---------------------------------------------------------------------------
# Reporting helpers (near-range/edge-fov counts, observability confusion).
# ---------------------------------------------------------------------------


def eligible_distance_bin_counts(rows: Iterable[Mapping[str, Any]]) -> dict[str, int]:
    counts = {name: 0 for name in _RANGE_BINS}
    for row in rows:
        if bool(row.get("eligible_visible")):
            name = row.get("distance_bin")
            if name in counts:
                counts[name] += 1
    return counts


def eligible_fov_region_counts(rows: Iterable[Mapping[str, Any]]) -> dict[str, int]:
    counts = {"center": 0, "mid_fov": 0, "edge_fov": 0, "outside": 0}
    for row in rows:
        if bool(row.get("eligible_visible")):
            name = row.get("fov_region")
            if name in counts:
                counts[name] += 1
    return counts


def observability_confusion_matrix(
    rows: Iterable[Mapping[str, Any]]
) -> dict[str, dict[str, int]]:
    """predicted_observability_class (runtime) x fov_region (GT) counts."""

    matrix: dict[str, dict[str, int]] = {}
    for row in rows:
        observability = row.get("predicted_observability_class")
        predicted = (
            observability.value
            if isinstance(observability, ObservabilityClass)
            else str(observability)
        )
        fov = str(row.get("fov_region"))
        matrix.setdefault(predicted, {}).setdefault(fov, 0)
        matrix[predicted][fov] += 1
    return matrix


# ---------------------------------------------------------------------------
# TOML loaders -- Task 9 is the first task that needs to build a coordinator
# from configs/f9c_robust_belief_v1.toml. No loader for
# ``RobustObservationConfig``/``FrozenBiasCorrection`` existed before this.
# ---------------------------------------------------------------------------


def load_robust_observation_config(path: str | Path) -> RobustObservationConfig:
    config_path = Path(path).resolve()
    with config_path.open("rb") as stream:
        data: dict[str, Any] = tomllib.load(stream)

    robust_observation = data["robust_observation"]
    switches = RobustObservationSwitches(
        bias_refit=bool(robust_observation["bias_refit"]),
        innovation_gate=bool(robust_observation["innovation_gate"]),
        temporal_association=bool(robust_observation["temporal_association"]),
        covariance_calibration=bool(robust_observation["covariance_calibration"]),
        conditional_detection=bool(robust_observation["conditional_detection"]),
    )
    gate = InnovationGateConfig(
        chi_square_threshold=float(data["innovation_gate"]["chi_square_threshold"])
    )
    association = AssociationConfig(
        chi_square_gate=float(data["association"]["chi_square_gate"]),
        initialization_rule=str(data["association"]["initialization_rule"]),
    )
    cc = data["covariance_calibration"]
    covariance = CovarianceCalibration(
        float(cc["range_scale"]),
        float(cc["bearing_scale"]),
        float(cc["range_posterior_floor_m"]),
        float(cc["bearing_posterior_floor_rad"]),
    )
    cd = data["conditional_detection"]
    effective_detection = EffectiveDetectionModel(
        {
            ObservabilityClass.CENTER: float(cd["detection_probability_center"]),
            ObservabilityClass.MID_FOV: float(cd["detection_probability_mid_fov"]),
            ObservabilityClass.EDGE_FOV: float(cd["detection_probability_edge_fov"]),
            ObservabilityClass.OUTSIDE_DOMAIN: float(
                cd["detection_probability_outside_domain"]
            ),
        },
        outside_domain_miss_policy=str(cd["outside_domain_miss_policy"]),
    )
    existence_track = data["existence_track"]
    return RobustObservationConfig(
        switches=switches,
        gate=gate,
        association=association,
        covariance=covariance,
        effective_detection=effective_detection,
        active_threshold=float(existence_track["active_threshold"]),
        delete_threshold=float(existence_track["delete_threshold"]),
        initialization_threshold=float(existence_track["initialization_threshold"]),
    )


def load_frozen_bias_correction(
    path: str | Path, *, section: str
) -> FrozenBiasCorrection:
    config_path = Path(path).resolve()
    with config_path.open("rb") as stream:
        data: dict[str, Any] = tomllib.load(stream)
    block = data[section]
    calibration_protocol = data["calibration_protocol"]
    range_bin_bias_m = block.get("range_bin_bias_m")
    return FrozenBiasCorrection(
        model=str(block["bias_model"]),
        range_bias_m=float(block.get("range_bias_m", 0.0)),
        bearing_bias_rad=float(block.get("bearing_bias_rad", 0.0)),
        range_bin_bias_m=(
            {name: float(value) for name, value in range_bin_bias_m.items()}
            if range_bin_bias_m
            else None
        ),
        near_max_m=float(calibration_protocol["distance_bin_near_max_m"]),
        medium_max_m=float(calibration_protocol["distance_bin_medium_max_m"]),
    )


def load_miss_likelihood_floor(path: str | Path) -> float:
    """Invariant I8. Reads ``[conditional_detection].miss_likelihood_floor``
    -- 0.0 (a strict no-op) before Task 10's freeze, the fitted LR_floor from
    the final calibration re-run afterward."""

    config_path = Path(path).resolve()
    with config_path.open("rb") as stream:
        data: dict[str, Any] = tomllib.load(stream)
    return float(data["conditional_detection"].get("miss_likelihood_floor", 0.0))
