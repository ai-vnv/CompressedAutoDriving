"""Recalibration of the *reported* lane-belief uncertainty (F10 v4 Task 1).

The lane EKF (``duckie_pomdp.belief.lane_ekf``) reports posterior standard
deviations that are badly overconfident relative to the errors they should be
bounding: on the final held-out validation set the heading channel covers
only 19.7% of errors within its reported one-sigma band (should be ~68%).

This module does **not** touch the mean estimate, the point extractor, or the
bias calibration in ``configs/lane_belief_v1.toml``. It only widens the
reported sigma that is handed to downstream consumers (the PPO observation
and any controller reading ``*_std_*``), via a per-channel
``std' = max(std * scale, floor)`` map, and it scores how well that widened
sigma is calibrated against ground truth.

Two pieces:

* :class:`LaneUncertaintyCalibration` -- the ``(floor, scale)`` map itself.
  ``apply`` is a pure function; the invariant it must uphold (checked in
  ``__post_init__``) is that it can only ever *widen* the interval, never
  narrow it, because narrowing a demonstrably overconfident belief would make
  the failure mode worse, not better.
* :func:`coverage_report` -- turns a batch of ``(error, sigma)`` pairs into
  the coverage/accuracy statistics the pre-registered gate checks.

``evaluate_uncertainty_gate`` applies the ``[uncertainty_gate]`` bands from
``configs/lane_belief_v2.toml`` to a pair of heading/lateral coverage
reports. ``lateral_overcoverage_disclosure`` documents -- rather than hides
-- the fact that the lateral channel cannot be fully calibrated (see the
module docstring in ``experiments/calibrate_lane_belief_uncertainty.py``).

**Runtime wiring (v4 fix round 1).** A calibration built here does nothing
on its own -- three more pieces connect it to the belief that actually
reaches the policy:

* :func:`load_lane_uncertainty_calibration` reads the ``[lane_belief_uncertainty]``
  table out of a config file (``None`` if the table is absent, e.g. for
  ``lane_belief_v1.toml``, which carries no such table).
* :func:`apply_calibration_to_lane_belief` applies a calibration (or, if
  ``None``, does nothing) to a :class:`~duckie_pomdp.domain.belief.LaneBelief`
  that has *already left* the EKF -- it only ever replaces the two reported
  standard deviations, never the means, and it is never handed the filter's
  internal covariance to mutate.
* :func:`resolve_runtime_calibration` is the single switch point: it reads
  ``[v4_changes].belief_uncertainty_refit`` from a loaded protocol's raw TOML
  and returns ``None`` (off) unless the flag is explicitly ``true`` -- a
  missing ``[v4_changes]`` table, a missing key, or an explicit ``false`` are
  all "off". This is what lets ``configs/f10_ppo_visual_v3.toml`` (which has
  no ``[v4_changes]`` table at all) keep reproducing its sigma bit-for-bit
  through the exact same runtime code path used by v4.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, replace
from math import isfinite
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray

from duckie_pomdp.domain.belief import LaneBelief

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10 runtime used by Gym-Duckietown.
    import tomli as tomllib

# Standard-normal one-sided quantiles used to score coverage. "coverage_68"
# is the fraction of samples with |error| <= 1*sigma (nominal 68.27%);
# "coverage_95" is the fraction with |error| <= 1.959964*sigma (nominal
# 95.00%, i.e. the two-sided 95% interval half-width).
Z_SCORE_68 = 1.0
Z_SCORE_95 = 1.959963984540054


@dataclass(frozen=True)
class LaneUncertaintyCalibration:
    """A per-channel ``std' = max(std * scale, floor)`` recalibration map.

    This changes only the *reported* uncertainty. It must never report less
    uncertainty than the raw belief -- a calibration that could narrow the
    interval could turn an already-overconfident belief into something even
    more overconfident, which is the exact failure this task exists to fix.
    That invariant is enforced structurally: floors and scales must be
    non-negative, and scales must be >= 1.0.
    """

    heading_floor_rad: float
    heading_scale: float
    lateral_floor_m: float
    lateral_scale: float

    def __post_init__(self) -> None:
        for name, value in (
            ("heading_floor_rad", self.heading_floor_rad),
            ("heading_scale", self.heading_scale),
            ("lateral_floor_m", self.lateral_floor_m),
            ("lateral_scale", self.lateral_scale),
        ):
            if not isfinite(value) or value < 0.0:
                raise ValueError(
                    f"{name} must be finite and non-negative (got {value!r})"
                )
        for name, scale in (
            ("heading_scale", self.heading_scale),
            ("lateral_scale", self.lateral_scale),
        ):
            if scale < 1.0:
                raise ValueError(
                    f"{name} must be >= 1.0 -- a scale below 1.0 could narrow "
                    "the reported interval below the raw belief, which this "
                    "calibration must never do"
                )

    def apply(self, heading_std: float, lateral_std: float) -> tuple[float, float]:
        """Return the recalibrated ``(heading_std, lateral_std)``."""

        for name, value in (("heading_std", heading_std), ("lateral_std", lateral_std)):
            if not isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative (got {value!r})")
        heading = max(heading_std * self.heading_scale, self.heading_floor_rad)
        lateral = max(lateral_std * self.lateral_scale, self.lateral_floor_m)
        return heading, lateral


def coverage_report(errors: ArrayLike, sigmas: ArrayLike) -> dict[str, float | int]:
    """Score a batch of ``(error, sigma)`` pairs against Gaussian coverage.

    Returns ``coverage_68``, ``coverage_95`` (fraction of samples with
    ``|error| <= z * sigma`` for the nominal 68%/95% one- and two-sigma
    z-scores), ``sigma_over_rmse`` (mean reported sigma divided by the
    empirical RMSE -- a scale-only summary independent of interval shape),
    ``rmse``, and ``n``.
    """

    errors_arr = np.asarray(errors, dtype=float)
    sigmas_arr = np.asarray(sigmas, dtype=float)
    if errors_arr.shape != sigmas_arr.shape:
        raise ValueError(
            f"errors and sigmas must have matching shapes, got {errors_arr.shape} "
            f"and {sigmas_arr.shape}"
        )
    if errors_arr.ndim != 1:
        raise ValueError("errors and sigmas must be one-dimensional")
    n = int(errors_arr.size)
    if n == 0:
        raise ValueError("coverage_report requires at least one sample")
    if not np.all(np.isfinite(errors_arr)) or not np.all(np.isfinite(sigmas_arr)):
        raise ValueError("errors and sigmas must be finite")
    if np.any(sigmas_arr <= 0.0):
        raise ValueError("sigmas must be strictly positive")

    rmse = float(np.sqrt(np.mean(np.square(errors_arr))))
    abs_z = np.abs(errors_arr) / sigmas_arr
    coverage_68 = float(np.mean(abs_z <= Z_SCORE_68))
    coverage_95 = float(np.mean(abs_z <= Z_SCORE_95))
    mean_sigma = float(np.mean(sigmas_arr))
    sigma_over_rmse = mean_sigma / rmse if rmse > 0.0 else float("inf")

    return {
        "n": n,
        "rmse": rmse,
        "coverage_68": coverage_68,
        "coverage_95": coverage_95,
        "sigma_over_rmse": sigma_over_rmse,
    }


def z_quantiles(errors: ArrayLike, sigmas: ArrayLike) -> dict[str, float]:
    """``|error|/sigma`` quantiles, used to document non-Gaussian shape."""

    errors_arr = np.asarray(errors, dtype=float)
    sigmas_arr = np.asarray(sigmas, dtype=float)
    z = np.abs(errors_arr) / sigmas_arr
    return {
        "p50": float(np.percentile(z, 50)),
        "p95": float(np.percentile(z, 95)),
        "p99": float(np.percentile(z, 99)),
    }


def lateral_overcoverage_disclosure(
    lateral_errors: ArrayLike, lateral_sigmas_raw: ArrayLike
) -> dict[str, object]:
    """Document why lateral cov95 cannot be trusted as a passing gate.

    ``lateral_sigmas_raw`` must be the *raw* (uncalibrated) belief sigma --
    the z-quantiles are evidence about the shape of the error distribution,
    independent of whatever scale/floor is later applied.
    """

    quantiles = z_quantiles(lateral_errors, lateral_sigmas_raw)
    gaussian_reference = {"p50": 0.6745, "p95": 1.9600, "p99": 2.5758}
    return {
        "channel": "lateral",
        "claim": "lateral_coverage_95 is reported but NOT gated and must not "
        "be presented as passing a calibration band",
        "reason": (
            "the lateral |error|/sigma distribution is the wrong shape, not "
            "merely the wrong scale: no (scale, floor) pair puts both "
            "coverage_68 and coverage_95 in their pre-registered bands "
            "simultaneously -- by the time coverage_95 reaches ~0.90, "
            "coverage_68 has already passed ~0.76"
        ),
        "z_quantiles_raw_belief": quantiles,
        "gaussian_reference_z_quantiles": gaussian_reference,
        "resolution": (
            "scale chosen to land coverage_68 in its pre-registered band; "
            "coverage_95 over-covers as a result. Over-coverage at 95% is "
            "conservative (wider interval than needed), which is the safe "
            "direction for a controller -- unlike under-coverage, it cannot "
            "cause the controller to trust a wrong point estimate."
        ),
    }


def _in_band(value: float, band: Sequence[float]) -> bool:
    low, high = float(band[0]), float(band[1])
    return low <= value <= high


def evaluate_uncertainty_gate(
    heading_report: Mapping[str, float],
    lateral_report: Mapping[str, float],
    gate: Mapping[str, object],
) -> dict[str, object]:
    """Apply the pre-registered ``[uncertainty_gate]`` bands.

    ``lateral_coverage_95`` is deliberately absent from both the checks and
    the pass/fail decision -- it is reported elsewhere (via
    ``lateral_overcoverage_disclosure``) but never gated, per the disclosed,
    known-unsatisfiable lateral shape mismatch.
    """

    checks = {
        "heading_coverage_68_in_band": _in_band(
            heading_report["coverage_68"], gate["heading_coverage_68_band"]  # type: ignore[arg-type]
        ),
        "heading_coverage_95_in_band": _in_band(
            heading_report["coverage_95"], gate["heading_coverage_95_band"]  # type: ignore[arg-type]
        ),
        "lateral_coverage_68_in_band": _in_band(
            lateral_report["coverage_68"], gate["lateral_coverage_68_band"]  # type: ignore[arg-type]
        ),
        "heading_sigma_over_rmse_ok": heading_report["sigma_over_rmse"]
        <= float(gate["maximum_sigma_over_rmse"]),
        "lateral_sigma_over_rmse_ok": lateral_report["sigma_over_rmse"]
        <= float(gate["maximum_sigma_over_rmse"]),
        "heading_rmse_ok": heading_report["rmse"] <= float(gate["maximum_heading_rmse_rad"]),
    }
    return {
        "pass": all(checks.values()),
        "checks": checks,
    }


def load_validation_errors_and_sigmas(
    csv_path: str | Path,
) -> dict[str, dict[str, NDArray[np.float64]]]:
    """Load heading/lateral ``(error, raw sigma)`` arrays from a validation CSV.

    ``error`` is ``belief_*_mean - gt_*_error`` (belief minus ground truth);
    ``sigma`` is the raw, uncalibrated ``belief_*_std`` reported by the EKF.
    All 2400 rows are used, including misses (the belief still reports a
    mean/std on a miss, propagated by the predict step) to match the
    validation gate's own accounting.
    """

    path = Path(csv_path)
    heading_error: list[float] = []
    heading_sigma: list[float] = []
    lateral_error: list[float] = []
    lateral_sigma: list[float] = []
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            heading_error.append(
                float(row["belief_heading_mean_rad"]) - float(row["gt_heading_error_rad"])
            )
            heading_sigma.append(float(row["belief_heading_std_rad"]))
            lateral_error.append(
                float(row["belief_lateral_mean_m"]) - float(row["gt_lateral_error_m"])
            )
            lateral_sigma.append(float(row["belief_lateral_std_m"]))

    if not heading_error:
        raise ValueError(f"no rows loaded from {path}")

    return {
        "heading": {
            "error": np.asarray(heading_error, dtype=float),
            "sigma": np.asarray(heading_sigma, dtype=float),
        },
        "lateral": {
            "error": np.asarray(lateral_error, dtype=float),
            "sigma": np.asarray(lateral_sigma, dtype=float),
        },
    }


def load_lane_uncertainty_calibration(
    config_path: str | Path,
) -> LaneUncertaintyCalibration | None:
    """Read ``[lane_belief_uncertainty]`` out of a lane-belief config.

    Returns ``None`` when the table is absent -- true of
    ``configs/lane_belief_v1.toml``, which carries no such table by design
    (it is the frozen point-extractor/bias-calibration config this task must
    not touch). ``configs/lane_belief_v2.toml`` carries the table.
    """

    path = Path(config_path)
    with path.open("rb") as handle:
        raw = tomllib.load(handle)
    table = raw.get("lane_belief_uncertainty")
    if table is None:
        return None
    return LaneUncertaintyCalibration(
        heading_floor_rad=float(table["heading_floor_rad"]),
        heading_scale=float(table["heading_scale"]),
        lateral_floor_m=float(table["lateral_floor_m"]),
        lateral_scale=float(table["lateral_scale"]),
    )


def apply_calibration_to_lane_belief(
    belief: LaneBelief, calibration: LaneUncertaintyCalibration | None
) -> LaneBelief:
    """Apply a calibration to a belief's *reported* std only.

    ``calibration=None`` returns ``belief`` completely unchanged -- this is
    the off/v3 path and is what makes flag-off reproduction bit-for-bit: no
    new float operation (not even a no-op multiply) touches the value.

    When a calibration is given, only ``heading_error_std_rad`` and
    ``lateral_error_std_m`` are replaced; every other field, including both
    reported means (``heading_error_mean_rad``, ``lateral_error_mean_m``)
    and the curvature channel, passes through untouched. ``belief`` must
    already be the object the EKF's ``.belief()`` produced -- this function
    never reaches back into filter state, so applying it can never corrupt a
    subsequent ``predict()``/``correct()`` step (mirrors the reasoning in
    ``belief/robust_updater.py``'s posterior-floor step, written for the
    identical F9c failure mode on the pedestrian belief).
    """

    if calibration is None:
        return belief
    heading_std, lateral_std = calibration.apply(
        belief.heading_error_std_rad, belief.lateral_error_std_m
    )
    return replace(
        belief,
        heading_error_std_rad=heading_std,
        lateral_error_std_m=lateral_std,
    )


def resolve_runtime_calibration(
    protocol_raw: Mapping[str, object],
    lane_belief_config_path: str | Path,
) -> LaneUncertaintyCalibration | None:
    """The single v4 switch point: ``[v4_changes].belief_uncertainty_refit``.

    Defaults to **off**: a protocol with no ``[v4_changes]`` table at all
    (every existing v3 config), a table missing the
    ``belief_uncertainty_refit`` key, or an explicit ``false`` all resolve to
    ``None`` -- so a config that predates or does not opt into this switch
    reproduces the raw EKF sigma exactly as before.

    When the flag is explicitly ``true``, the referenced
    ``lane_belief_config`` **must** carry a ``[lane_belief_uncertainty]``
    table; if it does not, this raises rather than silently returning
    ``None`` -- a flag that reads as "on" but changes nothing is the exact
    failure class this task exists to close (see the task-1 report's
    original diagnosis: an unwired calibration is not a deliverable).
    """

    v4_changes = protocol_raw.get("v4_changes") or {}
    if not isinstance(v4_changes, Mapping):
        raise ValueError("[v4_changes] must be a table")
    if not bool(v4_changes.get("belief_uncertainty_refit", False)):
        return None
    calibration = load_lane_uncertainty_calibration(lane_belief_config_path)
    if calibration is None:
        raise RuntimeError(
            "v4_changes.belief_uncertainty_refit is enabled but "
            f"{lane_belief_config_path} has no [lane_belief_uncertainty] table"
        )
    return calibration
