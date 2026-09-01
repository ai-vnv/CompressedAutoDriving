"""F10-PPO Visual-Lane v4, Task 1: recalibrate lane-belief uncertainty.

Fits and verifies the ``LaneUncertaintyCalibration`` (a ``std' = max(std *
scale, floor)`` widening map) against the already-collected, held-out
``artifacts/visual_lane/lane_belief_final_validation.csv`` (2400 rows, seeds
36501-36502, small_loop + experiment_loop, CCW). No simulator is run and no
mean estimate is touched -- this script only reads that CSV and scores the
reported sigma.

This does not re-search the calibration from scratch. The (floor, scale)
pair is pre-registered by the plan (see the v4 task brief) and this script's
job is to *verify* it reproduces on the real validation data, per "verify,
do not re-invent" -- widening the band to force a fit is explicitly out of
scope; a failure to reproduce is reported, not papered over.

For heading, the floor is documented as the minimal value for which
coverage_68 enters its pre-registered band [0.60, 0.76] from below -- shown
here as a diagnostic bisection, purely to explain where 0.051 rad comes
from; it is not used to pick a different number than the plan's.

For lateral, no (scale, floor) pair lands both coverage_68 and coverage_95
in band (the z-distribution is the wrong shape). The plan's resolution:
pick the scale that lands coverage_68, accept over-coverage at 95%, and
disclose it explicitly rather than gating on it.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from duckie_pomdp.control.lane_belief_uncertainty import (
    LaneUncertaintyCalibration,
    coverage_report,
    evaluate_uncertainty_gate,
    lateral_overcoverage_disclosure,
    load_validation_errors_and_sigmas,
)

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10 runtime used by Gym-Duckietown.
    import tomli as tomllib


ROOT = Path(__file__).resolve().parents[1]
VALIDATION_CSV = ROOT / "artifacts" / "visual_lane" / "lane_belief_final_validation.csv"
V2_CONFIG = ROOT / "configs" / "lane_belief_v2.toml"
METRICS_PATH = ROOT / "artifacts" / "visual_lane" / "lane_belief_uncertainty_v4_metrics.json"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def _minimal_floor_for_band_entry(
    errors: np.ndarray, sigma_raw: np.ndarray, scale: float, target_coverage_68: float
) -> float:
    """Bisect the smallest floor (at fixed scale) with coverage_68 >= target.

    Diagnostic only -- explains where a floor value comes from; the script
    below always scores the plan's pre-registered (floor, scale), not this
    bisection's output.
    """

    lo, hi = 0.0, 1.0
    for _ in range(60):
        mid = (lo + hi) / 2.0
        sigma = np.maximum(sigma_raw * scale, mid)
        coverage = float(np.mean(np.abs(errors) <= sigma))
        if coverage < target_coverage_68:
            lo = mid
        else:
            hi = mid
    return hi


def main() -> None:
    data = load_validation_errors_and_sigmas(VALIDATION_CSV)
    gate_config = tomllib.loads(V2_CONFIG.read_text())["uncertainty_gate"]
    calibration_config = tomllib.loads(V2_CONFIG.read_text())["lane_belief_uncertainty"]

    cal = LaneUncertaintyCalibration(
        heading_floor_rad=float(calibration_config["heading_floor_rad"]),
        heading_scale=float(calibration_config["heading_scale"]),
        lateral_floor_m=float(calibration_config["lateral_floor_m"]),
        lateral_scale=float(calibration_config["lateral_scale"]),
    )

    heading_error = data["heading"]["error"]
    heading_sigma_raw = data["heading"]["sigma"]
    lateral_error = data["lateral"]["error"]
    lateral_sigma_raw = data["lateral"]["sigma"]

    # --- Diagnostics: the pre-v4 (uncalibrated) picture that motivated this task.
    raw_heading_report = coverage_report(heading_error, heading_sigma_raw)
    raw_lateral_report = coverage_report(lateral_error, lateral_sigma_raw)
    effective_pre_v4_heading_floor = float(np.min(heading_sigma_raw))
    pinned_fraction = float(
        np.mean(heading_sigma_raw <= effective_pre_v4_heading_floor + 1e-6)
    )

    # --- Apply the calibration (verify, do not re-invent).
    heading_sigma_calibrated = np.array(
        [cal.apply(s, 0.0)[0] for s in heading_sigma_raw]
    )
    lateral_sigma_calibrated = np.array(
        [cal.apply(0.0, s)[1] for s in lateral_sigma_raw]
    )
    heading_report = coverage_report(heading_error, heading_sigma_calibrated)
    lateral_report = coverage_report(lateral_error, lateral_sigma_calibrated)

    # --- Diagnostic derivation of the heading floor (explains, does not re-pick).
    derived_heading_floor = _minimal_floor_for_band_entry(
        heading_error, heading_sigma_raw, scale=1.0, target_coverage_68=0.60
    )

    disclosure = lateral_overcoverage_disclosure(lateral_error, lateral_sigma_raw)

    gate_result = evaluate_uncertainty_gate(heading_report, lateral_report, gate_config)

    # A negative-control gate run: the pre-v4 effective floor must FAIL.
    undercalibrated = LaneUncertaintyCalibration(
        effective_pre_v4_heading_floor, 1.0, 0.0, 1.0
    )
    undercalibrated_heading_sigma = np.array(
        [undercalibrated.apply(s, 0.0)[0] for s in heading_sigma_raw]
    )
    undercalibrated_lateral_sigma = np.array(
        [undercalibrated.apply(0.0, s)[1] for s in lateral_sigma_raw]
    )
    undercalibrated_gate_result = evaluate_uncertainty_gate(
        coverage_report(heading_error, undercalibrated_heading_sigma),
        coverage_report(lateral_error, undercalibrated_lateral_sigma),
        gate_config,
    )

    metrics = {
        "gate": "F10-PPO visual-lane v4 uncertainty recalibration",
        "validation_csv": str(VALIDATION_CSV.relative_to(ROOT)),
        "validation_csv_sha256": _sha256(VALIDATION_CSV),
        "n_rows": int(heading_error.size),
        "config": str(V2_CONFIG.relative_to(ROOT)),
        "calibration": {
            "heading_floor_rad": cal.heading_floor_rad,
            "heading_scale": cal.heading_scale,
            "lateral_floor_m": cal.lateral_floor_m,
            "lateral_scale": cal.lateral_scale,
        },
        "pre_v4_uncalibrated": {
            "heading": raw_heading_report,
            "lateral": raw_lateral_report,
            "effective_heading_sigma_floor_rad": effective_pre_v4_heading_floor,
            "fraction_pinned_to_effective_floor": pinned_fraction,
            "note": (
                "the effective floor (~0.0197 rad) is the EKF's posterior "
                "Riccati fixed point, not the raw per-frame measurement "
                "floor (heading_std_floor_rad = 0.015 in lane_belief_v1.toml, "
                "perception/lane_measurement.py). Process noise "
                "(heading_process_std_rad_per_sqrt_s) is re-injected every "
                "predict step and only partially removed by the correction "
                "step because of the lateral/heading coupling in the motion "
                "Jacobian and the detection_validity_gain-damped Kalman gain "
                "(belief/lane_ekf.py) -- a property of the filter dynamics, "
                "not a single config key."
            ),
        },
        "calibrated": {
            "heading": heading_report,
            "lateral": lateral_report,
        },
        "heading_floor_derivation": {
            "method": "minimal floor (scale=1.0) for coverage_68 >= 0.60, "
            "the lower edge of the pre-registered band",
            "derived_floor_rad": derived_heading_floor,
            "configured_floor_rad": cal.heading_floor_rad,
        },
        "lateral_overcoverage_disclosure": disclosure,
        "gate_config": gate_config,
        "gate_result": gate_result,
        "negative_control": {
            "description": (
                "gate must FAIL when scored against the pre-v4 effective "
                "heading floor -- a gate that has never been seen to fail "
                "is not a gate"
            ),
            "heading_floor_rad_used": effective_pre_v4_heading_floor,
            "gate_result": undercalibrated_gate_result,
        },
    }

    METRICS_PATH.write_text(json.dumps(metrics, indent=2, sort_keys=False) + "\n")

    print(f"wrote {METRICS_PATH.relative_to(ROOT)}")
    print(
        "heading: floor={:.4f} scale={:.2f} -> cov68={:.4f} cov95={:.4f} "
        "sigma/rmse={:.4f} rmse={:.4f}".format(
            cal.heading_floor_rad,
            cal.heading_scale,
            heading_report["coverage_68"],
            heading_report["coverage_95"],
            heading_report["sigma_over_rmse"],
            heading_report["rmse"],
        )
    )
    print(
        "lateral: floor={:.4f} scale={:.2f} -> cov68={:.4f} cov95={:.4f} "
        "(over-covers, disclosed, not gated) sigma/rmse={:.4f} rmse={:.4f}".format(
            cal.lateral_floor_m,
            cal.lateral_scale,
            lateral_report["coverage_68"],
            lateral_report["coverage_95"],
            lateral_report["sigma_over_rmse"],
            lateral_report["rmse"],
        )
    )
    print(f"gate pass: {gate_result['pass']}  checks: {gate_result['checks']}")
    print(
        "negative control (pre-v4 floor) gate pass: "
        f"{undercalibrated_gate_result['pass']} "
        f"(expected False) checks: {undercalibrated_gate_result['checks']}"
    )
    if not gate_result["pass"]:
        raise SystemExit("uncertainty gate FAILED -- see metrics artifact for detail")


if __name__ == "__main__":
    main()
