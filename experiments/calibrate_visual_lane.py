"""Calibrate camera-only lane pose on real CCW Gym-Duckietown renders.

Privileged ``get_lane_pos2`` values are read only after RGB inference and are
used solely for offline fitting/evaluation.  The calibration and held-out seed
blocks are disjoint.  The resulting recommendation is not written into the
runtime TOML automatically; freezing it remains an explicit reviewed edit.
"""

from __future__ import annotations

import csv
import hashlib
import json
from math import pi
from pathlib import Path

import numpy as np

from duckie_pomdp.adapters.gym_duckietown import (
    GymDuckietownConfig,
    create_gym_duckietown,
)
from duckie_pomdp.perception.camera_geometry import CalibratedGroundProjector
from duckie_pomdp.perception.lane_measurement import (
    CameraLaneMeasurementEstimator,
    load_lane_perception_config,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "lane_belief_v1.toml"
ARTIFACT_DIR = ROOT / "artifacts" / "visual_lane"
CSV_PATH = ARTIFACT_DIR / "lane_pose_calibration.csv"
METRICS_PATH = ARTIFACT_DIR / "lane_pose_calibration_metrics.json"

CALIBRATION_SEEDS = (36001, 36002, 36003)
VALIDATION_SEEDS = (36101, 36102, 36103)
LATERAL_OFFSETS_M = (-0.06, -0.03, 0.0, 0.03, 0.06)
HEADING_OFFSETS_RAD = (-0.18, -0.09, 0.0, 0.09, 0.18)


def main() -> None:
    # The simulator queries static GL strings in every constructor. Repeating
    # that query across a pose sweep is unstable under headless llvmpipe; cache
    # it once. Rendering and physics remain the real simulator implementation.
    import gym_duckietown.simulator as simulator_module

    graphics_information = simulator_module.get_graphics_information()
    simulator_module.get_graphics_information = lambda: graphics_information
    rows: list[dict[str, object]] = []
    for split, seeds in (
        ("calibration", CALIBRATION_SEEDS),
        ("validation", VALIDATION_SEEDS),
    ):
        for seed in seeds:
            for lateral_offset in LATERAL_OFFSETS_M:
                for heading_offset in HEADING_OFFSETS_RAD:
                    rows.append(
                        _sample(
                            split=split,
                            seed=seed,
                            lateral_offset_m=lateral_offset,
                            heading_offset_rad=heading_offset,
                        )
                    )

    detected_calibration = [
        row for row in rows if row["split"] == "calibration" and row["detected"]
    ]
    biases = {
        "lateral_bias_m": _mean_error(
            detected_calibration, "raw_lateral_error_m", "gt_lateral_error_m"
        ),
        "heading_bias_rad": _mean_error(
            detected_calibration, "raw_heading_error_rad", "gt_heading_error_rad"
        ),
        # The calibration poses are all on a straight segment.  A curvature
        # correction fitted here would not be identifiable/generalizable.
        "curvature_bias_inv_m": 0.0,
    }
    for row in rows:
        if not row["detected"]:
            row.update(
                corrected_lateral_error_m="",
                corrected_heading_error_rad="",
                corrected_curvature_inv_m="",
            )
            continue
        row["corrected_lateral_error_m"] = float(row["raw_lateral_error_m"]) - biases[
            "lateral_bias_m"
        ]
        row["corrected_heading_error_rad"] = _wrap(
            float(row["raw_heading_error_rad"]) - biases["heading_bias_rad"]
        )
        row["corrected_curvature_inv_m"] = float(row["raw_curvature_inv_m"])

    residual_sigmas = {
        "lateral_sigma_m": _residual_std(
            detected_calibration,
            "raw_lateral_error_m",
            "gt_lateral_error_m",
            biases["lateral_bias_m"],
        ),
        "heading_sigma_rad": _residual_std(
            detected_calibration,
            "raw_heading_error_rad",
            "gt_heading_error_rad",
            biases["heading_bias_rad"],
        ),
        "curvature_sigma_inv_m": _residual_std(
            detected_calibration,
            "raw_curvature_inv_m",
            "gt_curvature_inv_m",
            0.0,
        ),
    }
    metrics = {
        "gate": "visual-lane calibration-only pose sweep",
        "direction": "counter-clockwise",
        "map": "small_loop",
        "split_unit": "seed",
        "calibration_seeds": list(CALIBRATION_SEEDS),
        "validation_seeds": list(VALIDATION_SEEDS),
        "seed_overlap": sorted(set(CALIBRATION_SEEDS) & set(VALIDATION_SEEDS)),
        "samples": len(rows),
        "config": str(CONFIG.relative_to(ROOT)),
        "config_sha256": _sha256(CONFIG),
        "recommended_calibration": {**biases, **residual_sigmas},
        "frozen_runtime_decision": {
            "lateral_bias_m": biases["lateral_bias_m"],
            "heading_bias_rad": 0.0,
            "curvature_bias_inv_m": 0.0,
            **residual_sigmas,
            "reason": (
                "lateral additive correction improved held-out RMSE; heading "
                "correction worsened it and was rejected; straight poses do "
                "not identify curvature bias"
            ),
        },
        "calibration": _split_metrics(rows, "calibration"),
        "held_out_validation": _split_metrics(rows, "validation"),
        "curvature_limitation": (
            "straight-segment pose sweep does not identify curvature bias; "
            "runtime curvature bias remains zero and trajectory validation is required"
        ),
        "runtime_inputs": ["front_rgb", "fixed_camera_calibration"],
        "privileged_use": "offline comparison/fitting after RGB inference only",
    }
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    with CSV_PATH.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    METRICS_PATH.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metrics, indent=2))


def _sample(
    *,
    split: str,
    seed: int,
    lateral_offset_m: float,
    heading_offset_rad: float,
) -> dict[str, object]:
    # For the frozen CCW spawn, positive simulator lane distance corresponds
    # to a negative local-z offset. Heading error has the opposite sign of the
    # configured SE(2) heading perturbation.
    integration = create_gym_duckietown(
        GymDuckietownConfig(
            map_name="small_loop",
            seed=seed,
            domain_randomization=True,
            dynamics_randomization=False,
            maximum_steps=2,
            headless=True,
            start_tile=(1, 0),
            start_pose=(
                (0.520, 0.0, 0.1755 - lateral_offset_m),
                pi - heading_offset_rad,
            ),
        )
    )
    try:
        observation = integration.agent.reset(seed=seed)
        estimator = CameraLaneMeasurementEstimator(
            CalibratedGroundProjector(integration.camera_calibration.read()),
            load_lane_perception_config(CONFIG),
        )
        # Runtime inference ends here. Privileged truth is read only below.
        measurement = estimator.estimate(observation.front_rgb)
        privileged = integration.privileged.read()
        curvature = privileged.true_pomdp_state.road.curvature_inv_m
        return {
            "split": split,
            "seed": seed,
            "map": "small_loop",
            "direction": "counter-clockwise",
            "lateral_pose_offset_m": lateral_offset_m,
            "heading_pose_offset_rad": heading_offset_rad,
            "detected": measurement.detected,
            "visible_point_count": measurement.visible_point_count,
            "fit_residual_m": _optional(measurement.fit_residual_m),
            "gt_lateral_error_m": observation.ego.lateral_error_m,
            "gt_heading_error_rad": observation.ego.heading_error_rad,
            "gt_curvature_inv_m": 0.0 if curvature is None else curvature,
            "raw_lateral_error_m": _optional(measurement.lateral_error_m),
            "raw_heading_error_rad": _optional(measurement.heading_error_rad),
            "raw_curvature_inv_m": _optional(measurement.curvature_inv_m),
            "corrected_lateral_error_m": "",
            "corrected_heading_error_rad": "",
            "corrected_curvature_inv_m": "",
        }
    finally:
        integration.close()


def _split_metrics(rows: list[dict[str, object]], split: str) -> dict[str, object]:
    subset = [row for row in rows if row["split"] == split]
    detected = [row for row in subset if row["detected"]]
    return {
        "opportunities": len(subset),
        "detections": len(detected),
        "detection_rate": len(detected) / len(subset),
        "raw": {
            "lateral": _metric(detected, "raw_lateral_error_m", "gt_lateral_error_m"),
            "heading": _metric(detected, "raw_heading_error_rad", "gt_heading_error_rad"),
            "curvature": _metric(detected, "raw_curvature_inv_m", "gt_curvature_inv_m"),
        },
        "corrected": {
            "lateral": _metric(
                detected, "corrected_lateral_error_m", "gt_lateral_error_m"
            ),
            "heading": _metric(
                detected, "corrected_heading_error_rad", "gt_heading_error_rad"
            ),
            "curvature": _metric(
                detected, "corrected_curvature_inv_m", "gt_curvature_inv_m"
            ),
        },
    }


def _metric(
    rows: list[dict[str, object]], predicted: str, truth: str
) -> dict[str, float]:
    errors = np.asarray(
        [float(row[predicted]) - float(row[truth]) for row in rows], dtype=float
    )
    return {
        "n": int(errors.size),
        "bias": float(np.mean(errors)),
        "mae": float(np.mean(np.abs(errors))),
        "rmse": float(np.sqrt(np.mean(np.square(errors)))),
        "residual_sd": float(np.std(errors, ddof=1)) if errors.size > 1 else 0.0,
    }


def _mean_error(rows: list[dict[str, object]], predicted: str, truth: str) -> float:
    return float(
        np.mean([float(row[predicted]) - float(row[truth]) for row in rows])
    )


def _residual_std(
    rows: list[dict[str, object]], predicted: str, truth: str, bias: float
) -> float:
    residuals = np.asarray(
        [float(row[predicted]) - float(row[truth]) - bias for row in rows],
        dtype=float,
    )
    return float(np.std(residuals, ddof=1))


def _optional(value: float | None) -> float | str:
    return "" if value is None else float(value)


def _wrap(angle: float) -> float:
    return float(np.arctan2(np.sin(angle), np.cos(angle)))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


if __name__ == "__main__":
    main()
