"""Fit the fixed RGB-lane preview calibration on disjoint CCW trajectories.

The estimator sees only the rendered front RGB image.  Simulator lane values
are joined after inference and are used solely to fit/evaluate one small affine
calibration.  Seeds are split at trajectory level; no frame-level leakage is
possible.
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
from duckie_pomdp.domain.action import PolicyAction
from duckie_pomdp.perception.camera_geometry import CalibratedGroundProjector
from duckie_pomdp.perception.lane_measurement import (
    CameraLaneMeasurementEstimator,
    load_lane_perception_config,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "lane_belief_v1.toml"
OUTPUT_DIR = ROOT / "artifacts" / "visual_lane"
CSV_PATH = OUTPUT_DIR / "lane_dynamic_calibration.csv"
METRICS_PATH = OUTPUT_DIR / "lane_dynamic_calibration_metrics.json"

CALIBRATION_SEEDS = (36301, 36302)
DEVELOPMENT_SEEDS = (36401, 36402)
MAPS = ("small_loop", "experiment_loop")
STEPS_PER_EPISODE = 600
CAPTURE_STRIDE = 3


def main() -> None:
    import gym_duckietown.simulator as simulator_module

    graphics_information = simulator_module.get_graphics_information()
    simulator_module.get_graphics_information = lambda: graphics_information
    rows: list[dict[str, object]] = []
    for split, seeds in (
        ("calibration", CALIBRATION_SEEDS),
        ("development", DEVELOPMENT_SEEDS),
    ):
        for map_name in MAPS:
            for seed in seeds:
                rows.extend(_episode(split, map_name, seed))

    calibration_rows = [
        row
        for row in rows
        if row["split"] == "calibration" and row["detected"]
    ]
    matrix, offset, retained = _fit_robust_affine(calibration_rows)
    for row in rows:
        if not row["detected"]:
            row.update(
                calibrated_lateral_error_m="",
                calibrated_heading_error_rad="",
                calibrated_curvature_inv_m="",
            )
            continue
        raw = _raw_vector(row)
        calibrated = matrix @ raw + offset
        row.update(
            calibrated_lateral_error_m=float(calibrated[0]),
            calibrated_heading_error_rad=_wrap(float(calibrated[1])),
            calibrated_curvature_inv_m=float(calibrated[2]),
        )

    calibration_residuals = _residuals(
        [row for row in rows if row["split"] == "calibration" and row["detected"]],
        calibrated=True,
    )
    sigmas = np.std(calibration_residuals, axis=0, ddof=1)
    metrics = {
        "gate": "visual-lane dynamic affine calibration development gate",
        "direction": "counter-clockwise",
        "maps": list(MAPS),
        "split_unit": "seed/trajectory",
        "calibration_seeds": list(CALIBRATION_SEEDS),
        "development_seeds": list(DEVELOPMENT_SEEDS),
        "seed_overlap": sorted(set(CALIBRATION_SEEDS) & set(DEVELOPMENT_SEEDS)),
        "steps_per_episode": STEPS_PER_EPISODE,
        "capture_stride": CAPTURE_STRIDE,
        "rows": len(rows),
        "config": str(CONFIG.relative_to(ROOT)),
        "config_sha256": _sha256(CONFIG),
        "fit": {
            "type": "robust_affine",
            "input_order": ["raw_d_m", "raw_phi_rad", "raw_kappa_inv_m"],
            "output_order": ["d_m", "phi_rad", "kappa_inv_m"],
            "matrix": matrix.tolist(),
            "offset": offset.tolist(),
            "calibration_detections": len(calibration_rows),
            "robust_fit_retained": retained,
            "residual_sigma": {
                "lateral_m": float(sigmas[0]),
                "heading_rad": float(sigmas[1]),
                "curvature_inv_m": float(sigmas[2]),
            },
        },
        "calibration": _split_metrics(rows, "calibration"),
        "development": _split_metrics(rows, "development"),
        "runtime_inputs": ["front_rgb", "fixed_camera_calibration"],
        "privileged_use": "offline target after RGB inference only",
        "pre_registered_development_gate": {
            "minimum_detection_rate": 0.80,
            "maximum_lateral_measurement_rmse_m": 0.05,
            "maximum_heading_measurement_rmse_rad": 0.15,
        },
    }
    development = metrics["development"]
    assert isinstance(development, dict)
    calibrated = development["calibrated"]
    assert isinstance(calibrated, dict)
    metrics["development_gate_pass"] = bool(
        float(development["detection_rate"]) >= 0.80
        and float(calibrated["lateral"]["rmse"]) <= 0.05
        and float(calibrated["heading"]["rmse"]) <= 0.15
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with CSV_PATH.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    METRICS_PATH.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metrics, indent=2))


def _episode(split: str, map_name: str, seed: int) -> list[dict[str, object]]:
    integration = create_gym_duckietown(
        GymDuckietownConfig(
            map_name=map_name,
            seed=seed,
            domain_randomization=True,
            dynamics_randomization=False,
            frame_rate_hz=30,
            frame_skip=1,
            maximum_steps=STEPS_PER_EPISODE + 2,
            headless=True,
            start_tile=(1, 0),
            start_pose=((0.520, 0.0, 0.1755), pi),
        )
    )
    rows: list[dict[str, object]] = []
    try:
        observation = integration.agent.reset(seed=seed)
        estimator = CameraLaneMeasurementEstimator(
            CalibratedGroundProjector(integration.camera_calibration.read()),
            load_lane_perception_config(CONFIG),
        )
        for step in range(STEPS_PER_EPISODE):
            yaw_command = float(
                np.clip(
                    5.0 * observation.ego.lateral_error_m
                    + 3.0 * observation.ego.heading_error_rad,
                    -4.0,
                    4.0,
                )
            )
            transition = integration.agent.step(PolicyAction(0.18, yaw_command))
            observation = transition.observation
            if step % CAPTURE_STRIDE == 0:
                measurement = estimator.estimate(observation.front_rgb)
                # Runtime inference is complete before these evaluation targets
                # are attached to the offline calibration row.
                truth = integration.privileged.read().true_pomdp_state
                rows.append(
                    {
                        "split": split,
                        "map": map_name,
                        "direction": "counter-clockwise",
                        "seed": seed,
                        "step": step,
                        "detected": measurement.detected,
                        "visible_point_count": measurement.visible_point_count,
                        "raw_lateral_error_m": _optional(measurement.lateral_error_m),
                        "raw_heading_error_rad": _optional(measurement.heading_error_rad),
                        "raw_curvature_inv_m": _optional(measurement.curvature_inv_m),
                        "gt_lateral_error_m": observation.ego.lateral_error_m,
                        "gt_heading_error_rad": observation.ego.heading_error_rad,
                        "gt_curvature_inv_m": _optional(truth.road.curvature_inv_m),
                        "calibrated_lateral_error_m": "",
                        "calibrated_heading_error_rad": "",
                        "calibrated_curvature_inv_m": "",
                    }
                )
            if transition.terminated or transition.truncated:
                break
    finally:
        integration.close()
    return rows


def _fit_robust_affine(
    rows: list[dict[str, object]],
) -> tuple[np.ndarray, np.ndarray, int]:
    x = np.asarray([_raw_vector(row) for row in rows], dtype=float)
    y = np.asarray([_truth_vector(row) for row in rows], dtype=float)
    design = np.column_stack((x, np.ones(x.shape[0], dtype=float)))
    keep = np.ones(x.shape[0], dtype=bool)
    coefficients = np.linalg.lstsq(design, y, rcond=None)[0]
    for _ in range(5):
        coefficients = np.linalg.lstsq(design[keep], y[keep], rcond=None)[0]
        residuals = y - design @ coefficients
        median = np.median(residuals[keep], axis=0)
        mad = np.median(np.abs(residuals[keep] - median), axis=0)
        scale = np.maximum(1.4826 * mad, np.asarray((0.005, 0.015, 0.10)))
        updated = np.all(np.abs(residuals - median) <= 3.5 * scale, axis=1)
        if np.array_equal(updated, keep):
            break
        keep = updated
    coefficients = np.linalg.lstsq(design[keep], y[keep], rcond=None)[0]
    return coefficients[:3].T, coefficients[3], int(np.count_nonzero(keep))


def _split_metrics(rows: list[dict[str, object]], split: str) -> dict[str, object]:
    subset = [row for row in rows if row["split"] == split]
    detected = [row for row in subset if row["detected"]]
    return {
        "opportunities": len(subset),
        "detections": len(detected),
        "detection_rate": len(detected) / len(subset),
        "raw": _metrics(_residuals(detected, calibrated=False)),
        "calibrated": _metrics(_residuals(detected, calibrated=True)),
        "by_map": {
            map_name: {
                "n": sum(row["map"] == map_name for row in subset),
                "detection_rate": float(
                    np.mean(
                        [
                            bool(row["detected"])
                            for row in subset
                            if row["map"] == map_name
                        ]
                    )
                ),
                "calibrated": _metrics(
                    _residuals(
                        [
                            row
                            for row in detected
                            if row["map"] == map_name
                        ],
                        calibrated=True,
                    )
                ),
            }
            for map_name in MAPS
        },
    }


def _residuals(
    rows: list[dict[str, object]], *, calibrated: bool
) -> np.ndarray:
    prefix = "calibrated" if calibrated else "raw"
    predicted = np.asarray(
        [
            [
                float(row[f"{prefix}_lateral_error_m"]),
                float(row[f"{prefix}_heading_error_rad"]),
                float(row[f"{prefix}_curvature_inv_m"]),
            ]
            for row in rows
        ],
        dtype=float,
    )
    truth = np.asarray([_truth_vector(row) for row in rows], dtype=float)
    residuals = predicted - truth
    residuals[:, 1] = np.arctan2(np.sin(residuals[:, 1]), np.cos(residuals[:, 1]))
    return residuals


def _metrics(residuals: np.ndarray) -> dict[str, object]:
    labels = ("lateral", "heading", "curvature")
    return {
        label: {
            "n": int(residuals.shape[0]),
            "bias": float(np.mean(residuals[:, index])),
            "mae": float(np.mean(np.abs(residuals[:, index]))),
            "rmse": float(np.sqrt(np.mean(np.square(residuals[:, index])))),
            "residual_sd": float(np.std(residuals[:, index], ddof=1)),
        }
        for index, label in enumerate(labels)
    }


def _raw_vector(row: dict[str, object]) -> np.ndarray:
    return np.asarray(
        [
            float(row["raw_lateral_error_m"]),
            float(row["raw_heading_error_rad"]),
            float(row["raw_curvature_inv_m"]),
        ],
        dtype=float,
    )


def _truth_vector(row: dict[str, object]) -> np.ndarray:
    return np.asarray(
        [
            float(row["gt_lateral_error_m"]),
            float(row["gt_heading_error_rad"]),
            float(row["gt_curvature_inv_m"]),
        ],
        dtype=float,
    )


def _optional(value: float | None) -> float | str:
    return "" if value is None else float(value)


def _wrap(angle: float) -> float:
    return float(np.arctan2(np.sin(angle), np.cos(angle)))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


if __name__ == "__main__":
    main()
