"""Fit visual-lane calibration on deliberately excited ego poses.

Earlier calibration trajectories stayed close to the lane centre.  Their
affine fit therefore looked accurate while becoming wrong-signed once a
closed-loop controller accumulated heading error.  This offline experiment
renders disjoint, deterministic lateral/heading pose perturbations on every
tile of both native loop maps.  Runtime inference still consumes RGB only;
simulator lane pose is joined after inference solely as the fit/evaluation
target.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path

import numpy as np

from duckie_pomdp.adapters.gym_duckietown import (
    GymDuckietownConfig,
    create_gym_duckietown,
)
from duckie_pomdp.control.start_sampler import DrivableTile, load_small_loop_tiles
from duckie_pomdp.perception.camera_geometry import CalibratedGroundProjector
from duckie_pomdp.perception.lane_measurement import (
    CameraLaneMeasurementEstimator,
    load_lane_perception_config,
)


ROOT = Path(__file__).resolve().parents[1]
SOURCE_CONFIG = ROOT / "configs" / "lane_belief_v4_transfer.toml"
DEFAULT_OUTPUT = ROOT / "artifacts" / "f10_ppo_visual_v6" / "lane_calibration"
MAP_SEED_BASES = {
    "calibration": {"small_loop": 70_000, "experiment_loop": 70_300},
    "development": {"small_loop": 71_000, "experiment_loop": 71_300},
}
POSES = (
    ("heading_negative_large", 0.0, -0.30),
    ("heading_negative", 0.0, -0.15),
    ("centred", 0.0, 0.0),
    ("heading_positive", 0.0, 0.15),
    ("heading_positive_large", 0.0, 0.30),
    ("lateral_right", -0.045, 0.0),
    ("lateral_left", 0.045, 0.0),
    ("right_heading_negative", -0.045, -0.30),
    ("right_heading_negative_small", -0.045, -0.15),
    ("right_heading_positive_small", -0.045, 0.15),
    ("right_heading_positive", -0.045, 0.30),
    ("left_heading_negative", 0.045, -0.30),
    ("left_heading_negative_small", 0.045, -0.15),
    ("left_heading_positive_small", 0.045, 0.15),
    ("left_heading_positive", 0.045, 0.30),
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    csv_path = args.output / "lane_pose_calibration.csv"
    metrics_path = args.output / "lane_pose_calibration_metrics.json"
    if csv_path.exists() or metrics_path.exists():
        raise FileExistsError(f"refusing to overwrite pose calibration in {args.output}")

    rows: list[dict[str, object]] = []
    for split, maps in MAP_SEED_BASES.items():
        for map_name, seed_base in maps.items():
            rows.extend(_map_samples(split, map_name, seed_base))

    fit_rows = [row for row in rows if row["split"] == "calibration" and row["detected"]]
    matrix, quadratic_matrix, offset = _fit_balanced_ridge(fit_rows)
    for row in rows:
        if not row["detected"]:
            row.update(calibrated_d="", calibrated_phi="", calibrated_kappa="")
            continue
        raw_vector = _raw(row)
        corrected = (
            matrix @ raw_vector
            + quadratic_matrix @ _quadratic_features(raw_vector)
            + offset
        )
        row.update(
            calibrated_d=float(corrected[0]),
            calibrated_phi=_wrap(float(corrected[1])),
            calibrated_kappa=float(corrected[2]),
        )

    calibration_errors = _error_matrix(rows, "calibration")
    metrics = {
        "purpose": "pose-excited C0/C1 visual-lane calibration; runtime RGB only",
        "source_config": str(SOURCE_CONFIG.relative_to(ROOT)),
        "split_unit": "seed/tile/pose",
        "pose_grid": [
            {"name": name, "lateral_offset_m": lateral, "heading_offset_rad": heading}
            for name, lateral, heading in POSES
        ],
        "seed_bases": MAP_SEED_BASES,
        "seed_overlap": [],
        "fit": {
            "type": "map_turn_pose_balanced_ridge_quadratic",
            "ridge_lambda": 1.0e-3,
            "input_order": ["raw_d_m", "raw_phi_rad", "raw_kappa_inv_m"],
            "output_order": ["d_m", "phi_rad", "kappa_inv_m"],
            "matrix": matrix.tolist(),
            "quadratic_feature_order": [
                "d_squared",
                "phi_squared",
                "kappa_squared",
                "d_phi",
                "d_kappa",
                "phi_kappa",
            ],
            "quadratic_matrix": quadratic_matrix.tolist(),
            "offset": offset.tolist(),
            "residual_sigma": {
                "lateral_m": float(np.std(calibration_errors[:, 0], ddof=1)),
                "heading_rad": float(np.std(calibration_errors[:, 1], ddof=1)),
                "curvature_inv_m": float(np.std(calibration_errors[:, 2], ddof=1)),
            },
        },
        "calibration": _split_metrics(rows, "calibration"),
        "development": _split_metrics(rows, "development"),
        "runtime_inputs": ["front_rgb", "fixed_camera_calibration"],
        "privileged_use": "offline target after image inference",
        "pre_registered_gate": {
            "minimum_detection_rate": 0.55,
            "maximum_lateral_rmse_m": 0.050,
            "maximum_heading_rmse_rad": 0.180,
            "minimum_excited_heading_sign_accuracy": 0.80,
            "minimum_excited_heading_correlation": 0.70,
        },
    }
    dev = metrics["development"]
    metrics["gate_pass"] = bool(
        dev["detection_rate"] >= 0.55
        and dev["calibrated"]["lateral"]["rmse"] <= 0.050
        and dev["calibrated"]["heading"]["rmse"] <= 0.180
        and dev["excited_heading_sign_accuracy"] >= 0.80
        and dev["excited_heading_correlation"] >= 0.70
    )

    args.output.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    metrics_path.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metrics, indent=2))
    if not metrics["gate_pass"]:
        raise SystemExit(1)


def _map_samples(split: str, map_name: str, seed_base: int) -> list[dict[str, object]]:
    tiles = load_small_loop_tiles(
        map_name=map_name,
        anchor_tile=(1, 0),
        anchor_heading_rad=np.pi,
    )
    first = _config(map_name, seed_base, tiles[0], *POSES[0][1:])
    integration = create_gym_duckietown(first)
    rows: list[dict[str, object]] = []
    try:
        for tile_index, tile in enumerate(tiles):
            for pose_index, (pose_name, lateral, heading) in enumerate(POSES):
                seed = seed_base + 20 * tile_index + pose_index
                config = _config(map_name, seed, tile, lateral, heading)
                integration.reconfigure_native_episode(config)
                observation = integration.agent.reset(seed=seed)
                estimator = CameraLaneMeasurementEstimator(
                    CalibratedGroundProjector(integration.camera_calibration.read()),
                    load_lane_perception_config(SOURCE_CONFIG),
                )
                measurement, diagnostics = estimator.estimate_with_diagnostics(
                    observation.front_rgb
                )
                truth = integration.privileged.read().true_pomdp_state
                rows.append(
                    {
                        "split": split,
                        "map": map_name,
                        "seed": seed,
                        "tile_i": tile.coords[0],
                        "tile_j": tile.coords[1],
                        "tile_kind": tile.kind,
                        "turn_family": _turn_family(truth.road.curvature_inv_m),
                        "pose_name": pose_name,
                        "requested_lateral_offset_m": lateral,
                        "requested_heading_offset_rad": heading,
                        "detected": measurement.detected,
                        "source": diagnostics.source,
                        "raw_d": "" if measurement.lateral_error_m is None else measurement.lateral_error_m,
                        "raw_phi": "" if measurement.heading_error_rad is None else measurement.heading_error_rad,
                        "raw_kappa": "" if measurement.curvature_inv_m is None else measurement.curvature_inv_m,
                        "gt_d": observation.ego.lateral_error_m,
                        "gt_phi": observation.ego.heading_error_rad,
                        "gt_kappa": truth.road.curvature_inv_m,
                        "calibrated_d": "",
                        "calibrated_phi": "",
                        "calibrated_kappa": "",
                    }
                )
    finally:
        integration.close()
    return rows


def _config(
    map_name: str,
    seed: int,
    tile: DrivableTile,
    lateral_offset_m: float,
    heading_offset_rad: float,
) -> GymDuckietownConfig:
    direction_x = math.cos(tile.base_heading_rad)
    direction_z = -math.sin(tile.base_heading_rad)
    right_x, right_z = -direction_z, direction_x
    return GymDuckietownConfig(
        map_name=map_name,
        seed=seed,
        domain_randomization=True,
        dynamics_randomization=False,
        frame_rate_hz=30,
        frame_skip=1,
        maximum_steps=3,
        camera_width=640,
        camera_height=480,
        headless=True,
        start_tile=tile.coords,
        start_pose=(
            (
                tile.base_local_x_m + lateral_offset_m * right_x,
                0.0,
                tile.base_local_z_m + lateral_offset_m * right_z,
            ),
            tile.base_heading_rad + heading_offset_rad,
        ),
    )


def _fit_balanced_ridge(rows):
    x = np.asarray([_raw(row) for row in rows], dtype=float)
    y = np.asarray([_truth(row) for row in rows], dtype=float)
    groups = [
        (str(row["map"]), str(row["turn_family"]), str(row["pose_name"]))
        for row in rows
    ]
    counts = Counter(groups)
    weights = np.asarray([1.0 / counts[group] for group in groups], dtype=float)
    weights *= len(weights) / np.sum(weights)
    features = np.column_stack((x, np.asarray([_quadratic_features(row) for row in x])))
    mean = np.average(features, axis=0, weights=weights)
    variance = np.average(np.square(features - mean), axis=0, weights=weights)
    scale = np.maximum(np.sqrt(variance), 1.0e-6)
    design = np.column_stack(((features - mean) / scale, np.ones(x.shape[0])))
    normal = design.T @ (weights[:, None] * design)
    normal += np.diag((1.0e-3,) * features.shape[1] + (0.0,))
    coefficients = np.linalg.solve(normal, design.T @ (weights[:, None] * y))
    slopes = coefficients[:-1].T / scale[None, :]
    offset = coefficients[-1] - slopes @ mean
    return slopes[:, :3], slopes[:, 3:], offset


def _quadratic_features(raw_vector):
    d_value, phi_value, kappa_value = raw_vector
    return np.asarray(
        (
            d_value * d_value,
            phi_value * phi_value,
            kappa_value * kappa_value,
            d_value * phi_value,
            d_value * kappa_value,
            phi_value * kappa_value,
        ),
        dtype=float,
    )


def _split_metrics(rows, split: str):
    selected = [row for row in rows if row["split"] == split]
    detected = [row for row in selected if row["detected"]]
    excited = [row for row in detected if abs(float(row["gt_phi"])) >= 0.10]
    gt_heading = np.asarray([float(row["gt_phi"]) for row in excited], dtype=float)
    predicted_heading = np.asarray(
        [float(row["calibrated_phi"]) for row in excited], dtype=float
    )
    raw_errors = np.asarray([_raw(row) - _truth(row) for row in detected])
    calibrated_errors = _error_matrix(detected, split=None)
    return {
        "rows": len(selected),
        "detections": len(detected),
        "detection_rate": len(detected) / len(selected),
        "raw": _channels(raw_errors),
        "calibrated": _channels(calibrated_errors),
        "excited_heading_samples": len(excited),
        "excited_heading_sign_accuracy": float(
            np.mean(np.sign(predicted_heading) == np.sign(gt_heading))
        ),
        "excited_heading_correlation": float(
            np.corrcoef(predicted_heading, gt_heading)[0, 1]
        ),
        "by_map": {
            name: _group_metrics([row for row in selected if row["map"] == name])
            for name in ("small_loop", "experiment_loop")
        },
        "by_turn": {
            name: _group_metrics(
                [row for row in selected if row["turn_family"] == name]
            )
            for name in ("right", "straight", "left")
        },
    }


def _group_metrics(rows):
    detected = [row for row in rows if row["detected"]]
    return {
        "rows": len(rows),
        "detections": len(detected),
        "detection_rate": len(detected) / len(rows) if rows else 0.0,
        "calibrated": _channels(_error_matrix(detected, split=None)),
    }


def _error_matrix(rows, split: str | None):
    selected = [
        row
        for row in rows
        if row["detected"] and (split is None or row["split"] == split)
    ]
    return np.asarray(
        [
            (
                float(row["calibrated_d"]) - float(row["gt_d"]),
                _wrap(float(row["calibrated_phi"]) - float(row["gt_phi"])),
                float(row["calibrated_kappa"]) - float(row["gt_kappa"]),
            )
            for row in selected
        ],
        dtype=float,
    )


def _channels(errors):
    names = ("lateral", "heading", "curvature")
    return {name: _metric(errors[:, index]) for index, name in enumerate(names)}


def _metric(values):
    return {
        "n": int(values.size),
        "bias": float(np.mean(values)),
        "mae": float(np.mean(np.abs(values))),
        "rmse": float(np.sqrt(np.mean(np.square(values)))),
    }


def _raw(row):
    return np.asarray(
        [float(row["raw_d"]), float(row["raw_phi"]), float(row["raw_kappa"])],
        dtype=float,
    )


def _truth(row):
    return np.asarray(
        [float(row["gt_d"]), float(row["gt_phi"]), float(row["gt_kappa"])],
        dtype=float,
    )


def _turn_family(curvature: float) -> str:
    if curvature < -0.75:
        return "right"
    if curvature > 0.75:
        return "left"
    return "straight"


def _wrap(value: float) -> float:
    return math.atan2(math.sin(value), math.cos(value))


if __name__ == "__main__":
    main()
