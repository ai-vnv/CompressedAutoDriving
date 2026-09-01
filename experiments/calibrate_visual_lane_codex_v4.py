"""Disjoint real-simulator calibration for the Codex dual-boundary lane mean.

The estimator sees front RGB and fixed camera calibration only.  Simulator
lane state is joined after inference as an offline regression/evaluation
target.  Split assignment is by seed/trajectory, never by frame.
"""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from duckie_pomdp.adapters.gym_duckietown import GymDuckietownConfig, create_gym_duckietown
from duckie_pomdp.control.start_sampler import load_small_loop_tiles
from duckie_pomdp.domain.action import PolicyAction
from duckie_pomdp.perception.camera_geometry import CalibratedGroundProjector
from duckie_pomdp.perception.lane_measurement import (
    CameraLaneMeasurementEstimator,
    load_lane_perception_config,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "lane_belief_v3_codex.toml"
OUTPUT_DIR = ROOT / "artifacts" / "f10_ppo_visual_v4_codex" / "lane_calibration"
CALIBRATION_SEEDS = tuple(range(61101, 61109))
DEVELOPMENT_SEEDS = tuple(range(61201, 61209))
STEPS = 120
CAPTURE_STRIDE = 3


def main() -> None:
    csv_path = OUTPUT_DIR / "lane_measurement_calibration.csv"
    metrics_path = OUTPUT_DIR / "lane_measurement_calibration_metrics.json"
    if csv_path.exists() or metrics_path.exists():
        raise RuntimeError("calibration artifacts already exist; refusing overwrite")
    tiles = load_small_loop_tiles(
        map_name="small_loop", anchor_tile=(1, 0), anchor_heading_rad=np.pi
    )
    rows: list[dict[str, object]] = []
    for split, seeds in (
        ("calibration", CALIBRATION_SEEDS),
        ("development", DEVELOPMENT_SEEDS),
    ):
        for episode_index, seed in enumerate(seeds):
            rows.extend(_episode(split, episode_index, seed, tiles[episode_index % len(tiles)]))

    calibration = [row for row in rows if row["split"] == "calibration" and row["detected"]]
    matrix, offset, retained = _fit_robust_affine(calibration)
    for row in rows:
        if not row["detected"]:
            row.update(calibrated_d="", calibrated_phi="", calibrated_kappa="")
            continue
        corrected = matrix @ _raw_vector(row) + offset
        row.update(
            calibrated_d=float(corrected[0]),
            calibrated_phi=_wrap(float(corrected[1])),
            calibrated_kappa=float(corrected[2]),
        )

    calibration_errors = _errors(rows, "calibration")
    residual_sigma = np.std(calibration_errors, axis=0, ddof=1)
    metrics = {
        "gate": "Codex dual-boundary lane measurement calibration",
        "direction": "counter-clockwise",
        "map": "small_loop",
        "split_unit": "seed/trajectory",
        "calibration_seeds": list(CALIBRATION_SEEDS),
        "development_seeds": list(DEVELOPMENT_SEEDS),
        "seed_overlap": sorted(set(CALIBRATION_SEEDS) & set(DEVELOPMENT_SEEDS)),
        "steps_per_episode": STEPS,
        "capture_stride": CAPTURE_STRIDE,
        "config": str(CONFIG.relative_to(ROOT)),
        "config_sha256_before_fit": _sha256(CONFIG),
        "runtime_inputs": ["front_rgb", "fixed_camera_calibration"],
        "privileged_use": "offline target read after RGB inference",
        "fit": {
            "type": "robust_affine",
            "input_order": ["raw_d_m", "raw_phi_rad", "raw_kappa_inv_m"],
            "output_order": ["d_m", "phi_rad", "kappa_inv_m"],
            "matrix": matrix.tolist(),
            "offset": offset.tolist(),
            "retained": retained,
            "residual_sigma": {
                "lateral_m": float(residual_sigma[0]),
                "heading_rad": float(residual_sigma[1]),
                "curvature_inv_m": float(residual_sigma[2]),
            },
        },
        "calibration": _split_metrics(rows, "calibration"),
        "development": _split_metrics(rows, "development"),
        "source_counts": _source_counts(rows),
        "pre_registered_gate": {
            "minimum_detection_rate": 0.80,
            "maximum_lateral_rmse_m": 0.050,
            "maximum_heading_rmse_rad": 0.150,
        },
    }
    dev = metrics["development"]
    metrics["gate_pass"] = bool(
        dev["detection_rate"] >= 0.80
        and dev["calibrated"]["lateral"]["rmse"] <= 0.050
        and dev["calibrated"]["heading"]["rmse"] <= 0.150
    )
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    metrics_path.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metrics, indent=2))


def _episode(split, episode_index, seed, tile):
    integration = create_gym_duckietown(
        GymDuckietownConfig(
            map_name="small_loop",
            seed=seed,
            domain_randomization=True,
            dynamics_randomization=False,
            frame_rate_hz=30,
            frame_skip=1,
            maximum_steps=STEPS + 2,
            camera_width=640,
            camera_height=480,
            headless=True,
            start_tile=tile.coords,
            start_pose=((tile.base_local_x_m, 0.0, tile.base_local_z_m), tile.base_heading_rad),
        )
    )
    rows = []
    try:
        observation = integration.agent.reset(seed=seed)
        estimator = CameraLaneMeasurementEstimator(
            CalibratedGroundProjector(integration.camera_calibration.read()),
            load_lane_perception_config(CONFIG),
        )
        for step in range(STEPS):
            if step % CAPTURE_STRIDE == 0:
                measurement, diagnostics = estimator.estimate_with_diagnostics(observation.front_rgb)
                truth = integration.privileged.read().true_pomdp_state
                rows.append(
                    {
                        "split": split,
                        "episode": episode_index,
                        "seed": seed,
                        "step": step,
                        "tile_i": tile.coords[0],
                        "tile_j": tile.coords[1],
                        "detected": measurement.detected,
                        "source": diagnostics.source,
                        "boundary_disagreement_m": "" if diagnostics.boundary_disagreement_m is None else diagnostics.boundary_disagreement_m,
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
            yaw = float(np.clip(5.0 * observation.ego.lateral_error_m + 3.0 * observation.ego.heading_error_rad, -4.0, 4.0))
            transition = integration.agent.step(PolicyAction(0.16, yaw))
            observation = transition.observation
            if transition.terminated or transition.truncated:
                break
    finally:
        integration.close()
    return rows


def _fit_robust_affine(rows):
    x = np.asarray([_raw_vector(row) for row in rows], dtype=float)
    y = np.asarray([_truth_vector(row) for row in rows], dtype=float)
    design = np.column_stack((x, np.ones(x.shape[0])))
    keep = np.ones(x.shape[0], dtype=bool)
    coefficients = np.linalg.lstsq(design, y, rcond=None)[0]
    for _ in range(6):
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


def _raw_vector(row):
    return np.asarray([float(row["raw_d"]), float(row["raw_phi"]), float(row["raw_kappa"])])


def _truth_vector(row):
    return np.asarray([float(row["gt_d"]), float(row["gt_phi"]), float(row["gt_kappa"])])


def _errors(rows, split):
    subset = [row for row in rows if row["split"] == split and row["detected"]]
    return np.asarray(
        [[float(row["calibrated_d"]) - float(row["gt_d"]), _wrap(float(row["calibrated_phi"]) - float(row["gt_phi"])), float(row["calibrated_kappa"]) - float(row["gt_kappa"])] for row in subset]
    )


def _metric(errors):
    return {"n": int(errors.size), "bias": float(np.mean(errors)), "mae": float(np.mean(np.abs(errors))), "rmse": float(np.sqrt(np.mean(np.square(errors)))), "residual_sd": float(np.std(errors, ddof=1))}


def _split_metrics(rows, split):
    subset = [row for row in rows if row["split"] == split]
    detected = [row for row in subset if row["detected"]]
    raw_errors = np.asarray([_raw_vector(row) - _truth_vector(row) for row in detected])
    corrected = _errors(rows, split)
    names = ("lateral", "heading", "curvature")
    return {
        "n": len(subset),
        "detections": len(detected),
        "detection_rate": len(detected) / len(subset),
        "raw": {name: _metric(raw_errors[:, index]) for index, name in enumerate(names)},
        "calibrated": {name: _metric(corrected[:, index]) for index, name in enumerate(names)},
    }


def _source_counts(rows):
    result = {}
    for row in rows:
        key = f"{row['split']}:{row['source']}"
        result[key] = result.get(key, 0) + 1
    return result


def _wrap(value):
    return float((value + np.pi) % (2.0 * np.pi) - np.pi)


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


if __name__ == "__main__":
    main()
