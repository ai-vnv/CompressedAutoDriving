"""Fit a map-balanced visual-lane calibration for C0 -> C1 transfer.

The estimator receives RGB plus fixed camera calibration. Ground-truth lane
pose is joined only after inference as an offline calibration/evaluation
target. Splits are disjoint by seed and trajectory, and right turns receive
the same aggregate fit weight as straight/left geometry.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np

from duckie_pomdp.adapters.gym_duckietown import (
    GymDuckietownConfig,
    create_gym_duckietown,
)
from duckie_pomdp.control.start_sampler import load_small_loop_tiles
from duckie_pomdp.domain.action import PolicyAction
from duckie_pomdp.perception.camera_geometry import CalibratedGroundProjector
from duckie_pomdp.perception.lane_measurement import (
    CameraLaneMeasurementEstimator,
    load_lane_perception_config,
)


ROOT = Path(__file__).resolve().parents[1]
SOURCE_CONFIG = ROOT / "configs" / "lane_belief_v3_codex.toml"
DEFAULT_OUTPUT = ROOT / "artifacts" / "f10_ppo_visual_v5_c1" / "lane_calibration"
STEPS = 60
CAPTURE_STRIDE = 2
SPLITS = {
    "calibration": {
        "small_loop": tuple(range(64001, 64009)),
        "experiment_loop": tuple(range(64021, 64033)),
    },
    "development": {
        "small_loop": tuple(range(64101, 64109)),
        "experiment_loop": tuple(range(64121, 64133)),
    },
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    csv_path = args.output / "lane_transfer_calibration.csv"
    metrics_path = args.output / "lane_transfer_calibration_metrics.json"
    if csv_path.exists() or metrics_path.exists():
        raise FileExistsError(f"refusing to overwrite lane calibration in {args.output}")

    rows: list[dict[str, object]] = []
    for split, maps in SPLITS.items():
        for map_name, seeds in maps.items():
            tiles = load_small_loop_tiles(
                map_name=map_name,
                anchor_tile=(1, 0),
                anchor_heading_rad=np.pi,
            )
            if len(seeds) != len(tiles):
                raise RuntimeError(
                    f"{split}/{map_name}: need one seed per drivable tile "
                    f"({len(tiles)}), got {len(seeds)}"
                )
            for episode, (seed, tile) in enumerate(zip(seeds, tiles)):
                rows.extend(_episode(split, map_name, episode, seed, tile))

    calibration = [row for row in rows if row["split"] == "calibration" and row["detected"]]
    matrix, offset, retained = _fit_map_balanced_affine(calibration)
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

    residuals = _errors(rows, split="calibration")
    metrics = {
        "purpose": "C1 bidirectional lane calibration; runtime remains RGB-only",
        "source_config": str(SOURCE_CONFIG.relative_to(ROOT)),
        "steps_per_episode": STEPS,
        "capture_stride": CAPTURE_STRIDE,
        "split_unit": "seed/trajectory",
        "splits": {
            split: {name: list(seeds) for name, seeds in maps.items()}
            for split, maps in SPLITS.items()
        },
        "seed_overlap": sorted(
            set(value for seeds in SPLITS["calibration"].values() for value in seeds)
            & set(value for seeds in SPLITS["development"].values() for value in seeds)
        ),
        "fit": {
            "type": "robust_map_and_turn_balanced_affine",
            "input_order": ["raw_d_m", "raw_phi_rad", "raw_kappa_inv_m"],
            "output_order": ["d_m", "phi_rad", "kappa_inv_m"],
            "matrix": matrix.tolist(),
            "offset": offset.tolist(),
            "retained": retained,
            "residual_sigma": {
                "lateral_m": float(np.std(residuals[:, 0], ddof=1)),
                "heading_rad": float(np.std(residuals[:, 1], ddof=1)),
                "curvature_inv_m": float(np.std(residuals[:, 2], ddof=1)),
            },
        },
        "calibration": _split_metrics(rows, "calibration"),
        "development": _split_metrics(rows, "development"),
        "runtime_inputs": ["front_rgb", "fixed_camera_calibration"],
        "privileged_use": "offline fit/evaluation target after image inference",
        "pre_registered_gate": {
            "minimum_detection_rate": 0.80,
            "maximum_lateral_rmse_m": 0.055,
            "maximum_heading_rmse_rad": 0.160,
            "maximum_right_heading_rmse_rad": 0.220,
            "must_improve_right_heading_rmse": True,
        },
    }
    dev = metrics["development"]
    right = dev["by_turn"]["right"]
    metrics["gate_pass"] = bool(
        dev["detection_rate"] >= 0.80
        and dev["calibrated"]["lateral"]["rmse"] <= 0.055
        and dev["calibrated"]["heading"]["rmse"] <= 0.160
        and right["calibrated"]["heading"]["rmse"] <= 0.220
        and right["calibrated"]["heading"]["rmse"]
        < right["raw"]["heading"]["rmse"]
    )

    args.output.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    metrics_path.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metrics, indent=2))


def _episode(split: str, map_name: str, episode: int, seed: int, tile):
    integration = create_gym_duckietown(
        GymDuckietownConfig(
            map_name=map_name,
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
            start_pose=(
                (tile.base_local_x_m, 0.0, tile.base_local_z_m),
                tile.base_heading_rad,
            ),
        )
    )
    rows: list[dict[str, object]] = []
    try:
        observation = integration.agent.reset(seed=seed)
        estimator = CameraLaneMeasurementEstimator(
            CalibratedGroundProjector(integration.camera_calibration.read()),
            load_lane_perception_config(SOURCE_CONFIG),
        )
        for step in range(STEPS):
            if step % CAPTURE_STRIDE == 0:
                measurement, diagnostics = estimator.estimate_with_diagnostics(
                    observation.front_rgb
                )
                truth = integration.privileged.read().true_pomdp_state
                rows.append(
                    {
                        "split": split,
                        "map": map_name,
                        "episode": episode,
                        "seed": seed,
                        "step": step,
                        "start_tile_i": tile.coords[0],
                        "start_tile_j": tile.coords[1],
                        "start_tile_kind": tile.kind,
                        "turn_family": _turn_family(truth.road.curvature_inv_m),
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
            yaw = float(
                np.clip(
                    5.0 * observation.ego.lateral_error_m
                    + 3.0 * observation.ego.heading_error_rad,
                    -4.0,
                    4.0,
                )
            )
            transition = integration.agent.step(PolicyAction(0.16, yaw))
            observation = transition.observation
            if transition.terminated or transition.truncated:
                break
    finally:
        integration.close()
    return rows


def _fit_map_balanced_affine(rows):
    x = np.asarray([_raw_vector(row) for row in rows], dtype=float)
    y = np.asarray([_truth_vector(row) for row in rows], dtype=float)
    design = np.column_stack((x, np.ones(x.shape[0])))
    groups = [(str(row["map"]), str(row["turn_family"])) for row in rows]
    counts = Counter(groups)
    base_weights = np.asarray([1.0 / counts[group] for group in groups], dtype=float)
    base_weights *= len(base_weights) / np.sum(base_weights)
    keep = np.ones(x.shape[0], dtype=bool)
    coefficients = np.zeros((4, 3), dtype=float)
    for _ in range(8):
        weights = np.sqrt(base_weights[keep])[:, None]
        coefficients = np.linalg.lstsq(
            design[keep] * weights,
            y[keep] * weights,
            rcond=None,
        )[0]
        residuals = y - design @ coefficients
        median = np.median(residuals[keep], axis=0)
        mad = np.median(np.abs(residuals[keep] - median), axis=0)
        scale = np.maximum(1.4826 * mad, np.asarray((0.005, 0.015, 0.10)))
        updated = np.all(np.abs(residuals - median) <= 3.5 * scale, axis=1)
        if np.array_equal(updated, keep):
            break
        keep = updated
    weights = np.sqrt(base_weights[keep])[:, None]
    coefficients = np.linalg.lstsq(
        design[keep] * weights,
        y[keep] * weights,
        rcond=None,
    )[0]
    return coefficients[:3].T, coefficients[3], int(np.count_nonzero(keep))


def _raw_vector(row):
    return np.asarray(
        [float(row["raw_d"]), float(row["raw_phi"]), float(row["raw_kappa"])],
        dtype=float,
    )


def _truth_vector(row):
    return np.asarray(
        [float(row["gt_d"]), float(row["gt_phi"]), float(row["gt_kappa"])],
        dtype=float,
    )


def _errors(rows, *, split: str):
    selected = [row for row in rows if row["split"] == split and row["detected"]]
    return np.asarray(
        [
            [
                float(row["calibrated_d"]) - float(row["gt_d"]),
                _wrap(float(row["calibrated_phi"]) - float(row["gt_phi"])),
                float(row["calibrated_kappa"]) - float(row["gt_kappa"]),
            ]
            for row in selected
        ],
        dtype=float,
    )


def _metric(values):
    return {
        "n": int(values.size),
        "bias": float(np.mean(values)),
        "mae": float(np.mean(np.abs(values))),
        "rmse": float(np.sqrt(np.mean(np.square(values)))),
    }


def _metrics_for(rows):
    detected = [row for row in rows if row["detected"]]
    raw = np.asarray([_raw_vector(row) - _truth_vector(row) for row in detected])
    calibrated = np.asarray(
        [
            [
                float(row["calibrated_d"]) - float(row["gt_d"]),
                _wrap(float(row["calibrated_phi"]) - float(row["gt_phi"])),
                float(row["calibrated_kappa"]) - float(row["gt_kappa"]),
            ]
            for row in detected
        ]
    )
    def channels(values):
        return {
            "lateral": _metric(values[:, 0]),
            "heading": _metric(values[:, 1]),
            "curvature": _metric(values[:, 2]),
        }
    return {
        "rows": len(rows),
        "detections": len(detected),
        "detection_rate": len(detected) / len(rows) if rows else 0.0,
        "raw": channels(raw),
        "calibrated": channels(calibrated),
    }


def _split_metrics(rows, split):
    selected = [row for row in rows if row["split"] == split]
    result = _metrics_for(selected)
    result["by_map"] = {
        map_name: _metrics_for([row for row in selected if row["map"] == map_name])
        for map_name in ("small_loop", "experiment_loop")
    }
    result["by_turn"] = {
        family: _metrics_for([row for row in selected if row["turn_family"] == family])
        for family in ("right", "straight", "left")
    }
    return result


def _turn_family(curvature: float) -> str:
    if curvature < -0.75:
        return "right"
    if curvature > 0.75:
        return "left"
    return "straight"


def _wrap(value: float) -> float:
    return float(np.arctan2(np.sin(value), np.cos(value)))


if __name__ == "__main__":
    main()
