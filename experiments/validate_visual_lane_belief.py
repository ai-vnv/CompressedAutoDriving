"""Held-out real-simulator validation for RGB -> lane belief.

The manual controller uses simulator lane pose only to keep the validation car
on a repeatable CCW trajectory.  The lane runtime under test receives RGB and
actual chassis motion only.  Privileged values are joined after each runtime
step for evaluation.
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
from duckie_pomdp.control.lane_belief_runtime import VisualLaneBeliefRuntime
from duckie_pomdp.domain.action import PolicyAction
from duckie_pomdp.perception.camera_geometry import CalibratedGroundProjector


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "lane_belief_v1.toml"
OUTPUT_DIR = ROOT / "artifacts" / "visual_lane"
CSV_PATH = OUTPUT_DIR / "lane_belief_final_validation.csv"
METRICS_PATH = OUTPUT_DIR / "lane_belief_final_validation_metrics.json"
SEEDS = (36501, 36502)
MAPS = ("small_loop", "experiment_loop")
STEPS_PER_EPISODE = 600
DT_S = 1.0 / 30.0


def main() -> None:
    if CSV_PATH.exists() or METRICS_PATH.exists():
        raise RuntimeError(
            "visual-lane final validation is once-only; existing artifacts refuse overwrite"
        )
    import gym_duckietown.simulator as simulator_module

    graphics_information = simulator_module.get_graphics_information()
    simulator_module.get_graphics_information = lambda: graphics_information
    rows: list[dict[str, object]] = []
    for map_name in MAPS:
        for seed in SEEDS:
            rows.extend(_episode(map_name, seed))

    detected = [row for row in rows if row["lane_detected"]]
    metrics = {
        "gate": "visual lane belief held-out real-simulator validation",
        "direction": "counter-clockwise",
        "maps": list(MAPS),
        "seeds": list(SEEDS),
        "seed_role": "once-only final visual-lane gate",
        "steps_per_episode": STEPS_PER_EPISODE,
        "rows": len(rows),
        "detections": len(detected),
        "detection_rate": len(detected) / len(rows),
        "config": str(CONFIG.relative_to(ROOT)),
        "config_sha256": _sha256(CONFIG),
        "runtime_inputs": ["front_rgb", "actual_linear_velocity", "actual_yaw_rate", "dt"],
        "driver_truth_use": "evaluation trajectory generation only",
        "measurement": _measurement_metrics(detected),
        "belief": _belief_metrics(rows),
        "by_map": {
            map_name: {
                "n": sum(row["map"] == map_name for row in rows),
                "detection_rate": np.mean(
                    [bool(row["lane_detected"]) for row in rows if row["map"] == map_name]
                ),
                **_belief_metrics([row for row in rows if row["map"] == map_name]),
            }
            for map_name in MAPS
        },
        "pre_registered_gate": {
            "minimum_detection_rate": 0.80,
            "maximum_lateral_belief_rmse_m": 0.05,
            "maximum_heading_belief_rmse_rad": 0.15,
        },
    }
    metrics["gate_pass"] = bool(
        metrics["detection_rate"] >= 0.80
        and metrics["belief"]["lateral"]["rmse"] <= 0.05
        and metrics["belief"]["heading"]["rmse"] <= 0.15
    )
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with CSV_PATH.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    METRICS_PATH.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metrics, indent=2))


def _episode(map_name: str, seed: int) -> list[dict[str, object]]:
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
        runtime = VisualLaneBeliefRuntime(
            CalibratedGroundProjector(integration.camera_calibration.read()),
            config_path=CONFIG,
        )
        lane_step = runtime.reset(observation.front_rgb)
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
            lane_step = runtime.update(
                observation.front_rgb,
                actual_ego_motion=observation.ego.motion,
                dt_s=DT_S,
            )
            # The runtime output is complete before evaluation truth is read.
            privileged = integration.privileged.read()
            truth = privileged.true_pomdp_state
            measurement = lane_step.measurement
            belief = lane_step.belief
            rows.append(
                {
                    "map": map_name,
                    "direction": "counter-clockwise",
                    "seed": seed,
                    "step": step,
                    "lane_detected": measurement.detected,
                    "visible_point_count": measurement.visible_point_count,
                    "fit_residual_m": _optional(measurement.fit_residual_m),
                    "gt_lateral_error_m": observation.ego.lateral_error_m,
                    "gt_heading_error_rad": observation.ego.heading_error_rad,
                    "gt_curvature_inv_m": _optional(truth.road.curvature_inv_m),
                    "measurement_lateral_error_m": _optional(measurement.lateral_error_m),
                    "measurement_heading_error_rad": _optional(measurement.heading_error_rad),
                    "measurement_curvature_inv_m": _optional(measurement.curvature_inv_m),
                    "lane_validity_probability": belief.validity_probability,
                    "belief_lateral_mean_m": belief.lateral_error_mean_m,
                    "belief_lateral_std_m": belief.lateral_error_std_m,
                    "belief_heading_mean_rad": belief.heading_error_mean_rad,
                    "belief_heading_std_rad": belief.heading_error_std_rad,
                    "belief_curvature_mean_inv_m": belief.curvature_mean_inv_m,
                    "belief_curvature_std_inv_m": belief.curvature_std_inv_m,
                    "actual_linear_velocity_mps": observation.ego.linear_velocity_mps,
                    "actual_yaw_rate_rad_s": observation.ego.yaw_rate_rad_s,
                    "evaluation_yaw_command_rad_s": yaw_command,
                    "terminated": transition.terminated,
                    "truncated": transition.truncated,
                }
            )
            if transition.terminated or transition.truncated:
                break
    finally:
        integration.close()
    return rows


def _measurement_metrics(rows: list[dict[str, object]]) -> dict[str, object]:
    return {
        "lateral": _metric(rows, "measurement_lateral_error_m", "gt_lateral_error_m"),
        "heading": _metric(rows, "measurement_heading_error_rad", "gt_heading_error_rad"),
        "curvature": _metric(rows, "measurement_curvature_inv_m", "gt_curvature_inv_m"),
    }


def _belief_metrics(rows: list[dict[str, object]]) -> dict[str, object]:
    return {
        "lateral": _metric_with_coverage(
            rows,
            "belief_lateral_mean_m",
            "belief_lateral_std_m",
            "gt_lateral_error_m",
        ),
        "heading": _metric_with_coverage(
            rows,
            "belief_heading_mean_rad",
            "belief_heading_std_rad",
            "gt_heading_error_rad",
        ),
        "curvature": _metric_with_coverage(
            rows,
            "belief_curvature_mean_inv_m",
            "belief_curvature_std_inv_m",
            "gt_curvature_inv_m",
        ),
        "mean_validity_probability": float(
            np.mean([float(row["lane_validity_probability"]) for row in rows])
        ),
    }


def _metric(rows: list[dict[str, object]], mean: str, truth: str) -> dict[str, float]:
    errors = np.asarray([float(row[mean]) - float(row[truth]) for row in rows])
    return {
        "n": int(errors.size),
        "bias": float(np.mean(errors)),
        "mae": float(np.mean(np.abs(errors))),
        "rmse": float(np.sqrt(np.mean(np.square(errors)))),
        "residual_sd": float(np.std(errors, ddof=1)) if errors.size > 1 else 0.0,
    }


def _metric_with_coverage(
    rows: list[dict[str, object]], mean: str, std: str, truth: str
) -> dict[str, float]:
    result = _metric(rows, mean, truth)
    errors = np.abs(np.asarray([float(row[mean]) - float(row[truth]) for row in rows]))
    sigmas = np.asarray([float(row[std]) for row in rows])
    result.update(
        mean_predicted_std=float(np.mean(sigmas)),
        coverage_68=float(np.mean(errors <= sigmas)),
        coverage_95=float(np.mean(errors <= 1.96 * sigmas)),
    )
    return result


def _optional(value: float | None) -> float | str:
    return "" if value is None else float(value)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


if __name__ == "__main__":
    main()
