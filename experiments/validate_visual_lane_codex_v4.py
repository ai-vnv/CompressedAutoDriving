"""Sequential real-simulator lane-belief development/final validation."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from duckie_pomdp.adapters.gym_duckietown import GymDuckietownConfig, create_gym_duckietown
from duckie_pomdp.control.lane_belief_runtime import VisualLaneBeliefRuntime
from duckie_pomdp.control.lane_belief_uncertainty import load_lane_uncertainty_calibration
from duckie_pomdp.control.start_sampler import load_small_loop_tiles
from duckie_pomdp.domain.action import PolicyAction
from duckie_pomdp.perception.camera_geometry import CalibratedGroundProjector


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "lane_belief_v3_codex.toml"
OUTPUT_DIR = ROOT / "artifacts" / "f10_ppo_visual_v4_codex" / "lane_belief_gate"
SEEDS = {
    "development": tuple(range(61201, 61209)),
    "final": tuple(range(61301, 61309)),
}
STEPS = 150
DT_S = 1.0 / 30.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=tuple(SEEDS), required=True)
    args = parser.parse_args()
    split = str(args.split)
    csv_path = OUTPUT_DIR / f"{split}.csv"
    metrics_path = OUTPUT_DIR / f"{split}_metrics.json"
    if csv_path.exists() or metrics_path.exists():
        raise RuntimeError(f"{split} lane-belief artifacts already exist; refusing overwrite")
    tiles = load_small_loop_tiles(
        map_name="small_loop", anchor_tile=(1, 0), anchor_heading_rad=np.pi
    )
    rows = []
    for episode_index, seed in enumerate(SEEDS[split]):
        rows.extend(_episode(split, episode_index, seed, tiles[episode_index % len(tiles)]))
    metrics = _metrics(split, rows)
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
        runtime = VisualLaneBeliefRuntime(
            CalibratedGroundProjector(integration.camera_calibration.read()),
            config_path=CONFIG,
            uncertainty_calibration=load_lane_uncertainty_calibration(CONFIG),
        )
        lane_step = runtime.reset(observation.front_rgb)
        for step in range(STEPS):
            measurement = lane_step.measurement
            belief = lane_step.belief
            diagnostics = lane_step.boundary_diagnostics
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
                    "source": "" if diagnostics is None else diagnostics.source,
                    "gt_d": observation.ego.lateral_error_m,
                    "gt_phi": observation.ego.heading_error_rad,
                    "gt_kappa": truth.road.curvature_inv_m,
                    "belief_validity": belief.validity_probability,
                    "belief_d": belief.lateral_error_mean_m,
                    "belief_d_std": belief.lateral_error_std_m,
                    "belief_phi": belief.heading_error_mean_rad,
                    "belief_phi_std": belief.heading_error_std_rad,
                    "belief_kappa": belief.curvature_mean_inv_m,
                    "belief_kappa_std": belief.curvature_std_inv_m,
                }
            )
            yaw = float(np.clip(5.0 * observation.ego.lateral_error_m + 3.0 * observation.ego.heading_error_rad, -4.0, 4.0))
            transition = integration.agent.step(PolicyAction(0.16, yaw))
            observation = transition.observation
            if transition.terminated or transition.truncated:
                break
            lane_step = runtime.update(
                observation.front_rgb,
                actual_ego_motion=observation.ego.motion,
                dt_s=DT_S,
            )
    finally:
        integration.close()
    return rows


def _channel(rows, mean_name, std_name, gt_name):
    errors = np.asarray([float(row[mean_name]) - float(row[gt_name]) for row in rows])
    sigmas = np.asarray([float(row[std_name]) for row in rows])
    abs_z = np.abs(errors) / sigmas
    return {
        "n": int(errors.size),
        "bias": float(np.mean(errors)),
        "mae": float(np.mean(np.abs(errors))),
        "rmse": float(np.sqrt(np.mean(np.square(errors)))),
        "mean_predicted_std": float(np.mean(sigmas)),
        "coverage_68": float(np.mean(abs_z <= 1.0)),
        "coverage_95": float(np.mean(abs_z <= 1.959963984540054)),
        "development_scale_for_68": float(max(1.0, np.percentile(abs_z, 68.0))),
    }


def _metrics(split, rows):
    detected = sum(bool(row["detected"]) for row in rows)
    lateral = _channel(rows, "belief_d", "belief_d_std", "gt_d")
    heading = _channel(rows, "belief_phi", "belief_phi_std", "gt_phi")
    curvature = _channel(rows, "belief_kappa", "belief_kappa_std", "gt_kappa")
    gate = {
        "minimum_detection_rate": 0.80,
        "maximum_lateral_rmse_m": 0.050,
        "maximum_heading_rmse_rad": 0.150,
        "lateral_coverage_68_band": [0.50, 0.85],
        "heading_coverage_68_band": [0.50, 0.85],
        "lateral_coverage_95_band": [0.85, 1.00],
        "heading_coverage_95_band": [0.85, 1.00],
    }
    result = {
        "gate": f"Codex dual-boundary lane-belief {split}",
        "direction": "counter-clockwise",
        "map": "small_loop",
        "split": split,
        "seeds": list(SEEDS[split]),
        "seed_role": "development uncertainty fitting" if split == "development" else "once-only held-out gate",
        "config": str(CONFIG.relative_to(ROOT)),
        "config_sha256": hashlib.sha256(CONFIG.read_bytes()).hexdigest(),
        "rows": len(rows),
        "detections": detected,
        "detection_rate": detected / len(rows),
        "lateral": lateral,
        "heading": heading,
        "curvature": curvature,
        "runtime_inputs": ["front_rgb", "actual_linear_velocity", "actual_yaw_rate", "dt"],
        "privileged_use": "evaluation target after lane belief output",
        "pre_registered_gate": gate,
    }
    result["gate_pass"] = bool(
        result["detection_rate"] >= gate["minimum_detection_rate"]
        and lateral["rmse"] <= gate["maximum_lateral_rmse_m"]
        and heading["rmse"] <= gate["maximum_heading_rmse_rad"]
        and gate["lateral_coverage_68_band"][0] <= lateral["coverage_68"] <= gate["lateral_coverage_68_band"][1]
        and gate["heading_coverage_68_band"][0] <= heading["coverage_68"] <= gate["heading_coverage_68_band"][1]
        and gate["lateral_coverage_95_band"][0] <= lateral["coverage_95"] <= gate["lateral_coverage_95_band"][1]
        and gate["heading_coverage_95_band"][0] <= heading["coverage_95"] <= gate["heading_coverage_95_band"][1]
    )
    return result


if __name__ == "__main__":
    main()
