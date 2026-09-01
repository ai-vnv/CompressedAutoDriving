"""Dynamic held-out validation for the bidirectional visual lane belief."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from duckie_pomdp.adapters.gym_duckietown import (
    GymDuckietownConfig,
    create_gym_duckietown,
)
from duckie_pomdp.control.lane_belief_runtime import VisualLaneBeliefRuntime
from duckie_pomdp.control.lane_belief_uncertainty import (
    load_lane_uncertainty_calibration,
)
from duckie_pomdp.control.start_sampler import load_small_loop_tiles
from duckie_pomdp.domain.action import PolicyAction
from duckie_pomdp.perception.camera_geometry import CalibratedGroundProjector


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "lane_belief_v4_transfer.toml"
OUTPUT = ROOT / "artifacts" / "f10_ppo_visual_v5_c1" / "lane_belief_gate"
STEPS = 60
SPLITS = {
    "development": {
        "small_loop": tuple(range(64301, 64309)),
        "experiment_loop": tuple(range(64321, 64333)),
    },
    "final": {
        "small_loop": tuple(range(64401, 64409)),
        "experiment_loop": tuple(range(64421, 64433)),
    },
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=tuple(SPLITS), required=True)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    csv_path = args.output / f"{args.split}.csv"
    metrics_path = args.output / f"{args.split}_metrics.json"
    if csv_path.exists() or metrics_path.exists():
        raise FileExistsError(f"refusing to overwrite {args.split} lane gate")

    rows: list[dict[str, object]] = []
    for map_name, seeds in SPLITS[args.split].items():
        tiles = load_small_loop_tiles(
            map_name=map_name,
            anchor_tile=(1, 0),
            anchor_heading_rad=np.pi,
        )
        if len(tiles) != len(seeds):
            raise RuntimeError(f"{map_name}: expected one seed per tile")
        for episode, (seed, tile) in enumerate(zip(seeds, tiles)):
            rows.extend(_episode(args.split, map_name, episode, seed, tile))

    metrics = _metrics(args.split, rows)
    args.output.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    metrics_path.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metrics, indent=2))
    if args.split == "final" and not metrics["gate_pass"]:
        raise SystemExit(1)


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
        runtime = VisualLaneBeliefRuntime(
            CalibratedGroundProjector(integration.camera_calibration.read()),
            config_path=CONFIG,
            uncertainty_calibration=(
                None
                if split == "development"
                else load_lane_uncertainty_calibration(CONFIG)
            ),
        )
        lane_step = runtime.reset(observation.front_rgb)
        for step in range(STEPS):
            belief = lane_step.belief
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
                    "detected": lane_step.measurement.detected,
                    "validity": belief.validity_probability,
                    "belief_d": belief.lateral_error_mean_m,
                    "belief_d_std": belief.lateral_error_std_m,
                    "belief_phi": belief.heading_error_mean_rad,
                    "belief_phi_std": belief.heading_error_std_rad,
                    "gt_d": observation.ego.lateral_error_m,
                    "gt_phi": observation.ego.heading_error_rad,
                    "gt_kappa": truth.road.curvature_inv_m,
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
            lane_step = runtime.update(
                observation.front_rgb,
                actual_ego_motion=observation.ego.motion,
                dt_s=1.0 / 30.0,
            )
    finally:
        integration.close()
    return rows


def _channel(rows, mean_name, std_name, gt_name):
    errors = np.asarray(
        [float(row[mean_name]) - float(row[gt_name]) for row in rows], dtype=float
    )
    sigmas = np.asarray([float(row[std_name]) for row in rows], dtype=float)
    z = np.abs(errors) / sigmas
    return {
        "n": int(errors.size),
        "bias": float(np.mean(errors)),
        "mae": float(np.mean(np.abs(errors))),
        "rmse": float(np.sqrt(np.mean(np.square(errors)))),
        "mean_predicted_std": float(np.mean(sigmas)),
        "coverage_68": float(np.mean(z <= 1.0)),
        "coverage_95": float(np.mean(z <= 1.959963984540054)),
        "development_scale_for_68": float(max(1.0, np.percentile(z, 68.0))),
    }


def _group(rows):
    return {
        "rows": len(rows),
        "detection_rate": float(np.mean([bool(row["detected"]) for row in rows])),
        "lateral": _channel(rows, "belief_d", "belief_d_std", "gt_d"),
        "heading": _channel(rows, "belief_phi", "belief_phi_std", "gt_phi"),
    }


def _metrics(split: str, rows):
    result = {
        "gate": f"bidirectional visual lane belief {split}",
        "direction": "counter-clockwise",
        "split": split,
        "seed_role": (
            "development uncertainty fitting"
            if split == "development"
            else "once-only held-out gate"
        ),
        "seeds": {
            name: list(seeds) for name, seeds in SPLITS[split].items()
        },
        "config": str(CONFIG.relative_to(ROOT)),
        "config_sha256": hashlib.sha256(CONFIG.read_bytes()).hexdigest(),
        **_group(rows),
        "by_map": {
            name: _group([row for row in rows if row["map"] == name])
            for name in ("small_loop", "experiment_loop")
        },
        "by_turn": {
            name: _group([row for row in rows if row["turn_family"] == name])
            for name in ("right", "straight", "left")
        },
        "runtime_inputs": ["front_rgb", "actual_linear_velocity", "actual_yaw_rate", "dt"],
        "privileged_use": "evaluation target after lane belief output",
        "pre_registered_gate": {
            "minimum_detection_rate": 0.80,
            "maximum_lateral_rmse_m": 0.055,
            "maximum_heading_rmse_rad": 0.160,
            "maximum_right_heading_rmse_rad": 0.220,
            "coverage_68_band": [0.50, 0.85],
            "coverage_95_band": [0.85, 1.00],
        },
    }
    gate = result["pre_registered_gate"]
    result["gate_pass"] = bool(
        split == "final"
        and result["detection_rate"] >= gate["minimum_detection_rate"]
        and result["lateral"]["rmse"] <= gate["maximum_lateral_rmse_m"]
        and result["heading"]["rmse"] <= gate["maximum_heading_rmse_rad"]
        and result["by_turn"]["right"]["heading"]["rmse"]
        <= gate["maximum_right_heading_rmse_rad"]
        and gate["coverage_68_band"][0]
        <= result["lateral"]["coverage_68"]
        <= gate["coverage_68_band"][1]
        and gate["coverage_68_band"][0]
        <= result["heading"]["coverage_68"]
        <= gate["coverage_68_band"][1]
        and gate["coverage_95_band"][0]
        <= result["lateral"]["coverage_95"]
        <= gate["coverage_95_band"][1]
        and gate["coverage_95_band"][0]
        <= result["heading"]["coverage_95"]
        <= gate["coverage_95_band"][1]
    )
    return result


def _turn_family(curvature: float) -> str:
    if curvature < -0.75:
        return "right"
    if curvature > 0.75:
        return "left"
    return "straight"


if __name__ == "__main__":
    main()
