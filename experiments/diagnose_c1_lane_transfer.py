"""Measure camera-lane belief transfer on the native experiment loop.

This is an offline diagnostic: the runtime estimator receives RGB and measured
ego motion only. Simulator lane state is read strictly after inference and is
used solely as the evaluation target.
"""

from __future__ import annotations

import argparse
import csv
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
LANE_CONFIG = ROOT / "configs" / "lane_belief_v3_codex.toml"
DEFAULT_OUTPUT = (
    ROOT
    / "artifacts"
    / "f10_ppo_visual_v4_codex"
    / "c1_diagnosis"
)
DT_S = 1.0 / 30.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--map", default="experiment_loop")
    parser.add_argument("--seed-start", type=int, default=63901)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--steps", type=int, default=150)
    parser.add_argument("--tile-offset", type=int, default=0)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    csv_path = args.output / "lane_transfer.csv"
    metrics_path = args.output / "lane_transfer_metrics.json"
    if csv_path.exists() or metrics_path.exists():
        raise FileExistsError(f"refusing to overwrite diagnostic artifacts in {args.output}")

    seeds = tuple(range(args.seed_start, args.seed_start + args.episodes))
    tiles = load_small_loop_tiles(
        map_name=args.map,
        anchor_tile=(1, 0),
        anchor_heading_rad=np.pi,
    )
    rows: list[dict[str, object]] = []
    for episode, seed in enumerate(seeds):
        rows.extend(
            _episode(
                args.map,
                episode,
                seed,
                tiles[(args.tile_offset + episode) % len(tiles)],
                args.steps,
            )
        )

    metrics = _metrics(args.map, seeds, rows)
    args.output.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    metrics_path.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metrics, indent=2))


def _episode(map_name: str, episode: int, seed: int, tile, maximum_steps: int):
    integration = create_gym_duckietown(
        GymDuckietownConfig(
            map_name=map_name,
            seed=seed,
            domain_randomization=True,
            dynamics_randomization=False,
            frame_rate_hz=30,
            frame_skip=1,
            maximum_steps=maximum_steps + 2,
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
            config_path=LANE_CONFIG,
            uncertainty_calibration=load_lane_uncertainty_calibration(LANE_CONFIG),
        )
        lane_step = runtime.reset(observation.front_rgb)
        for step in range(maximum_steps):
            belief = lane_step.belief
            measurement = lane_step.measurement
            boundary = lane_step.boundary_diagnostics
            raw_measurement, _ = runtime.estimator.estimate_with_diagnostics(
                observation.front_rgb
            )
            rows.append(
                {
                    "episode": episode,
                    "seed": seed,
                    "step": step,
                    "map": map_name,
                    "start_tile_i": tile.coords[0],
                    "start_tile_j": tile.coords[1],
                    "start_tile_kind": tile.kind,
                    "detected": measurement.detected,
                    "boundary_source": "" if boundary is None else boundary.source,
                    "yellow_pixels": (
                        0 if boundary is None else boundary.strict_yellow_pixel_count
                    ),
                    "white_pixels": (
                        0 if boundary is None else boundary.strict_white_pixel_count
                    ),
                    "adaptive_pixels": (
                        0 if boundary is None else boundary.adaptive_unknown_pixel_count
                    ),
                    "raw_detected": raw_measurement.detected,
                    "raw_d": (
                        "" if raw_measurement.lateral_error_m is None
                        else raw_measurement.lateral_error_m
                    ),
                    "raw_phi": (
                        "" if raw_measurement.heading_error_rad is None
                        else raw_measurement.heading_error_rad
                    ),
                    "raw_kappa": (
                        "" if raw_measurement.curvature_inv_m is None
                        else raw_measurement.curvature_inv_m
                    ),
                    "belief_validity": belief.validity_probability,
                    "belief_d": belief.lateral_error_mean_m,
                    "belief_d_std": belief.lateral_error_std_m,
                    "belief_phi": belief.heading_error_mean_rad,
                    "belief_phi_std": belief.heading_error_std_rad,
                    "gt_d": observation.ego.lateral_error_m,
                    "gt_phi": observation.ego.heading_error_rad,
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
                dt_s=DT_S,
            )
    finally:
        integration.close()
    return rows


def _metrics(map_name: str, seeds: tuple[int, ...], rows):
    d_error = np.asarray(
        [float(row["belief_d"]) - float(row["gt_d"]) for row in rows], dtype=float
    )
    phi_error = np.asarray(
        [
            np.arctan2(
                np.sin(float(row["belief_phi"]) - float(row["gt_phi"])),
                np.cos(float(row["belief_phi"]) - float(row["gt_phi"])),
            )
            for row in rows
        ],
        dtype=float,
    )
    return {
        "purpose": "C1 lane-perception transfer diagnosis; never policy input",
        "map": map_name,
        "seeds": list(seeds),
        "seed_role": "diagnostic only",
        "rows": len(rows),
        "detection_rate": float(np.mean([bool(row["detected"]) for row in rows])),
        "lateral_bias_m": float(np.mean(d_error)),
        "lateral_rmse_m": float(np.sqrt(np.mean(np.square(d_error)))),
        "heading_bias_rad": float(np.mean(phi_error)),
        "heading_rmse_rad": float(np.sqrt(np.mean(np.square(phi_error)))),
        "runtime_inputs": ["front_rgb", "actual_ego_motion", "dt_s"],
        "privileged_use": "offline target read only after lane-belief inference",
    }


if __name__ == "__main__":
    main()
