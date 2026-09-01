"""Real-simulator, evaluation-only comparison of lane-boundary extraction modes.

Both estimators receive exactly the same front RGB frame.  Simulator lane pose
is read only after both estimates are complete and is used solely for this
offline diagnostic.  The artifact is not a training or calibration set.
"""

from __future__ import annotations

import csv
import json
from dataclasses import replace
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
CONFIG = ROOT / "configs" / "lane_belief_v1.toml"
OUTPUT_DIR = ROOT / "artifacts" / "f10_ppo_visual_v4_codex" / "lane_boundary_diagnostic"
SEEDS = tuple(range(61001, 61009))
STEPS = 90
CAPTURE_STRIDE = 3


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    tiles = load_small_loop_tiles(
        map_name="small_loop", anchor_tile=(1, 0), anchor_heading_rad=np.pi
    )
    raw_config = load_lane_perception_config(CONFIG)
    for episode_index, seed in enumerate(SEEDS):
        tile = tiles[episode_index % len(tiles)]
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
                start_pose=(
                    (tile.base_local_x_m, 0.0, tile.base_local_z_m),
                    tile.base_heading_rad,
                ),
            )
        )
        try:
            observation = integration.agent.reset(seed=seed)
            projector = CalibratedGroundProjector(integration.camera_calibration.read())
            legacy = CameraLaneMeasurementEstimator(projector, raw_config)
            fusion = CameraLaneMeasurementEstimator(
                projector, replace(raw_config, dual_boundary_fusion_enabled=True)
            )
            for step in range(STEPS):
                if step % CAPTURE_STRIDE == 0:
                    legacy_m, legacy_d = legacy.estimate_with_diagnostics(observation.front_rgb)
                    fusion_m, fusion_d = fusion.estimate_with_diagnostics(observation.front_rgb)
                    # Runtime inference above is complete before evaluation truth is attached.
                    rows.extend(
                        _rows_for_frame(
                            episode_index,
                            seed,
                            step,
                            tile.coords,
                            observation,
                            legacy_m,
                            legacy_d,
                            fusion_m,
                            fusion_d,
                        )
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

    csv_path = OUTPUT_DIR / "comparison.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    metrics = {
        "purpose": "diagnostic only; never used for fitting or PPO evaluation",
        "direction": "counter-clockwise",
        "seeds": list(SEEDS),
        "split_unit": "seed/trajectory",
        "runtime_inputs": ["front_rgb", "fixed_camera_calibration"],
        "privileged_use": "offline evaluation target after both RGB estimators",
        "modes": {mode: _metrics(rows, mode) for mode in ("legacy", "dual_boundary")},
        "source_counts": _source_counts(rows),
    }
    (OUTPUT_DIR / "metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(metrics, indent=2))


def _rows_for_frame(episode, seed, step, tile, observation, legacy_m, legacy_d, fusion_m, fusion_d):
    base = {
        "episode": episode,
        "seed": seed,
        "step": step,
        "tile_i": tile[0],
        "tile_j": tile[1],
        "gt_lateral_error_m": observation.ego.lateral_error_m,
        "gt_heading_error_rad": observation.ego.heading_error_rad,
    }
    result = []
    for mode, measurement, diagnostics in (
        ("legacy", legacy_m, legacy_d),
        ("dual_boundary", fusion_m, fusion_d),
    ):
        result.append(
            {
                **base,
                "mode": mode,
                "detected": measurement.detected,
                "lateral_error_m": "" if measurement.lateral_error_m is None else measurement.lateral_error_m,
                "heading_error_rad": "" if measurement.heading_error_rad is None else measurement.heading_error_rad,
                "source": diagnostics.source,
                "strict_yellow_pixels": diagnostics.strict_yellow_pixel_count,
                "strict_white_pixels": diagnostics.strict_white_pixel_count,
                "adaptive_unknown_pixels": diagnostics.adaptive_unknown_pixel_count,
                "yellow_center_points": diagnostics.yellow_center_point_count,
                "white_center_points": diagnostics.white_center_point_count,
                "boundary_disagreement_m": "" if diagnostics.boundary_disagreement_m is None else diagnostics.boundary_disagreement_m,
            }
        )
    return result


def _metrics(rows, mode):
    subset = [row for row in rows if row["mode"] == mode]
    detected = [row for row in subset if row["detected"]]
    result = {"n": len(subset), "detections": len(detected), "detection_rate": len(detected) / len(subset)}
    for name, gt in (("lateral_error_m", "gt_lateral_error_m"), ("heading_error_rad", "gt_heading_error_rad")):
        errors = np.asarray([float(row[name]) - float(row[gt]) for row in detected])
        result[name] = {
            "bias": float(np.mean(errors)),
            "mae": float(np.mean(np.abs(errors))),
            "rmse": float(np.sqrt(np.mean(np.square(errors)))),
        }
    return result


def _source_counts(rows):
    result: dict[str, int] = {}
    for row in rows:
        key = f"{row['mode']}:{row['source']}"
        result[key] = result.get(key, 0) + 1
    return result


if __name__ == "__main__":
    main()
