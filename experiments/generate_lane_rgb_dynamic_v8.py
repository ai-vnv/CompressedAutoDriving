"""Generate trajectory-wide RGB lane supervision with recovery excursions."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image

from duckie_pomdp.adapters.gym_duckietown import GymDuckietownConfig, create_gym_duckietown
from duckie_pomdp.control.start_sampler import load_small_loop_tiles
from duckie_pomdp.domain.action import PolicyAction


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "datasets" / "lane_rgb_dynamic_v2"
SPLITS = {
    "train": {"small_loop": tuple(range(92_001, 92_009)), "experiment_loop": tuple(range(92_021, 92_029))},
    "development": {"small_loop": tuple(range(92_101, 92_105)), "experiment_loop": tuple(range(92_111, 92_115))},
    "final": {"small_loop": tuple(range(92_201, 92_205)), "experiment_loop": tuple(range(92_211, 92_215))},
}
REPLACEMENT_FINAL_SPLITS = {
    "final": {
        "small_loop": tuple(range(92_301, 92_305)),
        "experiment_loop": tuple(range(92_311, 92_315)),
    }
}
START_EXCITATIONS = (
    (-0.040, -0.25),
    (-0.040, 0.25),
    (0.040, -0.25),
    (0.040, 0.25),
    (-0.020, -0.10),
    (-0.020, 0.10),
    (0.020, -0.10),
    (0.020, 0.10),
)
STEPS = 720
CAPTURE_STRIDE = 4


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--replacement-final-only", action="store_true")
    parser.add_argument("--replacement-final-index", type=int, default=1)
    args = parser.parse_args()
    output = args.output.resolve()
    if args.replacement_final_only:
        if args.replacement_final_index <= 0:
            raise ValueError("replacement final index must be positive")
        seed_offset = 100 * (args.replacement_final_index - 1)
        splits = {
            split: {
                map_name: tuple(seed + seed_offset for seed in seeds)
                for map_name, seeds in maps.items()
            }
            for split, maps in REPLACEMENT_FINAL_SPLITS.items()
        }
    else:
        splits = SPLITS
    metadata_path = output / "metadata.csv"
    manifest_path = output / "manifest.json"
    if metadata_path.exists() or manifest_path.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    rows: list[dict[str, object]] = []
    integrations = {}
    try:
        for split, maps in splits.items():
            image_dir = output / "images" / split
            image_dir.mkdir(parents=True, exist_ok=True)
            for map_name, seeds in maps.items():
                tiles = load_small_loop_tiles(
                    map_name=map_name,
                    anchor_tile=(1, 0),
                    anchor_heading_rad=np.pi,
                )
                initial = _episode_config(map_name, seeds[0], tiles[0], 0.0, 0.0)
                if map_name not in integrations:
                    integrations[map_name] = create_gym_duckietown(initial)
                integration = integrations[map_name]
                for episode, seed in enumerate(seeds):
                    tile = tiles[episode % len(tiles)]
                    lateral, heading = START_EXCITATIONS[episode % len(START_EXCITATIONS)]
                    config = _episode_config(map_name, seed, tile, lateral, heading)
                    integration.reconfigure_native_episode(config)
                    observation = integration.agent.reset(seed=seed)
                    for step in range(STEPS):
                        truth = integration.privileged.read().true_pomdp_state
                        if step % CAPTURE_STRIDE == 0:
                            image_id = f"{split}_{map_name}_e{episode:02d}_s{seed}_f{step:04d}"
                            relative = Path("images") / split / f"{image_id}.png"
                            Image.fromarray(observation.front_rgb, mode="RGB").save(output / relative)
                            rows.append(
                                {
                                    "image_id": image_id,
                                    "image_path": relative.as_posix(),
                                    "split": split,
                                    "map": map_name,
                                    "episode": episode,
                                    "seed": seed,
                                    "frame": step,
                                    "start_tile_i": tile.coords[0],
                                    "start_tile_j": tile.coords[1],
                                    "start_lateral_offset_m": lateral,
                                    "start_heading_offset_rad": heading,
                                    "gt_lateral_error_m": observation.ego.lateral_error_m,
                                    "gt_heading_error_rad": observation.ego.heading_error_rad,
                                    "gt_curvature_inv_m": truth.road.curvature_inv_m,
                                    "width_px": observation.front_rgb.shape[1],
                                    "height_px": observation.front_rgb.shape[0],
                                }
                            )
                        # Offline expert: privileged lane pose supplies the
                        # supervision trajectory only, never runtime inference.
                        d = observation.ego.lateral_error_m
                        phi = observation.ego.heading_error_rad
                        disturbance = 0.15 * math.sin(0.071 * step + episode) if split == "train" else 0.0
                        omega = float(np.clip(5.0 * d + 3.0 * phi + disturbance, -4.0, 4.0))
                        transition = integration.agent.step(PolicyAction(0.16, omega))
                        observation = transition.observation
                        if transition.terminated or transition.truncated:
                            break
    finally:
        for integration in integrations.values():
            integration.close()

    output.mkdir(parents=True, exist_ok=True)
    with metadata_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    seed_sets = {
        split: {seed for seeds in maps.values() for seed in seeds}
        for split, maps in splits.items()
    }
    overlaps = {
        f"{first}_{second}": sorted(seed_sets[first] & seed_sets[second])
        for index, first in enumerate(splits)
        for second in tuple(splits)[index + 1 :]
    }
    manifest = {
        "schema_version": 2,
        "dataset": output.name,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "runtime_input": "front_rgb_only",
        "privileged_use": "offline expert trajectory and d/phi/k labels only",
        "direction": "counter-clockwise",
        "steps_per_episode_maximum": STEPS,
        "capture_stride": CAPTURE_STRIDE,
        "seed_sets": {split: {name: list(values) for name, values in maps.items()} for split, maps in splits.items()},
        "seed_overlaps": overlaps,
        "counts": {split: sum(row["split"] == split for row in rows) for split in splits},
        "metadata_sha256": hashlib.sha256(metadata_path.read_bytes()).hexdigest(),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


def _episode_config(map_name, seed, tile, lateral, heading):
    forward_x = math.cos(tile.base_heading_rad)
    forward_z = -math.sin(tile.base_heading_rad)
    left_x, left_z = -forward_z, forward_x
    return GymDuckietownConfig(
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
            (
                tile.base_local_x_m + lateral * left_x,
                0.0,
                tile.base_local_z_m + lateral * left_z,
            ),
            tile.base_heading_rad + heading,
        ),
    )


if __name__ == "__main__":
    main()
