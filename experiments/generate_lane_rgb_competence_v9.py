"""Generate camera-lane supervision at recovery poses on every loop tile.

Each RGB frame is a normal Gym-Duckietown render. Simulator lane pose is
written only as an offline label. The runtime lane estimator never receives
the pose, tile, or turn family.
"""

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
from duckie_pomdp.control.start_sampler import DrivableTile, load_small_loop_tiles


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "datasets" / "lane_rgb_competence_v9"
SPLIT_BASES = {"train": 100_000, "development": 102_000, "final": 104_000}
LONGITUDINAL_OFFSETS_M = (-0.18, 0.0, 0.18)
POSES = (
    ("centred", 0.0, 0.0),
    ("lateral_right_mid", -0.05, 0.0),
    ("lateral_left_mid", 0.05, 0.0),
    ("lateral_right_edge", -0.09, 0.0),
    ("lateral_left_edge", 0.09, 0.0),
    ("heading_right_mid", 0.0, -0.18),
    ("heading_left_mid", 0.0, 0.18),
    ("heading_right_large", 0.0, -0.35),
    ("heading_left_large", 0.0, 0.35),
    ("right_edge_heading_right", -0.09, -0.35),
    ("right_edge_heading_left", -0.09, 0.35),
    ("left_edge_heading_right", 0.09, -0.35),
    ("left_edge_heading_left", 0.09, 0.35),
    ("right_edge_recovery", -0.09, 0.18),
    ("left_edge_recovery", 0.09, -0.18),
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    output = args.output.resolve()
    metadata_path = output / "metadata.csv"
    manifest_path = output / "manifest.json"
    if metadata_path.exists() or manifest_path.exists():
        raise FileExistsError(f"refusing to overwrite competence dataset: {output}")

    rows: list[dict[str, object]] = []
    split_seeds: dict[str, set[int]] = {name: set() for name in SPLIT_BASES}
    integrations = {}
    try:
        for split, seed_base in SPLIT_BASES.items():
            next_seed = seed_base
            image_dir = output / "images" / split
            image_dir.mkdir(parents=True, exist_ok=True)
            for map_name in ("small_loop", "experiment_loop"):
                tiles = load_small_loop_tiles(
                    map_name=map_name,
                    anchor_tile=(1, 0),
                    anchor_heading_rad=np.pi,
                )
                if map_name not in integrations:
                    integrations[map_name] = create_gym_duckietown(
                        _config(map_name, seed_base, tiles[0], 0.0, 0.0, 0.0)
                    )
                integration = integrations[map_name]
                for tile_index, tile in enumerate(tiles):
                    repeats = _repeat_count(split, tile.kind)
                    for repeat in range(repeats):
                        for longitudinal in LONGITUDINAL_OFFSETS_M:
                            for pose_index, (pose_name, lateral, heading) in enumerate(POSES):
                                seed = next_seed
                                next_seed += 1
                                split_seeds[split].add(seed)
                                config = _config(
                                    map_name,
                                    seed,
                                    tile,
                                    longitudinal,
                                    lateral,
                                    heading,
                                )
                                _set_start(integration, tile.coords, config.start_pose)
                                observation = integration.agent.reset(seed=seed)
                                truth = integration.privileged.read().true_pomdp_state
                                image_id = (
                                    f"{split}_{map_name}_t{tile_index:02d}_r{repeat}_"
                                    f"l{longitudinal:+.2f}_p{pose_index:02d}_s{seed}"
                                )
                                relative = Path("images") / split / f"{image_id}.png"
                                Image.fromarray(observation.front_rgb, mode="RGB").save(
                                    output / relative
                                )
                                rows.append(
                                    {
                                        "image_id": image_id,
                                        "image_path": relative.as_posix(),
                                        "split": split,
                                        "map": map_name,
                                        "repeat": repeat,
                                        "seed": seed,
                                        "tile_i": tile.coords[0],
                                        "tile_j": tile.coords[1],
                                        "tile_kind": tile.kind,
                                        "turn_family": _turn_family(
                                            truth.road.curvature_inv_m
                                        ),
                                        "longitudinal_offset_m": longitudinal,
                                        "pose_name": pose_name,
                                        "requested_lateral_offset_m": lateral,
                                        "requested_heading_offset_rad": heading,
                                        "gt_lateral_error_m": observation.ego.lateral_error_m,
                                        "gt_heading_error_rad": observation.ego.heading_error_rad,
                                        "gt_curvature_inv_m": truth.road.curvature_inv_m,
                                        "width_px": observation.front_rgb.shape[1],
                                        "height_px": observation.front_rgb.shape[0],
                                    }
                                )
    finally:
        for integration in integrations.values():
            integration.close()

    with metadata_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    overlaps = {
        f"{left}_{right}": sorted(split_seeds[left] & split_seeds[right])
        for index, left in enumerate(SPLIT_BASES)
        for right in tuple(SPLIT_BASES)[index + 1 :]
    }
    manifest = {
        "schema_version": 1,
        "dataset": output.name,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "runtime_input": "front_rgb_only",
        "privileged_use": "offline d/phi/k labels after RGB capture",
        "direction": "counter-clockwise",
        "maps": ["small_loop", "experiment_loop"],
        "split_unit": "seed/map/tile/longitudinal/pose/repeat",
        "seed_ranges": {
            split: [min(values), max(values)] for split, values in split_seeds.items()
        },
        "seed_overlaps": overlaps,
        "longitudinal_offsets_m": list(LONGITUDINAL_OFFSETS_M),
        "poses": [
            {"name": name, "lateral_m": lateral, "heading_rad": heading}
            for name, lateral, heading in POSES
        ],
        "right_curve_repeat_policy": {"train": 4, "development": 2, "final": 2},
        "counts": {
            split: sum(row["split"] == split for row in rows)
            for split in SPLIT_BASES
        },
        "metadata_sha256": hashlib.sha256(metadata_path.read_bytes()).hexdigest(),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


def _repeat_count(split: str, tile_kind: str) -> int:
    if tile_kind == "curve_right":
        return 4 if split == "train" else 2
    return 1


def _config(
    map_name: str,
    seed: int,
    tile: DrivableTile,
    longitudinal_m: float,
    lateral_m: float,
    heading_rad: float,
) -> GymDuckietownConfig:
    forward_x = math.cos(tile.base_heading_rad)
    forward_z = -math.sin(tile.base_heading_rad)
    right_x, right_z = -forward_z, forward_x
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
                tile.base_local_x_m + longitudinal_m * forward_x + lateral_m * right_x,
                0.0,
                tile.base_local_z_m + longitudinal_m * forward_z + lateral_m * right_z,
            ),
            tile.base_heading_rad + heading_rad,
        ),
    )


def _set_start(integration, tile_coords, start_pose) -> None:
    simulator = integration.agent._session._simulator  # noqa: SLF001
    tile = simulator._get_tile(*tile_coords)  # noqa: SLF001
    if tile is None or not bool(tile.get("drivable", False)):
        raise RuntimeError(f"non-drivable dataset tile {tile_coords}")
    simulator.start_tile = tile
    simulator.start_pose = start_pose


def _turn_family(curvature: float) -> str:
    if curvature < -0.75:
        return "right"
    if curvature > 0.75:
        return "left"
    return "straight"


if __name__ == "__main__":
    main()
