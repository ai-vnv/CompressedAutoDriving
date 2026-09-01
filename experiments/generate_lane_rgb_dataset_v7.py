"""Generate disjoint real-simulator RGB supervision for lane pose.

Simulator lane pose is an offline label only.  The saved model input is the
front RGB frame; no privileged channel is written into the image or accepted
by the runtime estimator.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image

from duckie_pomdp.adapters.gym_duckietown import create_gym_duckietown
from duckie_pomdp.control.start_sampler import load_small_loop_tiles

from calibrate_lane_pose_excited_v6 import POSES, _config, _turn_family


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "datasets" / "lane_rgb_v1"
SPLITS = {
    "train": {
        "small_loop": (80_000, 80_400),
        "experiment_loop": (81_000, 81_400),
    },
    "development": {"small_loop": (82_000,), "experiment_loop": (82_500,)},
    "final": {"small_loop": (83_000,), "experiment_loop": (83_500,)},
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    output = args.output.resolve()
    manifest_path = output / "manifest.json"
    metadata_path = output / "metadata.csv"
    if manifest_path.exists() or metadata_path.exists():
        raise FileExistsError(f"refusing to overwrite lane RGB dataset at {output}")

    rows: list[dict[str, object]] = []
    integrations = {}
    try:
        for split, maps in SPLITS.items():
            image_dir = output / "images" / split
            image_dir.mkdir(parents=True, exist_ok=True)
            for map_name, seed_bases in maps.items():
                tiles = load_small_loop_tiles(
                    map_name=map_name,
                    anchor_tile=(1, 0),
                    anchor_heading_rad=np.pi,
                )
                if map_name not in integrations:
                    integrations[map_name] = create_gym_duckietown(
                        _config(map_name, seed_bases[0], tiles[0], 0.0, 0.0)
                    )
                integration = integrations[map_name]
                for repeat, seed_base in enumerate(seed_bases):
                    for tile_index, tile in enumerate(tiles):
                        for pose_index, (pose_name, lateral, heading) in enumerate(POSES):
                            seed = seed_base + 20 * tile_index + pose_index
                            image_id = (
                                f"{split}_{map_name}_r{repeat}_t{tile.coords[0]}-"
                                f"{tile.coords[1]}_p{pose_index:02d}_s{seed}"
                            )
                            pose_config = _config(
                                map_name, seed, tile, lateral, heading
                            )
                            _set_start(integration, tile.coords, pose_config.start_pose)
                            observation = integration.agent.reset(seed=seed)
                            rgb = np.asarray(observation.front_rgb, dtype=np.uint8)
                            truth = integration.privileged.read().true_pomdp_state
                            relative_image = Path("images") / split / f"{image_id}.png"
                            Image.fromarray(rgb, mode="RGB").save(output / relative_image)
                            rows.append(
                            {
                                "image_id": image_id,
                                "image_path": relative_image.as_posix(),
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
                                "pose_name": pose_name,
                                "requested_lateral_offset_m": lateral,
                                "requested_heading_offset_rad": heading,
                                "gt_lateral_error_m": observation.ego.lateral_error_m,
                                "gt_heading_error_rad": observation.ego.heading_error_rad,
                                "gt_curvature_inv_m": truth.road.curvature_inv_m,
                                "width_px": rgb.shape[1],
                                "height_px": rgb.shape[0],
                            }
                            )
    finally:
        for integration in integrations.values():
            integration.close()

    output.mkdir(parents=True, exist_ok=True)
    with metadata_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    counts = {
        split: sum(row["split"] == split for row in rows) for split in SPLITS
    }
    seed_sets = {
        split: {int(row["seed"]) for row in rows if row["split"] == split}
        for split in SPLITS
    }
    overlaps = {
        f"{a}_{b}": sorted(seed_sets[a] & seed_sets[b])
        for index, a in enumerate(SPLITS)
        for b in tuple(SPLITS)[index + 1 :]
    }
    manifest = {
        "schema_version": 1,
        "dataset": "lane_rgb_v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "runtime_input": "front_rgb_only",
        "privileged_use": "offline d/phi/k labels after RGB capture",
        "direction": "counter-clockwise",
        "maps": ["small_loop", "experiment_loop"],
        "image_resolution": [640, 480],
        "split_unit": "seed/map/tile/pose/repeat",
        "seed_bases": SPLITS,
        "seed_overlaps": overlaps,
        "counts": counts,
        "pose_count": len(POSES),
        "metadata_sha256": hashlib.sha256(metadata_path.read_bytes()).hexdigest(),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


def _set_start(integration, tile_coords, start_pose) -> None:
    """Offline generator hook: reuse one GL context while varying reset pose."""

    session = integration.agent._session
    simulator = session._simulator
    tile = simulator._get_tile(*tile_coords)
    if tile is None or not bool(tile.get("drivable", False)):
        raise RuntimeError(f"non-drivable dataset tile {tile_coords}")
    simulator.start_tile = tile
    simulator.start_pose = start_pose


if __name__ == "__main__":
    main()
