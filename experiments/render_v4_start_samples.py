"""F10-PPO v4 Task 2, Step 5: sanity-render sampled loop-wide start poses.

Renders one top-down frame (whole small_loop, car marker at its sampled
pose) and one front-camera frame per sampled start, so the loop-wide
distribution can be checked *visually*: every sample should be on-road and
facing the counter-clockwise loop direction, and the set of 20 samples
should visibly spread around all four sides of the loop, not cluster on
v3's single tile.

No training, no policy, no PPO config -- this only exercises
``LoopStartSampler``/``load_small_loop_tiles`` and the real simulator's
rendering, both already covered by ``tests/test_v4_start_sampler.py``.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import cv2
import numpy as np

from duckie_pomdp.adapters.gym_duckietown import (
    GymDuckietownConfig,
    create_gym_duckietown,
)
from duckie_pomdp.control.start_sampler import LoopStartSampler, load_small_loop_tiles

ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "artifacts" / "f10_ppo_visual_v4" / "start_samples"

ANCHOR_TILE = (1, 0)
ANCHOR_HEADING_RAD = math.pi
SAMPLE_COUNT = 20
RNG_SEED = 47201  # v4 stage_final seed for c0 -- arbitrary but fixed/traceable


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    tiles = load_small_loop_tiles(
        map_name="small_loop",
        anchor_tile=ANCHOR_TILE,
        anchor_heading_rad=ANCHOR_HEADING_RAD,
    )
    sampler = LoopStartSampler(tiles, rng_seed=RNG_SEED)

    manifest: list[dict[str, object]] = []
    tile_counts: dict[str, int] = {}
    for episode_index in range(SAMPLE_COUNT):
        pose = sampler.sample(episode_index)
        config = GymDuckietownConfig(
            map_name="small_loop",
            seed=1000 + episode_index,
            domain_randomization=False,
            dynamics_randomization=False,
            frame_rate_hz=30,
            frame_skip=1,
            maximum_steps=5,
            camera_width=640,
            camera_height=480,
            headless=True,
            start_tile=pose.tile,
            start_pose=((pose.local_x_m, 0.0, pose.local_z_m), pose.heading_rad),
        )
        integration = create_gym_duckietown(config)
        try:
            simulator = integration.agent._session._simulator
            top_down = np.asarray(simulator.render(mode="top_down")).copy()
            front_rgb = integration.agent.reset(seed=config.seed).front_rgb
        finally:
            integration.close()

        tile_key = f"{pose.tile[0]}_{pose.tile[1]}"
        tile_counts[tile_key] = tile_counts.get(tile_key, 0) + 1
        stem = f"{episode_index:02d}_tile{pose.tile[0]}{pose.tile[1]}"
        top_down_path = OUTPUT_DIR / f"{stem}_topdown.png"
        front_path = OUTPUT_DIR / f"{stem}_front.png"
        cv2.imwrite(str(top_down_path), cv2.cvtColor(top_down, cv2.COLOR_RGB2BGR))
        cv2.imwrite(str(front_path), cv2.cvtColor(np.asarray(front_rgb), cv2.COLOR_RGB2BGR))

        manifest.append(
            {
                "episode_index": episode_index,
                "tile": list(pose.tile),
                "local_x_m": pose.local_x_m,
                "local_z_m": pose.local_z_m,
                "heading_rad": pose.heading_rad,
                "top_down_png": top_down_path.name,
                "front_png": front_path.name,
            }
        )

    # Contact sheet of all 20 top-down frames, 5 columns x 4 rows.
    thumbnails = []
    for entry in manifest:
        image = cv2.imread(str(OUTPUT_DIR / entry["top_down_png"]))
        thumbnails.append(cv2.resize(image, (200, 200)))
    columns, rows = 5, math.ceil(len(thumbnails) / 5)
    canvas = np.full((rows * 200, columns * 200, 3), 40, dtype=np.uint8)
    for index, thumbnail in enumerate(thumbnails):
        row, column = divmod(index, columns)
        canvas[row * 200 : row * 200 + 200, column * 200 : column * 200 + 200] = thumbnail
        cv2.putText(
            canvas,
            f"ep{index} tile{manifest[index]['tile']}",
            (column * 200 + 4, row * 200 + 16),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (0, 255, 255),
            1,
            cv2.LINE_AA,
        )
    contact_sheet_path = OUTPUT_DIR / "contact_sheet.png"
    cv2.imwrite(str(contact_sheet_path), canvas)

    manifest_payload = {
        "schema_version": 1,
        "stage": "F10_PPO_VISUAL_V4_TASK2_START_SAMPLE_RENDER",
        "map": "small_loop",
        "rng_seed": RNG_SEED,
        "sample_count": SAMPLE_COUNT,
        "distinct_tiles": len(tile_counts),
        "drivable_tile_count": len(tiles),
        "tile_counts": tile_counts,
        "rejection_rate": sampler.rejection_rate,
        "contact_sheet": contact_sheet_path.name,
        "samples": manifest,
    }
    (OUTPUT_DIR / "manifest.json").write_text(
        json.dumps(manifest_payload, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({k: v for k, v in manifest_payload.items() if k != "samples"}, indent=2))


if __name__ == "__main__":
    main()
