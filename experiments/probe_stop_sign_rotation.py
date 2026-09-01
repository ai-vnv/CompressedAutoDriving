"""Render a contact sheet for stop-sign map rotations at one fixed camera pose."""

from __future__ import annotations

import argparse
import tempfile
from dataclasses import replace
from pathlib import Path

import yaml
from PIL import Image, ImageDraw

from duckie_pomdp.adapters.gym_duckietown import (
    GymDuckietownConfig,
    create_gym_duckietown,
)
from duckie_pomdp.scenario import PedestrianMode, load_scenario


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--scenario",
        type=Path,
        default=Path("configs/scenario_pomdp_v1.toml"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/stop_sign_rotation_sweep.png"),
    )
    args = parser.parse_args()

    base = load_scenario(args.scenario).with_pedestrian_mode(PedestrianMode.STATIONARY)
    map_data = yaml.safe_load(base.map_path.read_text(encoding="utf-8"))
    rotations = (-180, -90, 0, 90, 180)
    tiles: list[Image.Image] = []
    with tempfile.TemporaryDirectory(prefix="duckie-sign-rotation-") as temporary:
        temporary_root = Path(temporary)
        for rotation in rotations:
            candidate = dict(map_data)
            candidate["objects"] = {
                name: dict(value) for name, value in map_data["objects"].items()
            }
            candidate["objects"]["stop_sign"]["rotate"] = rotation
            map_path = temporary_root / f"pomdp_v1_sign_{rotation}.yaml"
            map_path.write_text(yaml.safe_dump(candidate), encoding="utf-8")
            scenario = replace(
                base,
                map_path=map_path,
                ego_start_pose_m=(0.18, base.ego_start_pose_m[1], base.ego_start_pose_m[2]),
                ego_heading_rad=-0.25,
            )
            integration = create_gym_duckietown(
                GymDuckietownConfig(scenario=scenario)
            )
            try:
                observation = integration.agent.reset(seed=scenario.seed)
                sign = next(
                    sample
                    for sample in integration.projection_validation.sample_object_silhouettes(
                        ("sign_stop",)
                    )
                )
                box = sign.bounding_box
                margin = 24
                left = max(0, int(box.x_min_px) - margin)
                top = max(0, int(box.y_min_px) - margin)
                right = min(640, int(box.x_max_px) + margin)
                bottom = min(480, int(box.y_max_px) + margin)
                crop = Image.fromarray(observation.front_rgb, mode="RGB").crop(
                    (left, top, right, bottom)
                )
                crop.thumbnail((260, 350))
                tile = Image.new("RGB", (280, 400), "white")
                tile.paste(crop, ((280 - crop.width) // 2, 35))
                draw = ImageDraw.Draw(tile)
                draw.text((10, 10), f"rotate: {rotation} deg", fill="black")
                tiles.append(tile)
            finally:
                integration.close()

    sheet = Image.new("RGB", (280 * len(tiles), 400), "white")
    for index, tile in enumerate(tiles):
        sheet.paste(tile, (280 * index, 0))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(args.output)
    print(args.output)


if __name__ == "__main__":
    main()
