"""Render one auditable RGB frame with simulator-derived annotation boxes."""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

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
        default=Path("artifacts/detection_annotation_probe.png"),
    )
    parser.add_argument("--ego-local-x", type=float, default=0.18)
    parser.add_argument("--ego-heading", type=float, default=-0.25)
    args = parser.parse_args()

    scenario = load_scenario(args.scenario).with_pedestrian_mode(
        PedestrianMode.STATIONARY
    )
    scenario = replace(
        scenario,
        ego_start_pose_m=(
            args.ego_local_x,
            scenario.ego_start_pose_m[1],
            scenario.ego_start_pose_m[2],
        ),
        ego_heading_rad=args.ego_heading,
    )
    integration = create_gym_duckietown(GymDuckietownConfig(scenario=scenario))
    try:
        observation = integration.agent.reset(seed=scenario.seed)
        samples = integration.projection_validation.sample_object_silhouettes(
            ("sign_stop", "duckie")
        )
        image = Image.fromarray(observation.front_rgb, mode="RGB")
        draw = ImageDraw.Draw(image)
        colors = {"sign_stop": "#ff3030", "duckie": "#00d7ff"}
        for sample in samples:
            box = sample.bounding_box
            coordinates = (
                box.x_min_px,
                box.y_min_px,
                box.x_max_px,
                box.y_max_px,
            )
            draw.rectangle(coordinates, outline=colors[sample.object_kind], width=3)
            draw.text(
                (box.x_min_px + 2, box.y_min_px + 2),
                sample.object_kind,
                fill=colors[sample.object_kind],
            )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        image.save(args.output)
        print(args.output)
    finally:
        integration.close()


if __name__ == "__main__":
    main()
