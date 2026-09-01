"""Validate calibrated pixel-to-ground projection against simulator truth."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from duckie_pomdp.adapters.gym_duckietown import (
    GymDuckietownConfig,
    create_gym_duckietown,
)
from duckie_pomdp.domain.action import PolicyAction
from duckie_pomdp.evaluation import ProjectionValidationLogger
from duckie_pomdp.perception.camera_geometry import (
    CalibratedGroundProjector,
    world_to_ego,
)
from duckie_pomdp.scenario import PedestrianMode, load_scenario


def run_validation(
    scenario_path: Path,
    output_path: Path,
    metrics_path: Path,
) -> ProjectionValidationLogger:
    base_scenario = load_scenario(scenario_path)
    logger = ProjectionValidationLogger()
    frame = 0
    episodes = (
        ("straight_approach", PedestrianMode.STATIONARY, PolicyAction(0.4, 0.0), 50, 4),
        ("turn_left", PedestrianMode.STATIONARY, PolicyAction(0.0, 1.5), 25, 2),
        ("turn_right", PedestrianMode.STATIONARY, PolicyAction(0.0, -1.5), 25, 2),
        ("pedestrian_crossing", PedestrianMode.CROSS_LEFT_TO_RIGHT, PolicyAction(0.0, 0.0), 70, 5),
    )
    for episode, mode, action, steps, sample_every in episodes:
        scenario = base_scenario.with_pedestrian_mode(mode)
        integration = create_gym_duckietown(
            GymDuckietownConfig(scenario=scenario)
        )
        try:
            integration.agent.reset(seed=scenario.seed)
            projector = CalibratedGroundProjector(
                integration.camera_calibration.read()
            )
            frame = _sample_frame(
                logger,
                integration,
                projector,
                frame=frame,
                episode=episode,
                step=0,
            )
            for step in range(1, steps + 1):
                transition = integration.agent.step(action)
                if transition.terminated or transition.truncated:
                    break
                if step % sample_every == 0:
                    frame = _sample_frame(
                        logger,
                        integration,
                        projector,
                        frame=frame,
                        episode=episode,
                        step=step,
                    )
        finally:
            integration.close()

    logger.write_csv(output_path)
    metrics = logger.metrics()
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return logger


def _sample_frame(
    logger: ProjectionValidationLogger,
    integration,
    projector: CalibratedGroundProjector,
    *,
    frame: int,
    episode: str,
    step: int,
) -> int:
    privileged = integration.privileged.read()
    samples = integration.projection_validation.sample_visible_objects(
        ("sign_stop", "duckie")
    )
    for sample in samples:
        world_position = (
            privileged.stop_sign_world_position
            if sample.object_kind == "sign_stop"
            else privileged.pedestrian_world_position
        )
        if world_position is None:
            raise RuntimeError(f"missing privileged position for {sample.object_kind}")
        logger.record(
            frame=frame,
            episode=episode,
            step=step,
            object_type=sample.object_kind,
            pixel=sample.pixel,
            silhouette_pixel_count=sample.silhouette_pixel_count,
            ground_truth=world_to_ego(
                world_position,
                privileged.ego_world_pose,
            ),
            projector=projector,
        )
    return frame + 1


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
        default=Path("artifacts/ground_projection_validation.csv"),
    )
    parser.add_argument(
        "--metrics-output",
        type=Path,
        default=Path("artifacts/ground_projection_metrics.json"),
    )
    args = parser.parse_args()
    logger = run_validation(args.scenario, args.output, args.metrics_output)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "metrics_output": str(args.metrics_output),
                "rows": len(logger.rows),
                **logger.metrics(),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
