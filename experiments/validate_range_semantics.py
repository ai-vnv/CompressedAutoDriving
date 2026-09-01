"""Run F5b range-semantics and held-out measurement calibration validation."""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

from duckie_pomdp.adapters.gym_duckietown import (
    GymDuckietownConfig,
    create_gym_duckietown,
)
from duckie_pomdp.domain.action import PolicyAction
from duckie_pomdp.evaluation.range_calibration import (
    RangeSemanticsSample,
    fit_and_evaluate_range_calibration,
    surface_point_in_ego,
    write_rows_csv,
)
from duckie_pomdp.perception.camera_geometry import (
    CalibratedGroundProjector,
    world_to_ego,
)
from duckie_pomdp.scenario import PedestrianMode, load_scenario


def collect_samples(scenario_path: Path) -> tuple[RangeSemanticsSample, ...]:
    base_scenario = load_scenario(scenario_path)
    samples: list[RangeSemanticsSample] = []
    frame = 0
    episodes = (
        (
            "calibration_approach",
            PedestrianMode.STATIONARY,
            PolicyAction(0.2, 0.0),
            10,
            2,
            (0.300, 0.0, 0.400),
        ),
        (
            "straight_approach",
            PedestrianMode.STATIONARY,
            PolicyAction(0.4, 0.0),
            50,
            4,
            None,
        ),
        ("turn_left", PedestrianMode.STATIONARY, PolicyAction(0.0, 1.5), 25, 2, None),
        ("turn_right", PedestrianMode.STATIONARY, PolicyAction(0.0, -1.5), 25, 2, None),
        (
            "pedestrian_crossing",
            PedestrianMode.CROSS_LEFT_TO_RIGHT,
            PolicyAction(0.0, 0.0),
            70,
            5,
            None,
        ),
    )
    for episode, mode, action, steps, sample_every, start_pose in episodes:
        scenario = base_scenario.with_pedestrian_mode(mode)
        if start_pose is not None:
            scenario = replace(scenario, ego_start_pose_m=start_pose)
        integration = create_gym_duckietown(GymDuckietownConfig(scenario=scenario))
        try:
            integration.agent.reset(seed=scenario.seed)
            projector = CalibratedGroundProjector(integration.camera_calibration.read())
            frame = _sample_frame(
                samples,
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
                        samples,
                        integration,
                        projector,
                        frame=frame,
                        episode=episode,
                        step=step,
                    )
        finally:
            integration.close()
    return tuple(samples)


def _sample_frame(
    samples: list[RangeSemanticsSample],
    integration,
    projector: CalibratedGroundProjector,
    *,
    frame: int,
    episode: str,
    step: int,
) -> int:
    privileged = integration.privileged.read()
    visible = integration.projection_validation.sample_visible_objects(
        ("sign_stop", "duckie")
    )
    for sample in visible:
        if sample.object_kind == "sign_stop":
            world_position = privileged.stop_sign_world_position
            footprint = privileged.stop_sign_world_footprint
        else:
            world_position = privileged.pedestrian_world_position
            footprint = privileged.pedestrian_world_footprint
        if world_position is None or footprint is None:
            raise RuntimeError(f"missing privileged geometry for {sample.object_kind}")
        normalized_offset = abs(
            sample.pixel.x_px - 0.5 * projector.calibration.image_width_px
        ) / (0.5 * projector.calibration.image_width_px)
        samples.append(
            RangeSemanticsSample(
                episode=episode,
                frame=frame,
                step=step,
                object_type=sample.object_kind,
                pixel_u=sample.pixel.x_px,
                pixel_v=sample.pixel.y_px,
                projected=projector.pixel_to_ground(sample.pixel),
                origin=world_to_ego(world_position, privileged.ego_world_pose),
                surface=surface_point_in_ego(
                    footprint,
                    privileged.ego_world_pose,
                ),
                fov_region=_fov_region(normalized_offset),
                silhouette_pixel_count=sample.silhouette_pixel_count,
            )
        )
    return frame + 1


def _fov_region(normalized_offset: float) -> str:
    if normalized_offset < 1.0 / 3.0:
        return "center"
    if normalized_offset < 2.0 / 3.0:
        return "mid_fov"
    return "edge_fov"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--scenario",
        type=Path,
        default=Path("configs/scenario_pomdp_v1.toml"),
    )
    parser.add_argument(
        "--semantics-output",
        type=Path,
        default=Path("artifacts/range_semantics_validation.csv"),
    )
    parser.add_argument(
        "--validation-output",
        type=Path,
        default=Path("artifacts/range_calibration_validation.csv"),
    )
    parser.add_argument(
        "--noise-output",
        type=Path,
        default=Path("artifacts/measurement_noise_v1.json"),
    )
    parser.add_argument(
        "--historical-f5-metrics",
        type=Path,
        default=Path("artifacts/ground_projection_metrics_pre_f5b.json"),
    )
    args = parser.parse_args()

    result = fit_and_evaluate_range_calibration(collect_samples(args.scenario))
    write_rows_csv(result.rows, args.semantics_output)
    write_rows_csv(
        (row for row in result.rows if row.split == "validation"),
        args.validation_output,
    )

    historical = None
    if args.historical_f5_metrics.is_file():
        historical = json.loads(args.historical_f5_metrics.read_text(encoding="utf-8"))
    report = {
        "camera_geometry_correction": {
            "issue": "look-at target incorrectly used ground y instead of camera y",
            "historical_pre_fix_metrics": historical,
        },
        **result.summary,
    }
    args.noise_output.parent.mkdir(parents=True, exist_ok=True)
    args.noise_output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "semantics_output": str(args.semantics_output),
                "validation_output": str(args.validation_output),
                "noise_output": str(args.noise_output),
                "rows": len(result.rows),
                **result.summary,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
