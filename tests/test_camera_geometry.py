from __future__ import annotations

from pathlib import Path

import pytest

from duckie_pomdp.adapters.gym_duckietown import (
    GymDuckietownConfig,
    create_gym_duckietown,
)
from duckie_pomdp.domain.measurement import GroundPoint
from duckie_pomdp.evaluation.range_calibration import surface_point_in_ego
from duckie_pomdp.perception.camera_geometry import (
    CalibratedGroundProjector,
    CameraCalibration,
    ground_to_polar,
    world_to_ego,
)
from duckie_pomdp.scenario import PedestrianMode, load_scenario


def test_pixel_ground_round_trip_uses_locked_coordinates() -> None:
    projector = CalibratedGroundProjector(
        CameraCalibration(
            image_width_px=640,
            image_height_px=480,
            vertical_fov_deg=75.0,
            camera_height_m=0.108,
            camera_pitch_deg=19.15,
            camera_forward_offset_m=0.066,
        )
    )

    for ground in (
        GroundPoint(0.0, 0.4),
        GroundPoint(0.2, 0.7),
        GroundPoint(-0.3, 1.1),
    ):
        pixel = projector.ground_to_pixel(ground)
        recovered = projector.pixel_to_ground(pixel)
        assert recovered.x_left_m == pytest.approx(ground.x_left_m, abs=1e-10)
        assert recovered.y_forward_m == pytest.approx(ground.y_forward_m, abs=1e-10)
        polar = ground_to_polar(recovered)
        assert (polar.bearing_rad > 0.0) is (ground.x_left_m > 0.0)


def test_f5_real_camera_calibration_and_rendered_object_pixels() -> None:
    scenario = load_scenario(
        Path("configs/scenario_pomdp_v1.toml")
    ).with_pedestrian_mode(PedestrianMode.STATIONARY)
    integration = create_gym_duckietown(GymDuckietownConfig(scenario=scenario))
    try:
        integration.agent.reset(seed=scenario.seed)
        calibration = integration.camera_calibration.read()
        projector = CalibratedGroundProjector(calibration)
        privileged = integration.privileged.read()
        samples = integration.projection_validation.sample_visible_objects(
            ("sign_stop", "duckie")
        )

        assert calibration.image_width_px == 640
        assert calibration.image_height_px == 480
        assert calibration.vertical_fov_deg == pytest.approx(75.0)
        assert calibration.camera_height_m == pytest.approx(0.108)
        assert calibration.camera_pitch_deg == pytest.approx(19.15)
        assert calibration.camera_forward_offset_m == pytest.approx(0.066)
        assert not calibration.distortion_enabled
        assert {sample.object_kind for sample in samples} == {"sign_stop", "duckie"}

        for sample in samples:
            world = (
                privileged.stop_sign_world_position
                if sample.object_kind == "sign_stop"
                else privileged.pedestrian_world_position
            )
            footprint = (
                privileged.stop_sign_world_footprint
                if sample.object_kind == "sign_stop"
                else privileged.pedestrian_world_footprint
            )
            truth = world_to_ego(world, privileged.ego_world_pose)
            surface = surface_point_in_ego(footprint, privileged.ego_world_pose)
            prediction = projector.pixel_to_ground(sample.pixel)
            assert prediction.y_forward_m > 0.0
            assert abs(prediction.x_left_m - truth.x_left_m) < 0.20
            assert abs(
                ground_to_polar(prediction).range_m
                - ground_to_polar(truth).range_m
            ) < 0.04
            assert ground_to_polar(surface).range_m < ground_to_polar(truth).range_m
            assert abs(
                ground_to_polar(prediction).bearing_rad
                - ground_to_polar(truth).bearing_rad
            ) < 0.03
    finally:
        integration.close()
