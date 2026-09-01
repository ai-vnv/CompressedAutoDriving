from __future__ import annotations

from math import cos, sin

import pytest

from duckie_pomdp.adapters.ego_motion import (
    SimulatorPoseSample,
    estimate_actual_motion,
)


def test_straight_motion_in_simulator_coordinates() -> None:
    previous = SimulatorPoseSample(1.0, 2.0, 0.0, 3.0)
    current = SimulatorPoseSample(1.2, 2.0, 0.0, 4.0)

    motion = estimate_actual_motion(previous, current)

    assert motion.linear_velocity_mps == pytest.approx(0.2)
    assert motion.yaw_rate_rad_s == pytest.approx(0.0)


def test_positive_heading_change_is_counter_clockwise() -> None:
    previous = SimulatorPoseSample(0.0, 0.0, 0.0, 0.0)
    delta_heading = 0.2
    distance = 0.3
    current = SimulatorPoseSample(
        x_m=distance * cos(0.5 * delta_heading),
        z_m=-distance * sin(0.5 * delta_heading),
        heading_rad=delta_heading,
        timestamp_s=1.0,
    )

    motion = estimate_actual_motion(previous, current)

    assert motion.linear_velocity_mps == pytest.approx(distance)
    assert motion.yaw_rate_rad_s == pytest.approx(delta_heading)


def test_heading_wrap_is_continuous() -> None:
    previous = SimulatorPoseSample(0.0, 0.0, 3.1, 0.0)
    current = SimulatorPoseSample(0.0, 0.0, -3.1, 0.1)

    motion = estimate_actual_motion(previous, current)

    assert motion.yaw_rate_rad_s > 0.0


def test_timestamps_must_increase() -> None:
    sample = SimulatorPoseSample(0.0, 0.0, 0.0, 1.0)

    with pytest.raises(ValueError, match="must increase"):
        estimate_actual_motion(sample, sample)
