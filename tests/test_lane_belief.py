from __future__ import annotations

from math import pi

import numpy as np
import pytest

from duckie_pomdp.belief.lane_ekf import (
    LaneBeliefUpdater,
    LaneEKFConfig,
    lane_motion_function,
    lane_motion_jacobian,
)
from duckie_pomdp.domain.measurement import LaneMeasurement
from duckie_pomdp.domain.state import EgoMotion


def _config() -> LaneEKFConfig:
    return LaneEKFConfig(
        lateral_process_std_m_per_sqrt_s=0.01,
        heading_process_std_rad_per_sqrt_s=0.02,
        curvature_process_std_inv_m_per_sqrt_s=0.1,
        initial_lateral_std_m=0.2,
        initial_heading_std_rad=0.4,
        initial_curvature_std_inv_m=1.0,
        covariance_floor=1e-10,
        detection_validity_gain=0.6,
        miss_validity_decay=0.8,
    )


def _measurement(d: float = 0.04, phi: float = 0.1, kappa: float = 0.2) -> LaneMeasurement:
    return LaneMeasurement(
        detected=True,
        lateral_error_m=d,
        lateral_error_std_m=0.01,
        heading_error_rad=phi,
        heading_error_std_rad=0.02,
        curvature_inv_m=kappa,
        curvature_std_inv_m=0.1,
        visible_point_count=30,
        fit_residual_m=0.005,
    )


def test_lane_motion_uses_actual_motion_with_locked_signs() -> None:
    predicted = lane_motion_function(
        np.array([0.0, 0.1, 0.2]),
        EgoMotion(linear_velocity_mps=0.3, yaw_rate_rad_s=0.4),
        0.1,
    )
    assert predicted[0] > 0.0
    assert predicted[1] < 0.1  # CCW ego yaw dominates the left-curve tangent rate.


def test_lane_motion_jacobian_matches_finite_difference() -> None:
    state = np.array([0.02, -0.17, 0.45], dtype=float)
    motion = EgoMotion(0.27, -0.31)
    dt_s = 0.07
    analytical = lane_motion_jacobian(state, motion, dt_s)
    numerical = np.zeros((3, 3), dtype=float)
    epsilon = 1e-6
    for column in range(3):
        offset = np.zeros(3)
        offset[column] = epsilon
        difference = (
            lane_motion_function(state + offset, motion, dt_s)
            - lane_motion_function(state - offset, motion, dt_s)
        ) / (2.0 * epsilon)
        if column == 1:
            difference[1] = ((difference[1] + pi) % (2.0 * pi)) - pi
        numerical[:, column] = difference
    assert analytical == pytest.approx(numerical, abs=1e-7)


def test_misses_grow_covariance_and_reduce_lane_validity() -> None:
    updater = LaneBeliefUpdater(_config())
    updater.correct(_measurement())
    covariance_before = np.trace(updater.covariance)
    validity_before = updater.belief().validity_probability
    for _ in range(3):
        updater.step(LaneMeasurement.missing(), EgoMotion(0.2, 0.0), 0.1)
    assert np.trace(updater.covariance) > covariance_before
    assert updater.belief().validity_probability < validity_before


def test_reobservation_contracts_uncertainty() -> None:
    updater = LaneBeliefUpdater(_config())
    updater.correct(_measurement())
    for _ in range(5):
        updater.step(LaneMeasurement.missing(), EgoMotion(0.2, 0.1), 0.1)
    trace_before = np.trace(updater.covariance)
    diagnostics = updater.step(_measurement(), EgoMotion(0.2, 0.1), 0.1)
    assert diagnostics.corrected
    assert np.trace(updater.covariance) < trace_before


def test_uninitialized_lane_belief_is_not_zero_range_hazard_alias() -> None:
    belief = LaneBeliefUpdater(_config()).belief()
    assert belief.validity_probability == 0.0
    assert belief.lateral_error_mean_m == 0.0
    assert belief.lateral_error_std_m == pytest.approx(0.2)
    assert belief.heading_error_std_rad == pytest.approx(0.4)
    assert belief.curvature_std_inv_m == pytest.approx(1.0)
