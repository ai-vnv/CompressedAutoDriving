"""Machine-readable logging for F4 true-state validation."""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
from pathlib import Path

from duckie_pomdp.domain.action import PolicyAction
from duckie_pomdp.domain.privileged import PrivilegedSimulatorState


@dataclass(frozen=True)
class StateValidationRow:
    episode: str
    step: int
    timestamp: float
    ego_world_x: float
    ego_world_z: float
    ego_heading: float
    d: float
    phi: float
    v_actual: float
    omega_actual: float
    kappa: float
    stop_line_distance: float
    sign_world_x: float
    sign_world_z: float
    sign_r: float
    sign_beta: float
    ped_world_x: float
    ped_world_z: float
    ped_r: float
    ped_beta: float
    ped_rdot: float
    ped_betadot: float
    v_cmd: float
    omega_cmd: float


class StateValidationLogger:
    def __init__(self) -> None:
        self.rows: list[StateValidationRow] = []

    def record(
        self,
        *,
        episode: str,
        step: int,
        timestamp: float,
        privileged: PrivilegedSimulatorState,
        action: PolicyAction,
    ) -> StateValidationRow:
        state = privileged.true_pomdp_state
        sign_world = _required(privileged.stop_sign_world_position, "stop sign")
        pedestrian_world = _required(
            privileged.pedestrian_world_position,
            "pedestrian",
        )
        curvature = _required(state.road.curvature_inv_m, "road curvature")
        stop_distance = _required(
            state.road.stop_line_distance_m,
            "stop-line distance",
        )
        sign_range = _required(state.stop_sign.range_m, "stop-sign range")
        sign_bearing = _required(state.stop_sign.bearing_rad, "stop-sign bearing")
        pedestrian_range = _required(state.pedestrian.range_m, "pedestrian range")
        pedestrian_bearing = _required(
            state.pedestrian.bearing_rad,
            "pedestrian bearing",
        )
        radial_velocity = _required(
            state.pedestrian.radial_velocity_mps,
            "pedestrian radial velocity",
        )
        bearing_rate = _required(
            state.pedestrian.bearing_rate_rad_s,
            "pedestrian bearing rate",
        )
        row = StateValidationRow(
            episode=episode,
            step=step,
            timestamp=timestamp,
            ego_world_x=privileged.ego_world_pose.x_m,
            ego_world_z=privileged.ego_world_pose.z_m,
            ego_heading=privileged.ego_world_pose.heading_rad,
            d=state.ego.lateral_error_m,
            phi=state.ego.heading_error_rad,
            v_actual=state.ego.linear_velocity_mps,
            omega_actual=state.ego.yaw_rate_rad_s,
            kappa=curvature,
            stop_line_distance=stop_distance,
            sign_world_x=sign_world.x_m,
            sign_world_z=sign_world.z_m,
            sign_r=sign_range,
            sign_beta=sign_bearing,
            ped_world_x=pedestrian_world.x_m,
            ped_world_z=pedestrian_world.z_m,
            ped_r=pedestrian_range,
            ped_beta=pedestrian_bearing,
            ped_rdot=radial_velocity,
            ped_betadot=bearing_rate,
            v_cmd=action.linear_velocity_mps,
            omega_cmd=action.angular_velocity_rad_s,
        )
        self.rows.append(row)
        return row

    def write_csv(self, output_path: str | Path) -> None:
        if not self.rows:
            raise ValueError("cannot write an empty state-validation artifact")
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        rows = [asdict(row) for row in self.rows]
        with path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def _required(value, name: str):
    if value is None:
        raise RuntimeError(f"F4 logger requires {name}")
    return value
