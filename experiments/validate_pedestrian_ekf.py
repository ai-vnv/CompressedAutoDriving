"""Run the F7 pedestrian EKF scenario matrix on real simulator trajectories."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from math import pi, sqrt
from pathlib import Path

import numpy as np

from duckie_pomdp.adapters.gym_duckietown import (
    GymDuckietownConfig,
    create_gym_duckietown,
)
from duckie_pomdp.belief.pedestrian_ekf import (
    MeasurementProfile,
    PedestrianEKFConfig,
    load_pedestrian_ekf_config,
)
from duckie_pomdp.belief.updater import (
    PedestrianBeliefUpdater,
    initial_belief,
    load_existence_filter_config,
)
from duckie_pomdp.domain.action import PolicyAction
from duckie_pomdp.domain.detection import ObjectClass
from duckie_pomdp.domain.measurement import PerceptionObservation
from duckie_pomdp.domain.observation import EgoObservation
from duckie_pomdp.domain.privileged import PrivilegedSimulatorState
from duckie_pomdp.perception.measurement_calibration import wrap_angle
from duckie_pomdp.perception.measurement_noise import (
    PolarMeasurementNoiseModel,
    load_polar_measurement_noise,
)
from duckie_pomdp.perception.oracle_measurement import (
    OracleDetectionConfig,
    OracleMode,
    OracleObservationModel,
    load_oracle_detection_config,
)
from duckie_pomdp.scenario import PedestrianMode, load_scenario


@dataclass(frozen=True)
class ScenarioSpec:
    name: str
    pedestrian_mode: PedestrianMode
    action: PolicyAction
    steps: int


@dataclass(frozen=True)
class TrajectoryFrame:
    episode: str
    frame: int
    timestamp_s: float
    dt_s: float
    action: PolicyAction
    ego: EgoObservation
    privileged: PrivilegedSimulatorState


@dataclass(frozen=True)
class TrackingRow:
    episode: str
    mode: str
    frame: int
    timestamp_s: float
    seed: int
    detected: bool
    gt_range_m: float
    gt_bearing_rad: float
    gt_range_rate_mps: float
    gt_bearing_rate_rad_s: float
    measurement_range_m: float | None
    measurement_bearing_rad: float | None
    belief_range_m: float
    belief_range_std_m: float
    belief_bearing_rad: float
    belief_bearing_std_rad: float
    belief_range_rate_mps: float
    belief_range_rate_std_mps: float
    belief_bearing_rate_rad_s: float
    belief_bearing_rate_std_rad_s: float
    existence_probability: float
    internal_x_left_m: float
    internal_y_forward_m: float
    internal_velocity_left_mps: float
    internal_velocity_forward_mps: float
    gt_velocity_left_mps: float
    gt_velocity_forward_mps: float
    actual_ego_velocity_mps: float
    actual_ego_yaw_rate_rad_s: float
    commanded_velocity_mps: float
    commanded_yaw_rate_rad_s: float
    covariance_trace: float
    range_error_m: float
    bearing_error_rad: float
    range_rate_error_mps: float
    bearing_rate_error_rad_s: float


SCENARIOS = (
    ScenarioSpec(
        "stationary_ped_stationary_ego",
        PedestrianMode.STATIONARY,
        PolicyAction(0.0, 0.0),
        70,
    ),
    ScenarioSpec(
        "stationary_ped_moving_ego",
        PedestrianMode.STATIONARY,
        PolicyAction(0.2, 0.0),
        70,
    ),
    ScenarioSpec(
        "stationary_ped_turning_ego",
        PedestrianMode.STATIONARY,
        PolicyAction(0.0, 1.0),
        55,
    ),
    ScenarioSpec(
        "cross_left_to_right",
        PedestrianMode.CROSS_LEFT_TO_RIGHT,
        PolicyAction(0.0, 0.0),
        90,
    ),
    ScenarioSpec(
        "cross_right_to_left",
        PedestrianMode.CROSS_RIGHT_TO_LEFT,
        PolicyAction(0.0, 0.0),
        90,
    ),
    ScenarioSpec(
        "crossing_moving_turning_ego",
        PedestrianMode.CROSS_LEFT_TO_RIGHT,
        PolicyAction(0.15, 0.35),
        70,
    ),
)


def collect_trajectories(
    scenario_path: Path,
) -> dict[str, tuple[TrajectoryFrame, ...]]:
    base_scenario = load_scenario(scenario_path)
    trajectories: dict[str, tuple[TrajectoryFrame, ...]] = {}
    for spec in SCENARIOS:
        scenario = base_scenario.with_pedestrian_mode(spec.pedestrian_mode)
        integration = create_gym_duckietown(
            GymDuckietownConfig(
                scenario=scenario,
                camera_width=80,
                camera_height=60,
            )
        )
        frames: list[TrajectoryFrame] = []
        try:
            observation = integration.agent.reset(seed=scenario.seed)
            frames.append(
                TrajectoryFrame(
                    episode=spec.name,
                    frame=0,
                    timestamp_s=0.0,
                    dt_s=1.0 / 30.0,
                    action=PolicyAction(0.0, 0.0),
                    ego=observation.ego,
                    privileged=integration.privileged.read(),
                )
            )
            previous_timestamp = 0.0
            for frame_index in range(1, spec.steps + 1):
                transition = integration.agent.step(spec.action)
                diagnostics = integration.diagnostics.read()
                if transition.terminated or transition.truncated:
                    raise RuntimeError(
                        f"{spec.name} ended early: {diagnostics.done_code}"
                    )
                dt_s = diagnostics.timestamp_s - previous_timestamp
                previous_timestamp = diagnostics.timestamp_s
                frames.append(
                    TrajectoryFrame(
                        episode=spec.name,
                        frame=frame_index,
                        timestamp_s=diagnostics.timestamp_s,
                        dt_s=dt_s,
                        action=spec.action,
                        ego=transition.observation.ego,
                        privileged=integration.privileged.read(),
                    )
                )
        finally:
            integration.close()
        trajectories[spec.name] = tuple(frames)
    return trajectories


def replay(
    trajectories: dict[str, tuple[TrajectoryFrame, ...]],
    *,
    mode: OracleMode,
    seed: int,
    ekf_config: PedestrianEKFConfig,
    measurement_noise: PolarMeasurementNoiseModel,
    detection: OracleDetectionConfig,
    oracle_config_path: Path,
) -> list[TrackingRow]:
    rows: list[TrackingRow] = []
    profile = (
        MeasurementProfile.CLEAN
        if mode is OracleMode.CLEAN
        else MeasurementProfile.CALIBRATED_RESIDUAL
    )
    existence_config = load_existence_filter_config(oracle_config_path)
    for episode_index, (episode, frames) in enumerate(trajectories.items()):
        pedestrian_oracle = OracleObservationModel(
            mode=mode,
            measurement_noise=measurement_noise,
            detection=detection,
            seed=seed + 1000 * episode_index,
        )
        stop_sign_oracle = OracleObservationModel(
            mode=mode,
            measurement_noise=measurement_noise,
            detection=detection,
            seed=seed + 1000 * episode_index + 997,
        )
        updater = PedestrianBeliefUpdater(
            ekf_config=ekf_config,
            existence_config=existence_config,
            measurement_noise=measurement_noise,
            measurement_profile=profile,
        )
        belief = initial_belief(
            frames[0].ego,
            existence_prior=existence_config.prior_probability,
        )
        for frame in frames:
            pedestrian_measurement = pedestrian_oracle.observe(
                frame.privileged,
                ObjectClass.DUCKIE,
            )
            stop_sign_measurement = stop_sign_oracle.observe(
                frame.privileged,
                ObjectClass.STOP_SIGN,
            )
            perception = PerceptionObservation(
                ego=frame.ego,
                road=None,
                stop_sign=stop_sign_measurement,
                pedestrian=pedestrian_measurement,
            )
            belief = updater.update(
                previous_belief=belief,
                previous_action=frame.action,
                ego_motion=frame.ego.motion,
                perception=perception,
                dt_s=frame.dt_s,
            )
            if updater.ekf.initialized:
                rows.append(
                    _tracking_row(
                        frame,
                        mode,
                        seed,
                        pedestrian_measurement,
                        belief.pedestrian,
                        updater,
                    )
                )
    return rows


def _tracking_row(
    frame: TrajectoryFrame,
    mode: OracleMode,
    seed: int,
    measurement,
    belief,
    updater: PedestrianBeliefUpdater,
) -> TrackingRow:
    truth = frame.privileged.true_pomdp_state.pedestrian
    if (
        not truth.exists
        or truth.range_m is None
        or truth.bearing_rad is None
        or truth.radial_velocity_mps is None
        or truth.bearing_rate_rad_s is None
    ):
        raise RuntimeError("F7 trajectory lost pedestrian truth")
    velocity_left, velocity_forward = _true_physical_velocity_ego(frame.privileged)
    state = updater.ekf.state
    return TrackingRow(
        episode=frame.episode,
        mode=mode.value,
        frame=frame.frame,
        timestamp_s=frame.timestamp_s,
        seed=seed,
        detected=measurement.detected,
        gt_range_m=truth.range_m,
        gt_bearing_rad=truth.bearing_rad,
        gt_range_rate_mps=truth.radial_velocity_mps,
        gt_bearing_rate_rad_s=truth.bearing_rate_rad_s,
        measurement_range_m=measurement.range_m,
        measurement_bearing_rad=measurement.bearing_rad,
        belief_range_m=belief.range_mean_m,
        belief_range_std_m=belief.range_std_m,
        belief_bearing_rad=belief.bearing_mean_rad,
        belief_bearing_std_rad=belief.bearing_std_rad,
        belief_range_rate_mps=belief.radial_velocity_mean_mps,
        belief_range_rate_std_mps=belief.radial_velocity_std_mps,
        belief_bearing_rate_rad_s=belief.bearing_rate_mean_rad_s,
        belief_bearing_rate_std_rad_s=belief.bearing_rate_std_rad_s,
        existence_probability=belief.existence_probability,
        internal_x_left_m=float(state[0]),
        internal_y_forward_m=float(state[1]),
        internal_velocity_left_mps=float(state[2]),
        internal_velocity_forward_mps=float(state[3]),
        gt_velocity_left_mps=velocity_left,
        gt_velocity_forward_mps=velocity_forward,
        actual_ego_velocity_mps=frame.ego.linear_velocity_mps,
        actual_ego_yaw_rate_rad_s=frame.ego.yaw_rate_rad_s,
        commanded_velocity_mps=frame.action.linear_velocity_mps,
        commanded_yaw_rate_rad_s=frame.action.angular_velocity_rad_s,
        covariance_trace=float(np.trace(updater.ekf.covariance)),
        range_error_m=belief.range_mean_m - truth.range_m,
        bearing_error_rad=wrap_angle(belief.bearing_mean_rad - truth.bearing_rad),
        range_rate_error_mps=(
            belief.radial_velocity_mean_mps - truth.radial_velocity_mps
        ),
        bearing_rate_error_rad_s=(
            belief.bearing_rate_mean_rad_s - truth.bearing_rate_rad_s
        ),
    )


def _true_physical_velocity_ego(
    privileged: PrivilegedSimulatorState,
) -> tuple[float, float]:
    velocity = privileged.pedestrian_world_velocity
    if velocity is None:
        raise RuntimeError("F7 trajectory has no pedestrian world velocity")
    heading = privileged.ego_world_pose.heading_rad
    sine = np.sin(heading)
    cosine = np.cos(heading)
    return (
        -velocity.x_velocity_mps * sine - velocity.z_velocity_mps * cosine,
        velocity.x_velocity_mps * cosine - velocity.z_velocity_mps * sine,
    )


def summarize(rows: list[TrackingRow]) -> dict[str, object]:
    return {
        "global": _metrics(rows),
        "by_mode": {
            mode.value: _metrics([row for row in rows if row.mode == mode.value])
            for mode in OracleMode
        },
        "by_scenario_and_mode": {
            f"{scenario.name}/{mode.value}": _metrics(
                [
                    row
                    for row in rows
                    if row.episode == scenario.name and row.mode == mode.value
                ]
            )
            for scenario in SCENARIOS
            for mode in OracleMode
        },
    }


def _metrics(rows: list[TrackingRow]) -> dict[str, object]:
    if not rows:
        return {"count": 0}
    detected = [row for row in rows if row.detected]
    variables = {
        "range": (
            np.array([row.range_error_m for row in rows]),
            np.array([row.belief_range_std_m for row in rows]),
        ),
        "bearing": (
            np.array([row.bearing_error_rad for row in rows]),
            np.array([row.belief_bearing_std_rad for row in rows]),
        ),
        "range_rate": (
            np.array([row.range_rate_error_mps for row in rows]),
            np.array([row.belief_range_rate_std_mps for row in rows]),
        ),
        "bearing_rate": (
            np.array([row.bearing_rate_error_rad_s for row in rows]),
            np.array([row.belief_bearing_rate_std_rad_s for row in rows]),
        ),
    }
    probabilistic: dict[str, object] = {}
    for name, (errors, standard_deviations) in variables.items():
        safe_sigma = np.maximum(standard_deviations, 1.0e-12)
        probabilistic[name] = {
            "rmse": float(np.sqrt(np.mean(errors**2))),
            "mean_predicted_std": float(np.mean(standard_deviations)),
            "mean_marginal_nll": float(
                np.mean(
                    0.5
                    * (
                        np.log(2.0 * pi * safe_sigma**2)
                        + errors**2 / safe_sigma**2
                    )
                )
            ),
            "coverage_68": float(np.mean(np.abs(errors) <= safe_sigma)),
            "coverage_95": float(np.mean(np.abs(errors) <= 1.96 * safe_sigma)),
        }
    current_range_errors = np.array(
        [row.measurement_range_m - row.gt_range_m for row in detected],
        dtype=float,
    )
    current_bearing_errors = np.array(
        [wrap_angle(row.measurement_bearing_rad - row.gt_bearing_rad) for row in detected],
        dtype=float,
    )
    velocity_left_errors = np.array(
        [row.internal_velocity_left_mps - row.gt_velocity_left_mps for row in rows]
    )
    velocity_forward_errors = np.array(
        [row.internal_velocity_forward_mps - row.gt_velocity_forward_mps for row in rows]
    )
    return {
        "count": len(rows),
        "detected_count": len(detected),
        "empirical_dropout_probability": 1.0 - len(detected) / len(rows),
        "current_frame_observation": {
            "range_rmse_m": _rmse(current_range_errors),
            "bearing_rmse_rad": _rmse(current_bearing_errors),
            "rates_available": False,
        },
        "ekf": probabilistic,
        "physical_velocity": {
            "left_rmse_mps": _rmse(velocity_left_errors),
            "forward_rmse_mps": _rmse(velocity_forward_errors),
        },
        "existence": {
            "mean_probability": float(
                np.mean([row.existence_probability for row in rows])
            ),
            "minimum_probability": float(
                np.min([row.existence_probability for row in rows])
            ),
        },
    }


def q_sensitivity(
    trajectories: dict[str, tuple[TrajectoryFrame, ...]],
    *,
    base_config: PedestrianEKFConfig,
    measurement_noise: PolarMeasurementNoiseModel,
    detection: OracleDetectionConfig,
    oracle_config_path: Path,
    seed: int,
) -> list[dict[str, object]]:
    results: list[dict[str, object]] = []
    for position_std in (0.001, 0.002, 0.005):
        for velocity_std in (0.005, 0.020, 0.050):
            config = base_config.with_process_noise(
                position_std=position_std,
                velocity_std=velocity_std,
            )
            rows = replay(
                trajectories,
                mode=OracleMode.DROPOUT,
                seed=seed,
                ekf_config=config,
                measurement_noise=measurement_noise,
                detection=detection,
                oracle_config_path=oracle_config_path,
            )
            metrics = _metrics(rows)
            results.append(
                {
                    "position_process_std_m_per_sqrt_s": position_std,
                    "velocity_process_std_mps_per_sqrt_s": velocity_std,
                    "selected_v1_choice": (
                        position_std
                        == base_config.position_process_std_m_per_sqrt_s
                        and velocity_std
                        == base_config.velocity_process_std_mps_per_sqrt_s
                    ),
                    "range_rmse_m": metrics["ekf"]["range"]["rmse"],
                    "bearing_rmse_rad": metrics["ekf"]["bearing"]["rmse"],
                    "range_rate_rmse_mps": metrics["ekf"]["range_rate"]["rmse"],
                    "bearing_rate_rmse_rad_s": metrics["ekf"]["bearing_rate"]["rmse"],
                    "range_coverage_68": metrics["ekf"]["range"]["coverage_68"],
                    "bearing_coverage_68": metrics["ekf"]["bearing"]["coverage_68"],
                    "range_rate_coverage_68": metrics["ekf"]["range_rate"]["coverage_68"],
                    "bearing_rate_coverage_68": metrics["ekf"]["bearing_rate"]["coverage_68"],
                }
            )
    return results


def _rmse(values: np.ndarray) -> float | None:
    if len(values) == 0:
        return None
    return float(sqrt(float(np.mean(values**2))))


def _write_csv(rows: list[TrackingRow], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=tuple(TrackingRow.__annotations__))
        writer.writeheader()
        for row in rows:
            writer.writerow(row.__dict__)


def run_validation(
    *,
    scenario_path: Path,
    measurement_config_path: Path,
    oracle_config_path: Path,
    tracking_output_path: Path,
    metrics_output_path: Path,
    seed: int,
) -> dict[str, object]:
    trajectories = collect_trajectories(scenario_path)
    measurement_noise = load_polar_measurement_noise(measurement_config_path)
    detection = load_oracle_detection_config(oracle_config_path)
    ekf_config = load_pedestrian_ekf_config(oracle_config_path)
    all_rows: list[TrackingRow] = []
    for mode_index, mode in enumerate(OracleMode):
        all_rows.extend(
            replay(
                trajectories,
                mode=mode,
                seed=seed + mode_index * 100_000,
                ekf_config=ekf_config,
                measurement_noise=measurement_noise,
                detection=detection,
                oracle_config_path=oracle_config_path,
            )
        )
    _write_csv(all_rows, tracking_output_path)
    report = {
        "seed": seed,
        "scenario_config": str(scenario_path),
        "measurement_config": str(measurement_config_path),
        "oracle_ekf_config": str(oracle_config_path),
        "internal_velocity_semantics": (
            "pedestrian physical world velocity expressed in current ego axes"
        ),
        "relative_rate_semantics": (
            "derived from physical pedestrian velocity plus actual ego translation/yaw"
        ),
        "process_noise": {
            "position_process_std_m_per_sqrt_s": (
                ekf_config.position_process_std_m_per_sqrt_s
            ),
            "velocity_process_std_mps_per_sqrt_s": (
                ekf_config.velocity_process_std_mps_per_sqrt_s
            ),
            "selection": (
                "selected once from the common 3x3 sensitivity grid for lower "
                "RMSE and less-conservative coverage; never tuned per scenario"
            ),
        },
        "metrics": summarize(all_rows),
        "q_sensitivity_oracle_dropout": q_sensitivity(
            trajectories,
            base_config=ekf_config,
            measurement_noise=measurement_noise,
            detection=detection,
            oracle_config_path=oracle_config_path,
            seed=seed + 200_000,
        ),
    }
    metrics_output_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_output_path.write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--scenario",
        type=Path,
        default=Path("configs/scenario_pomdp_v1.toml"),
    )
    parser.add_argument(
        "--measurement-config",
        type=Path,
        default=Path("configs/measurement_model_v1.toml"),
    )
    parser.add_argument(
        "--oracle-config",
        type=Path,
        default=Path("configs/oracle_ekf_v1.toml"),
    )
    parser.add_argument(
        "--tracking-output",
        type=Path,
        default=Path("artifacts/ekf_tracking_validation.csv"),
    )
    parser.add_argument(
        "--metrics-output",
        type=Path,
        default=Path("artifacts/belief_calibration_metrics.json"),
    )
    parser.add_argument("--seed", type=int, default=20260807)
    args = parser.parse_args()
    report = run_validation(
        scenario_path=args.scenario,
        measurement_config_path=args.measurement_config,
        oracle_config_path=args.oracle_config,
        tracking_output_path=args.tracking_output,
        metrics_output_path=args.metrics_output,
        seed=args.seed,
    )
    print(
        json.dumps(
            {
                "tracking_output": str(args.tracking_output),
                "metrics_output": str(args.metrics_output),
                "process_noise": report["process_noise"],
                "by_mode": report["metrics"]["by_mode"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
