"""Exercise the action adapter against Gym-Duckietown dynamics.

This gate bypasses ``DuckietownEnv.step`` deliberately. The local adapter has
already converted ``(v_cmd, omega_cmd)`` to wheel duty, so calling that wrapper
would convert the action a second time.
"""

from __future__ import annotations

import argparse
import json
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
from statistics import mean
from typing import Any

import numpy as np

from duckie_pomdp.adapters.differential_drive import (
    DifferentialDriveActionAdapter,
    DifferentialDriveCalibration,
)
from duckie_pomdp.adapters.ego_motion import (
    SimulatorPoseSample,
    estimate_actual_motion,
)
from duckie_pomdp.domain.action import PolicyAction


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--map", default="small_loop")
    parser.add_argument("--seed", type=int, default=73)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--steady-steps", type=int, default=10)
    parser.add_argument("--candidate-v-max", type=float, default=0.4)
    parser.add_argument("--candidate-omega-max", type=float, default=4.0)
    parser.add_argument("--duty-headroom-limit", type=float, default=0.75)
    parser.add_argument("--maximum-turning-radius-m", type=float, default=0.30)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/action_gate/report.json"),
    )
    return parser.parse_args()


def pose_sample(environment: Any) -> SimulatorPoseSample:
    return SimulatorPoseSample(
        x_m=float(environment.cur_pos[0]),
        z_m=float(environment.cur_pos[2]),
        heading_rad=float(environment.cur_angle),
        timestamp_s=float(environment.timestamp),
    )


def reset_trial(environment: Any, initial: dict[str, Any]) -> None:
    environment.cur_pos = initial["position"].copy()
    environment.cur_angle = initial["heading"]
    environment.state = deepcopy(initial["dynamics_state"])
    environment.timestamp = 0.0
    environment.step_count = 0
    environment.speed = 0.0
    environment.last_action = np.zeros(2, dtype=float)
    environment.wheelVels = np.zeros(2, dtype=float)


def run_trial(
    environment: Any,
    adapter: DifferentialDriveActionAdapter,
    initial: dict[str, Any],
    action: PolicyAction,
    steps: int,
    steady_steps: int,
) -> dict[str, Any]:
    reset_trial(environment, initial)
    conversion = adapter.convert(action)
    duty = np.array(
        [conversion.wheel_command.left, conversion.wheel_command.right],
        dtype=float,
    )
    motion_samples = []
    remained_valid = True
    for _ in range(steps):
        previous = pose_sample(environment)
        environment.update_physics(duty)
        current = pose_sample(environment)
        motion_samples.append(estimate_actual_motion(previous, current))
        remained_valid = remained_valid and bool(
            environment._valid_pose(environment.cur_pos, environment.cur_angle)
        )

    steady = motion_samples[-steady_steps:]
    measured_linear = mean(item.linear_velocity_mps for item in steady)
    measured_angular = mean(item.yaw_rate_rad_s for item in steady)
    turning_radius = None
    if abs(measured_angular) > 1e-6:
        turning_radius = abs(measured_linear / measured_angular)

    return {
        "command": asdict(action),
        "wheel_rate_rad_s": asdict(conversion.wheel_angular_velocity),
        "unclipped_wheel_duty": asdict(conversion.unclipped_wheel_command),
        "wheel_duty": asdict(conversion.wheel_command),
        "saturated": conversion.saturated,
        "actual_linear_velocity_mps": measured_linear,
        "actual_yaw_rate_rad_s": measured_angular,
        "turning_radius_m": turning_radius,
        "remained_valid": remained_valid,
    }


def find_trial(
    trials: list[dict[str, Any]],
    linear_velocity_mps: float,
    angular_velocity_rad_s: float,
) -> dict[str, Any]:
    for trial in trials:
        command = trial["command"]
        if (
            command["linear_velocity_mps"] == linear_velocity_mps
            and command["angular_velocity_rad_s"] == angular_velocity_rad_s
        ):
            return trial
    raise LookupError("requested trial is absent from the sweep")


def main() -> None:
    args = parse_args()
    if args.steps <= 0 or not 0 < args.steady_steps <= args.steps:
        raise SystemExit("steady-steps must be in [1, steps]")

    from gym_duckietown.envs import DuckietownEnv

    environment = DuckietownEnv(
        map_name=args.map,
        domain_rand=False,
        dynamics_rand=False,
        frame_rate=30,
        frame_skip=1,
        seed=args.seed,
    )
    try:
        # Fixed centerline pose on the eastbound straight of small_loop.
        # ``start_pose`` is tile-relative, while the resulting pose is world x/z.
        if args.map != "small_loop":
            raise SystemExit("the deterministic gate pose is defined for small_loop")
        environment.start_tile = environment._get_tile(1, 0)
        environment.start_pose = ([0.065, 0.0, 0.4095], 0.0)
        environment.reset()

        calibration = DifferentialDriveCalibration(
            wheel_radius_m=float(environment.radius),
            wheel_separation_m=float(environment.wheel_dist),
            motor_constant_rad_s_per_unit=float(environment.k),
            gain=float(environment.gain),
            trim=float(environment.trim),
            command_limit=float(environment.limit),
        )
        adapter = DifferentialDriveActionAdapter(calibration)
        initial = {
            "position": environment.cur_pos.copy(),
            "heading": float(environment.cur_angle),
            "dynamics_state": environment.state,
        }

        linear_values = sorted(
            {0.1, 0.2, 0.3, 0.4, 0.5, args.candidate_v_max}
        )
        angular_values = sorted(
            {
                -4.0,
                -3.0,
                -2.0,
                -1.0,
                0.0,
                1.0,
                2.0,
                3.0,
                4.0,
                -args.candidate_omega_max,
                args.candidate_omega_max,
            }
        )
        trials = [
            run_trial(
                environment,
                adapter,
                initial,
                PolicyAction(linear, angular),
                args.steps,
                args.steady_steps,
            )
            for linear in linear_values
            for angular in angular_values
        ]

        positive_sign = run_trial(
            environment,
            adapter,
            initial,
            PolicyAction(0.12, 0.8),
            args.steps,
            args.steady_steps,
        )
        negative_sign = run_trial(
            environment,
            adapter,
            initial,
            PolicyAction(0.12, -0.8),
            args.steps,
            args.steady_steps,
        )
        zero_sign = run_trial(
            environment,
            adapter,
            initial,
            PolicyAction(0.12, 0.0),
            args.steps,
            args.steady_steps,
        )
        sign_passed = (
            positive_sign["actual_yaw_rate_rad_s"] > 0.0
            and negative_sign["actual_yaw_rate_rad_s"] < 0.0
            and abs(zero_sign["actual_yaw_rate_rad_s"]) < 1e-6
        )

        candidate_straight = find_trial(
            trials, args.candidate_v_max, 0.0
        )
        candidate_left = find_trial(
            trials, args.candidate_v_max, args.candidate_omega_max
        )
        candidate_right = find_trial(
            trials, args.candidate_v_max, -args.candidate_omega_max
        )
        corner_trials = (candidate_left, candidate_right)
        maximum_corner_duty = max(
            abs(trial["wheel_duty"][side])
            for trial in corner_trials
            for side in ("left", "right")
        )
        maximum_turning_radius = max(
            float(trial["turning_radius_m"]) for trial in corner_trials
        )
        corner_trials_remained_valid = all(
            trial["remained_valid"] for trial in corner_trials
        )
        candidate_passed = (
            sign_passed
            and candidate_straight["remained_valid"]
            and corner_trials_remained_valid
            and not any(trial["saturated"] for trial in corner_trials)
            and maximum_corner_duty <= args.duty_headroom_limit
            and maximum_turning_radius <= args.maximum_turning_radius_m
        )

        report = {
            "gate": "A0_action_adapter_and_envelope",
            "passed": candidate_passed,
            "map": args.map,
            "seed": args.seed,
            "frame_rate_hz": 30,
            "trial_steps": args.steps,
            "calibration": asdict(calibration),
            "sign_convention": {
                "definition": "positive omega is counter-clockwise",
                "passed": sign_passed,
                "negative_trial": negative_sign,
                "zero_trial": zero_sign,
                "positive_trial": positive_sign,
            },
            "candidate_bounds": {
                "maximum_linear_velocity_mps": args.candidate_v_max,
                "maximum_angular_velocity_rad_s": args.candidate_omega_max,
                "maximum_corner_duty": maximum_corner_duty,
                "duty_headroom_limit": args.duty_headroom_limit,
                "maximum_measured_turning_radius_m": maximum_turning_radius,
                "turning_radius_limit_m": args.maximum_turning_radius_m,
                "straight_trial_remained_valid": candidate_straight[
                    "remained_valid"
                ],
                "corner_trials_remained_valid": corner_trials_remained_valid,
                "passed": candidate_passed,
            },
            "sweep": trials,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report, indent=2))
        if not candidate_passed:
            raise SystemExit(1)
    finally:
        environment.close()


if __name__ == "__main__":
    main()
