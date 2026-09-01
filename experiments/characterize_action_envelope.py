"""Measure the real Gym-Duckietown chassis-command response envelope."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import fmean, pstdev

from duckie_pomdp.adapters.gym_duckietown import (
    GymDuckietownConfig,
    create_gym_duckietown,
)
from duckie_pomdp.domain.action import PolicyAction


@dataclass(frozen=True)
class SweepCase:
    v_cmd: float
    omega_cmd: float


@dataclass(frozen=True)
class EnvelopeResult:
    seed: int
    v_cmd: float
    omega_cmd: float
    v_actual_mean: float
    v_actual_std: float
    omega_actual_mean: float
    omega_actual_std: float
    wheel_left: float
    wheel_right: float
    raw_wheel_left: float
    raw_wheel_right: float
    clipped: bool
    offroad: bool
    terminated: bool
    truncated: bool
    transient_steps: int
    steady_window_start_step: int
    steady_sample_count: int
    transient_v_actual_mean: float
    transient_omega_actual_mean: float
    max_abs_lane_deviation_m: float
    final_x_m: float
    final_z_m: float
    final_heading_rad: float
    done_code: str


def _default_cases() -> list[SweepCase]:
    cases = [SweepCase(v_cmd, 0.0) for v_cmd in (0.0, 0.1, 0.2, 0.3, 0.4, 0.5)]
    cases.extend(
        SweepCase(0.2, omega_cmd)
        for omega_cmd in (-4.0, -2.0, -0.8, 0.8, 2.0, 4.0)
    )
    cases.extend((SweepCase(0.4, -4.0), SweepCase(0.4, 4.0)))
    return cases


def _stats(values: list[float]) -> tuple[float, float]:
    if not values:
        return 0.0, 0.0
    return fmean(values), pstdev(values)


def _run_case(
    integration,
    case: SweepCase,
    *,
    seed: int,
    transient_steps: int,
    steady_steps: int,
) -> EnvelopeResult:
    integration.agent.reset(seed=seed)
    command = PolicyAction(case.v_cmd, case.omega_cmd)
    speeds: list[float] = []
    yaw_rates: list[float] = []
    lane_deviations: list[float] = []
    any_offroad = False
    any_terminated = False
    any_truncated = False
    diagnostics = None

    for _ in range(transient_steps + steady_steps):
        transition = integration.agent.step(command)
        diagnostics = integration.diagnostics.read()
        speeds.append(transition.observation.ego.linear_velocity_mps)
        yaw_rates.append(transition.observation.ego.yaw_rate_rad_s)
        lane_deviations.append(transition.observation.ego.lateral_error_m)
        any_offroad = any_offroad or diagnostics.offroad
        any_terminated = any_terminated or transition.terminated
        any_truncated = any_truncated or transition.truncated
        if transition.terminated or transition.truncated:
            break

    if diagnostics is None:
        raise RuntimeError("envelope case produced no simulator step")

    steady_speeds = speeds[transient_steps:]
    steady_yaw_rates = yaw_rates[transient_steps:]
    if not steady_speeds:
        steady_speeds = speeds
        steady_yaw_rates = yaw_rates
    v_mean, v_std = _stats(steady_speeds)
    omega_mean, omega_std = _stats(steady_yaw_rates)
    conversion = diagnostics.action_conversion
    return EnvelopeResult(
        seed=seed,
        v_cmd=case.v_cmd,
        omega_cmd=case.omega_cmd,
        v_actual_mean=v_mean,
        v_actual_std=v_std,
        omega_actual_mean=omega_mean,
        omega_actual_std=omega_std,
        wheel_left=conversion.wheel_command.left,
        wheel_right=conversion.wheel_command.right,
        raw_wheel_left=conversion.unclipped_wheel_command.left,
        raw_wheel_right=conversion.unclipped_wheel_command.right,
        clipped=conversion.saturated,
        offroad=any_offroad,
        terminated=any_terminated,
        truncated=any_truncated,
        transient_steps=min(transient_steps, len(speeds)),
        steady_window_start_step=transient_steps + 1,
        steady_sample_count=len(steady_speeds),
        transient_v_actual_mean=fmean(speeds[:transient_steps]),
        transient_omega_actual_mean=fmean(yaw_rates[:transient_steps]),
        max_abs_lane_deviation_m=max(abs(value) for value in lane_deviations),
        final_x_m=diagnostics.world_pose.x_m,
        final_z_m=diagnostics.world_pose.z_m,
        final_heading_rad=diagnostics.world_pose.heading_rad,
        done_code=diagnostics.done_code,
    )


def run_sweep(
    output_path: Path,
    *,
    seed: int,
    transient_steps: int,
    steady_steps: int,
) -> list[EnvelopeResult]:
    integration = create_gym_duckietown(
        GymDuckietownConfig(
            map_name="small_loop",
            seed=seed,
            domain_randomization=False,
            dynamics_randomization=False,
            headless=True,
            start_tile=(1, 0),
            start_pose=((0.065, 0.0, 0.4095), 0.0),
        )
    )
    try:
        results = [
            _run_case(
                integration,
                case,
                seed=seed,
                transient_steps=transient_steps,
                steady_steps=steady_steps,
            )
            for case in _default_cases()
        ]
    finally:
        integration.close()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows = [asdict(result) for result in results]
    with output_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("artifacts/action_envelope.csv"))
    parser.add_argument("--seed", type=int, default=73)
    parser.add_argument("--transient-steps", type=int, default=10)
    parser.add_argument("--steady-steps", type=int, default=20)
    args = parser.parse_args()
    if args.transient_steps <= 0 or args.steady_steps <= 0:
        parser.error("transient and steady step counts must be positive")

    results = run_sweep(
        args.output,
        seed=args.seed,
        transient_steps=args.transient_steps,
        steady_steps=args.steady_steps,
    )
    print(
        json.dumps(
            {
                "output": str(args.output),
                "cases": len(results),
                "clipped_cases": sum(result.clipped for result in results),
                "offroad_cases": sum(result.offroad for result in results),
                "terminated_cases": sum(result.terminated for result in results),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
