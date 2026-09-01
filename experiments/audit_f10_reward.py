"""Real-simulator pre-SAC sanity audit for the frozen F10 reward."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, replace
from pathlib import Path
from typing import Callable

import numpy as np

from duckie_pomdp.adapters.gym_duckietown import GymDuckietownConfig, create_gym_duckietown
from duckie_pomdp.control import (
    F10RewardConfig,
    F10RewardEvaluator,
    F10RoadObserver,
    load_f10_protocol,
)
from duckie_pomdp.control.f10_protocol import file_sha256
from duckie_pomdp.domain.action import PolicyAction
from duckie_pomdp.domain.observation import SensorObservation
from duckie_pomdp.scenario import PedestrianMode, load_scenario


ROOT = Path(__file__).resolve().parents[1]


class _SimpleController:
    def __init__(self) -> None:
        self._hold = 0
        self._stop_completed = False

    def __call__(self, observation: SensorObservation, _: np.random.Generator) -> PolicyAction:
        if observation.road is None:
            raise RuntimeError("simple controller requires road measurement")
        distance = observation.road.stop_line_distance_m
        if not self._stop_completed and 0.0 <= distance <= 0.24:
            if observation.ego.linear_velocity_mps <= 0.025:
                self._hold += 1
            if self._hold >= 15:
                self._stop_completed = True
            velocity = 0.0
        elif not self._stop_completed and distance <= 0.40:
            velocity = 0.08
        else:
            velocity = 0.28
        omega = np.clip(
            -1.5 * observation.ego.lateral_error_m
            - 0.8 * observation.ego.heading_error_rad,
            -0.8,
            0.8,
        )
        return PolicyAction(velocity, float(omega))


def _random(_: SensorObservation, rng: np.random.Generator) -> PolicyAction:
    return PolicyAction(float(rng.uniform(0.0, 0.4)), float(rng.uniform(-4.0, 4.0)))


def _always_stop(_: SensorObservation, __: np.random.Generator) -> PolicyAction:
    return PolicyAction(0.0, 0.0)


def _constant_forward(_: SensorObservation, __: np.random.Generator) -> PolicyAction:
    return PolicyAction(0.30, 0.0)


def _scenario(protocol, seed: int, index: int):
    modes = tuple(PedestrianMode(value) for value in protocol.raw["scenario_distribution"]["pedestrian_modes"])
    base = load_scenario(protocol.scenario_path).with_pedestrian_mode(modes[index % len(modes)])
    distribution = protocol.raw["scenario_distribution"]
    rng = np.random.default_rng(seed + 100_003 * index)
    pose = base.ego_start_pose_m
    return replace(
        base,
        seed=seed,
        ego_start_pose_m=(
            float(rng.uniform(*distribution["start_longitudinal_range_m"])),
            pose[1],
            pose[2] + float(rng.uniform(*distribution["start_lateral_offset_range_m"])),
        ),
        ego_heading_rad=base.ego_heading_rad + float(
            rng.uniform(*distribution["start_heading_offset_range_rad"])
        ),
    )


def _episode(protocol, seed: int, index: int, policy_name: str) -> dict[str, object]:
    scenario = _scenario(protocol, seed, index)
    simulator = protocol.raw["simulator"]
    integration = create_gym_duckietown(
        GymDuckietownConfig(
            scenario=scenario,
            domain_randomization=bool(simulator["domain_randomization"]),
            dynamics_randomization=bool(simulator["dynamics_randomization"]),
            frame_rate_hz=int(simulator["frame_rate_hz"]),
            frame_skip=int(simulator["frame_skip"]),
            maximum_steps=int(simulator["episode_horizon_steps"]) + 2,
            camera_width=int(simulator["camera_width_px"]),
            camera_height=int(simulator["camera_height_px"]),
        )
    )
    controller: Callable[[SensorObservation, np.random.Generator], PolicyAction]
    controller = {
        "random": _random,
        "always_stop": _always_stop,
        "constant_forward": _constant_forward,
        "simple_controller": _SimpleController(),
    }[policy_name]
    rng = np.random.default_rng(seed + 300_007 * index)
    reward = F10RewardEvaluator(
        F10RewardConfig.from_protocol(protocol),
        route_heading_rad=scenario.stop_line.route_heading_rad,
    )
    totals = {name: 0.0 for name in ("progress", "lane", "stop", "pedestrian", "comfort", "terminal")}
    try:
        road_observer = F10RoadObserver(
            scenario,
            map_tile_size_m=float(protocol.raw["road_observer"]["map_tile_size_m"]),
        )
        raw_observation = integration.agent.reset(seed=seed)
        observation = SensorObservation(
            raw_observation.front_rgb,
            raw_observation.ego,
            road_observer.reset(),
        )
        reward.reset(observation, integration.privileged.read())
        outcome = None
        timestamp_s = 0.0
        for step in range(int(simulator["episode_horizon_steps"])):
            action = controller(observation, rng)
            transition = integration.agent.step(action)
            diagnostics = integration.diagnostics.read()
            dt_s = diagnostics.timestamp_s - timestamp_s
            if dt_s <= 0.0:
                dt_s = 1.0 / float(simulator["frame_rate_hz"])
            timestamp_s = diagnostics.timestamp_s
            observation = SensorObservation(
                transition.observation.front_rgb,
                transition.observation.ego,
                road_observer.update(transition.observation.ego, dt_s=dt_s),
            )
            outcome = reward.evaluate(
                action=action,
                observation=observation,
                privileged=integration.privileged.read(),
                simulator_terminated=transition.terminated,
                simulator_truncated=transition.truncated,
                simulator_done_code=diagnostics.done_code,
                horizon_reached=step + 1 >= int(simulator["episode_horizon_steps"]),
            )
            for name in totals:
                totals[name] += getattr(outcome.reward_terms, name)
            if outcome.terminated or outcome.truncated:
                break
        assert outcome is not None
        return {
            "policy": policy_name,
            "seed": seed,
            "pedestrian_mode": scenario.pedestrian.mode.value,
            "steps": step + 1,
            "return": sum(totals.values()),
            "progress_m": outcome.progress_m,
            "collision": outcome.pedestrian_collision,
            "invalid_pose": outcome.invalid_pose,
            "stop_completed": outcome.stop_completed,
            "stop_violation": outcome.stop_violation,
            "termination_reason": outcome.termination_reason,
            "truncation_reason": outcome.truncation_reason,
            "reward_terms": totals,
        }
    finally:
        integration.close()


def run(config: Path, output: Path) -> dict[str, object]:
    protocol = load_f10_protocol(config)
    seeds = protocol.seeds.training[:3]
    policies = ("random", "always_stop", "constant_forward", "simple_controller")
    episodes = [
        _episode(protocol, seed, index, policy)
        for policy in policies
        for index, seed in enumerate(seeds)
    ]
    summaries = {}
    for policy in policies:
        rows = [row for row in episodes if row["policy"] == policy]
        summaries[policy] = {
            "episodes": len(rows),
            "mean_return": float(np.mean([row["return"] for row in rows])),
            "mean_progress_m": float(np.mean([row["progress_m"] for row in rows])),
            "collision_rate": float(np.mean([row["collision"] for row in rows])),
            "invalid_pose_rate": float(np.mean([row["invalid_pose"] for row in rows])),
            "stop_completion_rate": float(np.mean([row["stop_completed"] for row in rows])),
            "stop_violation_rate": float(np.mean([row["stop_violation"] for row in rows])),
        }
    checks = {
        "simple_beats_always_stop_return": summaries["simple_controller"]["mean_return"] > summaries["always_stop"]["mean_return"],
        "simple_beats_random_return": summaries["simple_controller"]["mean_return"] > summaries["random"]["mean_return"],
        "simple_beats_always_stop_progress": summaries["simple_controller"]["mean_progress_m"] > summaries["always_stop"]["mean_progress_m"],
        "reckless_forward_not_best": summaries["constant_forward"]["mean_return"] < summaries["simple_controller"]["mean_return"],
        "always_stop_not_strong": summaries["always_stop"]["mean_return"] < 0.0,
    }
    result = {
        "schema_version": 1,
        "config": str(config.resolve()),
        "config_sha256": file_sha256(config),
        "seeds": list(seeds),
        "summaries": summaries,
        "checks": checks,
        "passed": all(checks.values()),
        "episodes": episodes,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "f10_sac_v1.toml")
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts" / "f10" / "reward_audit.json")
    args = parser.parse_args()
    result = run(args.config, args.output)
    print(json.dumps({"passed": result["passed"], "summaries": result["summaries"], "checks": result["checks"]}, indent=2))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
