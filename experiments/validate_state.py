"""Run the deterministic F4 true-state validation cases on the real simulator."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

from duckie_pomdp.adapters.gym_duckietown import (
    GymDuckietownConfig,
    create_gym_duckietown,
)
from duckie_pomdp.domain.action import PolicyAction
from duckie_pomdp.evaluation import StateValidationLogger
from duckie_pomdp.scenario import PedestrianMode, load_scenario


@dataclass(frozen=True)
class EpisodeSpec:
    name: str
    pedestrian_mode: PedestrianMode
    action: PolicyAction
    maximum_steps: int
    stop_after_line_distance_m: float | None = None


EPISODES = (
    EpisodeSpec("A_stationary", PedestrianMode.STATIONARY, PolicyAction(0.0, 0.0), 10),
    EpisodeSpec("B_ego_forward", PedestrianMode.STATIONARY, PolicyAction(0.2, 0.0), 30),
    EpisodeSpec("C_ego_turn", PedestrianMode.STATIONARY, PolicyAction(0.0, 1.0), 20),
    EpisodeSpec("D_ped_cross_ltr", PedestrianMode.CROSS_LEFT_TO_RIGHT, PolicyAction(0.0, 0.0), 80),
    EpisodeSpec(
        "E_stop_line_crossing",
        PedestrianMode.STATIONARY,
        PolicyAction(0.4, 0.0),
        90,
        stop_after_line_distance_m=-0.05,
    ),
)


def run_validation(
    scenario_path: Path,
    output_path: Path,
) -> StateValidationLogger:
    base_scenario = load_scenario(scenario_path)
    logger = StateValidationLogger()
    for episode in EPISODES:
        scenario = base_scenario.with_pedestrian_mode(episode.pedestrian_mode)
        integration = create_gym_duckietown(
            GymDuckietownConfig(
                scenario=scenario,
                camera_width=160,
                camera_height=120,
            )
        )
        try:
            integration.agent.reset(seed=scenario.seed)
            logger.record(
                episode=episode.name,
                step=0,
                timestamp=0.0,
                privileged=integration.privileged.read(),
                action=PolicyAction(0.0, 0.0),
            )
            for step in range(1, episode.maximum_steps + 1):
                transition = integration.agent.step(episode.action)
                diagnostics = integration.diagnostics.read()
                privileged = integration.privileged.read()
                row = logger.record(
                    episode=episode.name,
                    step=step,
                    timestamp=diagnostics.timestamp_s,
                    privileged=privileged,
                    action=episode.action,
                )
                if transition.terminated or transition.truncated:
                    raise RuntimeError(
                        f"{episode.name} ended early with {diagnostics.done_code}"
                    )
                threshold = episode.stop_after_line_distance_m
                if threshold is not None and row.stop_line_distance < threshold:
                    break
        finally:
            integration.close()
    logger.write_csv(output_path)
    return logger


def _summary(logger: StateValidationLogger, output: Path) -> dict[str, object]:
    episodes: dict[str, list] = {}
    for row in logger.rows:
        episodes.setdefault(row.episode, []).append(row)
    stationary = episodes["A_stationary"][-1]
    forward = episodes["B_ego_forward"]
    turn = episodes["C_ego_turn"]
    crossing = episodes["D_ped_cross_ltr"]
    stop_line = episodes["E_stop_line_crossing"]
    return {
        "output": str(output),
        "rows": len(logger.rows),
        "stationary_rates_zero": abs(stationary.ped_rdot) < 1e-9
        and abs(stationary.ped_betadot) < 1e-9,
        "forward_range_decreased": forward[-1].ped_r < forward[0].ped_r,
        "forward_relative_rdot_negative": min(row.ped_rdot for row in forward[1:]) < 0.0,
        "turn_changed_bearing": abs(turn[-1].ped_beta - turn[0].ped_beta) > 0.05,
        "crossing_changed_bearing_sign": crossing[0].ped_beta > 0.0
        and crossing[-1].ped_beta < 0.0,
        "stop_line_crossed": stop_line[0].stop_line_distance > 0.0
        and stop_line[-1].stop_line_distance < 0.0,
    }


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
        default=Path("artifacts/state_validation.csv"),
    )
    args = parser.parse_args()
    logger = run_validation(args.scenario, args.output)
    print(json.dumps(_summary(logger, args.output), indent=2))


if __name__ == "__main__":
    main()
