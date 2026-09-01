"""Real-simulator launch gate for separated C2/C3 experiment-loop objects."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from math import hypot
from pathlib import Path

from PIL import Image

from duckie_pomdp.adapters.gym_duckietown import (
    GymDuckietownConfig,
    create_gym_duckietown,
)
from duckie_pomdp.control import PPOAgent, PPOCurriculumEnvironment
from duckie_pomdp.control.f10_protocol import file_sha256
from duckie_pomdp.control.ppo_protocol import (
    load_ppo_curriculum_protocol,
    require_stage_in_protocol_scope,
)
from duckie_pomdp.control.start_sampler import load_small_loop_tiles
from duckie_pomdp.domain.action import PolicyAction
from duckie_pomdp.evaluation.f10_ppo_policy import (
    BeliefAwareSimpleController,
    PPODeterministicPolicy,
)
from duckie_pomdp.scenario import (
    PedestrianMode,
    load_scenario,
    validate_route_object_geometry,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=Path, default=Path("configs/f10_ppo_visual_objects_v10.toml")
    )
    parser.add_argument(
        "--output", type=Path, default=Path("artifacts/f10_ppo_visual_objects_v10/object_scenario_gate.json")
    )
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    protocol = load_ppo_curriculum_protocol(args.config)
    c2 = protocol.stage("c2")
    c3 = protocol.stage("c3")
    assert c2.scenario_config_path is not None
    assert c3.scenario_config_path is not None
    duckie_scenario = load_scenario(c2.scenario_config_path)
    stop_scenario = load_scenario(c3.scenario_config_path)
    path = duckie_scenario.pedestrian.path_for_mode()
    assert path is not None

    geometry = validate_route_object_geometry(
        load_small_loop_tiles(
            map_name="experiment_loop",
            anchor_tile=(1, 0),
            anchor_heading_rad=3.141592653589793,
        ),
        pedestrian_start_world=path[0],
        pedestrian_end_world=path[1],
        stop_line_world=(
            stop_scenario.stop_line.world_x_m,
            stop_scenario.stop_line.world_z_m,
        ),
        stop_sign_world=(2.3, 1.405),
    )
    if not geometry.pedestrian_crosses_route:
        raise RuntimeError("Duckie path does not cross the ego centreline")
    if geometry.stop_route_error_m > 0.01:
        raise RuntimeError("stop line is not on the ego centreline")
    if geometry.forward_route_separation_m < 0.75:
        raise RuntimeError("Duckie crossing and stop line are not separated")

    motion = {
        mode.value: _validate_motion(duckie_scenario.with_pedestrian_mode(mode))
        for mode in (
            PedestrianMode.CROSS_LEFT_TO_RIGHT,
            PedestrianMode.CROSS_RIGHT_TO_LEFT,
        )
    }
    image_dir = args.output.parent / "object_scenario_gate_images"
    image_dir.mkdir(parents=True, exist_ok=True)
    isolation = {
        "c2": _validate_stage_isolation(
            args.config, "c2", c2.development_seeds[0], image_dir / "c2_duckie_only.png"
        ),
        "c3": _validate_stage_isolation(
            args.config, "c3", c3.development_seeds[0], image_dir / "c3_stop_only.png"
        ),
    }
    if not isolation["c2"]["stop_sign_physically_absent"]:
        raise RuntimeError("C2 stop sign is not physically absent")
    if not isolation["c3"]["pedestrian_physically_absent"]:
        raise RuntimeError("C3 pedestrian is not physically absent")
    if isolation["c3"]["stop_sign_existence_probability"] < 0.70:
        raise RuntimeError("C3 rendered stop sign is not detected by frozen YOLO")

    combined = None
    try:
        require_stage_in_protocol_scope(protocol, "c4")
    except RuntimeError:
        pass
    else:
        c4 = protocol.stage("c4")
        assert c4.scenario_config_path is not None
        isolation["c4"] = _validate_stage_isolation(
            args.config,
            "c4",
            c4.development_seeds[0],
            image_dir / "c4_combined.png",
        )
        if isolation["c4"]["pedestrian_physically_absent"]:
            raise RuntimeError("C4 Duckie is not physically present")
        if isolation["c4"]["stop_sign_physically_absent"]:
            raise RuntimeError("C4 stop sign is not physically present")
        combined_policy = BeliefAwareSimpleController(protocol)
        combined = [
            _temporal_conflict(
                args.config,
                stage="c4",
                split="training",
                seed=seed,
                policy=combined_policy,
                maximum_steps=c4.episode_horizon_steps,
            )
            for seed in c4.training_seeds[:2]
        ]
        if not all(
            row["duckie_detection_frames"] > 0
            and row["stop_sign_detection_frames"] > 0
            and row["minimum_clearance_m"] <= 0.65
            and row["pedestrian_speed_at_minimum_clearance_mps"] >= 0.15
            and row["pedestrian_exited_after_crossing"] is True
            and row["stop_completed"] is True
            and row["restarted"] is True
            and row["collision"] is False
            and row["stop_violation"] is False
            and row["completed"] is True
            for row in combined
        ):
            raise RuntimeError(
                "C4 reference did not encounter both objects and finish safely"
            )

    imported_checkpoint = (
        args.config.resolve().parent
        / str(protocol.raw["curriculum_import"]["c1"]["selected_checkpoint"])
    ).resolve()
    agent, payload = PPOAgent.load(
        imported_checkpoint,
        device=args.device,
    )
    policy = PPODeterministicPolicy(agent)
    temporal = [
        _temporal_conflict(
            args.config,
            seed=seed,
            policy=policy,
            stage="c2",
            split="development",
            maximum_steps=400,
        )
        for seed in c2.development_seeds[:2]
    ]
    if not all(
        row["duckie_detection_frames"] > 0
        and row["minimum_clearance_m"] <= 0.18
        for row in temporal
    ):
        raise RuntimeError("C1 policy did not encounter both crossing directions")

    output = {
        "schema_version": 1,
        "passed": True,
        "seed_role": "development_geometry_gate_only",
        "config": str(args.config.resolve()),
        "config_sha256": file_sha256(args.config),
        "map": str(duckie_scenario.map_path),
        "map_sha256": file_sha256(duckie_scenario.map_path),
        "imported_c1_checkpoint_sha256": file_sha256(
            imported_checkpoint
        ),
        "imported_c1_global_step": int(payload["global_step"]),
        "geometry": asdict(geometry) | {
            "pedestrian_crosses_route": geometry.pedestrian_crosses_route
        },
        "motion": motion,
        "stage_isolation": isolation,
        "temporal_conflict": temporal,
        "combined_temporal": combined,
        "images": [
            str((image_dir / "c2_duckie_only.png").resolve()),
            str((image_dir / "c3_stop_only.png").resolve()),
        ]
        + (
            [str((image_dir / "c4_combined.png").resolve())]
            if combined is not None
            else []
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(output, indent=2))


def _validate_motion(scenario) -> dict[str, object]:
    integration = create_gym_duckietown(
        GymDuckietownConfig(
            scenario=scenario,
            scenario_stop_sign_enabled=False,
            maximum_steps=260,
        )
    )
    try:
        integration.agent.reset(seed=scenario.seed)
        first = integration.privileged.read().pedestrian_world_position
        if first is None:
            raise RuntimeError("Duckie missing at crossing start")
        history = [(first.x_m, first.z_m)]
        for _ in range(210):
            integration.agent.step(PolicyAction(0.0, 0.0))
            point = integration.privileged.read().pedestrian_world_position
            if point is None:
                raise RuntimeError("Duckie disappeared during crossing")
            history.append((point.x_m, point.z_m))
        final = history[-1]
        expected = scenario.pedestrian.path_for_mode()
        assert expected is not None
        endpoint_error = (
            (final[0] - expected[1][0]) ** 2 + (final[1] - expected[1][1]) ** 2
        ) ** 0.5
        return {
            "start_world": history[0],
            "final_world": final,
            "expected_final_world": expected[1],
            "endpoint_error_m": endpoint_error,
            "minimum_z_m": min(row[1] for row in history),
            "maximum_z_m": max(row[1] for row in history),
            "crossed_lane_z_m": min(row[1] for row in history)
            <= 2.1645
            <= max(row[1] for row in history),
        }
    finally:
        integration.close()


def _validate_stage_isolation(
    config: Path, stage: str, seed: int, image_path: Path
) -> dict[str, object]:
    env = PPOCurriculumEnvironment(config, stage=stage, split="development")
    try:
        _, info = env.reset(seed=seed)
        Image.fromarray(env.latest_rgb(), mode="RGB").save(image_path)
        privileged = env._integration.privileged.read()  # evaluation gate only
        policy = info["policy"]
        return {
            "seed": seed,
            "pedestrian_mode": info["pedestrian_mode"],
            "pedestrian_physically_absent": privileged.pedestrian_world_position is None,
            "stop_sign_physically_absent": privileged.stop_sign_world_position is None,
            "pedestrian_existence_probability": policy[
                "pedestrian_existence_probability"
            ],
            "stop_sign_existence_probability": policy[
                "stop_sign_existence_probability"
            ],
            "stop_line_distance_m": policy["stop_line_distance_m"],
        }
    finally:
        env.close()


def _temporal_conflict(
    config: Path,
    *,
    stage: str,
    split: str,
    seed: int,
    policy,
    maximum_steps: int,
) -> dict[str, object]:
    env = PPOCurriculumEnvironment(config, stage=stage, split=split)
    try:
        observation, reset_info = env.reset(seed=seed)
        policy.reset(seed)
        minimum_clearance = float("inf")
        minimum_step = -1
        pedestrian_speed_at_minimum_clearance = 0.0
        pedestrian_seen_in_world = False
        pedestrian_exited_after_crossing = False
        detected = 0
        stop_detected = 0
        stop_completed = False
        restarted = False
        collision = False
        stop_violation = False
        completed = False
        termination_reason = None
        for step in range(maximum_steps):
            observation, _, terminated, truncated, info = env.step(
                policy.act(observation)
            )
            detected += int(info["perception"]["duckie_detection_count"] > 0)
            stop_detected += int(
                info["perception"]["stop_sign_detection_count"] > 0
            )
            stop_completed = stop_completed or bool(info["stop_completed"])
            restarted = restarted or (
                stop_completed and float(info["v_cmd"]) > 0.05
            )
            collision = collision or bool(info["collision"])
            stop_violation = stop_violation or bool(info["stop_violation"])
            completed = completed or bool(info["completed"])
            clearance = info["pedestrian_clearance_m"]
            privileged = env._integration.privileged.read()  # evaluation gate only
            pedestrian_present = privileged.pedestrian_world_position is not None
            pedestrian_seen_in_world = pedestrian_seen_in_world or pedestrian_present
            if pedestrian_seen_in_world and not pedestrian_present:
                pedestrian_exited_after_crossing = True
            if clearance is not None and clearance < minimum_clearance:
                minimum_clearance = float(clearance)
                minimum_step = step
                velocity = privileged.pedestrian_world_velocity
                pedestrian_speed_at_minimum_clearance = (
                    0.0
                    if velocity is None
                    else hypot(velocity.x_velocity_mps, velocity.z_velocity_mps)
                )
            termination_reason = info["termination_reason"]
            if terminated or truncated:
                break
        return {
            "seed": seed,
            "pedestrian_mode": reset_info["pedestrian_mode"],
            "steps": step + 1,
            "minimum_clearance_m": minimum_clearance,
            "minimum_clearance_step": minimum_step,
            "pedestrian_speed_at_minimum_clearance_mps": (
                pedestrian_speed_at_minimum_clearance
            ),
            "pedestrian_exited_after_crossing": pedestrian_exited_after_crossing,
            "duckie_detection_frames": detected,
            "stop_sign_detection_frames": stop_detected,
            "stop_completed": stop_completed,
            "restarted": restarted,
            "collision": collision,
            "stop_violation": stop_violation,
            "completed": completed,
            "termination_reason": termination_reason,
        }
    finally:
        env.close()


if __name__ == "__main__":
    main()
