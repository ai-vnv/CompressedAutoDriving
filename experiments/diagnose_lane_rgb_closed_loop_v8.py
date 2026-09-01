"""Development-only trace for camera-lane closed-loop failures.

The controller consumes only the normalized policy vector. Simulator truth is
read afterwards and written under explicit ``evaluation_gt_*`` columns.
"""

from __future__ import annotations

import argparse
from collections import deque
import csv
import json
from pathlib import Path

import cv2
import numpy as np

from duckie_pomdp.control import PPOCurriculumEnvironment, load_ppo_curriculum_protocol
from duckie_pomdp.evaluation.f10_ppo_policy import BeliefAwareSimpleController


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=Path, default=ROOT / "configs" / "f10_ppo_visual_v8.toml"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "artifacts" / "f10_ppo_visual_v8" / "lane_closed_loop_diagnostic",
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=(93011, 93012, 93013))
    parser.add_argument("--snapshot-window", type=int, default=48)
    args = parser.parse_args()

    config = args.config.resolve()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite diagnostic output: {output}")
    output.mkdir(parents=True)
    protocol = load_ppo_curriculum_protocol(config)
    environment = PPOCurriculumEnvironment(
        config,
        stage="c1",
        split="lane_gate_diagnostic",
        seeds=tuple(args.seeds),
    )
    controller = BeliefAwareSimpleController(protocol)
    summaries: list[dict[str, object]] = []
    try:
        for seed in args.seeds:
            rows: list[dict[str, object]] = []
            frames: deque[tuple[int, np.ndarray]] = deque(maxlen=args.snapshot_window)
            observation, _ = environment.reset(seed=seed)
            controller.reset(seed)
            last_info: dict[str, object] | None = None
            for step in range(protocol.stage("c1").episode_horizon_steps):
                action = controller.act(observation)
                observation, reward, terminated, truncated, info = environment.step(action)
                last_info = info
                belief = environment.current_belief.lane
                if belief is None:
                    raise RuntimeError("visual diagnostic requires lane belief")
                diagnostic = environment._integration.diagnostics.read()  # noqa: SLF001
                privileged = environment._integration.privileged.read()  # noqa: SLF001
                pose = diagnostic.world_pose
                gt = info["evaluation_gt"]
                rows.append(
                    {
                        "seed": seed,
                        "step": step + 1,
                        "reward": reward,
                        "progress_m": info["progress_m"],
                        "policy_lane_validity": belief.validity_probability,
                        "policy_lane_lateral_mean_m": belief.lateral_error_mean_m,
                        "policy_lane_lateral_std_m": belief.lateral_error_std_m,
                        "policy_lane_heading_mean_rad": belief.heading_error_mean_rad,
                        "policy_lane_heading_std_rad": belief.heading_error_std_rad,
                        "policy_lane_curvature_mean_inv_m": belief.curvature_mean_inv_m,
                        "policy_lane_curvature_std_inv_m": belief.curvature_std_inv_m,
                        "evaluation_gt_lateral_error_m": gt["lane_lateral_error_m"],
                        "evaluation_gt_heading_error_rad": gt["lane_heading_error_rad"],
                        "evaluation_gt_curvature_inv_m": privileged.true_pomdp_state.road.curvature_inv_m,
                        "v_cmd_mps": info["v_cmd"],
                        "omega_cmd_rad_s": info["omega_cmd"],
                        "v_actual_mps": info["v_actual"],
                        "omega_actual_rad_s": info["omega_actual"],
                        "world_x_m": pose.x_m,
                        "world_z_m": pose.z_m,
                        "world_heading_rad": pose.heading_rad,
                        "terminated": terminated,
                        "truncated": truncated,
                        "termination_reason": info["termination_reason"],
                        "truncation_reason": info["truncation_reason"],
                    }
                )
                frames.append((step + 1, environment.latest_rgb()))
                if terminated or truncated:
                    break
            if last_info is None:
                raise RuntimeError("diagnostic episode produced no step")
            csv_path = output / f"seed_{seed}.csv"
            with csv_path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            frame_dir = output / f"seed_{seed}_tail"
            frame_dir.mkdir()
            for frame_step, rgb in frames:
                cv2.imwrite(
                    str(frame_dir / f"step_{frame_step:05d}.png"),
                    cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR),
                )
            summaries.append(
                {
                    "seed": seed,
                    "steps": len(rows),
                    "completed": bool(last_info["completed"]),
                    "invalid_pose": bool(last_info["invalid_pose"]),
                    "termination_reason": last_info["termination_reason"],
                    "trace": csv_path.name,
                    "tail_frames": len(frames),
                }
            )
    finally:
        environment.close()
    payload = {
        "schema_version": 1,
        "seed_role": "development diagnostic only",
        "runtime_chain": "front_rgb -> lane model -> lane EKF -> simple controller",
        "controller_inputs": "normalized policy observation only",
        "privileged_use": "post-action diagnostic columns only",
        "episodes": summaries,
    }
    (output / "manifest.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
