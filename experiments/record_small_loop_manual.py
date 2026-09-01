"""Record a deterministic manual-controller drive on Gym-Duckietown small_loop.

This is a visual smoke test, not a learned policy evaluation. The controller
uses only the agent-visible lane deviation and heading error to issue the
project's chassis-level ``PolicyAction(v_cmd, omega_cmd)``.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from math import pi
from pathlib import Path

import cv2
import imageio.v2 as imageio
import numpy as np

from duckie_pomdp.adapters.gym_duckietown import (
    GymDuckietownConfig,
    create_gym_duckietown,
)
from duckie_pomdp.domain.action import PolicyAction
from duckie_pomdp.domain.observation import SensorObservation


@dataclass(frozen=True)
class ManualControllerConfig:
    cruise_velocity_mps: float = 0.20
    minimum_velocity_mps: float = 0.10
    lateral_gain_per_s: float = 5.0
    heading_gain_per_s: float = 3.0
    maximum_yaw_rate_rad_s: float = 4.0


class ManualLaneController:
    """Small deterministic P-controller for visual simulator inspection."""

    def __init__(self, config: ManualControllerConfig) -> None:
        self._config = config

    def command(self, observation: SensorObservation) -> PolicyAction:
        ego = observation.ego
        # Gym-Duckietown's lane-pose errors are error coordinates, so the
        # stabilizing chassis yaw command has the same feedback sign here.
        yaw_rate = (
            self._config.lateral_gain_per_s * ego.lateral_error_m
            + self._config.heading_gain_per_s * ego.heading_error_rad
        )
        yaw_rate = float(
            np.clip(
                yaw_rate,
                -self._config.maximum_yaw_rate_rad_s,
                self._config.maximum_yaw_rate_rad_s,
            )
        )

        turn_fraction = min(
            abs(yaw_rate) / self._config.maximum_yaw_rate_rad_s,
            1.0,
        )
        velocity = self._config.cruise_velocity_mps * (1.0 - 0.45 * turn_fraction)
        velocity = max(velocity, self._config.minimum_velocity_mps)
        return PolicyAction(velocity, yaw_rate)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--duration-s", type=float, default=120.0)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--seed", type=int, default=73)
    parser.add_argument(
        "--direction",
        choices=("clockwise", "counterclockwise"),
        default="clockwise",
    )
    parser.add_argument("--cruise-velocity-mps", type=float, default=0.20)
    parser.add_argument("--maximum-yaw-rate-rad-s", type=float, default=4.0)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/videos/small_loop_manual_120s.mp4"),
    )
    return parser.parse_args()


def _annotate_frame(
    frame_rgb: np.ndarray,
    *,
    frame_index: int,
    fps: int,
    observation: SensorObservation,
    action: PolicyAction,
    direction: str,
) -> np.ndarray:
    canvas = np.ascontiguousarray(frame_rgb.copy())
    panel = canvas.copy()
    cv2.rectangle(panel, (0, 0), (canvas.shape[1], 112), (0, 0, 0), -1)
    cv2.addWeighted(panel, 0.62, canvas, 0.38, 0.0, canvas)

    ego = observation.ego
    lines = (
        f"small_loop | {direction} | manual P-controller | no learned policy",
        f"time {frame_index / fps:6.2f} s    frame {frame_index:04d}",
        f"cmd:    v {action.linear_velocity_mps:+.3f} m/s    "
        f"omega {action.angular_velocity_rad_s:+.3f} rad/s",
        f"actual: v {ego.linear_velocity_mps:+.3f} m/s    "
        f"omega {ego.yaw_rate_rad_s:+.3f} rad/s    "
        f"d {ego.lateral_error_m:+.3f} m    phi {ego.heading_error_rad:+.3f} rad",
    )
    for row, line in enumerate(lines):
        cv2.putText(
            canvas,
            line,
            (14, 25 + 27 * row),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
    return canvas


def main() -> None:
    args = parse_args()
    if args.duration_s <= 0.0:
        raise SystemExit("duration-s must be positive")
    if args.fps <= 0:
        raise SystemExit("fps must be positive")

    total_frames = int(round(args.duration_s * args.fps))
    controller_config = ManualControllerConfig(
        cruise_velocity_mps=args.cruise_velocity_mps,
        maximum_yaw_rate_rad_s=args.maximum_yaw_rate_rad_s,
    )
    controller = ManualLaneController(controller_config)
    start_pose = (
        ((0.065, 0.0, 0.4095), 0.0)
        if args.direction == "clockwise"
        else ((0.520, 0.0, 0.1755), pi)
    )
    integration = create_gym_duckietown(
        GymDuckietownConfig(
            map_name="small_loop",
            seed=args.seed,
            domain_randomization=False,
            dynamics_randomization=False,
            frame_rate_hz=args.fps,
            frame_skip=1,
            maximum_steps=total_frames + 5,
            headless=True,
            start_tile=(1, 0),
            start_pose=start_pose,
        )
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    metadata_path = args.output.with_suffix(".json")
    observation = integration.agent.reset(seed=args.seed)
    lane_errors: list[float] = []
    heading_errors: list[float] = []
    actual_velocities: list[float] = []
    actual_yaw_rates: list[float] = []
    saturation_count = 0
    termination_count = 0

    writer = imageio.get_writer(
        args.output,
        fps=args.fps,
        codec="libx264",
        quality=8,
        pixelformat="yuv420p",
        macro_block_size=None,
        ffmpeg_log_level="warning",
    )
    try:
        for frame_index in range(total_frames):
            action = controller.command(observation)
            transition = integration.agent.step(action)
            observation = transition.observation
            diagnostics = integration.diagnostics.read()

            writer.append_data(
                _annotate_frame(
                    observation.front_rgb,
                    frame_index=frame_index,
                    fps=args.fps,
                    observation=observation,
                    action=action,
                    direction=args.direction,
                )
            )
            lane_errors.append(observation.ego.lateral_error_m)
            heading_errors.append(observation.ego.heading_error_rad)
            actual_velocities.append(observation.ego.linear_velocity_mps)
            actual_yaw_rates.append(observation.ego.yaw_rate_rad_s)
            saturation_count += int(diagnostics.action_conversion.saturated)

            if transition.terminated or transition.truncated:
                termination_count += 1
                observation = integration.agent.reset(seed=args.seed)
    finally:
        writer.close()
        integration.close()

    metadata = {
        "map": "small_loop",
        "direction": args.direction,
        "seed": args.seed,
        "duration_s": total_frames / args.fps,
        "fps": args.fps,
        "frames": total_frames,
        "controller": asdict(controller_config),
        "termination_count": termination_count,
        "saturation_count": saturation_count,
        "maximum_absolute_lane_error_m": max(abs(value) for value in lane_errors),
        "maximum_absolute_heading_error_rad": max(
            abs(value) for value in heading_errors
        ),
        "mean_actual_velocity_mps": float(np.mean(actual_velocities)),
        "mean_absolute_actual_yaw_rate_rad_s": float(
            np.mean(np.abs(actual_yaw_rates))
        ),
        "video": str(args.output),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
