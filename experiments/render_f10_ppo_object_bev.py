"""Render frozen PPO object stages as front RGB + evaluation-only BEV.

The front panel is the exact RGB sensor consumed by the runtime pipeline.  A
second invocation of the same frozen YOLO checkpoint is used only to draw audit
boxes; it never feeds the policy or belief updater.  The simulator top-down
panel and privileged labels are evaluation-only.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import cv2
import imageio.v2 as imageio
import numpy as np

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10
    import tomli as tomllib

from duckie_pomdp.control import PPOAgent, PPOCurriculumEnvironment
from duckie_pomdp.control.f10_protocol import file_sha256
from duckie_pomdp.control.ppo_protocol import load_ppo_curriculum_protocol
from duckie_pomdp.perception.yolo_detector import YoloObjectDetector


ROOT = Path(__file__).resolve().parents[1]


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--stage", choices=("c2", "c3", "c4"), required=True)
    parser.add_argument(
        "--split", choices=("development", "stage_final"), default="stage_final"
    )
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _draw_detections(front: np.ndarray, detections) -> np.ndarray:
    canvas = np.ascontiguousarray(front.copy())
    colours = {"duckie": (255, 215, 0), "stop_sign": (255, 80, 80)}
    for detection in detections:
        box = detection.bounding_box
        name = detection.object_class.value
        colour = colours.get(name, (120, 240, 160))
        p1 = (int(round(box.x_min_px)), int(round(box.y_min_px)))
        p2 = (int(round(box.x_max_px)), int(round(box.y_max_px)))
        cv2.rectangle(canvas, p1, p2, colour, 2)
        cv2.putText(
            canvas,
            f"{name} {detection.confidence:.2f}",
            (p1[0], max(16, p1[1] - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            colour,
            1,
            cv2.LINE_AA,
        )
    return canvas


def _compose(front: np.ndarray, bev: np.ndarray) -> np.ndarray:
    height = int(bev.shape[0])
    front_width = int(round(front.shape[1] * height / front.shape[0]))
    scaled = cv2.resize(front, (front_width, height), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((height, front_width + bev.shape[1], 3), dtype=np.uint8)
    canvas[:, :front_width] = scaled
    canvas[:, front_width:] = bev
    cv2.line(canvas, (front_width, 0), (front_width, height), (50, 50, 50), 2)
    cv2.putText(
        canvas,
        "FRONT RGB + YOLO AUDIT OVERLAY",
        (10, height - 14),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (220, 220, 220),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        canvas,
        "BEV (EVALUATION ONLY)",
        (front_width + 10, height - 14),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (220, 220, 220),
        1,
        cv2.LINE_AA,
    )
    return canvas


def _put_lines(canvas: np.ndarray, lines: list[str], x: int, y: int) -> None:
    for index, line in enumerate(lines):
        cv2.putText(
            canvas,
            line,
            (x, y + 19 * index),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (245, 245, 245),
            1,
            cv2.LINE_AA,
        )


def _annotate(
    canvas: np.ndarray,
    *,
    stage: str,
    seed: int,
    step: int,
    info: dict[str, Any],
    reward: float,
    total_return: float,
) -> np.ndarray:
    # Telemetry must not occlude the top/north part of the BEV loop.  Preserve
    # the complete composed image and append a separate information strip.
    canvas = np.ascontiguousarray(canvas)
    image_height, image_width = canvas.shape[:2]
    strip_height = 126
    result = np.zeros((image_height + strip_height, image_width, 3), dtype=np.uint8)
    result[:image_height] = canvas
    cv2.line(
        result,
        (0, image_height + 2),
        (image_width, image_height + 2),
        (55, 55, 55),
        2,
    )
    policy = info.get("policy", {}) or {}
    perception = info.get("perception", {}) or {}
    _put_lines(
        result,
        [
            f"F10-PPO {stage.upper()} | seed {seed} | step {step}",
            f"YOLO duckie detections: {int(perception.get('duckie_detection_count', 0))}",
            "belief: P(e) %.3f  r %.3f +/- %.3f m"
            % (
                float(policy.get("pedestrian_existence_probability", 0.0)),
                float(policy.get("pedestrian_range_mean_m", 0.0)),
                float(policy.get("pedestrian_range_std_m", 0.0)),
            ),
            "belief: beta %+.3f +/- %.3f rad"
            % (
                float(policy.get("pedestrian_bearing_mean_rad", 0.0)),
                float(policy.get("pedestrian_bearing_std_rad", 0.0)),
            ),
            "lane: valid %.2f  d %+.3f m  phi %+.3f rad"
            % (
                float(policy.get("lane_validity_probability", 0.0)),
                float(policy.get("lane_lateral_error_mean_m", 0.0)),
                float(policy.get("lane_heading_error_mean_rad", 0.0)),
            ),
        ],
        10,
        image_height + 21,
    )
    _put_lines(
        result,
        [
            "action: v %.3f m/s  omega %+.3f rad/s"
            % (float(info.get("v_cmd", 0.0)), float(info.get("omega_cmd", 0.0))),
            f"progress {float(info.get('progress_m', 0.0)):.2f} m",
            f"reward {reward:+.3f}  return {total_return:+.2f}",
            "event: %s"
            % (
                info.get("termination_reason")
                or info.get("truncation_reason")
                or "running"
            ),
            "GT/BEV are visualization only; PPO input remains 29D belief",
        ],
        result.shape[1] // 2,
        image_height + 21,
    )
    return result


def render(args: argparse.Namespace) -> dict[str, Any]:
    config = args.config.resolve()
    checkpoint = args.checkpoint.resolve()
    output = args.output.resolve()
    protocol = load_ppo_curriculum_protocol(config)
    stage = protocol.stage(args.stage)
    allowed = (
        stage.development_seeds
        if args.split == "development"
        else stage.stage_final_seeds
    )
    if args.seed not in allowed:
        raise ValueError(f"seed {args.seed} is outside frozen {args.split}: {allowed}")
    if file_sha256(checkpoint) != args.checkpoint_sha256:
        raise RuntimeError("frozen checkpoint SHA256 mismatch")

    destinations = (output, output.with_suffix(".json"), output.with_suffix(".png"))
    for destination in destinations:
        if destination.exists() and not args.overwrite:
            raise FileExistsError(f"refusing to overwrite {destination}")
        destination.parent.mkdir(parents=True, exist_ok=True)

    with protocol.belief_config_path.open("rb") as stream:
        detector_settings = tomllib.load(stream)["detector"]
    detector = YoloObjectDetector(
        protocol.detector_checkpoint_path,
        confidence_threshold=float(detector_settings["confidence_threshold"]),
        iou_threshold=float(detector_settings["nms_iou_threshold"]),
        image_size=int(detector_settings["image_size"]),
        device=str(detector_settings["device"]),
        max_detections=int(detector_settings["max_detections"]),
    )
    agent, payload = PPOAgent.load(checkpoint, device=args.device)
    environment = PPOCurriculumEnvironment(config, stage=args.stage, split=args.split)
    writer = imageio.get_writer(
        output,
        fps=args.fps,
        codec="libx264",
        quality=8,
        pixelformat="yuv420p",
        macro_block_size=None,
        ffmpeg_log_level="warning",
    )
    frames = 0
    total_return = 0.0
    last_info: dict[str, Any] | None = None
    last_frame: np.ndarray | None = None
    try:
        observation, _ = environment.reset(seed=args.seed)
        simulator = environment._integration.agent._session._simulator
        limit = min(
            stage.episode_horizon_steps,
            args.max_steps if args.max_steps is not None else stage.episode_horizon_steps,
        )
        for step in range(1, limit + 1):
            action = agent.act(observation, deterministic=True).environment_action
            observation, reward, terminated, truncated, info = environment.step(action)
            front = environment.latest_rgb()
            # Visualization-only inference; the policy already consumed its own
            # frozen RGB->YOLO->belief result inside environment.step().
            detections = detector.detect(front)
            bev = np.asarray(simulator.render(mode="top_down")).copy()
            total_return += float(reward)
            last_info = info
            last_frame = _annotate(
                _compose(_draw_detections(front, detections), bev),
                stage=args.stage,
                seed=args.seed,
                step=step,
                info=info,
                reward=float(reward),
                total_return=total_return,
            )
            writer.append_data(last_frame)
            frames += 1
            if terminated or truncated:
                break
    finally:
        writer.close()
        environment.close()
    if last_info is None or last_frame is None:
        raise RuntimeError("render produced no frames")
    cv2.imwrite(str(output.with_suffix(".png")), cv2.cvtColor(last_frame, cv2.COLOR_RGB2BGR))
    manifest = {
        "schema_version": 1,
        "stage": args.stage,
        "split": args.split,
        "seed": args.seed,
        "config": str(config),
        "config_sha256": file_sha256(config),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": file_sha256(checkpoint),
        "checkpoint_step": int(payload["global_step"]),
        "detector_checkpoint_sha256": protocol.detector_checkpoint_sha256,
        "runtime_policy_path": "front RGB -> YOLO -> metric measurement -> F9c belief -> 29D PPO",
        "visualization_only": ["second YOLO overlay", "simulator top_down"],
        "video": str(output),
        "video_sha256": file_sha256(output),
        "frames": frames,
        "fps": args.fps,
        "duration_s": frames / float(args.fps),
        "progress_m": float(last_info.get("progress_m", 0.0)),
        "completed": bool(last_info.get("completed", False)),
        "collision": bool(last_info.get("collision", False)),
        "termination_reason": last_info.get("termination_reason"),
        "truncation_reason": last_info.get("truncation_reason"),
    }
    output.with_suffix(".json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def main() -> None:
    print(json.dumps(render(_args()), indent=2))


if __name__ == "__main__":
    main()
