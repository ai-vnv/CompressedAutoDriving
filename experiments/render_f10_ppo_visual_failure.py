"""Render an audited visual-belief PPO episode and its terminal failure.

The policy path is unchanged from evaluation: front RGB is converted to the
runtime lane belief, the frozen PPO receives the normalized 29-D belief vector,
and its action passes through the existing environment action adapter.  Ground
truth lane values are displayed only in an explicitly marked evaluation panel.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import cv2
import imageio.v2 as imageio
import numpy as np

from duckie_pomdp.control import PPOAgent, PPOCurriculumEnvironment
from duckie_pomdp.control.f10_protocol import file_sha256
from duckie_pomdp.control.ppo_protocol import load_ppo_curriculum_protocol


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "f10_ppo_visual_v2.toml"
DEFAULT_STAGE_DIR = ROOT / "artifacts" / "f10_ppo_visual_v2" / "c0"
DEFAULT_OUTPUT = DEFAULT_STAGE_DIR / "c0_yellow_crossing_seed40101.mp4"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--stage-dir", type=Path, default=DEFAULT_STAGE_DIR)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--seed", type=int, default=40101)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _put_lines(
    canvas: np.ndarray,
    lines: tuple[tuple[str, tuple[int, int, int]], ...],
    *,
    x: int,
    y: int,
    spacing: int = 21,
) -> None:
    for index, (line, color) in enumerate(lines):
        cv2.putText(
            canvas,
            line,
            (x, y + spacing * index),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.43,
            color,
            1,
            cv2.LINE_AA,
        )


def _annotate(
    frame_rgb: np.ndarray,
    *,
    step: int,
    checkpoint_step: int,
    info: dict[str, Any],
    reward: float,
    total_return: float,
) -> np.ndarray:
    canvas = np.ascontiguousarray(frame_rgb.copy())
    panel = canvas.copy()
    panel_height = min(210, canvas.shape[0])
    cv2.rectangle(panel, (0, 0), (canvas.shape[1], panel_height), (0, 0, 0), -1)
    cv2.addWeighted(panel, 0.72, canvas, 0.28, 0.0, canvas)

    policy = info["policy"]
    truth = info["evaluation_gt"]
    perception = info["perception"]
    white = (255, 255, 255)
    cyan = (80, 240, 255)
    yellow = (255, 230, 70)
    red = (255, 90, 90)
    green = (100, 255, 120)
    event = info["termination_reason"] or info["truncation_reason"] or "running"
    event_color = red if event != "running" else green

    left = (
        ("F10 PPO VISUAL BELIEF | C0 small_loop | COUNTER-CLOCKWISE", white),
        (f"checkpoint {checkpoint_step:,} | seed 40101 | step {step:04d}", white),
        ("POLICY BELIEF (camera-derived)", cyan),
        (
            "valid %.3f | points %d | detected %s"
            % (
                float(policy["lane_validity_probability"]),
                int(perception.get("lane_visible_point_count", 0)),
                bool(perception.get("lane_detected", False)),
            ),
            cyan,
        ),
        (
            "d %.3f +/- %.3f m"
            % (
                float(policy["lane_lateral_error_mean_m"]),
                float(policy["lane_lateral_error_std_m"]),
            ),
            cyan,
        ),
        (
            "phi %.3f +/- %.3f rad"
            % (
                float(policy["lane_heading_error_mean_rad"]),
                float(policy["lane_heading_error_std_rad"]),
            ),
            cyan,
        ),
        (
            "kappa %.3f +/- %.3f 1/m"
            % (
                float(policy["lane_curvature_mean_inv_m"]),
                float(policy["lane_curvature_std_inv_m"]),
            ),
            cyan,
        ),
        (
            "cmd v %.3f m/s | omega %+.3f rad/s"
            % (float(info["v_cmd"]), float(info["omega_cmd"])),
            white,
        ),
        (
            "actual v %.3f m/s | omega %+.3f rad/s"
            % (float(info["v_actual"]), float(info["omega_actual"])),
            white,
        ),
    )
    right_x = max(330, canvas.shape[1] // 2 + 15)
    right = (
        ("EVAL ONLY (never enters PPO)", yellow),
        (f"GT d {float(truth['lane_lateral_error_m']):+.3f} m", yellow),
        (f"GT phi {float(truth['lane_heading_error_rad']):+.3f} rad", yellow),
        (f"progress {float(info['progress_m']):.3f} m", yellow),
        (f"reward {reward:+.3f} | return {total_return:+.3f}", white),
        (f"event: {event}", event_color),
        (f"lane_failure: {bool(info['lane_failure'])}", event_color),
    )
    _put_lines(canvas, left, x=10, y=20)
    _put_lines(canvas, right, x=right_x, y=62)
    return canvas


def render(args: argparse.Namespace) -> dict[str, Any]:
    config_path = args.config.resolve()
    stage_dir = args.stage_dir.resolve()
    protocol = load_ppo_curriculum_protocol(config_path)
    stage = protocol.stage("c0")
    if args.seed not in stage.development_seeds:
        raise ValueError("render seed must belong to the frozen C0 development split")

    checkpoint_manifest = json.loads(
        (stage_dir / "checkpoint_manifest.json").read_text(encoding="utf-8")
    )
    if checkpoint_manifest["config_sha256"] != file_sha256(config_path):
        raise RuntimeError("checkpoint/config provenance mismatch")
    selected = checkpoint_manifest["artifacts"]["selected"]
    checkpoint = (
        args.checkpoint.resolve()
        if args.checkpoint is not None
        else Path(selected["path"]).resolve()
    )
    if checkpoint == Path(selected["path"]).resolve() and file_sha256(checkpoint) != selected["sha256"]:
        raise RuntimeError("selected checkpoint hash mismatch")

    output = args.output.resolve()
    manifest_path = output.with_suffix(".json")
    preview_path = output.with_name(f"{output.stem}_terminal.png")
    for destination in (output, manifest_path, preview_path):
        if destination.exists() and not args.overwrite:
            raise FileExistsError(f"refusing to overwrite: {destination}")
        destination.parent.mkdir(parents=True, exist_ok=True)

    agent, payload = PPOAgent.load(checkpoint, device=args.device)
    environment = PPOCurriculumEnvironment(config_path, stage="c0", split="development")
    writer = imageio.get_writer(
        output,
        fps=args.fps,
        codec="libx264",
        quality=8,
        pixelformat="yuv420p",
        macro_block_size=None,
        ffmpeg_log_level="warning",
    )
    total_return = 0.0
    frames = 0
    last_frame: np.ndarray | None = None
    last_info: dict[str, Any] | None = None
    try:
        observation, _ = environment.reset(seed=args.seed)
        for step in range(1, stage.episode_horizon_steps + 1):
            action = agent.act(observation, deterministic=True).environment_action
            observation, reward, terminated, truncated, info = environment.step(action)
            total_return += float(reward)
            last_info = info
            last_frame = _annotate(
                environment.latest_rgb(),
                step=step,
                checkpoint_step=int(payload["global_step"]),
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

    if last_info is None or last_frame is None or frames == 0:
        raise RuntimeError("render produced no transition")
    cv2.imwrite(str(preview_path), cv2.cvtColor(last_frame, cv2.COLOR_RGB2BGR))
    manifest = {
        "schema_version": 1,
        "stage": "F10_PPO_VISUAL_C0_FAILURE_VIDEO",
        "runtime": "real Gym-Duckietown small_loop",
        "direction": "counterclockwise",
        "seed": args.seed,
        "seed_role": "frozen_c0_development_failure_replay",
        "config": str(config_path),
        "config_sha256": file_sha256(config_path),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": file_sha256(checkpoint),
        "checkpoint_step": int(payload["global_step"]),
        "policy_observation": "29-D visual lane belief with neutral hazards",
        "evaluation_only_overlay": [
            "lane_lateral_error_m",
            "lane_heading_error_rad",
            "progress_m",
            "lane_failure",
            "termination_reason",
        ],
        "video": str(output),
        "video_sha256": _sha256(output),
        "preview": str(preview_path),
        "preview_sha256": _sha256(preview_path),
        "fps": args.fps,
        "frames": frames,
        "duration_s": frames / float(args.fps),
        "total_return": total_return,
        "progress_m": float(last_info["progress_m"]),
        "completed": bool(last_info["completed"]),
        "lane_failure": bool(last_info["lane_failure"]),
        "termination_reason": last_info["termination_reason"],
        "truncation_reason": last_info["truncation_reason"],
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def main() -> None:
    args = _parse_args()
    if args.fps <= 0:
        raise SystemExit("fps must be positive")
    print(json.dumps(render(args), indent=2))


if __name__ == "__main__":
    main()
