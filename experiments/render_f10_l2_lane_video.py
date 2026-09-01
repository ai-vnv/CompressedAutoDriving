"""Render one audited F10-L2 transfer-policy lap on ``experiment_loop``.

The SAC policy receives only the frozen six-dimensional lane observation.
World path length and yellow-line clearance are evaluation-only overlay data.
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

from duckie_pomdp.control import (
    LaneTransferEnvironment,
    SACAgent,
    load_lane_transfer_protocol,
)
from duckie_pomdp.control.f10_protocol import file_sha256


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "f10_l2_transfer_v1.toml"
DEFAULT_ARTIFACT_DIR = ROOT / "artifacts" / "f10_l2"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--artifact-dir", type=Path, default=DEFAULT_ARTIFACT_DIR)
    parser.add_argument("--seed", type=int, default=17001)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--capture-every", type=int, default=1)
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_ARTIFACT_DIR / "sac_lane_transfer_demo.mp4",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _annotate(
    frame_rgb: np.ndarray,
    *,
    step: int,
    fps: int,
    checkpoint_step: int,
    info: dict[str, Any],
) -> np.ndarray:
    canvas = np.ascontiguousarray(frame_rgb.copy())
    panel = canvas.copy()
    cv2.rectangle(panel, (0, 0), (canvas.shape[1], 146), (0, 0, 0), -1)
    cv2.addWeighted(panel, 0.68, canvas, 0.32, 0.0, canvas)
    lines = (
        "F10-L2 SAC TRANSFER | experiment_loop | MIXED TURNS | real simulator",
        f"checkpoint {checkpoint_step:,} | time {step / fps:6.2f} s | step {step:04d}",
        f"command: v {float(info['v_cmd']):+.3f} m/s   "
        f"omega {float(info['omega_cmd']):+.3f} rad/s",
        f"actual:  v {float(info['v_actual']):+.3f} m/s   "
        f"omega {float(info['omega_actual']):+.3f} rad/s",
        f"lane: d {float(info['lateral_error_m']):+.3f} m   "
        f"phi {float(info['heading_error_rad']):+.3f} rad",
        f"EVAL ONLY: yellow clearance {float(info['yellow_clearance_m']):+.3f} m   "
        f"path {float(info['path_length_m']):.2f} m",
    )
    for row, line in enumerate(lines):
        color = (255, 230, 80) if row == 5 else (255, 255, 255)
        cv2.putText(
            canvas,
            line,
            (12, 22 + 23 * row),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.46,
            color,
            1,
            cv2.LINE_AA,
        )
    return canvas


def render(args: argparse.Namespace) -> dict[str, Any]:
    config_path = args.config.resolve()
    artifact_dir = args.artifact_dir.resolve()
    output = args.output.resolve()
    manifest_path = output.with_suffix(".json")
    preview_path = output.with_name(f"{output.stem}_preview.png")
    for destination in (output, manifest_path, preview_path):
        if destination.exists() and not args.overwrite:
            raise FileExistsError(f"refusing to overwrite: {destination}")
        destination.parent.mkdir(parents=True, exist_ok=True)

    protocol = load_lane_transfer_protocol(config_path)
    if args.seed not in protocol.seeds.development:
        raise ValueError("proof-video seed must belong to the development split")
    checkpoint_manifest = json.loads(
        (artifact_dir / "checkpoint_manifest.json").read_text(encoding="utf-8")
    )
    if checkpoint_manifest["config_sha256"] != file_sha256(config_path):
        raise RuntimeError("checkpoint manifest does not match the frozen config")
    selected = checkpoint_manifest["artifacts"]["safety_selected"]
    checkpoint_path = Path(selected["path"])
    if file_sha256(checkpoint_path) != selected["sha256"]:
        raise RuntimeError("selected checkpoint hash mismatch")

    agent, payload = SACAgent.load(checkpoint_path, device=args.device)
    environment = LaneTransferEnvironment(config_path, split="development")
    writer = imageio.get_writer(
        output,
        fps=args.fps,
        codec="libx264",
        quality=8,
        pixelformat="yuv420p",
        macro_block_size=None,
        ffmpeg_log_level="warning",
    )
    reward_names = (
        "reward_progress",
        "reward_lane",
        "reward_yellow",
        "reward_comfort",
        "reward_living",
        "reward_terminal",
    )
    rewards = {name: 0.0 for name in reward_names}
    lateral: list[float] = []
    heading: list[float] = []
    clearance: list[float] = []
    actual_velocity: list[float] = []
    encoded_frames = 0
    preview: np.ndarray | None = None
    last_info: dict[str, Any] | None = None
    total_return = 0.0
    try:
        observation, _ = environment.reset(seed=args.seed)
        horizon = int(protocol.raw["simulator"]["episode_horizon_steps"])
        for step in range(1, horizon + 1):
            action = agent.act(observation, deterministic=True)
            observation, reward, terminated, truncated, info = environment.step(action)
            last_info = info
            total_return += float(reward)
            for name in rewards:
                rewards[name] += float(info[name])
            lateral.append(abs(float(info["lateral_error_m"])))
            heading.append(abs(float(info["heading_error_rad"])))
            clearance.append(float(info["yellow_clearance_m"]))
            actual_velocity.append(float(info["v_actual"]))
            if step % args.capture_every == 0:
                frame = _annotate(
                    environment.latest_rgb(),
                    step=step,
                    fps=args.fps,
                    checkpoint_step=int(payload["global_step"]),
                    info=info,
                )
                writer.append_data(frame)
                encoded_frames += 1
                if preview is None or abs(float(info["omega_cmd"])) > 1.5:
                    preview = frame.copy()
            if terminated or truncated:
                break
    finally:
        writer.close()
        environment.close()

    if last_info is None or encoded_frames == 0 or preview is None:
        raise RuntimeError("proof render produced no complete transition")
    cv2.imwrite(str(preview_path), cv2.cvtColor(preview, cv2.COLOR_RGB2BGR))
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "stage": "F10_L2_SAC_TRANSFER_VIDEO_PROOF",
        "runtime": "real Gym-Duckietown experiment_loop",
        "turning_geometry": "mixed_left_and_right",
        "policy_observation": "six-dimensional agent-visible lane state only",
        "evaluation_only_overlay": ["yellow_clearance_m", "path_length_m"],
        "seed": args.seed,
        "seed_role": "development_proof_only_after_checkpoint_selection",
        "config": str(config_path),
        "config_sha256": file_sha256(config_path),
        "checkpoint": str(checkpoint_path.resolve()),
        "checkpoint_sha256": selected["sha256"],
        "checkpoint_step": int(payload["global_step"]),
        "video": str(output),
        "video_sha256": _sha256(output),
        "preview": str(preview_path),
        "preview_sha256": _sha256(preview_path),
        "fps": args.fps,
        "capture_every": args.capture_every,
        "encoded_frames": encoded_frames,
        "duration_s": encoded_frames / float(args.fps),
        "steps": len(lateral),
        "lap_completed": bool(last_info["lap_completed"]),
        "yellow_crossing": bool(last_info["yellow_crossing"]),
        "lane_departure": bool(last_info["lane_departure"]),
        "invalid_pose": bool(last_info["invalid_pose"]),
        "termination_reason": last_info["termination_reason"],
        "truncation_reason": last_info["truncation_reason"],
        "total_return": total_return,
        "path_length_m": float(last_info["path_length_m"]),
        "mean_abs_lateral_error_m": float(np.mean(lateral)),
        "p95_abs_lateral_error_m": float(np.percentile(lateral, 95)),
        "mean_abs_heading_error_rad": float(np.mean(heading)),
        "minimum_yellow_clearance_m": min(clearance),
        "mean_actual_velocity_mps": float(np.mean(actual_velocity)),
        "reward_components": rewards,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def main() -> None:
    args = _parse_args()
    if args.fps <= 0 or args.capture_every <= 0:
        raise SystemExit("fps and capture-every must be positive")
    print(json.dumps(render(args), indent=2))


if __name__ == "__main__":
    main()
