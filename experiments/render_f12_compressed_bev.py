"""Render the frozen F12 compressed A7 actor in C4 with front RGB + audit BEV."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import imageio.v2 as imageio
import numpy as np
import torch

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib

from duckie_pomdp.control import PPOCurriculumEnvironment
from duckie_pomdp.control.f10_protocol import file_sha256
from duckie_pomdp.control.ppo_protocol import load_ppo_curriculum_protocol
from duckie_pomdp.perception.yolo_detector import YoloObjectDetector

from render_f10_ppo_object_bev import _compose, _draw_detections, _put_lines


ROOT = Path(__file__).resolve().parents[1]


def args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--f12-config", type=Path, default=ROOT / "configs/f12_belief_ppo_compression_v1.toml")
    parser.add_argument("--seed", type=int, default=178021)
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/f12_belief_ppo_compression_v1/final/a7_c4_front_bev.mp4")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--max-steps", type=int)
    return parser.parse_args()


def annotate(canvas: np.ndarray, *, seed: int, step: int, info: dict, reward: float, total_return: float) -> np.ndarray:
    height, width = canvas.shape[:2]
    result = np.zeros((height + 126, width, 3), dtype=np.uint8)
    result[:height] = canvas
    cv2.line(result, (0, height + 2), (width, height + 2), (55, 55, 55), 2)
    policy = info.get("policy", {}) or {}
    perception = info.get("perception", {}) or {}
    _put_lines(result, [
        f"F12 A7 COMPRESSED BELIEF-PPO | C4 | seed {seed} | step {step}",
        f"YOLO: duckie detections {int(perception.get('duckie_detection_count', 0))}",
        "pedestrian: P(e) %.3f  r %.3f +/- %.3f m" % (
            float(policy.get("pedestrian_existence_probability", 0.0)),
            float(policy.get("pedestrian_range_mean_m", 0.0)),
            float(policy.get("pedestrian_range_std_m", 0.0)),
        ),
        "stop: P %.3f  range %.3f m  mode %s" % (
            float(policy.get("stop_sign_existence_probability", 0.0)),
            float(policy.get("stop_sign_range_mean_m", 0.0)),
            str(policy.get("stop_mode", "unknown")),
        ),
        "lane: valid %.2f  d %+.3f m  phi %+.3f rad" % (
            float(policy.get("lane_validity_probability", 0.0)),
            float(policy.get("lane_lateral_error_mean_m", 0.0)),
            float(policy.get("lane_heading_error_mean_rad", 0.0)),
        ),
    ], 10, height + 21)
    _put_lines(result, [
        "action: v %.3f m/s  omega %+.3f rad/s" % (
            float(info.get("v_cmd", 0.0)), float(info.get("omega_cmd", 0.0))
        ),
        f"progress {float(info.get('progress_m', 0.0)):.2f} m",
        f"reward {reward:+.3f}  return {total_return:+.2f}",
        "event: %s" % (info.get("termination_reason") or info.get("truncation_reason") or "running"),
        "BEV/GT are visualization-only; actor input is public 29D belief",
    ], width // 2, height + 21)
    return result


def main() -> None:
    cli = args()
    with cli.f12_config.resolve().open("rb") as stream:
        f12 = tomllib.load(stream)
    if cli.seed not in tuple(f12["seeds"]["compression_selection"]):
        raise ValueError("qualitative video seed must come from compression-selection, never final holdout")
    policy_config = (cli.f12_config.resolve().parent / f12["frozen"]["policy_config"]).resolve()
    selection_path = (cli.f12_config.resolve().parent / f12["artifacts"]["directory"] / "final/model_selection.json").resolve()
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    if selection["selected_variant"] != "A7":
        raise RuntimeError("frozen selected variant is not A7")
    actor_path = Path(selection["checkpoint_path"])
    if file_sha256(actor_path) != selection["checkpoint_sha256"]:
        raise RuntimeError("A7 checkpoint hash mismatch")
    actor = torch.jit.load(str(actor_path), map_location="cpu").eval()
    protocol = load_ppo_curriculum_protocol(policy_config)
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
    output = cli.output.resolve()
    for path in (output, output.with_suffix(".png"), output.with_suffix(".json")):
        if path.exists():
            raise FileExistsError(f"refusing to overwrite {path}")
        path.parent.mkdir(parents=True, exist_ok=True)
    env = PPOCurriculumEnvironment(policy_config, stage="c4", split="f12_a7_qualitative_video", seeds=(cli.seed,))
    writer = imageio.get_writer(output, fps=cli.fps, codec="libx264", quality=8, pixelformat="yuv420p", macro_block_size=None, ffmpeg_log_level="warning")
    frames = 0
    total_return = 0.0
    last_frame = None
    last_info = None
    try:
        observation, _ = env.reset(seed=cli.seed)
        simulator = env._integration.agent._session._simulator
        horizon = protocol.stage("c4").episode_horizon_steps
        limit = min(horizon, cli.max_steps or horizon)
        for step in range(1, limit + 1):
            with torch.inference_mode():
                action = actor(torch.as_tensor(observation, dtype=torch.float32).unsqueeze(0)).squeeze(0).numpy()
            observation, reward, terminated, truncated, info = env.step(np.clip(action, -1.0, 1.0))
            front = env.latest_rgb()
            detections = detector.detect(front)
            bev = np.asarray(simulator.render(mode="top_down")).copy()
            total_return += float(reward)
            last_frame = annotate(
                _compose(_draw_detections(front, detections), bev), seed=cli.seed,
                step=step, info=info, reward=float(reward), total_return=total_return,
            )
            writer.append_data(last_frame)
            frames += 1
            last_info = info
            if terminated or truncated:
                break
    finally:
        writer.close()
        env.close()
    if last_frame is None or last_info is None:
        raise RuntimeError("video produced no frames")
    cv2.imwrite(str(output.with_suffix(".png")), cv2.cvtColor(last_frame, cv2.COLOR_RGB2BGR))
    manifest = {
        "schema_version": 1,
        "role": "qualitative_example_only",
        "stage": "c4",
        "seed": cli.seed,
        "seed_split": "compression_selection",
        "variant": "A7",
        "checkpoint_sha256": selection["checkpoint_sha256"],
        "video": str(output),
        "video_sha256": file_sha256(output),
        "preview_sha256": file_sha256(output.with_suffix(".png")),
        "frames": frames,
        "fps": cli.fps,
        "duration_s": frames / cli.fps,
        "completed": bool(last_info.get("completed", False)),
        "collision": bool(last_info.get("collision", False)),
        "progress_m": float(last_info.get("progress_m", 0.0)),
        "runtime_path": "front RGB -> MobileNet/YOLO -> beliefs -> public 29D -> compressed A7 actor",
        "visualization_only": ["second YOLO audit overlay", "simulator top_down BEV"],
    }
    output.with_suffix(".json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
