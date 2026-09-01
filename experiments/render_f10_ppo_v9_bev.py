"""Render a bird's-eye-view video of a frozen F10-PPO checkpoint driving a lap.

Generalises the v3 failure renderer in two ways: the stage and split are
selectable (so C1/experiment_loop and the stage-final splits are reachable, not
just C0 development), and the frame is the simulator's top-down view rather than
the front camera.

Evaluation-only overlay. Privileged ground truth is read strictly after the
policy has acted, exactly as the v3 renderer does, so nothing here can leak into
the policy's observation.
"""

from __future__ import annotations

import argparse
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

WHITE = (255, 255, 255)
CYAN = (120, 220, 255)
YELLOW = (255, 220, 110)
RED = (255, 110, 110)
GREEN = (140, 240, 160)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "f10_ppo_visual_v9.toml")
    parser.add_argument("--stage", default="c0", choices=("c0", "c1"))
    parser.add_argument("--split", default="stage_final", choices=("development", "stage_final"))
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--stage-dir", type=Path)
    parser.add_argument(
        "--layout",
        default="side_by_side",
        choices=("bev", "side_by_side"),
        help="bev = top-down only; side_by_side = front camera beside top-down",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _text(canvas: np.ndarray, lines: list[tuple[str, tuple[int, int, int]]], x: int, y0: int) -> None:
    for index, (line, colour) in enumerate(lines):
        cv2.putText(
            canvas,
            line,
            (x, y0 + index * 19),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            tuple(int(c) for c in colour),
            1,
            cv2.LINE_AA,
        )


def _compose(front: np.ndarray | None, bev: np.ndarray) -> np.ndarray:
    """Place the front camera beside the top-down view, matched on height.

    The two sources differ in size (640x480 front, 800x600 top-down), so the
    front frame is scaled to the top-down height and the panels are labelled --
    otherwise a reader cannot tell which view is which once the banner is added.
    """

    if front is None:
        return np.ascontiguousarray(bev.copy())

    height = bev.shape[0]
    width = int(round(front.shape[1] * height / front.shape[0]))
    scaled = cv2.resize(front, (width, height), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((height, width + bev.shape[1], 3), dtype=np.uint8)
    canvas[:, :width] = scaled
    canvas[:, width:] = bev
    cv2.line(canvas, (width, 0), (width, height), (40, 40, 40), 2)

    for label, x in (("FRONT CAMERA (policy input)", 12), ("TOP-DOWN (evaluation only)", width + 12)):
        cv2.putText(
            canvas, label, (x, height - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA
        )
    return canvas


def _annotate(
    frame: np.ndarray,
    *,
    stage: str,
    map_name: str,
    seed: int,
    step: int,
    checkpoint_step: int,
    info: dict[str, Any],
    reward: float,
    total_return: float,
    layout: str,
) -> np.ndarray:
    # Keep every simulator pixel visible.  The older renderer painted a black
    # banner over y=0..120, hiding the north/top segment of the BEV loop.  Put
    # telemetry in a dedicated strip below the views instead.
    frame = np.ascontiguousarray(frame)
    height, width = frame.shape[:2]
    strip_height = 108
    canvas = np.zeros((height + strip_height, width, 3), dtype=np.uint8)
    canvas[:height] = frame
    cv2.line(canvas, (0, height + 2), (width, height + 2), (55, 55, 55), 2)

    truth = info.get("evaluation_gt", {}) or {}

    def number(source: dict[str, Any], key: str) -> str:
        value = source.get(key)
        return "n/a" if value is None else f"{float(value):+.3f}"

    _text(
        canvas,
        [
            (
                f"F10-PPO v9 {'FRONT+BEV' if layout == 'side_by_side' else 'BEV'}"
                f" | {stage.upper()} {map_name} | seed {seed}",
                WHITE,
            ),
            (f"checkpoint {checkpoint_step:,} | step {step:04d}", CYAN),
            (f"progress {float(info.get('progress_m', 0.0)):.2f} m", GREEN),
        ],
        10,
        height + 22,
    )
    event = info.get("termination_reason") or info.get("truncation_reason") or "running"
    _text(
        canvas,
        [
            (f"EVAL ONLY  d {number(truth, 'lane_lateral_error_m')} m", YELLOW),
            (f"           phi {number(truth, 'lane_heading_error_rad')} rad", YELLOW),
            (f"reward {reward:+.3f} | return {total_return:+.2f}", WHITE),
            (f"event: {event}", RED if event != "running" else GREEN),
        ],
        max(10, width // 2),
        height + 22,
    )
    return canvas


def render(args: argparse.Namespace) -> dict[str, Any]:
    config_path = args.config.resolve()
    protocol = load_ppo_curriculum_protocol(config_path)
    stage = protocol.stage(args.stage)

    allowed = (
        stage.development_seeds if args.split == "development" else stage.stage_final_seeds
    )
    if args.seed not in allowed:
        raise ValueError(
            f"seed {args.seed} is not in the frozen {args.stage} {args.split} split {allowed}"
        )

    stage_dir = (
        args.stage_dir.resolve()
        if args.stage_dir is not None
        else (ROOT / "artifacts" / "f10_ppo_visual_v9" / args.stage)
    )
    manifest = json.loads((stage_dir / "checkpoint_manifest.json").read_text(encoding="utf-8"))
    if manifest.get("config_sha256") not in (None, file_sha256(config_path)):
        raise RuntimeError("checkpoint/config provenance mismatch")
    selected = manifest["artifacts"]["selected"]
    checkpoint = (
        args.checkpoint.resolve() if args.checkpoint is not None else Path(selected["path"]).resolve()
    )
    if checkpoint == Path(selected["path"]).resolve():
        if file_sha256(checkpoint) != selected["sha256"]:
            raise RuntimeError("selected checkpoint hash mismatch")

    output = (
        args.output.resolve()
        if args.output is not None
        else stage_dir / f"{args.stage}_v9_{args.layout}_{args.split}_seed{args.seed}.mp4"
    )
    manifest_path = output.with_suffix(".json")
    preview_path = output.with_name(f"{output.stem}_final.png")
    for destination in (output, manifest_path, preview_path):
        if destination.exists() and not args.overwrite:
            raise FileExistsError(f"refusing to overwrite: {destination}")
        destination.parent.mkdir(parents=True, exist_ok=True)

    agent, payload = PPOAgent.load(checkpoint, device=args.device)
    environment = PPOCurriculumEnvironment(config_path, stage=args.stage, split=args.split)
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
        simulator = environment._integration.agent._session._simulator
        for step in range(1, stage.episode_horizon_steps + 1):
            action = agent.act(observation, deterministic=True).environment_action
            observation, reward, terminated, truncated, info = environment.step(action)
            total_return += float(reward)
            last_info = info
            bev = np.asarray(simulator.render(mode="top_down")).copy()
            front = environment.latest_rgb() if args.layout == "side_by_side" else None
            last_frame = _annotate(
                _compose(front, bev),
                stage=args.stage,
                map_name=stage.map_name,
                seed=args.seed,
                step=step,
                checkpoint_step=int(payload["global_step"]),
                info=info,
                reward=float(reward),
                total_return=total_return,
                layout=args.layout,
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
    record = {
        "schema_version": 1,
        "stage": f"F10_PPO_VISUAL_V9_{args.stage.upper()}_{args.layout.upper()}",
        "view": (
            "front camera (policy input) beside simulator top_down (evaluation only)"
            if args.layout == "side_by_side"
            else "simulator top_down (bird's eye)"
        ),
        "layout": args.layout,
        "runtime": f"real Gym-Duckietown {stage.map_name}",
        "split": args.split,
        "seed": args.seed,
        "config": str(config_path),
        "config_sha256": file_sha256(config_path),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": file_sha256(checkpoint),
        "checkpoint_step": int(payload["global_step"]),
        "evaluation_only_overlay": [
            "lane_lateral_error_m",
            "lane_heading_error_rad",
            "progress_m",
            "termination_reason",
        ],
        "video": str(output),
        "preview": str(preview_path),
        "fps": args.fps,
        "frames": frames,
        "duration_s": frames / float(args.fps),
        "total_return": total_return,
        "progress_m": float(last_info.get("progress_m", 0.0)),
        "completed": bool(last_info.get("completed", False)),
        "termination_reason": last_info.get("termination_reason"),
        "truncation_reason": last_info.get("truncation_reason"),
    }
    manifest_path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    return record


def main() -> None:
    args = _parse_args()
    record = render(args)
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
