#!/usr/bin/env python3
"""Render objective F15 failure animations directly from frozen telemetry.

These plots are not simulator or camera reconstructions.  Every plotted value comes
from the immutable localization traces selected before visualization.  This provides
human-readable visual evidence even when the active simulator cannot reproduce an old
closed-loop trajectory closely enough for a valid RGB replay.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import cv2
import imageio.v2 as imageio
import numpy as np
from PIL import Image

from duckie_pomdp.optimization.cross_curriculum_recovery import file_sha256


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_ROOT = ROOT / "artifacts/f15_cross_curriculum_recovery_v1"
WIDTH, HEIGHT, FPS = 1280, 720, 10
BLUE = (178, 114, 0)       # OpenCV BGR: manuscript blue
ORANGE = (0, 94, 213)      # OpenCV BGR: vermillion/orange
RED = (60, 60, 210)
GRID = (220, 220, 220)
TEXT = (35, 35, 35)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def load_trace(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        names = [str(value) for value in archive["feature_names"]]
        physical = np.asarray(archive["public_physical_29d"], dtype=np.float32)
        return {
            "step": np.arange(len(archive["progress_m"]), dtype=np.int32),
            "progress": np.asarray(archive["progress_m"], dtype=np.float32),
            "v_cmd": np.asarray(archive["physical_action"], dtype=np.float32)[:, 0],
            "omega_cmd": np.asarray(archive["physical_action"], dtype=np.float32)[:, 1],
            "lateral_error": physical[:, names.index("lane_lateral_error_mean_m")],
            "heading_error": physical[:, names.index("lane_heading_error_mean_rad")],
            "completed": np.asarray(archive["completed"], dtype=bool),
            "lane_failure": np.asarray(archive["lane_failure"], dtype=bool),
            "invalid_pose": np.asarray(archive["invalid_pose"], dtype=bool),
            "collision": np.asarray(archive["collision"], dtype=bool),
            "stop_violation": np.asarray(archive["stop_violation"], dtype=bool),
        }


def map_x(step: np.ndarray | float, start: int, stop: int, left: int, right: int) -> np.ndarray:
    denominator = max(1, stop - start)
    return left + (np.asarray(step, dtype=np.float64) - start) * (right - left) / denominator


def limits(*arrays: np.ndarray) -> tuple[float, float]:
    values = np.concatenate([np.asarray(array, dtype=np.float64)[np.isfinite(array)] for array in arrays])
    low, high = float(np.min(values)), float(np.max(values))
    if np.isclose(low, high):
        margin = max(0.05, abs(low) * 0.1)
    else:
        margin = 0.08 * (high - low)
    return low - margin, high + margin


def polyline(
    canvas: np.ndarray, steps: np.ndarray, values: np.ndarray, *, start: int, stop: int,
    bounds: tuple[int, int, int, int], value_limits: tuple[float, float], color: tuple[int, int, int],
) -> None:
    left, top, right, bottom = bounds
    mask = (steps >= start) & (steps <= stop) & np.isfinite(values)
    if not np.any(mask):
        return
    x = map_x(steps[mask], start, stop, left, right)
    low, high = value_limits
    y = bottom - (values[mask] - low) * (bottom - top) / max(1e-12, high - low)
    points = np.column_stack((x, y)).round().astype(np.int32)
    if len(points) == 1:
        cv2.circle(canvas, tuple(points[0]), 2, color, -1, cv2.LINE_AA)
    else:
        cv2.polylines(canvas, [points], False, color, 2, cv2.LINE_AA)


def draw_panel(
    canvas: np.ndarray, title: str, unit: str, original: dict[str, np.ndarray], compressed: dict[str, np.ndarray],
    key: str, *, start: int, stop: int, bounds: tuple[int, int, int, int], event_step: int,
) -> None:
    left, top, right, bottom = bounds
    values_limits = limits(original[key], compressed[key])
    cv2.rectangle(canvas, (left, top), (right, bottom), (252, 252, 252), -1)
    cv2.rectangle(canvas, (left, top), (right, bottom), GRID, 1)
    for fraction in (0.25, 0.5, 0.75):
        y = round(top + fraction * (bottom - top))
        cv2.line(canvas, (left, y), (right, y), GRID, 1, cv2.LINE_AA)
    polyline(canvas, original["step"], original[key], start=start, stop=stop, bounds=bounds,
             value_limits=values_limits, color=BLUE)
    polyline(canvas, compressed["step"], compressed[key], start=start, stop=stop, bounds=bounds,
             value_limits=values_limits, color=ORANGE)
    event_x = int(round(map_x(event_step, start, stop, left, right)))
    cv2.line(canvas, (event_x, top), (event_x, bottom), RED, 2, cv2.LINE_AA)
    cv2.putText(canvas, f"{title} ({unit})", (left, top - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.58, TEXT, 1, cv2.LINE_AA)
    cv2.putText(canvas, f"{values_limits[1]:+.3f}", (left + 4, top + 17), cv2.FONT_HERSHEY_SIMPLEX, 0.38, TEXT, 1, cv2.LINE_AA)
    cv2.putText(canvas, f"{values_limits[0]:+.3f}", (left + 4, bottom - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.38, TEXT, 1, cv2.LINE_AA)


def build_background(
    event: dict[str, Any], original: dict[str, np.ndarray], compressed: dict[str, np.ndarray], start: int, stop: int,
) -> tuple[np.ndarray, list[tuple[int, int, int, int]]]:
    canvas = np.full((HEIGHT, WIDTH, 3), 255, dtype=np.uint8)
    model_name = str(event.get("model_name", event["model_id"]))
    title = str(event.get("visual_title", "Original vs Compressed Policy at the First Objective Failure"))
    marker_label = str(event.get("marker_label", "Objective failure step"))
    cv2.putText(canvas, title, (42, 42),
                cv2.FONT_HERSHEY_SIMPLEX, 0.85, TEXT, 2, cv2.LINE_AA)
    cv2.putText(canvas, f"{model_name} | {str(event['curriculum']).upper()} | seed {event['seed']} | {event['event_labels']}",
                (42, 72), cv2.FONT_HERSHEY_SIMPLEX, 0.55, TEXT, 1, cv2.LINE_AA)
    cv2.line(canvas, (45, 98), (80, 98), BLUE, 3, cv2.LINE_AA)
    cv2.putText(canvas, "Original Policy", (88, 104), cv2.FONT_HERSHEY_SIMPLEX, 0.5, TEXT, 1, cv2.LINE_AA)
    cv2.line(canvas, (245, 98), (280, 98), ORANGE, 3, cv2.LINE_AA)
    cv2.putText(canvas, model_name, (288, 104), cv2.FONT_HERSHEY_SIMPLEX, 0.5, TEXT, 1, cv2.LINE_AA)
    cv2.line(canvas, (545, 98), (580, 98), RED, 3, cv2.LINE_AA)
    cv2.putText(canvas, marker_label, (588, 104), cv2.FONT_HERSHEY_SIMPLEX, 0.5, TEXT, 1, cv2.LINE_AA)
    panels = [(70, 145, 610, 365), (680, 145, 1220, 365), (70, 440, 610, 660), (680, 440, 1220, 660)]
    for title, unit, key, bounds in (
        ("Forward progress", "m", "progress", panels[0]),
        ("Linear velocity command", "m/s", "v_cmd", panels[1]),
        ("Angular velocity command", "rad/s", "omega_cmd", panels[2]),
        ("Lane lateral error", "m", "lateral_error", panels[3]),
    ):
        draw_panel(canvas, title, unit, original, compressed, key, start=start, stop=stop,
                   bounds=bounds, event_step=int(event["event_step"]))
    cv2.putText(canvas, "Direct visualization of frozen primary telemetry; no simulator replay or policy inference.",
                (70, 704), cv2.FONT_HERSHEY_SIMPLEX, 0.47, (80, 80, 80), 1, cv2.LINE_AA)
    return canvas, panels


def render_event(event: dict[str, Any], matrix_rows: list[dict[str, str]]) -> dict[str, Any]:
    model_id, curriculum, seed = str(event["model_id"]), str(event["curriculum"]), int(event["seed"])
    kind = str(event.get("visual_event_kind", "failure"))
    if kind not in {"failure", "success"}:
        raise ValueError(f"unsupported visual event kind: {kind}")
    output = ARTIFACT_ROOT / f"{kind}_telemetry" / model_id / curriculum / f"seed_{seed}"
    metadata_path = output / f"{kind}_telemetry_event.json"
    if metadata_path.exists():
        return read_json(metadata_path)
    output.mkdir(parents=True, exist_ok=True)
    compressed_path = Path(event["trace_path"])
    original_row = next(
        row for row in matrix_rows
        if row["model_id"] == "A0" and row["curriculum"] == curriculum and int(row["seed"]) == seed
    )
    original_path = Path(original_row["trace_path"])
    original, compressed = load_trace(original_path), load_trace(compressed_path)
    event_step = int(event["event_step"])
    if kind == "success":
        start, stop = 0, event_step
    else:
        start, stop = max(0, event_step - 90), event_step + 45
    background, panels = build_background(event, original, compressed, start, stop)
    stem = "success_telemetry_episode" if kind == "success" else "failure_telemetry_window"
    video_path = output / f"{stem}.mp4"
    writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (WIDTH, HEIGHT))
    if not writer.isOpened():
        raise RuntimeError(f"cannot open video writer: {video_path}")
    if kind == "success":
        sampled_steps = np.linspace(start, event_step, min(90, event_step - start + 1), dtype=int).tolist()
    else:
        sampled_steps = list(range(start, event_step + 1, 3))
    if not sampled_steps or sampled_steps[-1] != event_step:
        sampled_steps.append(event_step)
    frames: list[np.ndarray] = []
    for step in sampled_steps:
        frame = background.copy()
        for left, top, right, bottom in panels:
            x = int(round(map_x(step, start, stop, left, right)))
            cv2.line(frame, (x, top), (x, bottom), (80, 80, 80), 1, cv2.LINE_AA)
        cv2.putText(frame, f"Current telemetry step: {step}", (900, 104), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, TEXT, 1, cv2.LINE_AA)
        writer.write(frame)
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    for _ in range(FPS):
        writer.write(cv2.cvtColor(frames[-1], cv2.COLOR_RGB2BGR))
    writer.release()
    gif_path = output / f"{stem}.gif"
    imageio.mimsave(gif_path, frames, duration=0.12, loop=0)
    indexes = np.linspace(0, len(frames) - 1, min(8, len(frames)), dtype=int)
    thumbs = [Image.fromarray(frames[index]).resize((640, 360)) for index in indexes]
    sheet = Image.new("RGB", (1280, 360 * int(np.ceil(len(thumbs) / 2))), "white")
    for index, thumb in enumerate(thumbs):
        sheet.paste(thumb, ((index % 2) * 640, (index // 2) * 360))
    sheet_path = output / f"{stem}_contact_sheet.png"
    sheet.save(sheet_path, dpi=(300, 300))

    csv_path = output / f"{kind}_telemetry.csv"
    fields = ("policy", "step", "progress_m", "v_cmd_mps", "omega_cmd_rad_s", "lane_lateral_error_m",
              "heading_error_rad", "completed", "lane_failure", "invalid_pose", "collision", "stop_violation")
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        csv_writer = csv.DictWriter(stream, fieldnames=fields)
        csv_writer.writeheader()
        for policy_name, trace in (("Original Policy", original), (str(event.get("model_name", model_id)), compressed)):
            for step in range(start, min(stop + 1, len(trace["step"]))):
                csv_writer.writerow({
                    "policy": policy_name, "step": step, "progress_m": float(trace["progress"][step]),
                    "v_cmd_mps": float(trace["v_cmd"][step]), "omega_cmd_rad_s": float(trace["omega_cmd"][step]),
                    "lane_lateral_error_m": float(trace["lateral_error"][step]),
                    "heading_error_rad": float(trace["heading_error"][step]),
                    "completed": bool(trace["completed"][step]), "lane_failure": bool(trace["lane_failure"][step]),
                    "invalid_pose": bool(trace["invalid_pose"][step]), "collision": bool(trace["collision"][step]),
                    "stop_violation": bool(trace["stop_violation"][step]),
                })
    files = {}
    for path in (video_path, gif_path, sheet_path, csv_path):
        files[path.name] = {"path": str(path), "sha256": file_sha256(path)}
    metadata = {
        **event,
        "visualization_type": "direct_frozen_telemetry_timeline",
        "visual_event_kind": kind,
        "visualization_status": "VERIFIED_FROM_PRIMARY_TELEMETRY",
        "camera_or_simulator_frames_included": False,
        "reason_camera_replay_absent": (
            "the active simulator did not reproduce the frozen episode within the preregistered visual replay criteria"
        ),
        "original_trace_path": str(original_path), "original_trace_sha256": file_sha256(original_path),
        "compressed_trace_path": str(compressed_path), "compressed_trace_sha256": file_sha256(compressed_path),
        "selection_rule": "frozen lowest-seed then first-objective-failure rule",
        "pairing_claim": "same-seed recorded telemetry comparison; not a causal paired trajectory",
        "files": files,
    }
    write_json(metadata_path, metadata)
    return metadata


def main() -> None:
    decision_path = ARTIFACT_ROOT / "localization/failure_localization_decision.json"
    decision = read_json(decision_path)
    if decision.get("classification") != "FROZEN":
        raise RuntimeError("failure localization must be frozen before visualization")
    matrix_rows = read_csv(ARTIFACT_ROOT / "localization/matrix_episodes.csv")
    rendered = [render_event(dict(event), matrix_rows) for event in decision["failure_events"]]
    manifest = {
        "schema_version": 1,
        "source_failure_decision": str(decision_path),
        "source_failure_decision_sha256": file_sha256(decision_path),
        "visualization_type": "direct_frozen_telemetry_timeline",
        "events": rendered,
    }
    write_json(ARTIFACT_ROOT / "failure_telemetry/failure_telemetry_manifest.json", manifest)

    successful: list[dict[str, Any]] = []
    for family, decisions_path, episode_path in (
        ("matrix", ARTIFACT_ROOT / "localization/matrix_results.json", ARTIFACT_ROOT / "localization/matrix_episodes.csv"),
        ("pruning", ARTIFACT_ROOT / "localization/pruning_results.json", ARTIFACT_ROOT / "localization/pruning_episodes.csv"),
    ):
        decisions = read_json(decisions_path)["decisions"]
        rows = read_csv(episode_path)
        for model_id, curricula in decisions.items():
            for curriculum, cell in curricula.items():
                if cell["status"] not in {"PASS", "REFERENCE"}:
                    continue
                eligible = [
                    row for row in rows
                    if row["model_id"] == model_id and row["curriculum"] == curriculum
                    and row["completed"].lower() == "true"
                    and row["collision"].lower() == "false"
                    and row["lane_failure"].lower() == "false"
                    and row["invalid_pose"].lower() == "false"
                    and row["stop_violation"].lower() == "false"
                    and row["timeout"].lower() == "false"
                    and int(row["unsafe_proximity_events"]) == 0
                ]
                if not eligible:
                    raise RuntimeError(f"passing cell lacks an objectively successful episode: {model_id}/{curriculum}")
                selected = min(eligible, key=lambda row: int(row["seed"]))
                trace_path = Path(selected["trace_path"])
                trace = load_trace(trace_path)
                success_event = {
                    "family": family, "model_id": model_id, "model_name": selected["model_name"],
                    "curriculum": curriculum, "seed": int(selected["seed"]),
                    "event_step": int(len(trace["step"]) - 1), "event_labels": "recorded_completion",
                    "trace_path": str(trace_path), "trace_sha256": file_sha256(trace_path),
                    "selection_rule": "lowest seed with completion and no objective failure flag",
                    "visual_event_kind": "success",
                    "visual_title": "Frozen Telemetry for an Objectively Successful Episode",
                    "marker_label": "Recorded completion step",
                }
                successful.append(render_event(success_event, matrix_rows))
    unique_successful = {
        (str(item["model_id"]), str(item["curriculum"]), int(item["seed"])): item
        for item in successful
    }
    successful = [unique_successful[key] for key in sorted(unique_successful)]
    success_manifest = {
        "schema_version": 1,
        "source_failure_decision": str(decision_path),
        "source_failure_decision_sha256": file_sha256(decision_path),
        "visualization_type": "direct_frozen_telemetry_timeline",
        "selection_rule": "lowest seed with completion and no objective failure flag",
        "episodes": successful,
    }
    write_json(ARTIFACT_ROOT / "success_telemetry/success_telemetry_manifest.json", success_manifest)
    print(json.dumps({"rendered_failure_telemetry_events": len(rendered), "rendered_success_telemetry_episodes": len(successful)}, indent=2))


if __name__ == "__main__":
    main()
