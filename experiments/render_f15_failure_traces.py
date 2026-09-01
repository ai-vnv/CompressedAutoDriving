#!/usr/bin/env python3
"""Render objectively selected F15 failure evidence after localization freeze."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Mapping

import cv2
import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from duckie_pomdp.control.ppo_environment import PPOCurriculumEnvironment
from duckie_pomdp.control.ppo_protocol import load_ppo_curriculum_protocol
from duckie_pomdp.optimization.cross_curriculum_recovery import (
    file_sha256,
    first_objective_failure_event,
    verify_registry,
)

from run_f15_cross_curriculum_recovery import (
    ActorPolicy,
    artifact_root,
    frozen_paths,
    load_actor,
    load_config,
    read_json,
    trace_path,
    verify_protocol,
    write_json,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/f15_cross_curriculum_recovery_v1.toml"

# ---------------------------------------------------------------------------
# Frozen recorded-action replay validation.
#
# These tolerances are preregistered in docs/F15_VISUAL_REPLAY_IMPLEMENTATION_AMENDMENT.md
# and were fixed before any F15 video was rendered.  They are derived from simulator
# geometry, not tuned to make a replay pass:
#
#   EXACT     1 mm  -- indistinguishable from bitwise reproduction.
#   TOLERANT  5 cm  -- about five simulator steps of travel at the 0.4 m/s action
#                      bound, and roughly a quarter of the Duckietown half-lane
#                      width, so a deviation inside this band cannot by itself
#                      change a lane-departure or invalid-pose verdict.
#   EVENT     +/-2 steps, against a 90-before/45-after extraction window.
#
# Labels and termination reason must match exactly at every level.
# ---------------------------------------------------------------------------
REPLAY_EXACT_PROGRESS_M = 1.0e-3
REPLAY_TOLERANT_PROGRESS_M = 0.05
REPLAY_TOLERANT_EVENT_STEP = 2


def validate_recorded_action_replay(
    *,
    recorded_trace: Path,
    expected_trace_sha256: str,
    replayed_actions: np.ndarray,
    recorded_actions: np.ndarray,
    replay_progress: list[float],
    recorded_progress: np.ndarray,
    replay_step_rows: list[dict[str, Any]],
    recorded_event_step: int | None,
    recorded_event_labels: str,
) -> dict[str, Any]:
    """Three-level check that a recorded-action replay reproduces primary telemetry.

    Level 1 binds the replayed action sequence to the frozen trace by hash and value.
    Level 2 compares the reconstructed trajectory against the recorded trajectory.
    Level 3 requires the objective failure event to reproduce.

    Returns a report whose ``status`` is VERIFIED_EXACT, VERIFIED_WITHIN_TOLERANCE, or
    UNRESOLVED.  UNRESOLVED never raises: the caller quarantines the media instead, and
    the frozen primary telemetry remains the authoritative evidence either way.
    """
    report: dict[str, Any] = {"levels": {}}

    actual_sha = file_sha256(recorded_trace)
    action_identical = (
        replayed_actions.shape == recorded_actions.shape
        and bool(np.array_equal(replayed_actions, recorded_actions))
    )
    level1 = actual_sha == expected_trace_sha256 and action_identical
    report["levels"]["level1_recorded_action_integrity"] = {
        "pass": level1,
        "trace_sha256_matches_frozen_event": actual_sha == expected_trace_sha256,
        "replayed_actions_identical_to_telemetry": action_identical,
        "steps_replayed": int(len(replayed_actions)),
    }

    common = min(len(replay_progress), len(recorded_progress))
    max_progress_error = (
        float(np.max(np.abs(np.asarray(replay_progress[:common]) - recorded_progress[:common])))
        if common else float("inf")
    )
    report["levels"]["level2_trajectory_reproduction"] = {
        "max_absolute_progress_error_m": max_progress_error,
        "steps_compared": int(common),
        "exact_threshold_m": REPLAY_EXACT_PROGRESS_M,
        "tolerant_threshold_m": REPLAY_TOLERANT_PROGRESS_M,
        "pass_exact": max_progress_error <= REPLAY_EXACT_PROGRESS_M,
        "pass_tolerant": max_progress_error <= REPLAY_TOLERANT_PROGRESS_M,
    }

    replay_event = first_objective_failure_event(replay_step_rows)
    replay_step = None if replay_event is None else int(replay_event["step"])
    replay_labels = "" if replay_event is None else "|".join(replay_event["event_labels"])
    labels_match = replay_labels == recorded_event_labels
    if replay_step is None or recorded_event_step is None:
        step_delta = None
        step_exact = step_tolerant = False
    else:
        step_delta = abs(replay_step - int(recorded_event_step))
        step_exact = step_delta == 0
        step_tolerant = step_delta <= REPLAY_TOLERANT_EVENT_STEP
    report["levels"]["level3_failure_reproduction"] = {
        "recorded_event_step": recorded_event_step,
        "replay_event_step": replay_step,
        "event_step_delta": step_delta,
        "recorded_event_labels": recorded_event_labels,
        "replay_event_labels": replay_labels,
        "event_labels_match": labels_match,
        "pass_exact": bool(step_exact and labels_match),
        "pass_tolerant": bool(step_tolerant and labels_match),
    }

    if (
        level1
        and report["levels"]["level2_trajectory_reproduction"]["pass_exact"]
        and report["levels"]["level3_failure_reproduction"]["pass_exact"]
    ):
        report["status"] = "VERIFIED_EXACT"
    elif (
        level1
        and report["levels"]["level2_trajectory_reproduction"]["pass_tolerant"]
        and report["levels"]["level3_failure_reproduction"]["pass_tolerant"]
    ):
        report["status"] = "VERIFIED_WITHIN_TOLERANCE"
    else:
        report["status"] = "UNRESOLVED"
    report["usable_as_visual_evidence"] = report["status"] != "UNRESOLVED"
    report["primary_evidence_unaffected"] = True
    return report


def _writer(path: Path, width: int, height: int, fps: float = 30.0):
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"cannot open video writer: {path}")
    return writer


def prepare_output_directory(output: Path, completion_name: str, root: Path) -> dict[str, Any] | None:
    """Resume completed renders and preserve interrupted attempts without deletion."""
    completion = output / completion_name
    if completion.exists():
        return read_json(completion)
    if output.exists():
        relative = output.relative_to(root)
        base = root / "integrity/incomplete_visual_replay_attempts" / relative
        destination = base
        attempt = 1
        while destination.exists():
            attempt += 1
            destination = base.with_name(f"{base.name}_attempt_{attempt}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        output.replace(destination)
    output.mkdir(parents=True)
    return None


def annotate(frame_rgb: np.ndarray, lines: list[str], *, failure: bool) -> np.ndarray:
    frame = cv2.cvtColor(np.asarray(frame_rgb), cv2.COLOR_RGB2BGR)
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (frame.shape[1], 24 + 20 * len(lines)), (20, 20, 20), -1)
    frame = cv2.addWeighted(overlay, 0.72, frame, 0.28, 0)
    for index, line in enumerate(lines):
        color = (80, 80, 255) if failure and index == 0 else (245, 245, 245)
        cv2.putText(frame, line, (10, 22 + 20 * index), cv2.FONT_HERSHEY_SIMPLEX, 0.48, color, 1, cv2.LINE_AA)
    return frame


def render_one(
    config: Mapping[str, Any], config_path: Path, event: Mapping[str, Any], entry: Mapping[str, Any],
    original_entry: Mapping[str, Any],
) -> dict[str, Any]:
    root = artifact_root(config, config_path)
    model_id, curriculum, seed = event["model_id"], event["curriculum"], int(event["seed"])
    failure_step = int(event["event_step"])
    before, after = int(config["evaluation"]["failure_window_steps_before"]), int(config["evaluation"]["failure_window_steps_after"])
    start, stop = max(0, failure_step - before), failure_step + after
    output = root / "failure_traces" / model_id / curriculum / f"seed_{seed}"
    completed = prepare_output_directory(output, "failure_event.json", root)
    if completed is not None:
        return completed
    paths = frozen_paths(config, config_path)
    protocol = load_ppo_curriculum_protocol(paths["policy_config"])
    actor = load_actor(entry)
    policy = ActorPolicy(entry.get("name", model_id), actor)
    environment = PPOCurriculumEnvironment(
        paths["policy_config"], stage=curriculum,
        split=f"f15_visual_replay_{model_id}_{curriculum}", seeds=(seed,),
    )
    recorded = Path(event["trace_path"])
    with np.load(recorded, allow_pickle=False) as archive:
        recorded_actions = np.asarray(archive["normalized_action"], dtype=np.float32)
        recorded_physical = np.asarray(archive["physical_action"], dtype=np.float32)
        recorded_progress = np.asarray(archive["progress_m"], dtype=np.float32)
        physical_fields = np.asarray(archive["public_physical_29d"], dtype=np.float32)
        feature_names = [str(value) for value in archive["feature_names"]]
        recorded_flags = {
            name: np.asarray(archive[name], dtype=bool)
            for name in ("collision", "unsafe", "lane_failure", "invalid_pose", "stop_violation", "completed")
        }
    lateral_index = feature_names.index("lane_lateral_error_mean_m")
    heading_index = feature_names.index("lane_heading_error_mean_rad")
    observation, info = environment.reset(seed=seed)
    frame0 = environment.latest_rgb()
    height, width = frame0.shape[:2]
    full_writer = _writer(output / "full_episode.mp4", width, height)
    window_writer = _writer(output / "failure_window.mp4", width, height)
    window_frames: list[np.ndarray] = []
    # This CSV is derived from the frozen primary telemetry, independently of whether
    # the descriptive simulator replay reaches the requested visual window.
    trace_rows: list[dict[str, Any]] = []
    for trace_step in range(start, min(stop + 1, len(recorded_actions))):
        trace_row = {
            "model_id": model_id, "model_sha256": entry["sha256"], "curriculum": curriculum,
            "seed": seed, "step": trace_step, "progress_m": float(recorded_progress[trace_step]),
            "v_cmd_mps": float(recorded_physical[trace_step][0]),
            "omega_cmd_rad_s": float(recorded_physical[trace_step][1]),
            "collision": bool(recorded_flags["collision"][trace_step]),
            "unsafe": bool(recorded_flags["unsafe"][trace_step]),
            "lane_failure": bool(recorded_flags["lane_failure"][trace_step]),
            "invalid_pose": bool(recorded_flags["invalid_pose"][trace_step]),
            "stop_violation": bool(recorded_flags["stop_violation"][trace_step]),
            "completed": bool(recorded_flags["completed"][trace_step]),
        }
        trace_row.update({
            name: float(value)
            for name, value in zip(feature_names, physical_fields[trace_step], strict=True)
        })
        trace_rows.append(trace_row)
    replayed_actions: list[np.ndarray] = []
    replay_progress: list[float] = []
    replay_step_rows: list[dict[str, Any]] = []
    try:
        for step in range(protocol.stage(curriculum).episode_horizon_steps):
            if step >= len(recorded_actions):
                break
            # Drive the simulator with the exact recorded normalized action rather than
            # re-executing the policy.  The frozen perception front-end runs
            # nondeterministic CUDA kernels, so a policy re-execution diverges from the
            # recorded episode; replaying the recorded actions reproduces the trajectory
            # that the primary telemetry actually describes.
            action = recorded_actions[step]
            observation, _, terminated, truncated, next_info = environment.step(action)
            replayed_actions.append(np.asarray(action, dtype=np.float32))
            replay_progress.append(float(next_info["progress_m"]))
            replay_step_rows.append({
                "step": step,
                "collision": bool(next_info["collision"]),
                "unsafe": bool(next_info["unsafe_proximity"]),
                "stop_violation": bool(next_info["stop_violation"]),
                "lane_failure": bool(next_info["lane_failure"]),
                "invalid_pose": bool(next_info["invalid_pose"]),
                "timeout": bool(next_info["truncation_reason"]),
                "terminated": bool(terminated),
                "truncated": bool(truncated),
                "completed": bool(next_info["completed"]),
            })
            failure_now = step == failure_step
            annotated = annotate(
                environment.latest_rgb(),
                [
                    f"{entry.get('name', model_id)} | {curriculum.upper()} | seed {seed} | step {step}",
                    f"progress={float(recorded_progress[step]):.2f} m | v_cmd={float(recorded_physical[step][0]):.3f} m/s | omega_cmd={float(recorded_physical[step][1]):.3f} rad/s",
                    f"lat_err={float(physical_fields[step][lateral_index]):+.3f} m | head_err={float(physical_fields[step][heading_index]):+.3f} rad",
                    f"event={event['event_labels'] if failure_now else 'none'}",
                ],
                failure=failure_now,
            )
            full_writer.write(annotated)
            if start <= step <= stop:
                window_writer.write(annotated)
                window_frames.append(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB))
            info = next_info
            if terminated or truncated:
                break
    finally:
        full_writer.release(); window_writer.release(); environment.close()
    validation = validate_recorded_action_replay(
        recorded_trace=recorded,
        expected_trace_sha256=str(event["trace_sha256"]),
        replayed_actions=np.asarray(replayed_actions, dtype=np.float32),
        recorded_actions=recorded_actions,
        replay_progress=replay_progress,
        recorded_progress=recorded_progress,
        replay_step_rows=replay_step_rows,
        recorded_event_step=None if event["event_step"] is None else int(event["event_step"]),
        recorded_event_labels=str(event["event_labels"]),
    )

    # Render the paired Original rollout separately to avoid mixing policies or
    # privileged state.  The visual comparison is a same-seed paired rollout.
    with (root / "localization/matrix_episodes.csv").open(newline="", encoding="utf-8") as stream:
        matrix_rows = list(csv.DictReader(stream))
    original_row = next(
        row for row in matrix_rows
        if row["model_id"] == "A0" and row["curriculum"] == curriculum and int(row["seed"]) == seed
    )
    original_recorded = Path(original_row["trace_path"])
    with np.load(original_recorded, allow_pickle=False) as archive:
        original_actions = np.asarray(archive["normalized_action"], dtype=np.float32)
        original_physical = np.asarray(archive["physical_action"], dtype=np.float32)
        original_progress = np.asarray(archive["progress_m"], dtype=np.float32)
    original_environment = PPOCurriculumEnvironment(
        paths["policy_config"], stage=curriculum,
        split=f"f15_visual_replay_A0_{curriculum}", seeds=(seed,),
    )
    original_observation, _ = original_environment.reset(seed=seed)
    original_full_writer = _writer(output / "original_full_episode.mp4", width, height)
    original_window_writer = _writer(output / "original_failure_window.mp4", width, height)
    original_window_frames: list[np.ndarray] = []
    original_replay_error = 0.0
    original_steps_replayed = 0
    original_last_raw: np.ndarray | None = None
    original_last_step = -1
    try:
        for step in range(protocol.stage(curriculum).episode_horizon_steps):
            if step >= len(original_actions):
                break
            original_action = original_actions[step]
            original_observation, _, terminated, truncated, original_info = original_environment.step(original_action)
            original_steps_replayed += 1
            original_last_raw = np.asarray(original_environment.latest_rgb()).copy()
            original_last_step = step
            original_replay_error = max(
                original_replay_error,
                abs(float(original_info["progress_m"]) - float(original_progress[step])),
            )
            frame = annotate(
                original_environment.latest_rgb(),
                [
                    f"Original Policy | {curriculum.upper()} | seed {seed} | step {step}",
                    f"progress={float(original_progress[step]):.2f} m | v_cmd={float(original_physical[step][0]):.3f} m/s | omega_cmd={float(original_physical[step][1]):.3f} rad/s",
                    "paired reference at compressed-policy failure time" if step == failure_step else "event=none",
                ],
                failure=step == failure_step,
            )
            original_full_writer.write(frame)
            if start <= step <= stop:
                original_window_writer.write(frame)
                original_window_frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            if terminated or truncated:
                break
    finally:
        original_full_writer.release(); original_window_writer.release(); original_environment.close()
    # The paired Original rollout carries no frozen failure event of its own, so it is
    # validated on trajectory reproduction only.
    original_validation = {
        "max_absolute_progress_error_m": original_replay_error,
        "pass_exact": original_replay_error <= REPLAY_EXACT_PROGRESS_M,
        "pass_tolerant": original_replay_error <= REPLAY_TOLERANT_PROGRESS_M,
        "steps_replayed": original_steps_replayed,
        "steps_recorded": int(len(original_actions)),
        "full_length_reproduced": original_steps_replayed == len(original_actions),
    }

    original_padding_frames = 0
    if len(original_window_frames) < len(window_frames):
        if original_last_raw is None:
            raise RuntimeError("Original recorded-action replay produced no frame")
        held = annotate(
            original_last_raw,
            [
                f"Original Policy | {curriculum.upper()} | seed {seed} | replay step {original_last_step}",
                "Recorded-action reconstruction ended before the compressed failure window",
                "Last reconstructed frame held; paired visual is quarantined unless validation passes",
            ],
            failure=False,
        )
        held_rgb = cv2.cvtColor(held, cv2.COLOR_BGR2RGB)
        original_padding_frames = len(window_frames) - len(original_window_frames)
        original_window_frames.extend([held_rgb.copy() for _ in range(original_padding_frames)])
    original_validation["held_frame_padding_count"] = original_padding_frames
    original_validation["usable_as_paired_visual_evidence"] = bool(
        original_validation["pass_tolerant"]
        and original_validation["full_length_reproduced"]
        and original_padding_frames == 0
    )

    pair_count = min(len(original_window_frames), len(window_frames))
    if pair_count == 0:
        with (output / "trace.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(trace_rows[0]))
            writer.writeheader(); writer.writerows(trace_rows)
        media_names = (
            "full_episode.mp4", "failure_window.mp4",
            "original_full_episode.mp4", "original_failure_window.mp4",
        )
        files = {
            "trace.csv": {"path": str(output / "trace.csv"), "sha256": file_sha256(output / "trace.csv")}
        }
        quarantine = output / "unresolved"
        quarantine.mkdir(exist_ok=True)
        for name in media_names:
            source = output / name
            if source.exists():
                destination = quarantine / name
                source.replace(destination)
                files[name] = {"path": str(destination), "sha256": file_sha256(destination)}
        unresolved_event = {
            **dict(event), "model_sha256": entry["sha256"], "config_sha256": config["_sha256"],
            "primary_trace_sha256": file_sha256(recorded),
            "original_primary_trace_sha256": file_sha256(original_recorded),
            "replay_method": "recorded_action_same_seed_replay",
            "policy_reinferred_during_rendering": False,
            "overlay_source": "frozen primary telemetry (not recomputed during replay)",
            "visual_reconstruction_status": "UNRESOLVED",
            "visual_reconstruction_validation": validation,
            "original_paired_replay_validation": original_validation,
            "pairing_claim": "same-seed paired rollout; not a causal paired trajectory",
            "visual_replay_is_descriptive_only": True,
            "amendment": "docs/F15_VISUAL_REPLAY_IMPLEMENTATION_AMENDMENT.md",
            "files": files,
            "quarantined": True,
            "quarantine_reason": (
                "the descriptive replay ended before the frozen failure window, so no aligned "
                "frames could be reconstructed; the primary telemetry CSV remains authoritative"
            ),
        }
        write_json(output / "failure_event.json", unresolved_event)
        return unresolved_event
    paired_frames = [
        np.concatenate((original_window_frames[index], window_frames[index]), axis=1)
        for index in range(pair_count)
    ]
    paired_writer = _writer(output / "paired_failure_window.mp4", width * 2, height)
    try:
        for frame in paired_frames:
            paired_writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    finally:
        paired_writer.release()
    imageio.mimsave(
        output / "paired_failure_window.gif",
        [frame for index, frame in enumerate(paired_frames) if index % 3 == 0],
        duration=0.1,
        loop=0,
    )
    with (output / "trace.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(trace_rows[0]))
        writer.writeheader(); writer.writerows(trace_rows)
    gif_frames = [frame for index, frame in enumerate(window_frames) if index % 3 == 0]
    imageio.mimsave(output / "failure_window.gif", gif_frames, duration=0.1, loop=0)
    indexes = np.linspace(0, len(window_frames) - 1, min(8, len(window_frames)), dtype=int)
    thumbs = [Image.fromarray(window_frames[index]).resize((320, 240)) for index in indexes]
    sheet = Image.new("RGB", (320 * 4, 240 * int(np.ceil(len(thumbs) / 4))), "white")
    for index, thumb in enumerate(thumbs):
        sheet.paste(thumb, ((index % 4) * 320, (index // 4) * 240))
    sheet.save(output / "contact_sheet.png", dpi=(300, 300))
    paired_indexes = np.linspace(0, len(paired_frames) - 1, min(8, len(paired_frames)), dtype=int)
    paired_thumbs = [Image.fromarray(paired_frames[index]).resize((640, 240)) for index in paired_indexes]
    paired_sheet = Image.new("RGB", (640 * 2, 240 * int(np.ceil(len(paired_thumbs) / 2))), "white")
    for index, thumb in enumerate(paired_thumbs):
        paired_sheet.paste(thumb, ((index % 2) * 640, (index // 2) * 240))
    paired_sheet.save(output / "paired_contact_sheet.png", dpi=(300, 300))
    event_json = {
        **dict(event), "model_sha256": entry["sha256"], "config_sha256": config["_sha256"],
        "primary_trace_sha256": file_sha256(recorded),
        "original_primary_trace_sha256": file_sha256(original_recorded),
        "replay_method": "recorded_action_same_seed_replay",
        "replay_method_label": "Recorded-Action Same-Seed Replay",
        "replay_method_reason": (
            "the frozen F10 perception front-end runs nondeterministic CUDA kernels, so "
            "re-executing the policy does not reproduce the recorded episode; the simulator "
            "is therefore driven by the exact recorded normalized actions"
        ),
        "policy_reinferred_during_rendering": False,
        "overlay_source": "frozen primary telemetry (not recomputed during replay)",
        "visual_reconstruction_status": validation["status"],
        "visual_reconstruction_validation": validation,
        "original_paired_replay_validation": original_validation,
        "pairing_claim": "same-seed paired rollout; not a causal paired trajectory",
        "visual_replay_is_descriptive_only": True,
        "amendment": "docs/F15_VISUAL_REPLAY_IMPLEMENTATION_AMENDMENT.md",
        "files": {name: {"path": str(output / name), "sha256": file_sha256(output / name)} for name in (
            "full_episode.mp4", "failure_window.mp4", "failure_window.gif", "contact_sheet.png", "trace.csv",
            "original_full_episode.mp4", "original_failure_window.mp4", "paired_failure_window.mp4",
            "paired_failure_window.gif", "paired_contact_sheet.png",
        )},
    }
    if not validation["usable_as_visual_evidence"] or not original_validation["usable_as_paired_visual_evidence"]:
        # The reconstruction did not reproduce the frozen telemetry. Quarantine the media
        # so it cannot be mistaken for evidence, but keep it (and the validation report)
        # as the record of the attempt. The primary telemetry remains authoritative.
        quarantine = output / "unresolved"
        quarantine.mkdir(exist_ok=True)
        for name in list(event_json["files"]):
            source = output / name
            if source.exists():
                source.replace(quarantine / name)
            event_json["files"][name]["path"] = str(quarantine / name)
        event_json["quarantined"] = True
        event_json["quarantine_reason"] = (
            "compressed or paired-Original recorded-action replay did not reproduce the "
            "frozen primary telemetry within the preregistered tolerances; media retained "
            "for audit but not usable as evidence"
        )
    write_json(output / "failure_event.json", event_json)
    return event_json


def _truth(value: Any) -> bool:
    return value if isinstance(value, bool) else str(value).lower() == "true"


def select_success_row(rows: list[dict[str, Any]], model_id: str, curriculum: str) -> dict[str, Any] | None:
    """Choose the lowest-seed objectively successful episode for one cell."""
    eligible = [
        row for row in rows
        if row["model_id"] == model_id
        and row["curriculum"] == curriculum
        and _truth(row["completed"])
        and not _truth(row["collision"])
        and int(row["unsafe_proximity_events"]) == 0
        and not _truth(row["stop_violation"])
        and not _truth(row["lane_failure"])
        and not _truth(row["invalid_pose"])
        and not _truth(row["timeout"])
    ]
    return min(eligible, key=lambda row: int(row["seed"])) if eligible else None


def render_success_one(
    config: Mapping[str, Any], config_path: Path, row: Mapping[str, Any], entry: Mapping[str, Any]
) -> dict[str, Any]:
    """Render one deterministic successful episode; primary telemetry remains canonical."""
    root = artifact_root(config, config_path)
    model_id, curriculum, seed = str(row["model_id"]), str(row["curriculum"]), int(row["seed"])
    output = root / "success_traces" / model_id / curriculum / f"seed_{seed}"
    completed = prepare_output_directory(output, "success_event.json", root)
    if completed is not None:
        return completed
    paths = frozen_paths(config, config_path)
    protocol = load_ppo_curriculum_protocol(paths["policy_config"])
    recorded = Path(row["trace_path"])
    with np.load(recorded, allow_pickle=False) as archive:
        recorded_actions = np.asarray(archive["normalized_action"], dtype=np.float32)
        recorded_physical = np.asarray(archive["physical_action"], dtype=np.float32)
        recorded_progress = np.asarray(archive["progress_m"], dtype=np.float32)
        recorded_completed = np.asarray(archive["completed"], dtype=bool)
    environment = PPOCurriculumEnvironment(
        paths["policy_config"], stage=curriculum,
        split=f"f15_success_visual_replay_{model_id}_{curriculum}", seeds=(seed,),
    )
    observation, _ = environment.reset(seed=seed)
    first = environment.latest_rgb()
    height, width = first.shape[:2]
    video_path = output / "representative_success.mp4"
    writer = _writer(video_path, width, height)
    sample_steps = set(np.linspace(0, max(0, len(recorded_actions) - 1), min(8, len(recorded_actions)), dtype=int).tolist())
    sampled: list[np.ndarray] = []
    replay_error = 0.0
    replayed_actions: list[np.ndarray] = []
    replay_step_rows: list[dict[str, Any]] = []
    replay_completed = False
    try:
        for step in range(protocol.stage(curriculum).episode_horizon_steps):
            if step >= len(recorded_actions):
                break
            # Recorded-action playback; see render_one for why the policy is not re-executed.
            action = recorded_actions[step]
            observation, _, terminated, truncated, info = environment.step(action)
            replayed_actions.append(np.asarray(action, dtype=np.float32))
            replay_error = max(replay_error, abs(float(info["progress_m"]) - float(recorded_progress[step])))
            replay_step_rows.append({
                "step": step,
                "collision": bool(info["collision"]),
                "unsafe": bool(info["unsafe_proximity"]),
                "stop_violation": bool(info["stop_violation"]),
                "lane_failure": bool(info["lane_failure"]),
                "invalid_pose": bool(info["invalid_pose"]),
                "timeout": bool(info["truncation_reason"]),
                "terminated": bool(terminated),
                "truncated": bool(truncated),
                "completed": bool(info["completed"]),
            })
            frame = annotate(
                environment.latest_rgb(),
                [
                    f"{entry.get('name', model_id)} | {curriculum.upper()} | seed {seed} | step {step}",
                    f"progress={float(recorded_progress[step]):.2f} m | v_cmd={float(recorded_physical[step][0]):.3f} m/s | omega_cmd={float(recorded_physical[step][1]):.3f} rad/s",
                    "objectively selected representative success",
                ],
                failure=False,
            )
            writer.write(frame)
            if step in sample_steps:
                sampled.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            if terminated or truncated:
                replay_completed = bool(info["completed"])
                break
    finally:
        writer.release(); environment.close()
    recorded_success = bool(recorded_completed[-1])
    trace_hash_matches = file_sha256(recorded) == str(row["trace_sha256"])
    replayed_action_array = np.asarray(replayed_actions, dtype=np.float32)
    action_identical = bool(
        replayed_action_array.shape == recorded_actions.shape
        and np.array_equal(replayed_action_array, recorded_actions)
    )
    replay_failure = first_objective_failure_event(replay_step_rows)
    if not recorded_success:
        raise RuntimeError("selected primary telemetry row is not a recorded completion")
    exact = bool(
        trace_hash_matches and action_identical and replay_error <= REPLAY_EXACT_PROGRESS_M
        and replay_completed and replay_failure is None
    )
    tolerant = bool(
        trace_hash_matches and action_identical and replay_error <= REPLAY_TOLERANT_PROGRESS_M
        and replay_completed and replay_failure is None
    )
    status = "VERIFIED_EXACT" if exact else ("VERIFIED_WITHIN_TOLERANCE" if tolerant else "UNRESOLVED")
    thumbs = [Image.fromarray(frame).resize((320, 240)) for frame in sampled]
    sheet = Image.new("RGB", (320 * 4, 240 * int(np.ceil(len(thumbs) / 4))), "white")
    for index, thumb in enumerate(thumbs):
        sheet.paste(thumb, ((index % 4) * 320, (index // 4) * 240))
    sheet_path = output / "contact_sheet.png"
    sheet.save(sheet_path, dpi=(300, 300))
    metadata = {
        "model_id": model_id, "model_name": entry.get("name", model_id), "model_sha256": entry["sha256"],
        "curriculum": curriculum, "seed": seed,
        "selection_rule": "lowest seed with completion and no objective failure flag",
        "primary_trace_path": str(recorded), "primary_trace_sha256": file_sha256(recorded),
        "replay_method": "recorded_normalized_action_playback",
        "overlay_source": "frozen primary telemetry (not recomputed during replay)",
        "visual_replay_max_progress_error_m": replay_error,
        "visual_reconstruction_status": status,
        "visual_reconstruction_validation": {
            "trace_sha256_matches_primary_row": trace_hash_matches,
            "replayed_actions_identical_to_telemetry": action_identical,
            "exact_progress_threshold_m": REPLAY_EXACT_PROGRESS_M,
            "tolerant_progress_threshold_m": REPLAY_TOLERANT_PROGRESS_M,
            "replay_completed": replay_completed,
            "replay_failure_event": replay_failure,
        },
        "visual_replay_is_descriptive_only": True,
        "files": {
            "representative_success.mp4": {"path": str(video_path), "sha256": file_sha256(video_path)},
            "contact_sheet.png": {"path": str(sheet_path), "sha256": file_sha256(sheet_path)},
        },
    }
    if status == "UNRESOLVED":
        quarantine = output / "unresolved"
        quarantine.mkdir(exist_ok=True)
        for name in list(metadata["files"]):
            source = output / name
            if source.exists():
                source.replace(quarantine / name)
            metadata["files"][name]["path"] = str(quarantine / name)
        metadata["quarantined"] = True
        metadata["quarantine_reason"] = (
            "recorded-action replay did not reproduce the successful primary telemetry "
            "within the preregistered tolerances"
        )
    write_json(output / "success_event.json", metadata)
    return metadata


def main() -> None:
    config = load_config(CONFIG)
    verify_protocol(config, CONFIG)
    root = artifact_root(config, CONFIG)
    decision = read_json(root / "localization/failure_localization_decision.json")
    paths = frozen_paths(config, CONFIG)
    matrix = verify_registry(paths["ablation_registry"], expected_registry_sha256=config["frozen"]["f12_ablation_registry_sha256"], collection_key="variants")
    pruning = verify_registry(paths["pruning_registry"], expected_registry_sha256=config["frozen"]["f12_pruning_registry_sha256"], collection_key="candidates")
    rendered = []
    for event in decision["failure_events"]:
        registry = matrix if event["family"] == "matrix" else pruning
        rendered.append(render_one(config, CONFIG, event, registry[event["model_id"]], matrix["A0"]))
    status_counts: dict[str, int] = {}
    for item in rendered:
        key = str(item.get("visual_reconstruction_status", "UNKNOWN"))
        status_counts[key] = status_counts.get(key, 0) + 1
    write_json(root / "failure_traces/failure_trace_manifest.json", {
        "schema_version": 1, "created_at_utc": decision["created_at_utc"], "events": rendered,
        "selection_rule": config["evaluation"]["representative_failure_rule"],
        "replay_method": "recorded_action_same_seed_replay",
        "replay_method_label": "Recorded-Action Same-Seed Replay",
        "policy_reinferred_during_rendering": False,
        "amendment": "docs/F15_VISUAL_REPLAY_IMPLEMENTATION_AMENDMENT.md",
        "visual_reconstruction_status_counts": status_counts,
        "primary_evidence": "frozen F15 telemetry; video is descriptive reconstruction only",
    })
    successful = []
    for family, registry, result_name, episode_name in (
        ("matrix", matrix, "matrix_results.json", "matrix_episodes.csv"),
        ("pruning", pruning, "pruning_results.json", "pruning_episodes.csv"),
    ):
        result = read_json(root / "localization" / result_name)
        with (root / "localization" / episode_name).open(newline="", encoding="utf-8") as stream:
            episode_rows = list(csv.DictReader(stream))
        for model_id, curricula in result["decisions"].items():
            for curriculum, decision_cell in curricula.items():
                if decision_cell["status"] not in {"REFERENCE", "PASS"}:
                    continue
                row = select_success_row(episode_rows, model_id, curriculum)
                if row is None:
                    raise RuntimeError(f"passing {family} cell has no objectively successful episode: {model_id}/{curriculum}")
                successful.append(render_success_one(config, CONFIG, row, registry[model_id]))
    write_json(root / "success_traces/success_trace_manifest.json", {
        "schema_version": 1, "created_at_utc": decision["created_at_utc"], "episodes": successful,
        "selection_rule": "lowest seed with completion and no objective failure flag",
        "visual_reconstruction_status_counts": {
            status: sum(item.get("visual_reconstruction_status") == status for item in successful)
            for status in ("VERIFIED_EXACT", "VERIFIED_WITHIN_TOLERANCE", "UNRESOLVED")
        },
    })
    print(json.dumps({"rendered_failure_traces": len(rendered), "rendered_success_traces": len(successful)}, indent=2))


if __name__ == "__main__":
    main()
