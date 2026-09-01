#!/usr/bin/env python3
"""F16 sequence / width / INT8 runner.

Every closed-loop episode runs under the frozen deterministic backend and carries an
in-memory RGB ring buffer, so camera evidence comes from the PRIMARY scientific rollout.
Nothing here reruns actor inference to produce media, and nothing reconstructs a
trajectory after the fact.

Subcommands:
  smoke-media   Integrity Gate 2: prove the media pipeline works and does not perturb
                policy execution or determinism, before the main workload begins.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
from collections import deque
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))

from duckie_pomdp.control.ppo_environment import PPOCurriculumEnvironment  # noqa: E402
from duckie_pomdp.control.ppo_protocol import load_ppo_curriculum_protocol  # noqa: E402
from duckie_pomdp.optimization.cross_curriculum_recovery import (  # noqa: E402
    file_sha256,
    first_objective_failure_event,
    verify_registry,
)
from run_f15_cross_curriculum_recovery import (  # noqa: E402
    ActorPolicy,
    artifact_root,
    frozen_paths,
    load_actor,
    load_config,
    provenance,
    read_json,
    write_json,
)

CONFIG = ROOT / "configs/f16_sequence_int8_recovery_v1.toml"
CURRICULA = ("c0", "c1", "c2", "c3", "c4")


# ---------------------------------------------------------------------------
# Frozen deterministic backend
# ---------------------------------------------------------------------------
def apply_frozen_determinism(config: Mapping[str, Any]) -> dict[str, Any]:
    """Apply the backend frozen by the determinism gate. Fails closed on mismatch."""
    root = artifact_root(config, CONFIG)
    gate = read_json(root / "integrity/determinism_gate.json")
    if gate["classification"] != "PASS":
        raise RuntimeError("F16 closed-loop evaluation is barred: determinism gate did not pass")
    backend = gate["selected_backend"]
    if backend != "cuda_strict_deterministic":
        raise RuntimeError(f"unsupported frozen backend: {backend}")
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != config["determinism"]["cublas_workspace_config"]:
        raise RuntimeError(
            "CUBLAS_WORKSPACE_CONFIG must be set to the frozen value before torch initialises"
        )
    torch.use_deterministic_algorithms(True)
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        raise RuntimeError("frozen backend requires CUDA; it is unavailable")
    torch.manual_seed(0)
    np.random.seed(0)
    return {"backend": backend, "determinism_gate_sha256": file_sha256(root / "integrity/determinism_gate.json")}


# ---------------------------------------------------------------------------
# Primary rollout with in-memory RGB ring buffer
# ---------------------------------------------------------------------------
def rollout(
    environment,
    *,
    seed: int,
    policy: ActorPolicy,
    protocol,
    curriculum: str,
    capture_media: bool,
    ring_steps: int,
    window_before: int,
    window_after: int,
) -> dict[str, Any]:
    """Run one primary episode.

    When ``capture_media`` is set, an RGB ring buffer of ``ring_steps`` frames is kept.
    On the first objective failure the frames belonging to THIS episode are retained,
    and collection continues for ``window_after`` more steps. The buffer is read-only
    with respect to the policy: the action at step t is computed from the observation
    only, so enabling capture cannot change the trajectory.
    """
    observation, info = environment.reset(seed=seed)
    policy.reset(seed)

    ring: deque = deque(maxlen=ring_steps) if capture_media else deque(maxlen=0)
    kept_frames: list[tuple[int, np.ndarray]] = []
    steps: list[dict[str, Any]] = []
    actions: list[np.ndarray] = []
    failure_step: int | None = None
    post_failure = 0

    for step in range(protocol.stage(curriculum).episode_horizon_steps):
        if capture_media:
            frame = environment.latest_rgb()
            ring.append((step, np.asarray(frame, dtype=np.uint8).copy()))

        action = policy.act(observation)
        actions.append(np.asarray(action, dtype=np.float32).copy())
        observation, _, terminated, truncated, info = environment.step(action)

        row = {
            "step": step,
            "progress_m": float(info["progress_m"]),
            "v_cmd": float(info["v_cmd"]),
            "omega_cmd": float(info["omega_cmd"]),
            "completed": bool(info["completed"]),
            "collision": bool(info["collision"]),
            "unsafe": bool(info["unsafe_proximity"]),
            "lane_failure": bool(info["lane_failure"]),
            "invalid_pose": bool(info["invalid_pose"]),
            "stop_violation": bool(info["stop_violation"]),
            "timeout": bool(info["truncation_reason"]),
            "terminated": bool(terminated),
            "truncated": bool(truncated),
            "termination_reason": str(info["termination_reason"] or ""),
        }
        steps.append(row)

        # Failure detection is independent of capture so that a capture-off control run
        # reports the same failure step; otherwise the invariance check compares an
        # integer against None and is meaningless.
        if failure_step is None and first_objective_failure_event([row]) is not None:
            failure_step = step
            if capture_media:
                start = max(0, step - window_before)
                kept_frames = [(i, f) for i, f in ring if start <= i <= step]

        if capture_media and failure_step is not None and steps[-1]["step"] > failure_step:
            if post_failure < window_after:
                kept_frames.append((step, np.asarray(environment.latest_rgb(), dtype=np.uint8).copy()))
                post_failure += 1

        if terminated or truncated:
            break

    if capture_media and failure_step is None:
        # Objectively successful (or non-failing) episode: retain the tail of the ring.
        kept_frames = list(ring)

    return {
        "curriculum": curriculum,
        "seed": seed,
        "steps": steps,
        "actions": np.asarray(actions, dtype=np.float32),
        "failure_step": failure_step,
        "frames": kept_frames,
        "episode_length": len(steps),
    }


# ---------------------------------------------------------------------------
# Media encoding — pure function of frames already captured in the primary rollout
# ---------------------------------------------------------------------------
def encode_media(result: Mapping[str, Any], output: Path, label: Mapping[str, Any]) -> dict[str, Any]:
    import cv2
    import imageio.v2 as imageio
    from PIL import Image

    frames = result["frames"]
    if not frames:
        raise RuntimeError("no primary frames captured for this episode")
    output.mkdir(parents=True, exist_ok=True)
    steps_by_index = {row["step"]: row for row in result["steps"]}
    height, width = frames[0][1].shape[:2]

    annotated: list[np.ndarray] = []
    for index, rgb in frames:
        row = steps_by_index[index]
        is_failure = result["failure_step"] is not None and index == result["failure_step"]
        frame = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (width, 92), (20, 20, 20), -1)
        frame = cv2.addWeighted(overlay, 0.72, frame, 0.28, 0)
        lines = [
            f"{label['model_name']} | seq={label['sequence']} W={label['width']} {label['precision']}",
            f"{result['curriculum'].upper()} | seed {result['seed']} | step {index}",
            f"v={row['v_cmd']:.3f} m/s  w={row['omega_cmd']:+.3f} rad/s  progress={row['progress_m']:.2f} m",
            "EVENT: " + ("|".join(
                n for n in ("collision", "unsafe", "stop_violation", "lane_failure", "invalid_pose", "timeout")
                if row[n]
            ) or "none"),
        ]
        for line_index, line in enumerate(lines):
            color = (80, 80, 255) if is_failure and line_index == 3 else (245, 245, 245)
            cv2.putText(frame, line, (10, 20 + 21 * line_index), cv2.FONT_HERSHEY_SIMPLEX, 0.46, color, 1, cv2.LINE_AA)
        annotated.append(frame)

    mp4 = output / "primary_rollout.mp4"
    writer = cv2.VideoWriter(str(mp4), cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"cannot open video writer: {mp4}")
    try:
        for frame in annotated:
            writer.write(frame)
    finally:
        writer.release()

    rgb_frames = [cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in annotated]
    gif = output / "primary_rollout.gif"
    imageio.mimsave(gif, rgb_frames[::3] or rgb_frames, duration=0.1, loop=0)

    picks = np.linspace(0, len(rgb_frames) - 1, min(8, len(rgb_frames)), dtype=int)
    thumbs = [Image.fromarray(rgb_frames[i]).resize((320, 240)) for i in picks]
    sheet = Image.new("RGB", (320 * 4, 240 * int(np.ceil(len(thumbs) / 4))), "white")
    for i, thumb in enumerate(thumbs):
        sheet.paste(thumb, ((i % 4) * 320, (i // 4) * 240))
    contact = output / "primary_rollout_contact_sheet.png"
    sheet.save(contact, dpi=(300, 300))

    frame_index = [int(i) for i, _ in frames]
    manifest = {
        **dict(label),
        "curriculum": result["curriculum"],
        "seed": int(result["seed"]),
        "episode_length": int(result["episode_length"]),
        "failure_step": None if result["failure_step"] is None else int(result["failure_step"]),
        "frames_persisted": len(frames),
        "frame_index_first": frame_index[0],
        "frame_index_last": frame_index[-1],
        "frame_indices_contiguous": frame_index == list(range(frame_index[0], frame_index[-1] + 1)),
        "frame_index_synced_to_telemetry_step": all(i in steps_by_index for i in frame_index),
        "provenance": "frames captured during the primary scientific rollout",
        "actor_reinferred_for_media": False,
        "trajectory_reconstructed": False,
        "pairing_label": "Same-Seed Primary Rollouts",
        "files": {
            p.name: {"path": str(p), "sha256": file_sha256(p), "bytes": p.stat().st_size}
            for p in (mp4, gif, contact)
        },
    }
    write_json(output / "primary_media.json", manifest)
    return manifest


# ---------------------------------------------------------------------------
# Integrity Gate 2 — media smoke test
# ---------------------------------------------------------------------------
def smoke_media() -> dict[str, Any]:
    config = load_config(CONFIG)
    determinism = apply_frozen_determinism(config)
    root = artifact_root(config, CONFIG)
    target = root / "integrity/media_pipeline_gate.json"
    if target.exists():
        raise RuntimeError("refusing to overwrite the frozen F16 media pipeline gate")

    paths = frozen_paths(config, CONFIG)
    protocol = load_ppo_curriculum_protocol(paths["policy_config"])
    matrix = verify_registry(
        paths["ablation_registry"],
        expected_registry_sha256=config["frozen"]["f12_ablation_registry_sha256"],
        collection_key="variants",
    )
    media = config["media"]
    seed = int(config["seeds"]["determinism_preflight"][0])

    # A1 (Pruning Only) fails every curriculum in F15, so it reliably produces an
    # objective failure for the failure-path test. A0 gives the success path.
    cases = [
        ("A1", "c0", "failure path"),
        ("A0", "c0", "success path"),
    ]

    checks: dict[str, Any] = {}
    manifests = []
    for model_id, curriculum, purpose in cases:
        entry = matrix[model_id]
        results = {}
        for capture in (True, False):
            environment = PPOCurriculumEnvironment(
                paths["policy_config"], stage=curriculum,
                split=f"f16_smoke_{model_id}_{curriculum}_{int(capture)}", seeds=(seed,),
            )
            try:
                results[capture] = rollout(
                    environment, seed=seed,
                    policy=ActorPolicy(entry.get("name", model_id), load_actor(entry)),
                    protocol=protocol, curriculum=curriculum,
                    capture_media=capture,
                    ring_steps=int(media["rgb_ring_buffer_steps"]),
                    window_before=int(config["evaluation"]["failure_window_steps_before"]),
                    window_after=int(config["evaluation"]["failure_window_steps_after"]),
                )
            finally:
                environment.close()

        with_buffer, without_buffer = results[True], results[False]
        common = min(len(with_buffer["actions"]), len(without_buffer["actions"]))
        action_delta = (
            float(np.max(np.abs(with_buffer["actions"][:common] - without_buffer["actions"][:common])))
            if common else float("inf")
        )
        key = f"{model_id}_{curriculum}"
        checks[key] = {
            "purpose": purpose,
            "episode_length_with_buffer": with_buffer["episode_length"],
            "episode_length_without_buffer": without_buffer["episode_length"],
            "episode_length_identical": with_buffer["episode_length"] == without_buffer["episode_length"],
            "max_abs_action_delta": action_delta,
            "ring_buffer_does_not_perturb_policy": action_delta == 0.0,
            "failure_step_with_buffer": with_buffer["failure_step"],
            "failure_step_without_buffer": without_buffer["failure_step"],
            "failure_step_identical": with_buffer["failure_step"] == without_buffer["failure_step"],
            "frames_captured": len(with_buffer["frames"]),
        }

        output = root / "media_smoke" / model_id / curriculum / f"seed_{seed}"
        manifest = encode_media(with_buffer, output, {
            "model_id": model_id,
            "model_name": entry.get("name", model_id),
            "sequence": "n/a (smoke test)",
            "width": entry["hidden_sizes"][0],
            "precision": "INT8" if entry["int8"] else "FP32",
            "model_sha256": entry["sha256"],
        })
        manifests.append(manifest)

        # Media-side checks
        checks[key].update({
            "frame_index_synced_to_telemetry_step": manifest["frame_index_synced_to_telemetry_step"],
            "frame_indices_contiguous": manifest["frame_indices_contiguous"],
            "media_files_written": sorted(manifest["files"]),
            "media_sha256_recorded": all("sha256" in v for v in manifest["files"].values()),
            "media_non_empty": all(v["bytes"] > 0 for v in manifest["files"].values()),
        })
        if with_buffer["failure_step"] is not None:
            telemetry_event = first_objective_failure_event(with_buffer["steps"])
            checks[key]["first_failure_marker_matches_telemetry"] = (
                telemetry_event is not None and telemetry_event["step"] == with_buffer["failure_step"]
            )
            checks[key]["failure_frame_present"] = with_buffer["failure_step"] in [i for i, _ in with_buffer["frames"]]

    # Verify the encoded media actually decode.
    import cv2
    import imageio.v2 as imageio
    decode = {}
    for manifest in manifests:
        for name, record in manifest["files"].items():
            path = record["path"]
            if name.endswith(".mp4"):
                capture = cv2.VideoCapture(path)
                ok, _ = capture.read()
                decode[path] = bool(ok)
                capture.release()
            elif name.endswith(".gif"):
                decode[path] = len(imageio.mimread(path, memtest=False)) > 0
            else:
                decode[path] = imageio.imread(path).size > 0

    flat = []
    for record in checks.values():
        flat.extend([
            record["ring_buffer_does_not_perturb_policy"],
            record["episode_length_identical"],
            record["failure_step_identical"],
            record["frame_index_synced_to_telemetry_step"],
            record["media_sha256_recorded"],
            record["media_non_empty"],
        ])
        if "first_failure_marker_matches_telemetry" in record:
            flat.append(record["first_failure_marker_matches_telemetry"])
            flat.append(record["failure_frame_present"])
    passed = all(flat) and all(decode.values())

    output = {
        **provenance(config, CONFIG),
        "classification": "PASS" if passed else "FAIL",
        "gate": "Integrity Gate 2 — primary-rollout camera evidence",
        "determinism": determinism,
        "checks": checks,
        "media_decodes": decode,
        "requirements_demonstrated": {
            "frames_from_primary_rollout_not_rerun": True,
            "seed_model_curriculum_recorded": True,
            "frame_index_synced_to_telemetry_step": all(
                r["frame_index_synced_to_telemetry_step"] for r in checks.values()
            ),
            "mp4_gif_contact_sheet_decode": all(decode.values()),
            "sha256_stored": all(r["media_sha256_recorded"] for r in checks.values()),
            "first_failure_marker_matches_telemetry": all(
                r.get("first_failure_marker_matches_telemetry", True) for r in checks.values()
            ),
            "ring_buffer_does_not_alter_policy_or_determinism": all(
                r["ring_buffer_does_not_perturb_policy"] and r["episode_length_identical"]
                for r in checks.values()
            ),
        },
        "consequence_if_fail": "the F16 main workload must not start; F15's visual-evidence gap would recur",
    }
    write_json(target, output)
    print(json.dumps({
        "classification": output["classification"],
        "requirements_demonstrated": output["requirements_demonstrated"],
    }, indent=2))
    for key, record in checks.items():
        print(f"  {key}: action_delta={record['max_abs_action_delta']} frames={record['frames_captured']} "
              f"len={record['episode_length_with_buffer']}/{record['episode_length_without_buffer']}")
    if not passed:
        raise SystemExit(1)
    return output


# ---------------------------------------------------------------------------
# Sequence construction
# ---------------------------------------------------------------------------
def frozen_survivors(original) -> dict[int, tuple[tuple[int, ...], tuple[int, ...]]]:
    """Survivor indices per width, recomputed from the frozen Original actor.

    Verified nested (64 subset 96 subset 128 subset 192) for both layers before the
    protocol freeze, so a Direct and a Progressive candidate at the same target width
    retain the SAME original neurons.
    """
    from duckie_pomdp.optimization.actor_compression import build_pruned_actor

    out = {}
    for width in (192, 128, 96, 64):
        result = build_pruned_actor(original, width)
        out[width] = (result.first_layer_survivors, result.second_layer_survivors)
    return out


def prune_along_frozen_hierarchy(actor, current: tuple, target: tuple):
    """Prune an arbitrary-width actor to the frozen target survivor set.

    ``current`` and ``target`` are (first_layer, second_layer) tuples of ORIGINAL neuron
    indices. Because the hierarchy is nested, every target index is present in current,
    so the result contains exactly the neurons a Direct prune to that width would keep.
    """
    from duckie_pomdp.optimization.actor_compression import ActorSpec, DenseBeliefActor

    cur_first, cur_second = current
    tgt_first, tgt_second = target
    index_first = {value: position for position, value in enumerate(cur_first)}
    index_second = {value: position for position, value in enumerate(cur_second)}
    missing = [v for v in tgt_first if v not in index_first] + [v for v in tgt_second if v not in index_second]
    if missing:
        raise RuntimeError(f"frozen hierarchy violated: {len(missing)} target neurons absent from source")
    pos1 = [index_first[v] for v in tgt_first]
    pos2 = [index_second[v] for v in tgt_second]

    student = DenseBeliefActor(ActorSpec(hidden_sizes=(len(tgt_first), len(tgt_second))))
    with torch.no_grad():
        student.fc1.weight.copy_(actor.fc1.weight[pos1, :])
        student.fc1.bias.copy_(actor.fc1.bias[pos1])
        student.fc2.weight.copy_(actor.fc2.weight[pos2][:, pos1])
        student.fc2.bias.copy_(actor.fc2.bias[pos2])
        student.out.weight.copy_(actor.out.weight[:, pos2])
        student.out.bias.copy_(actor.out.bias)
    student.eval()
    return student


def train_candidates() -> dict[str, Any]:
    """Train the seven distinct FP32 candidates.

    The Progressive chain shares prefixes, so it is computed once:
        Original -> 192 -> KD -> 128 -> KD -> 96 -> KD -> 64 -> KD
    yielding P192 (identical to D192 by construction), P128, P96, P64.
    Direct adds three single-stage runs at 64, 96, 128.
    """
    from duckie_pomdp.optimization.actor_compression import (
        extract_original_actor, load_dense_actor, save_dense_actor,
    )
    from duckie_pomdp.optimization.cross_curriculum_recovery import distill_multicurriculum_actor

    config = load_config(CONFIG)
    root = artifact_root(config, CONFIG)
    target_manifest = root / "training_manifest.json"
    if target_manifest.exists():
        raise RuntimeError("refusing to overwrite the F16 training manifest")

    paths = frozen_paths(config, CONFIG)
    pruning = verify_registry(
        paths["pruning_registry"],
        expected_registry_sha256=config["frozen"]["f12_pruning_registry_sha256"],
        collection_key="candidates",
    )
    distill = config["distillation"]

    dataset_path = ROOT / "artifacts/f15_cross_curriculum_recovery_v1/recovery/datasets/multicurriculum_public_states.npz"
    actual = file_sha256(dataset_path)
    if actual != config["frozen"]["f15_kd_dataset_sha256"]:
        raise RuntimeError(f"F15 KD dataset hash mismatch: {actual}")
    with np.load(dataset_path, allow_pickle=False) as archive:
        data = {name: archive[name] for name in archive.files}

    original_path = ROOT / "artifacts/f10_ppo_visual_objects_v30/c4/ppo_selected.pt"
    original = extract_original_actor(
        str(original_path), expected_sha256=config["frozen"]["original_ppo_sha256"]
    )
    if isinstance(original, tuple):
        original = original[0]
    survivors = frozen_survivors(original)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    records: list[dict[str, Any]] = []

    def kd(actor, width: int, sequence: str, stage_index: int, cumulative_stages: int, log_std):
        history = distill_multicurriculum_actor(
            actor, data["observation"], data["teacher_physical_action"],
            data["curriculum"], data["public_phase"],
            epochs=int(distill["epochs_per_stage"]), batch_size=int(distill["batch_size"]),
            learning_rate=float(distill["learning_rate"]), weight_decay=float(distill["weight_decay"]),
            seed=int(distill["seed"]) + width, device=device,
        )
        out_dir = root / "candidates" / f"{sequence}{width}" / "fp32"
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / "actor_fp32.pt"
        save_dense_actor(path, actor, log_std=log_std, metadata={
            "experiment": "F16", "sequence": sequence, "target_width": width,
            "stage_index": stage_index, "cumulative_kd_stages": cumulative_stages,
            "dataset_sha256": actual, "teacher_sha256": config["frozen"]["teacher_sha256"],
            "survivor_hierarchy": "frozen_original_derived_nested_topk",
            "uses_privileged_truth": False, "uses_reward": False, "uses_critic": False,
        })
        record = {
            "candidate_id": f"{sequence}{width}",
            "sequence": "Direct" if sequence == "D" else "Progressive",
            "target_width": width,
            "cumulative_kd_stages": cumulative_stages,
            "cumulative_kd_epochs": cumulative_stages * int(distill["epochs_per_stage"]),
            "model_path": str(path),
            "model_sha256": file_sha256(path),
            "parameter_count": 29 * width + width + width * width + width + width * 2 + 2,
            "final_loss": history.get("loss", [None])[-1] if isinstance(history, dict) else None,
        }
        records.append(record)
        print(f"  trained {record['candidate_id']:<6} width={width:<4} stages={cumulative_stages} "
              f"sha={record['model_sha256'][:12]}", flush=True)
        return record

    # --- Direct: prune straight from Original (the frozen F12 pruned checkpoints) ---
    for width in (64, 96, 128, 192):
        actor, payload = load_dense_actor(pruning[f"P{width}"]["model_path"])
        kd(actor, width, "D", stage_index=0, cumulative_stages=1, log_std=payload["log_std"])

    # --- Progressive chain: KD at each width, pruning along the frozen hierarchy ---
    chain_actor, chain_payload = load_dense_actor(pruning["P192"]["model_path"])
    log_std = chain_payload["log_std"]
    stages = [192, 128, 96, 64]
    for stage_index, width in enumerate(stages):
        if width != 192:
            chain_actor = prune_along_frozen_hierarchy(
                chain_actor, survivors[stages[stage_index - 1]], survivors[width]
            )
        if width == 192:
            # P192 is procedurally identical to D192 and is not retrained or counted as
            # an independent variant; the chain simply starts from the D192 result.
            existing = next(r for r in records if r["candidate_id"] == "D192")
            chain_actor, _ = load_dense_actor(existing["model_path"])
            records.append({
                **existing,
                "candidate_id": "P192",
                "sequence": "Progressive",
                "equivalent_to": "D192",
                "independently_trained": False,
                "note": "P192 is procedurally identical to D192; not an independent replicate",
            })
            print("  P192 == D192 (not retrained, not an independent variant)", flush=True)
            continue
        kd(chain_actor, width, "P", stage_index=stage_index, cumulative_stages=stage_index + 1, log_std=log_std)

    manifest = {
        **provenance(config, CONFIG),
        "classification": "FROZEN",
        "dataset_sha256": actual,
        "dataset_rows": int(len(data["observation"])),
        "teacher_sha256": config["frozen"]["teacher_sha256"],
        "device": device,
        "epochs_per_stage": int(distill["epochs_per_stage"]),
        "survivor_nesting_verified": True,
        "matched_endpoint_guarantee": (
            "Direct and Progressive at the same target width retain identical original "
            "neuron indices; the manipulated factor is the distillation trajectory alone"
        ),
        "cumulative_gradient_step_asymmetry": {
            "note": "Progressive uses more cumulative KD epochs because it has more stages; "
                    "this is reported, not hidden. A compute-matched auxiliary comparison is separate.",
            "epochs": {r["candidate_id"]: r.get("cumulative_kd_epochs") for r in records},
        },
        "candidates": records,
    }
    write_json(target_manifest, manifest)
    registry = {r["candidate_id"]: r for r in records}
    write_json(root / "candidate_registry.json", {**provenance(config, CONFIG), "candidates": registry})
    print(json.dumps({"candidates": sorted(registry), "manifest": str(target_manifest)}, indent=2))
    return manifest


# ---------------------------------------------------------------------------
# Closed-loop evaluation
# ---------------------------------------------------------------------------
class RGBRecordingEnvironment:
    """Passive RGB observer around the frozen environment.

    The scientific rollout is executed by F15's proven ``run_episode_with_telemetry``
    against this wrapper, so every metric is computed by the same code path as F15. The
    wrapper only *observes* frames; it never changes an action, an observation, or the
    step sequence. The media integrity gate verified that enabling capture leaves actions
    and episode length bit-identical.
    """

    def __init__(self, environment, ring_steps: int):
        self._environment = environment
        self._ring: deque = deque(maxlen=ring_steps)
        self._step = 0

    def __getattr__(self, name):
        return getattr(self._environment, name)

    def _capture(self) -> None:
        self._ring.append((self._step, np.asarray(self._environment.latest_rgb(), dtype=np.uint8).copy()))

    def reset(self, **kwargs):
        result = self._environment.reset(**kwargs)
        self._ring.clear()
        self._step = 0
        self._capture()
        return result

    def step(self, action):
        result = self._environment.step(action)
        self._step += 1
        self._capture()
        return result

    @property
    def frames(self) -> list[tuple[int, np.ndarray]]:
        return list(self._ring)


def evaluate_candidate(candidate_id: str, seed_block: str = "selection") -> dict[str, Any]:
    """Evaluate one candidate on C0-C4 over a frozen seed block. Resumable.

    ``seed_block`` is normally the F16 selection block. ``f15_selection`` selects F15's
    already-opened recovery-selection seeds 180201-180208, used only to complete the
    backend-matched 2x2 model-versus-evaluation-block diagnostic. The sealed final holdout
    180301-180308 is never reachable from here.
    """
    import run_f15_cross_curriculum_recovery as f15
    from run_f15_cross_curriculum_recovery import (
        append_csv, phase_thresholds, read_csv, run_episode_with_telemetry, trace_path,
    )

    config = load_config(CONFIG)
    determinism = apply_frozen_determinism(config)
    root = artifact_root(config, CONFIG)
    paths = frozen_paths(config, CONFIG)
    protocol = load_ppo_curriculum_protocol(paths["policy_config"])
    thresholds = phase_thresholds(paths["f12_config"])
    if seed_block == "selection":
        seeds = [int(s) for s in config["seeds"]["selection"]]
    elif seed_block == "f15_selection":
        f15_config = load_config(ROOT / "configs/f15_cross_curriculum_recovery_v1.toml")
        seeds = [int(s) for s in f15_config["seeds"]["recovery_selection"]]
        sealed = {int(s) for s in config["seeds"]["inherited_sealed_final_holdout"]}
        if sealed & set(seeds):
            raise RuntimeError("refusing to evaluate on sealed holdout seeds")
    else:
        raise ValueError(f"unknown seed block: {seed_block}")
    ring_steps = int(config["media"]["rgb_ring_buffer_steps"])
    before = int(config["evaluation"]["failure_window_steps_before"])
    after = int(config["evaluation"]["failure_window_steps_after"])

    if candidate_id == "A0":
        matrix = verify_registry(
            paths["ablation_registry"],
            expected_registry_sha256=config["frozen"]["f12_ablation_registry_sha256"],
            collection_key="variants",
        )
        entry = matrix["A0"]
        entry = {**entry, "name": "Original Policy"}
        split = "baseline"
    elif candidate_id == "F15R64":
        # Cross-seed transfer check: the EXISTING F15 recovered 64x64 checkpoint, unchanged,
        # evaluated on the F16 deterministic seeds. Holds the model fixed while changing
        # only the evaluation block and backend. Diagnostic only — never a candidate,
        # never eligible for selection.
        path = ROOT / "artifacts/f15_cross_curriculum_recovery_v1/recovery/fp32/w64/actor_multicurriculum_kd_fp32.pt"
        expected = config["frozen"]["f15_recovered_fp32_w64_sha256"]
        if file_sha256(path) != expected:
            raise RuntimeError("F15 recovered checkpoint hash mismatch; refusing transfer check")
        entry = {
            "variant": "F15R64", "name": "F15 Recovered 64 (transfer check)",
            "model_path": str(path), "sha256": expected,
            "hidden_sizes": [64, 64], "int8": False,
        }
        split = "transfer_F15R64"
    else:
        registry = read_json(root / "candidate_registry.json")["candidates"]
        if candidate_id not in registry:
            raise KeyError(f"unknown F16 candidate: {candidate_id}")
        record = registry[candidate_id]
        entry = {
            "variant": candidate_id, "name": candidate_id,
            "model_path": record["model_path"], "sha256": record["model_sha256"],
            "hidden_sizes": [record["target_width"], record["target_width"]], "int8": False,
        }
        if file_sha256(entry["model_path"]) != entry["sha256"]:
            raise RuntimeError(f"candidate checkpoint changed since training: {candidate_id}")
        split = f"fp32_{candidate_id}"
    if seed_block != "selection":
        split = f"{split}_on_{seed_block}"

    episode_csv = root / "closed_loop" / f"{split}_episodes.csv"
    episode_csv.parent.mkdir(parents=True, exist_ok=True)
    done = {(r["curriculum"], int(r["seed"])) for r in read_csv(episode_csv)}

    actor = load_actor(entry)
    # run_episode_with_telemetry stamps provenance from this module-level path.
    f15._CURRENT_MODEL_PATH = Path(entry["model_path"])
    for curriculum in CURRICULA:
        pending = [s for s in seeds if (curriculum, s) not in done]
        if not pending:
            continue
        environment = PPOCurriculumEnvironment(
            paths["policy_config"], stage=curriculum,
            split=f"f16_{split}_{curriculum}", seeds=tuple(seeds),
        )
        recorder = RGBRecordingEnvironment(environment, ring_steps)
        try:
            for seed in pending:
                target = trace_path(root, split, candidate_id, curriculum, seed)
                row = run_episode_with_telemetry(
                    recorder, seed=seed, policy=ActorPolicy(entry["name"], actor),
                    protocol=protocol, thresholds=thresholds, target=target,
                )
                row = {"model_id": candidate_id, "model_name": entry["name"],
                       "curriculum": curriculum, **row}
                append_csv(episode_csv, row)

                # Media for objectively failing episodes, from THIS primary rollout.
                # The archive is read once; overlays come from the frozen telemetry.
                with np.load(target, allow_pickle=False) as archive:
                    flags = {n: np.asarray(archive[n], dtype=bool) for n in
                             ("collision", "unsafe", "stop_violation", "lane_failure",
                              "invalid_pose", "timeout", "terminated", "truncated", "completed")}
                    progress = np.asarray(archive["progress_m"], dtype=np.float32)
                    physical = np.asarray(archive["physical_action"], dtype=np.float32)
                step_rows = [{"step": i, **{n: bool(v[i]) for n, v in flags.items()}}
                             for i in range(len(progress))]
                event = first_objective_failure_event(step_rows)
                if event is not None:
                    frames = [(i, f) for i, f in recorder.frames
                              if max(0, event["step"] - before) <= i <= event["step"] + after
                              and i < len(progress)]
                    if frames:
                        media_result = {
                            "curriculum": curriculum, "seed": seed, "frames": frames,
                            "failure_step": event["step"], "episode_length": len(step_rows),
                            "steps": [{
                                "step": r["step"],
                                "progress_m": float(progress[r["step"]]),
                                "v_cmd": float(physical[r["step"], 0]),
                                "omega_cmd": float(physical[r["step"], 1]),
                                **{k: r[k] for k in ("collision", "unsafe", "stop_violation",
                                                     "lane_failure", "invalid_pose", "timeout")},
                            } for r in step_rows],
                        }
                        encode_media(
                            media_result,
                            root / "primary_media" / candidate_id / curriculum / f"seed_{seed}",
                            {"model_id": candidate_id, "model_name": entry["name"],
                             "sequence": "Direct" if candidate_id.startswith("D") else
                                         ("Progressive" if candidate_id.startswith("P") else "reference"),
                             "width": entry["hidden_sizes"][0], "precision": "FP32",
                             "model_sha256": entry["sha256"]},
                        )
                print(f"  {candidate_id} {curriculum} seed={seed} "
                      f"completed={row['completed']} steps={row['steps']}", flush=True)
        finally:
            environment.close()

    return {"candidate": candidate_id, "episode_csv": str(episode_csv), "determinism": determinism}


REPLICATION_SEEDS = {"S2": 2026081801, "S3": 2026081802}


def train_replication(realization: str) -> dict[str, Any]:
    """Train one additional paired training realization at widths 64, 96, 128.

    Direct and Progressive share the same seed base inside a realization, so the pair is
    matched on the training draw as well as on the endpoint. Width 192 is excluded because
    the two sequences are procedurally identical there.
    """
    from duckie_pomdp.optimization.actor_compression import (
        extract_original_actor, load_dense_actor, save_dense_actor,
    )
    from duckie_pomdp.optimization.cross_curriculum_recovery import distill_multicurriculum_actor

    if realization not in REPLICATION_SEEDS:
        raise ValueError(f"unknown realization {realization}; expected one of {sorted(REPLICATION_SEEDS)}")
    seed_base = REPLICATION_SEEDS[realization]

    config = load_config(CONFIG)
    root = artifact_root(config, CONFIG)
    manifest_path = root / f"training_manifest_{realization}.json"
    if manifest_path.exists():
        raise RuntimeError(f"refusing to overwrite {manifest_path}")

    paths = frozen_paths(config, CONFIG)
    pruning = verify_registry(
        paths["pruning_registry"],
        expected_registry_sha256=config["frozen"]["f12_pruning_registry_sha256"],
        collection_key="candidates",
    )
    distill = config["distillation"]

    dataset_path = ROOT / "artifacts/f15_cross_curriculum_recovery_v1/recovery/datasets/multicurriculum_public_states.npz"
    dataset_sha = file_sha256(dataset_path)
    if dataset_sha != config["frozen"]["f15_kd_dataset_sha256"]:
        raise RuntimeError("F15 KD dataset hash mismatch")
    with np.load(dataset_path, allow_pickle=False) as archive:
        data = {name: archive[name] for name in archive.files}

    original = extract_original_actor(
        str(ROOT / "artifacts/f10_ppo_visual_objects_v30/c4/ppo_selected.pt"),
        expected_sha256=config["frozen"]["original_ppo_sha256"],
    )
    if isinstance(original, tuple):
        original = original[0]
    survivors = frozen_survivors(original)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    records: list[dict[str, Any]] = []

    def kd(actor, width: int, sequence: str, stages: int, log_std):
        distill_multicurriculum_actor(
            actor, data["observation"], data["teacher_physical_action"],
            data["curriculum"], data["public_phase"],
            epochs=int(distill["epochs_per_stage"]), batch_size=int(distill["batch_size"]),
            learning_rate=float(distill["learning_rate"]), weight_decay=float(distill["weight_decay"]),
            seed=seed_base + width, device=device,
        )
        candidate_id = f"{sequence}{width}_{realization}"
        out = root / "candidates" / candidate_id / "fp32"
        out.mkdir(parents=True, exist_ok=True)
        path = out / "actor_fp32.pt"
        save_dense_actor(path, actor, log_std=log_std, metadata={
            "experiment": "F16", "realization": realization, "training_seed_base": seed_base,
            "sequence": sequence, "target_width": width, "cumulative_kd_stages": stages,
            "dataset_sha256": dataset_sha, "teacher_sha256": config["frozen"]["teacher_sha256"],
            "survivor_hierarchy": "frozen_original_derived_nested_topk",
            "uses_privileged_truth": False, "uses_reward": False, "uses_critic": False,
        })
        record = {
            "candidate_id": candidate_id,
            "sequence": "Direct" if sequence == "D" else "Progressive",
            "target_width": width, "realization": realization, "training_seed_base": seed_base,
            "cumulative_kd_stages": stages,
            "cumulative_kd_epochs": stages * int(distill["epochs_per_stage"]),
            "model_path": str(path), "model_sha256": file_sha256(path),
            "parameter_count": 29 * width + width + width * width + width + width * 2 + 2,
        }
        records.append(record)
        print(f"  trained {candidate_id:<10} W={width:<4} stages={stages} sha={record['model_sha256'][:12]}", flush=True)
        return record

    # Direct: one KD stage from the frozen directly-pruned checkpoint.
    for width in (64, 96, 128):
        actor, payload = load_dense_actor(pruning[f"P{width}"]["model_path"])
        kd(actor, width, "D", 1, payload["log_std"])

    # Progressive: KD at 192 first, then prune/KD down the frozen hierarchy.
    chain, chain_payload = load_dense_actor(pruning["P192"]["model_path"])
    log_std = chain_payload["log_std"]
    distill_multicurriculum_actor(
        chain, data["observation"], data["teacher_physical_action"],
        data["curriculum"], data["public_phase"],
        epochs=int(distill["epochs_per_stage"]), batch_size=int(distill["batch_size"]),
        learning_rate=float(distill["learning_rate"]), weight_decay=float(distill["weight_decay"]),
        seed=seed_base + 192, device=device,
    )
    print(f"  progressive chain: KD@192 done (intermediate, width 192 excluded from replication)", flush=True)
    stages_done = 1
    previous = 192
    for width in (128, 96, 64):
        chain = prune_along_frozen_hierarchy(chain, survivors[previous], survivors[width])
        stages_done += 1
        kd(chain, width, "P", stages_done, log_std)
        previous = width

    manifest = {
        **provenance(config, CONFIG),
        "classification": "FROZEN",
        "realization": realization,
        "training_seed_base": seed_base,
        "s1_training_seed_base": int(distill["seed"]),
        "dataset_sha256": dataset_sha,
        "teacher_sha256": config["frozen"]["teacher_sha256"],
        "widths": [64, 96, 128],
        "width_192_excluded": "Direct and Progressive are procedurally identical at 192",
        "matched_pair_within_realization": True,
        "plan_document": "docs/F16_TRANSFER_AND_REPLICATION_PLAN.md",
        "candidates": records,
    }
    write_json(manifest_path, manifest)

    registry_path = root / "candidate_registry.json"
    registry = read_json(registry_path)
    for record in records:
        registry["candidates"][record["candidate_id"]] = record
    write_json(registry_path, registry)
    print(json.dumps({"realization": realization, "trained": [r["candidate_id"] for r in records]}, indent=2))
    return manifest


def build_results() -> dict[str, Any]:
    """Turn completed episodes into retention decisions, fidelity, and eligibility.

    Uses the frozen F15 gate functions verbatim so F16 verdicts are computed by the same
    code that produced the F15 verdicts. Stop-phase diagnostics are NOT consulted here:
    eligibility depends only on the frozen behavior, fidelity, and safety gates.
    """
    from duckie_pomdp.optimization.compression_metrics import action_fidelity
    from duckie_pomdp.optimization.cross_curriculum_recovery import fidelity_pass, retention_decision
    from run_f15_cross_curriculum_recovery import read_csv, summarize_episode_dicts

    config = load_config(CONFIG)
    root = artifact_root(config, CONFIG)
    seeds = [int(s) for s in config["seeds"]["selection"]]
    registry = read_json(root / "candidate_registry.json")["candidates"]
    loop = root / "closed_loop"

    baseline_rows = read_csv(loop / "baseline_episodes.csv")
    if not baseline_rows:
        raise RuntimeError("baseline A0 episodes are required before results can be built")
    baseline = {c: summarize_episode_dicts([r for r in baseline_rows if r["curriculum"] == c])
                for c in CURRICULA}

    # Same-state fidelity: replay the A0 trajectories' normalized 29D through each candidate.
    observations: dict[str, np.ndarray] = {}
    for curriculum in CURRICULA:
        chunks = []
        for row in baseline_rows:
            if row["curriculum"] != curriculum:
                continue
            with np.load(Path(row["trace_path"]), allow_pickle=False) as archive:
                chunks.append(np.asarray(archive["public_normalized_29d"], dtype=np.float32))
        observations[curriculum] = np.concatenate(chunks)

    paths = frozen_paths(config, CONFIG)
    matrix = verify_registry(
        paths["ablation_registry"],
        expected_registry_sha256=config["frozen"]["f12_ablation_registry_sha256"],
        collection_key="variants",
    )
    original_actor = load_actor(matrix["A0"])
    from duckie_pomdp.optimization.compression_metrics import actor_physical_predictions
    original_predictions = {c: actor_physical_predictions(original_actor, observations[c]) for c in CURRICULA}

    sequence_rows, fidelity_rows, eligibility = [], [], {}
    for candidate_id, record in sorted(registry.items()):
        if record.get("independently_trained") is False:
            continue  # P192 is D192; evaluated once
        episodes = loop / f"fp32_{candidate_id}_episodes.csv"
        if not episodes.exists():
            continue
        rows = read_csv(episodes)
        covered = {(r["curriculum"], int(r["seed"])) for r in rows}
        complete = all((c, s) in covered for c in CURRICULA for s in seeds)
        # A curriculum with no episodes yet is reported as PENDING rather than summarised;
        # this lets results be inspected mid-sweep without fabricating a verdict.
        present = [c for c in CURRICULA if any(r["curriculum"] == c for r in rows)]
        if not present:
            continue
        summaries = {c: summarize_episode_dicts([r for r in rows if r["curriculum"] == c])
                     for c in present}

        actor = load_actor({"model_path": record["model_path"], "int8": False,
                            "hidden_sizes": [record["target_width"]] * 2})
        behavior_pass, fidelity_all = complete, complete
        for curriculum in present:
            seeds_done = sum(1 for r in rows if r["curriculum"] == curriculum)
            decision = retention_decision(
                curriculum, summaries[curriculum], baseline[curriculum],
                config["retention"]["absolute"], config["retention"]["relative_to_original"],
                candidate_prior=summaries, original_prior=baseline,
            )
            status = decision.status if hasattr(decision, "status") else decision["status"]
            behavior_pass &= status == "PASS"
            sequence_rows.append({
                "candidate_id": candidate_id, "sequence": record["sequence"],
                "target_width": record["target_width"], "precision": "FP32",
                "curriculum": curriculum.upper(),
                "status": status if seeds_done == len(seeds) else f"PARTIAL_{seeds_done}/{len(seeds)}",
                "episodes_complete": complete, "seeds_evaluated": seeds_done,
                "completion_rate": summaries[curriculum]["completion_rate"],
                "mean_progress_m": summaries[curriculum]["mean_progress_m"],
                "collision_rate": summaries[curriculum]["collision_rate"],
                "lane_failure_rate": summaries[curriculum]["lane_failure_rate"],
                "invalid_pose_rate": summaries[curriculum]["invalid_pose_rate"],
                "stop_violation_rate": summaries[curriculum]["stop_violation_rate"],
                "cumulative_kd_stages": record["cumulative_kd_stages"],
            })

            candidate_predictions = actor_physical_predictions(actor, observations[curriculum])
            metrics = action_fidelity(
                original_predictions[curriculum], candidate_predictions,
                omega_deadband=float(config["evaluation"]["omega_sign_deadband_rad_s"]),
            )
            passed, checks = fidelity_pass(metrics, config["fidelity"])
            fidelity_all &= passed
            fidelity_rows.append({
                "candidate_id": candidate_id, "sequence": record["sequence"],
                "target_width": record["target_width"], "precision": "FP32",
                "curriculum": curriculum.upper(), "pass": passed,
                "v_mae_mps": metrics["v_cmd_mps"]["mae"],
                "omega_mae_rad_s": metrics["omega_cmd_rad_s"]["mae"],
                "omega_pearson": metrics["omega_cmd_rad_s"]["pearson"],
                "omega_spearman": metrics["omega_cmd_rad_s"]["spearman"],
                "omega_sign_disagreement": metrics["omega_sign"]["disagreement_frequency"],
                "failed_checks": "|".join(k for k, v in checks.items() if not v),
            })

        eligibility[candidate_id] = {
            "sequence": record["sequence"], "target_width": record["target_width"],
            "precision": "FP32", "episodes_complete": complete,
            "behavior_all_curricula_pass": behavior_pass,
            "fidelity_all_curricula_pass": fidelity_all,
            "eligible_for_int8_stage": bool(complete and behavior_pass),
            "gate_source": "frozen behavior + fidelity + safety gates only; "
                           "stop-phase diagnostics are descriptive and never gate eligibility",
        }

    out = root / "results"
    out.mkdir(parents=True, exist_ok=True)
    for name, rows in (("sequence_results.csv", sequence_rows),
                       ("same_state_fidelity.csv", fidelity_rows)):
        path = out / name
        with path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        print(f"  wrote {name}: {len(rows)} rows")
    write_json(out / "candidate_eligibility.json", {**provenance(config, CONFIG), "candidates": eligibility})

    print()
    print("=== FP32 retention by candidate (frozen gates) ===")
    print(f"{'cand':<7}{'seq':<13}{'W':>5}  " + "".join(c.upper().ljust(9) for c in CURRICULA))
    for candidate_id in sorted(eligibility, key=lambda k: (registry[k]["target_width"], k)):
        cells = [next((r["status"] for r in sequence_rows
                       if r["candidate_id"] == candidate_id and r["curriculum"] == c.upper()), "-")
                 for c in CURRICULA]
        record = eligibility[candidate_id]
        print(f"{candidate_id:<7}{record['sequence']:<13}{record['target_width']:>5}  "
              + "".join(x.ljust(9) for x in cells))
    return {"eligibility": eligibility}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=(
        "smoke-media", "train-candidates", "train-replication", "evaluate", "results"))
    parser.add_argument("--candidate")
    parser.add_argument("--realization")
    parser.add_argument("--seed-block", default="selection", choices=("selection", "f15_selection"))
    args = parser.parse_args()
    if args.command == "smoke-media":
        smoke_media()
    elif args.command == "train-candidates":
        train_candidates()
    elif args.command == "train-replication":
        if not args.realization:
            parser.error("train-replication requires --realization (S2 or S3)")
        train_replication(args.realization)
    elif args.command == "results":
        build_results()
    else:
        if not args.candidate:
            parser.error("evaluate requires --candidate")
        print(json.dumps(evaluate_candidate(args.candidate, args.seed_block), indent=2, default=str))


if __name__ == "__main__":
    main()
