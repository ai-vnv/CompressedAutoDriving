"""F15 cross-curriculum retention, recovery, and provenance utilities.

This module deliberately contains no attribution code. It compares actor outputs,
closed-loop outcomes, and deployment efficiency while preserving the public 29D
boundary used by F10/F12.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from numpy.typing import NDArray
from torch import nn

from duckie_pomdp.optimization.actor_compression import physical_actions


CURRICULA = ("c0", "c1", "c2", "c3", "c4")
HUMAN_NAMES = {
    "A0": "Original Policy",
    "A1": "Pruning Only",
    "A2": "Pruning + Knowledge Distillation",
    "A3": "Post-Training Quantization (PTQ)",
    "A4": "Quantization-Aware Training + Distillation",
    "A5": "Pruning + PTQ",
    "A6": "Pruning + Distillation + PTQ",
    "A7": "Final INT8 Policy",
}


@dataclass(frozen=True)
class RetentionDecision:
    status: str
    original_absolute_pass: bool
    candidate_absolute_pass: bool
    relative_pass: bool
    absolute_checks: dict[str, bool]
    relative_checks: dict[str, bool]


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_json_sha256(value: Any) -> str:
    data = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


def verify_registry(
    registry_path: str | Path,
    *,
    expected_registry_sha256: str,
    collection_key: str,
) -> dict[str, dict[str, Any]]:
    path = Path(registry_path)
    if file_sha256(path) != expected_registry_sha256:
        raise RuntimeError(f"registry hash mismatch: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    entries = dict(payload[collection_key])
    for key, entry in entries.items():
        model = Path(entry["model_path"])
        if not model.exists() or file_sha256(model) != entry["sha256"]:
            raise RuntimeError(f"actor provenance mismatch: {key}")
        if int(entry["hidden_sizes"][0]) != int(entry["hidden_sizes"][1]):
            raise RuntimeError(f"F15 requires symmetric historical hidden widths: {key}")
    return entries


def validate_seed_partition(seed_mapping: Mapping[str, Sequence[int]], historical: Sequence[int]) -> None:
    names = tuple(seed_mapping)
    sets = {name: set(int(value) for value in seed_mapping[name]) for name in names}
    historical_set = set(int(value) for value in historical)
    for name, values in sets.items():
        if not values or len(values) != len(seed_mapping[name]):
            raise RuntimeError(f"seed block {name} is empty or contains duplicates")
        if values & historical_set:
            raise RuntimeError(f"seed block {name} intersects historical seeds")
    for index, name in enumerate(names):
        for other in names[index + 1 :]:
            if sets[name] & sets[other]:
                raise RuntimeError(f"F15 seed leakage: {name} intersects {other}")


def absolute_retention_checks(
    curriculum: str,
    summary: Mapping[str, Any],
    thresholds: Mapping[str, Any],
    *,
    prior_summaries: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, bool]:
    """Apply resolved F10 criteria without inventing unsupported metrics."""

    if curriculum not in CURRICULA:
        raise ValueError(f"unknown curriculum: {curriculum}")
    gate = thresholds[curriculum]
    prior = prior_summaries or {}
    checks: dict[str, bool] = {}
    direct = {
        "minimum_completion_rate": ("completion_rate", "minimum"),
        "maximum_lane_failure_rate": ("lane_failure_rate", "maximum"),
        "maximum_invalid_pose_rate": ("invalid_pose_rate", "maximum"),
        "maximum_mean_abs_lateral_error_m": ("mean_abs_lateral_error_m", "maximum"),
        "minimum_mean_progress_m": ("mean_progress_m", "minimum"),
        "maximum_collision_rate": ("collision_rate", "maximum"),
        "maximum_unsafe_episode_rate": ("unsafe_episode_rate", "maximum"),
        "maximum_stationary_fraction": ("stationary_fraction", "maximum"),
        "minimum_stop_completion_rate": ("stop_completion_rate", "minimum"),
        "maximum_stop_violation_rate": ("stop_violation_rate", "maximum"),
        "minimum_restart_rate": ("restart_rate", "minimum"),
    }
    for name, value in gate.items():
        if name in direct:
            metric, direction = direct[name]
            observed = float(summary[metric])
            checks[name] = observed >= float(value) if direction == "minimum" else observed <= float(value)
        elif name == "maximum_small_loop_completion_drop":
            checks[name] = "c0" in prior and (
                float(prior["c0"]["completion_rate"]) - float(summary["completion_rate"])
                <= float(value)
            )
        elif name == "maximum_c1_completion_drop":
            checks[name] = "c1" in prior and (
                float(prior["c1"]["completion_rate"]) - float(summary["completion_rate"])
                <= float(value)
            )
        elif name == "maximum_c2_collision_rate_increase":
            checks[name] = "c2" in prior and (
                float(summary["collision_rate"]) - float(prior["c2"]["collision_rate"])
                <= float(value)
            )
        else:
            raise KeyError(f"unsupported absolute retention criterion: {name}")
    return checks


def relative_behavior_checks(
    candidate: Mapping[str, Any], original: Mapping[str, Any], gate: Mapping[str, Any]
) -> dict[str, bool]:
    checks = {
        "new_collisions": int(candidate["collision_episodes"]) - int(original["collision_episodes"])
        <= int(gate["maximum_new_collisions"]),
        "collision_rate": float(candidate["collision_rate"]) - float(original["collision_rate"])
        <= float(gate["maximum_collision_rate_increase"]),
        "unsafe_episode_rate": float(candidate["unsafe_episode_rate"]) - float(original["unsafe_episode_rate"])
        <= float(gate["maximum_unsafe_episode_rate_increase"]),
        "stop_violation_rate": float(candidate["stop_violation_rate"]) - float(original["stop_violation_rate"])
        <= float(gate["maximum_stop_violation_rate_increase"]),
        "lane_failure_rate": float(candidate["lane_failure_rate"]) - float(original["lane_failure_rate"])
        <= float(gate["maximum_lane_failure_rate_increase"]),
        "invalid_pose_rate": float(candidate["invalid_pose_rate"]) - float(original["invalid_pose_rate"])
        <= float(gate["maximum_invalid_pose_rate_increase"]),
        "completion_rate": float(original["completion_rate"]) - float(candidate["completion_rate"])
        <= float(gate["maximum_completion_rate_drop"]),
        "restart_rate": float(original["restart_rate"]) - float(candidate["restart_rate"])
        <= float(gate["maximum_restart_rate_drop"]),
        "mean_progress_m": float(original["mean_progress_m"]) - float(candidate["mean_progress_m"])
        <= float(gate["maximum_mean_progress_drop_m"]),
    }
    original_clearance = original.get("minimum_pedestrian_clearance_m")
    candidate_clearance = candidate.get("minimum_pedestrian_clearance_m")
    checks["minimum_clearance"] = original_clearance is None or (
        candidate_clearance is not None
        and float(original_clearance) - float(candidate_clearance)
        <= float(gate["maximum_minimum_clearance_drop_m"])
    )
    return checks


def retention_decision(
    curriculum: str,
    candidate_summary: Mapping[str, Any],
    original_summary: Mapping[str, Any],
    absolute_thresholds: Mapping[str, Any],
    relative_gate: Mapping[str, Any],
    *,
    candidate_prior: Mapping[str, Mapping[str, Any]] | None = None,
    original_prior: Mapping[str, Mapping[str, Any]] | None = None,
) -> RetentionDecision:
    original_checks = absolute_retention_checks(
        curriculum, original_summary, absolute_thresholds, prior_summaries=original_prior
    )
    candidate_checks = absolute_retention_checks(
        curriculum, candidate_summary, absolute_thresholds, prior_summaries=candidate_prior
    )
    relative = relative_behavior_checks(candidate_summary, original_summary, relative_gate)
    original_pass = all(original_checks.values())
    candidate_pass = all(candidate_checks.values())
    relative_pass = all(relative.values())
    status = "UNRESOLVED" if not original_pass else ("PASS" if candidate_pass and relative_pass else "FAIL")
    return RetentionDecision(
        status=status,
        original_absolute_pass=original_pass,
        candidate_absolute_pass=candidate_pass,
        relative_pass=relative_pass,
        absolute_checks=candidate_checks,
        relative_checks=relative,
    )


def curriculum_balanced_probabilities(
    curricula: Sequence[str], phases: Sequence[str]
) -> NDArray[np.float64]:
    """Equal curriculum mass, then equal supported phase mass per curriculum."""

    curricula_array = np.asarray(curricula)
    phase_array = np.asarray(phases)
    if len(curricula_array) != len(phase_array) or len(curricula_array) == 0:
        raise ValueError("curriculum/phase labels must align and be non-empty")
    supported_curricula = tuple(sorted(set(str(value) for value in curricula_array)))
    if supported_curricula != CURRICULA:
        raise ValueError("F15 recovery dataset must support every C0-C4 curriculum")
    weights = np.zeros(len(curricula_array), dtype=np.float64)
    for curriculum in supported_curricula:
        curriculum_mask = curricula_array == curriculum
        supported_phases = tuple(sorted(set(str(value) for value in phase_array[curriculum_mask])))
        for phase in supported_phases:
            mask = curriculum_mask & (phase_array == phase)
            weights[mask] = 1.0 / (
                len(supported_curricula) * len(supported_phases) * int(mask.sum())
            )
    if not np.isclose(weights.sum(), 1.0):
        raise RuntimeError("curriculum-balanced weights do not sum to one")
    return weights


def distill_multicurriculum_actor(
    actor: nn.Module,
    observations: NDArray[np.float32],
    teacher_physical_actions: NDArray[np.float32],
    curricula: Sequence[str],
    phases: Sequence[str],
    *,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    seed: int,
    device: str,
) -> list[dict[str, float]]:
    matrix = np.asarray(observations, dtype=np.float32)
    targets = np.asarray(teacher_physical_actions, dtype=np.float32)
    if matrix.ndim != 2 or matrix.shape[1] != 29 or targets.shape != (len(matrix), 2):
        raise ValueError("invalid F15 public distillation dataset")
    if not np.isfinite(matrix).all() or not np.isfinite(targets).all():
        raise ValueError("F15 distillation dataset contains non-finite values")
    probabilities = curriculum_balanced_probabilities(curricula, phases)
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    actor.to(device).train()
    optimizer = torch.optim.Adam(actor.parameters(), lr=learning_rate, weight_decay=weight_decay)
    x = torch.as_tensor(matrix, dtype=torch.float32, device=device)
    y = torch.as_tensor(targets, dtype=torch.float32, device=device)
    scale = torch.as_tensor((0.4, 8.0), dtype=torch.float32, device=device)
    history: list[dict[str, float]] = []
    for epoch in range(epochs):
        indexes = rng.choice(len(matrix), size=len(matrix), replace=True, p=probabilities)
        losses: list[float] = []
        for start in range(0, len(indexes), batch_size):
            batch = torch.as_tensor(indexes[start : start + batch_size], device=device)
            prediction = physical_actions(actor(x[batch]))
            loss = torch.nn.functional.smooth_l1_loss(prediction / scale, y[batch] / scale)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        history.append({"epoch": float(epoch + 1), "loss": float(np.mean(losses))})
    actor.eval()
    return history


def first_objective_failure_event(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any] | None:
    """Return the earliest event and preserve all simultaneous labels."""

    for row in rows:
        labels = []
        for name in ("collision", "unsafe", "stop_violation", "lane_failure", "invalid_pose", "timeout"):
            if bool(row.get(name, False)):
                labels.append(name)
        terminal_without_completion = bool(row.get("terminated", False) or row.get("truncated", False)) and not bool(
            row.get("completed", False)
        )
        if terminal_without_completion:
            labels.append("termination_without_completion")
        if labels:
            return {"step": int(row["step"]), "event_labels": labels}
    return None


def fidelity_pass(metrics: Mapping[str, Any], gate: Mapping[str, Any]) -> tuple[bool, dict[str, bool]]:
    checks = {
        "v_mae": float(metrics["v_cmd_mps"]["mae"]) <= float(gate["maximum_v_mae_mps"]),
        "v_p95": float(metrics["v_cmd_mps"]["p95_absolute_error"]) <= float(gate["maximum_v_p95_abs_error_mps"]),
        "omega_mae": float(metrics["omega_cmd_rad_s"]["mae"]) <= float(gate["maximum_omega_mae_rad_s"]),
        "omega_p95": float(metrics["omega_cmd_rad_s"]["p95_absolute_error"]) <= float(gate["maximum_omega_p95_abs_error_rad_s"]),
        "omega_sign": float(metrics["omega_sign"]["disagreement_frequency"]) <= float(gate["maximum_omega_sign_disagreement"]),
        "pearson": min(float(metrics[name]["pearson"]) for name in ("v_cmd_mps", "omega_cmd_rad_s")) >= float(gate["minimum_pearson"]),
        "spearman": min(float(metrics[name]["spearman"]) for name in ("v_cmd_mps", "omega_cmd_rad_s")) >= float(gate["minimum_spearman"]),
    }
    return all(checks.values()), checks
