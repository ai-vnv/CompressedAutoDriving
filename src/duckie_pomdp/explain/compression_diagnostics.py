"""Shared F14 utilities for immutable compression-ablation diagnostics."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from numpy.typing import NDArray

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - project Python 3.10
    import tomli as tomllib

from duckie_pomdp.control.ppo_protocol import (
    PPOCurriculumProtocol,
    load_ppo_curriculum_protocol,
)
from duckie_pomdp.explain.development_protocol import apply_semantic_intervention
from duckie_pomdp.explain.group_shapley import GROUP_ORDER, validate_group_partition
from duckie_pomdp.optimization.actor_compression import load_dense_actor


ACTION_NAMES = ("v_cmd_mps", "omega_cmd_rad_s")
ACTION_RANGES = np.asarray((0.4, 8.0), dtype=np.float64)
PRIMARY_COUNTERFACTUAL_CELLS = (
    ("pedestrian_absent", "pedestrian_relevant", "v_cmd_mps"),
    ("stop_absent", "stop_required", "v_cmd_mps"),
    ("lane_centered", "lane_curve", "omega_cmd_rad_s"),
)


@dataclass(frozen=True)
class FrozenActor:
    variant: str
    name: str
    precision: str
    path: Path
    sha256: str
    architecture: tuple[int, ...]
    module: torch.nn.Module

    def physical(self, observations: NDArray[np.floating]) -> NDArray[np.float32]:
        matrix = np.asarray(observations, dtype=np.float32)
        if matrix.ndim != 2 or matrix.shape[1] != 29 or not np.isfinite(matrix).all():
            raise ValueError("actor observations must be finite [N,29]")
        chunks: list[NDArray[np.float32]] = []
        with torch.inference_mode():
            for start in range(0, len(matrix), 4096):
                raw = self.module(torch.from_numpy(matrix[start : start + 4096]))
                chunks.append(raw.detach().cpu().numpy().astype(np.float32))
        return normalized_to_physical(np.concatenate(chunks, axis=0))


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_json_sha256(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def unresolved_evidence(reason: str) -> dict[str, Any]:
    """Encode unavailable evidence explicitly; absence is never numerical zero."""

    if not str(reason).strip():
        raise ValueError("UNRESOLVED evidence requires a reason")
    return {"classification": "UNRESOLVED", "value": None, "reason": str(reason)}


def load_f14_config(path: str | Path) -> dict[str, Any]:
    target = Path(path).resolve()
    with target.open("rb") as stream:
        raw = tomllib.load(stream)
    raw["_path"] = str(target)
    raw["_sha256"] = file_sha256(target)
    return raw


def resolve_config_path(config: Mapping[str, Any], value: str) -> Path:
    return (Path(str(config["_path"])).parent / value).resolve()


def verify_frozen_file(config: Mapping[str, Any], path_key: str, hash_key: str) -> Path:
    path = resolve_config_path(config, str(config["frozen"][path_key]))
    expected = str(config["frozen"][hash_key])
    actual = file_sha256(path)
    if actual != expected:
        raise RuntimeError(f"frozen provenance mismatch: {path} {actual} != {expected}")
    return path


def load_policy_contract(
    config: Mapping[str, Any],
) -> tuple[PPOCurriculumProtocol, tuple[str, ...], dict[str, tuple[int, ...]]]:
    policy_path = verify_frozen_file(config, "policy_config", "policy_config_sha256")
    protocol = load_ppo_curriculum_protocol(policy_path)
    feature_names = tuple(str(value) for value in protocol.observation_order)
    groups = {
        str(name): tuple(str(field) for field in fields)
        for name, fields in config["groups"].items()
    }
    indexes = validate_group_partition(feature_names, groups)
    if len(feature_names) != int(config["frozen"]["observation_dimension"]):
        raise RuntimeError("frozen observation dimension mismatch")
    return protocol, feature_names, indexes


def load_frozen_actors(config: Mapping[str, Any]) -> dict[str, FrozenActor]:
    registry_path = verify_frozen_file(config, "actor_registry", "actor_registry_sha256")
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    actors: dict[str, FrozenActor] = {}
    if tuple(sorted(registry["variants"])) != tuple(f"A{i}" for i in range(8)):
        raise RuntimeError("F12 registry does not contain exact A0-A7 set")
    for variant in (f"A{i}" for i in range(8)):
        frozen = config["actors"][variant]
        registered = registry["variants"][variant]
        path = resolve_config_path(config, str(frozen["path"]))
        expected = str(frozen["sha256"])
        if file_sha256(path) != expected or str(registered["sha256"]) != expected:
            raise RuntimeError(f"actor hash mismatch: {variant}")
        if Path(str(registered["model_path"])).resolve() != path:
            raise RuntimeError(f"actor registry path mismatch: {variant}")
        precision = str(frozen["precision"])
        if precision == "INT8":
            module = torch.jit.load(str(path), map_location="cpu").eval()
            if str(module.inlined_graph).count("quantized::linear") < 3:
                raise RuntimeError(f"{variant} lacks deployed INT8 Linear kernels")
        elif precision == "FP32":
            module = load_dense_actor(path)[0].cpu().eval()
        else:
            raise RuntimeError(f"unsupported actor precision: {precision}")
        architecture = tuple(int(value) for value in frozen["architecture"])
        registered_architecture = (29, *tuple(registered["hidden_sizes"]), 2)
        if architecture != registered_architecture:
            raise RuntimeError(f"actor architecture mismatch: {variant}")
        actors[variant] = FrozenActor(
            variant=variant,
            name=str(frozen["name"]),
            precision=precision,
            path=path,
            sha256=expected,
            architecture=architecture,
            module=module,
        )
    return actors


def normalized_to_physical(action: NDArray[np.floating]) -> NDArray[np.float32]:
    raw = np.asarray(action, dtype=np.float32)
    result = np.empty_like(raw)
    result[..., 0] = (np.clip(raw[..., 0], -1.0, 1.0) + 1.0) * 0.2
    result[..., 1] = np.clip(raw[..., 1], -1.0, 1.0) * 4.0
    return result


def select_stratified_public_rows(
    phases: Sequence[str],
    seeds: NDArray[np.integer],
    steps: NDArray[np.integer],
    *,
    phase_order: Sequence[str],
    states_per_phase: int,
) -> NDArray[np.int64]:
    """Public-only, seed-balanced, evenly spaced deterministic state selection."""

    phase_values = np.asarray(phases)
    seed_values = np.asarray(seeds, dtype=np.int64)
    step_values = np.asarray(steps, dtype=np.int64)
    selected: list[int] = []
    for phase in phase_order:
        phase_seeds = sorted(int(value) for value in np.unique(seed_values[phase_values == phase]))
        if not phase_seeds:
            raise ValueError(f"no support for phase {phase}")
        base, remainder = divmod(int(states_per_phase), len(phase_seeds))
        for seed_offset, seed in enumerate(phase_seeds):
            quota = base + int(seed_offset < remainder)
            pool = np.flatnonzero((phase_values == phase) & (seed_values == seed))
            pool = pool[np.argsort(step_values[pool], kind="stable")]
            if len(pool) < quota:
                raise ValueError(f"insufficient support for phase={phase}, seed={seed}")
            positions = np.rint(np.linspace(0, len(pool) - 1, quota)).astype(np.int64)
            if len(np.unique(positions)) != quota:
                raise RuntimeError("deterministic selector produced duplicate positions")
            selected.extend(int(value) for value in pool[positions])
    indexes = np.asarray(selected, dtype=np.int64)
    if len(indexes) != len(phase_order) * int(states_per_phase):
        raise RuntimeError("diagnostic state selection count mismatch")
    return indexes


def assign_complete_references(
    observations: NDArray[np.float32],
    phases: Sequence[str],
    seeds: NDArray[np.integer],
    factual_indexes: NDArray[np.integer],
    *,
    draw_seeds: Sequence[int],
    references_per_draw: int,
) -> tuple[NDArray[np.float32], NDArray[np.int64]]:
    """Same-phase references, four distinct cross-seed trajectories per draw."""

    matrix = np.asarray(observations, dtype=np.float32)
    phase_values = np.asarray(phases)
    seed_values = np.asarray(seeds, dtype=np.int64)
    factual = np.asarray(factual_indexes, dtype=np.int64)
    index = np.empty((len(draw_seeds), references_per_draw, len(factual)), dtype=np.int64)
    for draw, draw_seed in enumerate(draw_seeds):
        rng = np.random.default_rng(int(draw_seed))
        for column, factual_index in enumerate(factual):
            phase = phase_values[factual_index]
            factual_seed = int(seed_values[factual_index])
            eligible_seeds = np.asarray(
                sorted(
                    int(value)
                    for value in np.unique(seed_values[phase_values == phase])
                    if int(value) != factual_seed
                ),
                dtype=np.int64,
            )
            if len(eligible_seeds) < references_per_draw:
                raise ValueError(f"insufficient cross-seed reference support for {phase}")
            chosen_seeds = rng.choice(
                eligible_seeds, size=references_per_draw, replace=False
            )
            for ref, reference_seed in enumerate(chosen_seeds):
                pool = np.flatnonzero(
                    (phase_values == phase) & (seed_values == reference_seed)
                )
                index[draw, ref, column] = int(rng.choice(pool))
    references = matrix[index]
    return np.asarray(references, dtype=np.float32), index


def summarize_group_attribution(
    attribution: NDArray[np.floating], phases: Sequence[str]
) -> list[dict[str, Any]]:
    """Aggregate state-level signed Shapley into phase/action/group statistics."""

    values = np.asarray(attribution, dtype=np.float64)
    if values.ndim != 3 or values.shape[1:] != (2, 6):
        raise ValueError("attribution must have shape [state,2,6]")
    phase_values = np.asarray(phases)
    rows: list[dict[str, Any]] = []
    for phase in ("overall", *tuple(dict.fromkeys(str(v) for v in phase_values))):
        mask = np.ones(len(values), dtype=bool) if phase == "overall" else phase_values == phase
        absolute = np.mean(np.abs(values[mask]), axis=0)
        signed = np.mean(values[mask], axis=0)
        share = absolute / np.maximum(absolute.sum(axis=1, keepdims=True), 1.0e-12)
        for action_index, action in enumerate(ACTION_NAMES):
            order = np.argsort(-share[action_index], kind="stable")
            ranks = np.empty(6, dtype=np.int64)
            ranks[order] = np.arange(1, 7)
            for group_index, group in enumerate(GROUP_ORDER):
                rows.append(
                    {
                        "phase": phase,
                        "action": action,
                        "group": group,
                        "states": int(np.sum(mask)),
                        "signed_mean": float(signed[action_index, group_index]),
                        "absolute_mean": float(absolute[action_index, group_index]),
                        "absolute_share": float(share[action_index, group_index]),
                        "rank": int(ranks[group_index]),
                        "top_group": str(GROUP_ORDER[int(order[0])]),
                        "top_two": "|".join(GROUP_ORDER[int(value)] for value in order[:2]),
                    }
                )
    return rows


def compare_group_summaries(
    reference_rows: Sequence[Mapping[str, Any]],
    candidate_rows: Sequence[Mapping[str, Any]],
    *,
    signed_deadband: float,
) -> list[dict[str, Any]]:
    left = {(row["phase"], row["action"], row["group"]): row for row in reference_rows}
    right = {(row["phase"], row["action"], row["group"]): row for row in candidate_rows}
    output: list[dict[str, Any]] = []
    for phase, action in sorted({(key[0], key[1]) for key in left}):
        a = np.asarray([left[(phase, action, g)]["absolute_share"] for g in GROUP_ORDER])
        b = np.asarray([right[(phase, action, g)]["absolute_share"] for g in GROUP_ORDER])
        signed_a = np.asarray([left[(phase, action, g)]["signed_mean"] for g in GROUP_ORDER])
        signed_b = np.asarray([right[(phase, action, g)]["signed_mean"] for g in GROUP_ORDER])
        eligible = (np.abs(signed_a) > signed_deadband) | (np.abs(signed_b) > signed_deadband)
        top_a = int(np.argmax(a)); top_b = int(np.argmax(b))
        top2_a = set(np.argsort(-a, kind="stable")[:2].tolist())
        top2_b = set(np.argsort(-b, kind="stable")[:2].tolist())
        output.append(
            {
                "phase": phase,
                "action": action,
                "group_spearman": spearman(a, b),
                "group_share_l1": float(np.sum(np.abs(a - b))),
                "top_group_reference": GROUP_ORDER[top_a],
                "top_group_candidate": GROUP_ORDER[top_b],
                "top_group_agreement": bool(top_a == top_b),
                "top_two_jaccard": float(len(top2_a & top2_b) / len(top2_a | top2_b)),
                "signed_direction_agreement": float(
                    np.mean(np.sign(signed_a[eligible]) == np.sign(signed_b[eligible]))
                    if np.any(eligible) else 1.0
                ),
            }
        )
    return output


def semantic_structure_classification(
    comparison: Sequence[Mapping[str, Any]], thresholds: Mapping[str, float]
) -> dict[str, Any]:
    cells = [row for row in comparison if row["phase"] != "overall"]
    for row in cells:
        row["preserved"] = bool(
            row["group_spearman"] >= thresholds["minimum_group_spearman"]
            and row["group_share_l1"] <= thresholds["maximum_group_share_l1"]
            and row["top_group_agreement"]
            and row["top_two_jaccard"] >= thresholds["minimum_top_two_jaccard"]
        )
    passed = sum(bool(row["preserved"]) for row in cells)
    minimum = int(thresholds.get("minimum_preserved_phase_action_cells", 8))
    classification = (
        "PRESERVED"
        if passed >= minimum
        else "PARTIAL"
        if passed >= max(1, minimum - 3)
        else "SHIFTED"
    )
    return {
        "classification": classification,
        "preserved_phase_action_cells": passed,
        "total_phase_action_cells": len(cells),
        "cells": cells,
    }


def evaluate_semantic_counterfactuals(
    actor: FrozenActor,
    normalized: NDArray[np.float32],
    physical: NDArray[np.float32],
    protocol: PPOCurriculumProtocol,
    interventions: Sequence[str],
    *,
    lane_low_confidence_validity: float,
    lane_low_confidence_min_lateral_std_m: float,
    lane_low_confidence_min_heading_std_rad: float,
    lane_low_confidence_min_curvature_std_inv_m: float,
) -> tuple[NDArray[np.float32], NDArray[np.float32], dict[str, tuple[str, ...]]]:
    factual = actor.physical(normalized)
    effects = np.empty((len(interventions), len(normalized), 2), dtype=np.float32)
    intervened_actions = np.empty_like(effects)
    intended: dict[str, tuple[str, ...]] = {}
    for intervention_index, intervention in enumerate(interventions):
        changed = np.empty_like(normalized)
        fields: tuple[str, ...] | None = None
        for row in range(len(normalized)):
            if intervention == "sham":
                changed[row] = normalized[row]
                current_fields = ()
            else:
                changed[row], current_fields = apply_semantic_intervention(
                    physical[row], intervention, protocol,
                    lane_low_confidence_validity=lane_low_confidence_validity,
                    lane_low_confidence_min_lateral_std_m=lane_low_confidence_min_lateral_std_m,
                    lane_low_confidence_min_heading_std_rad=lane_low_confidence_min_heading_std_rad,
                    lane_low_confidence_min_curvature_std_inv_m=lane_low_confidence_min_curvature_std_inv_m,
                )
            if fields is None:
                fields = current_fields
            elif fields != current_fields:
                raise RuntimeError("intervention field mask changed across rows")
        intervened = actor.physical(changed)
        intervened_actions[intervention_index] = intervened
        effects[intervention_index] = intervened - factual
        intended[intervention] = fields or ()
    return factual, effects, intended


def counterfactual_comparison(
    reference_effects: NDArray[np.floating],
    candidate_effects: NDArray[np.floating],
    phases: Sequence[str],
    interventions: Sequence[str],
    *,
    direction_deadband: float,
) -> list[dict[str, Any]]:
    left = np.asarray(reference_effects, dtype=np.float64)
    right = np.asarray(candidate_effects, dtype=np.float64)
    phase_values = np.asarray(phases)
    rows: list[dict[str, Any]] = []
    for intervention_index, intervention in enumerate(interventions):
        for phase in tuple(dict.fromkeys(str(value) for value in phase_values)):
            mask = phase_values == phase
            for action_index, action in enumerate(ACTION_NAMES):
                a = left[intervention_index, mask, action_index]
                b = right[intervention_index, mask, action_index]
                eligible = np.abs(a) > direction_deadband
                drift = b - a
                rows.append(
                    {
                        "intervention": intervention,
                        "phase": phase,
                        "action": action,
                        "states": int(np.sum(mask)),
                        "reference_mean": float(np.mean(a)),
                        "candidate_mean": float(np.mean(b)),
                        "reference_mean_absolute": float(np.mean(np.abs(a))),
                        "candidate_mean_absolute": float(np.mean(np.abs(b))),
                        "paired_direction_agreement": float(
                            np.mean(np.sign(a[eligible]) == np.sign(b[eligible]))
                            if np.any(eligible) else 1.0
                        ),
                        "direction_eligible_states": int(np.sum(eligible)),
                        "normalized_mean_effect_drift": float(abs(np.mean(b) - np.mean(a)) / ACTION_RANGES[action_index]),
                        "normalized_mean_absolute_effect_drift": float(abs(np.mean(np.abs(b)) - np.mean(np.abs(a))) / ACTION_RANGES[action_index]),
                        "normalized_p95_effect_drift": float(np.quantile(np.abs(drift), 0.95) / ACTION_RANGES[action_index]),
                        "maximum_absolute_drift": float(np.max(np.abs(drift))),
                    }
                )
    return rows


def counterfactual_preservation_classification(
    rows: Sequence[Mapping[str, Any]], thresholds: Mapping[str, float]
) -> dict[str, Any]:
    """Classify only the three preregistered primary cells; sham is a gate."""

    primary = [
        row for row in rows
        if (row["intervention"], row["phase"], row["action"])
        in PRIMARY_COUNTERFACTUAL_CELLS
    ]
    if len(primary) != 3:
        raise ValueError("counterfactual comparison lacks the three frozen primary cells")
    sham = [row for row in rows if row["intervention"] == "sham"]
    sham_valid = bool(sham) and all(
        row["reference_mean_absolute"] <= thresholds["sham_absolute_tolerance"]
        and row["candidate_mean_absolute"] <= thresholds["sham_absolute_tolerance"]
        for row in sham
    )
    preserved = [
        row for row in primary
        if row["paired_direction_agreement"] >= thresholds["minimum_direction_agreement"]
        and row["normalized_mean_effect_drift"] <= thresholds["maximum_normalized_mean_effect_drift"]
        and row["normalized_p95_effect_drift"] <= thresholds["maximum_normalized_p95_effect_drift"]
    ]
    if not sham_valid:
        classification = "INVALID"
    elif len(preserved) == 3:
        classification = "PRESERVED"
    elif len(preserved) == 2:
        classification = "PARTIAL"
    else:
        classification = "SHIFTED"
    return {
        "classification": classification,
        "preserved_primary_cells": len(preserved),
        "total_primary_cells": 3,
        "sham_gate": "PASS" if sham_valid else "FAILED",
        "primary_cells": primary,
        "cells": list(rows),
    }


def spearman(left: Sequence[float], right: Sequence[float]) -> float:
    a = _average_rank(np.asarray(left, dtype=np.float64))
    b = _average_rank(np.asarray(right, dtype=np.float64))
    if np.std(a) <= 1.0e-12 or np.std(b) <= 1.0e-12:
        return 1.0 if np.allclose(a, b) else 0.0
    return float(np.corrcoef(a, b)[0, 1])


def _average_rank(values: NDArray[np.float64]) -> NDArray[np.float64]:
    order = np.argsort(values, kind="stable")
    ranks = np.empty(len(values), dtype=np.float64)
    ranks[order] = np.arange(len(values), dtype=np.float64)
    unique, inverse, counts = np.unique(values, return_inverse=True, return_counts=True)
    del unique
    for group, count in enumerate(counts):
        if count > 1:
            ranks[inverse == group] = np.mean(ranks[inverse == group])
    return ranks
