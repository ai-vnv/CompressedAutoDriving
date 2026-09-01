#!/usr/bin/env python3
"""Run fixed, paired C4 diagnostics for Original and deployed A7 actors."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib

from duckie_pomdp.control.ppo_environment import PPOCurriculumEnvironment
from duckie_pomdp.control.ppo_protocol import load_ppo_curriculum_protocol
from duckie_pomdp.evaluation.f10_ppo_policy import run_episode, summarize_episodes
from duckie_pomdp.explain.compressed_policy_analysis import (
    file_sha256,
    require_quantized_linear_graph,
    verify_hash,
)
from duckie_pomdp.explain.development_protocol import PhaseThresholds, public_phase
from duckie_pomdp.optimization.actor_compression import extract_original_actor


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs/f13_explain_compressed_v1.toml"


class TracedActorPolicy:
    """Actor adapter that logs only public observation and actor outputs."""

    def __init__(self, name: str, actor: torch.nn.Module, protocol: Any, thresholds: PhaseThresholds) -> None:
        self.name = name
        self.actor = actor.cpu().eval()
        self._order = tuple(protocol.observation_order)
        self._scales = np.asarray(protocol.observation_scales, dtype=np.float64)
        self._thresholds = thresholds
        self.trace: list[dict[str, Any]] = []

    def reset(self, seed: int) -> None:
        del seed
        self.trace = []

    def act(self, observation: np.ndarray) -> np.ndarray:
        normalized = np.asarray(observation, dtype=np.float32)
        with torch.inference_mode():
            action = self.actor(torch.as_tensor(normalized).unsqueeze(0)).squeeze(0)
        value = np.clip(action.cpu().numpy(), -1.0, 1.0).astype(np.float32)
        physical = normalized.astype(np.float64) * self._scales
        phase = public_phase(physical, self._order, self._thresholds)
        self.trace.append({
            "step": len(self.trace),
            "public_phase": phase,
            "normalized_v_action": float(value[0]),
            "normalized_omega_action": float(value[1]),
        })
        return value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--split", choices=("exploratory", "confirmatory"), required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.config.resolve(), args.split), indent=2))


def run(config_path: Path, split: str) -> dict[str, Any]:
    config = load_config(config_path)
    root = resolve(config_path, config["artifacts"]["directory"])
    cf = read_json(root / "counterfactual/counterfactual_metrics.json")
    if cf["classification"] not in ("PRESERVED", "PARTIALLY PRESERVED", "SHIFTED"):
        raise RuntimeError("counterfactual analysis is not complete")
    stress_root = root / "failure_modes"
    stress_root.mkdir(parents=True, exist_ok=True)
    plan_path = stress_root / "probe_plan.json"
    if not plan_path.exists():
        plan = make_plan(config, cf)
        write_json(plan_path, plan)
    else:
        plan = read_json(plan_path)
    if split == "confirmatory" and not plan.get("confirmatory_authorized", False):
        target = stress_root / "confirmatory_not_run.json"
        if not target.exists():
            write_json(target, {
                "classification": "NOT_RUN",
                "reason": "no repeated Original-pass/A7-fail candidate in exploratory diagnostics",
                "seeds_opened": False,
            })
        return read_json(target)

    original_path = resolve(config_path, config["frozen"]["original"]["checkpoint"])
    a7_path = resolve(config_path, config["frozen"]["a7"]["checkpoint"])
    verify_hash(original_path, config["frozen"]["original"]["sha256"])
    verify_hash(a7_path, config["frozen"]["a7"]["sha256"])
    original, _, _ = extract_original_actor(
        original_path, expected_sha256=config["frozen"]["original"]["sha256"]
    )
    a7 = torch.jit.load(str(a7_path), map_location="cpu").eval()
    require_quantized_linear_graph(a7)
    policy_config = resolve(config_path, config["frozen"]["contract"]["policy_config"])
    protocol = load_ppo_curriculum_protocol(policy_config)
    thresholds = load_thresholds(config_path, config)
    seeds = tuple(int(seed) for seed in config["stress"][f"{split}_seeds"])
    output = stress_root / split
    output.mkdir(parents=True, exist_ok=True)
    csv_path = output / "paired_episodes.csv"
    existing = read_csv(csv_path)
    completed = {(row["policy"], int(row["seed"])) for row in existing}
    rows = list(existing)
    actors = (("Original", original), ("A7", a7))
    for name, actor in actors:
        policy = TracedActorPolicy(name, actor, protocol, thresholds)
        environment = PPOCurriculumEnvironment(
            policy_config,
            stage="c4",
            split=f"f13_{split}",
            seeds=seeds,
        )
        try:
            for seed in seeds:
                if (name, seed) in completed:
                    continue
                episode = run_episode(environment, seed=seed, policy=policy, protocol=protocol)
                row = {"policy": name, **asdict(episode)}
                row.update(action_summary(policy.trace, float(config["stress"]["action_saturation_threshold"])))
                append_csv(csv_path, row)
                rows.append({key: str(value) if value is None else value for key, value in row.items()})
                print(f"F13_STRESS {split} {name} seed={seed} completed={episode.completed}", flush=True)
        finally:
            environment.close()

    typed = [coerce_row(row) for row in rows]
    expected = 2 * len(seeds)
    if len(typed) != expected:
        raise RuntimeError(f"expected {expected} paired rows, found {len(typed)}")
    summary = summarize(split, typed, seeds, config)
    write_json(output / "summary.json", summary)
    if split == "exploratory":
        plan["confirmatory_authorized"] = summary["differential_failure_count"] > 0
        plan["confirmatory_reason"] = (
            "exploratory Original-pass/A7-fail candidate requires independent confirmation"
            if plan["confirmatory_authorized"]
            else "no exploratory closed-loop compression-related failure candidate"
        )
        write_json(plan_path, plan)
        if not plan["confirmatory_authorized"]:
            write_json(stress_root / "confirmatory_not_run.json", {
                "classification": "NOT_RUN",
                "reason": plan["confirmatory_reason"],
                "seeds_opened": False,
            })
    return summary


def make_plan(config: dict[str, Any], cf: dict[str, Any]) -> dict[str, Any]:
    failed = sorted(key for key, passed in cf["primary_checks"].items() if not passed)
    return {
        "schema_version": 1,
        "created_before_simulator_access": True,
        "fixed_sentinels": list(config["stress"]["sentinels"]),
        "functional_drift_triggers": failed,
        "targeted_focus": ["stop/restart"] if any("stop_absent" in key for key in failed) else [],
        "exploratory_seeds": list(config["stress"]["exploratory_seeds"]),
        "confirmatory_seeds_reserved": list(config["stress"]["confirmatory_seeds"]),
        "same_seed_pairing": True,
        "scenario_parameter_search": False,
        "confirmatory_authorized": False,
    }


def action_summary(trace: list[dict[str, Any]], threshold: float) -> dict[str, Any]:
    phases: dict[str, Any] = {}
    for phase in sorted({row["public_phase"] for row in trace}):
        selected = [row for row in trace if row["public_phase"] == phase]
        v = np.asarray([row["normalized_v_action"] for row in selected])
        omega = np.asarray([row["normalized_omega_action"] for row in selected])
        phases[phase] = {
            "steps": len(selected),
            "saturation_rate": float(np.mean((np.abs(v) >= threshold) | (np.abs(omega) >= threshold))),
            "v_mean": float(np.mean(v)),
            "omega_abs_mean": float(np.mean(np.abs(omega))),
        }
    return {"action_trace_steps": len(trace), "phase_action_json": json.dumps(phases, sort_keys=True)}


def summarize(split: str, rows: list[dict[str, Any]], seeds: tuple[int, ...], config: dict[str, Any]) -> dict[str, Any]:
    by_policy = {}
    for policy in ("Original", "A7"):
        selected = [row for row in rows if row["policy"] == policy]
        episodes = [row["episode"] for row in selected]
        by_policy[policy] = summarize_episodes(episodes)
        by_policy[policy]["phase_action"] = aggregate_phase_action(selected)
    differential: list[dict[str, Any]] = []
    both_fail: list[int] = []
    for seed in seeds:
        original = next(row for row in rows if row["policy"] == "Original" and row["seed"] == seed)["episode"]
        a7 = next(row for row in rows if row["policy"] == "A7" and row["seed"] == seed)["episode"]
        original_failure = failure_reasons(original)
        a7_failure = failure_reasons(a7)
        if not original_failure and a7_failure:
            differential.append({"seed": seed, "a7_failures": a7_failure})
        if original_failure and a7_failure:
            both_fail.append(seed)
    ped_o = by_policy["Original"]["phase_action"].get("pedestrian_relevant", {})
    ped_a = by_policy["A7"]["phase_action"].get("pedestrian_relevant", {})
    saturation_increase = float(ped_a.get("saturation_rate", 0.0) - ped_o.get("saturation_rate", 0.0))
    level3 = saturation_increase > float(config["stress"]["trigger_saturation_rate_increase"])
    minimum = int(config["classification"]["behavioral_new_failure_minimum_count"])
    level4 = len(differential) >= minimum
    return {
        "schema_version": 1,
        "split": split,
        "seeds": list(seeds),
        "pairing_verified": True,
        "results": by_policy,
        "pedestrian_saturation_rate_increase": saturation_increase,
        "level3_action_drift": level3,
        "differential_failure_count": len(differential),
        "differential_failures": differential,
        "both_fail_seeds": both_fail,
        "level4_closed_loop_failure": level4,
        "behavioral_classification": "DEGRADED" if level4 else "PRESERVED",
        "privileged_truth_used_for_action": False,
        "models_modified": False,
    }


def aggregate_phase_action(rows: list[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    decoded = [json.loads(str(row["phase_action_json"])) for row in rows]
    phases = {phase for payload in decoded for phase in payload}
    for phase in sorted(phases):
        count = sum(int(payload.get(phase, {}).get("steps", 0)) for payload in decoded)
        if not count:
            continue
        result[phase] = {
            "steps": count,
            "saturation_rate": sum(float(payload.get(phase, {}).get("saturation_rate", 0.0)) * int(payload.get(phase, {}).get("steps", 0)) for payload in decoded) / count,
            "v_mean": sum(float(payload.get(phase, {}).get("v_mean", 0.0)) * int(payload.get(phase, {}).get("steps", 0)) for payload in decoded) / count,
            "omega_abs_mean": sum(float(payload.get(phase, {}).get("omega_abs_mean", 0.0)) * int(payload.get(phase, {}).get("steps", 0)) for payload in decoded) / count,
        }
    return result


def failure_reasons(row: Any) -> list[str]:
    reasons = []
    for field in ("collision", "stop_violation", "lane_failure", "invalid_pose"):
        if bool(getattr(row, field)):
            reasons.append(field)
    if int(row.unsafe_proximity_events) > 0:
        reasons.append("unsafe_pedestrian")
    if not bool(row.completed):
        reasons.append("incomplete")
    if not bool(row.restarted_after_stop):
        reasons.append("restart_failed")
    return reasons


def load_thresholds(config_path: Path, config: dict[str, Any]) -> PhaseThresholds:
    path = resolve(config_path, config["frozen"]["f11"]["r003_config"])
    with path.open("rb") as stream:
        values = tomllib.load(stream)["phases"]
    return PhaseThresholds(
        pedestrian_existence=float(values["pedestrian_existence_threshold"]),
        pedestrian_max_range_m=float(values["pedestrian_relevant_max_range_m"]),
        lane_curve_min_abs_curvature_inv_m=float(values["lane_curve_min_abs_curvature_inv_m"]),
        stop_satisfied_vicinity_m=float(values["stop_satisfied_vicinity_m"]),
    )


def coerce_row(row: dict[str, Any]) -> dict[str, Any]:
    from duckie_pomdp.evaluation.f10_ppo_policy import PPOEpisodeEvaluation
    fields = PPOEpisodeEvaluation.__dataclass_fields__
    episode_values: dict[str, Any] = {}
    for key, field in fields.items():
        value = row[key]
        if field.type in (bool, "bool"):
            value = str(value).lower() == "true"
        elif field.type in (int, "int", "int | None"):
            value = None if value in ("", "None", None) else int(value)
        elif field.type in (float, "float", "float | None"):
            value = None if value in ("", "None", None) else float(value)
        episode_values[key] = value
    return {"policy": str(row["policy"]), "seed": int(row["seed"]), "episode": PPOEpisodeEvaluation(**episode_values), **row}


def load_config(path: Path) -> dict[str, Any]:
    with path.open("rb") as stream:
        return tomllib.load(stream)


def resolve(config_path: Path, value: str) -> Path:
    return (config_path.parent / value).resolve()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def read_csv(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def append_csv(path: Path, row: dict[str, Any]) -> None:
    first = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(row))
        if first:
            writer.writeheader()
        writer.writerow(row)


if __name__ == "__main__":
    main()
