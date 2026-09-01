"""Safety-first PPO checkpoint selection, stage-final, and retention evaluation."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path
from typing import Any, Iterable

from duckie_pomdp.control import (
    PPOAgent,
    PPOCurriculumEnvironment,
    classify_curriculum_stage,
    evaluate_retention_change,
    load_ppo_curriculum_protocol,
    require_stage_in_protocol_scope,
)
from duckie_pomdp.control.ppo_protocol import (
    load_retention_reference,
    protocol_artifact_root,
)
from duckie_pomdp.control.f10_protocol import file_sha256
from duckie_pomdp.evaluation.f10_ppo_policy import (
    AlwaysStopPolicy,
    BeliefAwareSimpleController,
    PPOEpisodeEvaluation,
    PPODeterministicPolicy,
    RandomPolicy,
    run_episode,
    summarize_episodes,
)


ROOT = Path(__file__).resolve().parents[1]


def _candidate_steps(protocol, stage) -> tuple[int, ...]:
    steps = tuple(
        range(stage.checkpoint_interval_steps, stage.training_steps + 1, stage.checkpoint_interval_steps)
    )
    if protocol.raw.get("behavior_warm_start", {}).get(stage.key, {}).get("enabled"):
        return (0,) + steps
    return steps


def _skill_checks(protocol, stage_key: str, summary: dict[str, Any]) -> dict[str, bool]:
    config = protocol.raw["acceptance"][stage_key]
    if stage_key == "c0":
        return {
            "completion": summary["completion_rate"] >= float(config["minimum_completion_rate"]),
            "lane_failure": summary["lane_failure_rate"] <= float(config["maximum_lane_failure_rate"]),
            "invalid_pose": summary["invalid_pose_rate"] <= float(config["maximum_invalid_pose_rate"]),
            "lateral_error": summary["mean_abs_lateral_error_m"] <= float(config["maximum_mean_abs_lateral_error_m"]),
            "progress": summary["mean_progress_m"] >= float(config["minimum_mean_progress_m"]),
        }
    if stage_key == "c1":
        return {
            "completion": summary["completion_rate"] >= float(config["minimum_completion_rate"]),
            "lane_failure": summary["lane_failure_rate"] <= float(config["maximum_lane_failure_rate"]),
            "invalid_pose": summary["invalid_pose_rate"] <= float(config["maximum_invalid_pose_rate"]),
        }
    if stage_key == "c2":
        return {
            "collision": summary["collision_rate"] <= float(config["maximum_collision_rate"]),
            "unsafe": summary["unsafe_episode_rate"] <= float(config["maximum_unsafe_episode_rate"]),
            "progress": summary["mean_progress_m"] >= float(config["minimum_mean_progress_m"]),
            "not_stationary": summary["stationary_fraction"] <= float(config["maximum_stationary_fraction"]),
        }
    if stage_key == "c3":
        return {
            "stop_completion": summary["stop_completion_rate"] >= float(config["minimum_stop_completion_rate"]),
            "stop_violation": summary["stop_violation_rate"] <= float(config["maximum_stop_violation_rate"]),
            "restart": summary["restart_rate"] >= float(config["minimum_restart_rate"]),
        }
    checks = {
        "collision": summary["collision_rate"] <= float(config["maximum_collision_rate"]),
        "stop_completion": summary["stop_completion_rate"] >= float(config["minimum_stop_completion_rate"]),
        "stop_violation": summary["stop_violation_rate"] <= float(config["maximum_stop_violation_rate"]),
        "progress": summary["mean_progress_m"] >= float(config["minimum_mean_progress_m"]),
    }
    optional_checks = {
        "unsafe": (
            "maximum_unsafe_episode_rate",
            summary["unsafe_episode_rate"],
            lambda actual, threshold: actual <= threshold,
        ),
        "restart": (
            "minimum_restart_rate",
            summary["restart_rate"],
            lambda actual, threshold: actual >= threshold,
        ),
        "completion": (
            "minimum_completion_rate",
            summary["completion_rate"],
            lambda actual, threshold: actual >= threshold,
        ),
        "lane_failure": (
            "maximum_lane_failure_rate",
            summary["lane_failure_rate"],
            lambda actual, threshold: actual <= threshold,
        ),
        "not_stationary": (
            "maximum_stationary_fraction",
            summary["stationary_fraction"],
            lambda actual, threshold: actual <= threshold,
        ),
    }
    for name, (key, actual, compare) in optional_checks.items():
        if key in config:
            checks[name] = compare(actual, float(config[key]))
    return checks


def _safety_pass(protocol, stage_key: str, summary: dict[str, Any]) -> bool:
    rule = protocol.raw["checkpoint_selection"]
    stage = protocol.stage(stage_key)
    return bool(
        summary["invalid_pose_rate"] <= float(rule["maximum_invalid_pose_rate"])
        and summary["lane_failure_rate"] <= float(rule["maximum_lane_failure_rate"])
        and (not stage.pedestrian_active or summary["collision_rate"] <= float(rule["maximum_collision_rate"]))
        and (not stage.stop_active or summary["stop_violation_rate"] <= float(rule["maximum_stop_violation_rate"]))
    )


def _rank(stage_key: str, summary: dict[str, Any], step: int):
    if stage_key in {"c0", "c1"}:
        return (summary["completion_rate"], summary["mean_progress_m"], -summary["mean_abs_lateral_error_m"], summary["mean_return"], step)
    if stage_key == "c2":
        return (-summary["collision_rate"], summary["mean_progress_m"], -summary["stationary_fraction"], summary["mean_return"], step)
    if stage_key == "c3":
        return (summary["stop_completion_rate"], summary["restart_rate"], -summary["stop_violation_rate"], summary["mean_progress_m"], step)
    return (-summary["collision_rate"], summary["stop_completion_rate"], -summary["stop_violation_rate"], summary["mean_progress_m"], step)


def _retention_checks(
    protocol,
    stage_key: str,
    artifact_root: Path,
    summaries: dict[str, dict[str, Any]],
    *,
    matched_import_baseline: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], bool]:
    """Apply only the pre-registered cross-stage degradation thresholds."""

    if stage_key == "c0":
        return {"c0_baseline_recorded": "c0" in summaries}, "c0" in summaries
    targets = {"c1": "c0", "c2": "c1", "c3": "c2"}
    if stage_key == "c4":
        expected = set(("c0", "c1", "c2", "c3", "c4"))
        complete = set(summaries) == expected
        config = protocol.raw["acceptance"].get("c4_retention", {})
        checks = {"complete_retention_matrix": complete}
        if complete and config:
            checks.update(
                {
                    "c0_completion": summaries["c0"]["completion_rate"]
                    >= float(config["minimum_c0_completion_rate"]),
                    "c0_lane": summaries["c0"]["lane_failure_rate"]
                    <= float(config["maximum_c0_lane_failure_rate"]),
                    "c1_completion": summaries["c1"]["completion_rate"]
                    >= float(config["minimum_c1_completion_rate"]),
                    "c1_lane": summaries["c1"]["lane_failure_rate"]
                    <= float(config["maximum_c1_lane_failure_rate"]),
                    "c2_collision": summaries["c2"]["collision_rate"]
                    <= float(config["maximum_c2_collision_rate"]),
                    "c2_progress": summaries["c2"]["mean_progress_m"]
                    >= float(config["minimum_c2_mean_progress_m"]),
                    "c3_stop_completion": summaries["c3"]["stop_completion_rate"]
                    >= float(config["minimum_c3_stop_completion_rate"]),
                    "c3_stop_violation": summaries["c3"]["stop_violation_rate"]
                    <= float(config["maximum_c3_stop_violation_rate"]),
                    "c3_restart": summaries["c3"]["restart_rate"]
                    >= float(config["minimum_c3_restart_rate"]),
                }
            )
        return checks, all(checks.values())
    target = targets[stage_key]
    if matched_import_baseline is None:
        baseline, provenance = load_retention_reference(
            protocol, target, artifact_root
        )
    else:
        baseline = matched_import_baseline["summary"]
        provenance = {
            "reference_checkpoint_stage": target,
            "reference_imported": True,
            "reference_protocol_sha256": matched_import_baseline[
                "source_protocol_sha256"
            ],
            "reference_checkpoint_sha256": matched_import_baseline[
                "checkpoint_sha256"
            ],
            "reference_evaluation": "same_config_same_development_seeds",
        }
    comparison, passed = evaluate_retention_change(
        protocol,
        stage_key,
        baseline,
        summaries[target],
    )
    return {
        **provenance,
        "target_task": target,
        **comparison,
    }, passed


def development(config_path: Path, stage_key: str, stage_dir: Path, *, device: str) -> dict:
    protocol = load_ppo_curriculum_protocol(config_path)
    require_stage_in_protocol_scope(protocol, stage_key)
    stage = protocol.stage(stage_key)
    output = stage_dir / "development_metrics.json"
    episodes_path = stage_dir / "development_episodes.csv"
    manifest_path = stage_dir / "checkpoint_manifest.json"
    _refuse((output, episodes_path, manifest_path, stage_dir / "ppo_selected.pt"))
    training_dir = stage_dir / "training"
    training_manifest = _load(training_dir / "training_run_manifest.json")
    config_sha = file_sha256(config_path)
    if training_manifest["config_sha256"] != config_sha:
        raise RuntimeError("training/config provenance mismatch")
    candidates = []
    rows: list[PPOEpisodeEvaluation] = []
    env = PPOCurriculumEnvironment(config_path, stage=stage_key, split="development")
    try:
        for step in _candidate_steps(protocol, stage):
            checkpoint = training_dir / "checkpoints" / f"ppo_{stage_key}_step_{step:07d}.pt"
            agent, payload = PPOAgent.load(checkpoint, device=device)
            if int(payload["global_step"]) != step or payload["metadata"]["config_sha256"] != config_sha:
                raise RuntimeError(f"invalid checkpoint provenance: {checkpoint}")
            policy = PPODeterministicPolicy(agent)
            current = [
                run_episode(env, seed=seed, policy=policy, protocol=protocol, checkpoint_step=step)
                for seed in stage.development_seeds
            ]
            summary = summarize_episodes(current)
            rows.extend(current)
            candidates.append({
                "path": str(checkpoint.resolve()),
                "sha256": file_sha256(checkpoint),
                "global_step": step,
                "summary": summary,
                "safety_pass": _safety_pass(protocol, stage_key, summary),
                "skill_checks": _skill_checks(protocol, stage_key, summary),
            })
    finally:
        env.close()
    minimum_updated_step = int(
        protocol.raw["checkpoint_selection"]
        .get("minimum_updated_global_step", {})
        .get(stage_key, 0)
    )
    for row in candidates:
        row["on_policy_update_eligible"] = (
            int(row["global_step"]) >= minimum_updated_step
        )
    eligible = [
        row
        for row in candidates
        if row["safety_pass"]
        and all(row["skill_checks"].values())
        and row["on_policy_update_eligible"]
    ]
    pool = eligible or candidates
    selected = max(pool, key=lambda row: _rank(stage_key, row["summary"], row["global_step"]))
    best_return = max(candidates, key=lambda row: (row["summary"]["mean_return"], row["global_step"]))
    last = candidates[-1]
    result = {
        "schema_version": 1,
        "stage": stage_key,
        "phase": "development_selection",
        "config_sha256": config_sha,
        "seeds": list(stage.development_seeds),
        "stage_final_seeds_used": False,
        "global_final_seeds_used": False,
        "rule": protocol.raw["checkpoint_selection"],
        "candidates": candidates,
        "selection": {
            "selected_step": selected["global_step"],
            "selected_sha256": selected["sha256"],
            "eligible": selected in eligible,
            "minimum_updated_global_step": minimum_updated_step,
            "reason": (
                "safety+stage-skill+on-policy-update filter, then frozen stage rank"
                if eligible
                else "no fully eligible updated checkpoint; diagnostic candidate only"
            ),
        },
    }
    _write_rows(episodes_path, rows)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    copies = {
        "selected": (Path(selected["path"]), stage_dir / "ppo_selected.pt"),
        "best_return": (Path(best_return["path"]), stage_dir / "ppo_best_return.pt"),
        "last": (Path(last["path"]), stage_dir / "ppo_last.pt"),
    }
    for source, target in copies.values():
        shutil.copy2(source, target)
    checkpoint_manifest = {
        "schema_version": 1,
        "stage": stage_key,
        "config_sha256": config_sha,
        "development_metrics_sha256": file_sha256(output),
        "artifacts": {
            name: {"path": str(target.resolve()), "sha256": file_sha256(target)}
            for name, (_, target) in copies.items()
        },
        "selected_is_gate_eligible": selected in eligible,
    }
    manifest_path.write_text(json.dumps(checkpoint_manifest, indent=2) + "\n", encoding="utf-8")
    return result


def stage_final(config_path: Path, stage_key: str, stage_dir: Path, *, device: str) -> dict:
    protocol = load_ppo_curriculum_protocol(config_path)
    require_stage_in_protocol_scope(protocol, stage_key)
    stage = protocol.stage(stage_key)
    metrics_path = stage_dir / "stage_final_metrics.json"
    rows_path = stage_dir / "stage_final_episodes.csv"
    _refuse((metrics_path, rows_path))
    manifest = _load(stage_dir / "checkpoint_manifest.json")
    retention_result = _load(stage_dir / "retention_metrics.json")
    config_sha = file_sha256(config_path)
    if manifest.get("config_sha256") != config_sha:
        raise RuntimeError("checkpoint manifest/config provenance mismatch")
    if manifest.get("selected_is_gate_eligible") is not True:
        raise RuntimeError(
            "once-only stage-final is forbidden without an eligible development checkpoint"
        )
    if retention_result.get("config_sha256") != config_sha or retention_result.get("stage") != stage_key:
        raise RuntimeError("retention/config or stage provenance mismatch")
    if retention_result.get("retention_pass") is not True:
        retention_pass = False
    else:
        retention_pass = True
    selected = stage_dir / "ppo_selected.pt"
    if file_sha256(selected) != manifest["artifacts"]["selected"]["sha256"]:
        raise RuntimeError("selected PPO checkpoint hash mismatch")
    agent, payload = PPOAgent.load(selected, device=device)
    policies = (
        RandomPolicy(), AlwaysStopPolicy(), BeliefAwareSimpleController(protocol), PPODeterministicPolicy(agent)
    )
    env = PPOCurriculumEnvironment(config_path, stage=stage_key, split="stage_final")
    rows: list[PPOEpisodeEvaluation] = []
    try:
        for policy in policies:
            rows.extend(
                run_episode(
                    env, seed=seed, policy=policy, protocol=protocol,
                    checkpoint_step=int(payload["global_step"]) if policy.name == "ppo" else None,
                )
                for seed in stage.stage_final_seeds
            )
    finally:
        env.close()
    summaries = {
        name: summarize_episodes([row for row in rows if row.policy == name])
        for name in ("random", "always_stop", "simple_controller", "ppo")
    }
    checks = _skill_checks(protocol, stage_key, summaries["ppo"])
    safety = _safety_pass(protocol, stage_key, summaries["ppo"])
    stage_skill = all(checks.values())
    classification, progression_permitted = classify_curriculum_stage(
        safety_pass=safety,
        skill_pass=stage_skill,
        retention_pass=retention_pass,
    )
    result = {
        "schema_version": 1,
        "stage": stage_key,
        "phase": "once_only_stage_final",
        "classification": classification,
        "config_sha256": config_sha,
        "checkpoint": {"path": str(selected.resolve()), "sha256": file_sha256(selected), "global_step": int(payload["global_step"])},
        "seeds": list(stage.stage_final_seeds),
        "checkpoint_selection_performed": False,
        "summaries": summaries,
        "skill_checks": checks,
        "safety_pass": safety,
        "retention_pass": retention_pass,
        "progression_permitted": progression_permitted,
    }
    _write_rows(rows_path, rows)
    result["episodes_sha256"] = file_sha256(rows_path)
    metrics_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def retention(config_path: Path, stage_key: str, stage_dir: Path, *, device: str) -> dict:
    protocol = load_ppo_curriculum_protocol(config_path)
    require_stage_in_protocol_scope(protocol, stage_key)
    index = ("c0", "c1", "c2", "c3", "c4").index(stage_key)
    target_keys = ("c0", "c1", "c2", "c3", "c4")[: index + 1]
    metrics_path = stage_dir / "retention_metrics.json"
    rows_path = stage_dir / "retention_episodes.csv"
    _refuse((metrics_path, rows_path))
    checkpoint = stage_dir / "ppo_selected.pt"
    agent, payload = PPOAgent.load(checkpoint, device=device)
    policy = PPODeterministicPolicy(agent)
    rows: list[PPOEpisodeEvaluation] = []
    summaries = {}
    for target in target_keys:
        target_stage = protocol.stage(target)
        env = PPOCurriculumEnvironment(config_path, stage=target, split="development")
        try:
            current = [
                run_episode(env, seed=seed, policy=policy, protocol=protocol, checkpoint_step=int(payload["global_step"]))
                for seed in target_stage.development_seeds
            ]
        finally:
            env.close()
        rows.extend(current)
        summaries[target] = summarize_episodes(current)

    matched_import_baseline = None
    target_for_stage = {"c1": "c0", "c2": "c1", "c3": "c2"}.get(stage_key)
    imported = (
        protocol.raw.get("curriculum_import", {}).get(target_for_stage)
        if target_for_stage is not None
        else None
    )
    if imported is not None:
        imported = dict(imported)
        source_checkpoint = (
            config_path.parent / str(imported["selected_checkpoint"])
        ).resolve()
        source_agent, source_payload = PPOAgent.load(source_checkpoint, device=device)
        if source_payload["stage"] != target_for_stage:
            raise RuntimeError("imported retention checkpoint stage mismatch")
        source_policy = PPODeterministicPolicy(source_agent)
        target_stage = protocol.stage(target_for_stage)
        source_env = PPOCurriculumEnvironment(
            config_path, stage=target_for_stage, split="development"
        )
        try:
            source_rows = [
                run_episode(
                    source_env,
                    seed=seed,
                    policy=source_policy,
                    protocol=protocol,
                    checkpoint_step=int(source_payload["global_step"]),
                )
                for seed in target_stage.development_seeds
            ]
        finally:
            source_env.close()
        matched_import_baseline = {
            "summary": summarize_episodes(source_rows),
            "checkpoint": str(source_checkpoint),
            "checkpoint_sha256": file_sha256(source_checkpoint),
            "source_protocol_sha256": imported["source_protocol_sha256"],
            "seeds": list(target_stage.development_seeds),
        }
    result = {
        "schema_version": 1,
        "stage": stage_key,
        "config_sha256": file_sha256(config_path),
        "checkpoint_stage": stage_key,
        "checkpoint_sha256": file_sha256(checkpoint),
        "development_seeds_only": True,
        "summaries": summaries,
    }
    checks, retention_pass = _retention_checks(
        protocol,
        stage_key,
        stage_dir.parent,
        summaries,
        matched_import_baseline=matched_import_baseline,
    )
    if matched_import_baseline is not None:
        result["matched_import_baseline"] = matched_import_baseline
    result["checks"] = checks
    result["retention_pass"] = retention_pass
    _write_rows(rows_path, rows)
    result["episodes_sha256"] = file_sha256(rows_path)
    metrics_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def global_final(config_path: Path, artifact_root: Path, *, device: str) -> dict:
    """Run the untouched five-task final holdout exactly once."""

    protocol = load_ppo_curriculum_protocol(config_path)
    require_stage_in_protocol_scope(protocol, "c4")
    metrics_path = artifact_root / "final_metrics.json"
    rows_path = artifact_root / "global_final_episodes.csv"
    matrix_path = artifact_root / "forgetting_matrix.csv"
    _refuse((metrics_path, rows_path, matrix_path))
    c4_dir = artifact_root / "c4"
    final_gate = _load(c4_dir / "stage_final_metrics.json")
    if final_gate.get("classification") != "PASS" or final_gate.get("progression_permitted") is not True:
        raise RuntimeError("global final is forbidden until C4 PASSES")
    checkpoint = c4_dir / "ppo_selected.pt"
    manifest = _load(c4_dir / "checkpoint_manifest.json")
    if file_sha256(checkpoint) != manifest["artifacts"]["selected"]["sha256"]:
        raise RuntimeError("C4 selected checkpoint hash mismatch")
    agent, payload = PPOAgent.load(checkpoint, device=device)
    policy = PPODeterministicPolicy(agent)
    task_stage = {
        "small_loop": "c0",
        "experiment_loop": "c1",
        "pedestrian": "c2",
        "stop": "c3",
        "combined": "c4",
    }
    rows: list[PPOEpisodeEvaluation] = []
    summaries: dict[str, dict[str, Any]] = {}
    for task, target_stage in task_stage.items():
        seeds = protocol.global_final[task]
        env = PPOCurriculumEnvironment(
            config_path,
            stage=target_stage,
            split="global_final",
            seeds=seeds,
        )
        try:
            current = [
                run_episode(
                    env,
                    seed=seed,
                    policy=policy,
                    protocol=protocol,
                    checkpoint_step=int(payload["global_step"]),
                )
                for seed in seeds
            ]
        finally:
            env.close()
        rows.extend(current)
        summaries[task] = summarize_episodes(current)
    _write_rows(rows_path, rows)
    _write_forgetting_matrix(artifact_root, matrix_path)
    result = {
        "schema_version": 1,
        "phase": "once_only_global_final",
        "config_sha256": file_sha256(config_path),
        "checkpoint": {
            "path": str(checkpoint.resolve()),
            "sha256": file_sha256(checkpoint),
            "global_step": int(payload["global_step"]),
        },
        "selection_performed": False,
        "training_performed": False,
        "seed_groups": {name: list(values) for name, values in protocol.global_final.items()},
        "summaries": summaries,
        "episodes_sha256": file_sha256(rows_path),
        "forgetting_matrix_sha256": file_sha256(matrix_path),
    }
    metrics_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def report(config_path: Path, artifact_root: Path) -> dict:
    """Assemble the review report from immutable stage/final artifacts."""

    protocol = load_ppo_curriculum_protocol(config_path)
    require_stage_in_protocol_scope(protocol, "c4")
    target = ROOT / "docs" / "F10_PPO_REPORT_FOR_REVIEW.md"
    _refuse((target,))
    stages = {
        key: _load(artifact_root / key / "stage_final_metrics.json")
        for key in ("c0", "c1", "c2", "c3", "c4")
    }
    if any(value.get("classification") != "PASS" for value in stages.values()):
        raise RuntimeError("review report requires every stage to PASS")
    final = _load(artifact_root / "final_metrics.json")
    checkpoint = final["checkpoint"]
    lines = [
        "# F10-PPO Report for Review",
        "",
        "## Classification",
        "",
        "PASS — all five pre-registered curriculum stages passed before the once-only global holdout.",
        "",
        "## Frozen policy",
        "",
        f"- Observation dimension: {len(protocol.observation_order)}",
        f"- Architecture: {list(protocol.ppo.hidden_sizes)} Tanh feed-forward actor and critic",
        f"- Final checkpoint: `{checkpoint['path']}`",
        f"- Checkpoint SHA256: `{checkpoint['sha256']}`",
        f"- Config SHA256: `{file_sha256(config_path)}`",
        "",
        "## Curriculum stage classifications",
        "",
        "| Stage | Classification | Selected step |",
        "| --- | --- | ---: |",
    ]
    for key, value in stages.items():
        lines.append(f"| {key.upper()} | {value['classification']} | {value['checkpoint']['global_step']} |")
    lines.extend((
        "",
        "## Once-only global holdout",
        "",
        "| Task | Completion | Progress (m) | Collision | Stop completion | Stop violation | Mean |d| (m) |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ))
    for task, summary in final["summaries"].items():
        lines.append(
            f"| {task} | {summary['completion_rate']:.3f} | {summary['mean_progress_m']:.3f} | "
            f"{summary['collision_rate']:.3f} | {summary['stop_completion_rate']:.3f} | "
            f"{summary['stop_violation_rate']:.3f} | {summary['mean_abs_lateral_error_m']:.3f} |"
        )
    lines.extend((
        "",
        "## Evidence",
        "",
        "- `artifacts/f10_ppo/forgetting_matrix.csv`",
        "- `artifacts/f10_ppo/global_final_episodes.csv`",
        "- `artifacts/f10_ppo/final_metrics.json`",
        "",
        "No explanation, optimization, recurrent policy, or post-holdout retraining was performed in F10-PPO.",
        "",
    ))
    target.write_text("\n".join(lines), encoding="utf-8")
    return {"report": str(target), "sha256": file_sha256(target)}


def _write_forgetting_matrix(artifact_root: Path, path: Path) -> None:
    rows: list[dict[str, Any]] = []
    for checkpoint_stage in ("c0", "c1", "c2", "c3", "c4"):
        retention_result = _load(artifact_root / checkpoint_stage / "retention_metrics.json")
        for target_stage, summary in retention_result["summaries"].items():
            rows.append({
                "checkpoint_stage": checkpoint_stage,
                "target_stage": target_stage,
                **summary,
            })
    if not rows:
        raise RuntimeError("forgetting matrix is empty")
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _write_rows(path: Path, rows: Iterable[PPOEpisodeEvaluation]) -> None:
    values = list(rows)
    if not values:
        raise ValueError("empty evaluation")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(values[0].to_row()))
        writer.writeheader()
        writer.writerows(row.to_row() for row in values)


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _refuse(paths: Iterable[Path]) -> None:
    existing = [str(path) for path in paths if path.exists()]
    if existing:
        raise FileExistsError(f"refusing to overwrite final PPO evidence: {existing}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("development", "retention", "stage-final", "global-final", "report"))
    parser.add_argument("stage", nargs="?", choices=("c0", "c1", "c2", "c3", "c4"))
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "f10_ppo_v1.toml")
    parser.add_argument("--stage-dir", type=Path)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.phase in {"global-final", "report"}:
        protocol = load_ppo_curriculum_protocol(args.config.resolve())
        root = protocol_artifact_root(protocol)
        value = global_final(args.config.resolve(), root, device=args.device) if args.phase == "global-final" else report(args.config.resolve(), root)
        print(json.dumps(value, indent=2))
        return
    if args.stage is None:
        parser.error("stage is required for development, retention, and stage-final")
    protocol = load_ppo_curriculum_protocol(args.config.resolve())
    directory = args.stage_dir or protocol_artifact_root(protocol) / args.stage
    function = {"development": development, "stage-final": stage_final, "retention": retention}[args.phase]
    print(json.dumps(function(args.config.resolve(), args.stage, directory.resolve(), device=args.device), indent=2))


if __name__ == "__main__":
    main()
