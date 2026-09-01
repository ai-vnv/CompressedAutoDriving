"""Development selection and one-shot final evaluation for F10 SAC."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path
from typing import Any, Iterable

from duckie_pomdp.control import F10GymEnvironment, SACAgent, load_f10_protocol
from duckie_pomdp.control.f10_protocol import file_sha256
from duckie_pomdp.evaluation.f10_policy import (
    AlwaysStopPolicy,
    CheckpointScore,
    EpisodeEvaluation,
    RandomPolicy,
    SACDeterministicPolicy,
    SimpleControllerPolicy,
    acceptance_checks,
    run_episode,
    select_checkpoint,
    summarize_episodes,
)


ROOT = Path(__file__).resolve().parents[1]


def evaluate_development(
    config_path: Path,
    artifact_dir: Path,
    *,
    device: str,
) -> dict[str, Any]:
    """Evaluate every planned checkpoint and freeze the safety-first choice."""

    protocol = load_f10_protocol(config_path)
    output_paths = (
        artifact_dir / "development_episodes.csv",
        artifact_dir / "dev_metrics.json",
        artifact_dir / "checkpoint_manifest.json",
        artifact_dir / "last.pt",
        artifact_dir / "best_return.pt",
        artifact_dir / "sac_baseline.pt",
    )
    _refuse_overwrite(output_paths)
    training_manifest = _load_json(artifact_dir / "training_run_manifest.json")
    config_sha = file_sha256(config_path)
    if training_manifest.get("f10_config_sha256") != config_sha:
        raise RuntimeError("training manifest does not match the frozen F10 config")
    expected_steps = tuple(
        range(
            protocol.sac.checkpoint_interval_steps,
            protocol.sac.training_steps + 1,
            protocol.sac.checkpoint_interval_steps,
        )
    )
    checkpoints = tuple(
        artifact_dir / "checkpoints" / f"sac_step_{step:07d}.pt"
        for step in expected_steps
    )
    missing = [str(path) for path in checkpoints if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"planned F10 checkpoints are missing: {missing}")

    environment = F10GymEnvironment(config_path, split="development")
    all_rows: list[EpisodeEvaluation] = []
    scores: list[CheckpointScore] = []
    try:
        for expected_step, checkpoint in zip(expected_steps, checkpoints, strict=True):
            agent, payload = SACAgent.load(checkpoint, device=device)
            actual_step = int(payload["global_step"])
            if actual_step != expected_step:
                raise RuntimeError(
                    f"checkpoint step mismatch for {checkpoint}: {actual_step}"
                )
            metadata = payload.get("metadata", {})
            if metadata.get("f10_config_sha256") != config_sha:
                raise RuntimeError(f"checkpoint config hash mismatch: {checkpoint}")
            policy = SACDeterministicPolicy(agent)
            rows = [
                run_episode(
                    environment,
                    seed=seed,
                    policy=policy,
                    protocol=protocol,
                    checkpoint_step=actual_step,
                )
                for seed in protocol.seeds.development
            ]
            summary = summarize_episodes(rows)
            all_rows.extend(rows)
            scores.append(
                CheckpointScore(
                    path=checkpoint.resolve(),
                    global_step=actual_step,
                    checkpoint_sha256=file_sha256(checkpoint),
                    summary=summary,
                )
            )
    finally:
        environment.close()

    rule = protocol.raw["checkpoint_selection"]
    selection = select_checkpoint(
        scores,
        maximum_collision_rate=float(rule["maximum_collision_rate"]),
        maximum_invalid_pose_rate=float(rule["maximum_invalid_pose_rate"]),
    )
    _write_episode_csv(output_paths[0], all_rows)
    development = {
        "schema_version": 1,
        "stage": "F10_DEVELOPMENT_CHECKPOINT_SELECTION",
        "f10_config": str(config_path.resolve()),
        "f10_config_sha256": config_sha,
        "seeds": list(protocol.seeds.development),
        "final_seeds_used": False,
        "checkpoint_rule": dict(rule),
        "candidates": [
            {
                "path": str(score.path),
                "global_step": score.global_step,
                "sha256": score.checkpoint_sha256,
                "summary": score.summary,
            }
            for score in scores
        ],
        "selection": {
            "selected_step": selection.selected.global_step,
            "selected_sha256": selection.selected.checkpoint_sha256,
            "best_return_step": selection.best_return.global_step,
            "last_step": selection.last.global_step,
            "safety_filter_passed": selection.safety_filter_passed,
            "reason": selection.selection_reason,
        },
    }
    output_paths[1].write_text(
        json.dumps(development, indent=2) + "\n", encoding="utf-8"
    )

    destinations = {
        "last": (selection.last.path, output_paths[3]),
        "best_return": (selection.best_return.path, output_paths[4]),
        "sac_baseline": (selection.selected.path, output_paths[5]),
    }
    for source, destination in destinations.values():
        shutil.copy2(source, destination)
    checkpoint_manifest = {
        "schema_version": 1,
        "stage": "F10_FROZEN_BASELINE_CHECKPOINT",
        "f10_config_sha256": config_sha,
        "development_metrics_sha256": file_sha256(output_paths[1]),
        "development_episodes_sha256": file_sha256(output_paths[0]),
        "selection_rule": str(rule["rule"]),
        "safety_filter_passed": selection.safety_filter_passed,
        "selection_reason": selection.selection_reason,
        "artifacts": {
            name: {
                "source": str(source),
                "path": str(destination.resolve()),
                "sha256": file_sha256(destination),
                "global_step": int(
                    SACAgent.load(destination, device="cpu")[1]["global_step"]
                ),
            }
            for name, (source, destination) in destinations.items()
        },
        "deployment_checkpoint": "sac_baseline",
    }
    output_paths[2].write_text(
        json.dumps(checkpoint_manifest, indent=2) + "\n", encoding="utf-8"
    )
    return development


def evaluate_final(
    config_path: Path,
    artifact_dir: Path,
    *,
    device: str,
) -> dict[str, Any]:
    """Evaluate the frozen baseline once; never perform checkpoint selection."""

    protocol = load_f10_protocol(config_path)
    episodes_path = artifact_dir / "evaluation_episodes.csv"
    metrics_path = artifact_dir / "final_metrics.json"
    _refuse_overwrite((episodes_path, metrics_path))
    checkpoint_manifest = _load_json(artifact_dir / "checkpoint_manifest.json")
    config_sha = file_sha256(config_path)
    if checkpoint_manifest.get("f10_config_sha256") != config_sha:
        raise RuntimeError("checkpoint manifest does not match frozen F10 config")
    baseline_record = checkpoint_manifest["artifacts"]["sac_baseline"]
    baseline_path = Path(baseline_record["path"])
    if file_sha256(baseline_path) != baseline_record["sha256"]:
        raise RuntimeError("frozen sac_baseline checkpoint hash mismatch")
    agent, payload = SACAgent.load(baseline_path, device=device)
    policies = (
        RandomPolicy(),
        AlwaysStopPolicy(),
        SimpleControllerPolicy(protocol),
        SACDeterministicPolicy(agent),
    )
    environment = F10GymEnvironment(config_path, split="final_evaluation")
    rows: list[EpisodeEvaluation] = []
    try:
        for policy in policies:
            rows.extend(
                run_episode(
                    environment,
                    seed=seed,
                    policy=policy,
                    protocol=protocol,
                    checkpoint_step=(
                        int(payload["global_step"])
                        if policy.name == "sac"
                        else None
                    ),
                )
                for seed in protocol.seeds.final_evaluation
            )
    finally:
        environment.close()
    _write_episode_csv(episodes_path, rows)
    summaries = {
        name: summarize_episodes([row for row in rows if row.policy == name])
        for name in ("random", "always_stop", "simple_controller", "sac")
    }
    checks = acceptance_checks(
        summaries["sac"], summaries["random"], summaries["always_stop"], protocol
    )
    safety_passed = bool(checkpoint_manifest["safety_filter_passed"])
    classification = "PASS" if safety_passed and all(checks.values()) else "LIMITED"
    metrics = {
        "schema_version": 1,
        "stage": "F10_FINAL_EVALUATION",
        "classification": classification,
        "f10_config_sha256": config_sha,
        "checkpoint": {
            "path": str(baseline_path.resolve()),
            "sha256": baseline_record["sha256"],
            "global_step": int(payload["global_step"]),
        },
        "seeds": list(protocol.seeds.final_evaluation),
        "checkpoint_selection_performed": False,
        "summaries": summaries,
        "acceptance_checks": checks,
        "safety_filter_passed_on_development": safety_passed,
        "evaluation_episodes_sha256": file_sha256(episodes_path),
    }
    metrics_path.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    return metrics


def _write_episode_csv(path: Path, rows: Iterable[EpisodeEvaluation]) -> None:
    materialized = list(rows)
    if not materialized:
        raise ValueError("cannot write an empty evaluation artifact")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(materialized[0].to_row()))
        writer.writeheader()
        writer.writerows(row.to_row() for row in materialized)


def _load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def _refuse_overwrite(paths: Iterable[Path]) -> None:
    existing = [str(path) for path in paths if path.exists()]
    if existing:
        raise FileExistsError(f"refusing to overwrite frozen F10 evaluation: {existing}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("development", "final"))
    parser.add_argument(
        "--config", type=Path, default=ROOT / "configs" / "f10_sac_v1.toml"
    )
    parser.add_argument(
        "--artifact-dir", type=Path, default=ROOT / "artifacts" / "f10"
    )
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    function = evaluate_development if args.phase == "development" else evaluate_final
    result = function(
        args.config.resolve(), args.artifact_dir.resolve(), device=args.device
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
