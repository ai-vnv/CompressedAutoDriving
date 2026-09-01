"""Development selection and one-shot final evaluation for F10-L1."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path
from typing import Any, Iterable

from duckie_pomdp.control import LaneCurriculumEnvironment, SACAgent, load_lane_protocol
from duckie_pomdp.control.f10_protocol import file_sha256
from duckie_pomdp.evaluation.lane_policy import (
    LaneAlwaysStopPolicy,
    LaneCheckpointScore,
    LaneEpisodeEvaluation,
    LaneRandomPolicy,
    LaneSACPolicy,
    LaneSimpleControllerPolicy,
    lane_acceptance_checks,
    run_lane_episode,
    select_lane_checkpoint,
    summarize_lane_episodes,
)


ROOT = Path(__file__).resolve().parents[1]


def evaluate_development(
    config_path: Path, artifact_dir: Path, *, device: str
) -> dict[str, Any]:
    protocol = load_lane_protocol(config_path)
    training_dir = artifact_dir / "training"
    checkpoints = sorted((training_dir / "checkpoints").glob("sac_step_*.pt"))
    if not checkpoints:
        raise FileNotFoundError("no F10-L1 training checkpoints")
    episodes_path = artifact_dir / "development_episodes.csv"
    metrics_path = artifact_dir / "development_metrics.json"
    manifest_path = artifact_dir / "checkpoint_manifest.json"
    destinations = {
        "last": artifact_dir / "lane_last.pt",
        "best_return": artifact_dir / "lane_best_return.pt",
        "safety_selected": artifact_dir / "sac_lane_baseline.pt",
    }
    _refuse_overwrite((episodes_path, metrics_path, manifest_path, *destinations.values()))
    config_sha = file_sha256(config_path)
    training_manifest = _load_json(training_dir / "training_run_manifest.json")
    if training_manifest.get("config_sha256") != config_sha:
        raise RuntimeError("training manifest does not match F10-L1 config")

    environment = LaneCurriculumEnvironment(config_path, split="development")
    all_rows: list[LaneEpisodeEvaluation] = []
    scores: list[LaneCheckpointScore] = []
    try:
        for checkpoint in checkpoints:
            agent, payload = SACAgent.load(checkpoint, device=device)
            step = int(payload["global_step"])
            policy = LaneSACPolicy(agent)
            rows = [
                run_lane_episode(
                    environment,
                    seed=seed,
                    policy=policy,
                    protocol=protocol,
                    checkpoint_step=step,
                )
                for seed in protocol.seeds.development
            ]
            all_rows.extend(rows)
            scores.append(
                LaneCheckpointScore(
                    path=checkpoint,
                    global_step=step,
                    sha256=file_sha256(checkpoint),
                    summary=summarize_lane_episodes(rows),
                )
            )
    finally:
        environment.close()
    rule = protocol.raw["checkpoint_selection"]
    selection = select_lane_checkpoint(
        scores,
        maximum_invalid_pose_rate=float(rule["maximum_invalid_pose_rate"]),
        maximum_yellow_crossing_rate=float(rule["maximum_yellow_crossing_rate"]),
        maximum_lane_departure_rate=float(rule["maximum_lane_departure_rate"]),
    )
    _write_rows(episodes_path, all_rows)
    result = {
        "schema_version": 1,
        "stage": "F10_L1_DEVELOPMENT_SELECTION",
        "config_sha256": config_sha,
        "seeds": list(protocol.seeds.development),
        "final_seeds_used": False,
        "selection_rule": dict(rule),
        "candidates": [
            {
                "path": str(score.path.resolve()),
                "global_step": score.global_step,
                "sha256": score.sha256,
                "summary": score.summary,
            }
            for score in scores
        ],
        "selection": {
            "selected_step": selection.selected.global_step,
            "selected_sha256": selection.selected.sha256,
            "best_return_step": selection.best_return.global_step,
            "last_step": selection.last.global_step,
            "safety_filter_passed": selection.safety_filter_passed,
            "reason": selection.reason,
        },
    }
    metrics_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    sources = {
        "last": selection.last,
        "best_return": selection.best_return,
        "safety_selected": selection.selected,
    }
    for name, score in sources.items():
        shutil.copy2(score.path, destinations[name])
    checkpoint_manifest = {
        "schema_version": 1,
        "stage": "F10_L1_FROZEN_LANE_CHECKPOINT",
        "config_sha256": config_sha,
        "development_metrics_sha256": file_sha256(metrics_path),
        "selection_rule": str(rule["rule"]),
        "safety_filter_passed": selection.safety_filter_passed,
        "selection_reason": selection.reason,
        "artifacts": {
            name: {
                "source": str(sources[name].path.resolve()),
                "path": str(path.resolve()),
                "sha256": file_sha256(path),
                "global_step": sources[name].global_step,
            }
            for name, path in destinations.items()
        },
        "deployment_for_lane_stage": "safety_selected",
        "full_pomdp_deployment_ready": False,
    }
    manifest_path.write_text(
        json.dumps(checkpoint_manifest, indent=2) + "\n", encoding="utf-8"
    )
    return result


def evaluate_final(
    config_path: Path, artifact_dir: Path, *, device: str
) -> dict[str, Any]:
    protocol = load_lane_protocol(config_path)
    episodes_path = artifact_dir / "final_evaluation_episodes.csv"
    metrics_path = artifact_dir / "final_metrics.json"
    _refuse_overwrite((episodes_path, metrics_path))
    checkpoint_manifest = _load_json(artifact_dir / "checkpoint_manifest.json")
    if checkpoint_manifest.get("config_sha256") != file_sha256(config_path):
        raise RuntimeError("checkpoint manifest does not match F10-L1 config")
    selected = checkpoint_manifest["artifacts"]["safety_selected"]
    checkpoint_path = Path(selected["path"])
    if file_sha256(checkpoint_path) != selected["sha256"]:
        raise RuntimeError("selected F10-L1 checkpoint hash mismatch")
    agent, payload = SACAgent.load(checkpoint_path, device=device)
    policies = (
        LaneRandomPolicy(),
        LaneAlwaysStopPolicy(),
        LaneSimpleControllerPolicy(protocol),
        LaneSACPolicy(agent),
    )
    environment = LaneCurriculumEnvironment(config_path, split="final_evaluation")
    rows: list[LaneEpisodeEvaluation] = []
    try:
        for policy in policies:
            rows.extend(
                run_lane_episode(
                    environment,
                    seed=seed,
                    policy=policy,
                    protocol=protocol,
                    checkpoint_step=(
                        int(payload["global_step"]) if policy.name == "sac" else None
                    ),
                )
                for seed in protocol.seeds.final_evaluation
            )
    finally:
        environment.close()
    _write_rows(episodes_path, rows)
    summaries = {
        name: summarize_lane_episodes([row for row in rows if row.policy == name])
        for name in ("random", "always_stop", "simple_controller", "sac")
    }
    checks = lane_acceptance_checks(
        summaries["sac"], summaries["random"], summaries["always_stop"], protocol
    )
    classification = (
        "PASS"
        if bool(checkpoint_manifest["safety_filter_passed"])
        and all(checks.values())
        else "LIMITED"
    )
    result = {
        "schema_version": 1,
        "stage": "F10_L1_FINAL_EVALUATION",
        "classification": classification,
        "config_sha256": file_sha256(config_path),
        "checkpoint": {
            "path": str(checkpoint_path.resolve()),
            "sha256": selected["sha256"],
            "global_step": int(payload["global_step"]),
        },
        "seeds": list(protocol.seeds.final_evaluation),
        "checkpoint_selection_performed": False,
        "summaries": summaries,
        "acceptance_checks": checks,
        "safety_filter_passed_on_development": bool(
            checkpoint_manifest["safety_filter_passed"]
        ),
        "evaluation_episodes_sha256": file_sha256(episodes_path),
    }
    metrics_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def _write_rows(path: Path, rows: Iterable[LaneEpisodeEvaluation]) -> None:
    materialized = list(rows)
    if not materialized:
        raise ValueError("cannot write empty F10-L1 evaluation")
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
        raise FileExistsError(f"refusing to overwrite F10-L1 evaluation: {existing}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("development", "final"))
    parser.add_argument(
        "--config", type=Path, default=ROOT / "configs" / "f10_l1_lane_v1.toml"
    )
    parser.add_argument(
        "--artifact-dir", type=Path, default=ROOT / "artifacts" / "f10_l1"
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
