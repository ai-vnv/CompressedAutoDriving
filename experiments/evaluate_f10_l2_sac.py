"""Safety-first development selection and final evaluation for F10-L2."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path
from typing import Any, Iterable

from duckie_pomdp.control import (
    LaneTransferEnvironment,
    SACAgent,
    load_lane_transfer_protocol,
)
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


class NamedSACPolicy(LaneSACPolicy):
    def __init__(self, agent: SACAgent, name: str) -> None:
        super().__init__(agent)
        self.name = name


def _load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def _refuse_overwrite(paths: Iterable[Path]) -> None:
    existing = [path for path in paths if path.exists()]
    if existing:
        raise FileExistsError(f"refusing to overwrite F10-L2 artifacts: {existing}")


def _write_rows(path: Path, rows: list[LaneEpisodeEvaluation]) -> None:
    if not rows:
        raise ValueError("cannot write empty F10-L2 evaluation")
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0].to_row()))
        writer.writeheader()
        writer.writerows(row.to_row() for row in rows)


def evaluate_development(
    config_path: Path, artifact_dir: Path, *, device: str
) -> dict[str, Any]:
    protocol = load_lane_transfer_protocol(config_path)
    checkpoints = sorted((artifact_dir / "training" / "checkpoints").glob("sac_transfer_step_*.pt"))
    if not checkpoints:
        raise FileNotFoundError("no F10-L2 transfer checkpoints")
    rows_path = artifact_dir / "development_episodes.csv"
    metrics_path = artifact_dir / "development_metrics.json"
    manifest_path = artifact_dir / "checkpoint_manifest.json"
    destinations = {
        "last": artifact_dir / "lane_transfer_last.pt",
        "best_return": artifact_dir / "lane_transfer_best_return.pt",
        "safety_selected": artifact_dir / "sac_lane_transfer_baseline.pt",
    }
    _refuse_overwrite((rows_path, metrics_path, manifest_path, *destinations.values()))
    training_manifest = _load_json(artifact_dir / "training" / "training_run_manifest.json")
    config_sha = file_sha256(config_path)
    if training_manifest.get("config_sha256") != config_sha:
        raise RuntimeError("F10-L2 training manifest does not match config")
    environment = LaneTransferEnvironment(config_path, split="development")
    all_rows: list[LaneEpisodeEvaluation] = []
    scores: list[LaneCheckpointScore] = []
    try:
        for checkpoint in checkpoints:
            agent, payload = SACAgent.load(checkpoint, device=device)
            step = int(payload["global_step"])
            policy = NamedSACPolicy(agent, "sac_transfer")
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
    _write_rows(rows_path, all_rows)
    result = {
        "schema_version": 1,
        "stage": "F10_L2_DEVELOPMENT_SELECTION",
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
    for name, source in sources.items():
        shutil.copy2(source.path, destinations[name])
    checkpoint_manifest = {
        "schema_version": 1,
        "stage": "F10_L2_FROZEN_TRANSFER_CHECKPOINT",
        "config_sha256": config_sha,
        "source_checkpoint_sha256": protocol.transfer_checkpoint_sha256,
        "development_metrics_sha256": file_sha256(metrics_path),
        "selection_rule": str(rule["rule"]),
        "safety_filter_passed": selection.safety_filter_passed,
        "selection_reason": selection.reason,
        "artifacts": {
            name: {
                "source": str(sources[name].path.resolve()),
                "path": str(destination.resolve()),
                "sha256": file_sha256(destination),
                "global_step": sources[name].global_step,
            }
            for name, destination in destinations.items()
        },
        "deployment_for_lane_transfer_stage": "safety_selected",
        "full_pomdp_deployment_ready": False,
    }
    manifest_path.write_text(
        json.dumps(checkpoint_manifest, indent=2) + "\n", encoding="utf-8"
    )
    return result


def evaluate_final(
    config_path: Path, artifact_dir: Path, *, device: str
) -> dict[str, Any]:
    protocol = load_lane_transfer_protocol(config_path)
    rows_path = artifact_dir / "final_evaluation_episodes.csv"
    metrics_path = artifact_dir / "final_metrics.json"
    _refuse_overwrite((rows_path, metrics_path))
    checkpoint_manifest = _load_json(artifact_dir / "checkpoint_manifest.json")
    if checkpoint_manifest.get("config_sha256") != file_sha256(config_path):
        raise RuntimeError("F10-L2 checkpoint manifest does not match config")
    selected = checkpoint_manifest["artifacts"]["safety_selected"]
    selected_path = Path(selected["path"])
    if file_sha256(selected_path) != selected["sha256"]:
        raise RuntimeError("selected F10-L2 checkpoint hash mismatch")
    source_agent, source_payload = SACAgent.load(
        protocol.transfer_checkpoint_path, device=device
    )
    transfer_agent, transfer_payload = SACAgent.load(selected_path, device=device)
    policies = (
        LaneRandomPolicy(),
        LaneAlwaysStopPolicy(),
        LaneSimpleControllerPolicy(protocol),
        NamedSACPolicy(source_agent, "source_f10_l1_sac"),
        NamedSACPolicy(transfer_agent, "sac_transfer"),
    )
    environment = LaneTransferEnvironment(config_path, split="final_evaluation")
    rows: list[LaneEpisodeEvaluation] = []
    try:
        for policy in policies:
            checkpoint_step = (
                int(source_payload["global_step"])
                if policy.name == "source_f10_l1_sac"
                else int(transfer_payload["global_step"])
                if policy.name == "sac_transfer"
                else None
            )
            rows.extend(
                run_lane_episode(
                    environment,
                    seed=seed,
                    policy=policy,
                    protocol=protocol,
                    checkpoint_step=checkpoint_step,
                )
                for seed in protocol.seeds.final_evaluation
            )
    finally:
        environment.close()
    _write_rows(rows_path, rows)
    names = (
        "random",
        "always_stop",
        "simple_controller",
        "source_f10_l1_sac",
        "sac_transfer",
    )
    summaries = {
        name: summarize_lane_episodes([row for row in rows if row.policy == name])
        for name in names
    }
    checks = lane_acceptance_checks(
        summaries["sac_transfer"],
        summaries["random"],
        summaries["always_stop"],
        protocol,
    )
    classification = (
        "PASS"
        if bool(checkpoint_manifest["safety_filter_passed"])
        and all(checks.values())
        else "LIMITED"
    )
    result = {
        "schema_version": 1,
        "stage": "F10_L2_FINAL_EVALUATION",
        "classification": classification,
        "config_sha256": file_sha256(config_path),
        "source_checkpoint": {
            "path": str(protocol.transfer_checkpoint_path),
            "sha256": protocol.transfer_checkpoint_sha256,
            "global_step": int(source_payload["global_step"]),
        },
        "selected_checkpoint": {
            "path": str(selected_path.resolve()),
            "sha256": selected["sha256"],
            "transfer_global_step": int(transfer_payload["global_step"]),
        },
        "seeds": list(protocol.seeds.final_evaluation),
        "checkpoint_selection_performed": False,
        "summaries": summaries,
        "acceptance_checks": checks,
        "safety_filter_passed_on_development": bool(
            checkpoint_manifest["safety_filter_passed"]
        ),
        "evaluation_episodes_sha256": file_sha256(rows_path),
    }
    metrics_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=Path, default=ROOT / "configs" / "f10_l2_transfer_v1.toml"
    )
    parser.add_argument(
        "--artifact-dir", type=Path, default=ROOT / "artifacts" / "f10_l2"
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--final", action="store_true")
    args = parser.parse_args()
    function = evaluate_final if args.final else evaluate_development
    print(
        json.dumps(
            function(
                args.config.resolve(), args.artifact_dir.resolve(), device=args.device
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

