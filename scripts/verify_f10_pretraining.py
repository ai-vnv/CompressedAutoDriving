"""Produce the machine-readable gate required before full F10 training."""

from __future__ import annotations

import argparse
import json
import platform
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any
from xml.etree import ElementTree

import numpy as np
import torch

from duckie_pomdp.control import load_f10_protocol
from duckie_pomdp.control.f10_protocol import file_sha256


ROOT = Path(__file__).resolve().parents[1]
SOURCE_FILES = (
    "configs/f10_sac_v1.toml",
    "docs/F10_FORMULATION.md",
    "experiments/audit_f10_reward.py",
    "experiments/train_f10_sac.py",
    "experiments/evaluate_f10_sac.py",
    "experiments/smoke_f10_evaluator.py",
    "scripts/probe_f10_wandb.py",
    "scripts/verify_f10_pretraining.py",
    "src/duckie_pomdp/control/f10_protocol.py",
    "src/duckie_pomdp/control/gym_environment.py",
    "src/duckie_pomdp/control/policy_observation.py",
    "src/duckie_pomdp/control/reward.py",
    "src/duckie_pomdp/control/road_observer.py",
    "src/duckie_pomdp/control/sac.py",
    "src/duckie_pomdp/evaluation/f10_policy.py",
)


def verify(
    config_path: Path,
    artifact_dir: Path,
    junit_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    protocol = load_f10_protocol(config_path)
    config_sha = file_sha256(config_path)
    reward = _load_json(artifact_dir / "reward_audit.json")
    smoke = _load_json(artifact_dir / "smoke" / "training_run_manifest.json")
    evaluator = _load_json(artifact_dir / "evaluator_smoke.json")
    wandb = _load_json(artifact_dir / "wandb_online_preflight.json")
    tests = _junit_summary(junit_path)
    source_hashes = {
        relative: file_sha256(ROOT / relative) for relative in SOURCE_FILES
    }
    leaked_secret_paths = _secret_paths(ROOT)
    training_outputs = (
        artifact_dir / "training_metrics.csv",
        artifact_dir / "episode_metrics.csv",
        artifact_dir / "training_run_manifest.json",
        artifact_dir / "checkpoints",
    )
    checks = {
        "frozen_dependencies_load": True,
        "reward_audit_passed": bool(reward.get("passed")),
        "reward_audit_config_matches": reward.get("config_sha256") == config_sha,
        "smoke_config_matches": smoke.get("f10_config_sha256") == config_sha,
        "smoke_has_gradient_updates": int(smoke.get("gradient_updates", 0)) > 0,
        "smoke_checkpoint_reload_verified": bool(
            smoke.get("checkpoint_reload_verified")
        ),
        "smoke_training_rows_complete": int(smoke.get("training_metrics_rows", 0))
        == int(smoke.get("environment_steps", -1)),
        "evaluator_smoke_passed": bool(evaluator.get("passed")),
        "evaluator_config_matches": evaluator.get("f10_config_sha256")
        == config_sha,
        "evaluator_uses_development_only": evaluator.get("split") == "development"
        and evaluator.get("final_seed_used") is False,
        "wandb_online_verified": bool(wandb.get("verified")),
        "wandb_config_matches": wandb.get("f10_config_sha256") == config_sha,
        "wandb_destination_matches": wandb.get("entity")
        == protocol.raw["wandb"]["entity"]
        and wandb.get("project") == protocol.raw["wandb"]["project"],
        "cuda_available": bool(torch.cuda.is_available()),
        "full_tests_passed": tests["failures"] == 0
        and tests["errors"] == 0
        and tests["tests"] > 0,
        "no_project_secret": not leaked_secret_paths,
        "full_training_not_started": not any(path.exists() for path in training_outputs),
    }
    result = {
        "schema_version": 1,
        "stage": "F10_PRETRAINING_GATE",
        "verified_at_utc": datetime.now(timezone.utc).isoformat(),
        "ready_for_full_training": all(checks.values()),
        "f10_config": str(config_path.resolve()),
        "f10_config_sha256": config_sha,
        "artifact_directory": str(artifact_dir.resolve()),
        "checks": checks,
        "source_sha256": source_hashes,
        "upstream": {
            "yolo_checkpoint_sha256": protocol.detector_checkpoint_sha256,
            "belief_config_sha256": protocol.belief_config_sha256,
            "action_config_sha256": protocol.action_config_sha256,
        },
        "expected_checkpoint_steps": list(
            range(
                protocol.sac.checkpoint_interval_steps,
                protocol.sac.training_steps + 1,
                protocol.sac.checkpoint_interval_steps,
            )
        ),
        "selection_rule": dict(protocol.raw["checkpoint_selection"]),
        "seed_split": {
            "training": list(protocol.seeds.training),
            "development": list(protocol.seeds.development),
            "final_evaluation": list(protocol.seeds.final_evaluation),
        },
        "reward_audit_sha256": file_sha256(artifact_dir / "reward_audit.json"),
        "smoke_manifest_sha256": file_sha256(
            artifact_dir / "smoke" / "training_run_manifest.json"
        ),
        "evaluator_smoke_sha256": file_sha256(artifact_dir / "evaluator_smoke.json"),
        "wandb_preflight_sha256": file_sha256(
            artifact_dir / "wandb_online_preflight.json"
        ),
        "test_report": {**tests, "sha256": file_sha256(junit_path)},
        "wandb": {
            "entity": wandb.get("entity"),
            "project": wandb.get("project"),
            "preflight_run_id": wandb.get("run_id"),
            "preflight_run_url": wandb.get("run_url"),
            "credential_storage": wandb.get("credential_storage"),
        },
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "cuda_available": torch.cuda.is_available(),
            "gpu": (
                torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
            ),
            "wandb": _package_version("wandb"),
            "ultralytics": _package_version("ultralytics"),
            "gym": _package_version("gym"),
            "gym_duckietown_source": "6.2.0",
        },
        "project_secret_paths": leaked_secret_paths,
        "training_command": (
            "python experiments/train_f10_sac.py --wandb-mode online"
        ),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def _load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def _junit_summary(path: Path) -> dict[str, int]:
    if not path.is_file():
        raise FileNotFoundError(path)
    root = ElementTree.parse(path).getroot()
    suites = [root] if root.tag == "testsuite" else list(root.findall("testsuite"))
    return {
        name: sum(int(suite.attrib.get(name, 0)) for suite in suites)
        for name in ("tests", "failures", "errors", "skipped")
    }


def _package_version(name: str) -> str | None:
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def _secret_paths(root: Path) -> list[str]:
    # Build the sentinels in pieces so this verifier does not flag its own source.
    patterns = (b"wandb" + b"_v1_", b"WANDB_" + b"API_KEY=")
    suffixes = {".py", ".toml", ".md", ".json", ".yaml", ".yml", ".txt", ".csv"}
    excluded = {".git", "graphify-out", "__pycache__"}
    matches: list[str] = []
    for path in root.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in suffixes:
            continue
        if any(part in excluded for part in path.relative_to(root).parts):
            continue
        try:
            data = path.read_bytes()
        except OSError:
            continue
        if any(pattern in data for pattern in patterns):
            matches.append(str(path.relative_to(root)))
    return sorted(matches)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=Path, default=ROOT / "configs" / "f10_sac_v1.toml"
    )
    parser.add_argument(
        "--artifact-dir", type=Path, default=ROOT / "artifacts" / "f10"
    )
    parser.add_argument(
        "--junit",
        type=Path,
        default=ROOT / "artifacts" / "f10" / "pretraining_tests.xml",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "artifacts" / "f10" / "pretraining_gate.json",
    )
    args = parser.parse_args()
    result = verify(
        args.config.resolve(),
        args.artifact_dir.resolve(),
        args.junit.resolve(),
        args.output.resolve(),
    )
    print(
        json.dumps(
            {
                "ready_for_full_training": result["ready_for_full_training"],
                "f10_config_sha256": result["f10_config_sha256"],
                "checks": result["checks"],
                "test_report": result["test_report"],
                "wandb": result["wandb"],
                "environment": result["environment"],
            },
            indent=2,
        )
    )
    if not result["ready_for_full_training"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
