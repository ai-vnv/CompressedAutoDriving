#!/usr/bin/env python3
"""Read-only artifact verifier for F11 R002 and R003 development gates."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib


ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify(config_path: Path) -> dict[str, object]:
    with config_path.open("rb") as stream:
        config = tomllib.load(stream)
    root = (config_path.parent / str(config["data"]["output_root"])).resolve()
    r002 = json.loads(
        (root / "r002" / "baseline_robustness.json").read_text(encoding="utf-8")
    )
    r003 = json.loads(
        (root / "r003" / "intervention_validation.json").read_text(
            encoding="utf-8"
        )
    )
    if r002["classification"] != "LIMITED":
        raise ValueError("R002 classification differs from frozen result")
    if r003["classification"] != "PASS" or not all(r003["criteria"].values()):
        raise ValueError("R003 does not pass every registered criterion")
    development = tuple(int(value) for value in config["data"]["development_seeds"])
    locked = set(int(value) for value in config["data"]["locked_evaluation_seeds"])
    if tuple(r002["seeds"]) != development or tuple(r003["seeds"]) != development:
        raise ValueError("development artifact seeds differ from config")
    if set(development) & locked:
        raise ValueError("development and locked seeds overlap")
    if r002["locked_evaluation_seeds_opened"] or r003["locked_evaluation_seeds_opened"]:
        raise ValueError("a locked evaluation seed was opened")

    trace_path = root / "r002" / "development_trace.npz"
    if sha256(trace_path) != r002["trace_sha256"]:
        raise ValueError("R002 trace hash mismatch")
    with np.load(trace_path, allow_pickle=False) as trace:
        if trace["observation"].shape != (8800, 29):
            raise ValueError("development trace shape mismatch")
        _forbid_schema(trace.files)
        if not np.isfinite(trace["observation"]).all():
            raise ValueError("development trace contains non-finite values")

    ig_path = root / "r002" / "integrated_gradients.npz"
    if sha256(ig_path) != r002["attribution_sha256"]:
        raise ValueError("R002 attribution hash mismatch")
    with np.load(ig_path, allow_pickle=False) as ig:
        _forbid_schema(ig.files)
        if ig["attribution"].shape != (3, 2, 2200, 29):
            raise ValueError("R002 attribution shape mismatch")
        if not np.isfinite(ig["attribution"]).all():
            raise ValueError("R002 attribution contains non-finite values")

    intervention_path = root / "r003" / "semantic_interventions.npz"
    if sha256(intervention_path) != r003["intervention_artifact_sha256"]:
        raise ValueError("R003 intervention hash mismatch")
    with np.load(intervention_path, allow_pickle=False) as intervention:
        _forbid_schema(intervention.files)
        if intervention["counterfactual_observation"].shape != (6, 2200, 29):
            raise ValueError("R003 intervention shape mismatch")
        names = tuple(str(value) for value in intervention["intervention_names"])
        sham = names.index("sham")
        np.testing.assert_array_equal(
            intervention["counterfactual_observation"][sham],
            intervention["factual_observation"],
        )
        if np.max(np.abs(intervention["delta_v_cmd_mps"][sham])) != 0.0:
            raise ValueError("R003 sham changed linear action")
        if np.max(np.abs(intervention["delta_omega_cmd_rad_s"][sham])) != 0.0:
            raise ValueError("R003 sham changed yaw action")

    if (root / "r004").exists():
        raise ValueError("R004 must remain unopened while R002 is LIMITED")
    full_test_log = root / "r002_r003_full_tests.log"
    test_text = full_test_log.read_text(encoding="utf-8")
    if "651 passed" not in test_text or " failed" in test_text or " skipped" in test_text:
        raise ValueError("full-test transcript does not prove 651/0/0")
    return {
        "classification": "PASS",
        "audit_scope": "R002/R003 artifact integrity",
        "r002_classification": "LIMITED",
        "r003_classification": "PASS",
        "development_samples": 2200,
        "locked_evaluation_seeds_opened": False,
        "r004_opened": False,
        "stored_privileged_truth": False,
        "full_tests": {"passed": 651, "failed": 0, "skipped": 0},
    }


def _forbid_schema(keys: list[str]) -> None:
    forbidden = ("privileged", "evaluation_gt", "ground_truth", "world_pose")
    offending = [key for key in keys if any(token in key.lower() for token in forbidden)]
    if offending:
        raise ValueError(f"privileged/evaluation fields found: {offending}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs" / "f11_ppo_explanation_development_v2.toml",
    )
    args = parser.parse_args()
    print(json.dumps(verify(args.config.resolve()), indent=2))


if __name__ == "__main__":
    main()
