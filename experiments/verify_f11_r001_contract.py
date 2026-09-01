#!/usr/bin/env python3
"""Read-only verification of frozen F11 R001 observation-contract artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10 project runtime
    import tomli as tomllib

from duckie_pomdp.control.ppo_protocol import load_ppo_curriculum_protocol
from duckie_pomdp.explain.observation_contract import validate_feature_group_partition


ROOT = Path(__file__).resolve().parents[1]
EXPECTED_TRACE_KEYS = (
    "step",
    "observation",
    "physical_observation",
    "deterministic_actor_mean",
    "environment_action",
    "physical_action",
    "critic_value",
    "rgb_sha256",
    "terminated",
    "truncated",
    "feature_names",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve(config_path: Path, value: str) -> Path:
    return (config_path.parent / value).resolve()


def verify(config_path: Path) -> dict[str, Any]:
    with config_path.open("rb") as stream:
        config = tomllib.load(stream)

    output = _resolve(config_path, str(config["r001"]["output_directory"]))
    audit_path = output / "contract_audit.json"
    trace_path = output / "public_trace.npz"
    manifest_path = output / "trace_manifest.json"
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    if manifest["classification"] != "PASS" or audit["classification"] != "PASS":
        raise ValueError("R001 is not classified PASS")
    if not all(bool(value) for value in audit["checks"].values()):
        raise ValueError("one or more R001 contract checks failed")
    if bool(manifest["stored_privileged_truth"]) or bool(
        audit["stored_privileged_truth"]
    ):
        raise ValueError("R001 trace reports stored privileged truth")
    if sha256(config_path) != manifest["config_sha256"]:
        raise ValueError("R001 config hash mismatch")
    if sha256(audit_path) != manifest["result_sha256"]:
        raise ValueError("R001 audit hash mismatch")
    if sha256(trace_path) != manifest["trace_sha256"]:
        raise ValueError("R001 trace hash mismatch")

    frozen = config["frozen_policy"]
    source_config = _resolve(config_path, str(frozen["config"]))
    protocol = load_ppo_curriculum_protocol(source_config)
    observation_order = tuple(protocol.observation_order)
    groups = {
        str(name): tuple(str(field) for field in fields)
        for name, fields in config["feature_groups"].items()
    }
    validate_feature_group_partition(observation_order, groups)
    if tuple(audit["observation_order"]) != observation_order:
        raise ValueError("audit observation order differs from frozen policy")
    if audit["primary_feature_groups"] != {
        name: list(fields) for name, fields in groups.items()
    }:
        raise ValueError("audit feature groups differ from frozen explanation config")

    expected_steps = int(config["r001"]["maximum_episode_steps"])
    with np.load(trace_path, allow_pickle=False) as trace:
        if tuple(trace.files) != EXPECTED_TRACE_KEYS:
            raise ValueError(f"unexpected public trace keys: {trace.files}")
        if trace["observation"].shape != (expected_steps, 29):
            raise ValueError("normalized observation trace has the wrong shape")
        if trace["physical_observation"].shape != (expected_steps, 29):
            raise ValueError("physical observation trace has the wrong shape")
        if trace["deterministic_actor_mean"].shape != (expected_steps, 2):
            raise ValueError("actor-mean trace has the wrong shape")
        if tuple(str(value) for value in trace["feature_names"]) != observation_order:
            raise ValueError("trace feature order differs from frozen policy")
        for key in (
            "step",
            "observation",
            "physical_observation",
            "deterministic_actor_mean",
            "environment_action",
            "physical_action",
            "critic_value",
        ):
            if not np.isfinite(trace[key]).all():
                raise ValueError(f"trace field {key!r} contains non-finite values")
        if len(set(str(value) for value in trace["rgb_sha256"])) < 2:
            raise ValueError("trace does not contain changing real RGB frames")

    tolerance = float(config["r001"]["replay_absolute_tolerance"])
    errors = audit["errors"]
    for key in (
        "maximum_actor_api_absolute",
        "maximum_action_mapping_absolute",
        "maximum_replay_actor_mean_absolute",
        "maximum_replay_critic_value_absolute",
    ):
        if float(errors[key]) > tolerance:
            raise ValueError(f"{key} exceeds the frozen tolerance")

    return {
        "classification": "PASS",
        "run_id": "R001",
        "steps": expected_steps,
        "observation_dimension": 29,
        "actor_target": audit["actor_target"],
        "maximum_replay_actor_mean_absolute": errors[
            "maximum_replay_actor_mean_absolute"
        ],
        "maximum_replay_critic_value_absolute": errors[
            "maximum_replay_critic_value_absolute"
        ],
        "stored_privileged_truth": False,
        "trace_sha256": manifest["trace_sha256"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs" / "f11_ppo_explanation_v2.toml",
    )
    args = parser.parse_args()
    print(json.dumps(verify(args.config.resolve()), indent=2))


if __name__ == "__main__":
    main()
