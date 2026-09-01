#!/usr/bin/env python3
"""Read-only verification of the immutable failed R006 attempt."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

import numpy as np
import torch

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))

from run_f11_r004_once import _load as load_r004  # noqa: E402
from duckie_pomdp.control.ppo import PPOAgent  # noqa: E402
from duckie_pomdp.explain.ppo_integrated_gradients import (  # noqa: E402
    PPOActionLimits,
    target_values,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs" / "f11_ppo_explanation_r006_v1.toml",
    )
    args = parser.parse_args()
    config_path = args.config.resolve()
    with config_path.open("rb") as stream:
        config = tomllib.load(stream)
    output = (config_path.parent / config["output"]["directory"]).resolve()
    claim_path = output / config["output"]["launch_claim"]
    failure_path = output / config["output"]["failure_marker"]
    actual_files = {path.name for path in output.iterdir() if path.is_file()}
    expected_files = {claim_path.name, failure_path.name}
    if actual_files != expected_files:
        raise RuntimeError("failed R006 output contains unexpected scientific artifacts")
    claim = json.loads(claim_path.read_text())
    failure = json.loads(failure_path.read_text())
    if sha256(config_path) != claim["config_sha256"]:
        raise RuntimeError("claim/config hash mismatch")
    if not claim["once_only"] or claim["rerun_permitted"]:
        raise RuntimeError("once-only claim is invalid")
    if failure != {
        "classification": "FAILED",
        "error_type": "RuntimeError",
        "error": "factual action replay differs from frozen R004 trace",
        "rerun_permitted": False,
    }:
        raise RuntimeError("failure marker changed")

    r004_config = (config_path.parent / config["frozen"]["r004_config"]).resolve()
    _, r001, _, checkpoint, _, _ = load_r004(r004_config)
    trace_path = (config_path.parent / config["frozen"]["r004_trace"]).resolve()
    attribution_path = (
        config_path.parent / config["frozen"]["r004_final_attribution"]
    ).resolve()
    if sha256(trace_path) != config["frozen"]["r004_trace_sha256"]:
        raise RuntimeError("R004 trace hash changed")
    with np.load(trace_path, allow_pickle=False) as archive:
        observations = archive["observation"]
        physical_action = archive["physical_action"]
    with np.load(attribution_path, allow_pickle=False) as archive:
        sample_index = archive["sample_index"]
    agent, _ = PPOAgent.load(
        checkpoint, device="cuda" if torch.cuda.is_available() else "cpu"
    )
    device = next(agent.model.parameters()).device
    tensor = torch.as_tensor(observations[sample_index], device=device)
    limits = PPOActionLimits(
        float(r001["frozen_policy"]["maximum_linear_velocity_mps"]),
        float(r001["frozen_policy"]["maximum_angular_velocity_rad_s"]),
    )
    with torch.no_grad():
        velocity = target_values(
            agent.model, tensor, target="v_cmd_mps", action_limits=limits
        ).cpu().numpy()
        yaw = target_values(
            agent.model, tensor, target="omega_cmd_rad_s", action_limits=limits
        ).cpu().numpy()
    replay = np.stack([velocity, yaw], axis=1)
    difference = np.abs(replay - physical_action[sample_index])
    maximum = float(np.max(difference))
    tolerance = float(config["gate"]["factual_action_replay_tolerance"])
    if not maximum > tolerance:
        raise RuntimeError("recorded R006 failure no longer reproduces")
    if maximum >= 5.0e-6:
        raise RuntimeError("replay divergence is larger than the documented near-miss")

    test_log = ROOT / "artifacts/f11_ppo_explanation_v2/r006_failed_full_tests.log"
    match = re.search(r"(\d+) passed, (\d+) warnings", test_log.read_text())
    if match is None or tuple(map(int, match.groups())) != (664, 426):
        raise RuntimeError("full-test transcript does not prove 664 passes")
    if (ROOT / "artifacts/f11_ppo_explanation_v2/r007").exists():
        raise RuntimeError("R007 must remain absent")
    print(
        json.dumps(
            {
                "classification": "FAILED",
                "verifier": "PASS",
                "maximum_replay_error": maximum,
                "frozen_tolerance": tolerance,
                "scientific_intervention_artifact_exists": False,
                "rerun_permitted": False,
                "r007_started": False,
                "full_tests": {"passed": 664, "failed": 0, "skipped": 0},
            },
            indent=2,
        )
    )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    main()
