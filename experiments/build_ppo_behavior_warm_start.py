"""Build an actor-imitation dataset from policy-visible training telemetry only."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from duckie_pomdp.control.f10_protocol import file_sha256
from duckie_pomdp.control.ppo_protocol import load_ppo_curriculum_protocol
from duckie_pomdp.evaluation.f10_ppo_policy import BeliefAwareSimpleController


def build(config: Path, source_csv: Path, output: Path, manifest: Path) -> dict:
    protocol = load_ppo_curriculum_protocol(config)
    columns = tuple(f"policy.{name}" for name in protocol.observation_order)
    observations: list[np.ndarray] = []
    with source_csv.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        if not set(columns).issubset(reader.fieldnames or ()):
            raise ValueError("warm-start source lacks the complete policy observation")
        for row in reader:
            physical = np.asarray([float(row[name]) for name in columns], dtype=np.float32)
            normalized = np.clip(
                physical / np.asarray(protocol.observation_scales, dtype=np.float32),
                -protocol.observation_clip,
                protocol.observation_clip,
            ).astype(np.float32)
            if not np.all(np.isfinite(normalized)):
                raise ValueError("warm-start source contains non-finite policy values")
            observations.append(normalized)
    x = np.asarray(observations, dtype=np.float32)
    controller = BeliefAwareSimpleController(protocol)
    y = np.asarray([controller.act(row) for row in x], dtype=np.float32)
    weights = np.where(y[:, 0] <= -0.99, 8.0, 1.0).astype(np.float32)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() or manifest.exists():
        raise FileExistsError("refusing to overwrite behavior warm-start artifacts")
    np.savez_compressed(output, observations=x, actions=y, weights=weights)
    payload = {
        "schema_version": 1,
        "source_role": "failed_c2_training_policy_observations_only",
        "source_csv": str(source_csv.resolve()),
        "source_csv_sha256": file_sha256(source_csv),
        "source_config_sha256": file_sha256(config),
        "builder_sha256": file_sha256(Path(__file__)),
        "rows": int(len(x)),
        "observation_dimension": int(x.shape[1]),
        "action_dimension": 2,
        "stop_action_rows": int(np.sum(y[:, 0] <= -0.99)),
        "uses_evaluation_gt": False,
        "dataset": str(output.resolve()),
        "dataset_sha256": file_sha256(output),
    }
    manifest.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--source-csv", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build(
        args.config.resolve(), args.source_csv.resolve(), args.output.resolve(),
        args.manifest.resolve(),
    ), indent=2))


if __name__ == "__main__":
    main()
