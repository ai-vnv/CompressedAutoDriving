"""Build C1-retention/C2-hazard actor distillation data without privileged truth."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

from duckie_pomdp.control.f10_protocol import file_sha256
from duckie_pomdp.control.ppo import PPOAgent
from duckie_pomdp.control.ppo_protocol import load_ppo_curriculum_protocol
from duckie_pomdp.evaluation.f10_ppo_policy import BeliefAwareSimpleController


def build(
    config: Path,
    source_csv: Path,
    source_checkpoint: Path,
    output: Path,
    manifest: Path,
) -> dict[str, object]:
    protocol = load_ppo_curriculum_protocol(config)
    columns = tuple(f"policy.{name}" for name in protocol.observation_order)
    roles: list[str] = []
    observations: list[np.ndarray] = []
    with source_csv.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        required = {"source_role", *columns}
        if not required.issubset(reader.fieldnames or ()):
            raise ValueError("multitask warm-start source is incomplete")
        for row in reader:
            roles.append(row["source_role"])
            physical = np.asarray([float(row[name]) for name in columns], dtype=np.float32)
            normalized = np.clip(
                physical / np.asarray(protocol.observation_scales, dtype=np.float32),
                -protocol.observation_clip,
                protocol.observation_clip,
            ).astype(np.float32)
            if not np.all(np.isfinite(normalized)):
                raise ValueError("multitask source contains non-finite policy values")
            observations.append(normalized)

    x = np.asarray(observations, dtype=np.float32)
    roles_array = np.asarray(roles)
    expected_roles = {"c1_anchor", "c2_hazard"}
    if set(roles) != expected_roles:
        raise ValueError(f"expected source roles {expected_roles}, got {set(roles)}")
    anchor_mask = roles_array == "c1_anchor"
    hazard_mask = roles_array == "c2_hazard"

    anchor_agent, checkpoint_payload = PPOAgent.load(source_checkpoint, device="cpu")
    anchor_tensor = torch.as_tensor(x[anchor_mask], dtype=torch.float32)
    with torch.no_grad():
        anchor_actions = anchor_agent.model.actor(anchor_tensor).cpu().numpy()
    anchor_actions = np.clip(anchor_actions, -1.0, 1.0).astype(np.float32)

    controller = BeliefAwareSimpleController(protocol)
    hazard_actions = np.asarray(
        [controller.act(row) for row in x[hazard_mask]], dtype=np.float32
    )
    actions = np.empty((len(x), 2), dtype=np.float32)
    actions[anchor_mask] = anchor_actions
    actions[hazard_mask] = hazard_actions

    hazard_weights = np.where(hazard_actions[:, 0] <= -0.99, 8.0, 1.0).astype(
        np.float32
    )
    anchor_weight = float(np.sum(hazard_weights) / np.sum(anchor_mask))
    weights = np.empty(len(x), dtype=np.float32)
    weights[anchor_mask] = anchor_weight
    weights[hazard_mask] = hazard_weights

    if output.exists() or manifest.exists():
        raise FileExistsError("refusing to overwrite multitask warm-start artifacts")
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, observations=x, actions=actions, weights=weights)
    payload = {
        "schema_version": 1,
        "source_role": "balanced_c1_anchor_and_c2_hazard_policy_observations",
        "source_csv": str(source_csv.resolve()),
        "source_csv_sha256": file_sha256(source_csv),
        "source_snapshot_rows": int(len(x)),
        "source_config_sha256": file_sha256(config),
        "builder_sha256": file_sha256(Path(__file__)),
        "rows": int(len(x)),
        "anchor_rows": int(np.sum(anchor_mask)),
        "hazard_rows": int(np.sum(hazard_mask)),
        "observation_dimension": int(x.shape[1]),
        "action_dimension": 2,
        "hazard_stop_action_rows": int(np.sum(hazard_actions[:, 0] <= -0.99)),
        "anchor_weight": anchor_weight,
        "anchor_weight_mass": float(np.sum(weights[anchor_mask])),
        "hazard_weight_mass": float(np.sum(weights[hazard_mask])),
        "source_checkpoint": str(source_checkpoint.resolve()),
        "source_checkpoint_sha256": file_sha256(source_checkpoint),
        "source_checkpoint_stage": checkpoint_payload["stage"],
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
    parser.add_argument("--source-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build(
        args.config.resolve(), args.source_csv.resolve(),
        args.source_checkpoint.resolve(), args.output.resolve(),
        args.manifest.resolve(),
    ), indent=2))


if __name__ == "__main__":
    main()
