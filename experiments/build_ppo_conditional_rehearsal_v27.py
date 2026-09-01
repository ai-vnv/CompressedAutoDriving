"""Build a leak-free conditional C2 correction / C3-C4 retention dataset."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from duckie_pomdp.control import PPOAgent
from duckie_pomdp.control.f10_protocol import file_sha256


def _task(role: str) -> str:
    if role.startswith("c2_"):
        return "c2_correction"
    if role.startswith("c3_"):
        return "c3_retention"
    if role.startswith("c4_"):
        return "c4_retention"
    raise ValueError(f"unsupported source role: {role}")


def build(
    *, source_dataset: Path,
    source_csv: Path,
    retained_checkpoint: Path,
    output: Path,
    manifest_path: Path,
    device: str,
) -> dict[str, object]:
    if output.exists() or manifest_path.exists():
        raise FileExistsError(output if output.exists() else manifest_path)
    with np.load(source_dataset) as data:
        observations = np.asarray(data["observations"], dtype=np.float32)
        teacher_actions = np.asarray(data["actions"], dtype=np.float32)
        value_targets = np.asarray(data["value_targets"], dtype=np.float32)
        value_weights = np.asarray(data["value_weights"], dtype=np.float32)
    with source_csv.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        roles = [row["source_role"] for row in reader]
    if len(roles) != len(observations):
        raise RuntimeError("source CSV and NPZ row counts differ")

    agent, payload = PPOAgent.load(retained_checkpoint, device=device)
    x = torch.as_tensor(observations, dtype=torch.float32, device=agent.device)
    with torch.no_grad():
        retained_actions = agent.model.actor(x).cpu().numpy().astype(np.float32)
    tasks = np.asarray([_task(role) for role in roles])
    actions = retained_actions.copy()
    c2_mask = tasks == "c2_correction"
    actions[c2_mask] = teacher_actions[c2_mask]

    counts = Counter(tasks.tolist())
    weights = np.empty(len(tasks), dtype=np.float32)
    for task, count in counts.items():
        # Equal total loss mass for C2 correction, C3 retention, and C4
        # retention regardless of their different trajectory lengths.
        weights[tasks == task] = len(tasks) / (len(counts) * count)
    if not np.isclose(float(weights.mean()), 1.0, atol=1.0e-6):
        raise RuntimeError("conditional rehearsal weights must have unit mean")

    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        observations=observations,
        actions=actions,
        weights=weights,
        value_targets=value_targets,
        value_weights=value_weights,
    )
    manifest = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "conditional C2 correction with exact C3/C4 actor retention",
        "rows": len(observations),
        "observation_dimension": observations.shape[1],
        "task_counts": dict(sorted(counts.items())),
        "task_weight_mass": {
            task: float(weights[tasks == task].sum()) for task in sorted(counts)
        },
        "c2_targets": "frozen source teacher actions",
        "c3_c4_targets": "frozen retained actor raw means",
        "source_dataset": str(source_dataset.resolve()),
        "source_dataset_sha256": file_sha256(source_dataset),
        "source_csv": str(source_csv.resolve()),
        "source_csv_sha256": file_sha256(source_csv),
        "retained_checkpoint": str(retained_checkpoint.resolve()),
        "retained_checkpoint_sha256": file_sha256(retained_checkpoint),
        "retained_checkpoint_stage": payload["stage"],
        "student_observation_uses_privileged_truth": False,
        "output": str(output.resolve()),
        "output_sha256": file_sha256(output),
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dataset", type=Path, required=True)
    parser.add_argument("--source-csv", type=Path, required=True)
    parser.add_argument("--retained-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    print(
        json.dumps(
            build(
                source_dataset=args.source_dataset.resolve(),
                source_csv=args.source_csv.resolve(),
                retained_checkpoint=args.retained_checkpoint.resolve(),
                output=args.output.resolve(),
                manifest_path=args.manifest.resolve(),
                device=args.device,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
