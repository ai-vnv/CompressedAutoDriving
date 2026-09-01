"""Build belief-gated pedestrian correction with counterfactual retention."""

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


PED_SLICE = slice(10, 19)
PED_NEUTRAL = np.asarray((0.0, 3.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0), dtype=np.float32)


def build(
    *,
    source_dataset: Path,
    source_csv: Path,
    retained_checkpoint: Path,
    output: Path,
    manifest_path: Path,
    minimum_existence: float,
    maximum_normalized_range: float,
    device: str,
) -> dict[str, object]:
    if output.exists() or manifest_path.exists():
        raise FileExistsError(output if output.exists() else manifest_path)
    with np.load(source_dataset) as data:
        source_x = np.asarray(data["observations"], dtype=np.float32)
        teacher_y = np.asarray(data["actions"], dtype=np.float32)
        source_value_targets = np.asarray(data["value_targets"], dtype=np.float32)
        source_value_weights = np.asarray(data["value_weights"], dtype=np.float32)
    with source_csv.open(newline="", encoding="utf-8") as handle:
        roles = np.asarray([row["source_role"] for row in csv.DictReader(handle)])
    if len(roles) != len(source_x):
        raise RuntimeError("source CSV and dataset row counts differ")

    retained, payload = PPOAgent.load(retained_checkpoint, device=device)
    x_tensor = torch.as_tensor(source_x, device=retained.device)
    with torch.no_grad():
        retained_y = retained.model.actor(x_tensor).cpu().numpy().astype(np.float32)

    c2 = roles == "c2_source_policy_rehearsal"
    active_hazard = (
        c2
        & (source_x[:, 10] >= minimum_existence)
        & (source_x[:, 11] <= maximum_normalized_range)
    )
    c2_nonhazard = c2 & ~active_hazard
    c3 = roles == "c3_source_policy_rehearsal"
    c4 = np.char.startswith(roles.astype(str), "c4_")
    if not all(np.any(mask) for mask in (active_hazard, c2_nonhazard, c3, c4)):
        raise RuntimeError("belief-gated rehearsal is missing a required role")

    counterfactual_x = source_x[active_hazard].copy()
    counterfactual_x[:, PED_SLICE] = PED_NEUTRAL
    with torch.no_grad():
        counterfactual_y = retained.model.actor(
            torch.as_tensor(counterfactual_x, device=retained.device)
        ).cpu().numpy().astype(np.float32)

    groups = (
        ("c2_hazard_correction", source_x[active_hazard], teacher_y[active_hazard]),
        ("c2_neutral_counterfactual", counterfactual_x, counterfactual_y),
        ("c2_nonhazard_retention", source_x[c2_nonhazard], retained_y[c2_nonhazard]),
        ("c3_retention", source_x[c3], retained_y[c3]),
        ("c4_retention", source_x[c4], retained_y[c4]),
    )
    observations = np.concatenate([group[1] for group in groups])
    actions = np.concatenate([group[2] for group in groups])
    group_names = np.concatenate(
        [np.full(len(group[1]), group[0], dtype="U32") for group in groups]
    )
    counts = Counter(group_names.tolist())
    weights = np.empty(len(group_names), dtype=np.float32)
    for name, count in counts.items():
        weights[group_names == name] = len(group_names) / (len(counts) * count)

    # Value supervision remains exclusively the frozen privileged guided rows;
    # duplicated counterfactual rows do not receive invented value targets.
    value_targets = np.concatenate(
        (
            source_value_targets[active_hazard],
            np.zeros(len(counterfactual_x), dtype=np.float32),
            source_value_targets[c2_nonhazard],
            source_value_targets[c3],
            source_value_targets[c4],
        )
    )
    value_weights = np.concatenate(
        (
            source_value_weights[active_hazard],
            np.zeros(len(counterfactual_x), dtype=np.float32),
            source_value_weights[c2_nonhazard],
            source_value_weights[c3],
            source_value_weights[c4],
        )
    )
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
        "purpose": "public-belief-gated C2 correction with counterfactual retention",
        "rows": len(observations),
        "observation_dimension": observations.shape[1],
        "minimum_existence_probability": minimum_existence,
        "maximum_normalized_range": maximum_normalized_range,
        "maximum_range_m": 2.0 * maximum_normalized_range,
        "group_counts": dict(sorted(counts.items())),
        "group_weight_mass": {
            name: float(weights[group_names == name].sum()) for name in sorted(counts)
        },
        "neutral_pedestrian_vector": PED_NEUTRAL.tolist(),
        "source_dataset": str(source_dataset.resolve()),
        "source_dataset_sha256": file_sha256(source_dataset),
        "source_csv": str(source_csv.resolve()),
        "source_csv_sha256": file_sha256(source_csv),
        "retained_checkpoint": str(retained_checkpoint.resolve()),
        "retained_checkpoint_sha256": file_sha256(retained_checkpoint),
        "retained_checkpoint_stage": payload["stage"],
        "privileged_truth_stored_in_npz": False,
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
    parser.add_argument("--minimum-existence", type=float, default=0.4)
    parser.add_argument("--maximum-normalized-range", type=float, default=0.7)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    print(json.dumps(build(
        source_dataset=args.source_dataset.resolve(),
        source_csv=args.source_csv.resolve(),
        retained_checkpoint=args.retained_checkpoint.resolve(),
        output=args.output.resolve(),
        manifest_path=args.manifest.resolve(),
        minimum_existence=args.minimum_existence,
        maximum_normalized_range=args.maximum_normalized_range,
        device=args.device,
    ), indent=2))


if __name__ == "__main__":
    main()
