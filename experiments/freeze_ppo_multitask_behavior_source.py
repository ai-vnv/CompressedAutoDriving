"""Freeze a balanced, policy-visible C1/C2 distillation source snapshot."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from duckie_pomdp.control.f10_protocol import file_sha256
from duckie_pomdp.control.ppo_protocol import load_ppo_curriculum_protocol


def _read_policy_rows(
    source: Path,
    columns: tuple[str, ...],
    count: int,
) -> list[dict[str, str]]:
    with source.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        if not set(columns).issubset(reader.fieldnames or ()):
            raise ValueError(f"policy columns are missing from {source}")
        rows = [{name: row[name] for name in columns} for row in reader]
    if len(rows) < count:
        raise ValueError(f"{source} contains only {len(rows)} rows; need {count}")
    indices = np.linspace(0, len(rows) - 1, count, dtype=np.int64)
    if len(np.unique(indices)) != count:
        raise RuntimeError("deterministic source sampling produced duplicates")
    return [rows[int(index)] for index in indices]


def freeze(
    config: Path,
    anchor_csv: Path,
    hazard_csv: Path,
    output: Path,
    rows_per_source: int,
) -> dict[str, object]:
    protocol = load_ppo_curriculum_protocol(config)
    columns = tuple(f"policy.{name}" for name in protocol.observation_order)
    anchor_rows = _read_policy_rows(anchor_csv, columns, rows_per_source)
    hazard_rows = _read_policy_rows(hazard_csv, columns, rows_per_source)
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=("source_role", *columns))
        writer.writeheader()
        writer.writerows(
            {"source_role": "c1_anchor", **row} for row in anchor_rows
        )
        writer.writerows(
            {"source_role": "c2_hazard", **row} for row in hazard_rows
        )
    return {
        "schema_version": 1,
        "rows": 2 * rows_per_source,
        "rows_per_source": rows_per_source,
        "observation_dimension": len(protocol.observation_order),
        "contains_evaluation_gt": False,
        "sampling": "deterministic_evenly_spaced_without_replacement",
        "anchor_source": str(anchor_csv.resolve()),
        "anchor_source_sha256": file_sha256(anchor_csv),
        "hazard_source": str(hazard_csv.resolve()),
        "hazard_source_sha256": file_sha256(hazard_csv),
        "snapshot": str(output.resolve()),
        "snapshot_sha256": file_sha256(output),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--anchor-csv", type=Path, required=True)
    parser.add_argument("--hazard-csv", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--rows-per-source", type=int, default=25_600)
    args = parser.parse_args()
    manifest = args.manifest.resolve()
    if manifest.exists():
        raise FileExistsError(manifest)
    payload = freeze(
        args.config.resolve(),
        args.anchor_csv.resolve(),
        args.hazard_csv.resolve(),
        args.output.resolve(),
        args.rows_per_source,
    )
    manifest.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
