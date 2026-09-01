"""Build a metadata-only, turn-balanced lane RGB training dataset."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DYNAMIC = ROOT / "datasets" / "lane_rgb_dynamic_v2"
COMPETENCE = ROOT / "datasets" / "lane_rgb_competence_v9"
OUTPUT = ROOT / "datasets" / "lane_rgb_combined_v9"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dynamic", type=Path, default=DYNAMIC)
    parser.add_argument("--competence", type=Path, default=COMPETENCE)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    sources = (args.dynamic.resolve(), args.competence.resolve())
    output = args.output.resolve()
    metadata_path = output / "metadata.csv"
    manifest_path = output / "manifest.json"
    if metadata_path.exists() or manifest_path.exists():
        raise FileExistsError(f"refusing to overwrite combined dataset: {output}")

    rows: list[dict[str, str]] = []
    for source in sources:
        source_rows = list(
            csv.DictReader((source / "metadata.csv").open(encoding="utf-8"))
        )
        for row in source_rows:
            if source == sources[0] and row["split"] == "final":
                continue
            materialized = dict(row)
            materialized["source_dataset"] = source.name
            materialized["source_image_id"] = row["image_id"]
            materialized["image_id"] = f"{source.name}__{row['image_id']}"
            materialized["image_path"] = (
                Path("..") / source.name / row["image_path"]
            ).as_posix()
            materialized["turn_family"] = row.get("turn_family") or _turn_family(
                float(row["gt_curvature_inv_m"])
            )
            materialized["pose_name"] = row.get("pose_name") or "dynamic_trajectory"
            rows.append(materialized)

    training = [row for row in rows if row["split"] == "train"]
    non_training = [row for row in rows if row["split"] != "train"]
    groups: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in training:
        groups[row["turn_family"]].append(row)
    if set(groups) != {"left", "right", "straight"}:
        raise RuntimeError(f"incomplete turn families: {sorted(groups)}")
    target = max(len(values) for values in groups.values())
    balanced: list[dict[str, str]] = []
    for family in ("left", "right", "straight"):
        values = groups[family]
        for index in range(target):
            row = dict(values[index % len(values)])
            row["image_id"] = f"{row['image_id']}__balanced_{index:05d}"
            row["balance_repeat"] = str(index // len(values))
            balanced.append(row)
    combined = balanced + non_training
    output.mkdir(parents=True)
    fields = sorted({name for row in combined for name in row})
    with metadata_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(combined)

    source_manifests = {
        source.name: hashlib.sha256((source / "manifest.json").read_bytes()).hexdigest()
        for source in sources
    }
    physical_training_counts = Counter(row["turn_family"] for row in training)
    logical_training_counts = Counter(row["turn_family"] for row in balanced)
    split_source_seeds: dict[str, set[tuple[str, int]]] = defaultdict(set)
    for row in combined:
        split_source_seeds[row["split"]].add(
            (row["source_dataset"], int(row["seed"]))
        )
    overlaps = {
        f"{left}_{right}": sorted(
            f"{source}:{seed}"
            for source, seed in split_source_seeds[left] & split_source_seeds[right]
        )
        for index, left in enumerate(("train", "development", "final"))
        for right in ("train", "development", "final")[index + 1 :]
    }
    manifest = {
        "schema_version": 1,
        "dataset": output.name,
        "runtime_input": "front_rgb_only",
        "privileged_use": "offline labels inherited from source datasets",
        "sources": source_manifests,
        "dynamic_final_excluded": True,
        "horizontal_flip_forbidden": True,
        "training_balance_unit": "turn_family metadata repetition; image pixels unchanged",
        "physical_training_counts_by_turn": dict(physical_training_counts),
        "logical_training_counts_by_turn": dict(logical_training_counts),
        "counts": {
            split: sum(row["split"] == split for row in combined)
            for split in ("train", "development", "final")
        },
        "split_overlaps": overlaps,
        "metadata_sha256": hashlib.sha256(metadata_path.read_bytes()).hexdigest(),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


def _turn_family(curvature: float) -> str:
    if curvature < -0.75:
        return "right"
    if curvature > 0.75:
        return "left"
    return "straight"


if __name__ == "__main__":
    main()
