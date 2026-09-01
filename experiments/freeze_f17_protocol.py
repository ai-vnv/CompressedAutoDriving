#!/usr/bin/env python3
"""Freeze the F17 protocol and audit every pathway checkpoint before any run.

Writes protocol_manifest.json, seed_manifest.json and pathway_registry.json. Fails closed
if the inherited determinism gate did not pass, if the anchor hash does not match, or if
any F17 scientific result already exists.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))

from duckie_pomdp.optimization.cross_curriculum_recovery import file_sha256  # noqa: E402
from run_f15_cross_curriculum_recovery import (  # noqa: E402
    load_config, provenance, read_json, write_json,
)

CONFIG = ROOT / "configs/f17_optimization_method_order_v1.toml"
PROTOCOL = ROOT / "docs/F17_PROTOCOL.md"


def resolve(value: str) -> Path:
    return (CONFIG.parent / value).resolve()


def main() -> None:
    config = load_config(CONFIG)
    root = resolve(config["artifacts"]["directory"])
    root.mkdir(parents=True, exist_ok=True)

    target = root / "protocol_manifest.json"
    if target.exists():
        raise RuntimeError(f"refusing to overwrite {target}")

    # Inherited determinism gate must have passed.
    gate_path = resolve(config["determinism"]["inherited_gate"])
    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    if gate["classification"] != "PASS":
        raise RuntimeError("inherited determinism gate did not pass; F17 evaluation is barred")

    # No F17 scientific result may exist yet.
    existing = sorted(str(p.relative_to(root)) for p in root.rglob("*")
                      if p.is_file() and p.suffix in {".csv", ".npz", ".mp4"})
    if existing:
        raise RuntimeError(f"refusing to freeze after results exist: {existing[:5]}")

    # Audit the anchor.
    anchor = config["anchor"]
    anchor_path = resolve(anchor["checkpoint"])
    anchor_actual = file_sha256(anchor_path)
    if anchor_actual != anchor["sha256"]:
        raise RuntimeError(f"anchor hash mismatch: {anchor_actual} != {anchor['sha256']}")

    verification = resolve(anchor["verification_source"])
    if not verification.exists():
        raise RuntimeError("anchor verification episodes are missing")
    verification_episodes = len(verification.read_text(encoding="utf-8").splitlines()) - 1

    # Audit every pathway checkpoint.
    registry = {}
    missing = []
    for pid, spec in sorted(config["pathways"].items()):
        path = resolve(spec["checkpoint"])
        if not path.exists():
            missing.append(f"{pid}: {path}")
            continue
        registry[pid] = {
            "pathway_id": pid,
            "label": spec["label"],
            "optimization_method_order": spec["pathway"],
            "checkpoint": str(path),
            "sha256": file_sha256(path),
            "bytes": path.stat().st_size,
            "width": int(spec["width"]),
            "precision": spec["precision"],
            "int8": spec["precision"] == "INT8",
            "pruning_schedule": "Direct" if int(spec["width"]) != 256 else "n/a",
            "role": spec.get("role", "pathway"),
            "is_anchor": bool(spec.get("is_anchor", False)),
            "parent": spec.get("parent"),
            "retrained_by_f17": False,
        }
    if missing:
        raise RuntimeError(f"pathway checkpoints missing: {missing}")

    seeds = config["seeds"]
    overlap = set(seeds["primary_evaluation"]) & set(seeds["sealed_final_holdout"])
    if overlap:
        raise RuntimeError(f"primary block overlaps the sealed holdout: {sorted(overlap)}")

    manifest = {
        **provenance(config, CONFIG),
        "classification": "FROZEN",
        "protocol_document": str(PROTOCOL),
        "protocol_document_sha256": file_sha256(PROTOCOL),
        "config_document_sha256": file_sha256(CONFIG),
        "primary_question": config["scope"]["primary_question"],
        "manipulated_factor": config["scope"]["manipulated_factor"],
        "held_fixed": list(config["scope"]["held_fixed"]),
        "not_studied_here": list(config["scope"]["not_studied_here"]),
        "terminology": {
            "actor_width": "64 / 96 / 128 / 192",
            "pruning_schedule": "Direct / Progressive — F16 subject, held fixed in F17",
            "optimization_method_order": "placement/order of pruning, distillation, PTQ, QAT — the F17 subject",
        },
        "inherited_determinism_gate": str(gate_path),
        "inherited_determinism_gate_sha256": file_sha256(gate_path),
        "determinism_backend": gate["selected_backend"],
        "anchor": {
            "checkpoint": str(anchor_path),
            "sha256_declared": anchor["sha256"],
            "sha256_actual": anchor_actual,
            "match": True,
            "width": int(anchor["width"]),
            "pruning_schedule": anchor["pruning_schedule"],
            "retrained": False,
            "modified": False,
            "already_verified_on_primary_block": True,
            "verification_episodes": verification_episodes,
            "verification_result": anchor["verification_result"],
        },
        "pathway_count": len(registry),
        "all_pathway_checkpoints_pre_existing": True,
        "new_training_in_primary_comparison": False,
        "no_width_sweep": True,
        "no_pruning_schedule_replication": True,
        "no_training_seed_matrix": True,
        "primary_evaluation_seeds": [int(s) for s in seeds["primary_evaluation"]],
        "sealed_final_holdout": [int(s) for s in seeds["sealed_final_holdout"]],
        "sealed_holdout_overlap": [],
        "results_present_at_freeze": False,
    }
    write_json(target, manifest)
    write_json(root / "pathway_registry.json", {**provenance(config, CONFIG), "pathways": registry})
    write_json(root / "seed_manifest.json", {
        **provenance(config, CONFIG),
        "primary_evaluation": manifest["primary_evaluation_seeds"],
        "sealed_final_holdout": manifest["sealed_final_holdout"],
        "same_block_for_every_branch": True,
        "note": "primary block is the already-opened F15 recovery-selection block; the sealed holdout stays closed",
    })

    print(json.dumps({
        "protocol_sha256": manifest["protocol_document_sha256"],
        "config_sha256": manifest["config_document_sha256"],
        "anchor_verified": manifest["anchor"]["match"],
        "anchor_verification_episodes": verification_episodes,
        "pathways": sorted(registry),
        "new_training_in_primary_comparison": False,
        "sealed_holdout_overlap": [],
    }, indent=2))
    print()
    print(f"{'id':<4}{'width':>6}{'prec':>6}  {'sha256':<14} pathway")
    for pid, entry in sorted(registry.items()):
        print(f"{pid:<4}{entry['width']:>6}{entry['precision']:>6}  "
              f"{entry['sha256'][:12]:<14} {entry['optimization_method_order']}")


if __name__ == "__main__":
    main()
