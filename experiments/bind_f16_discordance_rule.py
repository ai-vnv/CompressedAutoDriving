#!/usr/bin/env python3
"""Hash-bind the sequence-discordance rule before any matched pair completes.

Fails closed if a matched Direct/Progressive pair is already complete at any width,
because the rule must precede the observation it governs.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))

from duckie_pomdp.optimization.cross_curriculum_recovery import file_sha256  # noqa: E402
from run_f15_cross_curriculum_recovery import (  # noqa: E402
    artifact_root, load_config, provenance, read_csv, read_json, write_json,
)

CONFIG = ROOT / "configs/f16_sequence_int8_recovery_v1.toml"
DOC = ROOT / "docs/F16_SEQUENCE_DISCORDANCE_RULE.md"
PROTOCOL = ROOT / "docs/F16_PROTOCOL.md"


def main() -> None:
    config = load_config(CONFIG)
    root = artifact_root(config, CONFIG)
    target = root / "integrity/sequence_discordance_rule.json"
    if target.exists():
        raise RuntimeError(f"refusing to overwrite {target}")

    manifest = read_json(root / "protocol_manifest.json")
    if file_sha256(PROTOCOL) != manifest["protocol_document_sha256"]:
        raise RuntimeError("F16 protocol changed since freeze")

    seeds = [int(s) for s in config["seeds"]["selection"]]
    curricula = list(config["frozen"]["curricula"])
    loop = root / "closed_loop"

    def complete(candidate: str) -> bool:
        path = loop / f"fp32_{candidate}_episodes.csv"
        if not path.exists():
            return False
        covered = {(r["curriculum"], int(r["seed"])) for r in read_csv(path)}
        return all((c, s) in covered for c in curricula for s in seeds)

    matched_complete = [
        w for w in (64, 96, 128) if complete(f"D{w}") and complete(f"P{w}")
    ]
    if matched_complete:
        raise RuntimeError(
            f"refusing to freeze the discordance rule after matched pairs completed: {matched_complete}"
        )

    status = {c: complete(c) for c in ("D64", "P64", "D96", "P96", "D128", "P128", "D192")}
    payload = {
        **provenance(config, CONFIG),
        "classification": "FROZEN",
        "document": str(DOC),
        "document_sha256": file_sha256(DOC),
        "protocol_document_sha256_unchanged": file_sha256(PROTOCOL),
        "protocol_modified": False,
        "written_before_any_matched_pair_completed": True,
        "candidate_completion_at_freeze": status,
        "matched_pairs_complete_at_freeze": matched_complete,
        "rule": (
            "If Direct and Progressive disagree in verdict at a matched width x curriculum, "
            "the sequence effect is PROVISIONAL and triggers confirmatory training-seed "
            "replication of that discordant cell only."
        ),
        "classification_after_replication": {
            "same_direction_across_training_seeds": "SEQUENCE EFFECT SUPPORTED",
            "direction_changes_between_training_seeds": "TRAINING-SEED SENSITIVE / INCONCLUSIVE",
            "direct_and_progressive_never_disagree": "NO MATERIAL SEQUENCE EFFECT DETECTED",
        },
        "cross_seed_transfer_check": {
            "purpose": "hold the model fixed while changing only evaluation block and backend",
            "checkpoint": str(ROOT / "artifacts/f15_cross_curriculum_recovery_v1/recovery/fp32/w64/actor_multicurriculum_kd_fp32.pt"),
            "checkpoint_sha256": config["frozen"]["f15_recovered_fp32_w64_sha256"],
            "evaluation_seeds": seeds,
            "is_candidate": False,
            "eligible_for_selection": False,
        },
        "confound_at_freeze": (
            "D64 failed C4 in FP32 while the F15 recovered 64x64 actor had passed. Three factors "
            "differ between them (distillation seed, evaluation seed block, evaluation backend), "
            "so no single-factor attribution is permitted from that observation alone."
        ),
    }
    write_json(target, payload)
    print(json.dumps({
        "bound": str(target),
        "document_sha256": payload["document_sha256"],
        "matched_pairs_complete_at_freeze": matched_complete,
        "candidate_completion_at_freeze": status,
    }, indent=2))


if __name__ == "__main__":
    main()
