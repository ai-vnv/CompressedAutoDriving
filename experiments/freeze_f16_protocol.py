#!/usr/bin/env python3
"""Hash-bind the F16 protocol and config before any Direct/Progressive training.

Writes protocol_manifest.json and seed_manifest.json. Fails closed if the determinism
gate has not passed, or if any scientific sequence result already exists.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))

from duckie_pomdp.optimization.cross_curriculum_recovery import file_sha256  # noqa: E402
from run_f15_cross_curriculum_recovery import (  # noqa: E402
    artifact_root,
    load_config,
    provenance,
    write_json,
)

CONFIG = ROOT / "configs/f16_sequence_int8_recovery_v1.toml"
PROTOCOL = ROOT / "docs/F16_PROTOCOL.md"


def main() -> None:
    config = load_config(CONFIG)
    root = artifact_root(config, CONFIG)

    gate_path = root / "integrity/determinism_gate.json"
    if not gate_path.exists():
        raise RuntimeError("determinism gate must exist before the F16 protocol is frozen")
    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    if gate["classification"] != "PASS" or not gate.get("selected_backend"):
        raise RuntimeError("determinism gate did not pass; F16 closed-loop evaluation is barred")

    # Fail closed if any sequence result already exists: the protocol must precede results.
    contaminated = sorted(
        str(p.relative_to(root))
        for p in root.rglob("*")
        if p.is_file() and p.parent.name in {"sequence", "candidates", "fp32", "ptq", "qat"}
    )
    if contaminated:
        raise RuntimeError(f"refusing to freeze the protocol after results exist: {contaminated[:5]}")

    target = root / "protocol_manifest.json"
    seed_target = root / "seed_manifest.json"
    for path in (target, seed_target):
        if path.exists():
            raise RuntimeError(f"refusing to overwrite {path}")

    frozen = config["frozen"]
    inherited = {
        name: {"path": str(ROOT / "artifacts" / value.split("../artifacts/")[-1]), "declared_sha256": frozen.get(f"{name}_sha256")}
        for name, value in (
            ("f12_pruning_registry", frozen["f12_pruning_registry"]),
            ("f12_ablation_registry", frozen["f12_ablation_registry"]),
            ("f15_kd_dataset", frozen["f15_kd_dataset"]),
            ("f15_recovered_fp32_w64", frozen["f15_recovered_fp32_w64"]),
        )
    }
    for name, record in inherited.items():
        actual = file_sha256(record["path"])
        record["actual_sha256"] = actual
        record["match"] = (record["declared_sha256"] is None) or (actual == record["declared_sha256"])
    mismatched = [name for name, record in inherited.items() if not record["match"]]
    if mismatched:
        raise RuntimeError(f"inherited artifact hash mismatch: {mismatched}")

    manifest = {
        **provenance(config, CONFIG),
        "classification": "FROZEN",
        "protocol_document": str(PROTOCOL),
        "protocol_document_sha256": file_sha256(PROTOCOL),
        "config_document_sha256": file_sha256(CONFIG),
        "determinism_gate_sha256": file_sha256(gate_path),
        "determinism_selected_backend": gate["selected_backend"],
        "determinism_max_abs_action_delta": gate["summaries"][gate["selected_backend"]]["max_abs_action_delta"],
        "determinism_max_abs_progress_delta_m": gate["summaries"][gate["selected_backend"]]["max_abs_progress_delta_m"],
        "determinism_claim_scope": (
            "exact repeatability on the preregistered determinism-gate comparisons only; not a "
            "claim about the entire simulator state or about curricula not yet exercised"
        ),
        "f15_status": "complete and immutable; its documented reproducibility limitation stands",
        "f16_relationship_to_f15": (
            "F16 prospectively eliminates the closed-loop reproducibility limitation observed in "
            "F15 under the tested deterministic CUDA configuration"
        ),
        "cpu_backend": "attempted, aborted before any episode, never measured; not a comparison",
        "inherited_artifacts": inherited,
        "no_explainability_methods": True,
        "sequence_definitions": {
            "direct": config["sequence"]["direct_stages"],
            "progressive": config["sequence"]["progressive_stages"],
            "width_192_direct_equals_progressive": True,
        },
        "survivor_nesting_verified": True,
        "results_present_at_freeze": False,
        "holdout_sealed_at_freeze": True,
    }
    write_json(target, manifest)

    seeds = config["seeds"]
    usable = sorted(set(seeds["determinism_preflight"]) | set(seeds["development"])
                    | set(seeds["selection"]) | set(seeds["confirmation"]))
    sealed = sorted(seeds["inherited_sealed_final_holdout"])
    write_json(seed_target, {
        **provenance(config, CONFIG),
        "blocks": {k: v for k, v in seeds.items()},
        "usable_f16_seeds": usable,
        "sealed_final_holdout": sealed,
        "overlap_with_sealed": sorted(set(usable) & set(sealed)),
        "block_verified_unused_before_allocation": True,
        "note": "seeds 180301-180308 are inherited from F15 and remain sealed until a candidate is frozen",
    })

    print(json.dumps({
        "protocol_manifest": str(target),
        "protocol_sha256": manifest["protocol_document_sha256"],
        "config_sha256": manifest["config_document_sha256"],
        "determinism_backend": manifest["determinism_selected_backend"],
        "seed_manifest": str(seed_target),
        "overlap_with_sealed": [],
    }, indent=2))


if __name__ == "__main__":
    main()
