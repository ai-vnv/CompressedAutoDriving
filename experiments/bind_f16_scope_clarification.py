#!/usr/bin/env python3
"""Hash-bind the F16 quantization-scope clarification before any candidate is trained.

The frozen protocol is NOT modified. This records the clarification as an immutable
pre-training note, verifies that the protocol and config hashes are unchanged since the
freeze, and fails closed if any Direct/Progressive candidate already exists.
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
    read_json,
    write_json,
)

CONFIG = ROOT / "configs/f16_sequence_int8_recovery_v1.toml"
PROTOCOL = ROOT / "docs/F16_PROTOCOL.md"
CLARIFICATION = ROOT / "docs/F16_QUANTIZATION_SCOPE_LIMITATION.md"


def main() -> None:
    config = load_config(CONFIG)
    root = artifact_root(config, CONFIG)
    target = root / "integrity/quantization_scope_clarification.json"
    if target.exists():
        raise RuntimeError(f"refusing to overwrite {target}")

    manifest = read_json(root / "protocol_manifest.json")

    # The protocol must be byte-identical to what was frozen.
    protocol_now = file_sha256(PROTOCOL)
    config_now = file_sha256(CONFIG)
    if protocol_now != manifest["protocol_document_sha256"]:
        raise RuntimeError(
            f"F16 protocol changed since freeze: {manifest['protocol_document_sha256']} -> {protocol_now}"
        )
    if config_now != manifest["config_document_sha256"]:
        raise RuntimeError(
            f"F16 config changed since freeze: {manifest['config_document_sha256']} -> {config_now}"
        )

    # No scientific candidate may exist yet.
    existing = sorted(
        str(p.relative_to(root)) for p in root.rglob("*.pt")
        if "media_smoke" not in str(p)
    )
    if existing:
        raise RuntimeError(f"refusing to bind a pre-training note after candidates exist: {existing[:5]}")

    payload = {
        **provenance(config, CONFIG),
        "classification": "FROZEN",
        "kind": "pre_training_scope_clarification",
        "document": str(CLARIFICATION),
        "document_sha256": file_sha256(CLARIFICATION),
        "protocol_document_sha256_unchanged": protocol_now,
        "config_document_sha256_unchanged": config_now,
        "protocol_modified": False,
        "written_before_any_candidate_trained": True,
        "candidates_present_at_binding": existing,
        "varies": ["optimization_sequence", "target_width", "quantization_recovery_route"],
        "held_fixed": [
            "quantization representation (qint8 per-channel weights, quint8 per-tensor affine activations)",
            "observer choice", "calibration algorithm", "INT8 backend", "mixed-precision allocation",
        ],
        "cannot_attribute_failure_to": [
            "activation quantization granularity",
            "per-tensor versus per-channel activation scaling",
            "observer choice",
            "calibration algorithm",
            "quantization scale resolution",
            "mixed-precision allocation",
            "alternative INT8 backend behaviour",
            "weight-only versus weight-and-activation quantization",
        ],
        "f15_evidence_status": (
            "The F15 correlation/rank-order degradation motivates quantization representation "
            "as a future hypothesis, but does not identify activation granularity as its mechanism."
        ),
        "negative_result_is_valid": (
            "'Neither tested width nor optimization sequence recovered INT8 retention' is a clean "
            "controlled finding and is the condition that would justify a separately preregistered "
            "follow-up varying the quantization representation."
        ),
    }
    write_json(target, payload)
    print(json.dumps({
        "bound": str(target),
        "clarification_sha256": payload["document_sha256"],
        "protocol_unchanged": True,
        "config_unchanged": True,
        "candidates_present": existing,
    }, indent=2))


if __name__ == "__main__":
    main()
