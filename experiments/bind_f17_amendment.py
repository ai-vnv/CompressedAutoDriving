#!/usr/bin/env python3
"""Hash-bind the F17 comparison-interpretation amendment before any comparison is read."""
from __future__ import annotations
import json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))
from duckie_pomdp.optimization.cross_curriculum_recovery import file_sha256
from run_f15_cross_curriculum_recovery import load_config, provenance, read_json, write_json

CONFIG = ROOT / "configs/f17_optimization_method_order_v1.toml"
PROTOCOL = ROOT / "docs/F17_PROTOCOL.md"
DOC = ROOT / "docs/F17_COMPARISON_INTERPRETATION_AMENDMENT.md"

def main() -> None:
    config = load_config(CONFIG)
    root = (CONFIG.parent / config["artifacts"]["directory"]).resolve()
    target = root / "integrity" / "comparison_interpretation_amendment.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        raise RuntimeError(f"refusing to overwrite {target}")
    manifest = read_json(root / "protocol_manifest.json")
    if file_sha256(PROTOCOL) != manifest["protocol_document_sha256"]:
        raise RuntimeError("F17 protocol changed since freeze")
    if file_sha256(CONFIG) != manifest["config_document_sha256"]:
        raise RuntimeError("F17 config changed since freeze")
    results = root / "results"
    existing = sorted(p.name for p in results.glob("*")) if results.exists() else []
    if existing:
        raise RuntimeError(f"refusing to bind after comparisons exist: {existing}")
    payload = {
        **provenance(config, CONFIG),
        "classification": "FROZEN",
        "kind": "comparison_interpretation_amendment",
        "document": str(DOC), "document_sha256": file_sha256(DOC),
        "protocol_sha256_unchanged": file_sha256(PROTOCOL),
        "config_sha256_unchanged": file_sha256(CONFIG),
        "protocol_modified": False,
        "written_before_any_comparison_interpreted": True,
        "results_present_at_binding": existing,
        "clarifications": {
            "A5_vs_A6": {
                "licensed_question": ("Does inserting balanced distillation before PTQ preserve more "
                                      "cross-curriculum competence than quantizing the pruned actor directly?"),
                "what_actually_differs": "presence of a balanced-KD stage before quantization, not ordering alone",
                "forbidden": ["complete factorial proof of optimization order",
                              "general law that optimization order matters"],
            },
            "A6_vs_A8": {
                "licensed_question": ("Can a QAT+KD quantization route preserve or restore cross-curriculum "
                                      "retention relative to the PTQ route?"),
                "structural_fact": "A8 branches from the A3 FP32 anchor; it does not retrain the A6 INT8 graph",
                "forbidden": ["QAT repaired the failed PTQ model",
                              "QAT fixes quantization damage in general"],
            },
            "A3_vs_A6": {
                "licensed_question": ("Does PTQ introduce a new PASS->FAIL for this fixed recovered FP32 "
                                      "checkpoint on the identical deterministic block?"),
                "strength": "parent is a single fixed checkpoint already verified PASS on the same block; "
                            "no training-realization variability enters the statement",
            },
        },
    }
    write_json(target, payload)
    print(json.dumps({"bound": str(target), "document_sha256": payload["document_sha256"],
                      "protocol_unchanged": True, "results_present": existing}, indent=2))

if __name__ == "__main__":
    main()
