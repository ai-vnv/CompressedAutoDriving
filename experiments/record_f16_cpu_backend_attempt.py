#!/usr/bin/env python3
"""Record the attempted cpu_deterministic backend run.

The frozen F16 determinism gate selected `cuda_strict_deterministic`, which reproduced
every preflight cell bit-exactly. The protocol treats CPU perception as a *fallback*,
required only if strict CUDA determinism is unsupported or still non-reproducible
(F16 spec section 5). It was not needed.

The fallback was nevertheless attempted, and it aborted for a configuration reason that
has nothing to do with determinism: hiding CUDA via CUDA_VISIBLE_DEVICES="" makes the
YOLO detector's explicitly requested `device=0` invalid, and ultralytics raises before
any episode runs. This file records that attempt so the gate's evidence base is not
overstated. It does not modify the frozen gate.
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


def main() -> None:
    config = load_config(CONFIG)
    root = artifact_root(config, CONFIG)
    gate_path = root / "integrity/determinism_gate.json"
    target = root / "integrity/determinism_cpu_backend_attempt.json"
    if target.exists():
        raise RuntimeError(f"refusing to overwrite {target}")
    gate = json.loads(gate_path.read_text(encoding="utf-8"))

    payload = {
        **provenance(config, CONFIG),
        "backend": "cpu_deterministic",
        "status": "ATTEMPTED_ABORTED_BEFORE_ANY_EPISODE",
        "episodes_measured": 0,
        "reason": (
            "CUDA_VISIBLE_DEVICES=\"\" makes torch.cuda.is_available() false. The lane "
            "MobileNet resolves device \"auto\" and would have fallen back to CPU cleanly, "
            "but the YOLO detector requests device=0 explicitly, so ultralytics "
            "select_device raised ValueError before the first environment reset."
        ),
        "error_type": "ValueError",
        "error_message": "Invalid CUDA 'device=0' requested. Use 'device=cpu' or pass valid CUDA device(s)",
        "raised_at": "ultralytics/utils/torch_utils.py select_device, via yolo_detector.detect",
        "determinism_relevance": (
            "none: the abort is a device-configuration incompatibility, not evidence about "
            "reproducibility. No determinism conclusion may be drawn from it in either direction."
        ),
        "why_not_repaired": (
            "F16 spec section 5 defines CPU perception as a fallback conditional on strict "
            "CUDA determinism being unsupported or non-reproducible. Strict CUDA determinism "
            "passed every frozen criterion bit-exactly, so the fallback was not required. "
            "Repairing it would have meant overriding the frozen detector device configuration "
            "for no scientific gain."
        ),
        "selected_backend": gate["selected_backend"],
        "selected_backend_reproducible": gate["summaries"][gate["selected_backend"]]["reproducible"],
        "determinism_gate": str(gate_path),
        "determinism_gate_sha256": file_sha256(gate_path),
        "evidence_base_caveat": (
            "The gate's backends_tested list contains one backend. The selection was therefore "
            "not a comparison between two measured options; it was the first option in the "
            "frozen fallback chain succeeding outright."
        ),
    }
    write_json(target, payload)
    print(json.dumps({
        "written": str(target),
        "status": payload["status"],
        "selected_backend": payload["selected_backend"],
    }, indent=2))


if __name__ == "__main__":
    main()
