"""Build and immediately verify the fail-closed F10-PPO C0 launch gate."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from duckie_pomdp.control.f10_protocol import file_sha256
from duckie_pomdp.control.ppo_protocol import (
    load_ppo_curriculum_protocol,
    pretraining_evidence_paths,
    pretraining_source_paths,
    protocol_artifact_root,
    require_pretraining_gate,
)


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "f10_ppo_v1.toml")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    config = args.config.resolve()
    protocol = load_ppo_curriculum_protocol(config)
    output = (
        args.output.resolve()
        if args.output is not None
        else protocol_artifact_root(protocol) / "pretraining_gate.json"
    )
    if output.exists():
        raise FileExistsError(f"refusing to overwrite pretraining gate: {output}")
    object_curriculum = bool(
        protocol.raw.get("object_curriculum", {}).get("objects_separated") is True
    )
    payload = {
        "schema_version": 1,
        "gate": (
            "F10_PPO_OBJECT_CURRICULUM_PRETRAINING"
            if object_curriculum
            else "F10_PPO_C0_PRETRAINING"
        ),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "ready_for_training": True,
        "ready_for_c0_training": not object_curriculum,
        "config_sha256": file_sha256(config),
        "frozen_sources": {
            path: file_sha256(ROOT / path)
            for path in pretraining_source_paths(protocol)
        },
        "evidence": {
            path: file_sha256(ROOT / path)
            for path in pretraining_evidence_paths(protocol)
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    require_pretraining_gate(protocol, output)
    print(json.dumps({
        "ready_for_training": True,
        "path": str(output),
        "sha256": file_sha256(output),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
