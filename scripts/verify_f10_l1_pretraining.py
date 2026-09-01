"""Produce the guarded F10-L1 pre-training readiness artifact."""

from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch

from duckie_pomdp.control import load_lane_protocol
from duckie_pomdp.control.f10_protocol import file_sha256


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "f10_l1_lane_v1.toml"
ARTIFACT_DIR = ROOT / "artifacts" / "f10_l1"
GATE_PATH = ARTIFACT_DIR / "pretraining_gate.json"
SOURCE_FILES = (
    "experiments/train_f10_l1_sac.py",
    "src/duckie_pomdp/control/lane_environment.py",
    "src/duckie_pomdp/control/lane_policy_observation.py",
    "src/duckie_pomdp/control/lane_protocol.py",
    "src/duckie_pomdp/control/lane_reward.py",
    "src/duckie_pomdp/control/sac.py",
)
FOCUSED_TESTS = (
    "tests/test_f10_l1_protocol.py",
    "tests/test_f10_l1_observation.py",
    "tests/test_f10_l1_reward.py",
    "tests/test_f10_l1_environment.py",
    "tests/test_f10_l1_evaluation.py",
)


def _load(path: Path) -> dict[str, object]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    if GATE_PATH.exists():
        raise FileExistsError(f"refusing to overwrite frozen gate: {GATE_PATH}")
    protocol = load_lane_protocol(CONFIG)
    config_sha = file_sha256(CONFIG)
    source_sha = {name: file_sha256(ROOT / name) for name in SOURCE_FILES}
    reward_audit_path = ARTIFACT_DIR / "reward_audit.json"
    smoke_manifest_path = ARTIFACT_DIR / "smoke" / "training_run_manifest.json"
    reward_audit = _load(reward_audit_path)
    smoke = _load(smoke_manifest_path)
    expected_project = "https://wandb.ai/vnv/DuckiePOMDP/runs/"
    checks = {
        "config_loads_frozen": protocol.raw["stage"] == "F10-L1",
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_matches": (
            torch.cuda.is_available()
            and "4060" in torch.cuda.get_device_name(0)
        ),
        "reward_audit_passed": reward_audit.get("passed") is True,
        "reward_audit_config_matches": reward_audit.get("config_sha256")
        == config_sha,
        "smoke_config_matches": smoke.get("config_sha256") == config_sha,
        "smoke_sources_match": smoke.get("source_sha256") == source_sha,
        "smoke_gradient_updates": int(smoke.get("gradient_updates", 0)) > 0,
        "smoke_checkpoint_reload": smoke.get("checkpoint_reload_verified") is True,
        "wandb_online": smoke.get("wandb", {}).get("mode") == "online",
        "wandb_destination_exact": str(
            smoke.get("wandb", {}).get("run_url") or ""
        ).startswith(expected_project),
    }
    checkpoint = Path(str(smoke.get("final_checkpoint", "")))
    checks["smoke_checkpoint_exists"] = checkpoint.is_file()
    checks["smoke_checkpoint_hash_matches"] = bool(
        checkpoint.is_file()
        and file_sha256(checkpoint) == smoke.get("final_checkpoint_sha256")
    )

    test_command = [
        sys.executable,
        "-m",
        "pytest",
        *FOCUSED_TESTS,
        "-q",
        "--disable-warnings",
    ]
    test_result = subprocess.run(
        test_command,
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    checks["focused_tests_pass"] = test_result.returncode == 0
    ready = all(checks.values())
    gate = {
        "schema_version": 1,
        "stage": "F10_L1_PRETRAINING_GATE",
        "verified_at_utc": datetime.now(timezone.utc).isoformat(),
        "ready_for_training": ready,
        "config": str(CONFIG.resolve()),
        "config_sha256": config_sha,
        "source_sha256": source_sha,
        "reward_audit": {
            "path": str(reward_audit_path.resolve()),
            "sha256": file_sha256(reward_audit_path),
        },
        "smoke_manifest": {
            "path": str(smoke_manifest_path.resolve()),
            "sha256": file_sha256(smoke_manifest_path),
        },
        "wandb": smoke.get("wandb"),
        "cuda": {
            "torch": torch.__version__,
            "runtime": torch.version.cuda,
            "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        },
        "focused_test_command": test_command,
        "focused_test_returncode": test_result.returncode,
        "focused_test_stdout": test_result.stdout.strip(),
        "focused_test_stderr": test_result.stderr.strip(),
        "checks": checks,
    }
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    GATE_PATH.write_text(json.dumps(gate, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(gate, indent=2))
    if not ready:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

