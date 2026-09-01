from __future__ import annotations

import json
from pathlib import Path

import pytest

from duckie_pomdp.control import load_f10_protocol
from experiments.train_f10_sac import _require_pretraining_gate
from scripts.verify_f10_pretraining import _junit_summary, _secret_paths


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "f10_sac_v1.toml"


def test_frozen_wandb_destination_is_canonical_team_project() -> None:
    protocol = load_f10_protocol(CONFIG)
    assert protocol.raw["wandb"]["entity"] == "vnv"
    assert protocol.raw["wandb"]["project"] == "DuckiePOMDP"


def test_full_training_requires_ready_gate(tmp_path: Path) -> None:
    protocol = load_f10_protocol(CONFIG)
    with pytest.raises(RuntimeError, match="requires.*pretraining_gate"):
        _require_pretraining_gate(protocol, tmp_path)


def test_gate_rejects_changed_config_hash(tmp_path: Path) -> None:
    protocol = load_f10_protocol(CONFIG)
    (tmp_path / "pretraining_gate.json").write_text(
        json.dumps(
            {
                "ready_for_full_training": True,
                "f10_config_sha256": "wrong",
                "source_sha256": {},
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="config changed"):
        _require_pretraining_gate(protocol, tmp_path)


def test_junit_summary_counts_nested_suite(tmp_path: Path) -> None:
    report = tmp_path / "report.xml"
    report.write_text(
        '<testsuites><testsuite tests="7" failures="1" errors="0" skipped="2"/></testsuites>',
        encoding="utf-8",
    )
    assert _junit_summary(report) == {
        "tests": 7,
        "failures": 1,
        "errors": 0,
        "skipped": 2,
    }


def test_repository_contains_no_wandb_credential() -> None:
    assert _secret_paths(ROOT) == []
