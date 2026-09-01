from pathlib import Path

from duckie_pomdp.evaluation.f9d_protocol import load_f9d_protocol

import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))

from verify_f9d_artifacts import VerificationResults, verify_required_artifacts  # noqa: E402


def test_frozen_f9d_protocol_loads_after_task5():
    protocol = load_f9d_protocol(
        ROOT / "configs" / "f9d_evidence_closure_v1.toml",
        require_frozen=True,
    )
    assert protocol.parameters_frozen is True
    assert protocol.config_sha256 == (
        "7bbe6525c24e294b55a46808301249633236658814e906a68d0d804d5e8a8ca6"
    )


def test_final_verifier_fails_when_a_required_artifact_is_absent(tmp_path):
    results = VerificationResults()
    verify_required_artifacts(
        results,
        {"deliberately_missing": tmp_path / "missing.json"},
        final=True,
    )
    assert results.failed is True
    assert results.items == [
        {
            "check": "deliberately_missing",
            "status": "FAIL",
            "message": f"required final artifact missing: {tmp_path / 'missing.json'}",
        }
    ]


def test_development_verifier_skips_a_future_artifact(tmp_path):
    results = VerificationResults()
    verify_required_artifacts(
        results,
        {"future": tmp_path / "future.json"},
        final=False,
    )
    assert results.failed is False
    assert results.items[0]["status"] == "SKIP"
