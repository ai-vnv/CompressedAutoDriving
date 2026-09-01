"""Integrity checks for the documentation-only F14 explained package."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
FIGURES = (
    "01_project_pipeline_from_rgb_to_action",
    "02_policy_input_29d_explained",
    "03_original_policy_semantic_attribution",
    "04_pruning_failure_and_distillation_recovery",
    "05_ptq_and_qat_explained",
    "06_pruning_plus_ptq_failure",
    "07_successful_deployment_pathway",
    "08_failure_mode_diagnostic_matrix",
    "09_final_original_vs_int8_shapley",
    "10_final_counterfactual_response",
    "11_glossary_for_non_ai_readers",
)
OUTPUT = (
    ROOT
    / "artifacts"
    / "f14_explainability_aware_compression_v1"
    / "figures_explained_id"
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_all_explained_figures_have_valid_png_and_pdf() -> None:
    for stem in FIGURES:
        png = OUTPUT / f"{stem}.png"
        pdf = OUTPUT / f"{stem}.pdf"
        assert png.is_file() and pdf.is_file()
        assert pdf.read_bytes().startswith(b"%PDF-")
        with Image.open(png) as image:
            assert image.width >= 3000
            assert image.height >= 1200
            dpi = image.info.get("dpi", (0.0, 0.0))
            assert min(dpi) >= 299.0


def test_figure_manifest_hashes_and_sources_are_exact() -> None:
    manifest = json.loads((OUTPUT / "figure_source_manifest.json").read_text())
    assert manifest["language"] == "English"
    assert set(manifest["figures"]) == set(FIGURES)
    for entry in manifest["figures"].values():
        for source in entry["source_data_files"]:
            assert (ROOT / source).exists(), source
        for output in entry["outputs"]:
            path = ROOT / output
            assert path.is_file()
            assert entry["figure_sha256"][path.name] == _sha256(path)


def test_indonesian_guide_has_all_33_numbered_sections() -> None:
    guide = (ROOT / "docs/PANDUAN_PROYEK_DUCKIE_POMDP_DARI_NOL.md").read_text()
    sections = [int(value) for value in re.findall(r"^## (\d+)\.", guide, re.M)]
    assert sections == list(range(1, 34))
    assert guide.count("| `lane_") >= 7
    assert "representasi semantik policy 29 dimensi yang dikondisikan oleh belief" in guide
    assert "F12 `PASS` untuk deployment C4-only" in guide
    assert "F13 `LIMITED`" in guide
    assert "F14 `LIMITED`" in guide
    assert "S_j=" in guide
    assert "distill_dense_actor()" in guide
    assert "normalized Smooth-L1 loss" in guide
    assert "qint8" in guide and "quint8" in guide
    assert "Bias" in guide and "phase-balanced" in guide
    assert "3,04× actor-only CPU speedup" in guide


def test_primary_counterfactual_denominator_is_not_the_attempt_denominator() -> None:
    audit = (ROOT / "docs/F14_VISUAL_CONSISTENCY_AUDIT.md").read_text()
    guide = (ROOT / "docs/PANDUAN_PROYEK_DUCKIE_POMDP_DARI_NOL.md").read_text()
    assert "Pruning + PTQ = 1/3 primary counterfactual tests preserved" in audit
    assert "bukan denominator primary yang valid" in audit
    assert "1/3 primary tests preserved" in guide
    amendment = json.loads(
        (
            ROOT
            / "artifacts/f14_explainability_aware_compression_v1"
            / "protocol_alignment_amendment.json"
        ).read_text()
    )
    assert "eight primary cells" in amendment["reason"]


def test_pre_f14_historical_documents_remain_hash_exact() -> None:
    witness = json.loads(
        (
            ROOT
            / "artifacts/f14_explainability_aware_compression_v1/integrity"
            / "historical_integrity_manifest.json"
        ).read_text()
    )
    for relative, expected in witness["files"].items():
        assert _sha256(ROOT / relative) == expected, relative


def test_frozen_f14_reports_remain_hash_exact() -> None:
    manifest = json.loads(
        (
            ROOT
            / "artifacts/f14_explainability_aware_compression_v1"
            / "artifact_manifest.json"
        ).read_text()
    )
    reports = {
        relative: metadata["sha256"]
        for relative, metadata in manifest["files"].items()
        if relative.startswith("docs/F14_")
    }
    assert reports
    for relative, expected in reports.items():
        assert _sha256(ROOT / relative) == expected, relative
