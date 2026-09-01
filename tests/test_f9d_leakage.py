"""Structural leakage and freeze guards for the complete F9d gate."""

from __future__ import annotations

import ast
import inspect
from pathlib import Path

from duckie_pomdp.evaluation.f9_protocol import sha256
from duckie_pomdp.perception.detector_dropout import (
    DetectorDropout,
    TargetRemovalScheduler,
)

ROOT = Path(__file__).resolve().parents[1]
F9D_CONFIG_HASH = "7bbe6525c24e294b55a46808301249633236658814e906a68d0d804d5e8a8ca6"
F9C_CONFIG_HASH = "359dc52020421c248bf6c26e036234191b2b97d24b505a66d85daa85b563704e"


def _call_lines(source: str, suffix: tuple[str, ...]) -> list[int]:
    lines: list[int] = []
    for node in ast.walk(ast.parse(source)):
        if not isinstance(node, ast.Call):
            continue
        names: list[str] = []
        current = node.func
        while isinstance(current, ast.Attribute):
            names.append(current.attr)
            current = current.value
        if isinstance(current, ast.Name):
            names.append(current.id)
        names.reverse()
        if tuple(names[-len(suffix) :]) == suffix:
            lines.append(node.lineno)
    return sorted(lines)


def _assigned_attributes(source: str) -> set[str]:
    names: set[str] = set()
    for node in ast.walk(ast.parse(source)):
        targets = []
        if isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
            targets = getattr(node, "targets", [getattr(node, "target", None)])
        for target in targets:
            if isinstance(target, ast.Attribute):
                names.add(target.attr)
    return names


def test_dropout_and_removal_schedules_cannot_consult_runtime_truth_or_belief():
    assert set(inspect.signature(DetectorDropout.schedule_for).parameters) == {
        "self",
        "seed",
        "episode_length",
    }
    assert set(inspect.signature(TargetRemovalScheduler.schedule_for).parameters) == {
        "self",
        "seed",
        "episode_length",
    }


def test_f9d_render_paths_step_both_systems_before_reading_privileged_truth():
    for relative in (
        "experiments/probe_f9d_yield.py",
        "experiments/probe_f9d_absence_yield.py",
    ):
        source = (ROOT / relative).read_text(encoding="utf-8")
        step_lines = _call_lines(source, ("_step_both_systems",))
        truth_lines = _call_lines(source, ("privileged", "read"))
        assert step_lines and truth_lines, relative
        assert min(step_lines) < min(truth_lines), relative


def test_f9d_modules_never_assign_a_frozen_estimator_attribute():
    forbidden = {
        "range_scale",
        "bearing_scale",
        "chi_square_threshold",
        "miss_likelihood_floor",
        "survival_probability",
        "birth_probability",
        "process_position_std",
        "process_velocity_std",
    }
    paths = [
        *(ROOT / "src" / "duckie_pomdp" / "evaluation").glob("f9d_*.py"),
        ROOT / "src" / "duckie_pomdp" / "perception" / "detector_dropout.py",
        *(ROOT / "experiments").glob("*f9d*.py"),
    ]
    for path in paths:
        assigned = _assigned_attributes(path.read_text(encoding="utf-8"))
        assert not (assigned & forbidden), f"{path}: {sorted(assigned & forbidden)}"


def test_frozen_f9d_and_f9c_config_hashes_are_unchanged():
    assert sha256(ROOT / "configs" / "f9d_evidence_closure_v1.toml") == F9D_CONFIG_HASH
    assert sha256(ROOT / "configs" / "f9c_robust_belief_v1.toml") == F9C_CONFIG_HASH


def test_f9d_source_never_hardcodes_final_seed_values():
    forbidden = tuple(str(value) for value in range(8201, 8205)) + tuple(
        str(value) for value in range(8301, 8305)
    )
    for path in (ROOT / "src").rglob("*.py"):
        source = path.read_text(encoding="utf-8")
        for seed in forbidden:
            assert seed not in source, f"{path} hardcodes final seed {seed}"


def test_runtime_dropout_module_does_not_import_privileged_or_evaluation_code():
    path = ROOT / "src" / "duckie_pomdp" / "perception" / "detector_dropout.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    imports = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            imports.extend(alias.name for alias in node.names)
            if isinstance(node, ast.ImportFrom) and node.module:
                imports.append(node.module)
    assert not any("privileged" in name or ".evaluation" in name for name in imports)
