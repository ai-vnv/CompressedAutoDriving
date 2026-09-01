"""Gate F9c leakage tests.

These tests scan the frozen runtime belief modules for any *code* reference
(never prose) to privileged/ground-truth data, evaluation-package imports,
or hardcoded frozen test seeds, and check that the final evaluator steps
both belief systems before reading privileged truth. Any failure here is a
real leak in the source module -- never loosen these tests to make them
pass.

**Fix round 1 (coordinator-directed):** the original version of
``test_no_runtime_module_references_privileged_state`` and
``test_the_evaluator_steps_both_beliefs_before_reading_privileged_truth``
did a raw substring scan of the whole file, including comments and
docstrings. That tripped on three passages that *document* the
runtime/privileged boundary -- the central design property of this gate --
rather than violating it, and the fix at the time reworded those passages to
dodge the substring match. That was backwards: the passages were correct and
valuable, and the scan was too blunt to tell code from prose. Both scans are
now AST-based: they inspect identifiers, attribute names, string-literal
values, import targets, and (for the evaluator ordering check) actual
``Call`` nodes -- never raw source text -- and explicitly skip docstrings.
Comments never appear in the AST at all, so parsing already excludes them
for free. The three original passages have been restored verbatim.
"""

from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Task 13 note: the plan's original list of five modules has grown by one
# (bias_correction.py) since the runtime coordinator now applies the frozen
# F9c bias stage before association; it is exactly as privileged-blind as
# the other five and must be covered by the same scan.
RUNTIME_MODULES = [
    "src/duckie_pomdp/belief/innovation_gate.py",
    "src/duckie_pomdp/belief/bias_correction.py",
    "src/duckie_pomdp/belief/measurement_association.py",
    "src/duckie_pomdp/belief/covariance_calibration.py",
    "src/duckie_pomdp/belief/observability.py",
    "src/duckie_pomdp/belief/robust_updater.py",
]

FORBIDDEN = (
    "privileged",
    "PrivilegedState",
    "true_pomdp_state",
    "sample_object_silhouettes",
    "eligible_visible",
    "gt_range_m",
    "gt_bearing_rad",
    "selected_iou",
    "intersection_over_union",
)

FROZEN_TEST_SEEDS = ("5101", "5102", "5103", "5104")


def _docstring_constant_ids(tree: ast.AST) -> set[int]:
    """Return ``id()`` of every ``ast.Constant`` node that is a docstring --
    the first statement of the module or of a function/class body, when that
    statement is a bare string expression. Those constants are prose, not
    code, and must be excluded from the leak scan."""
    docstring_ids: set[int] = set()
    holders: list[ast.AST] = [tree]
    holders.extend(
        node
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
    )
    for holder in holders:
        body = getattr(holder, "body", None)
        if not body:
            continue
        first = body[0]
        if (
            isinstance(first, ast.Expr)
            and isinstance(first.value, ast.Constant)
            and isinstance(first.value.value, str)
        ):
            docstring_ids.add(id(first.value))
    return docstring_ids


def _forbidden_token_in_code(source: str, forbidden: tuple[str, ...]) -> str | None:
    """Scan only *code* for any token in ``forbidden``: identifiers
    (``ast.Name``), attribute names (``ast.Attribute``), import targets
    (``ast.Import``/``ast.ImportFrom``), and string-literal values that are
    not docstrings. Comments are never present in the AST; docstrings are
    explicitly excluded. Returns the first matching token, or ``None``."""
    tree = ast.parse(source)
    docstring_ids = _docstring_constant_ids(tree)
    for node in ast.walk(tree):
        haystacks: list[str] = []
        if isinstance(node, ast.Name):
            haystacks = [node.id]
        elif isinstance(node, ast.Attribute):
            haystacks = [node.attr]
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            haystacks = [getattr(node, "module", None) or ""]
            haystacks += [alias.name for alias in node.names]
            haystacks += [alias.asname or "" for alias in node.names]
        elif (
            isinstance(node, ast.Constant)
            and isinstance(node.value, str)
            and id(node) not in docstring_ids
        ):
            haystacks = [node.value]
        else:
            continue
        for token in forbidden:
            if any(token in haystack for haystack in haystacks):
                return token
    return None


def _first_call_lineno(tree: ast.AST, attr_chain: tuple[str, ...]) -> int | None:
    """Line number of the first ``ast.Call`` whose callee resolves exactly
    to the given dotted attribute chain, e.g. ``("integration", "privileged",
    "read")`` for ``integration.privileged.read()``. Only real ``Call``
    nodes are inspected, so a docstring merely *quoting* that call text
    cannot match -- it is prose, not an ``ast.Call``."""
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        names: list[str] = []
        cur = node.func
        while isinstance(cur, ast.Attribute):
            names.append(cur.attr)
            cur = cur.value
        if isinstance(cur, ast.Name):
            names.append(cur.id)
        names.reverse()
        if tuple(names) == attr_chain:
            return node.lineno
    return None


def test_no_runtime_module_references_privileged_state():
    for relative in RUNTIME_MODULES:
        source = (ROOT / relative).read_text(encoding="utf-8")
        found = _forbidden_token_in_code(source, FORBIDDEN)
        assert found is None, f"{relative} references {found} in code (not prose)"


def test_no_runtime_module_imports_the_evaluation_package():
    for relative in RUNTIME_MODULES:
        tree = ast.parse((ROOT / relative).read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                names = [getattr(node, "module", None) or ""] + [
                    alias.name for alias in node.names
                ]
                assert not any("evaluation" in str(name) for name in names), relative


def test_the_leakage_scan_reads_code_not_prose():
    """A comment or docstring documenting the absence of privileged access
    must not trip the scan; an actual reference must. The runtime modules
    describe their own boundary, and that documentation has to survive the
    guard."""
    prose_only = '''
"""Module docstring mentioning privileged truth for documentation only."""

# Uses only the predicted state and calibrated camera geometry -- never
# privileged simulator truth or the detector's own bounding box.


def f():
    """Never reads privileged simulator truth; public inputs only."""
    return 1
'''
    assert _forbidden_token_in_code(prose_only, ("privileged",)) is None

    real_reference = """
def f(observation):
    return observation.privileged.read()
"""
    assert _forbidden_token_in_code(real_reference, ("privileged",)) == "privileged"


def test_the_evaluator_steps_both_beliefs_before_reading_privileged_truth():
    source = (ROOT / "experiments" / "evaluate_f9c_robust_belief.py").read_text(
        encoding="utf-8"
    )
    tree = ast.parse(source)
    privileged_at = _first_call_lineno(tree, ("integration", "privileged", "read"))
    baseline_at = _first_call_lineno(tree, ("baseline_updater", "update"))
    robust_at = _first_call_lineno(tree, ("robust_updater", "update"))
    assert privileged_at is not None, "no integration.privileged.read() call found"
    assert baseline_at is not None, "no baseline_updater.update(...) call found"
    assert robust_at is not None, "no robust_updater.update(...) call found"
    assert baseline_at < privileged_at
    assert robust_at < privileged_at


def test_f9b_frozen_artifacts_are_untouched():
    from duckie_pomdp.evaluation.f9_protocol import sha256

    assert sha256(ROOT / "artifacts" / "f9_measurement_model.json") == (
        "eb09ea6c64b6cbf3306057092e254a0e049776b38581e5b873a8ef9e2e91b278"
    )


def test_f9c_source_never_names_a_frozen_test_seed():
    for path in (ROOT / "src").rglob("*.py"):
        source = path.read_text(encoding="utf-8")
        for seed in FROZEN_TEST_SEEDS:
            assert seed not in source, f"{path} hardcodes frozen test seed {seed}"
