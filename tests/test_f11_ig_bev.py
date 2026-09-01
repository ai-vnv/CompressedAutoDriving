from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pytest

from duckie_pomdp.explain.ig_bev import (
    POSE_TRACE_KEYS,
    aggregate_groups,
    align_pose_to_samples,
    resolve_feature_groups,
    signed_total,
    validate_pose_trace,
)


ROOT = Path(__file__).resolve().parents[1]


def pose_trace() -> dict[str, np.ndarray]:
    return {
        "seed": np.asarray([7, 7, 7], dtype=np.int64),
        "step": np.asarray([0, 1, 2], dtype=np.int32),
        "world_x_m": np.asarray([0.1, 0.2, 0.3]),
        "world_z_m": np.asarray([1.0, 1.1, 1.2]),
        "heading_rad": np.asarray([0.0, 0.1, 0.2]),
    }


def test_pose_trace_is_explicitly_evaluation_only_and_exact() -> None:
    trace = pose_trace()
    validate_pose_trace(trace)
    assert tuple(trace) == POSE_TRACE_KEYS
    with pytest.raises(ValueError, match="schema mismatch"):
        validate_pose_trace({**trace, "observation": np.zeros(3)})


def test_pose_alignment_uses_exact_seed_and_step_keys() -> None:
    aligned = align_pose_to_samples(
        pose_trace(),
        sample_seed=np.asarray([7, 7]),
        sample_step=np.asarray([2, 0]),
    )
    np.testing.assert_allclose(aligned["world_x_m"], [0.3, 0.1])
    with pytest.raises(ValueError, match="no evaluation pose"):
        align_pose_to_samples(
            pose_trace(),
            sample_seed=np.asarray([7]),
            sample_step=np.asarray([9]),
        )


def test_group_aggregation_has_exact_coverage_and_conserves_magnitude() -> None:
    features = ("lane", "ped_r", "ped_beta", "previous")
    groups = resolve_feature_groups(
        features,
        {
            "lane": ("lane",),
            "pedestrian": ("ped_r", "ped_beta"),
            "action": ("previous",),
        },
    )
    attribution = np.asarray(
        [[1.0, -2.0, 3.0, 0.5], [0.0, 0.0, 0.0, 0.0]]
    )
    result = aggregate_groups(attribution, groups)
    np.testing.assert_allclose(result.absolute[0], [1.0, 5.0, 0.5])
    np.testing.assert_allclose(result.share[0].sum(), 1.0)
    np.testing.assert_allclose(result.share[1], 0.0)
    assert result.names[result.dominant_index[0]] == "pedestrian"
    with pytest.raises(ValueError, match="exactly once"):
        resolve_feature_groups(features, {"incomplete": ("lane",)})


def test_signed_total_matches_integrated_gradients_completeness_identity() -> None:
    attribution = np.asarray([[0.1, -0.2, 0.3], [-1.0, 0.25, 0.5]])
    np.testing.assert_allclose(signed_total(attribution), [0.2, -0.25])


def test_renderer_reads_world_pose_only_after_actor_action_and_never_feeds_it_back() -> None:
    source = (ROOT / "experiments" / "render_f11_ig_bev.py").read_text(
        encoding="utf-8"
    )
    tree = ast.parse(source)
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "replay_evaluation_pose"
    )
    joined = "\n".join(ast.unparse(node) for node in function.body)
    assert joined.index("agent.act") < joined.index("privileged.read")
    assert "environment.step(action)" in joined
    assert "environment.step(action, privileged" not in joined
    assert "integrated_gradients(" not in source
