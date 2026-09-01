"""Gate F9d Task 2: association selection-rule diagnostic tests.

Reuses the ``RuntimeCacheFrame``/``TruthFrame`` construction pattern from
``tests/test_f9c_robust_updater.py``'s ``_small_cache_and_truth`` (that
file's own hand-written, no-simulator, no-detector cache fixture) rather
than inventing a new fixture shape.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from duckie_pomdp.belief.robust_updater import RobustPedestrianBeliefUpdater
from duckie_pomdp.evaluation.f9c_runtime_cache import (
    RuntimeCacheFrame,
    TruthFrame,
    read_evaluation_truth,
    read_runtime_cache,
)
from duckie_pomdp.evaluation.f9d_association import (
    compare_selection_rules,
    duplicate_frame_ranking,
)

ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "src" / "duckie_pomdp" / "evaluation" / "f9d_association.py"
RUNTIME_CACHE_PATH = ROOT / "artifacts" / "f9c_runtime_cache.npz"
EVALUATION_TRUTH_PATH = ROOT / "artifacts" / "f9c_evaluation_truth.npz"

# The hashes recorded in task-11-report.md from the ORIGINAL 2026-08
# final-evaluation render, identical to
# experiments/evaluate_f9c_robust_belief.py's own EXPECTED_*_SHA256
# constants -- these tests replay the SAME already-produced cache, seeds
# 7101-7104, never a fresh render.
RUNTIME_CACHE_SHA256 = "fe425c55aadd45af88d072c256010f5bddcbb82d952669e8fa988bd70722526d"
EVALUATION_TRUTH_SHA256 = "26663ebb85ebd2ed9fd00ffc0903679b298b9333ef3e10f58ff253d7fd1e2ae9"


# ---------------------------------------------------------------------------
# Fixtures.
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def real_cache_and_truth():
    """The real, already hash-verified F9c cache -- module-scoped so the
    (cheap, ~3328-frame) read happens once for every test in this file."""

    cache = read_runtime_cache(RUNTIME_CACHE_PATH, expected_sha256=RUNTIME_CACHE_SHA256)
    truth = read_evaluation_truth(EVALUATION_TRUTH_PATH, expected_sha256=EVALUATION_TRUTH_SHA256)
    return cache, truth


def _episode_slice(cache, truth, *, episode: str):
    cache_slice = tuple(frame for frame in cache if frame.episode == episode)
    truth_slice = {key: value for key, value in truth.items() if key[0] == episode}
    return cache_slice, truth_slice


def _two_frame_track_cache() -> tuple[tuple[RuntimeCacheFrame, ...], dict]:
    """A hand-written, two-frame, one-episode cache/truth pair -- no
    simulator, no detector. Frame 0 has a single candidate (establishes a
    track via association's "initialization" branch); frame 1 has TWO
    candidates once a predicted state already exists, so it is the one
    frame both scoring rules actually get to choose between."""

    episode = "synthetic_two_frame"
    frames = (
        RuntimeCacheFrame(
            episode=episode,
            seed=9999,
            scenario="stationary_ped_stationary_ego",
            frame=0,
            dt_s=1.0 / 30.0,
            raw_candidate_range_m=(0.70,),
            raw_candidate_bearing_rad=(0.02,),
            raw_candidate_confidence=(0.9,),
            raw_candidate_bbox=((250.0, 200.0, 350.0, 300.0),),
            raw_candidate_projection_failed=(False,),
            ego_linear_velocity_mps=0.0,
            ego_yaw_rate_rad_s=0.0,
        ),
        RuntimeCacheFrame(
            episode=episode,
            seed=9999,
            scenario="stationary_ped_stationary_ego",
            frame=1,
            dt_s=1.0 / 30.0,
            raw_candidate_range_m=(0.68, 0.90),
            raw_candidate_bearing_rad=(0.01, 0.05),
            raw_candidate_confidence=(0.9, 0.4),
            raw_candidate_bbox=((252.0, 202.0, 352.0, 302.0), (400.0, 150.0, 480.0, 250.0)),
            raw_candidate_projection_failed=(False, False),
            ego_linear_velocity_mps=0.0,
            ego_yaw_rate_rad_s=0.0,
        ),
    )
    truth = {
        (episode, 0): TruthFrame(
            episode=episode,
            frame=0,
            gt_exists=True,
            gt_range_m=0.70,
            gt_bearing_rad=0.02,
            gt_range_rate_mps=0.0,
            gt_bearing_rate_rad_s=0.0,
            eligible_visible=True,
            visible_pixel_count=500,
            gt_bbox=(250.0, 200.0, 350.0, 300.0),
            distance_bin="near",
            fov_region="center",
        ),
        (episode, 1): TruthFrame(
            episode=episode,
            frame=1,
            gt_exists=True,
            gt_range_m=0.68,
            gt_bearing_rad=0.01,
            gt_range_rate_mps=0.0,
            gt_bearing_rate_rad_s=0.0,
            eligible_visible=True,
            visible_pixel_count=500,
            gt_bbox=(252.0, 202.0, 352.0, 302.0),
            distance_bin="near",
            fov_region="center",
        ),
    }
    return frames, truth


# ---------------------------------------------------------------------------
# 1. The predicted state must be held fixed -- a single real trajectory.
# ---------------------------------------------------------------------------


def test_selection_comparison_holds_the_predicted_state_fixed(monkeypatch):
    """Both scorings must see the SAME predicted state and covariance on
    each frame. If the comparison re-ran the filter per rule, the
    trajectories would diverge and the result would measure drift, not
    selection quality.

    Proved structurally: RobustPedestrianBeliefUpdater.update -- the ONLY
    function that ever advances the real EKF/existence trajectory -- is
    called exactly ONCE per frame, never once per counterfactual scoring
    rule. If the implementation instead ran a separate trajectory per rule,
    this count would be doubled.
    """

    cache, truth = _two_frame_track_cache()

    calls = []
    original_update = RobustPedestrianBeliefUpdater.update

    def counting_update(self, *args, **kwargs):
        calls.append(1)
        return original_update(self, *args, **kwargs)

    monkeypatch.setattr(RobustPedestrianBeliefUpdater, "update", counting_update)

    result = compare_selection_rules(cache, truth)

    assert len(calls) == len(cache), (
        "update() must be called exactly once per frame -- one real "
        "trajectory, regardless of how many counterfactual selection rules "
        "are scored against each frame's snapshot"
    )
    # Frame 0 has no predicted state yet (track initializes there); frame 1
    # is the only frame with a fixed predicted state to compare rules on.
    assert result["frames_compared"] == 1


# ---------------------------------------------------------------------------
# 2. C1: lambda = 1 and the frozen lambda must be able to disagree on the
#    same frame, given the SAME predicted state.
# ---------------------------------------------------------------------------


def test_lambda_one_and_frozen_lambda_can_disagree_on_the_same_frame(real_cache_and_truth):
    """A real frame (evaluation_7101_cross_near_left_to_right, frame 91)
    where lambda = 1 selects a candidate with GT IoU 0.481 (a localization
    outlier) while the frozen lambda_r selects a different candidate with
    GT IoU 0.761 (not an outlier), given the identical predicted state. The
    comparison must report this disagreement rather than silently returning
    identical selections for both scorings.
    """

    cache, truth = real_cache_and_truth
    episode_cache, episode_truth = _episode_slice(
        cache, truth, episode="evaluation_7101_cross_near_left_to_right"
    )

    result = compare_selection_rules(episode_cache, episode_truth)

    frame_91 = next(record for record in result["frames"] if record["frame"] == 91)
    assert frame_91["agree"] is False
    assert frame_91["mode_a"] == "temporal"
    assert frame_91["mode_b"] == "temporal"
    assert frame_91["outlier_a"] is True
    assert frame_91["outlier_b"] is False
    assert frame_91["iou_a"] == pytest.approx(0.48142347549449277, abs=1e-9)
    assert frame_91["iou_b"] == pytest.approx(0.7607792274327276, abs=1e-9)


# ---------------------------------------------------------------------------
# 2b. Fix round 1: C1-abstention. The paired selection-quality comparison's
#     exclusion bucket ("one side made no selection") turned out to hold
#     most of C1's real disagreement (19-20 of 23 differing frames) --
#     abstention must be reported as a headline measure, with internally
#     consistent counts, not silently dropped.
# ---------------------------------------------------------------------------


def test_c1_abstention_is_reported_as_a_headline_measure_with_consistent_counts(
    real_cache_and_truth,
):
    """On the full real cache, lambda = 1's tighter (uninflated) S rejects
    materially more candidates outright than the frozen lambda_r does, and
    the candidate-level association-gate exceedance fraction moves in the
    same direction -- the direct mechanism check. Also pins the internal
    bookkeeping identities that must hold regardless of which cache slice
    produced them.
    """

    cache, truth = real_cache_and_truth
    result = compare_selection_rules(cache, truth)

    abstention = result["abstention"]

    # Internal bookkeeping identities -- must hold on ANY cache/slice.
    assert abstention["one_sided_abstention_frame_count"] == (
        abstention["rule_a_abstained_rule_b_selected"]
        + abstention["rule_b_abstained_rule_a_selected"]
    )
    assert (
        abstention["one_sided_abstention_frames_with_gt_available"]
        <= abstention["one_sided_abstention_frame_count"]
    )
    assert abstention["frames_rule_a_selected_nothing"] >= abstention[
        "rule_a_abstained_rule_b_selected"
    ]
    assert abstention["frames_rule_b_selected_nothing"] >= abstention[
        "rule_b_abstained_rule_a_selected"
    ]
    for fraction in abstention["candidate_gate_exceedance_fraction"].values():
        assert fraction is None or 0.0 <= fraction <= 1.0

    # The regression-pinned real counts behind this task's reported finding.
    assert abstention["frames_rule_a_selected_nothing"] == 42
    assert abstention["frames_rule_b_selected_nothing"] == 22
    assert abstention["rule_a_abstained_rule_b_selected"] == 20
    assert abstention["rule_b_abstained_rule_a_selected"] == 0
    assert abstention["candidate_gate_exceedance_fraction"]["rule_a"] > (
        abstention["candidate_gate_exceedance_fraction"]["rule_b"]
    )

    # The conclusion function's own verdict must be re-derivable from the
    # exact sub-fields it reports, so a future edit cannot desync them.
    conclusion = result["conclusion_abstention"]
    recomputed_supported = (
        conclusion["lambda_one_abstains_more_often"]
        and conclusion["gate_exceedance_moves_same_direction"]
    )
    assert conclusion["verdict"] == ("SUPPORTED" if recomputed_supported else "UNSUPPORTED")
    assert conclusion["verdict"] == "SUPPORTED"
    assert result["conclusion"]["abstention"] == conclusion["verdict"]


# ---------------------------------------------------------------------------
# 3. C2: on a duplicate frame, credit confidence and penalise min-NIS when
#    that is what the GT IoUs show.
# ---------------------------------------------------------------------------


def test_duplicate_ranking_scores_both_rules_against_gt_iou(real_cache_and_truth):
    """A real duplicate frame (evaluation_7101_cross_near_left_to_right,
    frame 96) where highest-confidence selects the higher-IoU box (0.777,
    not an outlier) and minimum-NIS at the frozen lambda selects the
    lower-IoU box (0.464, a localization outlier). The report must credit
    confidence and penalise min-NIS on this frame.
    """

    cache, truth = real_cache_and_truth
    episode_cache, episode_truth = _episode_slice(
        cache, truth, episode="evaluation_7101_cross_near_left_to_right"
    )

    result = duplicate_frame_ranking(episode_cache, episode_truth)

    frame_96 = next(record for record in result["frames"] if record["frame"] == 96)
    assert frame_96["duplicate_selection"] is True
    assert frame_96["agree"] is False
    assert frame_96["outlier_a"] is False  # rule A = highest confidence
    assert frame_96["outlier_b"] is True  # rule B = min-NIS at frozen lambda
    assert frame_96["iou_a"] > frame_96["iou_b"]

    # The aggregate must reflect this frame's contribution: at least one
    # differing, GT-available frame where rule A (confidence) has the
    # higher IoU.
    assert result["differing_frames_paired"]["rule_a_higher_iou"] >= 1


# ---------------------------------------------------------------------------
# 4. The diagnostic must never write an estimator parameter.
# ---------------------------------------------------------------------------

_FROZEN_ESTIMATOR_PARAMETER_NAMES = frozenset(
    {
        "range_scale",
        "bearing_scale",
        "range_posterior_floor_m",
        "bearing_posterior_floor_rad",
        "chi_square_gate",
        "chi_square_threshold",
        "active_threshold",
        "delete_threshold",
        "initialization_threshold",
        "prior_probability",
        "detection_probability",
        "false_positive_probability",
        "survival_probability",
        "birth_probability",
        "miss_likelihood_floor",
        "range_bias_m",
        "bearing_bias_rad",
        "range_bin_bias_m",
        "detection_probability_center",
        "detection_probability_mid_fov",
        "detection_probability_edge_fov",
        "detection_probability_outside_domain",
        "minimum_range_m",
        "position_process_std_m_per_sqrt_s",
        "velocity_process_std_mps_per_sqrt_s",
        "initial_velocity_std_mps",
    }
)


def test_the_diagnostic_never_writes_an_estimator_parameter():
    """Scan the module for assignment to any frozen parameter name. A
    diagnostic that mutates the estimator is not a diagnostic.

    Only plain-variable (``ast.Name``) and attribute (``ast.Attribute``)
    assignment TARGETS are checked -- a dict key such as
    ``result["matching_iou_threshold"] = ...`` is building this module's own
    OUTPUT report, not writing into any estimator config object, and must
    not be flagged.
    """

    source = MODULE_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(MODULE_PATH))

    offending: list[tuple[str, int]] = []

    def _target_names(target: ast.expr):
        if isinstance(target, ast.Name):
            yield target.id
        elif isinstance(target, ast.Attribute):
            yield target.attr
        elif isinstance(target, (ast.Tuple, ast.List)):
            for element in target.elts:
                yield from _target_names(element)

    for node in ast.walk(tree):
        targets: list[ast.expr] = []
        if isinstance(node, ast.Assign):
            targets = node.targets
        elif isinstance(node, (ast.AnnAssign, ast.AugAssign)):
            targets = [node.target]
        else:
            continue
        for target in targets:
            for name in _target_names(target):
                if name in _FROZEN_ESTIMATOR_PARAMETER_NAMES:
                    offending.append((name, node.lineno))

    assert not offending, f"assignment(s) to frozen estimator parameter name(s): {offending}"


# ---------------------------------------------------------------------------
# 5. The diagnostic constructs no detector and no simulator.
# ---------------------------------------------------------------------------


def test_the_diagnostic_constructs_no_detector_and_no_simulator(monkeypatch):
    """Monkeypatch YoloObjectDetector.__init__ and create_gym_duckietown to
    raise; the diagnostic must complete."""

    from duckie_pomdp.adapters import gym_duckietown as gym_duckietown_module
    from duckie_pomdp.perception import yolo_detector as yolo_detector_module

    def _raise(*args, **kwargs):
        raise AssertionError("the association diagnostic must not run inference or render")

    monkeypatch.setattr(yolo_detector_module.YoloObjectDetector, "__init__", _raise)
    monkeypatch.setattr(gym_duckietown_module, "create_gym_duckietown", _raise)

    cache, truth = _two_frame_track_cache()
    compare_selection_rules(cache, truth)
    duplicate_frame_ranking(cache, truth)
