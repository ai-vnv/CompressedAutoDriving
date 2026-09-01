from duckie_pomdp.evaluation.f9d_stress import (
    contiguous_outlier_events,
    is_localization_outlier_frame,
    outlier_yield,
)


def _row(
    *,
    episode="ep",
    seed=8101,
    scenario="scn",
    frame,
    eligible_visible=True,
    detected=True,
    iou,
    distance_bin="far",
    fov_region="edge_fov",
):
    """Build one synthetic row using the exact schema
    ``evaluate_f9c_robust_belief.build_row`` emits -- ``selected_bbox_x1``
    stands in for "a candidate was detected", ``selected_iou`` for the
    already-computed GT IoU of Baseline A's highest-confidence pick."""

    return {
        "episode": episode,
        "seed": seed,
        "scenario": scenario,
        "frame": frame,
        "eligible_visible": eligible_visible,
        "selected_bbox_x1": (10.0 if detected else None),
        "selected_iou": (None if iou is None else iou),
        "distance_bin": distance_bin,
        "fov_region": fov_region,
    }


def _bad(frame, **kwargs):
    return _row(frame=frame, iou=0.10, **kwargs)


def _good(frame, **kwargs):
    return _row(frame=frame, iou=0.90, **kwargs)


# ---------------------------------------------------------------------------
# is_localization_outlier_frame
# ---------------------------------------------------------------------------


def test_outlier_requires_low_iou():
    assert is_localization_outlier_frame(_row(frame=0, iou=0.10))
    assert not is_localization_outlier_frame(_row(frame=0, iou=0.90))


def test_outlier_boundary_is_strict_less_than():
    """IoU exactly at the threshold is NOT an outlier -- the gate's
    threshold is locked at 0.50 and must not be nudged in either direction
    by an off-by-one comparison here."""

    assert not is_localization_outlier_frame(
        _row(frame=0, iou=0.50), matching_iou_threshold=0.50
    )
    assert is_localization_outlier_frame(
        _row(frame=0, iou=0.4999), matching_iou_threshold=0.50
    )


def test_not_eligible_visible_is_never_an_outlier_even_with_low_iou():
    row = _row(frame=0, iou=0.0, eligible_visible=False)
    assert not is_localization_outlier_frame(row)


def test_not_detected_is_never_an_outlier():
    row = _row(frame=0, iou=None, detected=False)
    assert not is_localization_outlier_frame(row)


def test_missing_selected_iou_is_never_an_outlier():
    row = _row(frame=0, iou=None, detected=True)
    assert not is_localization_outlier_frame(row)


# ---------------------------------------------------------------------------
# contiguous_outlier_events -- the pinned event-definition test.
# ---------------------------------------------------------------------------


def test_pinned_bad_bad_bad_good_bad_is_two_events_four_frames():
    """The definition that matters most: a contiguous run of outlier frames
    is ONE event regardless of its length; a single non-outlier frame closes
    it. [bad, bad, bad, good, bad] in one episode must count 2 events and 4
    frames -- not 1 event of 5, and not 5 events of 1."""

    rows = [
        _bad(0),
        _bad(1),
        _bad(2),
        _good(3),
        _bad(4),
    ]
    events = contiguous_outlier_events(rows)
    assert len(events) == 2
    assert [event.length for event in events] == [3, 1]
    assert sum(event.length for event in events) == 4

    summary = outlier_yield(rows)
    assert summary["outlier_frames"] == 4
    assert summary["outlier_events"] == 2


def test_one_long_run_is_one_event_not_n_frames():
    rows = [_bad(frame) for frame in range(8)]
    events = contiguous_outlier_events(rows)
    assert len(events) == 1
    assert events[0].length == 8


def test_events_never_span_an_episode_boundary():
    """Two separate episodes, each ending/starting on an outlier frame, must
    never be merged into one event even though they share a seed and are
    adjacent in row order."""

    rows = [
        _bad(0, episode="ep_a"),
        _bad(1, episode="ep_a"),
        _bad(0, episode="ep_b"),
        _bad(1, episode="ep_b"),
    ]
    events = contiguous_outlier_events(rows)
    assert len(events) == 2
    assert {event.episode for event in events} == {"ep_a", "ep_b"}
    assert all(event.length == 2 for event in events)


def test_out_of_order_input_rows_are_sorted_by_frame_before_grouping():
    """Row order in the input list must not change the result -- events are
    detected against frame order within an episode, not list order."""

    rows = [
        _bad(2),
        _good(3),
        _bad(0),
        _bad(4),
        _bad(1),
    ]
    events = contiguous_outlier_events(rows)
    assert len(events) == 2
    assert [event.length for event in events] == [3, 1]


def test_a_non_outlier_frame_for_any_reason_closes_a_run():
    """The run-closing "non-outlier frame" is not-outlier for ANY reason --
    not eligible-visible, not detected, or IoU above threshold all count the
    same way."""

    rows = [
        _bad(0),
        _row(frame=1, iou=0.0, eligible_visible=False),  # closes the run
        _bad(2),
    ]
    events = contiguous_outlier_events(rows)
    assert len(events) == 2
    assert all(event.length == 1 for event in events)


# ---------------------------------------------------------------------------
# outlier_yield -- aggregate summary.
# ---------------------------------------------------------------------------


def test_yield_counts_total_and_visible_frames():
    rows = [
        _bad(0, eligible_visible=True),
        _row(frame=1, iou=None, eligible_visible=False, detected=False),
        _good(2, eligible_visible=True),
    ]
    summary = outlier_yield(rows)
    assert summary["total_frames"] == 3
    assert summary["visible_frames"] == 2


def test_yield_events_and_frames_per_seed():
    rows = [
        *[_bad(frame, episode="ep_a", seed=8101) for frame in range(3)],
        _good(3, episode="ep_a", seed=8101),
        _bad(0, episode="ep_b", seed=8102),
    ]
    summary = outlier_yield(rows)
    assert summary["events_per_seed"] == {8101: 1, 8102: 1}
    assert summary["outlier_frames_per_seed"] == {8101: 3, 8102: 1}
    assert summary["seeds_with_event"] == 2


def test_seeds_with_event_excludes_seeds_with_zero_events():
    rows = [
        _bad(0, episode="ep_a", seed=8101),
        _good(0, episode="ep_b", seed=8102),
    ]
    summary = outlier_yield(rows)
    assert summary["seeds_with_event"] == 1
    assert 8102 not in summary["events_per_seed"]


def test_yield_scenario_distribution_required_even_if_concentrated():
    rows = [
        *[_bad(frame, episode="ep_a", scenario="cross_edge") for frame in range(2)],
        _bad(0, episode="ep_b", scenario="cross_edge"),
        _bad(0, episode="ep_c", scenario="cross_near"),
    ]
    summary = outlier_yield(rows)
    assert summary["frames_per_scenario"] == {"cross_edge": 3, "cross_near": 1}
    assert summary["events_per_scenario"] == {"cross_edge": 2, "cross_near": 1}


def test_yield_distance_bin_and_fov_region_distribution():
    rows = [
        _bad(0, episode="ep_a", distance_bin="far", fov_region="edge_fov"),
        _bad(0, episode="ep_b", distance_bin="near", fov_region="center"),
        _bad(1, episode="ep_b", distance_bin="near", fov_region="center"),
    ]
    summary = outlier_yield(rows)
    assert summary["frames_per_distance_bin"] == {"far": 1, "near": 2}
    assert summary["frames_per_fov_region"] == {"edge_fov": 1, "center": 2}


def test_max_consecutive_and_median_event_length():
    rows = [
        *[_bad(frame, episode="ep_a") for frame in range(5)],  # one 5-run
        _bad(0, episode="ep_b"),  # one 1-run
        *[_bad(frame, episode="ep_c") for frame in range(3)],  # one 3-run
    ]
    summary = outlier_yield(rows)
    assert summary["max_consecutive_outlier_length"] == 5
    assert summary["median_outlier_event_length"] == 3  # lengths [1, 3, 5]


def test_empty_rows_yield_zeroes_not_errors():
    summary = outlier_yield([])
    assert summary["total_frames"] == 0
    assert summary["outlier_frames"] == 0
    assert summary["outlier_events"] == 0
    assert summary["max_consecutive_outlier_length"] == 0
    assert summary["median_outlier_event_length"] == 0.0


def test_matching_iou_threshold_is_configurable_and_used_consistently():
    """A tighter threshold can turn a previously-fine frame into an
    outlier; a looser one can do the reverse -- the parameter must actually
    flow through both the frame predicate and the event grouping."""

    row = _row(frame=0, iou=0.6)
    assert not is_localization_outlier_frame(row, matching_iou_threshold=0.5)
    assert is_localization_outlier_frame(row, matching_iou_threshold=0.7)

    summary_strict = outlier_yield([row], matching_iou_threshold=0.5)
    summary_loose = outlier_yield([row], matching_iou_threshold=0.7)
    assert summary_strict["outlier_frames"] == 0
    assert summary_loose["outlier_frames"] == 1
