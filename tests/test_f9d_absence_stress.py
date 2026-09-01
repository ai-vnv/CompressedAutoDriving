from duckie_pomdp.evaluation.f9d_absence_stress import (
    absence_yield,
    contiguous_absence_events,
    is_absence_frame,
    is_b1_absence_frame,
    is_b2_absence_frame,
    is_b3_absence_frame,
)


def _row(
    *,
    episode="ep",
    seed=8101,
    scenario="scn",
    frame,
    absence_kind,
    eligible_visible=True,
    gt_exists=True,
    dropout_frame=False,
):
    return {
        "episode": episode,
        "seed": seed,
        "scenario": scenario,
        "frame": frame,
        "absence_kind": absence_kind,
        "eligible_visible": eligible_visible,
        "gt_exists": gt_exists,
        "dropout_frame": dropout_frame,
    }


def _b1_absent(frame, **kwargs):
    return _row(frame=frame, absence_kind="B1", eligible_visible=False, gt_exists=True, **kwargs)


def _b1_present(frame, **kwargs):
    return _row(frame=frame, absence_kind="B1", eligible_visible=True, gt_exists=True, **kwargs)


def _b2_dropout(frame, **kwargs):
    kwargs.setdefault("eligible_visible", True)
    return _row(frame=frame, absence_kind="B2", dropout_frame=True, **kwargs)


def _b2_normal(frame, **kwargs):
    return _row(frame=frame, absence_kind="B2", dropout_frame=False, eligible_visible=True, **kwargs)


# ---------------------------------------------------------------------------
# is_b1_absence_frame
# ---------------------------------------------------------------------------


def test_b1_requires_kind_b1():
    row = _row(frame=0, absence_kind="B2", eligible_visible=False, gt_exists=True, dropout_frame=True)
    assert not is_b1_absence_frame(row)


def test_b1_requires_not_eligible_visible():
    assert is_b1_absence_frame(_b1_absent(0))
    assert not is_b1_absence_frame(_b1_present(0))


def test_b1_requires_gt_exists():
    """A frame where privileged truth lost the pedestrian entirely is not
    genuine out-of-domain absence -- that is not what B1 measures."""

    row = _row(frame=0, absence_kind="B1", eligible_visible=False, gt_exists=False)
    assert not is_b1_absence_frame(row)


def test_b1_does_not_trigger_on_warmup_frames():
    """The estimator's own predicted-observability classification puts the
    NEVER-initialized belief mean at outside_domain on every episode's very
    first frames -- B1's definition must not key off that, only off ground
    truth, or every episode (B1 or not) would spuriously start with a B1
    run."""

    row = _row(
        frame=0,
        absence_kind="B1",
        eligible_visible=True,
        gt_exists=True,
    )
    assert not is_b1_absence_frame(row)


# ---------------------------------------------------------------------------
# is_b2_absence_frame
# ---------------------------------------------------------------------------


def test_b2_requires_kind_b2():
    row = _b1_absent(0)
    row = dict(row, dropout_frame=True)
    assert not is_b2_absence_frame(row)


def test_b2_requires_dropout_frame_flag():
    assert is_b2_absence_frame(_b2_dropout(0))
    assert not is_b2_absence_frame(_b2_normal(0))


def test_b2_does_not_require_eligible_visible():
    """B2 counts the scheduled dropout window regardless of GT visibility --
    a B2 frame is defined by the intervention, not by ground truth."""

    row = _b2_dropout(0, eligible_visible=False)
    assert is_b2_absence_frame(row)


# ---------------------------------------------------------------------------
# is_b3_absence_frame / is_absence_frame
# ---------------------------------------------------------------------------


def test_b3_requires_gt_exists_false():
    present = _row(frame=0, absence_kind="B3", gt_exists=True)
    absent = _row(frame=0, absence_kind="B3", gt_exists=False)
    assert not is_b3_absence_frame(present)
    assert is_b3_absence_frame(absent)


def test_is_absence_frame_covers_all_three_kinds():
    assert is_absence_frame(_b1_absent(0))
    assert is_absence_frame(_b2_dropout(0))
    assert is_absence_frame(_row(frame=0, absence_kind="B3", gt_exists=False))
    assert not is_absence_frame(_b1_present(0))
    assert not is_absence_frame(_b2_normal(0))


# ---------------------------------------------------------------------------
# contiguous_absence_events -- same event definition as f9d_stress, applied
# to absence instead of outliers, kind-labelled throughout.
# ---------------------------------------------------------------------------


def test_pinned_absent_absent_absent_present_absent_is_two_events_four_frames():
    rows = [
        _b1_absent(0),
        _b1_absent(1),
        _b1_absent(2),
        _b1_present(3),
        _b1_absent(4),
    ]
    events = contiguous_absence_events(rows)
    assert len(events) == 2
    assert [event.length for event in events] == [3, 1]
    assert all(event.kind == "B1" for event in events)


def test_one_long_dropout_run_is_one_event_not_n_frames():
    rows = [_b2_dropout(frame) for frame in range(25)]
    events = contiguous_absence_events(rows)
    assert len(events) == 1
    assert events[0].length == 25
    assert events[0].kind == "B2"


def test_events_never_span_an_episode_boundary():
    rows = [
        _b1_absent(0, episode="ep_a"),
        _b1_absent(1, episode="ep_a"),
        _b1_absent(0, episode="ep_b"),
        _b1_absent(1, episode="ep_b"),
    ]
    events = contiguous_absence_events(rows)
    assert len(events) == 2
    assert {event.episode for event in events} == {"ep_a", "ep_b"}


def test_out_of_order_rows_are_sorted_by_frame_before_grouping():
    rows = [
        _b1_absent(2),
        _b1_present(3),
        _b1_absent(0),
        _b1_absent(4),
        _b1_absent(1),
    ]
    events = contiguous_absence_events(rows)
    assert len(events) == 2
    assert [event.length for event in events] == [3, 1]


def test_b1_and_b2_events_in_the_same_row_set_are_never_merged():
    """Different episodes, different kinds, same seed -- must stay two
    separate, correctly-kind-labelled events even when interleaved in the
    input list."""

    rows = [
        _b1_absent(0, episode="ep_b1"),
        _b1_absent(1, episode="ep_b1"),
        _b2_dropout(0, episode="ep_b2"),
        _b2_dropout(1, episode="ep_b2"),
        _b2_dropout(2, episode="ep_b2"),
    ]
    events = contiguous_absence_events(rows)
    assert len(events) == 2
    by_kind = {event.kind: event for event in events}
    assert by_kind["B1"].length == 2
    assert by_kind["B2"].length == 3


# ---------------------------------------------------------------------------
# absence_yield -- aggregate summary.
# ---------------------------------------------------------------------------


def test_yield_runs_ge_20_and_ge_40_thresholds():
    rows = [
        *[_b2_dropout(frame, episode="ep_20") for frame in range(20)],
        *[_b2_dropout(frame, episode="ep_40") for frame in range(40)],
        *[_b1_absent(frame, episode="ep_10") for frame in range(10)],
    ]
    summary = absence_yield(rows)
    assert summary["runs_ge_20"] == 2  # the 20-run and the 40-run
    assert summary["runs_ge_40"] == 1  # only the 40-run
    assert summary["events_total"] == 3


def test_yield_per_kind_breakdown_stays_separate():
    rows = [
        *[_b1_absent(frame, episode="ep_b1", seed=8101) for frame in range(45)],
        *[_b2_dropout(frame, episode="ep_b2", seed=8102) for frame in range(25)],
    ]
    summary = absence_yield(rows)
    assert summary["per_kind"]["B1"]["events"] == 1
    assert summary["per_kind"]["B1"]["frames"] == 45
    assert summary["per_kind"]["B1"]["runs_ge_40"] == 1
    assert summary["per_kind"]["B2"]["events"] == 1
    assert summary["per_kind"]["B2"]["frames"] == 25
    assert summary["per_kind"]["B2"]["runs_ge_40"] == 0
    assert summary["per_kind"]["B2"]["runs_ge_20"] == 1
    # Never merged: B1's frame count must not leak into B2's bucket.
    assert summary["per_kind"]["B1"]["events_per_seed"] == {8101: 1}
    assert summary["per_kind"]["B2"]["events_per_seed"] == {8102: 1}


def test_yield_flags_b2_frames_where_gt_also_went_invisible():
    """Disclosed, not hidden: if a B2 scenario's pedestrian happens to also
    leave the frame during the dropout window, that is a fact about the
    scenario worth surfacing separately from the dropout-frame count
    itself."""

    rows = [
        _b2_dropout(0, eligible_visible=True),
        _b2_dropout(1, eligible_visible=False),
        _b2_dropout(2, eligible_visible=False),
    ]
    summary = absence_yield(rows)
    assert summary["b2_dropout_frames"] == 3
    assert summary["b2_frames_with_gt_invisible"] == 2


def test_empty_rows_yield_zeroes_not_errors():
    summary = absence_yield([])
    assert summary["total_frames"] == 0
    assert summary["events_total"] == 0
    assert summary["runs_ge_20"] == 0
    assert summary["runs_ge_40"] == 0
    assert summary["per_kind"]["B1"]["events"] == 0
    assert summary["per_kind"]["B2"]["events"] == 0
    assert summary["per_kind"]["B3"]["events"] == 0
