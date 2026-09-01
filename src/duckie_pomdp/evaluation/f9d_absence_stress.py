"""F9d Task 4: the absence-yield metrics for B1 and B2.

Mirrors ``f9d_stress.py``'s outlier-yield shape (frame predicate -> event
grouping -> summary), applied to "absence" instead of "localization
outlier". **B1 and B2 are deliberately never merged into one predicate** --
each row carries its own ``absence_kind`` (set by the scenario matrix, not
inferred here), and every function in this module keeps that label attached
to the events and counts it produces, so a caller's report can never
accidentally attribute a controlled-dropout run to genuine out-of-domain
absence or vice versa.

**B1 -- genuine out-of-domain absence.** A row counts once ``absence_kind``
is ``"B1"`` AND ground truth says the pedestrian still exists in the world
AND is not eligible-visible (walked out of the camera's field of view).
This is a ground-truth-based definition, not the estimator's own
``robust_b_observability_class`` -- deliberately, because before any track
has ever initialized the belief mean sits at the origin
(``PredictedObservabilityModel`` classifies ``(0, 0)`` as
``outside_domain``), which would otherwise mislabel ordinary warm-up frames
as B1 absence on every single episode, B1 or not.

**B2 -- controlled in-domain detector dropout.** A row counts once
``absence_kind`` is ``"B2"`` AND ``dropout_frame`` is true -- the flag the
render loop sets from ``DetectorDropout``'s own schedule, independent of
ground truth. B2 never requires ``eligible_visible`` to still be true: if a
B2 scenario's pedestrian happens to also leave the frame during the
dropout window, that is a scenario-design fact worth surfacing (see
``absence_yield``'s ``b2_frames_with_gt_invisible``), not silently folded
into the count either as support for or against the definition.

**"Event" is the same run-based definition Task 3 used**: a contiguous run
of absence frames within one episode is one event, closed by any
non-absence frame (for the row's own kind), never spanning an episode
boundary.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence

B1 = "B1"
B2 = "B2"
B3 = "B3"


@dataclass(frozen=True)
class AbsenceEvent:
    """One contiguous run of absence frames within a single episode."""

    seed: int
    scenario: str
    kind: str
    episode: str
    start_frame: int
    end_frame: int
    length: int


def is_b1_absence_frame(row: Mapping[str, Any]) -> bool:
    """Genuine out-of-domain absence: still exists, not eligible-visible."""

    if row.get("absence_kind") != B1:
        return False
    if not row.get("gt_exists"):
        return False
    return not row.get("eligible_visible")


def is_b2_absence_frame(row: Mapping[str, Any]) -> bool:
    """Controlled in-domain detector dropout: this frame's Duckie detection
    was suppressed by the scheduled intervention."""

    if row.get("absence_kind") != B2:
        return False
    return bool(row.get("dropout_frame"))


def is_b3_absence_frame(row: Mapping[str, Any]) -> bool:
    """Controlled target disappearance, if B3 was implemented: privileged
    truth reports the scenario pedestrian absent from the switch frame on.
    Mirrors B1's ground-truth-based shape (existence is what changed, not
    mere visibility) -- a B3 row is absent because ``gt_exists`` itself is
    false, never because it merely left the frame."""

    if row.get("absence_kind") != B3:
        return False
    return not row.get("gt_exists")


def is_absence_frame(row: Mapping[str, Any]) -> bool:
    """Any of the three kinds -- used only for the combined support count
    against the pre-registered ``minimum_absence_runs_20``/``_40`` minima,
    never for a per-kind claim."""

    return (
        is_b1_absence_frame(row)
        or is_b2_absence_frame(row)
        or is_b3_absence_frame(row)
    )


def _predicate_for_row(row: Mapping[str, Any]) -> bool:
    kind = row.get("absence_kind")
    if kind == B1:
        return is_b1_absence_frame(row)
    if kind == B2:
        return is_b2_absence_frame(row)
    if kind == B3:
        return is_b3_absence_frame(row)
    return False


def _rows_by_episode_in_frame_order(
    rows: Sequence[Mapping[str, Any]]
) -> dict[str, list[Mapping[str, Any]]]:
    by_episode: dict[str, list[Mapping[str, Any]]] = {}
    for row in rows:
        by_episode.setdefault(str(row["episode"]), []).append(row)
    for episode_rows in by_episode.values():
        episode_rows.sort(key=lambda row: int(row["frame"]))
    return by_episode


def _event_from_run(episode: str, run: list[Mapping[str, Any]]) -> AbsenceEvent:
    first, last = run[0], run[-1]
    return AbsenceEvent(
        seed=int(first["seed"]),
        scenario=str(first["scenario"]),
        kind=str(first["absence_kind"]),
        episode=episode,
        start_frame=int(first["frame"]),
        end_frame=int(last["frame"]),
        length=len(run),
    )


def contiguous_absence_events(rows: Sequence[Mapping[str, Any]]) -> list[AbsenceEvent]:
    """Every contiguous absence run, grouped and ordered per episode.

    One episode carries exactly one ``absence_kind`` by construction (the
    scenario matrix assigns it), so grouping by episode alone -- without
    also grouping by kind -- cannot merge a B1 run with a B2 run. Rows are
    sorted by frame within each episode before scanning, so caller row
    order never affects the result, matching
    ``f9d_stress.contiguous_outlier_events``.
    """

    events: list[AbsenceEvent] = []
    for episode, episode_rows in _rows_by_episode_in_frame_order(rows).items():
        run: list[Mapping[str, Any]] = []
        for row in episode_rows:
            if _predicate_for_row(row):
                run.append(row)
                continue
            if run:
                events.append(_event_from_run(episode, run))
                run = []
        if run:
            events.append(_event_from_run(episode, run))
    return events


def _bump(counter: dict[Any, int], key: Any) -> None:
    counter[key] = counter.get(key, 0) + 1


def absence_yield(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """The full absence-yield summary for one collection of frame rows.

    Reports B1 and B2 (and B3, if present) separately throughout, plus a
    COMBINED count (events/frames of any kind) against the pre-registered
    ``minimum_absence_runs_20``/``minimum_absence_runs_40`` minima -- those
    minima do not distinguish kind, but every other field here does.
    """

    rows = list(rows)
    events = contiguous_absence_events(rows)

    def _events_of_kind(kind: str) -> list[AbsenceEvent]:
        return [event for event in events if event.kind == kind]

    events_by_kind = {kind: _events_of_kind(kind) for kind in (B1, B2, B3)}

    def _kind_summary(kind_events: list[AbsenceEvent]) -> dict[str, Any]:
        events_per_seed: dict[int, int] = {}
        events_per_scenario: dict[str, int] = {}
        frames_per_seed: dict[int, int] = {}
        for event in kind_events:
            _bump(events_per_seed, event.seed)
            _bump(events_per_scenario, event.scenario)
            frames_per_seed[event.seed] = frames_per_seed.get(event.seed, 0) + event.length
        return {
            "events": len(kind_events),
            "frames": sum(event.length for event in kind_events),
            "runs_ge_20": sum(1 for event in kind_events if event.length >= 20),
            "runs_ge_40": sum(1 for event in kind_events if event.length >= 40),
            "max_consecutive_length": max((event.length for event in kind_events), default=0),
            "seeds_with_event": len(events_per_seed),
            "events_per_seed": events_per_seed,
            "frames_per_seed": frames_per_seed,
            "events_per_scenario": events_per_scenario,
        }

    per_kind = {kind: _kind_summary(kind_events) for kind, kind_events in events_by_kind.items()}

    b2_dropout_rows = [row for row in rows if row.get("absence_kind") == B2 and row.get("dropout_frame")]
    b2_frames_with_gt_invisible = sum(
        1 for row in b2_dropout_rows if not row.get("eligible_visible")
    )

    return {
        "total_frames": len(rows),
        "events": [asdict(event) for event in events],
        "runs_ge_20": sum(1 for event in events if event.length >= 20),
        "runs_ge_40": sum(1 for event in events if event.length >= 40),
        "events_total": len(events),
        "per_kind": per_kind,
        "b2_dropout_frames": len(b2_dropout_rows),
        "b2_frames_with_gt_invisible": b2_frames_with_gt_invisible,
    }
