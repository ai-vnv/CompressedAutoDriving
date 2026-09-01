"""F9d Task 3: the outlier-yield metrics.

``outlier_yield(rows)`` is the single place that turns a sequence of
per-frame rows (the same schema ``evaluate_f9c_robust_belief.build_row``
produces) into the numbers the pre-registered support gate needs:

* how many frames are localization outliers,
* how many *distinct contiguous events* those frames form, and
* how those events/frames are spread across seeds, scenarios, range bins,
  and FOV regions.

**"Localization outlier" is locked**: Baseline A's highest-confidence
selection (``build_row``'s ``selected_iou``) has GT IoU below
``matching_iou_threshold`` on a frame that is eligible-visible and detected.
This module does not recompute IoU from bounding boxes -- it trusts
``selected_iou``, which ``build_row`` already sets to ``None`` unless
eligible_visible AND a GT box AND a selected box all exist for that frame --
so it cannot silently diverge from what the row-building path already
decided for the exact same frame. The same definition is used by
``f9d_association.py``'s C1/C2 diagnostic, both reading
``matching_iou_threshold`` from the same frozen F9c config section
(``calibration_protocol.matching_iou_threshold``) rather than a second
hard-coded literal.

**"Event" is the definition that matters most.** An event is a *contiguous
run* of outlier frames within one episode, not a frame count: one bbox
failure lasting 8 consecutive frames is 1 event, not 8. A run is closed only
when a non-outlier frame (for ANY reason: not eligible-visible, not
detected, or IoU >= threshold) intervenes within the same episode -- it
never spans two episodes, even if adjacent rows share a seed. This is what
makes ``[bad, bad, bad, good, bad]`` in one episode count 2 events and 4
frames, and it is what stops 50 outlier frames from two long bursts (a
single failure mode) from passing as the same evidence as 50 frames spread
across a dozen scattered failures. See
``duckie_pomdp.evaluation.f9d_protocol.outlier_support_satisfied``, which
this module's counts feed but does not itself implement.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from statistics import median
from typing import Any, Mapping, Sequence

DEFAULT_MATCHING_IOU_THRESHOLD = 0.50


@dataclass(frozen=True)
class OutlierEvent:
    """One contiguous run of outlier frames within a single episode."""

    seed: int
    scenario: str
    episode: str
    start_frame: int
    end_frame: int
    length: int


def is_localization_outlier_frame(
    row: Mapping[str, Any],
    *,
    matching_iou_threshold: float = DEFAULT_MATCHING_IOU_THRESHOLD,
) -> bool:
    """Whether one row is a natural localization-outlier frame.

    Three conditions, all required: eligible-visible, a selected (detected)
    bounding box, and GT IoU below ``matching_iou_threshold``. The three are
    checked explicitly (rather than only trusting ``selected_iou is not
    None``) so a schema change that stops populating ``selected_iou``
    correctly cannot silently manufacture or hide outliers.
    """

    if not row.get("eligible_visible"):
        return False
    if row.get("selected_bbox_x1") in (None, ""):
        return False
    iou = row.get("selected_iou")
    if iou in (None, ""):
        return False
    return float(iou) < matching_iou_threshold


def _rows_by_episode_in_frame_order(
    rows: Sequence[Mapping[str, Any]]
) -> dict[str, list[Mapping[str, Any]]]:
    by_episode: dict[str, list[Mapping[str, Any]]] = {}
    for row in rows:
        by_episode.setdefault(str(row["episode"]), []).append(row)
    for episode_rows in by_episode.values():
        episode_rows.sort(key=lambda row: int(row["frame"]))
    return by_episode


def _event_from_run(episode: str, run: list[Mapping[str, Any]]) -> OutlierEvent:
    first, last = run[0], run[-1]
    return OutlierEvent(
        seed=int(first["seed"]),
        scenario=str(first["scenario"]),
        episode=episode,
        start_frame=int(first["frame"]),
        end_frame=int(last["frame"]),
        length=len(run),
    )


def contiguous_outlier_events(
    rows: Sequence[Mapping[str, Any]],
    *,
    matching_iou_threshold: float = DEFAULT_MATCHING_IOU_THRESHOLD,
) -> list[OutlierEvent]:
    """Every contiguous outlier run, grouped and ordered per episode.

    Rows are sorted by frame number within each episode before scanning, so
    caller row order never affects the result. A run closes -- and a new one
    may open -- the instant a non-outlier frame is seen; it never survives
    an episode boundary.
    """

    events: list[OutlierEvent] = []
    for episode, episode_rows in _rows_by_episode_in_frame_order(rows).items():
        run: list[Mapping[str, Any]] = []
        for row in episode_rows:
            if is_localization_outlier_frame(
                row, matching_iou_threshold=matching_iou_threshold
            ):
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


def outlier_yield(
    rows: Sequence[Mapping[str, Any]],
    *,
    matching_iou_threshold: float = DEFAULT_MATCHING_IOU_THRESHOLD,
) -> dict[str, Any]:
    """The full yield-probe summary for one collection of frame rows.

    Returns every number the required report structure and the
    pre-registered support gate need: frame/event counts, their spread
    across seeds/scenarios/range-bins/FOV-regions, and event-length
    statistics. Does not itself judge PASS/FAIL against the minima -- that
    decision belongs to ``f9d_protocol.outlier_support_satisfied``, fed by
    this dict's ``outlier_frames``/``outlier_events``/``seeds_with_event``.
    """

    rows = list(rows)
    outlier_rows = [
        row
        for row in rows
        if is_localization_outlier_frame(row, matching_iou_threshold=matching_iou_threshold)
    ]
    events = contiguous_outlier_events(rows, matching_iou_threshold=matching_iou_threshold)

    events_per_seed: dict[int, int] = {}
    events_per_scenario: dict[str, int] = {}
    for event in events:
        _bump(events_per_seed, event.seed)
        _bump(events_per_scenario, event.scenario)

    outlier_frames_per_seed: dict[int, int] = {}
    frames_per_scenario: dict[str, int] = {}
    frames_per_distance_bin: dict[str, int] = {}
    frames_per_fov_region: dict[str, int] = {}
    for row in outlier_rows:
        _bump(outlier_frames_per_seed, int(row["seed"]))
        _bump(frames_per_scenario, str(row["scenario"]))
        _bump(frames_per_distance_bin, str(row.get("distance_bin")))
        _bump(frames_per_fov_region, str(row.get("fov_region")))

    lengths = [event.length for event in events]

    return {
        "matching_iou_threshold": matching_iou_threshold,
        "total_frames": len(rows),
        "visible_frames": sum(1 for row in rows if row.get("eligible_visible")),
        "outlier_frames": len(outlier_rows),
        "outlier_events": len(events),
        "seeds_with_event": len(events_per_seed),
        "events_per_seed": events_per_seed,
        "outlier_frames_per_seed": outlier_frames_per_seed,
        "frames_per_scenario": frames_per_scenario,
        "events_per_scenario": events_per_scenario,
        "frames_per_distance_bin": frames_per_distance_bin,
        "frames_per_fov_region": frames_per_fov_region,
        "max_consecutive_outlier_length": max(lengths) if lengths else 0,
        "median_outlier_event_length": median(lengths) if lengths else 0.0,
        "events": [asdict(event) for event in events],
    }
