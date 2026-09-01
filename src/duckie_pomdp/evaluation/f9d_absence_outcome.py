"""Pure long-absence outcome analysis for F9d-B.

B1 (natural out-of-domain), B2 (controlled in-domain detector dropout), and
B3 (controlled target removal) remain separate in every returned table.
No count is pooled to make a criterion easier to pass.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from duckie_pomdp.evaluation.f9d_absence_stress import (
    B1,
    B2,
    B3,
    AbsenceEvent,
    contiguous_absence_events,
)

CHECKPOINTS = (1, 5, 10, 20, 30, 40)
IN_DOMAIN = frozenset({"center", "mid_fov", "edge_fov"})


def analytic_existence_probability(
    initial_probability: float,
    frames: int,
    *,
    survival_probability: float,
    birth_probability: float,
) -> float:
    factor = survival_probability - birth_probability
    fixed_point = birth_probability / (1.0 - factor)
    return fixed_point + (initial_probability - fixed_point) * factor**frames


def _episode_rows(rows: Sequence[Mapping[str, Any]]) -> dict[str, list[Mapping[str, Any]]]:
    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row["episode"]), []).append(row)
    for episode in grouped:
        grouped[episode].sort(key=lambda row: int(row["frame"]))
    return grouped


def _row_by_frame(rows: Sequence[Mapping[str, Any]]) -> dict[int, Mapping[str, Any]]:
    return {int(row["frame"]): row for row in rows}


def _event_checkpoints(
    event: AbsenceEvent,
    episode_rows: Sequence[Mapping[str, Any]],
    *,
    survival_probability: float,
    birth_probability: float,
) -> dict[str, Any]:
    by_frame = _row_by_frame(episode_rows)
    previous = by_frame.get(event.start_frame - 1)
    initial_probability = (
        None if previous is None else float(previous["robust_b_existence_probability"])
    )
    values: dict[str, Any] = {}
    for checkpoint in CHECKPOINTS:
        if checkpoint > event.length:
            values[str(checkpoint)] = None
            continue
        row = by_frame[event.start_frame + checkpoint - 1]
        observed = float(row["robust_b_existence_probability"])
        analytic = (
            None
            if initial_probability is None
            else analytic_existence_probability(
                initial_probability,
                checkpoint,
                survival_probability=survival_probability,
                birth_probability=birth_probability,
            )
        )
        values[str(checkpoint)] = {
            "observed": observed,
            "analytic_prediction_only": analytic,
            "absolute_deviation": None if analytic is None else abs(observed - analytic),
            "runtime_observability_class": row.get("robust_b_observability_class"),
            "track_active": bool(row.get("robust_b_track_active")),
            "track_deleted": bool(row.get("robust_b_track_deleted")),
        }
    event_slice = [
        row
        for row in episode_rows
        if event.start_frame <= int(row["frame"]) <= event.end_frame
    ]
    return {
        "episode": event.episode,
        "seed": event.seed,
        "scenario": event.scenario,
        "kind": event.kind,
        "start_frame": event.start_frame,
        "end_frame": event.end_frame,
        "length": event.length,
        "initial_probability": initial_probability,
        "checkpoints": values,
        "track_active_throughout": all(bool(row.get("robust_b_track_active")) for row in event_slice),
        "track_deleted_during_event": any(bool(row.get("robust_b_track_deleted")) for row in event_slice),
    }


def _recovery_after_event(
    event: AbsenceEvent,
    episode_rows: Sequence[Mapping[str, Any]],
) -> int | None:
    for row in episode_rows:
        frame = int(row["frame"])
        if frame <= event.end_frame:
            continue
        if bool(row.get("detector_detected")) and bool(row.get("robust_b_track_active")):
            return frame - event.end_frame
    return None


def _b1_prediction_only_events(
    rows: Sequence[Mapping[str, Any]],
) -> list[AbsenceEvent]:
    """Contiguous B1 runs where invariant I3 truly applies.

    GT-out-of-FOV and runtime ``outside_domain`` are not sufficient by
    themselves: a positive detector output still applies detection evidence.
    The closed-form prediction recurrence is valid only when the detector is
    also silent.  Keeping this predicate explicit prevents a boundary
    detection from being misdiagnosed as a recurrence error.
    """

    events: list[AbsenceEvent] = []
    for episode, episode_rows in _episode_rows(rows).items():
        run: list[Mapping[str, Any]] = []
        for row in episode_rows:
            pure_prediction = (
                row.get("absence_kind") == B1
                and bool(row.get("gt_exists"))
                and not bool(row.get("eligible_visible"))
                and row.get("robust_b_observability_class") == "outside_domain"
                and not bool(row.get("detector_detected"))
            )
            if pure_prediction:
                run.append(row)
                continue
            if run:
                first, last = run[0], run[-1]
                events.append(
                    AbsenceEvent(
                        seed=int(first["seed"]),
                        scenario=str(first["scenario"]),
                        kind=B1,
                        episode=episode,
                        start_frame=int(first["frame"]),
                        end_frame=int(last["frame"]),
                        length=len(run),
                    )
                )
                run = []
        if run:
            first, last = run[0], run[-1]
            events.append(
                AbsenceEvent(
                    seed=int(first["seed"]),
                    scenario=str(first["scenario"]),
                    kind=B1,
                    episode=episode,
                    start_frame=int(first["frame"]),
                    end_frame=int(last["frame"]),
                    length=len(run),
                )
            )
    return events


def absence_outcome_metrics(
    rows: Sequence[Mapping[str, Any]],
    *,
    survival_probability: float,
    birth_probability: float,
    out_of_domain_floor: float,
    in_domain_ceiling: float,
    recurrence_tolerance: float,
    recovery_frames_max: int,
) -> dict[str, Any]:
    rows = list(rows)
    grouped = _episode_rows(rows)
    events = contiguous_absence_events(rows)
    result: dict[str, Any] = {}
    for kind in (B1, B2, B3):
        kind_events = [event for event in events if event.kind == kind]
        details = [
            _event_checkpoints(
                event,
                grouped[event.episode],
                survival_probability=survival_probability,
                birth_probability=birth_probability,
            )
            for event in kind_events
        ]
        recoveries = [
            _recovery_after_event(event, grouped[event.episode]) for event in kind_events
        ]
        observed_recoveries = [value for value in recoveries if value is not None]
        result[kind] = {
            "event_count": len(kind_events),
            "events": details,
            "recovery": {
                "recovered_event_count": len(observed_recoveries),
                "unrecovered_event_count": len(recoveries) - len(observed_recoveries),
                "mean_frames": (
                    None
                    if not observed_recoveries
                    else sum(observed_recoveries) / len(observed_recoveries)
                ),
                "max_frames": None if not observed_recoveries else max(observed_recoveries),
            },
        }

    b1_rows = [
        row
        for row in rows
        if row.get("absence_kind") == B1
        and bool(row.get("gt_exists"))
        and not bool(row.get("eligible_visible"))
    ]
    confusion: dict[str, int] = {}
    for row in b1_rows:
        key = str(row.get("robust_b_observability_class"))
        confusion[key] = confusion.get(key, 0) + 1
    result[B1]["gt_out_of_fov_vs_runtime_observability"] = confusion

    pure_b1_events = _b1_prediction_only_events(rows)
    pure_b1_details = [
        _event_checkpoints(
            event,
            grouped[event.episode],
            survival_probability=survival_probability,
            birth_probability=birth_probability,
        )
        for event in pure_b1_events
    ]
    result[B1]["prediction_only_events"] = pure_b1_details
    pure_b1_checkpoint_values = []
    for event in pure_b1_details:
        for checkpoint in CHECKPOINTS:
            value = event["checkpoints"][str(checkpoint)]
            if value is not None:
                pure_b1_checkpoint_values.append((checkpoint, value))
    floor_40 = [
        value["observed"]
        for checkpoint, value in pure_b1_checkpoint_values
        if checkpoint == 40
    ]
    deviations = [
        value["absolute_deviation"]
        for _, value in pure_b1_checkpoint_values
        if value["absolute_deviation"] is not None
    ]
    result[B1]["criterion"] = {
        "prediction_only_event_count": len(pure_b1_events),
        "checkpoint_40_count": len(floor_40),
        "floor": out_of_domain_floor,
        "all_checkpoint_40_above_floor": bool(floor_40)
        and all(value > out_of_domain_floor for value in floor_40),
        "maximum_analytic_absolute_deviation": None if not deviations else max(deviations),
        "recurrence_tolerance": recurrence_tolerance,
        "recurrence_matches": bool(deviations)
        and max(deviations) <= recurrence_tolerance,
    }

    b2_checkpoint_20 = [
        event["checkpoints"]["20"]["observed"]
        for event in result[B2]["events"]
        if event["checkpoints"]["20"] is not None
    ]
    result[B2]["criterion"] = {
        "checkpoint_20_count": len(b2_checkpoint_20),
        "ceiling": in_domain_ceiling,
        "below_ceiling_count": sum(
            value < in_domain_ceiling for value in b2_checkpoint_20
        ),
        "all_below_ceiling": bool(b2_checkpoint_20)
        and all(value < in_domain_ceiling for value in b2_checkpoint_20),
        "recovery_frames_max": recovery_frames_max,
        "all_observed_recoveries_within_limit": bool(
            result[B2]["recovery"]["recovered_event_count"]
        )
        and result[B2]["recovery"]["unrecovered_event_count"] == 0
        and result[B2]["recovery"]["max_frames"] <= recovery_frames_max,
        "interpretation": (
            "controlled dropout tests an unsupported in-domain track while the "
            "real pedestrian remains present; it is not natural disappearance"
        ),
    }
    result[B3]["interpretation"] = (
        "controlled target removal: privileged existence and rendered RGB both "
        "switch absent; no recovery is expected because the object does not return"
    )
    return result
