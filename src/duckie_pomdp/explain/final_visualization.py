"""Pure helpers for the final F11 explanation visualization.

This module only joins already-frozen public policy traces, R004 attribution
summaries, and a separately stored evaluation-only pose trace.  It never runs
the policy, detector, simulator, or attribution algorithm.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
from numpy.typing import NDArray


GROUP_ORDER = (
    "Lane",
    "Ego",
    "StopLine",
    "Pedestrian",
    "Stop",
    "PreviousAction",
)

PHASE_ORDER = (
    "nominal",
    "lane_curve",
    "pedestrian_relevant",
    "stop_required",
    "stop_satisfied",
)

TARGET_ORDER = ("v_cmd_mps", "omega_cmd_rad_s")

# Okabe-Ito-derived, colorblind-safe semantic identity.  Labels are always
# rendered alongside colors, so no figure relies on hue alone.
GROUP_COLOURS = {
    "Lane": "#0072B2",
    "Ego": "#56B4E9",
    "StopLine": "#009E73",
    "Pedestrian": "#E69F00",
    "Stop": "#D55E00",
    "PreviousAction": "#CC79A7",
}

PHASE_LABELS = {
    "nominal": "Nominal",
    "lane_curve": "Lane curve",
    "pedestrian_relevant": "Pedestrian relevant",
    "stop_required": "Stop required",
    "stop_satisfied": "Stop satisfied",
}


@dataclass(frozen=True)
class RepresentativeFrame:
    phase: str
    local_index: int
    simulator_step: int
    rule: str
    segment_start: int
    segment_end: int


def longest_contiguous_segment(indices: Sequence[int]) -> NDArray[np.int64]:
    """Return the longest contiguous run, breaking ties by earliest start."""

    values = np.asarray(indices, dtype=np.int64)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("indices must be a non-empty vector")
    if np.any(np.diff(values) <= 0):
        raise ValueError("indices must be strictly increasing")
    segments = np.split(values, np.flatnonzero(np.diff(values) > 1) + 1)
    return max(segments, key=lambda segment: (segment.size, -int(segment[0])))


def select_representative_frames(
    *,
    phases: Sequence[str],
    steps: Sequence[int],
    physical_observation: NDArray[np.floating],
    feature_names: Sequence[str],
    pedestrian_min_existence: float = 0.9,
) -> dict[str, RepresentativeFrame]:
    """Select deterministic public-only representative states.

    Midpoints of the longest continuous segment are used for every phase
    except ``pedestrian_relevant``.  That phase uses the state whose valid
    pedestrian range is nearest the median among strong public beliefs.  No
    attribution, world pose, RGB content, or privileged label enters selection.
    """

    phase_values = np.asarray(phases, dtype="U40")
    step_values = np.asarray(steps, dtype=np.int64)
    physical = np.asarray(physical_observation, dtype=np.float64)
    names = tuple(str(name) for name in feature_names)
    if phase_values.ndim != 1 or step_values.shape != phase_values.shape:
        raise ValueError("phase and step vectors must have matching shape")
    if physical.shape != (phase_values.size, len(names)):
        raise ValueError("physical observation shape does not match feature schema")
    if not np.all(np.isfinite(physical)):
        raise ValueError("physical observations must be finite")
    if len(names) != len(set(names)):
        raise ValueError("feature names must be unique")

    p_index = names.index("pedestrian_existence_probability")
    r_index = names.index("pedestrian_range_mean_m")
    selected: dict[str, RepresentativeFrame] = {}
    for phase in PHASE_ORDER:
        candidates = np.flatnonzero(phase_values == phase)
        if candidates.size == 0:
            raise ValueError(f"qualitative trace has no {phase!r} phase")
        segment = longest_contiguous_segment(candidates)
        if phase == "pedestrian_relevant":
            strong = segment[
                (physical[segment, p_index] >= pedestrian_min_existence)
                & (physical[segment, r_index] > 0.0)
            ]
            if strong.size == 0:
                raise ValueError("pedestrian segment has no strong valid public belief")
            ranges = physical[strong, r_index]
            median = float(np.median(ranges))
            local_index = int(strong[np.argmin(np.abs(ranges - median))])
            rule = (
                "nearest median valid pedestrian range within the longest "
                f"segment, restricted to P(e)>={pedestrian_min_existence:g}"
            )
        else:
            local_index = int(segment[segment.size // 2])
            rule = "temporal midpoint of the longest continuous phase segment"
        selected[phase] = RepresentativeFrame(
            phase=phase,
            local_index=local_index,
            simulator_step=int(step_values[local_index]),
            rule=rule,
            segment_start=int(step_values[int(segment[0])]),
            segment_end=int(step_values[int(segment[-1])]),
        )
    return selected


def validate_group_summary_rows(rows: Sequence[Mapping[str, object]]) -> None:
    """Require one exact R004 row for each target/phase/group cell."""

    seen: set[tuple[str, str, str]] = set()
    valid_phases = ("all",) + PHASE_ORDER
    for row in rows:
        key = (str(row["target"]), str(row["public_phase"]), str(row["group"]))
        if key in seen:
            raise ValueError(f"duplicate attribution summary row: {key}")
        seen.add(key)
        share = float(row["mean_absolute_group_share"])
        low = float(row["share_ci_low"])
        high = float(row["share_ci_high"])
        if not (0.0 <= share <= 1.0 and 0.0 <= low <= share <= high <= 1.0):
            raise ValueError(f"invalid attribution share/CI for {key}")
    expected = {
        (target, phase, group)
        for target in TARGET_ORDER
        for phase in valid_phases
        for group in GROUP_ORDER
    }
    if seen != expected:
        missing = sorted(expected - seen)
        extra = sorted(seen - expected)
        raise ValueError(f"summary grid mismatch; missing={missing}, extra={extra}")
    for target in TARGET_ORDER:
        for phase in valid_phases:
            total = sum(
                float(row["mean_absolute_group_share"])
                for row in rows
                if str(row["target"]) == target
                and str(row["public_phase"]) == phase
            )
            if not np.isclose(total, 1.0, atol=1.0e-9):
                raise ValueError(f"group shares do not sum to one for {target}/{phase}")


def summary_matrix(
    rows: Sequence[Mapping[str, object]], *, target: str
) -> NDArray[np.float64]:
    """Return the exact phase x group share matrix in frozen display order."""

    validate_group_summary_rows(rows)
    if target not in TARGET_ORDER:
        raise ValueError(f"unknown target {target!r}")
    lookup = {
        (str(row["target"]), str(row["public_phase"]), str(row["group"])): float(
            row["mean_absolute_group_share"]
        )
        for row in rows
    }
    return np.asarray(
        [
            [lookup[(target, phase, group)] for group in GROUP_ORDER]
            for phase in PHASE_ORDER
        ],
        dtype=np.float64,
    )


def phase_runs(phases: Sequence[str]) -> list[tuple[int, int, str]]:
    """Encode a phase vector as inclusive contiguous runs."""

    values = np.asarray(phases, dtype="U40")
    if values.ndim != 1 or values.size == 0:
        raise ValueError("phases must be a non-empty vector")
    changes = np.r_[0, np.flatnonzero(values[1:] != values[:-1]) + 1, values.size]
    return [
        (int(start), int(end - 1), str(values[start]))
        for start, end in zip(changes[:-1], changes[1:])
    ]


def pedestrian_belief_world(
    *,
    ego_x_m: float,
    ego_z_m: float,
    ego_heading_rad: float,
    range_mean_m: float,
    range_std_m: float,
    bearing_mean_rad: float,
    bearing_std_rad: float,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Map public polar pedestrian belief to world mean/covariance for display.

    The transformation follows the project convention ``x=left``,
    ``y=forward``, ``beta=atan2(x,y)``.  This is a visualization transform;
    simulator world pose remains evaluation-only and never enters PPO or IG.
    """

    values = np.asarray(
        [
            ego_x_m,
            ego_z_m,
            ego_heading_rad,
            range_mean_m,
            range_std_m,
            bearing_mean_rad,
            bearing_std_rad,
        ],
        dtype=np.float64,
    )
    if not np.all(np.isfinite(values)):
        raise ValueError("belief/world transform inputs must be finite")
    if range_mean_m < 0.0 or range_std_m < 0.0 or bearing_std_rad < 0.0:
        raise ValueError("range and standard deviations cannot be negative")

    r = float(range_mean_m)
    beta = float(bearing_mean_rad)
    x_left = r * np.sin(beta)
    y_forward = r * np.cos(beta)
    heading = float(ego_heading_rad)
    ego_to_world = np.asarray(
        [
            [np.sin(heading), np.cos(heading)],
            [np.cos(heading), -np.sin(heading)],
        ],
        dtype=np.float64,
    )
    mean = np.asarray([ego_x_m, ego_z_m], dtype=np.float64) + ego_to_world @ np.asarray(
        [x_left, y_forward], dtype=np.float64
    )
    polar_to_ego = np.asarray(
        [
            [np.sin(beta), r * np.cos(beta)],
            [np.cos(beta), -r * np.sin(beta)],
        ],
        dtype=np.float64,
    )
    polar_covariance = np.diag([range_std_m**2, bearing_std_rad**2])
    ego_covariance = polar_to_ego @ polar_covariance @ polar_to_ego.T
    world_covariance = ego_to_world @ ego_covariance @ ego_to_world.T
    return mean, world_covariance

