"""Gate F9c final-evaluation metrics: Baseline A vs Robust B.

Reuses ``duckie_pomdp.evaluation.f9_belief.scalar_error_metrics`` and
``belief_metrics`` for every quantity F9 already defines correctly (RMSE,
bias, coverage_68/95, mean_marginal_nll, existence summaries) rather than
recomputing them. This module adds only what F9's evaluation never needed:
coverage-error scalars, a std/RMSE overconfidence ratio, natural-miss-run
checkpoints, the three-way miss breakdown invariants I2/I3 require, outlier
impact, safety bias, and the pre-registered support check.

**Row convention.** Every function here expects one row per rendered frame
with a shared, system-independent ``detector_detected`` column (the same
single YOLO inference decides whether *any* candidate existed this frame,
for both systems) plus two per-system column families using the same
``{variant}_belief_*`` / ``{variant}_existence_probability`` /
``{variant}_nis`` naming ``f9_belief.py`` already established, with
``variant`` in ``("baseline_a", "robust_b")``. This is what lets
``belief_metrics``/``track_continuity`` be reused unmodified.

**Never pool miss-class retention (see ``robustness_metrics``).** Invariant
I3 makes ``detector_miss_outside_domain`` almost free to survive (no
likelihood is ever applied to it), so a single pooled "belief maintained
during a miss" number is dominated by whichever class is more common and
can read high while the belief has actually collapsed on every genuine
in-domain miss. The three counts below are always reported disjointly, and
retention is always reported per class.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np

from duckie_pomdp.evaluation.f9_belief import (
    belief_metrics,
    scalar_error_metrics,
    track_continuity,
)

DEFAULT_MISS_RUN_LENGTHS: tuple[int, ...] = (1, 3, 5, 10)
IN_DOMAIN_SUPPORT_MINIMUM = 20
BELIEF_VARIABLE_NAMES: tuple[str, ...] = (
    "range",
    "bearing",
    "range_rate",
    "bearing_rate",
)
IN_DOMAIN_OBSERVABILITY_CLASSES: frozenset[str] = frozenset({"center", "mid_fov", "edge_fov"})


# ---------------------------------------------------------------------------
# Coverage-error / overconfidence augmentation of f9_belief.belief_metrics.
# ---------------------------------------------------------------------------


def augment_belief_metrics_with_calibration(metrics: Mapping[str, Any]) -> dict[str, Any]:
    """Add ``coverage_error_68``/``coverage_error_95``/``std_over_rmse`` to
    every per-variable sub-dict of a ``f9_belief.belief_metrics`` result, in
    place semantics but returning the (mutated) dict for convenience.

    Named exactly ``coverage_error_68``/``coverage_error_95`` per the gate
    brief -- these are NOT "ECE" (expected calibration error), which is a
    different, histogram-based quantity; this is simply the absolute gap
    between the observed empirical coverage at the nominal 68%/95%
    intervals and the nominal target.
    """

    for name in BELIEF_VARIABLE_NAMES:
        stats = metrics.get(name)
        if not isinstance(stats, dict):
            continue
        if "coverage_68" in stats and stats["coverage_68"] is not None:
            stats["coverage_error_68"] = abs(float(stats["coverage_68"]) - 0.68)
        if "coverage_95" in stats and stats["coverage_95"] is not None:
            stats["coverage_error_95"] = abs(float(stats["coverage_95"]) - 0.95)
        rmse = stats.get("rmse")
        std = stats.get("mean_predicted_std")
        stats["std_over_rmse"] = (
            None if not rmse or std is None else float(std) / float(rmse)
        )
    return dict(metrics)


def belief_metrics_calibrated(rows: Sequence[Mapping[str, Any]], *, variant: str) -> dict[str, Any]:
    """``f9_belief.belief_metrics`` plus this module's calibration fields."""

    return augment_belief_metrics_with_calibration(
        belief_metrics(list(rows), variant=variant)
    )


# ---------------------------------------------------------------------------
# Natural-miss-run checkpoints.
# ---------------------------------------------------------------------------


def _episode_rows_sorted_by_frame(
    rows: Sequence[Mapping[str, Any]]
) -> dict[str, list[Mapping[str, Any]]]:
    by_episode: dict[str, list[Mapping[str, Any]]] = {}
    for row in rows:
        by_episode.setdefault(str(row["episode"]), []).append(row)
    for episode_rows in by_episode.values():
        episode_rows.sort(key=lambda row: int(row["frame"]))
    return by_episode


def _natural_miss_runs(
    rows: Sequence[Mapping[str, Any]]
) -> list[tuple[list[Mapping[str, Any]], int, int]]:
    """Consecutive-run boundaries of GENUINE natural misses -- GT
    ``eligible_visible`` True and the shared ``detector_detected`` False.
    Returns ``(episode_rows, start_index, end_index)`` with both indices
    0-based and inclusive, into that episode's frame-ordered row list."""

    runs: list[tuple[list[Mapping[str, Any]], int, int]] = []
    for episode_rows in _episode_rows_sorted_by_frame(rows).values():
        start: int | None = None
        for index, row in enumerate(episode_rows):
            is_miss = bool(row["eligible_visible"]) and not bool(row["detector_detected"])
            if is_miss:
                if start is None:
                    start = index
                continue
            if start is not None:
                runs.append((episode_rows, start, index - 1))
                start = None
        if start is not None:
            runs.append((episode_rows, start, len(episode_rows) - 1))
    return runs


def _is_active(row: Mapping[str, Any], *, variant: str, active_probability_threshold: float) -> bool:
    return bool(row[f"{variant}_belief_initialized"]) and float(
        row[f"{variant}_existence_probability"]
    ) >= active_probability_threshold


def miss_sequence_metrics(
    rows: Sequence[Mapping[str, Any]],
    *,
    variant: str,
    run_lengths: tuple[int, ...] = DEFAULT_MISS_RUN_LENGTHS,
    active_probability_threshold: float = 0.5,
    label: str = "natural",
) -> dict[str, Any]:
    """Belief state at natural-miss-run checkpoints, for one belief variant.

    For each target run length ``L`` in ``run_lengths``, considers every
    consecutive genuine-natural-miss run (GT ``eligible_visible`` True,
    shared ``detector_detected`` False, see ``_natural_miss_runs``) at
    least ``L`` frames long, reads ``variant``'s belief at the ``L``-th
    frame of that run, and reports the mean existence probability, mean
    reported range std, fraction with an active track, and -- among the
    subset of those runs that eventually see the belief become active again
    within the same episode -- the mean number of frames from run-end to
    that recovery.

    ``label`` must be ``"natural"`` for genuine miss runs (the only kind
    this function itself ever constructs) and ``"synthetic"`` for any
    caller-injected miss sequence, reported under a separate key -- this
    module never injects a synthetic sequence on its own; a caller wanting
    one must build its own row list and call this function again with
    ``label="synthetic"``.
    """

    if label not in ("natural", "synthetic"):
        raise ValueError(f"label must be 'natural' or 'synthetic', got {label!r}")

    runs = _natural_miss_runs(rows) if label == "natural" else []
    run_lengths_observed = [end - start + 1 for _, start, end in runs]

    checkpoints: dict[str, Any] = {}
    for length in run_lengths:
        eligible_runs = [
            (episode_rows, start, end)
            for episode_rows, start, end in runs
            if (end - start + 1) >= length
        ]
        checkpoint_rows = [
            episode_rows[start + length - 1] for episode_rows, start, _ in eligible_runs
        ]
        existence = [
            float(row[f"{variant}_existence_probability"]) for row in checkpoint_rows
        ]
        range_std = [
            float(row[f"{variant}_belief_range_std_m"])
            for row in checkpoint_rows
            if row.get(f"{variant}_belief_range_std_m") is not None
        ]
        active = [
            _is_active(
                row, variant=variant, active_probability_threshold=active_probability_threshold
            )
            for row in checkpoint_rows
        ]

        recovery_frames: list[int] = []
        for episode_rows, _start, end in eligible_runs:
            recovery = next(
                (
                    offset + 1
                    for offset, later in enumerate(episode_rows[end + 1 :])
                    if _is_active(
                        later,
                        variant=variant,
                        active_probability_threshold=active_probability_threshold,
                    )
                ),
                None,
            )
            if recovery is not None:
                recovery_frames.append(recovery)

        checkpoints[str(length)] = {
            "eligible_run_count": len(eligible_runs),
            "mean_existence_probability": (
                None if not existence else float(np.mean(existence))
            ),
            "mean_reported_range_std_m": (
                None if not range_std else float(np.mean(range_std))
            ),
            "active_fraction": None if not active else float(np.mean(active)),
            "recovered_run_count": len(recovery_frames),
            "mean_frames_to_recovery": (
                None if not recovery_frames else float(np.mean(recovery_frames))
            ),
        }

    return {
        "label": label,
        "run_count": len(runs),
        "run_length_distribution": {
            "mean": (
                None if not run_lengths_observed else float(np.mean(run_lengths_observed))
            ),
            "median": (
                None if not run_lengths_observed else float(np.median(run_lengths_observed))
            ),
            "max": max(run_lengths_observed, default=0),
        },
        "run_length_checkpoints": checkpoints,
    }


# ---------------------------------------------------------------------------
# Robustness metrics: the three-way miss breakdown (invariants I2/I3) plus
# association/gate/track-lifecycle diagnostics.
# ---------------------------------------------------------------------------


def _retention(rows: Sequence[Mapping[str, Any]], *, active_probability_threshold: float) -> dict[str, Any]:
    if not rows:
        return {"frame_count": 0, "active_belief_retained": 0, "retention_fraction": None}
    retained = sum(
        1
        for row in rows
        if _is_active(
            row,
            variant="robust_b",
            active_probability_threshold=active_probability_threshold,
        )
    )
    return {
        "frame_count": len(rows),
        "active_belief_retained": retained,
        "retention_fraction": retained / len(rows),
    }


def _lifecycle_events(rows: Sequence[Mapping[str, Any]]) -> tuple[int, int, int]:
    """Per-episode track-lifecycle counts, in frame order.

    ``false_track_initializations``: an ``"initialization"`` frame whose
    associated candidate did NOT actually match ground truth
    (``selected_correct_iou50`` False).
    ``track_deletions``: frames the coordinator itself flagged
    ``robust_b_track_deleted``.
    ``recoveries``: ``"initialization"`` frames that are NOT the first
    track this episode ever had -- i.e. the track existed before, was lost
    (deleted or otherwise dropped back to ``no_track``), and is now being
    re-established. The very first initialization in an episode is not a
    recovery; there was nothing to recover from.
    """

    false_track_initializations = 0
    track_deletions = 0
    recoveries = 0
    for episode_rows in _episode_rows_sorted_by_frame(rows).values():
        has_had_a_track = False
        for row in episode_rows:
            mode = row.get("robust_b_frame_mode")
            if mode == "initialization":
                if not bool(row.get("selected_correct_iou50")):
                    false_track_initializations += 1
                if has_had_a_track:
                    recoveries += 1
                has_had_a_track = True
            if bool(row.get("robust_b_track_deleted")):
                track_deletions += 1
    return false_track_initializations, track_deletions, recoveries


def robustness_metrics(
    rows: Sequence[Mapping[str, Any]],
    *,
    active_probability_threshold: float = 0.5,
    matching_iou_threshold: float = 0.5,
    in_domain_support_minimum: int = IN_DOMAIN_SUPPORT_MINIMUM,
) -> dict[str, Any]:
    """Robust-B robustness diagnostics, centered on the invariant-I2/I3
    three-way miss breakdown.

    Every row is classified into exactly one of three disjoint buckets
    (frames where the GT calls the pedestrian invisible -- i.e. NOT
    ``eligible_visible`` -- are excluded from all three, since they are not
    a miss at all):

    - ``detector_miss_in_domain``: no raw candidate this frame, and the
      coordinator's OWN predicted observability was CENTER/MID_FOV/EDGE_FOV
      (a class where the miss likelihood is actually applied). This is the
      only class where P_D^eff is doing any work, and the ONLY class this
      module treats as a control-readiness criterion.
    - ``detector_miss_outside_domain``: no raw candidate this frame, and
      predicted observability was OUTSIDE_DOMAIN -- invariant I3 applies no
      likelihood at all here, so retention is reported as a sanity check
      only ("belief doesn't collapse just because the camera isn't
      looking"), never as a pass/fail criterion.
    - ``gated_rejection``: a raw candidate existed and reached association,
      but the innovation gate rejected it. Invariant I2: this is scored as
      a DETECTION for existence purposes, never as a miss.

    Retention (fraction of frames in a bucket where the belief remained
    active, ``track_active AND existence_probability >= threshold``) is
    reported separately per bucket and is never pooled -- see the module
    docstring for why a pooled figure would be misleading.
    """

    visible = [row for row in rows if bool(row["eligible_visible"])]

    in_domain = [
        row
        for row in visible
        if not bool(row["detector_detected"])
        and str(row.get("robust_b_observability_class")) in IN_DOMAIN_OBSERVABILITY_CLASSES
    ]
    outside_domain = [
        row
        for row in visible
        if not bool(row["detector_detected"])
        and str(row.get("robust_b_observability_class")) == "outside_domain"
    ]
    gated_rejection = [
        row
        for row in visible
        if bool(row["detector_detected"])
        and row.get("robust_b_gate_decision") == "rejected"
    ]

    in_domain_retention = _retention(
        in_domain, active_probability_threshold=active_probability_threshold
    )
    outside_domain_retention = _retention(
        outside_domain, active_probability_threshold=active_probability_threshold
    )
    gated_rejection_retention = _retention(
        gated_rejection, active_probability_threshold=active_probability_threshold
    )

    accepted_by_gate = sum(
        1
        for row in visible
        if bool(row["detector_detected"]) and row.get("robust_b_gate_decision") == "accepted"
    )
    localization_outliers = [
        row
        for row in visible
        if row.get("robust_b_associated_iou") not in (None, "")
        and float(row["robust_b_associated_iou"]) < matching_iou_threshold
    ]
    duplicate_frames = sum(1 for row in rows if bool(row.get("duplicate_selection")))
    wrong_association_events = sum(
        1
        for row in visible
        if bool(row.get("robust_b_association_differed_from_highest_confidence"))
        and row.get("robust_b_associated_iou") not in (None, "")
        and float(row["robust_b_associated_iou"]) < matching_iou_threshold
    )
    false_track_initializations, track_deletions, recoveries = _lifecycle_events(rows)

    return {
        "matching_iou_threshold": matching_iou_threshold,
        "active_probability_threshold": active_probability_threshold,
        "miss_breakdown": {
            "detector_miss_in_domain": in_domain_retention,
            "detector_miss_outside_domain": outside_domain_retention,
            "gated_rejection": gated_rejection_retention,
        },
        "in_domain_control_readiness": {
            "frame_count": in_domain_retention["frame_count"],
            "minimum_required": in_domain_support_minimum,
            "under_powered": in_domain_retention["frame_count"] < in_domain_support_minimum,
            "retention_fraction": in_domain_retention["retention_fraction"],
        },
        "gate_accept_reject": {
            "accepted": accepted_by_gate,
            "rejected": len(gated_rejection),
        },
        "localization_outlier_count": len(localization_outliers),
        "duplicate_frames": duplicate_frames,
        "wrong_association_events": wrong_association_events,
        "natural_miss_count": in_domain_retention["frame_count"]
        + outside_domain_retention["frame_count"],
        "false_track_initializations": false_track_initializations,
        "track_deletions": track_deletions,
        "recoveries": recoveries,
    }


# ---------------------------------------------------------------------------
# Outlier impact and safety bias.
# ---------------------------------------------------------------------------


def outlier_impact(
    rows: Sequence[Mapping[str, Any]],
    *,
    matching_iou_threshold: float = 0.5,
) -> dict[str, Any]:
    """Measurement RMSE, Baseline-A RMSE, Robust-B RMSE, and max transient
    belief error, restricted to GT-labelled outlier frames: frames where a
    detection existed but its localization was wrong
    (``eligible_visible AND detector_detected AND selected_correct_iou50``
    is False, i.e. IoU vs GT < ``matching_iou_threshold`` or no correct
    match at all)."""

    outliers = [
        row
        for row in rows
        if bool(row["eligible_visible"])
        and bool(row["detector_detected"])
        and not bool(row.get("selected_correct_iou50"))
    ]

    measurement_errors = [
        float(row["raw_range_m"]) - float(row["gt_range_m"])
        for row in outliers
        if row.get("raw_range_m") not in (None, "")
    ]

    def _belief_errors(variant: str) -> list[float]:
        return [
            float(row[f"{variant}_belief_range_m"]) - float(row["gt_range_m"])
            for row in outliers
            if bool(row[f"{variant}_belief_initialized"])
        ]

    baseline_errors = _belief_errors("baseline_a")
    robust_errors = _belief_errors("robust_b")

    return {
        "matching_iou_threshold": matching_iou_threshold,
        "outlier_frame_count": len(outliers),
        "measurement_range_rmse": scalar_error_metrics(measurement_errors),
        "baseline_a_belief_range_rmse": scalar_error_metrics(baseline_errors),
        "robust_b_belief_range_rmse": scalar_error_metrics(robust_errors),
        "baseline_a_max_transient_belief_range_error_m": (
            None if not baseline_errors else float(np.max(np.abs(baseline_errors)))
        ),
        "robust_b_max_transient_belief_range_error_m": (
            None if not robust_errors else float(np.max(np.abs(robust_errors)))
        ),
    }


def safety_bias(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """``E[mu_r - r_GT]`` for both systems, over frames each system's own
    belief was initialized. Positive means the pedestrian is believed
    farther away than it actually is (a conservative error for a
    collision-avoidance consumer); negative means the pedestrian is
    believed closer than it actually is (an optimistic, riskier error)."""

    def _mean_signed_range_error(variant: str) -> float | None:
        errors = [
            float(row[f"{variant}_belief_range_m"]) - float(row["gt_range_m"])
            for row in rows
            if bool(row[f"{variant}_belief_initialized"])
        ]
        return None if not errors else float(np.mean(errors))

    return {
        "sign_interpretation": (
            "positive == pedestrian believed FARTHER than reality (conservative); "
            "negative == pedestrian believed CLOSER than reality (optimistic, "
            "riskier for a downstream collision-avoidance consumer)"
        ),
        "baseline_a_mean_signed_range_error_m": _mean_signed_range_error("baseline_a"),
        "robust_b_mean_signed_range_error_m": _mean_signed_range_error("robust_b"),
    }


# ---------------------------------------------------------------------------
# Pre-registered support check.
# ---------------------------------------------------------------------------


def support_check(
    rows: Sequence[Mapping[str, Any]],
    minimum_support: Mapping[str, int],
) -> dict[str, Any]:
    """Per-bin GT-eligible-visible frame counts against the pre-registered
    minima, and a boolean ``satisfied``. Must be read before any accuracy or
    calibration metric -- see the gate brief."""

    visible = [row for row in rows if bool(row["eligible_visible"])]
    counts = {
        "near": sum(1 for row in visible if row.get("distance_bin") == "near"),
        "medium": sum(1 for row in visible if row.get("distance_bin") == "medium"),
        "far": sum(1 for row in visible if row.get("distance_bin") == "far"),
        "edge_fov": sum(1 for row in visible if row.get("fov_region") == "edge_fov"),
    }
    shortfalls = {
        name: int(minimum) - counts.get(name, 0)
        for name, minimum in minimum_support.items()
        if counts.get(name, 0) < minimum
    }
    return {
        "counts": counts,
        "minimum_support": dict(minimum_support),
        "shortfalls": shortfalls,
        "satisfied": not shortfalls,
    }


# ---------------------------------------------------------------------------
# Top-level report assembly.
# ---------------------------------------------------------------------------


def summarize_f9c(rows: list[dict[str, Any]], *, protocol: Any) -> tuple[dict[str, Any], dict[str, Any]]:
    """Assemble the full F9c final-evaluation report: ``(metrics, nis)``,
    mirroring ``f9_belief.summarize_f9``'s shape.

    ``protocol`` is the loaded ``F9cProtocol``; the coordinator-config
    thresholds (existence-active threshold, innovation-gate chi-square
    threshold) are read from ``protocol.config_path`` via
    ``evaluation.f9c_calibration.load_robust_observation_config`` rather
    than duplicated onto ``F9cProtocol`` itself, which exists for
    freeze-boundary guards, not rendering/reporting plumbing (see
    ``f9c_protocol.py``'s own module docstring).
    """

    # Imported here, not at module scope, to avoid a needless import-time
    # dependency for callers (e.g. this module's own unit tests) that only
    # exercise the pure row-metric functions above with synthetic protocol
    # stand-ins.
    from duckie_pomdp.evaluation.f9c_calibration import load_robust_observation_config

    robust_config = load_robust_observation_config(protocol.config_path)
    active_threshold = robust_config.active_threshold
    nis_threshold = robust_config.gate.chi_square_threshold
    variants = ("baseline_a", "robust_b")

    ekf = {variant: belief_metrics_calibrated(rows, variant=variant) for variant in variants}
    continuity = {
        variant: track_continuity(
            rows, variant=variant, active_probability_threshold=active_threshold
        )
        for variant in variants
    }
    nis = {
        variant: _nis_summary(rows, variant=variant, threshold=nis_threshold)
        for variant in variants
    }

    metrics: dict[str, Any] = {
        "row_count": len(rows),
        "ekf": ekf,
        "track_continuity": continuity,
        "nis": nis,
        "miss_sequence": {
            variant: miss_sequence_metrics(
                rows, variant=variant, active_probability_threshold=active_threshold
            )
            for variant in variants
        },
        "robustness": robustness_metrics(rows, active_probability_threshold=active_threshold),
        "outlier_impact": outlier_impact(rows),
        "safety_bias": safety_bias(rows),
        "support_check": support_check(rows, protocol.minimum_support),
        "acceptance": {
            "coverage_68_low": protocol.acceptance.coverage_68_low,
            "coverage_68_high": protocol.acceptance.coverage_68_high,
            "coverage_95_low": protocol.acceptance.coverage_95_low,
            "coverage_95_high": protocol.acceptance.coverage_95_high,
            "max_std_over_rmse": protocol.acceptance.max_std_over_rmse,
            "max_rmse_ratio_vs_baseline": protocol.acceptance.max_rmse_ratio_vs_baseline,
        },
    }
    for grouping, values in (
        ("distance_bin", ("near", "medium", "far")),
        ("fov_region", ("center", "mid_fov", "edge_fov", "outside")),
    ):
        metrics[f"by_{grouping}"] = {
            value: {
                # Named frame_count, not row_count: a generic "row_count"
                # sub-key here would collide with
                # verify_f9c_artifacts.py's top-level row_count == len(rows)
                # cross-check, which walks every key literally named
                # row_count anywhere in the document -- this is a per-bin
                # SUBSET count, not the total.
                "frame_count": len(subset := [row for row in rows if row.get(grouping) == value]),
                "baseline_a_ekf": belief_metrics_calibrated(subset, variant="baseline_a"),
                "robust_b_ekf": belief_metrics_calibrated(subset, variant="robust_b"),
            }
            for value in values
        }

    return metrics, {"global": nis}


def _nis_summary(
    rows: Sequence[Mapping[str, Any]], *, variant: str, threshold: float
) -> dict[str, Any]:
    values = np.asarray(
        [
            float(row[f"{variant}_nis"])
            for row in rows
            if row.get(f"{variant}_nis") not in (None, "")
        ],
        dtype=float,
    )
    if not len(values):
        return {"count": 0}
    return {
        "count": int(len(values)),
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "p95": float(np.quantile(values, 0.95)),
        "fraction_above_chi_square_2d_95": float(np.mean(values > threshold)),
        "chi_square_2d_95_threshold": threshold,
    }
