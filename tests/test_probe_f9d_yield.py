"""Task 3: regression test for the yield-probe's projection arithmetic.

Fix-round note: the first probe run (development seeds 8101-8108) computed
``projected = mean(per_dev_seed_rate) * (final_seed_count /
dev_seed_count)``. That divides by the development-seed count TWICE -- once
implicitly inside the mean, once again in the scale factor -- so an 8-seed
run projecting onto 4 final seeds silently shrank the projection by an
extra 8x (2.5 frames/seed averaged down to 1.25 total instead of scaled up
to 10.0 total). This test pins the fix: projecting a known per-seed rate
onto a known final-seed count must equal ``mean_rate * final_seed_count``,
not ``mean_rate * (final_seed_count / dev_seed_count)``.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))

import probe_f9d_yield as probe  # noqa: E402


def _summary(outlier_frames_per_seed, events_per_seed):
    return {
        "outlier_frames_per_seed": outlier_frames_per_seed,
        "events_per_seed": events_per_seed,
        "seeds_with_event": len(events_per_seed),
    }


def test_projection_scales_up_not_down_when_final_seeds_fewer_than_dev_seeds():
    """The exact regression: 8 dev seeds averaging 2.5 outlier frames/seed,
    projected onto 4 final seeds, must give 10.0 -- NOT 1.25."""

    development_seeds = (8101, 8102, 8103, 8104, 8105, 8106, 8107, 8108)
    outlier_frames_per_seed = {8101: 2, 8102: 5, 8103: 3, 8104: 1, 8105: 2, 8106: 4, 8107: 1, 8108: 2}
    events_per_seed = {8101: 2, 8102: 2, 8103: 3, 8104: 1, 8105: 2, 8106: 4, 8107: 1, 8108: 2}
    summary = _summary(outlier_frames_per_seed, events_per_seed)

    projection = probe.project_to_final_seeds(summary, development_seeds, 4)

    assert projection["projected_outlier_frames"] == 10.0
    assert projection["projected_outlier_events"] == 8.5
    assert projection["projected_seeds_with_event"] == 4


def test_projection_of_a_uniform_rate_scales_linearly_with_final_seed_count():
    """A trivial, easy-to-hand-verify case: every dev seed contributes
    exactly 4 outlier frames and 1 event. Projected onto N final seeds, the
    total must be exactly 4*N frames and 1*N events -- independent of how
    many development seeds were averaged over."""

    development_seeds = tuple(range(10))
    outlier_frames_per_seed = {seed: 4 for seed in development_seeds}
    events_per_seed = {seed: 1 for seed in development_seeds}
    summary = _summary(outlier_frames_per_seed, events_per_seed)

    for final_seed_count in (2, 4, 10, 20):
        projection = probe.project_to_final_seeds(summary, development_seeds, final_seed_count)
        assert projection["projected_outlier_frames"] == 4 * final_seed_count
        assert projection["projected_outlier_events"] == 1 * final_seed_count


def test_seeds_with_zero_events_are_included_at_zero_in_the_mean():
    """A dev seed that contributed no outlier frames/events must still
    count toward the per-seed MEAN (as a zero) -- dropping it would inflate
    the projection by averaging over fewer seeds than actually ran."""

    development_seeds = (1, 2, 3, 4)
    outlier_frames_per_seed = {1: 8, 2: 0, 3: 0, 4: 0}  # only seed 1 has frames
    events_per_seed = {1: 2}  # seeds 2-4 have zero events (absent from the dict)
    summary = _summary(outlier_frames_per_seed, events_per_seed)

    projection = probe.project_to_final_seeds(summary, development_seeds, 4)

    # mean frames = (8+0+0+0)/4 = 2.0 -> projected = 2.0 * 4 = 8.0
    assert projection["projected_outlier_frames"] == 8.0
    # mean events = (2+0+0+0)/4 = 0.5 -> projected = 0.5 * 4 = 2.0
    assert projection["projected_outlier_events"] == 2.0


def test_projected_seeds_with_event_uses_the_observed_fraction():
    development_seeds = (1, 2, 3, 4, 5, 6, 7, 8)
    events_per_seed = {seed: 1 for seed in development_seeds[:6]}  # 6/8 seeds had an event
    summary = _summary({}, events_per_seed)

    projection = probe.project_to_final_seeds(summary, development_seeds, 4)

    # 6/8 = 0.75 fraction * 4 final seeds = 3.0 -> round() = 3
    assert projection["projected_seeds_with_event"] == 3
