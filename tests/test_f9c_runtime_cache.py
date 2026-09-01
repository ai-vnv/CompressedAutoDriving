"""Task 11: runtime cache round-trip -- invariants I4 and I5.

All synthetic data. Never touches the simulator, the detector, or seeds
7101-7104.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from duckie_pomdp.evaluation.f9_protocol import sha256
from duckie_pomdp.evaluation.f9c_runtime_cache import (
    RuntimeCacheFrame,
    TruthFrame,
    read_evaluation_truth,
    read_runtime_cache,
    write_evaluation_truth,
    write_runtime_cache,
)


def _frame(
    *,
    episode: str = "evaluation_7101_stationary_ped_stationary_ego",
    frame: int = 0,
    seed: int = 7101,
    scenario: str = "stationary_ped_stationary_ego",
    candidates: tuple[tuple[float, float, float, tuple, bool], ...] = (),
) -> RuntimeCacheFrame:
    return RuntimeCacheFrame(
        episode=episode,
        seed=seed,
        scenario=scenario,
        frame=frame,
        dt_s=1.0 / 30.0,
        raw_candidate_range_m=tuple(c[0] for c in candidates),
        raw_candidate_bearing_rad=tuple(c[1] for c in candidates),
        raw_candidate_confidence=tuple(c[2] for c in candidates),
        raw_candidate_bbox=tuple(c[3] for c in candidates),
        raw_candidate_projection_failed=tuple(c[4] for c in candidates),
        ego_linear_velocity_mps=0.2,
        ego_yaw_rate_rad_s=-0.05,
    )


def test_frame_rejects_mismatched_candidate_field_lengths():
    with pytest.raises(ValueError):
        RuntimeCacheFrame(
            episode="e",
            seed=1,
            scenario="s",
            frame=0,
            dt_s=1.0 / 30.0,
            raw_candidate_range_m=(1.0, 2.0),
            raw_candidate_bearing_rad=(0.1,),  # too short
            raw_candidate_confidence=(0.9, 0.5),
            raw_candidate_bbox=((0, 0, 1, 1), (0, 0, 1, 1)),
            raw_candidate_projection_failed=(False, False),
            ego_linear_velocity_mps=0.0,
            ego_yaw_rate_rad_s=0.0,
        )


def test_raw_candidate_count_matches_the_stored_arrays():
    frame = _frame(
        candidates=(
            (0.5, 0.1, 0.9, (10.0, 20.0, 30.0, 40.0), False),
            (float("nan"), float("nan"), 0.3, (100.0, 5.0, 140.0, 90.0), True),
        )
    )
    assert frame.raw_candidate_count == 2


def test_write_then_read_round_trips_every_field_exactly(tmp_path: Path):
    frames = (
        _frame(
            frame=0,
            candidates=(
                (0.812345, -0.13579, 0.87, (12.0, 34.0, 56.0, 78.0), False),
                (1.9, 0.02, 0.11, (200.5, 10.25, 260.75, 90.0), False),
            ),
        ),
        _frame(frame=1, candidates=()),
        _frame(
            frame=2,
            candidates=((float("nan"), float("nan"), 0.42, (1.0, 2.0, 3.0, 4.0), True),),
        ),
    )

    path = tmp_path / "cache.npz"
    returned_hash = write_runtime_cache(path, frames)
    assert returned_hash == sha256(path)

    read_back = read_runtime_cache(path)
    assert len(read_back) == len(frames)
    for original, restored in zip(frames, read_back):
        assert restored.episode == original.episode
        assert restored.seed == original.seed
        assert restored.scenario == original.scenario
        assert restored.frame == original.frame
        assert restored.dt_s == pytest.approx(original.dt_s)
        assert restored.raw_candidate_count == original.raw_candidate_count
        assert restored.ego_linear_velocity_mps == pytest.approx(
            original.ego_linear_velocity_mps
        )
        assert restored.ego_yaw_rate_rad_s == pytest.approx(original.ego_yaw_rate_rad_s)
        assert restored.raw_candidate_projection_failed == original.raw_candidate_projection_failed
        for a, b in zip(original.raw_candidate_bbox, restored.raw_candidate_bbox):
            assert a == pytest.approx(b)
        for a, b in zip(
            original.raw_candidate_confidence, restored.raw_candidate_confidence
        ):
            assert a == pytest.approx(b)
        # NaN entries (failed projections) round-trip as NaN, not as a
        # coincidentally-equal sentinel -- np.nan != np.nan by design, so
        # compare with isnan directly rather than pytest.approx.
        for a, b in zip(original.raw_candidate_range_m, restored.raw_candidate_range_m):
            if np.isnan(a):
                assert np.isnan(b)
            else:
                assert a == pytest.approx(b)


def test_read_runtime_cache_raises_on_hash_mismatch(tmp_path: Path):
    path = tmp_path / "cache.npz"
    write_runtime_cache(path, (_frame(),))

    with pytest.raises(ValueError):
        read_runtime_cache(path, expected_sha256="0" * 64)

    # A correct expectation does not raise.
    actual = sha256(path)
    read_runtime_cache(path, expected_sha256=actual)


def test_read_runtime_cache_detects_a_regenerated_file(tmp_path: Path):
    path = tmp_path / "cache.npz"
    original_hash = write_runtime_cache(path, (_frame(frame=0),))
    # Silently regenerate the file with different content but the same name.
    write_runtime_cache(path, (_frame(frame=0), _frame(frame=1)))
    with pytest.raises(ValueError):
        read_runtime_cache(path, expected_sha256=original_hash)


def test_write_runtime_cache_rejects_empty_frame_list(tmp_path: Path):
    with pytest.raises(ValueError):
        write_runtime_cache(tmp_path / "empty.npz", ())


def test_cache_loads_without_allow_pickle(tmp_path: Path):
    """Regression guard for I5's explicit-offsets requirement: a ragged
    dtype=object array would force allow_pickle=True to load."""

    path = tmp_path / "cache.npz"
    write_runtime_cache(
        path,
        (
            _frame(frame=0, candidates=((0.5, 0.0, 0.8, (0, 0, 1, 1), False),)),
            _frame(frame=1, candidates=()),
            _frame(
                frame=2,
                candidates=(
                    (0.4, 0.1, 0.7, (1, 1, 2, 2), False),
                    (0.9, -0.2, 0.6, (2, 2, 3, 3), False),
                    (1.4, 0.3, 0.2, (3, 3, 4, 4), False),
                ),
            ),
        ),
    )
    with np.load(path, allow_pickle=False) as data:
        assert "raw_candidate_offsets" in data
        assert data["raw_candidate_range_m"].dtype == np.float64


def test_evaluation_truth_round_trips_and_keeps_candidates_out(tmp_path: Path):
    frames = (
        TruthFrame(
            episode="evaluation_7101_a",
            frame=0,
            gt_exists=True,
            gt_range_m=0.734,
            gt_bearing_rad=-0.12,
            gt_range_rate_mps=0.01,
            gt_bearing_rate_rad_s=0.0,
            eligible_visible=True,
            visible_pixel_count=812,
            gt_bbox=(10.0, 20.0, 60.0, 120.0),
            distance_bin="medium",
            fov_region="center",
        ),
        TruthFrame(
            episode="evaluation_7101_a",
            frame=1,
            gt_exists=True,
            gt_range_m=None,
            gt_bearing_rad=None,
            gt_range_rate_mps=None,
            gt_bearing_rate_rad_s=None,
            eligible_visible=False,
            visible_pixel_count=0,
            gt_bbox=None,
            distance_bin="far",
            fov_region="outside",
        ),
    )
    path = tmp_path / "truth.npz"
    returned_hash = write_evaluation_truth(path, frames)
    assert returned_hash == sha256(path)

    restored = read_evaluation_truth(path)
    assert set(restored) == {("evaluation_7101_a", 0), ("evaluation_7101_a", 1)}
    first = restored[("evaluation_7101_a", 0)]
    assert first.gt_range_m == pytest.approx(0.734)
    assert first.gt_bbox == pytest.approx((10.0, 20.0, 60.0, 120.0))
    second = restored[("evaluation_7101_a", 1)]
    assert second.gt_range_m is None
    assert second.gt_bbox is None
    # RuntimeCacheFrame's raw_* fields have no counterpart here -- ground
    # truth and candidates are provably separate files.
    with np.load(path, allow_pickle=False) as data:
        assert not any(name.startswith("raw_candidate") for name in data.files)
