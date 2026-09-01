from __future__ import annotations

import numpy as np
import pytest

from duckie_pomdp.explain.final_visualization import (
    GROUP_ORDER,
    PHASE_ORDER,
    longest_contiguous_segment,
    pedestrian_belief_world,
    select_representative_frames,
    summary_matrix,
    validate_group_summary_rows,
)


def test_longest_segment_is_deterministic_and_earliest_on_tie() -> None:
    assert longest_contiguous_segment([1, 2, 5, 6]).tolist() == [1, 2]


def test_representative_selection_uses_public_fields_only() -> None:
    names = (
        "pedestrian_existence_probability",
        "pedestrian_range_mean_m",
    )
    phases = np.asarray(
        [
            "nominal",
            "nominal",
            "lane_curve",
            "lane_curve",
            "pedestrian_relevant",
            "pedestrian_relevant",
            "stop_required",
            "stop_required",
            "stop_satisfied",
            "stop_satisfied",
        ]
    )
    physical = np.zeros((10, 2), dtype=np.float32)
    physical[4:6, 0] = 0.95
    physical[4:6, 1] = [0.8, 0.4]
    selected = select_representative_frames(
        phases=phases,
        steps=np.arange(10),
        physical_observation=physical,
        feature_names=names,
    )
    assert tuple(selected) == PHASE_ORDER
    assert selected["nominal"].simulator_step == 1
    assert selected["pedestrian_relevant"].simulator_step == 4


def test_summary_grid_requires_exact_partition_and_unit_share() -> None:
    rows = []
    for target in ("v_cmd_mps", "omega_cmd_rad_s"):
        for phase in ("all",) + PHASE_ORDER:
            for group in GROUP_ORDER:
                rows.append(
                    {
                        "target": target,
                        "public_phase": phase,
                        "group": group,
                        "mean_absolute_group_share": 1.0 / len(GROUP_ORDER),
                        "share_ci_low": 0.0,
                        "share_ci_high": 1.0,
                    }
                )
    validate_group_summary_rows(rows)
    assert summary_matrix(rows, target="v_cmd_mps").shape == (5, 6)
    rows.pop()
    with pytest.raises(ValueError, match="summary grid mismatch"):
        validate_group_summary_rows(rows)


def test_pedestrian_world_transform_matches_project_axes() -> None:
    mean, covariance = pedestrian_belief_world(
        ego_x_m=1.0,
        ego_z_m=2.0,
        ego_heading_rad=0.0,
        range_mean_m=2.0,
        range_std_m=0.1,
        bearing_mean_rad=0.0,
        bearing_std_rad=0.05,
    )
    # heading=0 faces +world-x; zero bearing is directly forward.
    assert mean == pytest.approx([3.0, 2.0])
    assert covariance.shape == (2, 2)
    assert np.linalg.eigvalsh(covariance).min() >= -1.0e-12

