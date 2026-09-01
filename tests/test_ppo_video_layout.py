import numpy as np

from experiments.render_f10_ppo_object_bev import _annotate as object_annotate
from experiments.render_f10_ppo_v9_bev import _annotate as lane_annotate


def test_lane_video_telemetry_does_not_cover_bev_pixels() -> None:
    frame = np.full((60, 100, 3), 173, dtype=np.uint8)
    result = lane_annotate(
        frame,
        stage="c0",
        map_name="small_loop",
        seed=1,
        step=1,
        checkpoint_step=1,
        info={},
        reward=0.0,
        total_return=0.0,
        layout="bev",
    )
    assert result.shape == (168, 100, 3)
    assert np.array_equal(result[:60], frame)


def test_object_video_telemetry_does_not_cover_bev_pixels() -> None:
    frame = np.full((60, 100, 3), 211, dtype=np.uint8)
    result = object_annotate(
        frame,
        stage="c2",
        seed=1,
        step=1,
        info={},
        reward=0.0,
        total_return=0.0,
    )
    assert result.shape == (186, 100, 3)
    assert np.array_equal(result[:60], frame)
