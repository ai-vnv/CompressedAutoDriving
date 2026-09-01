from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from duckie_pomdp.visualization.belief_video import (
    BeliefVideoOverlay,
    DetectionOverlay,
    EvaluationTruthOverlay,
    render_belief_overlay,
)


def overlay() -> BeliefVideoOverlay:
    return BeliefVideoOverlay(
        frame_index=12,
        timestamp_s=0.4,
        detections=(
            DetectionOverlay(
                "duckie",
                0.91,
                (100.0, 120.0, 180.0, 260.0),
                associated=True,
                accepted=True,
            ),
        ),
        duckie_detection_count=1,
        measurement_range_m=0.81,
        measurement_bearing_rad=0.12,
        belief_range_m=0.79,
        belief_range_std_m=0.03,
        belief_bearing_rad=0.11,
        belief_bearing_std_rad=0.02,
        radial_velocity_mps=-0.04,
        bearing_rate_rad_s=0.08,
        existence_probability=0.98,
        track_active=True,
        frame_mode="temporal",
        observability_class="center",
        measurement_accepted=True,
        nis=1.2,
        truth=EvaluationTruthOverlay(0.78, 0.10),
    )


def test_renderer_produces_rgb_panel_without_mutating_input() -> None:
    rgb = np.zeros((480, 640, 3), dtype=np.uint8)
    before = rgb.copy()

    rendered = render_belief_overlay(rgb, overlay())

    assert rendered.shape == (480, 1000, 3)
    assert rendered.dtype == np.uint8
    assert np.array_equal(rgb, before)
    assert np.any(rendered[:, 640:, :])


def test_renderer_rejects_non_rgb_input() -> None:
    with pytest.raises(ValueError, match="HxWx3 uint8 RGB"):
        render_belief_overlay(np.zeros((480, 640), dtype=np.uint8), overlay())


def test_detection_overlay_validates_domain_values() -> None:
    with pytest.raises(ValueError, match="confidence"):
        DetectionOverlay("duckie", 1.1, (1.0, 1.0, 2.0, 2.0))
    with pytest.raises(ValueError, match="positive area"):
        DetectionOverlay("stop_sign", 0.5, (2.0, 1.0, 1.0, 2.0))


def test_demo_reads_privileged_truth_only_after_runtime_update() -> None:
    source = (
        Path(__file__).resolve().parents[1]
        / "experiments"
        / "render_yolo_belief_video.py"
    ).read_text(encoding="utf-8")

    observe = source.index("result = runtime.observe(observation.front_rgb)")
    update = source.index("belief, record = updater.update(")
    privileged = source.index("integration.privileged.read().true_pomdp_state")

    assert observe < update < privileged
