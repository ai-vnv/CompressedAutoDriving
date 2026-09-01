"""F9d Task 4, B3 feasibility gate: ``remove_scenario_pedestrian``.

Real-simulator tests (no mocking) for the minimal additive intervention the
task brief asked to attempt: mark the scenario pedestrian not-visible at a
scheduled frame so it leaves both the rendered frame and the
privileged-truth queries GT sampling uses. This mirrors
``test_minimal_scenario.py``'s real-integration style rather than a unit
test with a fake simulator, because the whole feasibility question is
about the REAL gym-duckietown renderer and REAL privileged-state plumbing,
which a fake could not answer.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from duckie_pomdp.adapters.gym_duckietown import (
    GymDuckietownConfig,
    create_gym_duckietown,
)
from duckie_pomdp.domain.action import PolicyAction
from duckie_pomdp.scenario import PedestrianMode, load_scenario

SCENARIO_PATH = Path("configs/scenario_pomdp_v1.toml")


def test_removal_makes_privileged_truth_report_the_pedestrian_absent() -> None:
    """(b) privileged truth reports it absent from the switch frame onward,
    and unchanged (present) on every frame before it."""

    scenario = load_scenario(SCENARIO_PATH).with_pedestrian_mode(PedestrianMode.STATIONARY)
    integration = create_gym_duckietown(
        GymDuckietownConfig(scenario=scenario, camera_width=160, camera_height=120)
    )
    try:
        integration.agent.reset(seed=scenario.seed)
        action = PolicyAction(0.0, 0.0)

        for _ in range(5):
            assert integration.privileged.read().true_pomdp_state.pedestrian.exists
            integration.agent.step(action)

        integration.projection_validation.remove_scenario_pedestrian()

        for _ in range(10):
            integration.agent.step(action)
            state = integration.privileged.read().true_pomdp_state.pedestrian
            assert state.exists is False
            assert state.range_m is None
            assert state.bearing_rad is None
    finally:
        integration.close()


def test_removal_makes_the_pedestrian_vanish_from_the_rendered_rgb() -> None:
    """(a) the pedestrian genuinely vanishes from the rendered RGB after the
    switch frame -- compared numerically against
    ``render_without_objects(("duckie",))``, which is already trusted by
    F9a's privileged counterfactual renderer to produce a duckie-free
    frame."""

    scenario = load_scenario(SCENARIO_PATH).with_pedestrian_mode(PedestrianMode.STATIONARY)
    integration = create_gym_duckietown(
        GymDuckietownConfig(scenario=scenario, camera_width=160, camera_height=120)
    )
    try:
        observation = integration.agent.reset(seed=scenario.seed)
        action = PolicyAction(0.0, 0.0)

        visible_frame = np.asarray(observation.front_rgb).astype(int)
        hidden_reference = integration.projection_validation.render_without_objects(
            ("duckie",)
        ).astype(int)

        # Before removal: the real rendered frame must differ substantially
        # from the hidden-pedestrian reference -- the pedestrian occupies
        # real, visible pixels.
        diff_before = int(np.max(np.abs(visible_frame - hidden_reference)))
        assert diff_before > 50

        integration.projection_validation.remove_scenario_pedestrian()
        transition = integration.agent.step(action)
        after_frame = np.asarray(transition.observation.front_rgb).astype(int)

        # After removal: the rendered frame must match the hidden-pedestrian
        # reference almost exactly -- the pedestrian is genuinely gone from
        # the render, not merely absent from privileged truth.
        diff_after = int(np.max(np.abs(after_frame - hidden_reference)))
        assert diff_after <= 2

        # And it stays gone on later frames -- not a one-frame flicker.
        for _ in range(5):
            transition = integration.agent.step(action)
        later_frame = np.asarray(transition.observation.front_rgb).astype(int)
        diff_later = int(np.max(np.abs(later_frame - hidden_reference)))
        assert diff_later <= 2
    finally:
        integration.close()


def test_removal_without_a_scenario_pedestrian_raises() -> None:
    """The method must fail loudly, not silently no-op, when the session
    was not built from a scenario at all (no ``_scenario_pedestrian``)."""

    integration = create_gym_duckietown(
        GymDuckietownConfig(
            map_name="small_loop",
            seed=73,
            start_tile=(1, 0),
            start_pose=((0.065, 0.0, 0.4095), 0.0),
        )
    )
    try:
        integration.agent.reset(seed=73)
        with pytest.raises(RuntimeError):
            integration.projection_validation.remove_scenario_pedestrian()
    finally:
        integration.close()


def test_image_and_privileged_truth_agree_on_the_first_absent_frame() -> None:
    """Fix round 1 pin (coordinator-directed). Task 7 will read
    ``P(e)`` at frames 1/5/10/20/30/40 relative to absence onset, exactly
    once, on seeds that cannot be re-rendered -- a one-frame ambiguity about
    which frame is "first absent" would shift every one of those readings.

    ``RemovalSchedule.switch_frame`` is contractually the first frame index
    for which BOTH privileged truth and the rendered image agree the
    pedestrian is gone (see its docstring). This test runs the EXACT
    call-ordering pattern ``probe_f9d_absence_yield.py::collect_absence_rows``
    uses after the fix: ``remove_scenario_pedestrian()`` is called
    immediately before the ``step()`` that RENDERS ``switch_frame``'s image
    -- never while a caller is already processing ``switch_frame`` itself,
    which would leave that frame's image one step stale relative to truth.
    """

    from duckie_pomdp.perception.detector_dropout import TargetRemovalScheduler

    scenario = load_scenario(SCENARIO_PATH).with_pedestrian_mode(PedestrianMode.STATIONARY)
    integration = create_gym_duckietown(
        GymDuckietownConfig(scenario=scenario, camera_width=160, camera_height=120)
    )
    try:
        observation = integration.agent.reset(seed=scenario.seed)
        action = PolicyAction(0.0, 0.0)

        scheduler = TargetRemovalScheduler(warmup_frames=5, tail_frames=10)
        episode_length = 25
        schedule = scheduler.schedule_for(seed=scenario.seed, episode_length=episode_length)
        assert 0 < schedule.switch_frame < episode_length - 1, (
            "test setup assumes an interior switch frame with a frame on "
            "each side to compare"
        )

        hidden_reference = integration.projection_validation.render_without_objects(
            ("duckie",)
        ).astype(int)

        removed = False
        images: dict[int, np.ndarray] = {}
        truth_exists: dict[int, bool] = {}

        for frame in range(episode_length):
            images[frame] = np.asarray(observation.front_rgb).astype(int)
            truth_exists[frame] = bool(
                integration.privileged.read().true_pomdp_state.pedestrian.exists
            )

            if frame == episode_length - 1:
                break

            # Same ordering as the fixed probe script: remove BEFORE the
            # step() that renders switch_frame's image, not after.
            if not removed and schedule.switch_frame == frame + 1:
                integration.projection_validation.remove_scenario_pedestrian()
                removed = True

            transition = integration.agent.step(action)
            observation = transition.observation

        switch = schedule.switch_frame

        # The frame immediately BEFORE the switch: both image and truth
        # still show the pedestrian present.
        assert truth_exists[switch - 1] is True
        assert int(np.max(np.abs(images[switch - 1] - hidden_reference))) > 50

        # The switch frame itself: both image and truth agree the
        # pedestrian is already gone, on the SAME frame index -- not one
        # frame apart in either direction.
        assert truth_exists[switch] is False
        assert int(np.max(np.abs(images[switch] - hidden_reference))) <= 2
    finally:
        integration.close()


def test_removal_is_not_exposed_through_the_agent_environment() -> None:
    """Deliberately not reachable from the runtime detector/EKF chain --
    the same discipline ``render_without_objects`` already follows."""

    from duckie_pomdp.adapters.gym_duckietown import GymDuckietownAgentEnvironment

    assert not hasattr(GymDuckietownAgentEnvironment, "remove_scenario_pedestrian")
