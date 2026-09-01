import inspect

from duckie_pomdp.domain.detection import BoundingBox, Detection, ObjectClass
from duckie_pomdp.perception.detector_dropout import (
    DetectorDropout,
    DropoutAwareDetector,
    DropoutSchedule,
    DropoutWindow,
    RemovalSchedule,
    TargetRemovalScheduler,
    suppress_duckie_detections,
)


def _duckie(x=10.0, confidence=0.9) -> Detection:
    return Detection(
        object_class=ObjectClass.DUCKIE,
        confidence=confidence,
        bounding_box=BoundingBox(x, 20.0, x + 30.0, 60.0),
    )


def _stop_sign(x=200.0, confidence=0.8) -> Detection:
    return Detection(
        object_class=ObjectClass.STOP_SIGN,
        confidence=confidence,
        bounding_box=BoundingBox(x, 20.0, x + 30.0, 60.0),
    )


# ---------------------------------------------------------------------------
# The five brief-specified tests.
# ---------------------------------------------------------------------------


def test_dropout_suppresses_detections_only_inside_the_scheduled_window():
    dropout = DetectorDropout(window_length=20, warmup_frames=5)
    schedule = dropout.schedule_for(seed=8101, episode_length=100)
    window = schedule.windows[0]

    detections = (_duckie(), _stop_sign())

    before = suppress_duckie_detections(detections, frame=window.start_frame - 1, schedule=schedule)
    after = suppress_duckie_detections(detections, frame=window.end_frame + 1, schedule=schedule)
    inside = suppress_duckie_detections(detections, frame=window.start_frame, schedule=schedule)

    assert before == detections
    assert after == detections
    assert inside != detections
    assert all(detection.object_class is ObjectClass.STOP_SIGN for detection in inside)


def test_the_dropout_schedule_is_seed_determined_and_reproducible():
    dropout = DetectorDropout(window_length=25, warmup_frames=10)
    first = dropout.schedule_for(seed=8104, episode_length=150)
    second = dropout.schedule_for(seed=8104, episode_length=150)
    assert first.windows == second.windows

    other_seed = dropout.schedule_for(seed=8105, episode_length=150)
    # Not asserting inequality (a collision is not a bug), only that the
    # SAME seed always reproduces the SAME schedule.
    assert isinstance(other_seed.windows[0], DropoutWindow)


def test_the_dropout_schedule_cannot_consult_ground_truth_or_belief():
    parameters = set(inspect.signature(DetectorDropout.schedule_for).parameters)
    assert parameters == {"self", "seed", "episode_length"}


def test_dropout_suppresses_duckie_detections_but_not_stop_signs():
    dropout = DetectorDropout(window_length=10, warmup_frames=0)
    schedule = dropout.schedule_for(seed=8101, episode_length=40)
    window = schedule.windows[0]

    detections = (_duckie(), _stop_sign())
    filtered = suppress_duckie_detections(detections, frame=window.start_frame, schedule=schedule)

    assert len(filtered) == 1
    assert filtered[0].object_class is ObjectClass.STOP_SIGN


def test_a_dropout_frame_is_flagged_for_the_analysis():
    dropout = DetectorDropout(window_length=15, warmup_frames=5)
    schedule = dropout.schedule_for(seed=8102, episode_length=90)
    window = schedule.windows[0]

    assert schedule.is_dropout_frame(window.start_frame) is True
    assert schedule.is_dropout_frame(window.end_frame) is True
    assert schedule.is_dropout_frame(window.start_frame - 1) is False
    assert schedule.is_dropout_frame(window.end_frame + 1) is False


# ---------------------------------------------------------------------------
# Additional coverage: window shape, boundary handling, determinism across
# runs/instances, empty-detection safety.
# ---------------------------------------------------------------------------


def test_window_length_matches_the_configured_value_when_it_fits():
    dropout = DetectorDropout(window_length=30, warmup_frames=10)
    schedule = dropout.schedule_for(seed=8103, episode_length=200)
    assert schedule.windows[0].length == 30


def test_window_is_clamped_to_fit_a_short_episode():
    """A caller must never be handed a window that runs past the episode's
    own length, even if the configured window_length would not fit."""

    dropout = DetectorDropout(window_length=50, warmup_frames=10)
    schedule = dropout.schedule_for(seed=8103, episode_length=30)
    window = schedule.windows[0]
    assert window.length <= 30
    assert window.end_frame < 30
    assert window.start_frame >= 0


def test_window_respects_warmup_when_the_episode_has_room():
    dropout = DetectorDropout(window_length=20, warmup_frames=25)
    schedule = dropout.schedule_for(seed=8106, episode_length=200)
    assert schedule.windows[0].start_frame >= 25


def test_two_different_window_lengths_are_independent_instances():
    """window_length lives on the DetectorDropout instance, not passed at
    call time -- two differently-configured instances for the same seed
    produce differently-sized windows, never colliding on a shared call
    signature."""

    short = DetectorDropout(window_length=20, warmup_frames=10)
    long = DetectorDropout(window_length=45, warmup_frames=10)
    short_schedule = short.schedule_for(seed=8107, episode_length=200)
    long_schedule = long.schedule_for(seed=8107, episode_length=200)
    assert short_schedule.windows[0].length == 20
    assert long_schedule.windows[0].length == 45


def test_negative_or_zero_window_length_rejected():
    import pytest

    with pytest.raises(ValueError):
        DetectorDropout(window_length=0)
    with pytest.raises(ValueError):
        DetectorDropout(window_length=-5)


def test_non_positive_episode_length_rejected():
    import pytest

    dropout = DetectorDropout(window_length=10)
    with pytest.raises(ValueError):
        dropout.schedule_for(seed=8101, episode_length=0)


def test_suppress_on_empty_detections_is_a_no_op():
    dropout = DetectorDropout(window_length=10, warmup_frames=0)
    schedule = dropout.schedule_for(seed=8101, episode_length=40)
    assert suppress_duckie_detections((), frame=schedule.windows[0].start_frame, schedule=schedule) == ()


# ---------------------------------------------------------------------------
# DropoutAwareDetector -- the detector-output-boundary wrapper.
# ---------------------------------------------------------------------------


class _FakeDetector:
    """Records every rgb it was called with (so tests can assert the base
    detector always ran, even on a suppressed frame) and always returns one
    Duckie plus one stop sign."""

    def __init__(self) -> None:
        self.calls: list = []

    def detect(self, rgb):
        self.calls.append(rgb)
        return (_duckie(), _stop_sign())


def test_dropout_aware_detector_always_calls_the_base_detector():
    base = _FakeDetector()
    dropout = DetectorDropout(window_length=10, warmup_frames=0)
    schedule = dropout.schedule_for(seed=8101, episode_length=40)
    wrapped = DropoutAwareDetector(base, schedule)

    wrapped.frame = schedule.windows[0].start_frame
    wrapped.detect("frame_a")
    wrapped.frame = schedule.windows[0].end_frame + 1
    wrapped.detect("frame_b")

    assert base.calls == ["frame_a", "frame_b"]


def test_dropout_aware_detector_suppresses_duckies_inside_the_window_only():
    base = _FakeDetector()
    dropout = DetectorDropout(window_length=10, warmup_frames=0)
    schedule = dropout.schedule_for(seed=8101, episode_length=40)
    wrapped = DropoutAwareDetector(base, schedule)

    wrapped.frame = schedule.windows[0].start_frame
    inside = wrapped.detect("frame")
    wrapped.frame = schedule.windows[0].end_frame + 1
    outside = wrapped.detect("frame")

    assert all(d.object_class is ObjectClass.STOP_SIGN for d in inside)
    assert any(d.object_class is ObjectClass.DUCKIE for d in outside)
    assert any(d.object_class is ObjectClass.STOP_SIGN for d in outside)


# ---------------------------------------------------------------------------
# TargetRemovalScheduler -- B3's single-switch-frame schedule.
# ---------------------------------------------------------------------------


def test_removal_schedule_is_seed_determined_and_reproducible():
    scheduler = TargetRemovalScheduler(warmup_frames=15, tail_frames=20)
    first = scheduler.schedule_for(seed=8301, episode_length=90)
    second = scheduler.schedule_for(seed=8301, episode_length=90)
    assert first.switch_frame == second.switch_frame


def test_removal_schedule_signature_cannot_consult_ground_truth_or_belief():
    parameters = set(inspect.signature(TargetRemovalScheduler.schedule_for).parameters)
    assert parameters == {"self", "seed", "episode_length"}


def test_removal_schedule_guarantees_the_tail_frames_floor():
    """Wherever the seed places the switch, at least tail_frames+1 frames
    (the switch frame itself plus tail_frames more) must remain absent
    before the episode ends -- this is what lets a scenario GUARANTEE a
    run length, the same way DetectorDropout's window_length does for B2."""

    scheduler = TargetRemovalScheduler(warmup_frames=10, tail_frames=40)
    for seed in range(8301, 8309):
        schedule = scheduler.schedule_for(seed=seed, episode_length=95)
        remaining = schedule.episode_length - schedule.switch_frame
        assert remaining >= 41


def test_removal_schedule_is_removed_at_boundary():
    scheduler = TargetRemovalScheduler(warmup_frames=5, tail_frames=5)
    schedule = scheduler.schedule_for(seed=8301, episode_length=50)
    assert schedule.is_removed_at(schedule.switch_frame) is True
    assert schedule.is_removed_at(schedule.switch_frame - 1) is False
    assert schedule.is_removed_at(schedule.episode_length - 1) is True


def test_removal_schedule_shape():
    scheduler = TargetRemovalScheduler(warmup_frames=5, tail_frames=5)
    schedule = scheduler.schedule_for(seed=8301, episode_length=50)
    assert isinstance(schedule, RemovalSchedule)
    assert schedule.seed == 8301
    assert schedule.episode_length == 50


def test_removal_schedule_rejects_non_positive_episode_length():
    import pytest

    scheduler = TargetRemovalScheduler(warmup_frames=0, tail_frames=0)
    with pytest.raises(ValueError):
        scheduler.schedule_for(seed=8301, episode_length=0)


def test_dropout_schedule_is_a_frozen_dataclass_shape():
    """Sanity check on the public shape callers rely on."""

    dropout = DetectorDropout(window_length=10, warmup_frames=0)
    schedule = dropout.schedule_for(seed=8101, episode_length=40)
    assert isinstance(schedule, DropoutSchedule)
    assert schedule.seed == 8101
    assert schedule.episode_length == 40
