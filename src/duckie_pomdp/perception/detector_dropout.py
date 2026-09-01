"""B2 controlled detector-dropout mechanism (gate F9d, Task 4).

F9c's own natural-miss statistics (``P_D ~= 0.98``) make a 20-frame
in-domain miss run astronomically unlikely to occur on its own -- about
10^-34. B2 exists to test the existence filter's response to prolonged
detector silence anyway, *without* pretending the silence occurred
naturally: this module suppresses the detector's own Duckie detections for
a scheduled window, entirely at the DETECTOR OUTPUT boundary.

**What this is not.** The pedestrian is never removed from the scene, never
marked not-visible, and the simulator/renderer are never touched. The
detector still runs against the real rendered frame; only its Duckie
detections are discarded for the scheduled frames. This is what lets B2
answer "does an *unsupported in-domain* track decay?" -- not "can the
estimator forget a pedestrian that left?" (that question, if answerable at
all, belongs to B3).

**The three rules that keep the intervention honest**, all enforced by this
module's shape rather than by convention:

1. The schedule is fixed **per episode from the seed alone**, decided
   *before* the episode runs. ``DetectorDropout.schedule_for`` takes
   exactly ``(self, seed, episode_length)`` -- no ground truth, no belief
   state, no detector output can reach it, because there is no parameter
   slot for any of them to arrive through. A caller that tried to thread
   ground truth into the schedule would have to change this method's
   signature, which the drift test in
   ``tests/test_f9d_detector_dropout.py`` pins directly via
   ``inspect.signature``.
2. Suppression is applied to an already-produced detection list -- it never
   reaches backward into the detector or the renderer.
3. Only ``ObjectClass.DUCKIE`` detections are discarded; stop-sign
   detections in the same frame pass through untouched, because the
   intervention targets the pedestrian-existence question only.

Every dropout frame is queryable via ``DropoutSchedule.is_dropout_frame``,
so callers can flag it in their own CSV output and keep B2 frames
distinguishable from natural misses -- this module never itself decides
what a caller's analysis calls a "natural" vs "controlled" absence; it only
answers "was this frame's Duckie detection suppressed."
"""

from __future__ import annotations

from dataclasses import dataclass
from random import Random
from typing import Sequence

import numpy as np
from numpy.typing import NDArray

from duckie_pomdp.domain.detection import Detection, ObjectClass
from duckie_pomdp.ports.detector import ObjectDetector

_SALT = "f9d-task4-detector-dropout-v1"


@dataclass(frozen=True)
class DropoutWindow:
    """One contiguous, inclusive frame range within which Duckie detections
    are suppressed."""

    start_frame: int
    end_frame: int

    def __post_init__(self) -> None:
        if self.start_frame < 0:
            raise ValueError("start_frame must be non-negative")
        if self.end_frame < self.start_frame:
            raise ValueError("end_frame must not precede start_frame")

    @property
    def length(self) -> int:
        return self.end_frame - self.start_frame + 1

    def contains(self, frame: int) -> bool:
        return self.start_frame <= frame <= self.end_frame


@dataclass(frozen=True)
class DropoutSchedule:
    """The full per-episode dropout schedule: which frames are suppressed."""

    seed: int
    episode_length: int
    windows: tuple[DropoutWindow, ...]

    def is_dropout_frame(self, frame: int) -> bool:
        return any(window.contains(frame) for window in self.windows)


class DetectorDropout:
    """Builds seed-determined, reproducible B2 dropout schedules.

    ``window_length`` and ``warmup_frames`` are configured per instance (at
    construction time) rather than per call, precisely so
    ``schedule_for(self, seed, episode_length)`` never needs a parameter
    beyond the seed and the episode length -- a caller wanting a 20-frame
    window and a caller wanting a 45-frame window build two different
    ``DetectorDropout`` instances, not two different call signatures.

    ``warmup_frames`` biases the window away from frame 0: an existence
    track needs some genuinely-detected frames to become active before a
    dropout can test whether an *established* track decays under silence.
    The bias is a fixed, seed-independent constructor parameter -- it
    never consults whether a track actually initialized, which would be
    reading belief state into the schedule.
    """

    def __init__(self, *, window_length: int, warmup_frames: int = 0) -> None:
        if window_length <= 0:
            raise ValueError("window_length must be positive")
        if warmup_frames < 0:
            raise ValueError("warmup_frames must be non-negative")
        self._window_length = window_length
        self._warmup_frames = warmup_frames

    def schedule_for(self, seed: int, episode_length: int) -> DropoutSchedule:
        """The one entry point. Exactly ``(self, seed, episode_length)`` --
        no other parameter may be added without breaking
        ``test_the_dropout_schedule_cannot_consult_ground_truth_or_belief``,
        which asserts this signature directly via ``inspect.signature``.
        """

        if episode_length <= 0:
            raise ValueError("episode_length must be positive")

        window_length = min(self._window_length, episode_length)
        latest_start = episode_length - window_length
        earliest_start = min(self._warmup_frames, latest_start)

        rng = Random(f"{_SALT}:{seed}:{episode_length}:{self._window_length}:{self._warmup_frames}")
        start = (
            earliest_start
            if latest_start <= earliest_start
            else rng.randint(earliest_start, latest_start)
        )
        window = DropoutWindow(start_frame=start, end_frame=start + window_length - 1)
        return DropoutSchedule(seed=seed, episode_length=episode_length, windows=(window,))


def suppress_duckie_detections(
    detections: Sequence[Detection],
    *,
    frame: int,
    schedule: DropoutSchedule,
) -> tuple[Detection, ...]:
    """Apply one episode's dropout schedule to one frame's raw detections.

    Outside the scheduled window every detection passes through unchanged
    (same objects, same order -- byte-identical). Inside it, every
    ``ObjectClass.DUCKIE`` detection is discarded; ``ObjectClass.STOP_SIGN``
    detections are never touched, in either case.
    """

    if not schedule.is_dropout_frame(frame):
        return tuple(detections)
    return tuple(
        detection
        for detection in detections
        if detection.object_class is not ObjectClass.DUCKIE
    )


class DropoutAwareDetector:
    """``ObjectDetector`` wrapper that applies one episode's schedule at
    exactly the detector-output boundary.

    The base detector always runs against the real rendered frame --
    ``detect()`` is never skipped or short-circuited. Only the RETURNED
    Duckie detections are discarded when the caller-advanced ``frame``
    counter falls inside the schedule. ``frame`` is a plain public
    attribute the caller sets once per simulated step, immediately before
    invoking anything that calls ``detect`` -- this keeps the
    ``ObjectDetector`` port's ``detect(rgb)`` signature completely
    unchanged (no frame index threads through it), so this wrapper can
    stand in anywhere a base detector is expected, including inside
    ``YoloPedestrianMeasurementPipeline`` unmodified.
    """

    def __init__(self, detector: ObjectDetector, schedule: DropoutSchedule) -> None:
        self._detector = detector
        self.schedule = schedule
        self.frame = 0

    def detect(self, rgb: NDArray[np.uint8]) -> Sequence[Detection]:
        detections = self._detector.detect(rgb)
        return suppress_duckie_detections(detections, frame=self.frame, schedule=self.schedule)


# ---------------------------------------------------------------------------
# B3's scheduler. A different intervention shape (a single, PERMANENT
# switch frame rather than a bounded suppression window -- once past it,
# the scenario pedestrian never returns), but the same three honesty
# rules: seed-and-episode-length-only, decided before the episode runs,
# never consulting ground truth/belief/detector output.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RemovalSchedule:
    """B3's schedule: one seed-determined frame at which the scenario
    pedestrian is removed, and never returns.

    **Contract a caller must honour (fix round 1, Task 4).** ``switch_frame``
    is defined as the first frame index for which BOTH privileged ground
    truth AND the rendered image agree the pedestrian is gone. Privileged
    truth flips the instant ``remove_scenario_pedestrian()`` is called
    (``is_removed_at`` reflects that immediately), but the rendered image
    a caller is holding at that moment is whatever was last rendered --
    typically one step stale. To keep the two in agreement at exactly
    ``switch_frame``, a caller must call ``remove_scenario_pedestrian()``
    BEFORE the simulator step that RENDERS ``switch_frame``'s image (i.e.
    between finishing frame ``switch_frame - 1`` and stepping to
    ``switch_frame``), not while processing ``switch_frame`` itself. See
    ``experiments/probe_f9d_absence_yield.py::collect_absence_rows`` for
    the reference implementation, and
    ``tests/test_f9d_b3_pedestrian_removal.py::
    test_image_and_privileged_truth_agree_on_the_first_absent_frame`` for
    the real-simulator pin.
    """

    seed: int
    episode_length: int
    switch_frame: int

    def is_removed_at(self, frame: int) -> bool:
        return frame >= self.switch_frame


class TargetRemovalScheduler:
    """Builds seed-determined, reproducible B3 removal schedules.

    ``warmup_frames`` (a track needs genuinely-detected frames to become
    active before removal can test whether an established track decays)
    and ``tail_frames`` (a floor on how many trailing absence frames the
    schedule guarantees, regardless of where within the eligible band the
    seed happens to place the switch) are constructor-level, exactly like
    ``DetectorDropout.window_length``/``warmup_frames`` -- so
    ``schedule_for`` never needs a parameter beyond the seed and the
    episode length.
    """

    def __init__(self, *, warmup_frames: int = 0, tail_frames: int = 0) -> None:
        if warmup_frames < 0:
            raise ValueError("warmup_frames must be non-negative")
        if tail_frames < 0:
            raise ValueError("tail_frames must be non-negative")
        self._warmup_frames = warmup_frames
        self._tail_frames = tail_frames

    def schedule_for(self, seed: int, episode_length: int) -> RemovalSchedule:
        if episode_length <= 0:
            raise ValueError("episode_length must be positive")

        latest_switch = max(0, episode_length - 1 - self._tail_frames)
        earliest_switch = min(self._warmup_frames, latest_switch)

        rng = Random(f"{_SALT}:b3:{seed}:{episode_length}:{self._warmup_frames}:{self._tail_frames}")
        switch_frame = (
            earliest_switch
            if latest_switch <= earliest_switch
            else rng.randint(earliest_switch, latest_switch)
        )
        return RemovalSchedule(seed=seed, episode_length=episode_length, switch_frame=switch_frame)
