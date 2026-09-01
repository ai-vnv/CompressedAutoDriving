# Task 5 Report: Perception emits candidates

## What I implemented

In `src/duckie_pomdp/perception/f9_pipeline.py`:

- New frozen dataclass `CandidateProjection` with fields `detection: Detection`,
  `measurement: ObjectMeasurement`, `projection_error: str | None`.
- `F9ImageObservation` gained a new **last** field
  `duckie_candidates: tuple[CandidateProjection, ...] = ()` (defaulted, so any existing
  positional construction elsewhere keeps working).
- New private helper `YoloPedestrianMeasurementPipeline._project_candidate(detection)` that
  calls `self._projector.project_raw(detection)` inside its own `try/except ValueError`,
  returning a `CandidateProjection` with either a populated `ObjectMeasurement` and
  `projection_error=None`, or `ObjectMeasurement.missing(ObjectClass.DUCKIE)` and
  `projection_error=str(error)`.
- In `observe()`, every Duckie detection (not just the one `select_single_duckie` picks) is
  now run through `_project_candidate` independently, each with its own try/except, so one bad
  box cannot discard the others. The resulting tuple is passed as `duckie_candidates` on all
  three existing return paths (no-selection, projection-failure, success). **The pre-existing
  selection logic — `select_single_duckie`, the `project_raw` call on `selection.selected`, and
  the construction of `pedestrian` / `selected_duckie` / `duplicate_selection` /
  `projection_error` / `duckie_detection_count` / `stop_sign_detections` — is untouched.**
  `duckie_candidates` is purely additive; nothing was rerouted through it.

## TDD evidence

### RED

Command:
```
wsl.exe -d Ubuntu-Baru -- bash -lc 'cd /home/pannntastic/aivnv/duckie-pomdp && export PYTHONPATH=src:/home/pannntastic/aivnv/duckie/src && export DUCKIETOWN_HEADLESS=1 && /home/pannntastic/aivnv/duckie/.venv/bin/python -m pytest tests/test_f9_pipeline.py -q'
```

Actual output (before implementing):
```
==================================== ERRORS ====================================
__________________ ERROR collecting tests/test_f9_pipeline.py __________________
ImportError while importing test module '/home/pannntastic/aivnv/duckie-pomdp/tests/test_f9_pipeline.py'.
Hint: make sure your test modules/packages have valid Python names.
Traceback:
/usr/lib/python3.10/importlib/__init__.py:126: in import_module
    return _bootstrap._gcd_import(name[level:], package, level)
tests/test_f9_pipeline.py:29: in <module>
    from duckie_pomdp.perception.f9_pipeline import (
E   ImportError: cannot import name 'CandidateProjection' from 'duckie_pomdp.perception.f9_pipeline' (/home/pannntastic/aivnv/duckie-pomdp/src/duckie_pomdp/perception/f9_pipeline.py)
=========================== short test summary info ============================
ERROR tests/test_f9_pipeline.py
!!!!!!!!!!!!!!!!!!!! Interrupted: 1 error during collection !!!!!!!!!!!!!!!!!!!!
1 error in 0.24s
```

**Deviation from the brief's predicted failure, and why it's correct:** the brief predicted
`AttributeError: 'F9ImageObservation' object has no attribute 'duckie_candidates'`. What I
actually got instead was an `ImportError` at collection time, because my test file imports
`CandidateProjection` by name at module load (`from duckie_pomdp.perception.f9_pipeline import
..., CandidateProjection, ...`), and that name did not exist yet in `f9_pipeline.py`. This is
the correct and expected RED for this implementation: it fails for the same underlying reason
(the feature doesn't exist yet) one import statement earlier than the brief's illustrative
snippet assumed, because the brief's snippet didn't show importing `CandidateProjection`
explicitly. I judged this an equivalent, honest RED and proceeded to implement.

### GREEN

Command:
```
wsl.exe -d Ubuntu-Baru -- bash -lc 'cd /home/pannntastic/aivnv/duckie-pomdp && export PYTHONPATH=src:/home/pannntastic/aivnv/duckie/src && export DUCKIETOWN_HEADLESS=1 && /home/pannntastic/aivnv/duckie/.venv/bin/python -m pytest tests/test_f9_pipeline.py -q'
```

Actual output (after implementing, run and observed by me directly):
```
..................                                                       [100%]
18 passed in 0.39s
```

18 = 14 pre-existing tests in this file + 4 new ones.

## Full-suite result

I ran this myself (not merely relaying a claim):
```
wsl.exe -d Ubuntu-Baru -- bash -lc 'cd /home/pannntastic/aivnv/duckie-pomdp && export PYTHONPATH=src:/home/pannntastic/aivnv/duckie/src && export DUCKIETOWN_HEADLESS=1 && /home/pannntastic/aivnv/duckie/.venv/bin/python -m pytest tests -q'
```
Output tail:
```
144 passed, 260 warnings in 11.91s
```
144 passed, 0 failed, confirmed directly by me. (The warnings are pre-existing pyparsing/gym
deprecation noise unrelated to this change.) This matches the number the coordinator had
separately reported after running it independently; I re-ran it myself rather than taking that
report on faith, and got the same result.

## How I triggered the projection failure in test 3

I used the **real** `YoloMeasurementProjector` (via the existing `projector()` helper already
defined in the test file — same camera calibration as the other tests), not a stub. I probed it
directly first to find a box that reliably raises `ValueError` from
`CalibratedGroundProjector.pixel_to_ground`:

```
wsl.exe -d Ubuntu-Baru -- bash -lc '... python -c "... BoundingBox(300.0, 0.0, 340.0, y) for y in [0,1,5,10,...,80] ..."'
```

Every box with `y_max_px` between 0 and 80 (i.e. bottom edge on/above the horizon band for this
camera pitch/height) raised `ValueError: pixel ray intersects the ground behind the camera`.
I used `detection(ObjectClass.DUCKIE, 0.8, (300.0, 0.0, 340.0, 80.0))` as the failing candidate
in `test_a_candidate_whose_projection_fails_is_kept_with_its_error_and_no_measurement`, paired
with a normal, successfully-projecting box `(10.0, 300.0, 60.0, 420.0)` (bottom of image,
consistent with the box conventions already used in
`test_runtime_pipeline_uses_raw_projection_and_preserves_stop_detections`). This is the "real
projector, box on/above the horizon" path the brief suggested as the natural case — no stub
projector was needed or used.

## Confirmation: pre-existing tests unmodified and green

`test_single_duckie_selection_is_highest_confidence_without_gt`,
`test_yolo_miss_is_structural_none_measurement`, and
`test_runtime_pipeline_uses_raw_projection_and_preserves_stop_detections` were not edited — the
only changes to `tests/test_f9_pipeline.py` were (a) adding `CandidateProjection` to the
existing import statement from `duckie_pomdp.perception.f9_pipeline`, and (b) inserting four new
test functions immediately after
`test_runtime_pipeline_uses_raw_projection_and_preserves_stop_detections` and before
`test_bias_correction_is_fixed_image_measurement_only_transform`. No existing test body,
assertion, or fixture was touched. All three are present in the 18-passed run above.

## Files changed

- `\\wsl.localhost\Ubuntu-Baru\home\pannntastic\aivnv\duckie-pomdp\src\duckie_pomdp\perception\f9_pipeline.py`
  — added `CandidateProjection`, added `duckie_candidates` field to `F9ImageObservation`, added
  `_project_candidate` helper, threaded `duckie_candidates` through all three return paths of
  `observe()`. No line of pre-existing logic was deleted or altered.
- `\\wsl.localhost\Ubuntu-Baru\home\pannntastic\aivnv\duckie-pomdp\tests\test_f9_pipeline.py` —
  added `CandidateProjection` to the import, added 4 new tests. No existing test modified.

## Self-review

- **Existing fields keep identical semantics:** yes. `select_single_duckie` is called exactly
  as before with no argument or behavior change. The success/no-selection/projection-error
  branches build `pedestrian`, `selected_duckie`, `duplicate_selection`, `projection_error`,
  `duckie_detection_count`, and `stop_sign_detections` with the exact same expressions as
  before my edit; I only appended `duckie_candidates=candidates` as a new keyword argument to
  each of the three existing `F9ImageObservation(...)` constructor calls. Nothing reads from or
  is derived from `duckie_candidates` to populate those existing fields.
- **`select_single_duckie` untouched:** confirmed — zero changes to that function.
- **`AdditiveMeasurementBias` untouched:** confirmed — zero changes.
- **`duckie_candidates` is the last field with a default:** confirmed (line order:
  `pedestrian, duckie_detection_count, selected_duckie, duplicate_selection, projection_error,
  stop_sign_detections, duckie_candidates=()`), preserving positional-construction
  compatibility.

### `project_raw` is now called twice for the selected detection — is that intentional?

Yes, and it is intentional, not an oversight. `observe()` still calls
`self._projector.project_raw(selection.selected)` directly (the original, untouched code path)
to build `pedestrian` / `projection_error` / etc., and separately `_project_candidate` calls
`self._projector.project_raw(detection)` again for that same detection while building the
`duckie_candidates` tuple. I considered reusing the already-computed candidate's
`CandidateProjection` for the selected detection instead of projecting it a second time, which
would save one redundant `project_raw` call per frame. I deliberately did **not** do that,
for two reasons:

1. **Risk minimization on validated code.** The task brief is explicit that this is the first
   task modifying already-validated code, and that `select_single_duckie` and the existing
   field population must stay "untouched." Reusing the candidate projection would mean
   `pedestrian`'s values originate from a shared code path with `duckie_candidates`, which is
   exactly the kind of rerouting the brief told me not to do ("You are adding
   `duckie_candidates`, not rerouting anything through it"). Keeping the two computations
   fully independent means a future bug in candidate-list construction structurally cannot
   change Baseline-A's frozen values, and vice versa — the two code paths can be reasoned about,
   tested, and later changed in isolation.
2. **`project_raw` is a pure, side-effect-free geometric computation** (pixel clamp → ground
   projection → polar conversion) with no I/O or randomness, so calling it twice for the same
   `Detection` produces bit-identical results at negligible extra cost (a few `numpy` matrix
   operations per frame). This is not a performance-sensitive hot path for the gate's purposes.

If a reviewer prefers the single-call version for efficiency, it is a safe follow-up refactor
*after* Baseline A has run through this exact code once, at which point deduplicating the two
calls could be verified byte-for-byte against a recorded baseline rather than trusted by
inspection alone. I did not make that change myself.

## Concerns

- None blocking. The one design tradeoff (double `project_raw` call for the selected
  detection) is discussed above and was a deliberate choice favoring isolation between the
  frozen Baseline-A path and the new candidate-list path over a minor efficiency gain.
- This project is not a git repository; no commit was made, per instructions.
- I did not touch `src/duckie_pomdp/belief/__init__.py` or any belief-layer file.
