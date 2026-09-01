# Task 7 Report: Predicted observability and effective detection probability

## Summary

Implemented predicted-observability classification (`ObservabilityClass`,
`PredictedObservability`, `PredictedObservabilityModel`) and the
per-class effective detection model (`EffectiveDetectionModel`) in a new
module `src/duckie_pomdp/belief/observability.py`. Extended
`ExistenceFilter.update` with keyword-only `detection_probability` and
`observation_informative` parameters implementing invariant I3: a miss
predicted to fall outside the camera's observable domain applies no
likelihood at all, only the `P_S`/`P_birth` survival prediction.

## Files changed

- Created: `src/duckie_pomdp/belief/observability.py`
- Modified: `src/duckie_pomdp/belief/existence_filter.py` (only `ExistenceFilter.update`;
  `ExistenceFilterConfig` and `P_S`/`P_birth` untouched)
- Created: `tests/test_f9c_observability.py` (brief's 8 tests + 1 added MID_FOV test = 9 total)
- Created: `tests/test_f9c_existence.py` (brief's 10 tests, verbatim)
- **Not modified**: `src/duckie_pomdp/belief/__init__.py`, `tests/test_pedestrian_ekf.py`

## Implementation notes

- `PredictedObservabilityModel.classify(predicted_state)` reads only
  `predicted_state[0]` (`x_left`) and `predicted_state[1]` (`y_forward`) — no
  privileged simulator input, verified by the
  `test_classification_uses_no_privileged_input` signature-introspection test.
- Returns `OUTSIDE_DOMAIN` when `y_forward <= 0`, when
  `CalibratedGroundProjector.ground_to_pixel` raises `ValueError` ("behind the
  camera"), or when the projected column falls outside `[0, W)`.
  `normalized_horizontal_offset` is `None` in all three of those cases.
- In-domain binning: `normalized = |x_px − W/2| / (W/2)`; `< 1/3` → CENTER,
  `< 2/3` → MID_FOV, else → EDGE_FOV. A comment in the source points at
  `experiments/validate_f9_yolo_ekf.py:62-71` (`_fov_region`) as the
  evaluation-side binning this must agree with.
- `EffectiveDetectionModel.__init__` rejects any mapping missing a class,
  message contains `"every observability class"`.
- `EffectiveDetectionModel.miss_is_informative` returns `False` only for
  `OUTSIDE_DOMAIN` when `outside_domain_miss_policy == "prediction_only"`;
  `probability()` still reports the OUTSIDE_DOMAIN class's fitted value as a
  diagnostic — it is just never applied to a miss.
- `ExistenceFilter.update(detected, *, detection_probability=None,
  observation_informative=True)`:
  1. computes `predicted` from `P_S`/`P_birth` exactly as before;
  2. if `not observation_informative`, clamps and returns `predicted`
     immediately — **no validation, no likelihood arithmetic** (this is why
     `test_an_uninformative_observation_ignores_any_detection_probability_passed`
     and the deliberately-invalid-probability-with-`observation_informative=False`
     case both pass without raising);
  3. otherwise resolves `probability = config.detection_probability if
     detection_probability is None else detection_probability`, validates it
     (error message contains the literal substring `false-positive`), then
     runs the pre-existing numerator/denominator arithmetic, substituting the
     resolved `probability` in place of `config.detection_probability`.
- Defaults `(None, True)` reproduce F9b behaviour exactly — proved by
  `tests/test_pedestrian_ekf.py` passing unmodified (12/12) and by
  `test_default_update_reproduces_the_frozen_f9b_collapse` in the new
  existence tests.
- `P_S` and `P_birth` were not touched anywhere.

## TDD evidence

### RED

Command:
```
wsl.exe -d Ubuntu-Baru -- bash -lc 'cd /home/pannntastic/aivnv/duckie-pomdp && export PYTHONPATH=src:/home/pannntastic/aivnv/duckie/src && export DUCKIETOWN_HEADLESS=1 && /home/pannntastic/aivnv/duckie/.venv/bin/python -m pytest tests/test_f9c_observability.py tests/test_f9c_existence.py -q'
```

`test_f9c_observability.py` failed collection as expected (module did not yet exist):
```
ERROR collecting tests/test_f9c_observability.py
ImportError while importing test module '.../tests/test_f9c_observability.py'.
tests/test_f9c_observability.py:5: in <module>
    from duckie_pomdp.belief.observability import (
E   ModuleNotFoundError: No module named 'duckie_pomdp.belief.observability'
1 error in 0.21s
```

Ran `test_f9c_existence.py` alone (the collection error above short-circuits the combined
run) to confirm the second expected failure mode:
```
8 failed, 2 passed in 0.15s
```
The 2 passes were `test_default_update_reproduces_the_frozen_f9b_collapse` and
`test_ps_is_untouched_by_f9c`, which only call `update(False)` with no new kwargs (or don't
call `update` at all) and so exercise pre-existing behaviour. The other 8 all failed with
`TypeError: ExistenceFilter.update() got an unexpected keyword argument
'observation_informative'` or `'detection_probability'`, matching the brief's expectation of
`TypeError: update() got an unexpected keyword argument 'detection_probability'`.

Both failures are exactly the ones the brief predicted (`ModuleNotFoundError` for
observability, `TypeError` on the new `update()` kwargs for existence).

### GREEN

Command:
```
wsl.exe -d Ubuntu-Baru -- bash -lc 'cd /home/pannntastic/aivnv/duckie-pomdp && export PYTHONPATH=src:/home/pannntastic/aivnv/duckie/src && export DUCKIETOWN_HEADLESS=1 && /home/pannntastic/aivnv/duckie/.venv/bin/python -m pytest tests/test_f9c_observability.py tests/test_f9c_existence.py -q'
```
Output:
```
...................                                                      [100%]
19 passed in 0.14s
```
(18 from the brief + 1 added `test_pedestrian_predicted_between_centre_and_edge_is_mid_fov`.)

### Full suite

Command:
```
wsl.exe -d Ubuntu-Baru -- bash -lc 'cd /home/pannntastic/aivnv/duckie-pomdp && export PYTHONPATH=src:/home/pannntastic/aivnv/duckie/src && export DUCKIETOWN_HEADLESS=1 && /home/pannntastic/aivnv/duckie/.venv/bin/python -m pytest tests -q'
```
Output:
```
172 passed, 260 warnings in 11.82s
```
Matches the orchestrator's expected ~172 (153 baseline + 19 new).

`tests/test_pedestrian_ekf.py` run in isolation:
```
............                                                             [100%]
12 passed in 0.13s
```
This file was never opened with Edit/Write during this task — it is unmodified, and it is
fully green, which is the required proof that the default `update(detected)` call path
(`detection_probability=None, observation_informative=True`) reproduces F9b behaviour exactly.

## Self-review findings

- Verified `EffectiveDetectionModel.__init__` computes the missing-class set via
  `set(ObservabilityClass) - set(probability_by_class)`, which iterates the four enum members
  regardless of dict order, and the message uses the literal substring `"every observability
  class"` required by the test's `pytest.raises(..., match=...)`.
  `ExistenceFilterConfig` validation error still requires probability `> false_positive_probability`
  (strict), consistent with the pre-existing config-level check and the brief's
  `probability <= config.false_positive_probability` raise condition.
- Confirmed `ground_to_pixel` is called with `GroundPoint(x_left, y_forward)` — same argument
  order used throughout `test_camera_geometry.py` and `camera_geometry.py`.
- Confirmed the sign convention noted by the orchestrator (positive `x_left` → smaller pixel
  column) is irrelevant to the `abs(...)` binning math, so no special-casing was needed.
- Confirmed `y_forward <= 0` short-circuits before calling `ground_to_pixel` at all, so the
  behind-camera `ValueError` path is only reached for points with `y_forward > 0` whose
  homogeneous clip-space `w` still comes out non-positive (not currently exercised by these
  tests, but handled defensively per the brief's explicit instruction to catch `ValueError`).
- Did not touch `belief/__init__.py`, per instruction; `observability.py`'s new classes are not
  re-exported there and must be imported from `duckie_pomdp.belief.observability` directly
  (matches how the brief's own tests import them).

## Concerns (original submission)

- None blocking. One documentation note: `EffectiveDetectionModel`'s `outside_domain_miss_policy`
  parameter currently only recognizes the literal string `"prediction_only"` as
  "treat OUTSIDE_DOMAIN misses as uninformative"; any other string value (including typos)
  silently falls through to "informative", matching the letter of the brief but with no
  validation guard on the policy string itself — the brief's tests don't exercise an invalid
  policy value, so I left it unvalidated rather than invent an error contract not specified
  anywhere in the brief.

---

## Fix round 1/5: validate `outside_domain_miss_policy` at construction

**Trigger:** reviewer confirmed the self-raised concern above as an Important finding — a
policy typo (or renamed key) silently makes OUTSIDE_DOMAIN misses informative again, which
re-enables the exact belief-collapse behaviour F9c exists to eliminate. No exception, no
warning, no failing test. Invariant I3's own guard rail had no guard.

### What changed

`src/duckie_pomdp/belief/observability.py` only. Nothing else touched — `existence_filter.py`,
the binning logic (`classify`), and the other pre-existing tests are unchanged.

1. Added a module-level constant:
   ```python
   PREDICTION_ONLY_POLICY = "prediction_only"
   ```
   so the accepted string exists in exactly one place.
2. `EffectiveDetectionModel.__init__` now validates `outside_domain_miss_policy` immediately
   after the missing-class check, raising:
   ```python
   if outside_domain_miss_policy != PREDICTION_ONLY_POLICY:
       raise ValueError(
           "unsupported outside_domain_miss_policy: "
           f"{outside_domain_miss_policy!r}; only {PREDICTION_ONLY_POLICY!r} "
           "is supported"
       )
   ```
   The message names both the received value and the one supported value, so a config typo is
   diagnosable from the error text alone (e.g. `unsupported outside_domain_miss_policy:
   'prediction-only'; only 'prediction_only' is supported`).
3. `miss_is_informative` now compares against `PREDICTION_ONLY_POLICY` instead of a duplicated
   string literal.
4. Did **not** add a second policy branch (e.g. `"ordinary_miss"`) — there is exactly one
   supported policy in F9c v1; the fix is reject-and-fail-loud, not enum-completion.

### Pre-fix verification that the hole was real

Added the reviewer-specified test to `tests/test_f9c_observability.py`
(`test_effective_detection_model_rejects_an_unknown_outside_domain_policy`, using the
hyphenated near-miss `"prediction-only"`) and ran it **against the pre-fix code** first:

Command:
```
wsl.exe -d Ubuntu-Baru -- bash -lc 'cd /home/pannntastic/aivnv/duckie-pomdp && export PYTHONPATH=src:/home/pannntastic/aivnv/duckie/src && export DUCKIETOWN_HEADLESS=1 && /home/pannntastic/aivnv/duckie/.venv/bin/python -m pytest tests/test_f9c_observability.py::test_effective_detection_model_rejects_an_unknown_outside_domain_policy -q'
```
Output:
```
F                                                                        [100%]
=================================== FAILURES ===================================
___ test_effective_detection_model_rejects_an_unknown_outside_domain_policy ____

    def test_effective_detection_model_rejects_an_unknown_outside_domain_policy():
        ...
>       with pytest.raises(ValueError, match="prediction_only"):
E       Failed: DID NOT RAISE ValueError

tests/test_f9c_observability.py:111: Failed
1 failed in 0.23s
```
This confirms the hyphenated typo `"prediction-only"` was silently accepted by the pre-fix
constructor — no exception, meaning `EffectiveDetectionModel` was constructed successfully with
a policy string that does not equal `"prediction_only"`, and `miss_is_informative` would then
fall through to `return True` for OUTSIDE_DOMAIN misses, exactly the hole the reviewer
described.

### Post-fix verification (GREEN)

Command:
```
wsl.exe -d Ubuntu-Baru -- bash -lc 'cd /home/pannntastic/aivnv/duckie-pomdp && export PYTHONPATH=src:/home/pannntastic/aivnv/duckie/src && export DUCKIETOWN_HEADLESS=1 && /home/pannntastic/aivnv/duckie/.venv/bin/python -m pytest tests/test_f9c_observability.py tests/test_f9c_existence.py -q'
```
Output:
```
....................                                                     [100%]
20 passed in 0.15s
```
(19 previous + 1 new guard-rail test.)

Full suite:
```
wsl.exe -d Ubuntu-Baru -- bash -lc 'cd /home/pannntastic/aivnv/duckie-pomdp && export PYTHONPATH=src:/home/pannntastic/aivnv/duckie/src && export DUCKIETOWN_HEADLESS=1 && /home/pannntastic/aivnv/duckie/.venv/bin/python -m pytest tests -q'
```
Output:
```
173 passed, 260 warnings in 11.76s
```
Matches the coordinator's expected 173 (172 previous + 1 new test).

`tests/test_pedestrian_ekf.py` re-run in isolation, unaffected by this round (fix touched only
`observability.py`):
```
wsl.exe -d Ubuntu-Baru -- bash -lc 'cd /home/pannntastic/aivnv/duckie-pomdp && export PYTHONPATH=src:/home/pannntastic/aivnv/duckie/src && export DUCKIETOWN_HEADLESS=1 && /home/pannntastic/aivnv/duckie/.venv/bin/python -m pytest tests/test_pedestrian_ekf.py -q'
```
Output:
```
............                                                             [100%]
12 passed in 0.16s
```

### Files touched in this fix round

- Modified: `src/duckie_pomdp/belief/observability.py` (added `PREDICTION_ONLY_POLICY`
  constant; added policy validation in `EffectiveDetectionModel.__init__`; `miss_is_informative`
  now references the constant instead of a duplicated literal)
- Modified: `tests/test_f9c_observability.py` (added
  `test_effective_detection_model_rejects_an_unknown_outside_domain_policy`, exactly as
  specified by the coordinator)
- Not touched: `src/duckie_pomdp/belief/existence_filter.py`, `tests/test_f9c_existence.py`,
  `classify()`/binning logic, any other test file

### Concerns after fix round 1

None. The guard rail now has its own guard, validated at construction with a single source of
truth for the accepted policy string, and the reviewer's exact test (with the hyphenated
near-miss) is green with a demonstrated pre-fix failure as evidence the hole was real.
