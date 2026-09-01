# Task 3b report: Frozen F9c bias correction as an explicit runtime stage

## What was implemented

- `src/duckie_pomdp/belief/bias_correction.py` (new): `FrozenBiasCorrection`, a
  frozen dataclass with fields, in this exact order (positional args in the
  tests rely on this order):
  `model, range_bias_m, bearing_bias_rad, range_bin_bias_m, near_max_m, medium_max_m`.
  - `model` must be one of `"identity"`, `"global_additive"`, `"per_range_bin"`
    (validated in `__post_init__`, raises `ValueError` otherwise).
  - `per_range_bin` requires `range_bin_bias_m` to be present with exactly the
    keys `near`, `medium`, `far`; otherwise raises `ValueError` with the
    literal substring `near, medium, far` in the message.
  - `correct(measurement)`:
    - returns the same object (`is` identity, not just equality) when
      `measurement.detected` is `False`, and also when `model == "identity"`
      (a true no-op regardless of what bias fields happen to be set — not
      merely a no-op because `identity()`'s classmethod zeroes them).
    - otherwise selects a range bias: for `global_additive`, `self.range_bias_m`
      directly; for `per_range_bin`, looks up the bin from the **measured**
      range (`measurement.range_m`, never a predicted/ground-truth range —
      documented in a comment on `_select_range_bias` to prevent a feedback
      loop into the estimator).
    - clamps with `max(0.0, range_m - bias)`, wraps the bearing with
      `wrap_angle` from `duckie_pomdp.perception.measurement_calibration`,
      and rebuilds `x_left_m`/`y_forward_m` from the corrected polar pair
      using `range * sin(bearing)` / `range * cos(bearing)` — same shape as
      `AdditiveMeasurementBias.correct` in `perception/f9_pipeline.py:42-61`.
  - `identity()` classmethod: returns `FrozenBiasCorrection(model="identity",
    range_bias_m=0.0, bearing_bias_rad=0.0, range_bin_bias_m=None,
    near_max_m=0.55, medium_max_m=0.80)`.
  - `from_config(data: Mapping)` classmethod: reads `model`, `range_bias_m`,
    `bearing_bias_rad`, `near_max_m`, `medium_max_m` by key, and
    `range_bin_bias_m` via `.get` (optional, defaults to `None`).
  - Bin edges: `range <= near_max_m` -> `near`; `near_max_m < range <=
    medium_max_m` -> `medium`; else `far`. Matches all three brief test
    fixtures (0.50/0.55 -> near, 0.70 within (0.55, 0.80] -> medium, 0.95 ->
    far).

- `tests/test_f9c_bias_correction.py` (new): the 9 tests exactly as given in
  the brief's Step 1, verbatim.

`src/duckie_pomdp/perception/f9_pipeline.py` (`AdditiveMeasurementBias`) and
`src/duckie_pomdp/belief/__init__.py` were **not** modified, per the brief and
the ambiguity resolutions.

## TDD evidence

### RED

Command:
```
wsl.exe -d Ubuntu-Baru -- bash -lc 'cd /home/pannntastic/aivnv/duckie-pomdp && export PYTHONPATH=src:/home/pannntastic/aivnv/duckie/src && export DUCKIETOWN_HEADLESS=1 && /home/pannntastic/aivnv/duckie/.venv/bin/python -m pytest tests/test_f9c_bias_correction.py -q'
```

Output (abridged):
```
==================================== ERRORS ====================================
______________ ERROR collecting tests/test_f9c_bias_correction.py ______________
ImportError while importing test module '.../tests/test_f9c_bias_correction.py'.
tests/test_f9c_bias_correction.py:4: in <module>
    from duckie_pomdp.belief.bias_correction import FrozenBiasCorrection
E   ModuleNotFoundError: No module named 'duckie_pomdp.belief.bias_correction'
=========================== short test summary info ============================
ERROR tests/test_f9c_bias_correction.py
!!!!!!!!!!!!!!!!!!!! Interrupted: 1 error during collection !!!!!!!!!!!!!!!!!!!!
1 error in 0.32s
```

Why this is the expected failure: the module `duckie_pomdp.belief.bias_correction`
did not exist yet at this point — exactly the brief's Step 2 expectation.

### GREEN

Command (same as above, after implementing the module):
```
wsl.exe -d Ubuntu-Baru -- bash -lc 'cd /home/pannntastic/aivnv/duckie-pomdp && export PYTHONPATH=src:/home/pannntastic/aivnv/duckie/src && export DUCKIETOWN_HEADLESS=1 && /home/pannntastic/aivnv/duckie/.venv/bin/python -m pytest tests/test_f9c_bias_correction.py -q'
```

Output:
```
.........                                                                [100%]
9 passed in 0.14s
```

## Full-suite result

Command:
```
wsl.exe -d Ubuntu-Baru -- bash -lc 'cd /home/pannntastic/aivnv/duckie-pomdp && export PYTHONPATH=src:/home/pannntastic/aivnv/duckie/src && export DUCKIETOWN_HEADLESS=1 && /home/pannntastic/aivnv/duckie/.venv/bin/python -m pytest tests -q'
```

Output tail:
```
132 passed, 260 warnings in 10.44s
```

Matches the brief's checkpoint (132, up from the recorded baseline of 123).
The 260 warnings are pre-existing deprecation/precision warnings from
third-party libraries (pyparsing, gym) unrelated to this change; re-verified
this by re-running the full suite after my one post-GREEN edit (see below)
and the count/pass total was unchanged.

## Files changed (complete list)

- Created: `src/duckie_pomdp/belief/bias_correction.py`
- Created: `tests/test_f9c_bias_correction.py`

No other files were touched. This is not a git repository (per the task's
ambiguity resolutions); no commit was made.

## Self-review

- **Completeness**: all 9 brief tests pass; `identity()`, `from_config()`,
  and the `per_range_bin` model are implemented as the stated additions over
  the reference `AdditiveMeasurementBias`. Field order matches the brief's
  positional-construction test exactly.
- **Naming**: `FrozenBiasCorrection`, `correct`, `identity`, `from_config`
  match the brief's interface spec verbatim. Internal helper
  `_select_range_bias` is private and named for what it does.
- **YAGNI**: no extra public surface was added. I considered but rejected
  adding a `__repr__`/config-serialization round-trip helper — not asked for.
- **One post-GREEN correctness change**: after the first GREEN run I revised
  `correct()` so `model == "identity"` short-circuits to a true no-op
  (`return measurement`) regardless of what `range_bias_m`/`bearing_bias_rad`
  happen to be set to on that instance, rather than relying only on
  `identity()`'s classmethod zeroing those fields. This makes "identity"
  mean identity by construction rather than by convention, matching the
  ambiguity resolution's "genuine no-op" requirement more literally. Since
  this is stricter than the reference shape (which has no identity concept),
  I re-ran both the targeted and full suites afterward to confirm no
  regression (9 passed, 132 passed).
- **Are the tests verifying real behavior**: yes — they exercise the
  additive correction sign, clamping at zero, angle wrap-around at ±π,
  bin selection at bin boundaries, missing-measurement pass-through by
  identity (`is`), and a signature check on `correct` that guards against a
  future edit smuggling extra (e.g. ground-truth) parameters into the
  method.
- **Output pristine**: the only build artifact touched by tests is
  `__pycache__`; no stray print/debug output in the module.

## Concerns

None. The task's scope was fully bounded by the brief and I did not need to
touch `f9_pipeline.py`, `belief/__init__.py`, or any config/YAML files that
might eventually carry F9c-fitted constants for `from_config` — wiring this
stage into the actual Robust-B runtime coordinator (association / gate / EKF
order from the brief's "Locked runtime order") is explicitly a later task,
not this one.

---

## Fix round 1/5

### What changed

Per the coordinator's ruling: kept the `if self.model == "identity": return
measurement` branch in `correct()` and added a test that actually exercises
it (rather than passing vacuously through zeroed fields). Two files touched,
both already in scope for this task:

- `tests/test_f9c_bias_correction.py`: added
  `test_identity_model_is_a_no_op_even_with_nonzero_bias_fields`, exactly as
  specified by the coordinator, appended after
  `test_correction_never_receives_ground_truth`. No other test was changed.
- `src/duckie_pomdp/belief/bias_correction.py`: added one docstring sentence
  to `correct()` recording why the identity branch is unconditional. No
  other production code changed — the branch itself is unchanged from the
  prior version.

`f9_pipeline.py` and `belief/__init__.py` were not touched.

### Branch-removal verification (the requested falsifiability check)

Before restoring the docstring update, I temporarily deleted the
`if self.model == "identity": return measurement` early-return from
`correct()` and ran only the new test in isolation:

Command:
```
wsl.exe -d Ubuntu-Baru -- bash -lc 'cd /home/pannntastic/aivnv/duckie-pomdp && export PYTHONPATH=src:/home/pannntastic/aivnv/duckie/src && export DUCKIETOWN_HEADLESS=1 && /home/pannntastic/aivnv/duckie/.venv/bin/python -m pytest tests/test_f9c_bias_correction.py::test_identity_model_is_a_no_op_even_with_nonzero_bias_fields -q'
```

Output:
```
F                                                                        [100%]
=================================== FAILURES ===================================
_________ test_identity_model_is_a_no_op_even_with_nonzero_bias_fields _________
...
    assert corrected.range_m == pytest.approx(0.900)
E       assert 0.9459000000000001 == 0.9 ± 9.0e-07
E
E       comparison failed
E       Obtained: 0.9459000000000001
E       Expected: 0.9 ± 9.0e-07

tests/test_f9c_bias_correction.py:132: AssertionError
=========================== short test summary info ============================
FAILED tests/test_f9c_bias_correction.py::test_identity_model_is_a_no_op_even_with_nonzero_bias_fields
1 failed in 0.25s
```

Without the branch, `model="identity"` falls through to the
`global_additive`/default case and applies `range_bias_m=-0.0459` directly
(`max(0.0, 0.900 - (-0.0459)) = 0.9459`), which is exactly what the failure
shows. This confirms the test is not vacuous — it fails specifically because
the identity semantics were not enforced. I then restored the branch
(re-added the exact same two lines) and added the docstring sentence.

### Covering tests re-run after restoring the branch

Targeted file:
```
wsl.exe -d Ubuntu-Baru -- bash -lc 'cd /home/pannntastic/aivnv/duckie-pomdp && export PYTHONPATH=src:/home/pannntastic/aivnv/duckie/src && export DUCKIETOWN_HEADLESS=1 && /home/pannntastic/aivnv/duckie/.venv/bin/python -m pytest tests/test_f9c_bias_correction.py -q'
```
Output:
```
..........                                                               [100%]
10 passed in 0.12s
```

Full suite:
```
wsl.exe -d Ubuntu-Baru -- bash -lc 'cd /home/pannntastic/aivnv/duckie-pomdp && export PYTHONPATH=src:/home/pannntastic/aivnv/duckie/src && export DUCKIETOWN_HEADLESS=1 && /home/pannntastic/aivnv/duckie/.venv/bin/python -m pytest tests -q'
```
Output tail:
```
133 passed, 260 warnings in 11.02s
```

Matches the coordinator's expected 10/10 and 133 (up from 132 before this
fix round). No other test count changed.

### Concerns

None. The branch is now covered by a test that fails when the branch is
absent, which was the exact gap the review identified.
