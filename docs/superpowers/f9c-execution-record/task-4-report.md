# Task 4 Report: Temporal measurement association

## What was implemented

Created `src/duckie_pomdp/belief/measurement_association.py`:

- `AssociationConfig(chi_square_gate: float, initialization_rule: str)` — validates the gate
  is finite and positive.
- `CandidateMeasurement(measurement: ObjectMeasurement, confidence: float, bbox_key: tuple[int,int,int,int])`.
- `AssociationResult(selected_index, selected, mode, candidate_nis, highest_confidence_index, differed_from_highest_confidence)`.
- `MeasurementAssociator.associate(candidates, *, predicted_measurement, innovation_covariance_for) -> AssociationResult`.

Behavior:
- Empty `candidates` → `mode="no_candidate"`, `selected=None`, `selected_index=None`,
  `candidate_nis=()`, `highest_confidence_index=None`.
- `predicted_measurement is None` (no active track) → NIS is skipped entirely; selection is by
  the same `(-confidence, bbox_key)` ordering as `select_single_duckie` in
  `perception/f9_pipeline.py` (mirrored, not imported, since that file must not be modified and
  I did not want a perception→belief reverse dependency); `mode="initialization"`;
  `candidate_nis` is a tuple of `None` the same length as `candidates`.
- Active track present → for every candidate, build innovation
  `[z_r − ẑ_r, wrap_angle(z_β − ẑ_β)]`, get `S = innovation_covariance_for(z_r)`, compute NIS via
  the Task-3 `normalized_innovation_squared`. Candidates with `nis <= chi_square_gate` are
  eligible; the eligible candidate with the smallest NIS is selected (`mode="temporal"`).
  If no candidate is eligible, `mode="all_gated_out"`, `selected=None`, `selected_index=None`,
  but `candidate_nis` still holds every candidate's actual NIS value (not None) — useful for
  diagnostics.
- `highest_confidence_index` and `differed_from_highest_confidence` are always computed/reported
  when candidates exist, in both modes, so a caller can compare the two selection strategies.

**Invariant I1**: the module never builds `S` itself. It only calls the injected
`innovation_covariance_for` callable and consumes its return value directly in
`normalized_innovation_squared`. No `np.diag(...)`, no `H P Hᵀ` anywhere in the module. This
is stated explicitly in the module docstring, closely following the brief's required wording:
"`associate` thresholds against exactly the covariance its caller will later correct with."

Created `tests/test_f9c_association.py` exactly as given in the brief's Step 1 (7 tests,
unmodified).

## TDD evidence

### RED

Command:
```
wsl.exe -d Ubuntu-Baru -- bash -lc 'cd /home/pannntastic/aivnv/duckie-pomdp && export PYTHONPATH=src:/home/pannntastic/aivnv/duckie/src && export DUCKIETOWN_HEADLESS=1 && /home/pannntastic/aivnv/duckie/.venv/bin/python -m pytest tests/test_f9c_association.py -q'
```

Output:
```
==================================== ERRORS ====================================
________________ ERROR collecting tests/test_f9c_association.py ________________
ImportError while importing test module '/home/pannntastic/aivnv/duckie-pomdp/tests/test_f9c_association.py'.
...
E   ModuleNotFoundError: No module named 'duckie_pomdp.belief.measurement_association'
=========================== short test summary info ============================
ERROR tests/test_f9c_association.py
!!!!!!!!!!!!!!!!!!!! Interrupted: 1 error during collection !!!!!!!!!!!!!!!!!!!!
1 error in 0.29s
```

This is the expected failure: the module did not yet exist. Matches the brief's Step 2
expectation exactly.

### GREEN

Command:
```
wsl.exe -d Ubuntu-Baru -- bash -lc 'cd /home/pannntastic/aivnv/duckie-pomdp && export PYTHONPATH=src:/home/pannntastic/aivnv/duckie/src && export DUCKIETOWN_HEADLESS=1 && /home/pannntastic/aivnv/duckie/.venv/bin/python -m pytest tests/test_f9c_association.py -v'
```

Output:
```
collecting ... collected 7 items

tests/test_f9c_association.py::test_without_an_active_track_the_highest_confidence_candidate_initializes PASSED [ 14%]
tests/test_f9c_association.py::test_initialization_breaks_exact_confidence_ties_deterministically PASSED [ 28%]
tests/test_f9c_association.py::test_with_an_active_track_the_most_consistent_candidate_wins_over_confidence PASSED [ 42%]
tests/test_f9c_association.py::test_association_wraps_bearing_across_pi PASSED [ 57%]
tests/test_f9c_association.py::test_every_candidate_outside_the_gate_yields_no_selection PASSED [ 71%]
tests/test_f9c_association.py::test_no_candidates_is_reported_distinctly_from_all_gated_out PASSED [ 85%]
tests/test_f9c_association.py::test_associate_signature_admits_no_privileged_argument PASSED [100%]

============================== 7 passed in 0.14s ===============================
```

## Full-suite result

Command:
```
wsl.exe -d Ubuntu-Baru -- bash -lc 'cd /home/pannntastic/aivnv/duckie-pomdp && export PYTHONPATH=src:/home/pannntastic/aivnv/duckie/src && export DUCKIETOWN_HEADLESS=1 && /home/pannntastic/aivnv/duckie/.venv/bin/python -m pytest tests -q'
```

Output (final line):
```
140 passed, 260 warnings in 11.60s
```

133 pre-existing + 7 new = **140 passed**. This matches the task's corrected expectation (the
brief's own checkpoint of 139 was stale relative to the current baseline of 133 stated in the
task instructions; observed actual count is 140). No failures, no errors, no skips.

## Files changed

- Created: `src/duckie_pomdp/belief/measurement_association.py`
- Created: `tests/test_f9c_association.py`

No other files were modified. `src/duckie_pomdp/belief/__init__.py` was left untouched per
instruction (exports belong to a later task). `src/duckie_pomdp/perception/f9_pipeline.py` was
read only, not modified.

## Self-review

**Completeness**: All four `AssociationResult.mode` values are reachable and distinguished by
dedicated tests: `initialization` (no active track), `temporal` (active track, at least one
candidate passes gate), `all_gated_out` (active track, candidates present, none pass gate),
`no_candidate` (empty candidate list, checked before even looking at `predicted_measurement`).
The empty-list check happens first, so `no_candidate` takes priority over `initialization` when
both `candidates` is empty and `predicted_measurement` is None — this ordering isn't tested by
the brief's suite but is the only sane reading (there is nothing to report on either strategy
when there are zero candidates, so "no candidate" is a more informative label than
"initialization").

**Naming**: Matches the brief's interface list exactly — `AssociationConfig`,
`CandidateMeasurement`, `AssociationResult`, `MeasurementAssociator.associate`. Field names
(`selected_index`, `selected`, `mode`, `candidate_nis`, `highest_confidence_index`,
`differed_from_highest_confidence`) are verbatim from the brief.

**YAGNI**: No extra fields, no extra methods, no configuration knobs beyond
`chi_square_gate` and `initialization_rule` (the latter is stored but not branched on beyond
the one rule the tests exercise — the brief specifies only one rule value and doesn't ask for
alternate-rule dispatch, so I did not build a rule registry that has no second rule to
register).

**Do the tests verify real behavior?** Yes — I checked each by hand:
- Test 1: candidate 1 (conf 0.81) beats candidate 0 (conf 0.42) under `-confidence` ordering.
- Test 2: equal confidence 0.50, bbox (10,0,20,10) < (30,0,40,10) lexicographically → index 1
  wins the tie, matching `select_single_duckie`'s ordering.
- Test 3: candidate 0 has confidence 0.95 but bearing innovation 0.30 rad → large NIS;
  candidate 1 has confidence 0.31 but matches the prediction exactly → NIS 0, wins on
  temporal consistency despite lower confidence. `differed_from_highest_confidence` correctly
  True.
- Test 4: verifies `wrap_angle` is actually used — a plain subtraction of
  `(-π+0.01) − (π−0.01) = -2π+0.02` would produce a huge (wrong) bearing innovation and fail
  the gate; wrapped it becomes `0.02`, matching the expected NIS to 1e-6 relative tolerance.
- Test 5/6: distinguish `all_gated_out` (candidates present, all exceed gate) from
  `no_candidate` (empty list) — both assert `selected is None` but check different `mode`
  values, and I implemented them as genuinely different code paths (early return on empty list
  vs. late return after computing NIS for every candidate).
- Test 7: locks the public signature to exactly `{self, candidates, predicted_measurement,
  innovation_covariance_for}` — no extra defaulted parameter could sneak in unnoticed
  (e.g., an accidental "prefer highest confidence" escape hatch).

**Output pristine?** Full suite run shows only pre-existing deprecation warnings from
third-party dependencies (`zuper_nodes`, `gym`), nothing from the new module or test file.

**Invariant I1 audit**: grepped the new module for `np.diag`, `H @ P`, `P @ H`, `.dot(` — no
matches outside the docstring's negative example. The only covariance the module ever touches
is the return value of `innovation_covariance_for`.

## Concerns

None blocking. One minor design note for the record: `initialization_rule` is stored on
`AssociationConfig` but the module does not currently dispatch on its string value — it always
applies the `(-confidence, bbox_key)` rule regardless of what string is passed. This matches
what the brief's tests require (they only ever pass
`"highest_confidence_then_bbox_lexicographic"`), but if a future task introduces a second rule
value, this module will need a dispatch branch added at that time. I did not build speculative
dispatch machinery for a single known rule (YAGNI), per the brief's own emphasis on avoiding
that kind of gold-plating, but flagging it so it isn't missed later.
