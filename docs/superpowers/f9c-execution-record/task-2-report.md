# Task 2 Report — Final-evaluation near-range scenarios

## What was implemented

`configs/f9c_robust_belief_v1.toml`: the two calibration-only entries
`calibration_near_stationary` and `calibration_medium_stationary`
(`use_for_final_evaluation = false`) were **removed** and replaced with four
scenarios usable by both calibration and final evaluation:

| Scenario | pedestrian_mode | linear_velocity_mps | steps | ego_start_x_offset_m |
| --- | --- | ---: | ---: | ---: |
| `approach_near_stationary_ego` | stationary | 0.0 | 60 | 0.50 |
| `approach_medium_stationary_ego` | stationary | 0.0 | 60 | 0.25 |
| `approach_near_moving_ego` | stationary | 0.20 | 90 | **0.35** (brief said 0.30 — see Deviation below) |
| `cross_near_left_to_right` | cross_left_to_right | 0.0 | 110 | 0.40 |

All four set `use_for_calibration = true` and `use_for_final_evaluation =
true`. The six original F9b scenarios (`stationary_ped_stationary_ego`,
`stationary_ped_moving_ego`, `stationary_ped_turning_ego`,
`cross_left_to_right`, `cross_right_to_left`,
`crossing_moving_turning_ego`) were left byte-identical — diffed by eye
against the pre-Task-2 file; unchanged.

`tests/test_f9c_protocol.py`: added
`test_f9c_scenario_matrix_supports_near_range_final_evaluation` (the exact
test given in the brief's Step 1, verbatim).

`IMPLEMENTATION_NOTES.md`: appended a "Task 2" subsection under the existing
"F9c" section documenting the new scenarios, the deviation, and the Step 5
dry-run counts.

## TDD evidence

**RED** — added the test first, ran it against the pre-edit config:

```
wsl.exe -d Ubuntu-Baru -- bash -lc 'cd /home/pannntastic/aivnv/duckie-pomdp && export PYTHONPATH=src:/home/pannntastic/aivnv/duckie/src && export DUCKIETOWN_HEADLESS=1 && /home/pannntastic/aivnv/duckie/.venv/bin/python -m pytest tests/test_f9c_protocol.py::test_f9c_scenario_matrix_supports_near_range_final_evaluation -q'
```

```
AssertionError: final evaluation must contain near-range scenarios
assert 0 >= 2
1 failed in 0.23s
```

This matches the brief's stated expected failure exactly (`assert 0 >= 2`).

**GREEN** — after editing the config (with the one deviation described
below), ran the focused file:

```
wsl.exe -d Ubuntu-Baru -- bash -lc 'cd /home/pannntastic/aivnv/duckie-pomdp && export PYTHONPATH=src:/home/pannntastic/aivnv/duckie/src && export DUCKIETOWN_HEADLESS=1 && /home/pannntastic/aivnv/duckie/.venv/bin/python -m pytest tests/test_f9c_protocol.py -q'
```

```
.......                                                                  [100%]
7 passed in 0.19s
```

**Checkpoint** — full suite:

```
wsl.exe -d Ubuntu-Baru -- bash -lc 'cd /home/pannntastic/aivnv/duckie-pomdp && export PYTHONPATH=src:/home/pannntastic/aivnv/duckie/src && export DUCKIETOWN_HEADLESS=1 && /home/pannntastic/aivnv/duckie/.venv/bin/python -m pytest tests -q'
```

```
115 passed, 260 warnings in 11.66s
```

The 260 warnings are pre-existing third-party `PyparsingDeprecationWarning`
and one gym `Box bound precision lowered` warning from unrelated tests
(`test_camera_geometry.py`, `test_detection_dataset.py`,
`test_gym_duckietown_integration.py`, `test_minimal_scenario.py`,
`test_true_state_validation.py`); none originate from
`test_f9c_protocol.py` or the changed config.

## Deviation from the brief — found by TDD, documented, not silently made

The brief's Step 3 TOML block specifies `ego_start_x_offset_m = 0.30` for
`approach_near_moving_ego`. The brief's own Step 1 test (which I copied
verbatim) defines "near" as `ego_start_x_offset_m >= 0.35` and then requires
at least one near, final-evaluation scenario with `linear_velocity_mps >
0.0`. At `0.30`, `approach_near_moving_ego` is excluded from the "near" set
by the test's own filter — no scenario is simultaneously near (by the
test's threshold) and moving, so the final assertion is unsatisfiable
regardless of any other scenario in the file. Running the exact Step-3 TOML
against the exact Step-1 test confirmed this: `assert False` on the last
assertion, with the two `len(...) >= 2` assertions already passing.

I raised `ego_start_x_offset_m` for `approach_near_moving_ego` from `0.30`
to `0.35` — the minimal change that resolves the contradiction, using the
test's own threshold value rather than an arbitrary number, and touching
nothing else in that scenario (name, mode, velocity, steps, both usage
flags all remain as specified). This is called out with an inline TOML
comment at the point of the change and documented in
`IMPLEMENTATION_NOTES.md`. No other number or name from the brief was
altered. This is worth a human's explicit sign-off since the assignment
states "every number in the brief is exact" — I judged the Step-4 checkpoint
("Expected: PASS (7 tests)" / "expect 115 passed") as the binding contract
when two brief-supplied numbers could not both hold, rather than guessing
silently or blocking a task that had an obvious, narrow, well-justified fix.

## Step 5 dry-run (arithmetic only — no simulator/render run, per assignment)

Command run:

```python
from pathlib import Path
from duckie_pomdp.evaluation.f9c_protocol import load_f9c_protocol
protocol = load_f9c_protocol(Path("configs/f9c_robust_belief_v1.toml"))
final = [s for s in protocol.scenarios if s.use_for_final_evaluation]
print(sum(s.steps + 1 for s in final) * len(protocol.final_evaluation_seeds), "final frames")
```

**Result: 3468 final frames** (10 final-evaluation scenarios x 4 seeds
`7101-7104`; `sum(steps+1)` = 867 frames/seed).

Additional diagnostic (not requested verbatim, but useful and recorded):
restricting to the three scenarios with `ego_start_x_offset_m >= 0.35`
(`approach_near_stationary_ego`, `approach_near_moving_ego`,
`cross_near_left_to_right`) gives **1052 frames** across the 4 final seeds
(263/seed). This is a scenario-placement proxy count, not a rendered
`distance_bin` count — it comfortably clears the proportional
`minimum_support.near / 4 = 25`-per-seed floor, but real near/medium/far
classification depends on the actual simulated `range_m` per frame
(especially for `approach_near_moving_ego`, whose range decreases
continuously rather than sitting at one static value), which requires
rendering.

**Deferred to Task 9** (the first task that renders calibration seeds
`6101-6108` through the not-yet-existing Task 11 collector), per the
assignment's explicit instruction: actual `distance_bin` counting on a
rendered calibration seed (e.g. 6101). No simulator was run, no frame was
rendered, and no `7101-7104` seed was touched in this task.

## Every file changed

- `\\wsl.localhost\Ubuntu-Baru\home\pannntastic\aivnv\duckie-pomdp\configs\f9c_robust_belief_v1.toml`
  — replaced the two calibration-only near/medium scenarios with four
  calibration+final scenarios (one field, `approach_near_moving_ego`'s
  `ego_start_x_offset_m`, deviates from the brief's literal value; see
  above). Six original F9b scenarios unchanged.
- `\\wsl.localhost\Ubuntu-Baru\home\pannntastic\aivnv\duckie-pomdp\tests\test_f9c_protocol.py`
  — added `test_f9c_scenario_matrix_supports_near_range_final_evaluation`
  (verbatim from the brief).
- `\\wsl.localhost\Ubuntu-Baru\home\pannntastic\aivnv\duckie-pomdp\IMPLEMENTATION_NOTES.md`
  — appended a "Task 2" subsection under "F9c" documenting the scenario
  table, the deviation and its rationale, and the Step 5 dry-run counts.

No other files were touched. `src/duckie_pomdp/evaluation/f9c_protocol.py`
and `src/duckie_pomdp/evaluation/f9_protocol.py` (Task 1 interfaces) were
read but not modified — Task 2 is config- and test-only, as scoped.

## Self-review

- **Completeness**: all six brief steps done except the explicitly deferred
  half of Step 5 (real-render distance-bin count), which the assignment
  itself says not to attempt. Checkpoint (115 passed) matches exactly.
- **YAGNI**: no new production code, no new fields beyond what the brief's
  TOML block specifies, no speculative helper functions added to the test
  file.
- **Naming**: scenario names match the brief exactly
  (`approach_near_stationary_ego`, `approach_medium_stationary_ego`,
  `approach_near_moving_ego`, `cross_near_left_to_right`); no leftover
  `calibration_near_stationary`/`calibration_medium_stationary` duplicate
  entries (verified no duplicate-name validation error and via manual read
  of the file section).
- **Test quality**: the added test is the brief's own test verbatim,
  exercising real behavior (loader + dataclass field access), not a mock.
  No test was weakened to force a pass — the config was changed to make the
  test's own stated intent achievable.
- **Pristine output**: full-suite run has zero failures, zero errors, zero
  new warnings attributable to this change; the 260 warnings present are
  pre-existing third-party deprecation noise from unrelated test files.

## Concerns

1. The one substantive concern is the deviation above
   (`ego_start_x_offset_m` 0.30 -> 0.35 for `approach_near_moving_ego`). It
   is small, mechanically forced by the brief's own test, documented in
   three places (TOML comment, `IMPLEMENTATION_NOTES.md`, this report), and
   does not touch scenario names, counts, seeds, or any other task's
   deliverable — but it does mean the literal Step-3 TOML block in the
   brief, taken in isolation, is not what's in the repository. Flagging for
   explicit sign-off per the task's "ask if unclear" guidance (surfaced
   after the fact, since it was only discoverable by running the brief's
   own RED/GREEN sequence).
2. Step 5's real distance-bin verification is deferred to Task 9 as
   instructed; until that render happens, the near-support claim rests on
   the scenario-count proxy (1052 frames / 263 per seed) rather than actual
   binned truth. This is a known, called-out limitation, not a hidden gap.

---

## Fix round 1/5 — coordinator ruling applied

**Ruling:** the near-range membership *test* was wrong, not the config.
`approach_near_moving_ego` deliberately starts at `ego_start_x_offset_m =
0.30` (medium range) and drives in at 0.20 m/s for 90 steps, so it
*traverses* into near range — that traversal is the scenario's entire
purpose ("sweeps range continuously downward, which is what makes the near
bin a traversed regime rather than a single static pose"). Classifying
near-range membership by start offset alone excludes exactly the scenario
designed to reach near range by motion rather than by static placement, and
my Fix-round-0 change (raising the offset to 0.35) would have shortened the
traversal and reduced the medium-range frames the scenario contributes. The
canonical plan file was updated to match this ruling before I applied it.

### What was changed

1. **`configs/f9c_robust_belief_v1.toml`** — reverted
   `approach_near_moving_ego`'s `ego_start_x_offset_m` from `0.35` back to
   `0.30`; removed the inline comment I had added justifying `0.35`. No
   other field in that scenario, or any other scenario, was touched.

2. **`tests/test_f9c_protocol.py`** —
   `test_f9c_scenario_matrix_supports_near_range_final_evaluation`'s
   near-range membership predicate was replaced with the coordinator's exact
   `reaches_near(spec)` helper (start-offset `>= 0.35`, OR moving with
   start-offset `>= 0.25`), applied to both `near_final` and
   `near_calibration`. The three downstream assertions
   (`len(near_final) >= 2`, `len(near_calibration) >= 2`, and the
   `any(... linear_velocity_mps > 0.0 ...)` check) were left unchanged, as
   instructed.

3. **`IMPLEMENTATION_NOTES.md`** — replaced the Fix-round-0 "deviation"
   paragraph with a "Resolved ambiguity ... (ruling recorded)" paragraph
   describing the ruling and the actual resolution (config keeps 0.30, test
   predicate now accounts for traversal). Also corrected the scenario table
   row, the Step 5 dry-run paragraph (recomputed the near-frame proxy count
   under the new predicate — still 1052 frames, since scenario `steps` did
   not change, only which scenarios qualify as "near" did), and the
   closing test-description sentence, all of which had referenced the
   now-reverted `0.35` value or the old `>= 0.35`-only filter.

### Covering tests run

Focused file:

```
wsl.exe -d Ubuntu-Baru -- bash -lc 'cd /home/pannntastic/aivnv/duckie-pomdp && export PYTHONPATH=src:/home/pannntastic/aivnv/duckie/src && export DUCKIETOWN_HEADLESS=1 && /home/pannntastic/aivnv/duckie/.venv/bin/python -m pytest tests/test_f9c_protocol.py -q'
```

```
.......                                                                  [100%]
7 passed in 0.19s
```

Full suite:

```
wsl.exe -d Ubuntu-Baru -- bash -lc 'cd /home/pannntastic/aivnv/duckie-pomdp && export PYTHONPATH=src:/home/pannntastic/aivnv/duckie/src && export DUCKIETOWN_HEADLESS=1 && /home/pannntastic/aivnv/duckie/.venv/bin/python -m pytest tests -q 2>&1 | tail -5'
```

```
  /home/pannntastic/aivnv/duckie/.venv/lib/python3.10/site-packages/gym/spaces/box.py:127: UserWarning: [33mWARN: Box bound precision lowered by casting to float32[0m
    logger.warn(f"Box bound precision lowered by casting to {self.dtype}")

-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html
115 passed, 260 warnings in 11.26s
```

Both green; no new warnings introduced by this fix. `approach_near_moving_ego`
now reads exactly as in the brief's Step 3 TOML (`ego_start_x_offset_m =
0.30`, no added comment), and the test's near-range predicate matches the
coordinator's ruling verbatim. No prior concern remains open from Fix round 0.
