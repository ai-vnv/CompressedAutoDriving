# Task 10 report — freeze the configuration

## Summary

`configs/f9c_robust_belief_v1.toml`'s `[measurement_model]`,
`[covariance_calibration]`, and `[conditional_detection]` sections were
filled from `artifacts/f9c_calibration_metrics.json` — the **final**
calibration re-run (fixed `approach_near_moving_ego` scenario, miss-likelihood
floor implemented, false-positive-rate contamination fixed; see
`task-9-report.md`, "Final internally-consistent calibration re-run") — and
all three `parameters_frozen` flags were set `true`. `artifacts/f9c_frozen_config.json`
was written recording the four provenance hashes, both seed lists, the full
fitted-parameter set, the pre-specified acceptance bands and minimum-support
floors, and `"final_evaluation_seeds_not_yet_rendered": true`.
`load_f9c_protocol(..., require_frozen=True)` now loads cleanly.
`experiments/verify_f9c_artifacts.py` was written as a read-only,
gracefully-degrading verifier. A carried-forward review gap
(`CovarianceCalibration` not validating non-negative floors) was closed.

**Frozen config SHA256: `359dc52020421c248bf6c26e036234191b2b97d24b505a66d85daa85b563704e`.**

No render, no experiment run, no read of seeds 7101–7104. Not a git
repository; no commits made.

## Values transcribed (all read programmatically, never hand-copied)

Extracted via `repr()` on Python floats read directly from
`artifacts/f9c_calibration_metrics.json` (guarantees exact IEEE-754
round-trip through the TOML literal):

```
measurement_model.bias_model = 'global_additive'
measurement_model.range_bias_m = -0.02986607430110723
measurement_model.bearing_bias_rad = 0.0012336629252072933
covariance_calibration.range_scale = 9.96243043243885
covariance_calibration.bearing_scale = 1.0
covariance_calibration.range_posterior_floor_m = 0.02041790926900693
covariance_calibration.bearing_posterior_floor_rad = 0.012546331734068323
conditional_detection.detection_probability_center = 0.9490486257928118
conditional_detection.detection_probability_mid_fov = 0.9801336146272855
conditional_detection.detection_probability_edge_fov = 0.997211155378486
conditional_detection.detection_probability_outside_domain = 0.5586734693877551
conditional_detection.false_positive_probability = 0.00078003120124805   (unchanged; already F9b's frozen value)
conditional_detection.miss_likelihood_floor = 0.37362469458201386
```

Source JSON key paths used (cross-checked against the artifact's own
`recommended_covariance_calibration` / `recommended_conditional_detection_miss_likelihood_floor`
convenience fields, which matched exactly):

| Config field | Artifact key path |
| --- | --- |
| `bias_model` | `bias.selected_model` |
| `range_bias_m` / `bearing_bias_rad` | `bias.fit.range_bias_m` / `bias.fit.bearing_bias_rad` |
| `range_scale` / `bearing_scale` | `covariance_scales.lambda_r` / `covariance_scales.lambda_beta` |
| `range_posterior_floor_m` | `variance_components.range.sigma_floor_m` |
| `bearing_posterior_floor_rad` | `variance_components.bearing.sigma_floor_rad` |
| `detection_probability_{center,mid_fov,edge_fov,outside_domain}` | `effective_detection.detection_probability_{...}` |
| `false_positive_probability` | `miss_likelihood_floor.adjusted_using_frozen_f9b_false_positive_rate.false_positive_probability_used` |
| `miss_likelihood_floor` | `miss_likelihood_floor.adjusted_using_frozen_f9b_false_positive_rate.lr_floor` |

**Confirmed: `lambda_r = 9.96243043243885`, not `10.125`.** The `10.125`
number appears only in `task-9-report.md`'s pre-floor, crashed-scenario
first-pass section and was never read for this task.

`[existence].detection_probability` (F9b's frozen scalar, `0.9766775777414075`)
was deliberately **left untouched** — it is outside the three sections this
task is scoped to fill, is intentionally unfrozen-but-not-this-task's-job
per the Task 1 freeze-boundary table, and is architecturally independent of
`[conditional_detection]`'s per-FOV-class values (`load_robust_observation_config`
in `src/duckie_pomdp/evaluation/f9c_calibration.py:947-996` builds
`EffectiveDetectionModel` from `[conditional_detection]` directly; it never
reads `[existence].detection_probability`).

## Full artifact-vs-config comparison (full float precision, bit-for-bit `==`)

Programmatically re-read both files and compared every value with Python
`==` (not `pytest.approx`) after writing the config:

```
OK  measurement_model.bias_model: config='global_additive' artifact='global_additive'
OK  measurement_model.range_bias_m: config=-0.02986607430110723 artifact=-0.02986607430110723
OK  measurement_model.bearing_bias_rad: config=0.0012336629252072933 artifact=0.0012336629252072933
OK  covariance_calibration.range_scale: config=9.96243043243885 artifact=9.96243043243885
OK  covariance_calibration.bearing_scale: config=1.0 artifact=1.0
OK  covariance_calibration.range_posterior_floor_m: config=0.02041790926900693 artifact=0.02041790926900693
OK  covariance_calibration.bearing_posterior_floor_rad: config=0.012546331734068323 artifact=0.012546331734068323
OK  conditional_detection.detection_probability_center: config=0.9490486257928118 artifact=0.9490486257928118
OK  conditional_detection.detection_probability_mid_fov: config=0.9801336146272855 artifact=0.9801336146272855
OK  conditional_detection.detection_probability_edge_fov: config=0.997211155378486 artifact=0.997211155378486
OK  conditional_detection.detection_probability_outside_domain: config=0.5586734693877551 artifact=0.5586734693877551
OK  conditional_detection.false_positive_probability: config=0.00078003120124805 artifact=0.00078003120124805
OK  conditional_detection.miss_likelihood_floor: config=0.37362469458201386 artifact=0.37362469458201386

parameters_frozen: True True True
ALL MATCH
```

`experiments/verify_f9c_artifacts.py` re-performs this exact comparison as
its `fitted_parameters_match_calibration_artifact` check and reports the
same result on every run.

## Frozen F7 physics / invariant I7

```
ekf identical: True
prior_probability 0.5 0.5 True
survival_probability 0.995 0.995 True
birth_probability 0.005 0.005 True
association gate: 13.815510557964274 innovation gate: 9.21034037197618 I7 holds (assoc > gate): True
```

`[ekf]` and the three frozen `[existence]` keys are byte-identical to
`configs/oracle_ekf_v1.toml` (dict `==` comparison after TOML parse, not a
lossy string diff). Invariant I7 holds strictly.

## `require_frozen=True` load

```
$ python -c "
from pathlib import Path
from duckie_pomdp.evaluation.f9c_protocol import load_f9c_protocol
p = load_f9c_protocol(Path('configs/f9c_robust_belief_v1.toml'), require_frozen=True)
print('frozen config sha256:', p.config_sha256)
"
frozen config sha256: 359dc52020421c248bf6c26e036234191b2b97d24b505a66d85daa85b563704e
```

No exception. Matches the hash recorded in `artifacts/f9c_frozen_config.json`
and the value `experiments/verify_f9c_artifacts.py` reports.

## `artifacts/f9c_frozen_config.json`

Written with: `schema_version`, `gate`, `status`, `config_sha256`,
`checkpoint_sha256`, `calibration_artifact_sha256`, `frozen_f7_config_sha256`,
`calibration_seeds` (`[6101..6108]`), `final_evaluation_seeds` (`[7101..7104]`),
`final_evaluation_seeds_not_yet_rendered: true`, an ISO-8601 UTC timestamp
(`2026-08-08T16:38:47Z`), a `source_calibration_artifact` pointer, a `note`
documenting the freeze provenance and the point-of-no-return warning, the
complete `fitted_parameters` block (mirroring the three frozen config
sections, each with its own `parameters_frozen: true`), the pre-specified
`acceptance_bands`, and the pre-specified `minimum_support`.

Hashes (computed via `duckie_pomdp.evaluation.f9_protocol.sha256`, the same
function `f9c_protocol._validate` uses):

```
config_sha256               359dc52020421c248bf6c26e036234191b2b97d24b505a66d85daa85b563704e
checkpoint_sha256            3d4f816d440690493b856d25403a84a3249e4250599319c32569b97cb8d7482c
calibration_artifact_sha256  653eef59725c9bda857d15164e251799ae1ccc3b14eecfb5883d1470a5d569aa
frozen_f7_config_sha256      a4815c8d0e17f1868d51619ae51d2183c72832a022edce88aa3c10302594d701
```

`checkpoint_sha256` and `frozen_f7_config_sha256` match the values already
frozen elsewhere in the repo (the config's own `provenance.checkpoint_sha256`,
and `verify_f9_artifacts.py`'s `FROZEN_BASELINE_HASHES["configs/oracle_ekf_v1.toml"]`
respectively) — an independent cross-check that nothing upstream drifted.

## Carried-forward review item: negative posterior floor validation

`src/duckie_pomdp/belief/covariance_calibration.py`,
`CovarianceCalibration.__post_init__` now raises `ValueError` for a
negative `range_posterior_floor_m` or `bearing_posterior_floor_rad`, with a
message explaining that a negative floor would *shrink* reported uncertainty
via `floor_polar_standard_deviation`'s quadrature sum instead of inflating
it — silently defeating the calibration this task freezes by hand. Two new
tests in `tests/test_f9c_covariance_calibration.py`:
`test_negative_range_posterior_floor_is_rejected` and
`test_negative_bearing_posterior_floor_is_rejected`. The fitted values this
task actually wrote (`0.02041790926900693` / `0.012546331734068323`) are
positive and pass unaffected — `posterior_floor_from_components` always
returns a `sqrt`, so the fitting path was already safe; this closes the
hand-edit path specifically.

## Pre-existing tests updated in lockstep with the freeze

Two tests directly read the shared config file and asserted on its
pre-freeze placeholder values; both had to change or the freeze itself
would break the suite. Neither was added or removed — same two test
functions, updated assertions/docstrings only:

1. `tests/test_f9c_protocol.py::test_load_robust_observation_config_builds_a_coordinator_config`
   — `config.covariance.range_scale == pytest.approx(1.0)` →
   `pytest.approx(9.96243043243885)`.
2. `tests/test_f9c_covariance_calibration.py::test_load_miss_likelihood_floor_defaults_to_a_no_op`
   — renamed to `test_load_miss_likelihood_floor_reads_the_frozen_fitted_value`;
   `floor == pytest.approx(0.0)` → `pytest.approx(0.37362469458201386)`. This
   function's own docstring already said "(0.0, a strict no-op, until Task
   10 freezes a fitted value)" — updated to describe both states — so this
   was a foreseen, not accidental, consequence of the freeze.

I verified no other test reads `configs/f9c_robust_belief_v1.toml` and
asserts on a covariance/bias/detection-probability/miss-floor value (grepped
every `test_f9c_*.py` file for `range_scale`, `bearing_scale`,
`posterior_floor`, `detection_probability_*`, `miss_likelihood_floor`,
`range_bias_m`, `bearing_bias_rad` in combination with the config path; the
only other hits were literal-constructor test fixtures unrelated to the
shared TOML file).

## `experiments/verify_f9c_artifacts.py`

Modeled on `experiments/verify_f9_artifacts.py`. Runs two tiers of checks:

**Runs now and passes (6 checks, all green):**
1. `frozen_config_loads` — `require_frozen=True` load succeeds.
2. `frozen_config_artifact_hashes` — all 4 hashes in `f9c_frozen_config.json`
   independently re-verified against the files they name; seed lists and the
   `final_evaluation_seeds_not_yet_rendered` literal checked too.
3. `fitted_parameters_match_calibration_artifact` — the same 13-value
   bit-for-bit comparison shown above.
4. `frozen_f7_physics_unperturbed` — `[ekf]` + 3 `[existence]` keys.
5. `invariant_i7_association_looser_than_gate`.
6. `upstream_baseline_hashes_unchanged` — F5b/F6/F7 artifacts (same
   `FROZEN_BASELINE_HASHES` dict as `verify_f9_artifacts.py`) untouched.

**Skipped now, will run after Task 11 (9 checks, all cleanly SKIP, not FAIL):**
existence of `f9c_validation.csv` / `f9c_belief_metrics.json` /
`f9c_nis_metrics.json`; final-evaluation seed/scenario/frame-matrix
completeness; belief-metrics and NIS-metrics config-hash consistency; miss
rows carrying empty geometry; error-case image presence; and a schema-tolerant
best-effort re-derivation of range RMSE/coverage directly from the CSV
(`rederive_belief_metrics_from_csv`), cross-checked against whatever the
belief-metrics JSON reports via a fuzzy recursive key search (`_find_key_paths`)
rather than a hardcoded schema import — because `evaluation.f9c_belief`
(Task 11's module) does not exist yet, this script deliberately does not
import anything from it.

A dedicated `SchemaSkip` exception (distinct from `AssertionError`) lets a
check report "I could not even attempt this because the expected
column/field isn't there" as `SKIP`, never as `FAIL` — so a future schema
choice by Task 11 that differs from this script's naming guess will not be
misreported as an artifact defect.

**Verified the script's behavior directly, not just by inspection:**
- Clean run today: 6 PASS, 9 SKIP, 0 FAIL, exit 0.
- Injected a corrupted `calibration_artifact_sha256` into
  `f9c_frozen_config.json` (backed up first): `frozen_config_artifact_hashes`
  correctly reported FAIL with the claimed-vs-actual hashes, and the process
  exited non-zero (confirmed via `cmd && echo ZERO || echo NONZERO`, not via
  `$?` capture — this sandbox's command wrapper resets `$?` between
  semicolon-separated statements, which cost some time to diagnose; see
  Concerns). Restored the file immediately after, then re-verified the clean
  exit-0 state.
- Wrote a **synthetic** Task-11-shaped `f9c_validation.csv` /
  `f9c_belief_metrics.json` / `f9c_nis_metrics.json` / one placeholder PNG in
  `f9c_error_cases/`, built from `protocol.scenarios`/`protocol.final_evaluation_seeds`
  so the seed/scenario/frame-matrix check would have real data to validate,
  but with placeholder measurement columns only (no belief-mean/std columns)
  to exercise the "Task 11 used different column names" path deliberately.
  Result: 12 PASS (including the newly-real seed-leakage, frame-matrix,
  hash, and miss-geometry checks) + 1 clean SKIP
  (`rederive_belief_metrics_from_csv`, naming exactly which 6 columns it
  couldn't find) + 0 FAIL, exit 0. **Deleted all synthetic files immediately
  after** (`artifacts/f9c_validation.csv`, `artifacts/f9c_belief_metrics.json`,
  `artifacts/f9c_nis_metrics.json`, `artifacts/f9c_error_cases/`); confirmed
  `artifacts/` is back to only the real Task 9/10 outputs and the verifier
  reports the original 6-PASS/9-SKIP clean state again. No seed 7101–7104
  data was read or referenced; the synthetic seed labels were placeholder
  strings, not rendered content.

## Files changed

- `configs/f9c_robust_belief_v1.toml` — filled and froze the three sections.
- `src/duckie_pomdp/belief/covariance_calibration.py` — non-negative floor
  validation.
- `src/duckie_pomdp/evaluation/f9c_calibration.py` — updated one docstring
  (`load_miss_likelihood_floor`) to describe both pre/post-freeze states;
  no behavior change.
- `tests/test_f9c_covariance_calibration.py` — 2 new negative-floor tests;
  1 existing test updated (frozen value, renamed).
- `tests/test_f9c_protocol.py` — 1 existing test updated (frozen value).
- `IMPLEMENTATION_NOTES.md` — Task 10 section appended under `## F9c`.
- `GATES.md` — F9c row added (`IN PROGRESS`, pre-final-run witness).

## Files created

- `artifacts/f9c_frozen_config.json`.
- `experiments/verify_f9c_artifacts.py`.

## Test suite

```
209 passed, 0 failed, 0 skipped
```
(207 pre-existing + 2 new: the two negative-floor tests. The two
pre-existing tests that needed updating were modified in place, not added,
so they do not change the count.)

## Self-review

- Every numeric value in the config was sourced by `repr()` on a Python
  float read directly from the JSON artifact, never typed by hand from
  looking at digits — the exact failure mode this task warns against.
- Cross-checked the artifact's own `recommended_covariance_calibration` and
  `recommended_conditional_detection_miss_likelihood_floor` convenience
  fields against the primary `covariance_scales`/`variance_components`/
  `miss_likelihood_floor` sources before using either; they agreed exactly.
- Did not touch `[existence].detection_probability` — confirmed by reading
  `load_robust_observation_config` that it is architecturally independent
  of `[conditional_detection]`, and it is outside this task's explicit
  three-section scope.
- Searched every `test_f9c_*.py` file for assertions against the shared
  config's now-changed values before declaring the freeze test-suite-safe,
  rather than assuming the two I found were the only two.
- Ran the numeric artifact-vs-config comparison as an independent script
  (not by eyeballing the diff), and had `verify_f9c_artifacts.py` re-do the
  identical comparison as one of its own checks — two independent
  implementations agreeing.
- Actually exercised the verifier's FAIL path (injected a real hash
  mismatch) and its SKIP-vs-FAIL distinction (synthetic mismatched-schema
  fixture), rather than trusting the code by inspection alone; found and
  fixed one real bug in the process (see Concerns).
- Cleaned up every synthetic/temporary file (`/tmp` scratch files and the
  synthetic Task-11 fixtures) before finishing; `artifacts/` and `configs/`
  contain only the files this task and its predecessors intentionally
  produced.
- Did not re-fit, re-run, or touch the simulator/detector/GPU anywhere in
  this task. Did not read seeds 7101–7104 (the synthetic-fixture seed
  *labels* were placeholder strings copied from `protocol.final_evaluation_seeds`
  for shape only — no simulator was invoked, no frame was rendered).
- Not a git repository; made no `git init`/`add`/`commit`.

## Concerns

1. **`experiments/verify_f9c_artifacts.py`'s Task-11-dependent re-derivation
   is a best-effort guess at Task 11's CSV/JSON schema**, because Task 11
   has not been planned or implemented yet (`task-11-brief.md` does not
   exist in this session). I modeled the guessed column names
   (`corrected_belief_range_m`, `reported_range_std_m`, etc.) on the
   existing precedent in `experiments/validate_f9_yolo_ekf.py`'s
   `_belief_fields` helper and `robust_updater.RobustStepRecord`'s field
   names, which is the strongest signal available, but if Task 11 names
   things differently, `rederive_belief_metrics_from_csv` will report
   `SKIP` (not silently pass, not crash — verified) until this script's
   candidate-column lists are updated. This is disclosed in the script's
   own module docstring as the expected maintenance point.
2. This sandbox's command wrapper appears to inject something (a DEBUG-trap-like
   mechanism) between semicolon-separated Bash statements that resets `$?`
   before a subsequent `echo "$?"` can observe it — even plain `false;
   echo "RC=$?"` reported `RC=0`. This is an artifact of the execution
   environment, not of the F9c code; I worked around it by testing exit
   codes with `cmd && echo A || echo B` (short-circuit evaluation, unaffected)
   instead of variable capture, and confirmed the verifier's exit-code
   behavior is correct both ways once the right test technique was used.
   Flagging in case it affects other tasks' verification methodology.
3. `GATES.md`'s new F9c row is marked `IN PROGRESS` with the calibration
   numbers as a "pre-final-run witness," per the brief's Step 5 instruction
   — it is deliberately not a `PASSED`/`FAILED` verdict, since Task 11
   (final evaluation) and Task 13 (leakage tests / gate report) have not
   run.
