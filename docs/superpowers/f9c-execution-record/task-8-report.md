# Task 8 report: Robust belief coordinator

## What was implemented

- `src/duckie_pomdp/belief/robust_updater.py` (new). Contains:
  - `RobustObservationSwitches` — the five ablation booleans, declared locally
    (mirrors `evaluation.f9c_protocol.RobustObservationSwitches` field-for-field
    by design, rather than importing it, so `belief` does not depend on the
    `evaluation` experiment-harness layer above it).
  - `RobustObservationConfig(switches, gate, association, covariance,
    effective_detection, active_threshold, delete_threshold,
    initialization_threshold)` with threshold validation
    (`delete_threshold < initialization_threshold`, all three in `[0, 1]`).
  - `RobustStepRecord` — the per-frame diagnostic record, exactly the fields
    listed in the task brief.
  - `_LambdaInflatedNoise` — a private duck-typed adapter installed on the
    frozen `PedestrianEKF`'s `measurement_noise` attribute at construction
    time. Its `.covariance(r)` returns `calibration.inflate(base.covariance(r))`;
    its `.bias(r)` passes through the base model unchanged (see Invariant I1
    discussion below for why that is safe).
  - `RobustPedestrianBeliefUpdater` — the coordinator. `__init__` builds the
    adapter once and installs it on a fresh `PedestrianEKF` (profile
    `CALIBRATED_RESIDUAL`), builds `ExistenceFilter`, `MeasurementAssociator`,
    `InnovationGate`. `_innovation_covariance(range_m)` is the single I1
    provider. `update(previous_belief, previous_action, ego_motion,
    candidates, dt_s) -> tuple[BeliefState, RobustStepRecord]` implements the
    12-step order of operations from the brief verbatim, including the
    track-lifecycle state machine (no-track / initialization / temporal /
    deleted) and the posterior-floor-at-report-only rule.
- `tests/test_f9c_robust_updater.py` (new). All 11 tests from the brief's
  Step 1, arrangements filled in from the docstring stubs. Reuses the
  `measurement()` helper shape from `tests/test_pedestrian_ekf.py:43`
  (redefined locally with a comment pointing at that line, since `tests/` is
  not a package and no test file in this repo imports another test file).
- `src/duckie_pomdp/belief/__init__.py` (modified). Added exports for every
  F9c belief submodule that had none yet (`bias_correction`,
  `covariance_calibration`, `innovation_gate`, `measurement_association`,
  `observability`) plus the four new `robust_updater` names, alphabetized
  into the existing `__all__`.

## TDD evidence

**RED** — module absent (verified by temporarily moving the file aside and
re-running; this is the genuine failure the brief's Step 2 describes):

```
$ mv src/duckie_pomdp/belief/robust_updater.py /tmp/robust_updater.py.bak
$ PYTHONPATH=src:.../duckie/src pytest tests/test_f9c_robust_updater.py -q
ERROR tests/test_f9c_robust_updater.py - ModuleNotFoundError: No module named
'duckie_pomdp.belief.robust_updater'
1 error in 0.19s
$ mv /tmp/robust_updater.py.bak src/duckie_pomdp/belief/robust_updater.py
```

Intermediate RED (during first real run with the module in place): 4 of 11
failed on first attempt — a test-arrangement bug, not an implementation bug.
`test_a_duplicate_frame_selects_the_temporally_consistent_candidate` compared
against the raw (pre-bias-correction) candidate range instead of the
bias-corrected one the coordinator actually reports selecting.
`test_belief_survives_five_consecutive_misses_when_predicted_edge_fov`,
`test_an_outside_domain_absence_preserves_belief_far_longer_than_an_in_domain_one`,
and `test_belief_still_dies_after_a_long_absence` asserted the observability
class on the very first (track-initializing) frame, which necessarily falls
back to the zero-mean `initial_belief()` and classifies `OUTSIDE_DOMAIN` by
construction (no `x̂⁻` exists yet). Fixed by asserting bias-corrected values
and by adding a one-frame "settle" observation before checking observability
class, matching the class the coordinator actually reports on subsequent
frames. These were arrangement fixes only; `robust_updater.py` was not
changed in response.

**GREEN**:

```
$ PYTHONPATH=src:.../duckie/src DUCKIETOWN_HEADLESS=1 \
  pytest tests/test_f9c_robust_updater.py -v
tests/test_f9c_robust_updater.py::test_a_gross_outlier_does_not_move_the_belief_more_than_the_baseline_does PASSED
tests/test_f9c_robust_updater.py::test_reported_range_std_is_never_below_the_posterior_floor PASSED
tests/test_f9c_robust_updater.py::test_a_duplicate_frame_selects_the_temporally_consistent_candidate PASSED
tests/test_f9c_robust_updater.py::test_a_rejected_localization_is_not_an_existence_miss PASSED
tests/test_f9c_robust_updater.py::test_association_gate_and_correction_share_one_innovation_covariance PASSED
tests/test_f9c_robust_updater.py::test_belief_survives_five_consecutive_misses_when_predicted_edge_fov PASSED
tests/test_f9c_robust_updater.py::test_an_outside_domain_absence_preserves_belief_far_longer_than_an_in_domain_one PASSED
tests/test_f9c_robust_updater.py::test_belief_still_dies_after_a_long_absence PASSED
tests/test_f9c_robust_updater.py::test_a_deleted_track_reinitializes_from_the_next_valid_candidate PASSED
tests/test_f9c_robust_updater.py::test_switching_every_robust_component_off_reproduces_the_f9b_path PASSED
tests/test_f9c_robust_updater.py::test_updater_never_receives_privileged_state PASSED
============================== 11 passed in 0.24s ===============================
```

## Full-suite count

```
$ PYTHONPATH=src:.../duckie/src DUCKIETOWN_HEADLESS=1 pytest tests -q
184 passed, 260 warnings in 11.34s
```

184, matching the brief's "expect roughly 184" note (173 prior + 11 new).
Re-ran after removing an unused test import; still 184 passed, no
regressions anywhere else in the suite.

## `grep -n 'covariance('` output and explanation of every hit

```
$ grep -n "covariance(" src/duckie_pomdp/belief/robust_updater.py
135:    ``initialize()`` -- use exactly ``calibration.inflate(base.covariance(r))``
153:    def covariance(self, range_m: float) -> NDArray[np.float64]:
154:        return self._calibration.inflate(self._base.covariance(range_m))
209:    def _innovation_covariance(self, range_m: float) -> NDArray[np.float64]:
229:            + self._inflated_noise.covariance(range_m)
342:                innovation_covariance = self._innovation_covariance(
```

- **135**: docstring prose, not code.
- **153**: `_LambdaInflatedNoise.covariance()` *definition* — the duck-typed
  accessor `PedestrianEKF.measurement_covariance()` calls on whatever object
  is installed as `self.measurement_noise`. Required interface, not an S
  construction.
- **154**: inside that adapter, `self._calibration.inflate(self._base.covariance(range_m))`
  — the λR(range_m) *term* (base R, then inflated). This is one operand of
  the eventual S sum, not the sum itself.
- **209**: `_innovation_covariance()` *definition* — the single I1 provider.
- **229**: `jacobian @ predicted_covariance @ jacobian.T + self._inflated_noise.covariance(range_m)`
  (line 227–229 together) — **this is the one and only expression in the
  module that constructs `S = H P⁻ Hᵀ + R`.** Nowhere else in
  `robust_updater.py` adds a jacobian-quadratic-form term to an R term.
- **342**: `self._innovation_covariance(measurement.range_m)` — the explicit
  step-7 gate call. This is a *call* to the one provider at line 209, not a
  second independent place that builds S; `association.associate()` (Task 4,
  a different file) also calls this same bound method once per candidate via
  its `innovation_covariance_for` callback, and `PedestrianEKF.correct()`
  (frozen, a different file) independently arrives at the identical value
  using the same installed adapter — verified numerically, not by grep, in
  `test_association_gate_and_correction_share_one_innovation_covariance`.

`PedestrianEKF.covariance` is a `@property`; every read of `self.ekf.covariance`
in this module is attribute access with no trailing `(`, so none of those
reads appear in the grep output.

## The two critical tests

- **`test_switching_every_robust_component_off_reproduces_the_f9b_path`**:
  PASSED. Runs 8 frames (mixed detections/misses, mixed ego motion) through
  a `RobustPedestrianBeliefUpdater` with all five switches `False`
  (`covariance_calibration=False` resolved to `CovarianceCalibration(1.0,
  1.0, 0.0, 0.0)` at config-construction time, per the brief's convention
  that the covariance switch is resolved once rather than branched on at
  runtime) and through a plain `PedestrianBeliefUpdater` fed the same raw
  measurements pre-corrected by the identical `FrozenBiasCorrection(
  "global_additive", -0.045904804710162034, 0.00414567890700929, ...)`
  constants outside the call. All nine `PedestrianBelief` fields
  (`existence_probability`, `range_mean_m`, `range_std_m`, `bearing_mean_rad`,
  `bearing_std_rad`, `radial_velocity_mean_mps`, `radial_velocity_std_mps`,
  `bearing_rate_mean_rad_s`, `bearing_rate_std_rad_s`) are asserted equal
  with `pytest.approx(..., abs=1e-12)`. I verified this is not a coincidence
  of the specific frames chosen by reasoning through *why* it must hold
  exactly: (1) both EKFs use the identical `PolarMeasurementNoiseModel`
  loaded from `configs/f9c_robust_belief_v1.toml`, whose `[range_noise.*]`
  and `[bearing]` sections all carry zero residual bias, so `PedestrianEKF`'s
  own internal bias term is exactly `[0, 0]` on both sides regardless of
  argument-evaluation order — the earlier concern about "predicted-range vs.
  measured-range bias lookup" ordering turned out to be moot because the
  looked-up value is always zero either way; (2) with
  `covariance_calibration=False` resolved to scale `1.0`/floor `0.0`, the
  adapter's `.covariance(r)` is `inflate` multiplying by exactly `1.0`
  (bit-preserving) and the posterior floor is `sqrt(std**2 + 0**2)`, which
  differs from `std` only by <1e-16 relative floating-point rounding, safely
  inside the `1e-12` tolerance; (3) with `temporal_association=False`,
  association is forced into its `predicted_measurement=None` branch, which
  is `MeasurementAssociator`'s own unconditional "pick highest confidence,
  no NIS test" path — identical to always correcting on any detected frame,
  exactly like `PedestrianBeliefUpdater.update()`'s unconditional
  `ekf.step()`; (4) with `innovation_gate=False`, nothing is ever rejected;
  (5) with `bias_refit=False`, the frozen F9b constants are applied
  externally to the robust path's candidates, identically to what the test
  applies externally to the baseline's raw measurement before feeding it in.

- **`test_a_rejected_localization_is_not_an_existence_miss`**: PASSED.
  Establishes a track at (0.85 m, 0 rad), then feeds 6 frames whose single
  candidate is offset by +0.30 m (a NIS on the order of hundreds against
  ~3–16 mm range sigmas, reliably gate-rejected). Asserted, every frame:
  `detector_detected is True`, `kinematic_measurement_accepted is False`,
  `track_active is True`. Existence stayed above `active_threshold=0.50`
  throughout (it in fact stays near 1.0, since `detected=True` always drives
  the Bayesian update toward presence regardless of whether the EKF accepted
  the coordinates). Contrast run: identical track, then 6 frames with **no**
  candidate at all — existence collapses to well below 0.50 after the very
  first miss (the calibration-derived `P_D≈0.99`/`P_FA≈0.0008` values in the
  test's `EffectiveDetectionModel` make a genuine miss extremely informative).
  Final asserted values differ (`rejected_existence[-1] != silent_existence[-1]`).

## Files changed

- `src/duckie_pomdp/belief/robust_updater.py` — new.
- `tests/test_f9c_robust_updater.py` — new.
- `src/duckie_pomdp/belief/__init__.py` — modified (exports only; no
  behavioral change to any existing symbol).

No other file was touched. `pedestrian_ekf.py`, `existence_filter.py`,
`f9_pipeline.py`, and every other already-completed F9c module are untouched,
confirmed by `git`-free diff review (this is not a git repo, so this was
confirmed by re-reading each file's content against what I read at the start
of the task and by the fact only the three files above show a `Write`/`Edit`
in this session).

## Self-review

- **Order of operations**: implemented in the exact 12-step order from the
  brief, with each step numbered in a comment in `update()`.
- **I1**: single provider (`_innovation_covariance`), single R-injection
  point (`_LambdaInflatedNoise` installed once in `__init__`), verified both
  by code inspection (`grep`, above) and by a runtime test that monkeypatches
  the provider and compares matrices to 1e-15, plus a direct comparison
  against the frozen EKF's own `EKFStepDiagnostics.innovation_covariance`
  (exposed via a new `self.last_ekf_diagnostics` instance attribute, mirroring
  the existing `PedestrianBeliefUpdater.last_diagnostics` pattern — not part
  of `RobustStepRecord`, since the brief didn't list it there, but useful for
  debugging/testing exactly as `last_diagnostics` is on the F9b updater).
- **I2**: `detector_detected` is computed from the raw `candidates` sequence
  at step 4, strictly before bias correction (step 4b) or association (steps
  5–8); `existence.update()`'s `detected` argument is always
  `detector_detected`, never `kinematic_measurement_accepted`.
- **Bias switch**: never calls `FrozenBiasCorrection.identity()`; the
  "off" state resolves to the F9b frozen constants
  (`-0.045904804710162034`, `0.00414567890700929`) via a caller-supplied
  `bias_frozen` argument, matching the brief's explicit values.
- **Posterior floor**: applied only inside the `if self.ekf.initialized:`
  reporting branch, via `self.config.covariance.floor_polar_standard_deviation(...)`,
  never assigned back into `self.ekf._covariance` (which the updater has no
  write access to at all beyond calling `predict`/`correct`/`reset` — there
  is structurally no code path that could do this). Commented in place as
  required.
- **Track lifecycle**: implements the exact six-row state table from the
  brief, including the deliberate "no track + accepted candidate but
  existence posterior didn't clear the initialization threshold" edge case
  (`self.ekf.reset()` discards the just-created kinematic state). This
  branch is reachable in principle but not exercised by any of the 11
  required tests, since with this project's fitted `P_D`/`P_FA` values a
  single detection from a cold existence prior almost always clears
  `initialization_threshold=0.50` in one step (verified by hand-computation
  during design; see reasoning notes below).
- **`RobustObservationSwitches` duplication**: I declared a fresh dataclass
  in `belief/robust_updater.py` rather than importing
  `evaluation.f9c_protocol.RobustObservationSwitches`, to avoid a
  `belief -> evaluation` dependency (no existing file in `evaluation/`
  imports from `belief/`, and I did not want this task to be the first to
  invert that direction). This is a deliberate design choice, flagged below
  as the one thing worth a second opinion.

## Concerns

1. **`RobustObservationSwitches` is now declared twice** (here and in
   `evaluation/f9c_protocol.py`), with identical fields. I judged this
   preferable to a layering inversion, but Task 9 (which will construct a
   `RobustPedestrianBeliefUpdater` from a loaded `F9cProtocol`) will need a
   short adapter line (`RobustObservationSwitches(**dataclasses.asdict(protocol.robust))`)
   to bridge them. If the plan intends a single shared type instead, this is
   the place to raise it before Task 9 starts.
2. **`RobustObservationConfig.effective_detection`, `.gate`, `.association`,
   and `.covariance` are not loaded from `configs/f9c_robust_belief_v1.toml`
   by this task** — the brief's interface list only specifies the
   `RobustObservationConfig` *shape*, not a loader function, and no test
   requires one, so I did not add one. Task 9/10/11 will need to build a
   `RobustObservationConfig` from the frozen TOML; I left that wiring for
   whichever task actually needs it, to avoid speculative untested code.
3. The `f9c_robust_belief_v1.toml` config's `[conditional_detection]` and
   `[covariance_calibration]` sections are still placeholders
   (`parameters_frozen = false`, all `ObservabilityClass` probabilities
   equal, floors at `0.0`) pending Task 9's calibration run. Tests 6/7/8 in
   this task therefore construct their own differentiated
   `EffectiveDetectionModel` (`CENTER=0.99, MID_FOV=0.95, EDGE_FOV=0.60,
   OUTSIDE_DOMAIN=0.05`) rather than loading the config, since the config's
   current uncalibrated values (all classes equal) would make the
   EDGE_FOV-survives / CENTER-dies contrast in those tests physically
   meaningless. This is expected and matches how Task 4's association tests
   also hand-construct configs rather than loading the not-yet-frozen file.

## Report file

`.superpowers/sdd/2026-08-08-f9c-robust-observation-belief-calibration/task-8-report.md`

---

# Fix round 1/5: Invariant I7 (association gate must be looser than the innovation gate)

## Root cause (plan defect, not code defect)

`[association].chi_square_gate` and `[innovation_gate].chi_square_threshold` were both set to
`9.21034037197618` in `configs/f9c_robust_belief_v1.toml`. `MeasurementAssociator.associate()`
discards every candidate whose NIS exceeds `chi_square_gate` *before* the coordinator's explicit
gate step ever runs `InnovationGate.evaluate()`. With the two thresholds equal, no candidate that
reaches the gate could ever fail it, so `innovation_gate=False` vs. `innovation_gate=True` produced
identical behavior whenever `temporal_association=True` — the switch was dead. This was a defect
in the plan/config values, not in `robust_updater.py`'s wiring, which already called
`InnovationGate.evaluate()` correctly; it simply never had a chance to reject anything.

## What changed

1. **`configs/f9c_robust_belief_v1.toml`** — `[association].chi_square_gate` raised from
   `9.21034037197618` (2 DOF, 99%) to `13.815510557964274` (2 DOF, 99.9%), with the
   coordinator-provided explanatory comment placed above it verbatim. `[innovation_gate]
   .chi_square_threshold` is unchanged at `9.21034037197618`.

2. **`src/duckie_pomdp/belief/robust_updater.py`** — added an Invariant-I7 guard at the top of
   `RobustPedestrianBeliefUpdater.__init__`, immediately after storing `self.config`:

   ```python
   if config.association.chi_square_gate <= config.gate.chi_square_threshold:
       raise ValueError(
           "invariant I7 violated: association.chi_square_gate "
           f"({config.association.chi_square_gate}) must be strictly "
           "greater than gate.chi_square_threshold "
           f"({config.gate.chi_square_threshold}); an association gate "
           "at or below the innovation-gate threshold makes the "
           "innovation gate unreachable -- association would already "
           "have discarded every candidate the gate could reject."
       )
   ```

   States both numeric values and explains the mechanism ("unreachable" appears in the message,
   matched by the new test below). This check runs unconditionally at construction — it does not
   look at `switches.innovation_gate`/`switches.temporal_association`, because a config that
   *could* produce this failure mode should never be constructible, whether or not a given run
   happens to have the switches positioned to expose it.

   No change to `_innovation_covariance` (I1), the existence wiring (I2), or the bias-switch
   logic — all three were confirmed untouched by re-reading the diff before running tests.

3. **`tests/test_f9c_robust_updater.py`**:
   - Added module constant `ASSOCIATION_GATE_THRESHOLD = 13.815510557964274` alongside the
     existing `GATE_THRESHOLD = 9.21034037197618` (now comment-labeled "innovation gate" vs.
     "association").
   - `robust_config()`'s default `association=` now builds
     `AssociationConfig(chi_square_gate=ASSOCIATION_GATE_THRESHOLD, ...)` instead of reusing
     `GATE_THRESHOLD`, so every existing test that doesn't override `association=` now runs
     against the loosened default, matching the corrected config. I re-verified none of the 11
     original tests depended on the two thresholds being equal (all outlier scenarios use gross
     ~0.25–0.30 m offsets whose NIS is in the hundreds, comfortably rejected by either threshold).
   - Added `test_association_gate_is_looser_than_the_innovation_gate` — constructs the
     coordinator with `gate=InnovationGateConfig(GATE_THRESHOLD)` and
     `association=AssociationConfig(chi_square_gate=GATE_THRESHOLD, ...)` (equal thresholds) and
     asserts `pytest.raises(ValueError, match="unreachable")`.
   - Added `test_a_candidate_between_the_two_thresholds_is_associated_then_gated` — establishes
     a track (init + one settle frame at range 0.85 m, bearing 0), then feeds a candidate at range
     0.935 m (a +0.085 m offset). Against the settled track's actual `S` this produces
     NIS ≈ 11.48, confirmed empirically (see below) to sit strictly between 9.21034037197618 and
     13.815510557964274. Asserts `association.mode == "temporal"`, `association.selected is not
     None`, `record.gate is not None`, `record.gate.accepted is False`,
     `record.gate.threshold == pytest.approx(GATE_THRESHOLD)`,
     `kinematic_measurement_accepted is False`, and `detector_detected is True`.

## Why the between-thresholds test genuinely exercises `InnovationGate.evaluate()`

`record.gate` is only ever populated in `update()`'s `association.mode != "initialization"`
("temporal") branch, and only by one of two code paths: `self.gate.evaluate(innovation,
innovation_covariance)` when `switches.innovation_gate` is `True` (the default the test uses), or
an unconditional `GateDecision(accepted=True, ...)` when the switch is `False`. The test asserts
`record.gate.accepted is False` — the only way to reach a non-`None`, *rejecting* `GateDecision`
is through the real `self.gate.evaluate()` call, since the switch-off branch always constructs
`accepted=True`. Separately, `association.mode == "temporal"` and `association.selected is not
None` (asserted first) prove the candidate was *not* filtered out by `MeasurementAssociator`'s own
internal `chi_square_gate` — if it had been, `association.selected` would be `None` and mode would
be `"all_gated_out"`, and the coordinator's code would never reach the gate-evaluation branch at
all (`record.gate` would stay `None`). The combination of "association selected it" + "gate
rejected it with a populated `GateDecision`" is only reachable through `InnovationGate.evaluate()`
actually running and actually returning `accepted=False`.

## Finding the between-thresholds NIS empirically

Hand-deriving NIS from H, P, and R analytically was avoidable and error-prone (an early probe that
didn't advance P through a third `predict()` call was off by roughly 4 NIS units); I instead
searched empirically against the exact fixture sequence the test uses:

```
$ PYTHONPATH=src:.../duckie/src DUCKIETOWN_HEADLESS=1 python - <<'PYEOF'
import sys; sys.path.insert(0, "tests")
from test_f9c_robust_updater import robust_updater, candidate, start_belief, STATIONARY, NO_ACTION, DT
for delta in [0.08, 0.085, 0.09, 0.095]:
    updater = robust_updater()
    belief = start_belief()
    belief, record = updater.update(belief, NO_ACTION, STATIONARY, [candidate(0.85, 0.0)], DT)
    belief, record = updater.update(belief, NO_ACTION, STATIONARY, [candidate(0.85, 0.0)], DT)
    belief, record = updater.update(belief, NO_ACTION, STATIONARY, [candidate(0.85+delta, 0.0)], DT)
    print(delta, record.association.mode, record.kinematic_measurement_accepted, record.nis)
PYEOF
0.08 temporal False 10.169393071606907
0.085 temporal False 11.480291397243736
0.09 temporal False 12.870638106252494
0.095 all_gated_out False None
```

`delta=0.085` (range 0.935 m) gives NIS ≈ 11.48, roughly centered between 9.21 and 13.82 with
~2.3 units of margin on each side — the value used in the committed test.

## Test run: `tests/test_f9c_robust_updater.py`

```
$ PYTHONPATH=src:/home/pannntastic/aivnv/duckie/src DUCKIETOWN_HEADLESS=1 \
  /home/pannntastic/aivnv/duckie/.venv/bin/python -m pytest tests/test_f9c_robust_updater.py -v
============================= test session starts ==============================
collecting ... collected 13 items

tests/test_f9c_robust_updater.py::test_a_gross_outlier_does_not_move_the_belief_more_than_the_baseline_does PASSED
tests/test_f9c_robust_updater.py::test_reported_range_std_is_never_below_the_posterior_floor PASSED
tests/test_f9c_robust_updater.py::test_a_duplicate_frame_selects_the_temporally_consistent_candidate PASSED
tests/test_f9c_robust_updater.py::test_a_rejected_localization_is_not_an_existence_miss PASSED
tests/test_f9c_robust_updater.py::test_association_gate_and_correction_share_one_innovation_covariance PASSED
tests/test_f9c_robust_updater.py::test_association_gate_is_looser_than_the_innovation_gate PASSED
tests/test_f9c_robust_updater.py::test_a_candidate_between_the_two_thresholds_is_associated_then_gated PASSED
tests/test_f9c_robust_updater.py::test_belief_survives_five_consecutive_misses_when_predicted_edge_fov PASSED
tests/test_f9c_robust_updater.py::test_an_outside_domain_absence_preserves_belief_far_longer_than_an_in_domain_one PASSED
tests/test_f9c_robust_updater.py::test_belief_still_dies_after_a_long_absence PASSED
tests/test_f9c_robust_updater.py::test_a_deleted_track_reinitializes_from_the_next_valid_candidate PASSED
tests/test_f9c_robust_updater.py::test_switching_every_robust_component_off_reproduces_the_f9b_path PASSED
tests/test_f9c_robust_updater.py::test_updater_never_receives_privileged_state PASSED

============================== 13 passed in 0.29s ==============================
```

(11 original + 2 new I7 tests = 13.)

## Full suite

```
$ PYTHONPATH=src:/home/pannntastic/aivnv/duckie/src DUCKIETOWN_HEADLESS=1 \
  /home/pannntastic/aivnv/duckie/.venv/bin/python -m pytest tests -q
186 passed, 260 warnings in 11.78s
```

186, matching the coordinator's expectation exactly (184 + 2 new tests).

## Files changed in this fix round

- `configs/f9c_robust_belief_v1.toml` — `[association].chi_square_gate` 9.21034037197618 →
  13.815510557964274, with explanatory comment.
- `src/duckie_pomdp/belief/robust_updater.py` — added the Invariant-I7 construction-time guard.
  No other change; `_innovation_covariance`, the existence wiring, and the bias-switch logic are
  byte-for-byte the same as before this fix round.
- `tests/test_f9c_robust_updater.py` — new `ASSOCIATION_GATE_THRESHOLD` constant,
  `robust_config()`'s default association gate now uses it, and two new tests.

## Self-review

- Verified via `grep` that no other file in the repo asserts the old
  `association.chi_square_gate == 9.21034037197618` value from this config (only
  `tests/test_f9c_association.py` and `tests/test_f9c_innovation_gate.py` hardcode
  `9.21034037197618`, and both use it as a self-contained fixture constant for testing those
  modules in isolation, not by loading `f9c_robust_belief_v1.toml` — unaffected by this change).
- Confirmed the guard is unconditional (not gated on any of the five ablation switches), so an
  invalid `RobustObservationConfig` cannot be constructed regardless of which switches a caller
  intends to flip later — this matches "fail loudly at construction rather than silently
  producing a meaningless ablation."
- Re-ran the full suite twice (once immediately after the config/code change, once after the test
  file edits) to confirm no other test implicitly depended on the two thresholds being equal.

## Concerns

None new. The three items listed in the original report (duplicated `RobustObservationSwitches`
declaration, no TOML loader for `RobustObservationConfig` yet, placeholder conditional-detection
values) still stand and are unaffected by this fix.
