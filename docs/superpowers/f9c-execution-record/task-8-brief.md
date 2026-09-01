## Task 8: Robust belief coordinator

**Files:**
- Create: `src/duckie_pomdp/belief/robust_updater.py`
- Modify: `src/duckie_pomdp/belief/__init__.py`
- Test: `tests/test_f9c_robust_updater.py`

**Interfaces:**
- Consumes: everything from Tasks 3, 4, 6, 7, plus the frozen `PedestrianEKF`, `ExistenceFilter`, `PedestrianBelief`.
- Produces:
  - `RobustObservationConfig(switches, gate, association, covariance, effective_detection, active_threshold, delete_threshold, initialization_threshold)`.
  - `RobustStepRecord(frame_mode: str, detector_detected: bool, kinematic_measurement_accepted: bool, association: AssociationResult, gate: GateDecision | None, effective_detection_probability: float, observation_informative: bool, observability_class: ObservabilityClass, existence_probability: float, track_active: bool, track_deleted: bool, nis: float | None, reported_range_std_m: float, reported_bearing_std_rad: float)`.
  - `RobustPedestrianBeliefUpdater.update(previous_belief, previous_action, ego_motion, candidates, dt_s) -> tuple[BeliefState, RobustStepRecord]`.
  - Private `_innovation_covariance(range_m) -> NDArray` — the **single** provider satisfying invariant I1, returning `H P⁻ Hᵀ + λR(range_m)` from the current predicted covariance.

Order of operations, fixed:

```text
 1. EKF predict with actual ego motion                          (frozen)
 2. classify predicted observability from x̂⁻                    (Task 7)
 3. resolve P_D^eff and miss_is_informative for this frame      (Task 7)
 4. detector_detected = any raw candidate exists in this frame  <- existence evidence
4b. apply the FROZEN F9c BIAS CORRECTION to every candidate     (Task 3b)
       z_raw -> z_corr, before anything reads a candidate range
 5. build ONE innovation-covariance provider from λR and P⁻     (I1)
 6. associate corrected candidates against h(x̂⁻) using it       (Task 4)
 7. gate the associated candidate on NIS using that provider    (Task 3)
 8. kinematic_measurement_accepted = association selected AND gate accepted
 9. if accepted -> EKF correct using that same provider's R     (Tasks 3+6)
    else        -> prediction only, filter state already advanced by step 1
10. existence update:
       observation_informative = miss_is_informative(observability) OR detector_detected
       existence.update(detected=detector_detected,
                        detection_probability=P_D^eff,
                        observation_informative=observation_informative)
11. delete the track if P(e) < delete_threshold
12. report belief with the posterior floor applied              (Task 6)
```

**Step 10 is the fix for invariant I2 and is the single most consequential line in this plan.** Existence is driven by `detector_detected` — the detector's answer to "is a Duckie in this image" — *not* by `kinematic_measurement_accepted`. A frame where YOLO finds the pedestrian but the gate rejects the bbox is scored as a **detection** for existence and as **prediction-only** for the EKF. That is the estimator saying "I believe it is there, I do not believe these coordinates."

The `OR detector_detected` in the `observation_informative` expression implements the asymmetry of invariant I3: a miss predicted outside the domain applies no likelihood, but a *detection* always does, whatever the predicted class was.

**Step 4b placement matters.** `detector_detected` is read from the **raw** candidate list at step 4, because whether the detector saw a Duckie cannot depend on a metric correction. Everything downstream — association, gating, correction — consumes only corrected candidates, so a candidate range never reaches an innovation computation uncorrected.

The updater holds exactly one `self._bias: FrozenBiasCorrection`, resolved from the switch:

```text
bias_refit = false  ->  F9c bias stage loaded with the F9b FROZEN constants
                        b_r = -0.045904804710162034, b_beta = +0.00414567890700929
bias_refit = true   ->  F9c bias stage loaded with the F9c FITTED constants
```

Not `identity()` when the switch is off — Baseline A applies the F9b correction, so an identity fallback would make `all switches off` differ from Baseline A by the entire F9b bias and would silently invalidate `test_switching_every_robust_component_off_reproduces_the_f9b_path`. `identity()` exists for unit tests only and must not appear in any ablation configuration.

**Initialization is the deliberate exception.** With no active track there is no innovation to test, so the two claims cannot be separated. A track is created only when `kinematic_measurement_accepted` is true — a candidate that fails projection or falls outside the association gate must not create a track. State transition, to be documented verbatim in `IMPLEMENTATION_NOTES.md`:

```text
no track  + accepted candidate + P(e) >= initialization_threshold  -> track created (frame_mode="initialization")
no track  + detection only                                          -> existence updates, no track
active    + accepted candidate                                      -> correct        (frame_mode="temporal")
active    + detection, gate rejected                                -> predict only, existence counts a DETECTION
active    + no detection, in-domain                                 -> predict only, existence counts a MISS
active    + no detection, outside domain                            -> predict only, existence prediction step only
active    + P(e) < delete_threshold                                 -> track deleted (frame_mode="deleted")
```

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_f9c_robust_updater.py
def test_a_gross_outlier_does_not_move_the_belief_more_than_the_baseline_does(...):
    """Feed 10 consistent measurements then one 0.30 m outlier.
    Robust belief range error after the outlier must be < 0.5 x baseline error."""


def test_reported_range_std_is_never_below_the_posterior_floor(...):
    """After 60 consistent updates the EKF std collapses; the reported std must
    still be >= range_posterior_floor_m."""


def test_a_duplicate_frame_selects_the_temporally_consistent_candidate(...):
    """Two candidates, the higher-confidence one 0.25 m off track.
    record.association.differed_from_highest_confidence is True and the selected
    measurement is the consistent one."""


def test_a_rejected_localization_is_not_an_existence_miss(...):
    """Invariant I2. Establish a track, then feed 6 frames whose single candidate
    is a 0.30 m gross outlier. Every frame must record
    detector_detected=True, kinematic_measurement_accepted=False, and
    track_active=True; existence must stay above the active threshold.
    Contrast case: the same 6 frames with NO candidate at all must drive
    existence below the active threshold. The two sequences must differ."""


def test_association_gate_and_correction_share_one_innovation_covariance(...):
    """Invariant I1. Monkeypatch the updater's _innovation_covariance to record
    every matrix it returns within one update() call, and assert that the
    matrix used by the associator, the matrix passed to the gate, and the
    matrix implied by the EKF correction's R are identical to 1e-15.
    Additionally assert the returned R equals calibration.inflate(base_R),
    not base_R -- a lambda fitted on inflated S must be applied to inflated S."""


def test_belief_survives_five_consecutive_misses_when_predicted_edge_fov(...):
    """record.track_active stays True through 5 misses at EDGE_FOV."""


def test_an_outside_domain_absence_preserves_belief_far_longer_than_an_in_domain_one(...):
    """Invariant I3 at the coordinator level. 30 missing frames with the belief
    predicted OUTSIDE_DOMAIN must leave P(e) > 0.80 and track_deleted False;
    the same 30 frames predicted CENTER must delete the track."""


def test_belief_still_dies_after_a_long_absence(...):
    """After 40 consecutive misses at CENTER, P(e) < delete_threshold and
    record.track_deleted becomes True exactly once."""


def test_a_deleted_track_reinitializes_from_the_next_valid_candidate(...):
    """After deletion, one good candidate produces frame_mode == 'initialization'
    and an initialized EKF."""


def test_switching_every_robust_component_off_reproduces_the_f9b_path(...):
    """With all five switches False, the belief sequence must equal
    PedestrianBeliefUpdater's output to within 1e-12 on the same inputs.
    This test is the Baseline-A regression guard."""


def test_updater_never_receives_privileged_state(...):
    import inspect
    parameters = set(
        inspect.signature(RobustPedestrianBeliefUpdater.update).parameters
    )
    assert parameters == {
        "self", "previous_belief", "previous_action", "ego_motion",
        "candidates", "dt_s",
    }
```

Fill each docstring stub with a concrete arrangement using the same synthetic-measurement helpers `tests/test_pedestrian_ekf.py` already uses — read that file and reuse its fixtures rather than inventing new ones.

- [ ] **Step 2: Run tests to verify they fail**

Run: `$PY -m pytest tests/test_f9c_robust_updater.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'duckie_pomdp.belief.robust_updater'`

- [ ] **Step 3: Implement**

Call the *existing* `PedestrianEKF.predict` / `PedestrianEKF.correct`; do not reimplement filter mathematics. For the λ-inflated correction, wrap the EKF's `measurement_noise` in a small adapter whose `covariance()` returns `calibration.inflate(base.covariance(r))` — this keeps `pedestrian_ekf.py` untouched, which is what "frozen" must mean in practice. Apply the posterior floor only in the reported `PedestrianBelief`, never to `self.ekf._covariance`; add a comment saying so, because writing the floor back into the filter state would corrupt the next prediction.

**Invariant I1 in code.** Construct the inflated noise adapter *once* in `__init__` and install it on the EKF. Then `_innovation_covariance(range_m)` reads `H` and `P⁻` from the EKF after `predict()` and returns `H @ P⁻ @ H.T + adapter.covariance(range_m)`. Hand that one bound method to the associator as `innovation_covariance_for`, use its output for the gate, and let the EKF correction consume the same adapter. There must be exactly one call site constructing `S` in the whole module:

```bash
grep -n "covariance(" src/duckie_pomdp/belief/robust_updater.py
```

If more than one expression builds `H P Hᵀ + R`, the invariant is already broken.

Two tests carry the weight of this task. `test_switching_every_robust_component_off_reproduces_the_f9b_path` — if it does not pass exactly, the ablation in Task 12 is meaningless. `test_a_rejected_localization_is_not_an_existence_miss` — if it does not pass, the gate makes existence collapse *worse* than F9b, which is the specific failure mode this whole gate exists to prevent.

- [ ] **Step 4: Run tests to verify they pass**

Run: `$PY -m pytest tests/test_f9c_robust_updater.py -q`
Expected: PASS (11 tests)

- [ ] **Step 5: Checkpoint**

Run: `$PY -m pytest tests -q` → expect 181 passed.

---

