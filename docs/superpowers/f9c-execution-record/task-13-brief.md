## Task 13: Leakage tests, regression guard, and gate report

**Files:**
- Create: `tests/test_f9c_leakage.py`
- Modify: `GATES.md`, `README.md`, `IMPLEMENTATION_NOTES.md`, `.aris/compute/local.md`

- [ ] **Step 1: Write the leakage and regression tests**

```python
# tests/test_f9c_leakage.py
import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNTIME_MODULES = [
    "src/duckie_pomdp/belief/innovation_gate.py",
    "src/duckie_pomdp/belief/measurement_association.py",
    "src/duckie_pomdp/belief/covariance_calibration.py",
    "src/duckie_pomdp/belief/observability.py",
    "src/duckie_pomdp/belief/robust_updater.py",
]
FORBIDDEN = (
    "privileged", "PrivilegedState", "true_pomdp_state",
    "sample_object_silhouettes", "eligible_visible",
    "gt_range_m", "gt_bearing_rad", "selected_iou", "intersection_over_union",
)


def test_no_runtime_module_references_privileged_state():
    for relative in RUNTIME_MODULES:
        source = (ROOT / relative).read_text(encoding="utf-8")
        for token in FORBIDDEN:
            assert token not in source, f"{relative} references {token}"


def test_no_runtime_module_imports_the_evaluation_package():
    for relative in RUNTIME_MODULES:
        tree = ast.parse((ROOT / relative).read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                names = [getattr(node, "module", None) or ""] + [
                    alias.name for alias in node.names
                ]
                assert not any("evaluation" in str(name) for name in names), relative


def test_the_evaluator_steps_both_beliefs_before_reading_privileged_truth():
    source = (ROOT / "experiments" / "evaluate_f9c_robust_belief.py").read_text(
        encoding="utf-8"
    )
    privileged_at = source.index("integration.privileged.read()")
    baseline_at = source.index("baseline_updater.update(")
    robust_at = source.index("robust_updater.update(")
    assert baseline_at < privileged_at
    assert robust_at < privileged_at


def test_f9b_frozen_artifacts_are_untouched():
    from duckie_pomdp.evaluation.f9_protocol import sha256

    assert sha256(ROOT / "artifacts" / "f9_measurement_model.json") == (
        "eb09ea6c64b6cbf3306057092e254a0e049776b38581e5b873a8ef9e2e91b278"
    )


def test_f9c_source_never_names_a_frozen_test_seed():
    for path in (ROOT / "src").rglob("*.py"):
        source = path.read_text(encoding="utf-8")
        for seed in ("5101", "5102", "5103", "5104"):
            assert seed not in source, f"{path} hardcodes frozen test seed {seed}"
```

- [ ] **Step 2: Run to verify they fail or pass honestly**

Run: `$PY -m pytest tests/test_f9c_leakage.py -q`
Expected: any failure here is a real leak — fix the source module, never the test.

- [ ] **Step 3: Run the complete suite**

Run: `$PY -m pytest tests -q`
Expected: PASS, ≈ 197 tests, 0 failed, 0 skipped. Record the exact count. (Per-task counts in this plan are expectations, not contracts — if your fixture layout yields a different total, reconcile it once here and move on. Zero failures and zero skips is the contract.)

- [ ] **Step 4: Write the gate report**

Add the F9c row to `GATES.md` and a `## F9c` section to `README.md` with the reproduction commands. Add an `### env:` witness block to `.aris/compute/local.md` following the existing format (`how` / `tier` / `validated` / `gotcha`).

The report must contain, in this order:

```text
calibration seeds / final seeds / frozen config SHA256
gate type + threshold
association rule
covariance inflation parameters (λ_r, λ_β) and posterior floors (σ_floor,r, σ_floor,β)
variance components τ̂ and σ̂_w that justify the floors
bias model chosen + LOSO evidence for the choice
effective detection probabilities per observability class
support counts near/medium/far/edge_fov vs the pre-specified minima

Baseline A vs Robust B:
  range bias / MAE / RMSE
  bearing bias / MAE / RMSE
  range-rate RMSE / bearing-rate RMSE
  coverage_68 / coverage_95 / coverage_error_68 / coverage_error_95 / NLL
  mean_predicted_std and std_over_rmse

natural misses maintained / duplicate handling / outlier handling
miss breakdown: detector_miss_in_domain / detector_miss_outside_domain / gated_rejection
fraction of gated_rejection frames that retained an active belief   (invariant I2 payoff)
false tracks / track deletions / recovery time
NIS diagnostics for accepted measurements AND for rejected ones, separately
ablation table
runtime-cache SHA256 shared by the headline run and the ablation    (invariant I4)
predicted-observability vs GT FOV-region confusion
full test count
```

Then answer, explicitly:

1. Did robust observation handling reduce localization-outlier impact?
2. Did temporal association improve duplicate frames?
3. Did range uncertainty become realistically calibrated?
4. Did the conditional detection model improve belief through natural misses?
5. Did separating detection evidence from kinematic acceptance prevent the gate from worsening existence collapse? Quantify against the counterfactual: how many `gated_rejection` frames were there, and what would `P(e)` have done had each been scored as a miss?
6. Was RMSE materially worsened to achieve calibration?
7. Is EKF + robust observation handling sufficient for Version-1 POMDP?
8. Is the system control-ready?

- [ ] **Step 5: Classify and STOP**

Classify as `CONTROL_READY`, `LIMITED`, or `FAILED` against the pre-specified acceptance bands and the minimum-support requirement. `CONTROL_READY` is unavailable if near-range support < 100.

**Then STOP.** Do not begin stop logic, reward, or SAC. Report and wait.

---

## Self-Review

**Spec coverage.** Every locked decision maps to a task: `P_S` untouched → Task 7 (`test_ps_is_untouched_by_f9c`) and the freeze table; existence unfrozen but only observation parameters → Task 1 validator (which checks `prior`/`survival`/`birth` but deliberately not `detection_probability`); bias refit on new calibration seeds before covariance → Task 9 ordering, with the per-bin decision rule pre-specified; 5101–5104 frozen → Task 1 `forbidden_seeds` plus Task 13 source scan; near-range required for `CONTROL_READY` → Task 2, Task 11 Step 4, Task 13 Step 5; `P_D → P_D^eff(predicted observability)` → Task 7; acceptance bands rather than exact 0.68/0.95 → Task 1 `[acceptance]`; candidates in perception, association in the belief layer → Tasks 5 and 4; four experiments A/B/C/D → Tasks 9, 7, 3+4, 6; ablation → Task 12; artifacts and required tests → Tasks 9–13.

**Additions beyond the operator's brief, flagged deliberately:** (i) the posterior variance floor and its random-effects justification, because `R` inflation provably cannot fix steady-state coverage under a frozen tiny `Q`; (ii) eight calibration seeds instead of four, to halve `SE(b̂)`; (iii) the LOSO rule for the per-bin bias decision; (iv) the anti-inflation guard `mean_predicted_std ≤ 1.5 × RMSE`, which operationalizes "not achieved by absurdly inflating uncertainty" as a number fixed before the run.

**Review round 4 — four corrections applied, plan then marked ready.** (1) *Cache holds raw candidates* — invariant I5, every cached field renamed `raw_*`, cache written at the output of `observe()` before Task 8 step 4b, and `test_runtime_cache_contains_pre_bias_raw_candidates` which asserts the two replay paths see ranges differing by exactly `b_F9c − b_F9b` (zero would prove the cache was written post-correction). (2) *`λ` fit de-circularized* — invariant I6; the fitting set is now selected by `eligible_visible AND valid projection AND correct class AND selected_correct_iou50`, an external GT criterion, rather than by gate acceptance which depends on the `λ` being fitted; excluded-sample count and NIS distribution are reported so the blind spot is visible. (3) *Existence retention un-pooled* — reported separately for `detector_miss_in_domain`, `detector_miss_outside_domain`, and `gated_rejection`, with the in-domain figure alone as the control-readiness criterion, since invariant I3 makes outside-domain retention nearly free and a pooled number could read 80% while the belief collapsed on every genuine detector miss; under-powered if in-domain misses < 20. (4) *Task 9 calls the nested estimator explicitly*, with the one-level function labelled as an internal primitive. The estimator is also now labelled an approximate nested variance-component estimator rather than a REML mixed-effects fit.

**Review round 3 — three corrections applied.** (1) *The bias refit must actually reach the runtime* — new Task 3b adds `FrozenBiasCorrection` as a named runtime stage locked at position 4b in Task 8, before association rather than after the gate, since association thresholds candidates against `h(x̂⁻)` and an uncorrected candidate would inject the full bias into every NIS. The switch selects between the F9b frozen constants and the F9c fitted ones — never `identity()` — and the ablation table now states the bias column for every row, so "all switches off == Baseline A" holds by construction; guarded by `test_bias_ablation_uses_f9b_bias_when_switch_off` and `test_bias_refit_switch_applies_f9c_frozen_bias_before_association`. (2) *`DOWNWEIGHT` removed* — it would have made the correction use `25λR` while association and the gate used `λR`, so `S_gate ≠ S_correction` and invariant I1 would fail; F9c v1 is hard-reject only, and `test_the_gate_exposes_no_covariance_scaling_knob` asserts the knob does not exist rather than merely going unused. (3) *Random-effects statistics corrected* — the `0.01562 rad` bearing figure was the SD of the episode means, not `τ̂`; recomputed with Task 6's own estimator the components are `τ̂_r = 0.01425 / σ̂_w,r = 0.00739` and `τ̂_β = 0.01203 / σ̂_w,β = 0.00455`. Verifying that also surfaced a structural fact that changes the floor: range offset is carried at the *seed* level and bearing offset at the *episode* level, so `SE(b̂)` must divide by the seed count for range. Task 6 now specifies a two-level nested fit and the projections are stated as bands, since F9a's four seeds cannot pin the split.

**Review round 2 — four corrections applied.** (1) *Detection evidence ≠ measurement acceptance* — Finding 6, invariant I2, Task 8 step 10 and its state-transition table, `detector_detected` / `kinematic_measurement_accepted` split through the record, the calibration CSV, the gate log, and the miss breakdown; guarded by `test_a_rejected_localization_is_not_an_existence_miss`. The earlier draft would have turned every gated bbox into an existence miss, making the gate a net negative — this was a real defect, not a wording issue. (2) *One `S` everywhere* — invariant I1, the single `_innovation_covariance` provider in Task 8, the injected `innovation_covariance_for` contract in Task 4, the `grep` check, and `test_association_gate_and_correction_share_one_innovation_covariance`; Task 9 additionally fits `λ` against that same provider. (3) *Outside-domain misses apply no likelihood* — Finding 7, invariant I3, `miss_is_informative` and `observation_informative`, the `outside_domain_miss_policy` config key, and the reworded in-domain decay criterion; `P_D^eff(OUTSIDE_DOMAIN)` survives as a reported diagnostic only, so it cannot become a tuning knob. (4) *Ablation replays an on-disk cache* — invariant I4, `f9c_runtime_cache.py`, hash-verified load, `--ablation` as a mutually exclusive mode with the detector and environment constructed only in the final-run branch, and `test_ablation_performs_no_inference_and_no_render`.

**Placeholder scan.** Task 8 Step 1 and Task 9 Step 1 carry docstring-only test stubs. That is deliberate and bounded: both instruct the implementer to reuse the fixtures already present in `tests/test_pedestrian_ekf.py`, which must be read first; inventing fixture code here that conflicts with the existing ones would be worse than pointing at them. Every assertion those tests must make is stated. No other step defers content.

**Type consistency.** `GateDecision.accepted` (Task 3) is the sole gate output consumed by `RobustPedestrianBeliefUpdater` (Task 8); the gate carries no covariance-scaling field, by invariant I1. `FrozenBiasCorrection.correct` (Task 3b) is applied to every candidate before `CandidateMeasurement` construction in Task 8, and its parameters come from `[measurement_model]` (Task 1) as fitted in Task 9. `AssociationResult.differed_from_highest_confidence` (Task 4) is read by Task 8's duplicate test and Task 11's `robustness_metrics`. `ObservabilityClass` (Task 7) keys `EffectiveDetectionModel`, drives `miss_is_informative`, and appears as `predicted_observability_class` in both the calibration CSV (Task 9) and the validation CSV (Task 11). `RobustObservationSwitches` field names (Task 1) are the ablation keys (Task 12). `CandidateMeasurement` (Task 4) is produced from `CandidateProjection` (Task 5) by the coordinator in Task 8, and is what `RuntimeCacheFrame` (Task 11) serializes for replay in Task 12. `RobustStepRecord.detector_detected` / `.kinematic_measurement_accepted` (Task 8) are the same two column names used by the calibration CSV (Task 9), the validation CSV and gate log (Task 11), and the miss breakdown in `robustness_metrics` (Task 11). `ExistenceFilter.update`'s `observation_informative` keyword (Task 7) is supplied from `EffectiveDetectionModel.miss_is_informative(...) or detector_detected` in Task 8 step 10.
