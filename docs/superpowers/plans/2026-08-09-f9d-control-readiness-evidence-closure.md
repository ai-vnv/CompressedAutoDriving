# F9d — Control-Readiness Evidence Closure Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Close the two evidence gaps that kept F9c at `LIMITED` — gross localization-outlier robustness and long-absence existence decay — by stress-testing the **completely frozen** F9c estimator on new unseen seeds, and separately diagnose why minimum-NIS association underperformed, using only the existing F9c cache.

**Architecture:** F9d adds **no estimator capability whatsoever**. Every F9c parameter, threshold, bias, covariance scale, floor and rule is imported and hash-verified, never re-fitted. A and B are fresh renders on new seeds that measure the frozen estimator under conditions F9c's data never produced. C touches no simulator at all — it replays the existing F9c runtime cache to answer a selection-rule question.

**Tech Stack:** Python 3.10, NumPy, frozen Ultralytics YOLO11n checkpoint (inference only), gym-duckietown 6.2.0, pytest, TOML config.

---

## Global Constraints

**The F9c estimator is frozen. This gate may not tune it.**

```
configs/f9c_robust_belief_v1.toml
  SHA256 359dc52020421c248bf6c26e036234191b2b97d24b505a66d85daa85b563704e
```

Every one of these is imported and asserted, never re-derived:

```
b_r = -0.02986607430110723          b_beta = 0.0012336629252072933
lambda_r = 9.96243043243885         lambda_beta = 1.0
sigma_floor_r = 0.02041790926900693 sigma_floor_beta = 0.012546331734068323
LR_floor = 0.37362469458201386
gate threshold        = 9.21034037197618    (chi-square 2 DOF, 99%)
association gate      = 13.815510557964274  (chi-square 2 DOF, 99.9%)
P_D^eff center/mid/edge/outside = 0.9490 / 0.9801 / 0.9973 / 0.5587
P_S = 0.995   P_birth = 0.005   prior = 0.50
YOLO checkpoint SHA256 3d4f816d440690493b856d25403a84a3249e4250599319c32569b97cb8d7482c
F7 [ekf] block / Q     byte-identical to configs/oracle_ekf_v1.toml
```

**If any F9d result is disappointing, that is the finding.** Re-fitting anything in response
to what F9d shows would destroy the only thing this gate is for: independent evidence about an
estimator whose parameters were fixed before it was tested.

**Seeds — every earlier band is consumed:**

```
detector train/val/test   1101-1106 / 2101-2102 / 3101-3102
F9a calibration           4101-4104
F9b final (FROZEN TEST)   5101-5104
F9c calibration           6101-6108
F9c final (FROZEN TEST)   7101-7104
F9d development / yield   8101-8108      <- scenario tuning happens ONLY here
F9d-A final               8201-8204
F9d-B final               8301-8304
```

Reading 5101-5104 or 7101-7104 is a gate failure. Scenario geometry may be adjusted on
8101-8108 and must be frozen before any 82xx or 83xx frame is rendered.

**Environment (identical to F9c):**

```bash
wsl.exe -d Ubuntu-Baru -- bash -lc 'cd /home/pannntastic/aivnv/duckie-pomdp && export PYTHONPATH=src:/home/pannntastic/aivnv/duckie/src && export DUCKIETOWN_HEADLESS=1 && export CUBLAS_WORKSPACE_CONFIG=:4096:8 && export PYTHONHASHSEED=8123 && export CUDA_VISIBLE_DEVICES=0 && /home/pannntastic/aivnv/duckie/.venv/bin/python <command>'
```

Windows UNC paths for edits; `wsl.exe` only for execution. `$?` between `;`-separated
statements is unreliable — use `&&`/`||`. Renders exceed the 10-minute Bash cap; use
`run_in_background: true`. Not a git repository — checkpoints are ledger entries plus a green
suite. Baseline: **251 passed**.

**Pre-registered minima and criteria — fixed now, before any F9d frame exists:**

```
F9d-A  localization-outlier stress

  THE HEADLINE OUTLIER SET IS DEFINED BY BASELINE A, NOT BY ROBUST B:
    frames, eligible + detected, where BASELINE A's highest-confidence
    selection has GT IoU < 0.5
  Robust B is then evaluated ON THOSE SAME FRAMES, whatever it chose to do
  (select a different candidate, reject, abstain, run prediction-only).
  Defining the set by Robust B's own associated box would be selection bias:
  the question is whether the robust layer improves the OUTCOME when baseline
  perception fails, so baseline failure defines the population.

  support required, ALL THREE:
    natural outlier frames                    >= 50
    independent contiguous outlier events     >= 12
    events represented in                     >= 3 of the 4 final seeds
  50 frames from 2 long bursts is not the same evidence as 50 scattered
  failures. If frames clear 50 but events fall below 12, or the events
  concentrate in fewer than 3 seeds, the verdict is INSUFFICIENT EVIDENCE.
  below 30 frames -> INSUFFICIENT EVIDENCE regardless of the other two.

  PRIMARY MEASURE IS OUTCOME, not rejection counts:
    frame-level   Robust B belief range RMSE <= Baseline A's on the outlier set
    event-level   per event: Baseline A RMSE, Robust B RMSE, and the paired
                  difference (Robust - Baseline); summarise the paired
                  differences with a bootstrap CI at the EVENT level, since
                  frames within an event are not independent
  also report: max transient belief error, recovery frames after each burst
  secondary diagnostics, reported separately: Robust-associated IoU < 0.5 count,
    the gate confusion table, abstention count

F9d-B  long-absence stress
  minimum absence episodes with >= 20 consecutive frames   >= 12
  minimum absence episodes with >= 40 consecutive frames   >= 4
  the absence kinds are reported SEPARATELY and never pooled.

  B1 genuine out-of-domain — pedestrian exists but leaves the field of view.
     Invariant I3 applies no likelihood, so belief follows the pure prediction
     recurrence with the frozen parameters:
         p_(t+1) = P_S*p_t + P_birth*(1 - p_t) = 0.99*p_t + 0.005
         fixed point p* = 0.5
         closed form  p_n = 0.5 + (p_0 - 0.5) * 0.99**n
     From p_0 = 0.99 this gives p_40 = 0.828 and NEVER crosses 0.5 from above.
     criterion B1a: P(e) > 0.50 through 40 out-of-domain frames
       -- retained, but it is near-guaranteed by the algebra, so it verifies
          the routing runs at all rather than stressing anything
     criterion B1b, the informative one: on frames the RUNTIME classified
       OUTSIDE_DOMAIN, observed P(e) must track the analytic recurrence.
       Report observed, analytic, and |deviation| at frames 1, 5, 10, 20, 30, 40.
     criterion B1c, the system-level one: report the confusion between
       GT genuinely out of FOV and runtime-predicted OUTSIDE_DOMAIN.
       Invariant I3 routes on PREDICTED observability. If the pedestrian has
       genuinely left but the estimator still says CENTER, the likelihood is
       applied and belief collapses -- a real failure that analysing only the
       correctly-classified subset would hide.

  B2 controlled in-domain detector silence — pedestrian REMAINS PRESENT AND
     VISIBLE while the Duckie detector output is suppressed.
     This tests: does an unsupported in-domain track eventually decay under
     prolonged detector silence?
     It does NOT test whether the estimator can forget a pedestrian that
     genuinely left -- the pedestrian is still there. Do not make that claim.
     criterion B2: P(e) < 0.10 after >= 20 consecutive suppressed in-domain frames
     Report the safety trade-off explicitly: with the pedestrian still present,
     P(e) < 0.10 is not unambiguously "safe" either. It is the correct
     behaviour for an unsupported track, and simultaneously a loss of a true
     positive. State both.

  B3 controlled target disappearance — OPTIONAL, subject to a feasibility gate.
     The pedestrian is removed from the scene at a seed-determined time; the
     detector runs normally. This is the only arm that answers "can the
     existence filter forget an object that no longer exists?" If the simulator
     cannot support clean mid-episode removal, drop B3 and say so; B2 still
     closes the criterion, under B2's narrower interpretation.

  criterion (all arms): re-detection restores an active belief within <= 2 frames

F9d-C  association diagnostic — DIAGNOSTIC ONLY, cannot change gate status
  unless it uncovers an implementation bug, in which case STOP and report.
  Conclusions must come from PAIRED COUNTS, never from a subjective reading of
  whether something "improved sharply". See Task 2 for the exact schema.
```

**Do not implement in F9d:** reward, stop logic, SAC/TD3/PPO, any estimator change, any
re-fit, any new filter. **STOP and report after F9d.**

---

## Why each part exists

### A — the outcome question F9c could not answer

F9c found only 9 Baseline-A localization-failure frames in 3,328. On those, the robust layer
did fire — 8 of 9 went prediction-only (7 filtered by association, 1 by the gate) — and the
gate's own statistics were near nominal (sensitivity 3/4, false-rejection 0.66% against the
~1% a χ²₂ 99% gate should produce).

But belief RMSE on those frames went the **wrong way**: Baseline A 0.0218 m vs Robust B
0.0346 m. The likely reason is that a box with IoU < 0.5 is not necessarily wrong in *range*,
while consecutive prediction-only frames accumulate drift — and Baseline A's EKF had already
attenuated the raw measurement error from 0.171 m to 0.0218 m unaided.

So the open question is **not** "does the gate reject bad boxes" — it does. It is **"does
refusing them leave the belief better or worse than letting the EKF suppress them?"** That is
an outcome question, and it needs enough outliers to answer.

### B — the condition that never occurred

F9c's longest natural in-domain miss run was 10 frames, so the `>= 20 consecutive misses ->
P(e) < 0.10` criterion was never exercised. Worse, the I8 floor (`LR_floor = 0.3736`) was
introduced precisely to slow collapse; nobody has checked that an unsupported track still
*does* decay when evidence stops arriving. A floor that is too strong would leave a track
alive indefinitely on no evidence — a failure in the opposite direction from F9b's collapse.

The absence kinds must be separated because invariant I3 makes them behave differently **by
design**: out-of-domain absence applies no likelihood and follows the pure prediction
recurrence, while in-domain silence applies the floored likelihood. Pooling them would hide
whichever is broken.

**A correction to an earlier draft of this plan.** It claimed out-of-domain absence "decays
only through `P_S = 0.995` — a half-life of about 138 frames". That is wrong twice over. The
recurrence is `p_(t+1) = P_S·p_t + P_birth·(1−p_t) = 0.99·p_t + 0.005`, so the decay factor is
`P_S − P_birth = 0.99`, not `P_S`; and it converges to the fixed point `p* = 0.5`, not to zero.
The half-life of the *deviation from 0.5* is `ln(0.5)/ln(0.99) ≈ 69` frames. Starting from
`p_0 = 0.99`, belief reaches only 0.828 by frame 40 and can never cross 0.5 from above. That is
why criterion B1a is retained but demoted, and why B1b and B1c carry the actual evidential
weight.

### C — two questions that look like one but are not

`association_only` in the F9c ablation was the worst row (range RMSE 0.03776 vs baseline
0.02580). A natural hypothesis: with `covariance_calibration` off, association scores
candidates against `S = H P⁻ Hᵀ + λR` using `λ = 1`, a covariance the calibration says is
~10× too small, so the minimum-NIS ranking is being computed against the wrong metric.

**That hypothesis cannot explain everything, and the distinction matters.** Robust B's final
run already used `λ_r = 9.9624`, and minimum-NIS association *still* produced 4 localization
outliers against highest-confidence's 2 — including 2 on duplicate frames where
highest-confidence produced none. Association rescued 0 frames and introduced 2.

So there are two separable claims:

```
C1  does the wrong S explain the ABLATION penalty?          (a calibration question)
C2  does min-NIS pick worse boxes than confidence even
    when S is correct?                                       (a selection-rule question)
```

If C1 improves sharply at the correct λ while C2 remains worse than confidence, the conclusion
is clean and publishable:

> the wrong `S` explains the ablation penalty, but not the selection-rule failure

and minimum-NIS is then **not** worth keeping merely because it is theoretically temporal. A
legitimate Version-1 outcome is that highest-confidence selection plus abstention and gating
beats minimum-NIS candidate selection on this data.

---

## File Structure

```text
configs/f9d_evidence_closure_v1.toml            new; imports and asserts F9c's frozen values

src/duckie_pomdp/evaluation/f9d_protocol.py     config load + frozen-F9c import guards + seed guards
src/duckie_pomdp/evaluation/f9d_stress.py       outlier-yield and absence-run metrics
src/duckie_pomdp/evaluation/f9d_association.py  C1/C2 selection-rule comparison (cache-only)
src/duckie_pomdp/perception/detector_dropout.py controlled detector-output dropout for B2

experiments/probe_f9d_yield.py                  dev-seed yield check (8101-8108)
experiments/evaluate_f9d_outlier_stress.py      A, final seeds 8201-8204
experiments/evaluate_f9d_absence_stress.py      B, final seeds 8301-8304
experiments/diagnose_f9d_association.py         C, replays the F9c cache
experiments/verify_f9d_artifacts.py             read-only verifier

tests/test_f9d_protocol.py
tests/test_f9d_detector_dropout.py
tests/test_f9d_stress.py
tests/test_f9d_association.py
tests/test_f9d_leakage.py

artifacts/f9d_yield_probe.json
artifacts/f9d_frozen_config.json
artifacts/f9d_outlier_stress.csv        artifacts/f9d_outlier_metrics.json
artifacts/f9d_absence_stress.csv        artifacts/f9d_absence_metrics.json
artifacts/f9d_association_diagnostic.json
```

---

## Task 1: F9d protocol with frozen-F9c import guards

**Files:**
- Create: `configs/f9d_evidence_closure_v1.toml`, `src/duckie_pomdp/evaluation/f9d_protocol.py`
- Test: `tests/test_f9d_protocol.py`

**Interfaces:**
- Consumes: `evaluation.f9c_protocol.load_f9c_protocol`, `evaluation.f9_protocol.sha256`.
- Produces: `F9dProtocol` with `f9c_config_path`, `f9c_config_sha256`, `development_seeds`,
  `outlier_final_seeds`, `absence_final_seeds`, `forbidden_seeds`, `minimum_outlier_frames`,
  `minimum_outlier_events`, `minimum_outlier_seeds`, `insufficient_outlier_frames`,
  `minimum_absence_runs_20`, `minimum_absence_runs_40`, `scenarios`, `artifacts`; plus
  `load_f9d_protocol(path, *, require_frozen=False) -> F9dProtocol`.

**The point of this task is that F9d cannot silently drift from F9c.** The protocol must load
the F9c config, assert its SHA256, and expose its parameters read-only. There must be no code
path in F9d that writes an estimator parameter.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_f9d_protocol.py
from pathlib import Path

import pytest

from duckie_pomdp.evaluation.f9d_protocol import load_f9d_protocol

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "f9d_evidence_closure_v1.toml"

F9C_SHA = "359dc52020421c248bf6c26e036234191b2b97d24b505a66d85daa85b563704e"


def test_f9d_pins_the_frozen_f9c_config_by_hash():
    protocol = load_f9d_protocol(CONFIG)
    assert protocol.f9c_config_sha256 == F9C_SHA


def test_f9d_rejects_a_drifted_f9c_config(tmp_path, monkeypatch):
    """If the F9c config ever changes, F9d must refuse to run rather than
    silently evaluate a different estimator than the one it claims to test."""
    text = CONFIG.read_text(encoding="utf-8").replace(F9C_SHA, "0" * 64)
    broken = ROOT / "configs" / "_tmp_f9d_drift_probe.toml"
    broken.write_text(text, encoding="utf-8")
    try:
        with pytest.raises(ValueError, match="frozen F9c"):
            load_f9d_protocol(broken)
    finally:
        broken.unlink(missing_ok=True)


def test_f9d_seeds_are_disjoint_from_every_earlier_band():
    protocol = load_f9d_protocol(CONFIG)
    development = set(protocol.development_seeds)
    outlier = set(protocol.outlier_final_seeds)
    absence = set(protocol.absence_final_seeds)
    assert development and outlier and absence
    assert not (development & outlier)
    assert not (development & absence)
    assert not (outlier & absence)
    forbidden = set(protocol.forbidden_seeds)
    assert {5101, 5102, 5103, 5104} <= forbidden
    assert {7101, 7102, 7103, 7104} <= forbidden
    assert {6101, 6102, 6103, 6104, 6105, 6106, 6107, 6108} <= forbidden
    assert not (development | outlier | absence) & forbidden


def test_f9d_exposes_f9c_parameters_but_defines_none_of_its_own():
    """F9d may read every estimator parameter and write none. A parameter
    defined in the F9d config would be an estimator change wearing a
    stress-test costume."""
    import tomli

    with CONFIG.open("rb") as stream:
        data = tomli.load(stream)
    forbidden_sections = {
        "measurement_model",
        "covariance_calibration",
        "conditional_detection",
        "innovation_gate",
        "association",
        "ekf",
        "existence",
    }
    assert not (forbidden_sections & set(data)), (
        "F9d must not redefine any estimator section; it imports F9c's"
    )


def test_f9d_minima_are_pre_registered():
    protocol = load_f9d_protocol(CONFIG)
    assert protocol.minimum_outlier_frames >= 50
    assert protocol.minimum_outlier_events >= 12
    assert protocol.minimum_outlier_seeds >= 3
    assert protocol.insufficient_outlier_frames == 30
    assert protocol.minimum_absence_runs_20 >= 12
    assert protocol.minimum_absence_runs_40 >= 4


def test_outlier_support_requires_all_three_conditions():
    """Frames alone can be satisfied by two long bursts. The support check must
    fail when events or seed spread fall short, even with frames well clear."""
    from duckie_pomdp.evaluation.f9d_protocol import outlier_support_satisfied

    protocol = load_f9d_protocol(CONFIG)
    assert outlier_support_satisfied(protocol, frames=60, events=15, seeds=4)
    assert not outlier_support_satisfied(protocol, frames=60, events=2, seeds=2)
    assert not outlier_support_satisfied(protocol, frames=60, events=15, seeds=2)
    assert not outlier_support_satisfied(protocol, frames=40, events=15, seeds=4)
```

- [ ] **Step 2: Run tests to verify they fail**

Run the suite with `tests/test_f9d_protocol.py`.
Expected: FAIL — `ModuleNotFoundError: No module named 'duckie_pomdp.evaluation.f9d_protocol'`

- [ ] **Step 3: Write the config**

```toml
schema_version = 1

[provenance]
# F9d tests the F9c estimator. It imports every parameter and defines none.
f9c_config = "f9c_robust_belief_v1.toml"
f9c_config_sha256 = "359dc52020421c248bf6c26e036234191b2b97d24b505a66d85daa85b563704e"
checkpoint_sha256 = "3d4f816d440690493b856d25403a84a3249e4250599319c32569b97cb8d7482c"

[split]
development_seeds = [8101, 8102, 8103, 8104, 8105, 8106, 8107, 8108]
outlier_final_seeds = [8201, 8202, 8203, 8204]
absence_final_seeds = [8301, 8302, 8303, 8304]
forbidden_seeds = [
  1101, 1102, 1103, 1104, 1105, 1106, 2101, 2102, 3101, 3102,
  4101, 4102, 4103, 4104, 5101, 5102, 5103, 5104,
  6101, 6102, 6103, 6104, 6105, 6106, 6107, 6108,
  7101, 7102, 7103, 7104,
]

[minima]
# Pre-registered before any F9d frame exists. Below the insufficient thresholds
# the gate reports INSUFFICIENT EVIDENCE rather than a pass, exactly as F9c did.
# A's support requires ALL THREE. Frames alone can be satisfied by two long
# bursts, which is not the same evidence as scattered failures.
minimum_outlier_frames = 50
minimum_outlier_events = 12
minimum_outlier_seeds = 3
insufficient_outlier_frames = 30
minimum_absence_runs_20 = 12
minimum_absence_runs_40 = 4

[criteria]
outlier_rmse_ratio_max = 1.0          # Robust B must not exceed Baseline A on outlier frames
absence_out_of_domain_floor = 0.50    # B1: P(e) stays above this through 40 frames
absence_in_domain_ceiling = 0.10      # B2: P(e) below this after >= 20 in-domain misses
recovery_frames_max = 2

[artifacts]
yield_probe_json = "../artifacts/f9d_yield_probe.json"
frozen_config_json = "../artifacts/f9d_frozen_config.json"
outlier_csv = "../artifacts/f9d_outlier_stress.csv"
outlier_metrics_json = "../artifacts/f9d_outlier_metrics.json"
absence_csv = "../artifacts/f9d_absence_stress.csv"
absence_metrics_json = "../artifacts/f9d_absence_metrics.json"
association_diagnostic_json = "../artifacts/f9d_association_diagnostic.json"
```

Scenario matrices are added in Tasks 3 and 4 after their yields are measured.

- [ ] **Step 4: Write `f9d_protocol.py`**

Model it on `f9c_protocol.py`. `_validate` must: verify the F9c config's SHA256 and raise
`ValueError` containing the literal `frozen F9c` on mismatch; verify the checkpoint hash;
enforce the three-way seed disjointness and the forbidden-seed exclusion; and assert that no
estimator section appears in the F9d config.

Expose the F9c parameters through a read-only accessor that loads them from the F9c config at
call time, so there is no copy to drift.

- [ ] **Step 5: Run tests to verify they pass** — expect 5 new tests green.

- [ ] **Step 6: Checkpoint** — full suite, expect 256 passed. Record the F9c hash assertion in
`IMPLEMENTATION_NOTES.md` under a new `## F9d` section.

---

## Task 2: F9d-C — association diagnostic, cache only

**Files:**
- Create: `src/duckie_pomdp/evaluation/f9d_association.py`,
  `experiments/diagnose_f9d_association.py`
- Test: `tests/test_f9d_association.py`

**Interfaces:**
- Consumes: `evaluation.f9c_runtime_cache.read_runtime_cache`,
  `belief.measurement_association.MeasurementAssociator`,
  `belief.robust_updater.RobustPedestrianBeliefUpdater`.
- Produces: `compare_selection_rules(cache, truth, *, lambda_scales) -> dict`,
  `duplicate_frame_ranking(cache, truth) -> dict`, and
  `artifacts/f9d_association_diagnostic.json`.

**This task runs first among the substantive ones because it needs no render.** It also cannot
change the gate's status — it is a diagnosis, not a criterion — unless it uncovers an
implementation bug, in which case STOP and report.

### C1 — does the wrong `S` explain the ablation penalty?

The subtlety that makes this non-trivial: if you simply re-run the pipeline with `λ = 1`, the
EKF corrections change, so the predicted state diverges from the reference within a few
frames and you are no longer comparing selection rules — you are comparing two different
trajectories.

**Hold the predicted state fixed.** Run the frozen Robust B configuration once over the cache
to produce the reference predicted-state sequence. Then, at each frame, ask *both* scorings
which candidate they would select **given that same predicted state and covariance**. Compare
the selections, not the trajectories.

### C2 — is min-NIS a worse selection rule than confidence, at the correct λ?

On duplicate frames only, rank candidates by (a) highest confidence with the frozen
lexicographic tie-break and (b) minimum NIS at the frozen `λ_r = 9.96243043243885`. Score each
choice against GT IoU. Report how often each rule picks the better box, how often they agree,
and the resulting localization-outlier counts.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_f9d_association.py
def test_selection_comparison_holds_the_predicted_state_fixed():
    """Both scorings must see the SAME predicted state and covariance on each
    frame. If the comparison re-ran the filter per rule, the trajectories would
    diverge and the result would measure drift, not selection quality."""


def test_lambda_one_and_frozen_lambda_can_disagree_on_the_same_frame():
    """Construct two candidates whose NIS ordering inverts between lambda=1 and
    the frozen lambda. The comparison must report the disagreement rather than
    silently returning identical selections."""


def test_duplicate_ranking_scores_both_rules_against_gt_iou():
    """On a synthetic duplicate frame where the higher-confidence box has the
    better IoU and the lower-NIS box does not, the report must credit
    confidence and penalise min-NIS."""


def test_the_diagnostic_never_writes_an_estimator_parameter():
    """Scan the module for assignment to any frozen parameter name. A diagnostic
    that mutates the estimator is not a diagnostic."""


def test_the_diagnostic_constructs_no_detector_and_no_simulator():
    """Monkeypatch YoloObjectDetector.__init__ and create_gym_duckietown to
    raise; the diagnostic must complete."""
```

Fill each stub using the fixture patterns already in `tests/test_f9c_robust_updater.py`.

- [ ] **Step 2: Run to verify they fail** — `ModuleNotFoundError`.

- [ ] **Step 3: Implement**, then **Step 4: run the diagnostic:**

```bash
<wsl wrapper> experiments/diagnose_f9d_association.py
```

- [ ] **Step 5: Report the paired counts — no subjective language**

Both comparisons emit the **same schema**, so C1 and C2 are read the same way. "Improves
sharply" is not a finding; these counts are.

```
for each comparison (C1: lambda=1 vs lambda=frozen ; C2: highest-confidence vs min-NIS@frozen)

  frames compared
  selections agree
  selections differ

  among the DIFFERING frames only (this is the paired part):
    rule A picks the higher-IoU box     n
    rule B picks the higher-IoU box     n
    tie                                 n

  localization-outlier count      rule A / rule B
  mean selected IoU               rule A / rule B
  median selected IoU             rule A / rule B
```

Conclusion rules, fixed now so they cannot be chosen after seeing the numbers:

```
C1  the S-scale hypothesis is SUPPORTED iff frozen-lambda yields BOTH
      fewer localization-outlier selections than lambda=1
      AND more wins than losses on the paired differing frames
    otherwise it is REFUTED, and the cause of association-only's degradation
      remains open -- say so, do not substitute a new hypothesis post hoc

C2  min-NIS is INFERIOR to highest-confidence iff, at the frozen lambda,
      it yields MORE localization-outlier selections
      AND fewer wins than losses on the paired differing frames
```

The joint reading that would be the cleanest scientific result — and the one F9c's evidence
points toward — is C1 supported **and** C2 inferior:

> the wrong `S` explains the ablation penalty, but not the selection-rule failure

in which case minimum-NIS is not worth keeping merely for being theoretically temporal, and a
legitimate Version-1 conclusion is that highest-confidence selection plus abstention and
gating beats minimum-NIS candidate selection on this data. **Record whichever outcome the
counts show. Do not adjust anything in response, and do not soften a refutation.**

- [ ] **Step 6: Checkpoint** — full suite green, ledger entry.

---

## Task 3: Outlier-eliciting scenarios and the yield probe

**Files:**
- Create: `experiments/probe_f9d_yield.py`, `src/duckie_pomdp/evaluation/f9d_stress.py`
- Modify: `configs/f9d_evidence_closure_v1.toml` (`[[outlier_scenario_matrix]]`)
- Test: `tests/test_f9d_stress.py`

**This task exists because of F9c's most expensive mistake.** F9c rendered its final seeds
without knowing whether they would produce enough localization outliers, and got 9 — too few
to conclude anything. F9d must know the yield **before** spending 8201-8204.

**Natural outliers only in the primary result.** Design scenarios that make YOLO localization
genuinely hard, do not inject GT-derived error. The conditions that plausibly elicit failure,
from F9c's own data:

```
truncation at the frame edge          edge_fov had 543 frames in F9c
small apparent size at far range      far bin had 1887 frames, lowest P_D (0.9717)
rapid viewpoint change under fast     turning scenarios existed but were slow;
  ego yaw (large inter-frame image    this is a geometry effect, NOT motion blur --
  displacement)                       the renderer does not simulate shutter
partial occlusion                     by the stop sign or map furniture
multiple overlapping detections       84 duplicate frames in F9c
```

Combine them: pedestrian near the frame edge **while** the ego turns quickly at far range,
crossing paths that pass behind the stop sign, and approaches that keep the pedestrian
partially truncated.

If natural yield still falls short, a **clearly-labelled synthetic arm** may supplement it —
but it is reported separately and never merged into the primary result, exactly as the F9c
specification required.

- [ ] **Step 1: Write `f9d_stress.py` with the yield metrics and their tests**

`outlier_yield(rows) -> dict` counting eligible+detected frames with GT IoU < 0.5, the
contiguous events they form, and their distribution over `distance_bin` and `fov_region`.

- [ ] **Step 2: Draft the scenario matrix** in the config, marked `use_for_final = false`
until the probe passes.

- [ ] **Step 3: Run the yield probe on development seeds 8101-8108 ONLY**

```bash
<wsl wrapper> experiments/probe_f9d_yield.py --config configs/f9d_evidence_closure_v1.toml
```

- [ ] **Step 4: Judge the yield against the pre-registered minimum**

Scale the observed per-seed outlier rate to the 4 final seeds. If the projection falls below
**50 outlier frames**, revise the scenarios and re-probe on the development seeds. Iterate
here as many times as needed — this is the only place scenario geometry may change.

If after reasonable effort the natural projection stays below **30**, STOP and report: the
honest outcome is that this simulator and detector do not produce enough natural localization
failure to answer the question, and the synthetic arm becomes the only available instrument.

- [ ] **Step 5: Record the probe result** in `artifacts/f9d_yield_probe.json` including the
per-seed rates, the projection, and the decision taken.

- [ ] **Step 6: Checkpoint.**

---

## Task 4: Long-absence scenarios and the controlled dropout mechanism

**Files:**
- Create: `src/duckie_pomdp/perception/detector_dropout.py`
- Modify: `configs/f9d_evidence_closure_v1.toml` (`[[absence_scenario_matrix]]`)
- Test: `tests/test_f9d_detector_dropout.py`

**Two absence kinds, kept apart:**

**B1 — genuine out-of-domain absence.** The pedestrian walks out of the field of view and
stays out for 40+ frames. Entirely natural; no intervention. Under invariant I3 this applies
no likelihood, so `P(e)` follows the pure prediction recurrence
`p_(t+1) = 0.99·p_t + 0.005`, converging to the fixed point `p* = 0.5` with the deviation from
0.5 halving every **69** frames. From `p_0 = 0.99` it reaches only 0.828 by frame 40 and never
crosses 0.5 from above.

An earlier draft of this plan said "decay only through `P_S = 0.995` — a half-life of about
138 frames". That was wrong twice: the factor is `P_S − P_birth = 0.99`, not `P_S`, and the
limit is 0.5, not zero. Do not use the old interpretation.

**B2 — controlled in-domain detector dropout.** A pedestrian that remains visible while the
detector output is suppressed for N frames. This cannot be produced naturally: F9c measured
`P_D ≈ 0.98`, so a natural 20-frame in-domain run has probability ~10⁻³⁴.

The intervention rules, which keep it honest:

```
- suppression happens at the DETECTOR OUTPUT boundary: the detector runs, its
  Duckie detections are discarded for the scheduled window
- the schedule is fixed per episode from the seed, decided BEFORE the episode runs
- the schedule may NOT consult ground truth, the belief, or the detector's output
- every dropout frame is flagged in the CSV so the analysis can separate them
- results are labelled `controlled_dropout`, never merged with natural misses
```

This tests the **existence filter's response to a detector failure**, which is a legitimate
question about the estimator. It is not a claim about how often the detector fails.

- [ ] **Step 1: Write the failing tests**

```python
def test_dropout_suppresses_detections_only_inside_the_scheduled_window():
    """Frames outside the window pass through byte-identical."""


def test_the_dropout_schedule_is_seed_determined_and_reproducible():
    """Two runs with the same seed produce the same windows."""


def test_the_dropout_schedule_cannot_consult_ground_truth_or_belief():
    import inspect
    parameters = set(inspect.signature(DetectorDropout.schedule_for).parameters)
    assert parameters == {"self", "seed", "episode_length"}


def test_dropout_suppresses_duckie_detections_but_not_stop_signs():
    """The intervention targets the pedestrian detector path only."""


def test_a_dropout_frame_is_flagged_for_the_analysis():
    """Every suppressed frame must be distinguishable from a natural miss."""
```

- [ ] **Step 2-4: Verify RED, implement, verify GREEN.**

- [ ] **Step 5: Probe the absence yield on development seeds 8101-8108**, confirming at least
12 runs of ≥20 frames and 4 of ≥40 across the projected final seeds. Adjust scenario lengths
here only.

- [ ] **Step 6: B3 feasibility gate — attempt, then decide**

B3 is the only arm that asks whether the existence filter can forget an object that **no
longer exists**. B2 cannot answer it: the pedestrian is still standing there.

The machinery may already exist. `gym_duckietown.py` has `render_without_objects(object_kinds)`
(line ~189/513), used by F9a's privileged counterfactual renderer to measure `P_FA`, and it
filters `self._simulator.objects`. The scenario pedestrian is held as `_scenario_pedestrian`,
and there are `_visible_objects` / `_first_visible_object` helpers keyed on a `visible`
attribute.

Attempt the minimal intervention: at a seed-determined frame, mark the scenario pedestrian
not-visible so it leaves both the rendered frame and the visible-object queries that GT
sampling uses.

Then verify all three, on a development seed:

```
(a) the pedestrian genuinely vanishes from the rendered RGB after the switch frame
(b) privileged truth reports it as absent from that frame onward
(c) no earlier gate's behaviour changes -- the full suite stays green
```

If any of the three fails, or the intervention requires restructuring the adapter that every
earlier gate depends on, **drop B3 and record why**. B2 still closes the criterion under its
narrower interpretation, and an honest "the simulator does not support clean mid-episode
removal" is a better outcome than a fragile arm.

The B3 schedule obeys the same rules as B2's: fixed per episode from the seed before the
episode runs, never consulting GT, the belief, or the detector's output.

- [ ] **Step 7: Checkpoint.** Record the B3 decision — attempted and working, or attempted and
dropped with the reason.

---

## Task 5: Freeze the F9d configuration

**Files:**
- Modify: `configs/f9d_evidence_closure_v1.toml`
- Create: `artifacts/f9d_frozen_config.json`, `experiments/verify_f9d_artifacts.py`

- [ ] **Step 1:** Set every scenario `use_for_final = true` as the probes justified, and set
`parameters_frozen = true`.

- [ ] **Step 2:** Write `artifacts/f9d_frozen_config.json` with the F9d config SHA256, the
imported F9c config SHA256, the checkpoint SHA256, all three seed groups, the pre-registered
minima and criteria, an ISO timestamp, and
`"final_evaluation_seeds_not_yet_rendered": true`.

- [ ] **Step 3:** Re-assert that the F9c config hash is **still**
`359dc52020421c248bf6c26e036234191b2b97d24b505a66d85daa85b563704e`. If it changed at any
point during Tasks 1-4, STOP — the estimator under test is not the one F9c reported.

- [ ] **Step 4:** Write the read-only verifier, modelled on `verify_f9c_artifacts.py`, with
**two modes**:

```
default          graceful SKIP for artifacts Tasks 6-7 have not yet produced,
                 used during development
--final          STRICT. Every required A, B and C artifact must exist, every
                 hash must match, and a missing artifact is a FAILURE, never a
                 SKIP. Exit non-zero on any absence or mismatch.
```

The strict mode exists because a graceful verifier is exactly the wrong instrument at report
time: F9c's verifier legitimately reported `12 PASS / 1 SKIP` while Task 11's artifacts did not
yet exist, and a final report built on that reading would have been certified by a check that
verified nothing. **Task 8 must run `--final` and record its output before writing the
report.** Add a test that `--final` fails when a required artifact is absent — a strict mode
that has never been seen to fail is not a strict mode.

- [ ] **Step 5: Checkpoint** — record the frozen F9d hash in the ledger and `GATES.md`.
**From here the config is read-only.**

---

## Task 6: F9d-A final run — outlier stress

**Files:**
- Create: `experiments/evaluate_f9d_outlier_stress.py`

Run **once** on seeds 8201-8204. Baseline A and Robust B side by side on the same frames and
the same single YOLO inference per frame, exactly as F9c did. Write a runtime cache with
`raw_*` pre-bias candidates, following F9c's invariant I5, so this run is also replayable if
post-processing fails — F9c's final run crashed after rendering and was only recoverable
because that cache existed.

- [ ] **Step 1:** Build the script, reusing F9c's `build_row` and `_step_both_systems` rather
than duplicating them. Test it on **development** seeds first.

- [ ] **Step 2:** Run once, in background.

- [ ] **Step 3: Read the support check FIRST — all three conditions, before any RMSE.**

Call `outlier_support_satisfied(protocol, frames=, events=, seeds=)` from Task 1 and report
each component:

```
natural outlier frames                >= 50   (frames < 30 -> INSUFFICIENT regardless)
independent contiguous outlier events >= 12
final seeds containing >= 1 event     >= 3
```

**All three must hold.** 50 frames arising from two long bursts is not the evidence this gate
needs, and a spread across only two seeds cannot distinguish an estimator property from a
seed artefact. If any condition fails, the verdict is INSUFFICIENT EVIDENCE and the RMSE
comparison is reported as a descriptive secondary, never as the headline.

Read the support numbers before looking at any RMSE — once accuracy figures are seen, the
support judgement is no longer independent.

- [ ] **Step 4: Then read the outcome measure** — belief range RMSE on outlier frames, Robust
B vs Baseline A — plus max transient belief error and recovery frames after each outlier
burst. Report the gate confusion table as a secondary diagnostic, using the same definitions
F9c used so the two gates are comparable.

- [ ] **Step 5:** Run the verifier; confirm exit 0.

- [ ] **Step 6: Checkpoint.**

---

## Task 7: F9d-B final run — long-absence stress

**Files:**
- Create: `experiments/evaluate_f9d_absence_stress.py`

Run **once** on seeds 8301-8304, with the controlled dropout schedule from Task 4.

- [ ] **Step 1:** Build and test on development seeds.

- [ ] **Step 2:** Run once, in background.

- [ ] **Step 3: Report B1 and B2 separately.** For each absence run, report `P(e)` at frames
1, 5, 10, 20, 30 and 40; whether the track stayed active; whether it was deleted; and the
recovery frames after re-detection.

- [ ] **Step 4: Check the criteria, each with its correct interpretation.**

**B1a** `P(e) > 0.50` through 40 out-of-domain frames — near-guaranteed by the algebra; treat
it as a routing check. **B1b** observed `P(e)` tracks `p_n = 0.5 + (p_0 − 0.5)·0.99ⁿ` on
runtime-classified OUTSIDE_DOMAIN frames; report observed / analytic / |deviation| at
1, 5, 10, 20, 30, 40. **B1c** the GT-out-of-FOV vs runtime-predicted-OUTSIDE_DOMAIN confusion
— a pedestrian that has genuinely left while the estimator still says CENTER gets the
likelihood applied and collapses, and that is a real system failure.

**B2** `P(e) < 0.10` after ≥20 consecutive suppressed in-domain frames. State the failure
meaning precisely:

> A B2 failure means the frozen existence model retains an unsupported in-domain track too
> strongly under prolonged detector silence.

It does **not** mean the estimator cannot forget a pedestrian that genuinely left — in B2 the
pedestrian is still present and visible, and only the detector output was suppressed. Only
**B3**, if its feasibility gate passed, may support a claim about genuine disappearance.

Report B2's safety trade-off in both directions: with the pedestrian still present,
`P(e) < 0.10` is correct behaviour for an unsupported track **and** a lost true positive.
Do not soften either half.

- [ ] **Step 5:** Verifier, exit 0. **Step 6: Checkpoint.**

---

## Task 8: Leakage tests, report, and classification

**Files:**
- Create: `tests/test_f9d_leakage.py`
- Modify: `GATES.md`, `README.md`, `IMPLEMENTATION_NOTES.md`, `.aris/compute/local.md`
- Create: `docs/superpowers/F9D_REPORT_FOR_REVIEW.md`

- [ ] **Step 0: Run the strict verifier FIRST**

```bash
<wsl wrapper> experiments/verify_f9d_artifacts.py --final
```
Exit 0 is a precondition for writing the report. Record its output verbatim. If it fails,
the report is not written until the missing or mismatched artifact is resolved.

- [ ] **Step 1:** Leakage tests covering `perception/detector_dropout.py` and the F9d
evaluation modules, AST-based as F9c's are — check code, not comments, so documentation
describing the boundary survives the guard.

- [ ] **Step 2:** Add one test that no F9d module writes any frozen estimator parameter, and
one that the F9c config hash is unchanged.

- [ ] **Step 3: Write the report** in the same two-layer form F9c used: functional criteria
separately from pre-registered statistical targets, with the weaknesses as findable as the
strengths. Include:

```
A: outlier frame count, outcome RMSE both systems, max transient error, recovery,
   gate confusion table, and whether the count cleared the pre-registered minimum
B1: out-of-domain decay curve and the P_S half-life comparison
B2: in-domain decay curve, labelled controlled_dropout throughout
C: C1 and C2 results and which of the three pre-stated outcomes obtained
whether F9c's two open criteria (5 and 10) are now closed, still open, or failed
```

- [ ] **Step 4: Classify.** `CONTROL_READY` requires **both** A and B criteria met with their
minima cleared. If either lacks the sample size, the honest label remains `LIMITED` with the
specific missing evidence named. C cannot change the label.

- [ ] **Step 5: STOP.** Do not begin stop logic, reward, or SAC.

---

## Self-Review

**Spec coverage.** A (outlier stress, outcome-primary) → Tasks 3, 6. B (long-absence, two
kinds separated) → Tasks 4, 7. C (cache-only association diagnostic, C1 and C2 distinct) →
Task 2. Frozen-F9c guarantee → Task 1's hash guard, Task 5's re-assertion, Task 8's test.
Pre-registration → Task 1's `[minima]`/`[criteria]`, frozen in Task 5 before any final frame.

**Deliberate departures from the operator's brief, flagged.** (i) A yield probe on development
seeds (Task 3) is not in the brief, but F9c's most expensive failure was rendering final seeds
without knowing the outlier yield; without it F9d risks repeating n=9. (ii) The B2 controlled
dropout is an intervention, not a natural condition — a 20-frame natural in-domain miss run
has probability ~10⁻³⁴ at the measured `P_D`, so the question cannot be asked without one; the
constraints in Task 4 keep it honest and the results are labelled throughout. (iii) Task 6
writes a runtime cache F9d does not strictly need, because F9c's final run crashed after
rendering and was recoverable only because that cache existed.

**Placeholder scan.** Task 2's five tests and Task 4's five tests are docstring stubs whose
assertions are fully specified in prose; both instruct the implementer to reuse existing
fixture patterns rather than invent conflicting ones. Scenario matrices in Tasks 3 and 4 are
deliberately left to be written from measured yields — writing invented `steps` values here
would be the exact false precision this plan exists to avoid.

**Type consistency.** `F9dProtocol` field names (Task 1) are used by Tasks 3, 6 and 7.
`outlier_yield` (Task 3) feeds Task 6's Step 3 gate. `DetectorDropout.schedule_for(seed,
episode_length)` (Task 4) is called by Task 7. `compare_selection_rules` and
`duplicate_frame_ranking` (Task 2) produce the C1/C2 sections of Task 8's report. Every
estimator parameter is read through Task 1's read-only accessor, never copied.
