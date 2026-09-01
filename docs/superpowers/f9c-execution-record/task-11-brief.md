## Task 11: Final evaluation on seeds 7101–7104

**Files:**
- Create: `experiments/evaluate_f9c_robust_belief.py`
- Create: `src/duckie_pomdp/evaluation/f9c_belief.py`
- Create: `src/duckie_pomdp/evaluation/f9c_runtime_cache.py`

**Interfaces:**
- Consumes: `F9cProtocol` (`require_frozen=True`), `RobustPedestrianBeliefUpdater`, `PedestrianBeliefUpdater`.
- Produces: `artifacts/f9c_validation.csv`, `artifacts/f9c_belief_metrics.json`, `artifacts/f9c_nis_metrics.json`, `artifacts/f9c_error_cases/`, `artifacts/f9c_runtime_cache.npz`, `artifacts/f9c_evaluation_truth.npz`; and `summarize_f9c(rows, *, protocol) -> tuple[dict, dict]`; and `write_runtime_cache(path, frames) -> str` / `read_runtime_cache(path) -> tuple[RuntimeCacheFrame, ...]`.

Two systems run side by side on the *same rendered frames and the same single YOLO inference per frame*, exactly as F9b ran raw and corrected in parallel:

```text
Baseline A: corrected YOLO (F9b frozen bias) → highest-confidence selection → frozen F7 EKF → frozen existence
Robust  B: candidates → association → gate → λ-inflated R → frozen F7 EKF → P_D^eff existence → posterior floor
```

Baseline A must be constructed from the *unmodified* `PedestrianBeliefUpdater` and the F9b bias constants, so it is a genuine control.

- [ ] **Step 1: Write `f9c_belief.py`**

Reuse `duckie_pomdp.evaluation.f9_belief.scalar_error_metrics` and `belief_metrics` for the shared quantities. Add:
- `coverage_error_68 = |coverage_68 − 0.68|`, `coverage_error_95 = |coverage_95 − 0.95|` (named exactly that — not "ECE");
- `std_over_rmse` per variable;
- `miss_sequence_metrics(rows)` reporting `P(exists)`, reported range std, `track_active`, and recovery frames at miss-run lengths 1, 3, 5, and 10 — using genuine natural misses only, labelled `natural`; any injected sequence must be labelled `synthetic` and reported separately;
- `robustness_metrics(rows)` reporting localization-outlier count (post-hoc GT IoU < 0.5), how many were accepted vs rejected by the gate, duplicate frames, wrong-association events, natural misses, belief maintained, false-track initializations, track deletions, and recoveries. Misses must be broken out into three disjoint counts, because invariant I2 and invariant I3 make them different events and collapsing them would hide whether the corrections worked:

```text
detector_miss_in_domain        no candidate, predicted CENTER/MID/EDGE   -> likelihood applied
detector_miss_outside_domain   no candidate, predicted OUTSIDE_DOMAIN    -> prediction only
gated_rejection                candidate present, gate rejected          -> counted as a DETECTION
```

  Report, explicitly, how many `gated_rejection` frames occurred and what fraction of them retained an active belief. Under the earlier draft of this plan every one of those frames would have been an existence miss; that number is the direct measure of what Finding 6 bought.

  **Report active-belief retention per class and never pool it.** Invariant I3 makes outside-domain misses nearly free to survive — they apply no likelihood at all — so a pooled retention figure is dominated by whichever class happens to be more common. A run with 80 outside-domain and 20 in-domain misses would report 80% retention while the belief collapsed on essentially every genuine detector miss. Pooled retention must not appear as a control-readiness criterion anywhere; the primary criterion is `detector_miss_in_domain` retention alone, because that is the only class where `P_D^eff` is actually doing work. Outside-domain retention is still reported, but its role is a sanity check on a different question — that belief does not collapse merely because the camera is not looking at the region.

  If `detector_miss_in_domain` has fewer than 20 frames in the final run, say so and treat the in-domain criterion as under-powered rather than passed.
- `outlier_impact(rows)` reporting measurement RMSE, Baseline-A RMSE, Robust-B RMSE, and max transient belief error over the GT-labelled outlier frames;
- `safety_bias(rows)` reporting `E[μ_r − r_GT]` separately for Baseline A and Robust B with an explicit `sign_interpretation` string — positive means the pedestrian is believed farther than reality;
- `support_check(rows, minimum_support)` returning per-bin counts and a boolean `satisfied`.

- [ ] **Step 2: Write `f9c_runtime_cache.py`**

Invariant I4 requires the ablation to be a pure replay, so the cache must survive process exit. An in-memory cache would silently let a second process re-render and re-infer, which would break the "one inference pass" guarantee and could even produce different candidates under domain randomization.

`RuntimeCacheFrame` is a frozen dataclass holding **only runtime-visible, pre-bias quantities**. Every candidate field is named `raw_*` — the naming is load-bearing, not cosmetic:

```text
episode, seed, scenario, frame, dt_s
raw_candidate_count
raw_candidate_range_m[], raw_candidate_bearing_rad[], raw_candidate_confidence[]
raw_candidate_bbox[4][], raw_candidate_projection_failed[]
ego_linear_velocity_mps, ego_yaw_rate_rad_s, ego_motion payload as consumed by predict()
```

**Invariant I5 — the runtime cache is written BEFORE any bias correction.** The ablation must be able to send the same candidate down two paths:

```text
raw candidate ─┬─ F9b bias → baseline row
               └─ F9c bias → robust row
```

If the cache stored F9c-corrected candidates, `bias_refit = false` would replay `z_F9c_corrected − b_F9b`, which is neither Baseline A nor anything else meaningful, and `test_switching_every_robust_component_off_reproduces_the_f9b_path` would be comparing against a corrupted control. Cache at the output of `YoloPedestrianMeasurementPipeline.observe`, before step 4b of Task 8's order of operations.

Ground truth goes to `artifacts/f9c_evaluation_truth.npz`, keyed by `(episode, frame)`, in a **separate file** so that a replay consumer can be handed the runtime cache alone. `write_runtime_cache` returns the SHA256 of the written file; record it in `artifacts/f9c_belief_metrics.json`. `read_runtime_cache` must verify that hash and raise on mismatch — an ablation replaying a cache that has been regenerated is not an ablation.

Ragged per-frame candidate lists must be stored with an explicit offsets array rather than `dtype=object`, so the `.npz` loads without `allow_pickle`.

- [ ] **Step 3: Write `evaluate_f9c_robust_belief.py`**

Structure it on `experiments/validate_f9_yolo_ekf.py`. Preserve the runtime/privileged boundary comment placement: both updaters must be stepped **before** `integration.privileged.read()` is called. Write the runtime cache and the truth file as part of this run. Log per gated measurement, as required: `frame, confidence, predicted range/bearing, measurement range/bearing, innovation, NIS, gate threshold, decision` — and, per invariant I2, also log `detector_detected` and `kinematic_measurement_accepted` so the gate log itself shows that a rejection did not become a miss.

- [ ] **Step 4: Run the final evaluation — ONCE**

```bash
$PY experiments/evaluate_f9c_robust_belief.py --config configs/f9c_robust_belief_v1.toml
```

- [ ] **Step 5: Check the pre-specified support minima FIRST, before reading any accuracy metric**

```bash
$PY -c "
import json
m = json.load(open('artifacts/f9c_belief_metrics.json'))
print(json.dumps(m['metrics']['support_check'], indent=2))
"
```

If `satisfied` is false, the gate is at best `LIMITED` and `CONTROL_READY` is unavailable. Do not re-run with different scenarios to fix it — record the shortfall and report it.

- [ ] **Step 6: Verify artifacts without re-running inference**

Run: `$PY experiments/verify_f9c_artifacts.py`
Expected: exit 0. The verifier must also confirm that the runtime-cache SHA256 recorded in `artifacts/f9c_belief_metrics.json` matches the file on disk.

- [ ] **Step 7: Checkpoint**

Run: `$PY -m pytest tests -q` → expect 186 passed. Record row count, support counts, runtime-cache SHA256, and the frozen config hash in `IMPLEMENTATION_NOTES.md`.

---

