## Task 12: Ablation

**Files:**
- Modify: `experiments/evaluate_f9c_robust_belief.py` (add `--ablation`)
- Produces: `artifacts/f9c_ablation_metrics.json`

Seven configurations replayed from `artifacts/f9c_runtime_cache.npz` — invariant I4 means the ablation constructs no detector and no simulator. **The bias stage is always present; only which constants it carries varies.** Writing the table this way is what makes the baseline row equal Baseline A by definition:

```text
row                         bias stage      other components
─────────────────────────── ─────────────── ──────────────────────────────────
baseline                    F9b frozen      none        (== Baseline A exactly)
+ bias refit only           F9c fitted      none
+ innovation gate only      F9b frozen      gate
+ temporal association only F9b frozen      association
+ covariance calibration    F9b frozen      lambda R + posterior floor
+ conditional detection     F9b frozen      P_D^eff + I3 outside-domain policy
all combined                F9c fitted      all of the above  (== Robust B exactly)
```

Note that `+ covariance calibration only` still uses `λR` in the invariant-I1 provider, so its association and gate — where present in other rows — remain internally consistent. No row mixes a raw-`R` provider with an inflated-`R` correction.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_f9c_robust_updater.py
def test_ablation_endpoints_match_the_two_headline_systems():
    """ablation['baseline'] must equal the Baseline-A metrics and
    ablation['all_combined'] must equal the Robust-B metrics, field by field,
    to within 1e-12."""


def test_ablation_performs_no_inference_and_no_render(monkeypatch, tmp_path):
    """Invariant I4. Import experiments.evaluate_f9c_robust_belief, monkeypatch
    YoloObjectDetector.__init__ and create_gym_duckietown to raise
    AssertionError('ablation must not run inference or render'), then call the
    ablation entry point on a small hand-written runtime cache written with
    write_runtime_cache. It must complete and produce seven result sets."""


def test_runtime_cache_contains_pre_bias_raw_candidates():
    """Invariant I5. Write a cache from a known pipeline observation, read it
    back, and assert raw_candidate_range_m equals the projector's raw output --
    not that value minus any bias constant. Then replay the same cache twice,
    once with bias_refit=False and once with True, and assert the two runs see
    candidate ranges differing by exactly (f9c_range_bias_m - f9b_range_bias_m).
    A cache written post-correction would make that difference zero."""


def test_ablation_refuses_a_runtime_cache_whose_hash_does_not_match():
    """read_runtime_cache must raise when the recorded SHA256 disagrees with the
    file, so a silently regenerated cache cannot be replayed as if it were the
    final run's."""


def test_bias_ablation_uses_f9b_bias_when_switch_off():
    """Every ablation row with bias_refit=False must carry a bias stage whose
    range_bias_m equals -0.045904804710162034 and bearing_bias_rad equals
    0.00414567890700929 -- the F9b frozen constants, not identity and not the
    F9c fitted values. Assert on the constructed updater's _bias, so the check
    cannot be satisfied by a coincidentally matching metric."""


def test_bias_refit_switch_applies_f9c_frozen_bias_before_association():
    """With bias_refit=True, feed one candidate and capture the range the
    associator receives. It must equal raw_range - f9c_range_bias_m, proving
    the correction happens upstream of association rather than after the gate
    or not at all. Run the same frame with bias_refit=False and assert the
    associator receives raw_range - f9b_range_bias_m instead."""
```

- [ ] **Step 2: Run to verify they fail**

Run: `$PY -m pytest tests/test_f9c_robust_updater.py -k "ablation or bias" -q`
Expected: FAIL

- [ ] **Step 3: Implement**

Add `--ablation` to `evaluate_f9c_robust_belief.py` as a **mutually exclusive** mode with the final run: `--ablation` loads `artifacts/f9c_runtime_cache.npz` plus `artifacts/f9c_evaluation_truth.npz` and never touches the detector or the simulator. Structure the module so the detector and the environment are constructed inside the final-run branch only — that is what makes the monkeypatch test meaningful rather than decorative.

Replaying candidates rather than images is what makes one inference pass sufficient. Record the runtime-cache SHA256 in `artifacts/f9c_ablation_metrics.json` so the ablation and the headline result are provably the same frames.

- [ ] **Step 4: Run and verify**

Run: `$PY experiments/evaluate_f9c_robust_belief.py --ablation` then `$PY -m pytest tests/test_f9c_robust_updater.py -k "ablation or bias" -q`
Expected: PASS

- [ ] **Step 5: Checkpoint**

Run: `$PY -m pytest tests -q` → expect 192 passed.

---

