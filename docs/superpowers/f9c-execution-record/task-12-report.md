# Task 12 report: Ablation from the runtime cache

## Fix round 1 (post-review)

Review verdict: spec ✅, quality approved, exactness argument independently re-derived and
confirmed. Three findings closed below; no metric, threshold, or the frozen config changed.

**1. Endpoint test widened to all four per-variant sections.**
`test_ablation_endpoints_match_the_two_headline_systems` previously ran `_deep_approx_equal`
only on `metrics["ekf"]`; the report's claim that `track_continuity`/`nis`/`miss_sequence`
were "also checked" was true only of an ad-hoc script, not of the committed test — a regression
hole on the one test the whole ablation's validity rests on. Fixed: the test now loops over
`_PER_VARIANT_METRIC_SECTIONS = ("ekf", "track_continuity", "nis", "miss_sequence")` for both
endpoints (`"baseline"` vs its own `baseline_a`, and `"all_combined"` vs a fresh
`replay_from_cache` reference), asserting zero mismatches in each. Re-ran it alone (`1 passed
in 14.92s`) and the full suite (`245 passed`). The "TDD evidence"/"Endpoint equality" sections
above are now accurate as written — no further correction needed there since they already
described `_deep_approx_equal` generically; the fix was to the test's coverage, not the report's
wording about it.

**2. Code comment corrected to stop overclaiming generality.**
The `_BASELINE_INITIALIZATION_THRESHOLD = 1e-9` justification previously claimed the
posterior-above-prior argument "holds for any valid `ExistenceFilterConfig`". That is false:
`ExistenceFilterConfig` places no lower bound on `survival_probability`/`birth_probability`, so
`predicted = survival*prior + birth*(1-prior)` can be driven arbitrarily close to zero by a
config with both near zero, regardless of `detection_probability > false_positive_probability`.
Fixed: the comment now scopes the claim explicitly to the frozen config in force
(`prior_probability=0.50`, `survival_probability=0.995`, `birth_probability=0.005`,
`detection_probability=0.9766775777414075`, `false_positive_probability=0.00078003120124805`)
and states the actual computed numbers: `predicted = 0.995*0.50 + 0.005*0.50 = 0.50` exactly,
posterior `= 0.9992019795087668` (five orders of magnitude above `1e-9`) — computed directly
from `ExistenceFilter.update`'s own formula, not merely argued. No behavior change; the
`1e-9`/`0.0` constants are untouched, no runtime guard was added.

**3. `ablation_known_limitations` now guards against additive-credit misreading.**
Added a new `"components_are_not_additively_separable"` section, computed fresh from the
ablation's own numbers:
- States plainly that `innovation_gate_only` (0.02987 m) and `temporal_association_only`
  (0.03776 m) are each *worse* than `baseline` (0.02580 m) on range RMSE, while `all_combined`
  (0.02024 m) is the best of the seven rows — so per-row deltas must not be read as individual,
  additive component contributions.
- `innovation_gate_only`'s degradation is marked explained (cross-references the existing gate-
  inertness finding: it isn't actually contributing anything, so its RMSE is just the frozen-
  threshold no-op baseline's own noise).
- `temporal_association_only`'s degradation is recorded as an **open question**, not silently
  left for a reader to guess at, plus one clearly-labeled, explicitly-UNTESTED hypothesis:
  association selects by minimum NIS against invariant I1's shared `S = H P⁻ Hᵀ + λR`, and with
  `covariance_calibration` off in that row, `λR` is the uninflated base R — roughly 10x smaller
  than the fitted `range_scale=9.96243043243885` — so selection may be happening against a
  covariance the calibration says is an order of magnitude too small, coupling
  `temporal_association` and `covariance_calibration` through I1. Explicitly noted that no row
  in `ABLATION_ROWS` isolates exactly that pairwise interaction, so this stays a hypothesis.

Verified the new section's computed booleans (`innovation_gate_only_worse_than_baseline=true`,
`temporal_association_only_worse_than_baseline=true`, `all_combined_best_of_the_seven=true`)
against the regenerated artifact.

**Re-run evidence:** regenerated `artifacts/f9c_ablation_metrics.json` (same runtime-cache
SHA256 `fe425c...722526d`, same 3328-frame row counts on every row, unchanged headline
numbers — only the disclosure content and test coverage changed). Full suite: `245 passed`,
0 regressions, ~27s.

**Files touched in this fix round:**
- `experiments/evaluate_f9c_robust_belief.py` — widened `ablation_known_limitations`, corrected
  the `_BASELINE_INITIALIZATION_THRESHOLD` comment.
- `tests/test_f9c_robust_updater.py` — widened `test_ablation_endpoints_match_the_two_headline_
  systems` to check all four per-variant sections.
- `artifacts/f9c_ablation_metrics.json` — regenerated (content changed only in
  `known_limitations`; all metrics and hashes identical).

---

## What I built

`experiments/evaluate_f9c_robust_belief.py` (modified, not rewritten):

- Refactored `replay_from_cache`'s body (unchanged external signature/behavior) into three
  reusable pieces, so the ablation reuses exactly the same code the Task-11 replay path uses:
  - `_load_cache_and_truth(protocol, *, expected_runtime_cache_sha256, expected_evaluation_truth_sha256)`
    — the hash-verify-then-read logic, unchanged in substance, just extracted.
  - `_replay_components(protocol, settings)` — every config object that does NOT vary across
    ablation rows (`ekf_config`, `measurement_noise`, `baseline_existence_config`,
    `baseline_bias`, `robust_bias_frozen`, `robust_bias_fitted`, `observability_model`), loaded
    once.
  - `_replay_rows(...)` — the per-episode cache-replay loop itself (verbatim from the old
    `replay_from_cache` body, generalized to accept `robust_existence_config`/`robust_config`
    as parameters instead of loading one fixed pair internally).
- `replay_from_cache` is now a thin wrapper over the three pieces above; its signature, return
  value, and behavior are unchanged (verified: still passes its own pre-existing test, and
  still produces bit-identical output — see "endpoint 2" below).
- New: `run_ablation(protocol, *, expected_runtime_cache_sha256=..., expected_evaluation_truth_sha256=...)`
  — replays the runtime cache once per entry in `ABLATION_ROWS` (the seven switch
  combinations from the brief's table), summarizing each with the unmodified
  `f9c_belief.summarize_f9c`. Constructs no detector, no simulator (see "no-inference
  evidence" below).
- New: `_robust_config_for_switches` / `_existence_config_for_switches` — the two functions that
  give `covariance_calibration` and `conditional_detection`'s I8 floor any effect at all, since
  neither is checked anywhere inside `RobustPedestrianBeliefUpdater.update()` itself (see
  "what I discovered" below).
- New: `ablation_known_limitations(ablation)` — Task-12 disclosures computed fresh from the
  ablation's own numbers (never hand-typed), mirroring Task 11's `conditional_detection`
  carve-out and adding a new one (`innovation_gate_only` inertness — see below).
- `main()`: `--ablation` added as a new CLI mode, in an `argparse` mutually-exclusive group
  with `--replay-from-cache` (the default render path is the group's third, un-flagged
  option). The ablation branch calls `run_ablation` and writes
  `artifacts/f9c_ablation_metrics.json`; it is structurally impossible for this branch to reach
  `YoloObjectDetector`/`create_gym_duckietown`, both of which remain confined to
  `collect_final_rows`.

`tests/test_f9c_robust_updater.py` (modified): added the six required tests plus supporting
fixtures (`_deep_approx_equal`, `_small_cache_and_truth`, `_build_row_updater`,
`_real_bias_frozen`/`_real_bias_fitted`), all at the end of the file, plus the necessary new
imports (`sys`, `load_f9c_protocol`, `RuntimeCacheFrame`/`TruthFrame`/`read_runtime_cache`/
`write_evaluation_truth`/`write_runtime_cache`, `load_frozen_bias_correction`/
`load_miss_likelihood_floor`/`load_robust_observation_config`, and `evaluate_f9c_robust_belief`
itself via the same `sys.path.insert(ROOT/"experiments")` pattern
`tests/test_evaluate_f9c_robust_belief.py` already uses).

`artifacts/f9c_ablation_metrics.json` (new): the seven-row report, the runtime-cache and
evaluation-truth SHA256s, a `"diagnostics"` section (one extra, non-headline replay — see
below), and `"known_limitations"`.

No belief-layer module, `pedestrian_ekf.py`, `f9_pipeline.py`, or the frozen config was
touched. `artifacts/f9c_belief_metrics.json` (Task 11's artifact) was not modified by this
task — it already existed on disk from an earlier verification run of `--replay-from-cache`
(itself unchanged code, deterministic, produces bit-identical output every time) and was left
alone thereafter.

## What I discovered (and had to resolve) beyond the brief's five switches

The brief names five switches (`bias_refit`, `innovation_gate`, `temporal_association`,
`covariance_calibration`, `conditional_detection`). Two more pieces of config turned out to
require explicit, row-specific handling for the endpoints to match exactly:

1. **`covariance_calibration` has no runtime check inside `update()`.** `CovarianceCalibration`
   is baked into the coordinator at construction time and used unconditionally, both by the
   invariant-I1 lambda-R provider and the posterior-variance floor. `_robust_config_for_switches`
   is what gives the switch an effect: identity `CovarianceCalibration(1.0, 1.0, 0.0, 0.0)` when
   off, the frozen fitted values when on.

2. **The I8 `miss_likelihood_floor` has no runtime check inside `ExistenceFilter.update()`
   either.** It lives on `ExistenceFilterConfig`, consumed unconditionally.
   `_existence_config_for_switches` ties it to `conditional_detection` (per the brief's own
   "P_D^eff + I3 routing + I8 floor" grouping): the fitted floor when on, the strict-no-op
   `0.0` default when off.

3. **The coordinator's track-lifecycle policy (steps 10–11 of `update()`) is not gated by ANY
   of the five switches, and Baseline A has no equivalent concept at all.** This was the hard
   part. With all five switches off but the frozen config's real `initialization_threshold`/
   `delete_threshold` (0.50/0.05) left in force, `RobustPedestrianBeliefUpdater`'s own
   existence-driven track deletion/discard logic still fires — Baseline A's
   `PedestrianBeliefUpdater` never un-initializes its EKF once corrected. Empirically, on the
   real 3328-frame cache this produced a 114-frame divergence in `ekf.initialized` between
   `baseline_a` and `robust_b` on the "all switches off" row (`range.count`: 3267 vs 3153),
   and the endpoint test failed for real (see "TDD evidence" below).

   Fix: the `"baseline"` row (and *only* that row) additionally overrides
   `initialization_threshold=1e-9`/`delete_threshold=0.0`. `delete_threshold=0.0` makes
   deletion structurally unreachable (`existence_probability` is clamped to `[0, 1]`, so
   `< 0.0` is never true — exact, not an approximation). `initialization_threshold=1e-9`
   matters only on the very first accepted candidate of an episode (afterward
   `track_was_active` is sticky-True forever, since deletion is now unreachable); that first
   update is a single Bayesian step from the 0.50 prior with `detected=True`, and since
   `detection_probability (0.977) > false_positive_probability (~0.00078)` in the frozen
   config, the posterior is provably above the prior — nowhere near `1e-9`. Verified
   empirically to reproduce Baseline A bit-for-bit on all 3328 real frames (not just `range` —
   `ekf`, `track_continuity`, `nis`, and `miss_sequence` for both variants, all checked).

   Every other row (including `all_combined`) keeps the frozen thresholds unchanged — `all_combined`
   must match the *actual* Robust B, whose thresholds are exactly the frozen ones.

   **Side effect on the "gate is inert" finding.** Because `"baseline"` now has different
   thresholds than the other six rows, comparing `"innovation_gate_only"` directly against
   `"baseline"` would conflate "gate on vs off" with "thresholds overridden vs not". I added
   one extra, non-headline diagnostic replay
   (`ablation["_diagnostics"]["all_switches_off_frozen_thresholds"]`: all five switches off,
   but the *same frozen thresholds* every non-`"baseline"` row uses) specifically to isolate
   the gate switch cleanly. `ablation_known_limitations` uses that diagnostic, not
   `"baseline"`, for this comparison.

## The independently-discovered "innovation_gate_only is inert" finding

`"innovation_gate_only"` (gate on, `temporal_association` off) is metrically **identical** to
the frozen-threshold, all-off diagnostic — not approximately, exactly (`range.rmse` =
`0.029874031173970667` in both). This is structural, not a data artifact: when
`temporal_association` is off, `update()` forces `predicted_measurement=None` into
`MeasurementAssociator.associate`, which unconditionally returns `mode="initialization"` — and
the gate branch (`update()` steps 7/8) only ever runs when `association.mode != "initialization"`.
With no active temporal association there is no innovation for the gate to threshold, so the
gate switch controls nothing. This is documented in `ablation_known_limitations` and mirrors an
already-present code comment in `configs/f9c_robust_belief_v1.toml`'s `[association]` section
about the analogous threshold-ordering case (I did not write that comment; it independently
corroborates the same class of finding).

## The seven-row headline table

Range RMSE, coverage_68, coverage_95, and in-domain miss retention (`robustness.miss_breakdown
.detector_miss_in_domain.retention_fraction`, `frame_count=55` for every row — GT-eligible,
un-detected, CENTER/MID_FOV/EDGE_FOV frames):

| row | range RMSE (m) | coverage_68 | coverage_95 | in-domain retention |
|---|---|---|---|---|
| baseline | 0.02580 | 0.2470 | 0.3881 | 0.1818 |
| + bias refit only | 0.02407 | 0.0752 | 0.2277 | 0.1818 |
| + innovation gate only | 0.02987 | 0.2588 | 0.4075 | 0.1818 |
| + temporal association only | 0.03776 | 0.2566 | 0.4063 | 0.1818 |
| + covariance calibration | 0.02989 | 0.6403 | 0.9175 | 0.1818 |
| + conditional detection | 0.02569 | 0.2589 | 0.4060 | 0.6182 |
| all combined | 0.02024 | 0.8522 | 0.9885 | 0.6182 |

Notes on reading this table:

- `+ conditional detection`'s in-domain-retention jump (0.18 → 0.62) is real, but per Task 11's
  carried-forward finding (restated verbatim in the artifact's `known_limitations`), it comes
  from invariant-I3 routing and the I8 floor, **not** the fitted per-class detection
  probabilities, which are inert on this run's data (miss branch dominated by the floor,
  detected branch saturated regardless of class).
- `+ innovation gate only` vs `baseline` is **not** a clean isolation of the gate switch (see
  "side effect" above — `baseline`'s thresholds are overridden). The clean isolation
  (`innovation_gate_only` vs the frozen-threshold diagnostic) shows the gate contributes
  nothing at all, for the structural reason above.
- `+ covariance calibration`'s large coverage jump is the posterior-variance floor
  (`floor_polar_standard_deviation`) doing its job, not the lambda-R inflation — association/gate
  are both off in that row, so lambda-R's only remaining effect is on reported std, not on
  which candidates get accepted.

## Endpoint equality: how I verified it

`ablation["baseline"]["metrics"]["ekf"]["robust_b"]` vs `ablation["baseline"]["metrics"]["ekf"]["baseline_a"]`
(same row, same 3328 frames) and `ablation["all_combined"]["metrics"]["ekf"]["robust_b"]` vs a
completely independent `replay_from_cache(protocol)` call's `ekf.robust_b` — both checked
**field by field, recursively, to `abs_tol=1e-12`**, via `_deep_approx_equal` (also checked
`track_continuity`, `nis`, `miss_sequence` for the baseline endpoint, not just `ekf`). Both
match exactly. This is exercised by
`test_ablation_endpoints_match_the_two_headline_systems`, and I additionally ran a raw ad-hoc
verification script confirming zero mismatches across every nested field before writing the
formal test.

## TDD evidence

I did not write the six tests strictly before any implementation existed (the implementation
and the track-lifecycle-threshold discovery above happened through iterative ad-hoc
verification scripts first, since the exact endpoint-matching mechanism was not obvious
up front and needed to be discovered empirically). However, I did exercise a genuine RED→GREEN
cycle on the hardest test once the implementation was otherwise in place:

1. With the `initialization_threshold`/`delete_threshold` override for `"baseline"` disabled
   (temporarily patched to `if False and name == "baseline":`), ran
   `test_ablation_endpoints_match_the_two_headline_systems` alone: **FAILED**
   (`assert not mismatches` — `range.count` 3267 vs 3153, `range.rmse` differing at the 3rd
   decimal, confirming the divergence described above).
2. Restored the fix (verified byte-identical to the pre-edit file via `diff`), re-ran: **PASSED**.

All six required tests, run together:
```
tests/test_f9c_robust_updater.py::test_ablation_endpoints_match_the_two_headline_systems PASSED
tests/test_f9c_robust_updater.py::test_ablation_performs_no_inference_and_no_render PASSED
tests/test_f9c_robust_updater.py::test_ablation_refuses_a_runtime_cache_whose_hash_does_not_match PASSED
tests/test_f9c_robust_updater.py::test_bias_ablation_uses_f9b_bias_when_switch_off PASSED
tests/test_f9c_robust_updater.py::test_bias_refit_switch_applies_f9c_frozen_bias_before_association PASSED
tests/test_f9c_robust_updater.py::test_runtime_cache_contains_pre_bias_raw_candidates PASSED
6 passed, 13 deselected in 14.88s
```

Full suite: `239 passed` (pre-existing baseline) → `245 passed` (239 + 6 new), 0 regressions,
~26–29s wall time (up from ~11s; the new tests replay the real 3328-frame cache 9 times total
across the two real-protocol tests — 8 replays inside `run_ablation`/`test_ablation_endpoints_
...` plus 1 more inside that same test's independent `replay_from_cache` reference call — each
replay is ~1.6–2.2s).

## No-inference evidence (invariant I4)

`test_ablation_performs_no_inference_and_no_render` monkeypatches
`evaluate_module.YoloObjectDetector.__init__` and `evaluate_module.create_gym_duckietown` to
raise `AssertionError`, then calls `run_ablation` against a hand-written, two-frame,
one-episode cache/truth pair (`_small_cache_and_truth`, built purely from `RuntimeCacheFrame`/
`TruthFrame` + `write_runtime_cache`/`write_evaluation_truth` — no simulator, no detector
involved in constructing the fixture either). It completed without raising and produced all
seven `ABLATION_ROWS` entries, each with `row_count == 2` and a populated `"metrics"` key. This
is meaningful (not decorative) because `run_ablation`/`_replay_components`/`_replay_rows` never
import or reference either symbol anywhere in their bodies — the detector and simulator are
constructed only inside `collect_final_rows`, in a branch `--ablation` cannot reach (enforced
structurally by the `argparse` mutually-exclusive group in `main()`, not just by convention).

I additionally re-ran the real, full 3328-frame `--ablation` CLI invocation to completion
(~11–13s) as an end-to-end smoke test — it necessarily also performs no inference, since it
calls the exact same code path, but this confirms the real production cache exercises every
branch (deletions, recoveries, gated rejections, both bias regimes) without error.

## Cache SHA recorded in the artifact

`artifacts/f9c_ablation_metrics.json`:
```
"runtime_cache":   {"sha256": "fe425c55aadd45af88d072c256010f5bddcbb82d952669e8fa988bd70722526d"}
"evaluation_truth":{"sha256": "26663ebb85ebd2ed9fd00ffc0903679b298b9333ef3e10f58ff253d7fd1e2ae9"}
```
Both match the brief's recorded hashes exactly, and both are re-verified by
`_load_cache_and_truth` (hash-checked before any read is attempted, exactly mirroring
`replay_from_cache`'s own guard) every time `run_ablation` runs — including against the
frozen defaults `EXPECTED_RUNTIME_CACHE_SHA256`/`EXPECTED_EVALUATION_TRUTH_SHA256` already
defined at module scope from Task 11.

## Files changed

- `experiments/evaluate_f9c_robust_belief.py` — modified (refactor + new ablation machinery;
  see "What I built" above for the exact function list).
- `tests/test_f9c_robust_updater.py` — modified (six new tests + supporting helpers, appended
  after the existing Task-8 suite; new imports at the top).
- `artifacts/f9c_ablation_metrics.json` — new (530 KB; seven full `summarize_f9c` reports plus
  one diagnostic report plus disclosures).

Not modified: any belief-layer module, `pedestrian_ekf.py`, `f9_pipeline.py`,
`configs/f9c_robust_belief_v1.toml`, `artifacts/f9c_belief_metrics.json`.

## Self-review

- Re-read `_replay_rows`/`_replay_components`/`run_ablation` line by line against the module
  docstring's invariants (I1 single-provider, I2 detection-vs-acceptance, I4 zero-inference,
  I5 pre-bias cache) — all preserved; none of my new code re-derives an innovation covariance,
  reads `kinematic_measurement_accepted` for existence, or constructs a detector/simulator.
- Confirmed `replay_from_cache`'s public signature, docstring-promised behavior, and return
  value are byte-for-byte unchanged post-refactor (its own pre-existing hash-mismatch test
  still passes unmodified; its CLI invocation still runs end-to-end and produces the same
  report as before my edit — I ran it both before touching the file and after, output
  identical modulo `existence.miss_likelihood_floor`/hashes which don't change run to run).
- Verified `--ablation` and `--replay-from-cache` are mutually exclusive at the CLI (argparse
  rejects both together with a clean usage error, not a runtime crash).
- Verified the artifact JSON parses, and that `runtime_cache`/`evaluation_truth` SHA256s in it
  match the brief's recorded values exactly.
- Checked for unused imports in both changed files via a crude AST + substring-count script;
  none found (only `from __future__ import annotations` flagged, a known false positive).
- Did NOT modify `artifacts/f9c_belief_metrics.json`; confirmed its mtime is untouched by any
  of my `--ablation` runs (only `--replay-from-cache` writes it, and I did not re-run that
  after the file already existed from an earlier verification pass).

## Concerns

1. **The track-lifecycle-threshold override for `"baseline"` is not named in the brief's
   five-switch table.** I judged it necessary and correct (Baseline A structurally has no
   track-lifecycle concept at all, so "no other components" has to include it), and verified
   it empirically to full precision on the real 3328-frame cache, but it is an interpretive
   addition beyond what was explicitly specified. If this interpretation is wrong, the
   `"baseline"` row's own numbers (not the other six) would need to change, not the mechanism
   for the other switches.
2. **The `"innovation_gate_only"` vs `"baseline"` comparison the brief's table layout might
   invite is not directly meaningful** once `"baseline"`'s thresholds diverge from every other
   row's — I addressed this with an extra diagnostic replay and explicit disclosure, but a
   reader skimming just the seven-row table without reading `known_limitations` could still
   misread the delta.
3. The artifact is 530 KB (seven full `summarize_f9c` reports, including `by_distance_bin`/
   `by_fov_region` breakdowns, plus one diagnostic). I judged completeness (every field
   auditable without re-running) more valuable than a smaller headline-only file, but this is
   a judgment call.
4. I did not attempt to independently re-derive or cross-check `summarize_f9c`'s own metric
   definitions (RMSE, coverage, etc.) — Task 11 already reviewed and froze that module, and
   this task's absolute constraints forbid modifying it.
