# SDD ledger — plan: docs/superpowers/plans/2026-08-08-f9c-robust-observation-belief-calibration.md

Mode: no-git (operator decision, 2026-08-08). `sdd-workspace` and
`review-package` both require `git rev-parse` and cannot run here, so this
workspace was created by hand and review packages come from `snap.py`:

    python .superpowers/sdd/<plan>/snap.py save    task<N>-base
    python .superpowers/sdd/<plan>/snap.py diff     task<N>-base review-task<N>.diff
    python .superpowers/sdd/<plan>/snap.py restore  task<N>-base   # rollback

Recovery map: this ledger plus `snapshots/` replace `git log`. A task with a
`Task <N>: complete` line is DONE — do not re-dispatch it.

Execution skills in use: run-experiment (Task 9, Task 11 runs) and
analyze-results (Task 9 Step 5, Task 11 metric interpretation), matching the
skills Codex used for F9a/F9b.

## Log

Baseline verified before Task 1: 108 passed, 0 failed, 0 skipped, 260 warnings.

Task 1: complete (snapshot task01-base -> current, spec OK, quality approved)
Task 1: adjudicated (controller error, not implementer) — reviewer raised as Important
  that `docs/superpowers/plans/...md` was modified without disclosure in the implementer
  report. That edit was MINE: I snapshotted task01-base, then added the WSL environment
  block to the plan, then dispatched. The implementer never touched the plan file.
  Ruling: no fix required; finding is a snapshot-timing artifact. Process corrected —
  snapshot is now taken immediately before each dispatch, after all controller edits.
Task 1: minor (deferred): `_scenario_spec` in f9c_protocol.py duplicates the private
  helper of the same name in the frozen f9_protocol.py. Defensible (source module is
  frozen) but it is logic duplication. Triage at final review.
Task 1: minor (deferred): brief's Consumes line names `load_scenario`, which
  f9c_protocol.py does not use. Harmless; relevant if a later task needs scenario_for.

Task 2: implementer escalated a genuine plan contradiction instead of forcing a pass —
  Step 1's test classified near-range by START offset (>=0.35) while Step 3 gave the only
  moving near scenario 0.30, making the test unsatisfiable. Escalated to human partner.
  RULING (human, 2026-08-08): the TEST was wrong; config keeps 0.30. approach_near_moving_ego
  starts near the medium bin and drives in at 0.20 m/s for 90 steps, so it TRAVERSES into
  near range; classifying by start offset excludes exactly the scenario built to sweep it.
  Canonical plan file updated to match the ruling.
Task 2: fix round 1/5 (1 addressed, 0 open — config reverted to 0.30, test predicate
  replaced with reaches_near(), IMPLEMENTATION_NOTES resynced)
Task 2: complete (snapshot task02-base -> current, spec OK, quality approved, 115/115)
Task 2: deferred to Task 9: whether ego_start_x_offset_m maps to true simulated distance_bin
  closely enough that reaches_near() is physically faithful rather than a plausible proxy.
  Task 9 is the first task that renders calibration seeds and can measure it. If the near
  bin comes up short there, scenario geometry must be adjusted BEFORE any 7101 frame.

Task 3: implementer reported a failing boundary test rather than loosening the gate's `<=`
  or padding the threshold. Correct instinct: under invariant I1 the gate boundary is
  calibration-sensitive, so widening it silently would have decoupled the gate from the
  lambda fitted to the NIS median.
  RULING (controller, no escalation): the TEST construction was at fault, not the module.
  `[THRESHOLD**0.5, 0.0]` squared back lands ~2e-15 above THRESHOLD. Replaced with an
  exactly-representable boundary (innovation [3.0, 0.0] over the identity -> NIS == 9.0)
  plus a companion assertion that a value just above the threshold is still rejected.
  Module left untouched. Canonical plan updated. Decided without escalation because the
  property under test is unchanged and the estimator is unaffected — only the arithmetic
  used to reach the boundary changed. Reviewer independently verified the new construction
  is exact in IEEE-754 rather than machine-lucky.
Task 3: fix round 1/5 (1 addressed, 0 open — boundary test replaced)
Task 3: complete (snapshot task03-base -> current, spec OK, quality approved, 123/123,
  no Critical/Important findings)

Task 3b: implementer self-disclosed a post-GREEN change (hardened the `identity` model to
  be an unconditional no-op inside correct(), not merely via identity()'s zeroed fields).
  Review found the branch was therefore exercised only vacuously — the existing test would
  have passed with the branch deleted.
  RULING (controller): keep the branch, test it. `model` is a config-driven string and
  from_config can build model="identity" carrying stray nonzero bias fields without going
  through the identity() classmethod, so the unconditional branch closes a real hole; what
  was missing was a test pinning the semantics down. Deleting it would have left the no-op
  guarantee dependent on the accident that one constructor zeroes the fields.
Task 3b: fix round 1/5 (1 addressed, 0 open — test added with nonzero bias fields asserting
  all four outputs; implementer verified by deleting the branch and observing 0.9459 vs
  0.900, exactly the un-applied bias, then restoring)
Task 3b: complete (snapshot task03b-base -> current, spec OK, quality approved, 133/133)
Task 3b: minor (deferred): from_config accepts model="identity" alongside stray nonzero bias
  fields and silently ignores them rather than rejecting. Tolerable since identity() is
  test-only and must never appear in an ablation config, but unverified either way.

PROCESS NOTE: snapshots are now taken both before dispatch AND after the implementer's
  first report, so fix rounds get a properly scoped diff. Task 3b's re-review had to be
  scoped by instruction instead, because only the pre-dispatch snapshot existed.

Task 4: complete (snapshot task04-base -> current, spec OK, quality approved, 140/140,
  no Critical/Important findings, no fix round needed)
Task 4: invariant I1 audited explicitly by the reviewer — the only covariance the module
  touches is the return value of the injected innovation_covariance_for(); no np.diag, no
  recomputed H P HT, no default/fallback matrix anywhere in module code. Invariant holds.
Task 4: reviewer hand-verified the +-pi wrap test is discriminating (naive subtraction gives
  NIS ~245,000 -> all_gated_out; wrapped gives 2.5, the asserted value) and that argmin runs
  only over gate-passing candidates, not argmin-then-check-winner.
Task 4: CARRY FORWARD TO TASK 8 (controller-resolved "cannot verify from diff"): the
  equivalence between MeasurementAssociator's (-confidence, bbox_key) ordering and
  select_single_duckie's (-confidence, x_min, y_min, x_max, y_max) holds ONLY if callers
  populate bbox_key in exactly (x_min_px, y_min_px, x_max_px, y_max_px) order. Task 8 builds
  CandidateMeasurement and must do so. A divergent order silently corrupts the duplicate-
  handling metric this gate reports, with no test failing. Must be stated in Task 8's dispatch.
Task 4: minor (deferred): AssociationConfig.initialization_rule is stored but never
  dispatched on and never validated against its one known value; a typo'd rule name would
  silently no-op forever.
Task 4: minor (deferred): the `nis is not None` guard when building gated_indices is
  unreachable in the temporal branch — harmless dead defensiveness.

Task 5: implementer was terminated mid-task by an API session limit (10pm Asia/Jakarta
  reset). Its final message read "Now implement", implying nothing was written.
  CONTROLLER VERIFICATION (do not trust the truncated report): checked disk directly —
  CandidateProjection, duckie_candidates, and _project_candidate ARE present in
  perception/f9_pipeline.py; the 4 new tests ARE present in tests/test_f9_pipeline.py;
  full suite re-run by controller = 144 passed, 0 failed (the expected count).
  Snapshot diff shows ZERO deleted lines — pure addition. observe()'s pre-existing
  selection path (selection.selected -> pedestrian/duplicate_selection/projection_error)
  is untouched, so Baseline A parity holds. Work is complete; only the report was missing.
  Implementer resumed solely to write its report from context, explicitly forbidden from
  touching code.
Task 5: controller observation for the reviewer: project_raw is now called TWICE for the
  selected detection — once via _project_candidate in the candidates loop, once directly
  in observe(). Projection is pure, so this is redundancy rather than a correctness bug,
  but it should be on the record.
Task 5: RULING on the double projection — KEEP IT. Implementer and reviewer independently
  reached the same conclusion: project_raw is pure geometry with no I/O, so the cost is a
  few numpy ops per frame, while sharing the candidate's projection with the frozen path
  would create exactly the coupling this task is forbidden to introduce. A future bug in
  the new candidate loop could otherwise leak silently into Baseline A. Revisit only after
  Baseline A has been recorded once as a verified reference.
Task 5: reviewer performed the Baseline-A parity audit by reading observe() in full rather
  than trusting the zero-deletion diff: the frozen path's expressions are byte-identical
  and `candidates` is never read by the frozen branches. Reviewer also independently re-ran
  the suite (144 passed) and confirmed the projection failure in test 3 uses a REAL
  projector with a near-horizon box (300,0,340,80), not a stub.
Task 5: complete (snapshot task05-base -> current, spec OK, quality approved, 144/144,
  no Critical/Important findings, no fix round needed)
Task 5: minor (deferred): duckie_detections re-filters by ObjectClass.DUCKIE independently
  of select_single_duckie's internal filter — deliberate duplication given the
  no-rerouting constraint.

Task 6: complete (snapshot task06-base -> current, spec OK, quality approved, 153/153,
  no Critical/Important findings, no fix round needed)
Task 6: reviewer verified EVERY formula term by term against the spec rather than trusting
  green tests — MS_within denominator (N-k), MS_between denominator (k-1), n_effective's
  use of (N - sum(n_i^2)/N), the zero-clamp being on the VARIANCE not the SD, per-seed
  (not global) centring in the nested pass, and the floor dividing the seed term by
  seed_count and the episode term by episode_count. No deviations found. This mattered
  because a wrong estimator here would silently corrupt the gate's central calibration
  claim with no other test catching it.
Task 6: RULING on the "between-group" vs "between-episode" docstring wording. The reviewer
  flagged the implementer's "between-group" as a literal deviation from the brief's quoted
  sentence. It is a CORRECTION, not a deviation, and the implementer's wording stands: the
  brief's "between-episode" was written when the plan still used a ONE-level estimator.
  After the estimator became two-level (seed + episode|seed), "between-episode" names only
  half of what the floor now measures. No change required.
Task 6: seed-SD recovery 0.01727 vs true 0.0155 (tolerance +-0.006) assessed as pure
  small-k sampling noise, not estimator bias: with k=8 groups MS_between has 7 df, giving
  a relative SE of ~sqrt(2/7)=53% on the variance scale (~25% on the SD scale), so an
  11.4% deviation sits well inside one SE. Formula transcription independently confirmed
  exact, so there is no coding-level bias to explain it.
Task 6: minor (deferred), CARRY TO TASK 10: floor fields are not validated non-negative in
  __post_init__. posterior_floor_from_components always returns a sqrt (non-negative), but
  Task 10 writes floor values into config by hand, where a negative could shrink reported
  uncertainty instead of increasing it. Task 10's dispatch must state this.
Task 6: minor (deferred): two extra ValueError guards (k<2, empty group) beyond the brief;
  harmless, prevent div-by-zero.

Task 7: controller found a gap in the BRIEF before dispatch and closed it: none of the
  brief's tests ever produced MID_FOV, so one of four observability classes would have gone
  unexercised and a boundary bug could have made it unreachable with nothing failing.
  Controller computed a state that lands mid-band (x_left=0.30, y_fwd=0.85 -> x_px=199.09,
  normalized 0.3778) and required an added test. Controller also pre-verified all four of
  the brief's states against the REAL projector so the implementer was not chasing bad
  values, and corrected the brief's implied import path (GroundPoint lives in
  domain.measurement, not domain.coordinates).
Task 7: reviewer confirmed binning parity with _fov_region in validate_f9_yolo_ekf.py:62-71
  including strict-vs-inclusive behaviour at the 1/3 and 2/3 boundaries. Had these
  disagreed, the fitted per-class detection probabilities would be applied to the wrong
  frames — silent and untested.
Task 7: reviewer traced update(False, detection_probability=0.0001,
  observation_informative=False) and confirmed the early return precedes validation (no
  raise, result equals the pure P_S/P_birth prediction), and that the default path is
  bit-identical to the pre-change body. test_pedestrian_ekf.py unmodified and green.
Task 7: fix round 1/5 IN PROGRESS — Important finding, escalated ABOVE the implementer's own
  "non-blocking" framing by the reviewer: outside_domain_miss_policy is compared by exact
  string equality, so any typo ("prediction-only") falls through to return True, silently
  making OUTSIDE_DOMAIN misses informative again and re-enabling the very belief-collapse
  behaviour this gate exists to fix. Invariant I3's guard rail had no guard.
  RULING: validate and reject at construction; do NOT add a second policy branch to make an
  enum look natural — one supported policy exists, and an unused branch that could violate a
  stated invariant is worse than no branch (same reasoning that removed the gate's
  DOWNWEIGHT mode).
  NOTE: this is a DIFFERENT severity from Task 4's deferred initialization_rule minor.
  Nothing reads initialization_rule, so a typo there is inert; this policy string IS read
  and a typo flips behaviour.
Task 7: fix round 1/5 (1 addressed, 0 open) — PREDICTION_ONLY_POLICY constant defined once,
  constructor rejects any other value with an error naming both received and supported,
  miss_is_informative now compares against the constant rather than a repeated literal, and
  no second policy branch was added. Implementer verified pre-fix that the hyphenated string
  was silently accepted ("DID NOT RAISE ValueError"), proving the hole was real.
Task 7: complete (snapshot task07-base -> current, spec OK, quality approved, 173/173,
  scoped re-review confirms ADDRESSED, no new breakage)

Task 8: the integration task. 184/184 green on first pass; spec OK, quality approved.
  Reviewer confirmed the I1 trap was AVOIDED: all three consumers (association loop, the
  coordinator's gate call, and PedestrianEKF.correct internally) look R up with the
  corrected candidate's MEASURED range, not the predicted range, and the coordinator's
  own test cross-checks last_ekf_diagnostics.innovation_covariance against its provider to
  1e-15. Controller's structural grep confirmed exactly one expression builds H P H^T + R.
  Bias-switch-off uses the F9b frozen constants; identity() never appears in the module.
Task 8: PLAN DEFECT found by the reviewer, not by controller or implementer — INVARIANT I7.
  [association].chi_square_gate and [innovation_gate].chi_square_threshold were BOTH set to
  9.21034037197618. Association discards candidates above its gate BEFORE the coordinator
  calls gate.evaluate(), so every surviving candidate trivially passed the gate: the
  innovation_gate switch had NO observable effect while temporal_association was on, and
  any contribution Task 12 attributed to it would have been spurious. Root cause was mine —
  I gave two components the same threshold despite them answering different questions.
  RULING (human partner, 2026-08-08): loosen association to chi-square 99.9% (13.815510557964274),
  keep the innovation gate at 99% (9.21034037197618). Association asks "which candidate is
  plausibly the track?" and should reject only wild outliers; the gate asks "do I trust
  these coordinates enough to correct with them?" and is the real accept/reject decision.
  Canonical plan updated with the new value AND with invariant I7.
Task 8: fix round 1/5 IN PROGRESS — config threshold change, a construction-time guard that
  raises when the association gate is not strictly looser than the innovation threshold, a
  test that the coordinator refuses the degenerate configuration, and a test feeding a
  candidate whose NIS falls BETWEEN the two thresholds to prove the gate is now genuinely
  reachable with association enabled.
Task 8: minor (deferred): the "detection but existence below initialization_threshold" reset
  branch is not exercised by any test (hand-computed only).
Task 8: minor (deferred): RobustStepRecord.nis is None when association's internal filter
  rejects, even though a real NIS exists in association.candidate_nis.
Task 8: CARRY TO TASK 9: add a field-parity test between belief.RobustObservationSwitches and
  evaluation.f9c_protocol.RobustObservationSwitches. The duplication is deliberate (importing
  evaluation from belief would invert the layering) but nothing currently guards against drift.
Task 8: CARRY TO TASK 9/10/11: no TOML loader for RobustObservationConfig exists yet;
  correctly out of scope for Task 8, but Task 9 is the first task that needs one.
Task 8: fix round 1/5 (1 addressed, 0 open) — association gate raised to 13.815510557964274,
  construction-time I7 guard (strict <=, message names both values and says "unreachable"),
  test that the degenerate equal-threshold config is refused, and a test feeding a candidate
  whose NIS = 11.48 lands BETWEEN the two thresholds (margin ~2.3 each side), proving the
  gate now genuinely rejects while association accepts. robust_config() helper updated so the
  other 11 tests no longer run in the degenerate configuration.
Task 8: complete (snapshot task08-base -> current, spec OK, quality approved, 186/186,
  scoped re-review confirms ADDRESSED, no new breakage)

=== PHASE CHANGE: Tasks 1-8 were pure code/test. Task 9 onward renders the simulator and
=== runs YOLO on GPU. Task 11 may be run EXACTLY ONCE, after Task 10 freezes the config.

Task 9: first calibration pass complete — 6,910 rows, 80 episodes, seeds 6101-6108.
Task 9: BOTH STRUCTURAL PREDICTIONS CONFIRMED. Range offset is seed-carried (seed_variance
  3.36e-4 vs episode 3.48e-5); bearing offset is episode-carried (episode 1.23e-4 vs seed
  2.70e-5). This is what justified the two-level nested estimator over the one-level form.
  sigma_floor_r = 0.02033 (predicted band 0.015-0.018, slightly above); sigma_floor_beta =
  0.01245 (band 0.012-0.016, inside).
Task 9: bias model = global_additive. Per-bin LOSO came in at -0.9%, far below the +10%
  pre-specified bar, so the per-bin temptation was correctly refused. b_r = -0.02987,
  b_beta = +0.00073.
Task 9: near-range verification (deferred from Task 2) PASSED — 1017 near-range eligible
  frames across 8 calibration seeds, scaling to ~508 for 4 final seeds, about 5x the
  pre-specified minimum of 100. reaches_near() is physically faithful, not just a proxy.
Task 9: FINDING — P_D^eff inversion is a RANGE CONFOUND, verified by controller cross-tab:
  center is 77.9% far-range while edge_fov is 1.9% far-range; within each range bin the
  detection rate is flat (0.946-1.000). Pooling by range gives near 0.9969 / medium 0.9925 /
  far 0.9717, monotone and physically sensible.
Task 9: DEEPER FINDING — no single-frame P_D conditioning can fix belief collapse. Those
  per-range values give miss likelihood ratios 0.003 / 0.0075 / 0.028 against F9b's global
  0.0233, so conditioning makes collapse FASTER in near and medium. The mis-specification is
  the INDEPENDENCE assumption, not the value of P_D: real misses are bursty (F9b mean run
  7.125 frames) and an independent-Bernoulli likelihood cannot express that at any P_D.
  RULING (human partner): add a miss-likelihood floor — INVARIANT I8. LR_used =
  max(LR_miss, LR_floor) with LR_floor = LR_nominal ** (1 / L_mean), L_mean measured on
  6101-6108. Rationale: a burst of correlated misses carries the evidence of ONE independent
  miss, not L_mean of them. Derived from data, never tuned to a retention target. Same
  species of correction as the posterior variance floor. Plan updated with I8.
Task 9: RULING (human partner) on the crashing scenario: fix geometry BEFORE the freeze.
  calibration_6102_approach_near_moving_ego terminated at frame 64/90 with invalid-pose.
  Task 11 runs once, so a crash on a final seed loses that episode permanently. Shorten the
  scenario, verify zero early terminations across all 8 calibration seeds, then re-run the
  full calibration so every frozen value comes from one internally consistent run.
Task 9: DISCLOSED LIMITATION (kept, not tuned away): calibration predicted_nis median is
  0.036 against a chi2_2 target of 1.386 — the innovation distribution is not chi-square, so
  a single lambda cannot fix a shape mismatch. Tail behaviour is nonetheless near nominal
  (0.63% of frames exceed the 99% gate threshold vs 1% expected), so the gate threshold
  remains usable. lambda_r = 10.125, lambda_beta = 1.0.
Task 9: fix round 1/5 (2 addressed, 0 open) — I8 miss-likelihood floor implemented via TDD
  with a no-op default (0.0) so every pre-existing test passes unchanged; scenario
  approach_near_moving_ego shortened 90->55 steps with zero early terminations across all
  8 seeds; full calibration re-run so every frozen value comes from one consistent run.
Task 9: L_mean = 4.0333 measured fresh on 6101-6108 (NOT F9b's 7.125). LR_nominal = 0.018858.
  LR_floor = 0.373625. Controller verified the arithmetic independently: floor**L_mean =
  0.018858 = LR_nominal exactly, so a run of typical length reproduces precisely the nominal
  single-miss evidence, as I8 intends.
Task 9: P_FA SCARE RESOLVED. The implementer's self-fit P_FA came out at 31.2% and it
  disclosed this rather than using it. Controller verified independently: of 209
  not-eligible-visible frames, 65 had a detection and ALL 65 carry a GT range — the
  pedestrian was physically present in every one. These are correct detections of a real
  pedestrian that the conservative eligible_visible rule excluded, not false alarms. Using
  the frozen F9b P_FA (measured properly from counterfactual renders with the Duckie hidden,
  a genuine negative) is the correct call.
Task 9: complete (snapshot task09-base -> current, spec OK, quality approved, 207/207,
  reviewer issued an explicit safe-to-freeze verdict on all five fitted quantities).
Task 9: ADJUDICATED (controller, no escalation): reviewer flagged as Important that
  fit_covariance_scales uses per-axis marginal fitting rather than the brief's literal joint
  NIS-median match, and that this was self-authorized. ACCEPTED — the joint form is one
  equation in two unknowns and is provably underdetermined; the implementer caught it via a
  seeded convergence test (1.0 vs 10.0 diverging). The only well-posed alternative, a single
  shared scalar, would force lambda_beta up to ~10 when the data says bearing needs no
  inflation at all (fitted 1.0). This is a correction of an under-specified instruction, not
  a choice between valid options, and it changes neither what evidence is collected nor any
  gate conclusion. Not escalated for that reason.
Task 9: DISCLOSED FOR THE GATE REPORT: reproducibility evidence is n=2 full runs under
  domain_randomization=true, agreeing to ~0.16% (lambda_r 9.9779 -> 9.9624). The frozen value
  is therefore one sample from a distribution, not a deterministic constant. Must be stated
  in Task 13's report rather than implied.
Task 9: minor (deferred to Task 13, which already owns that file): the brief asked for the
  seeds-vs-plan comparison in IMPLEMENTATION_NOTES.md; it lives in task-9-report.md instead.
Task 9: CARRY TO TASK 10 — THE FINAL-RUN VALUES ARE THE ONLY VALID ONES. The first pass
  produced lambda_r = 10.125; the re-run produced 9.96243043243885. Config must be filled
  from artifacts/f9c_calibration_metrics.json (the final run), never from the first report.
  Config currently still holds placeholders (parameters_frozen = false, 1.0/0.0), which is
  correct — Task 9 fits, Task 10 transcribes and freezes.
  Final-run values: lambda_r=9.96243043243885, lambda_beta=1.0,
  sigma_floor_r=0.02041790926900693, sigma_floor_beta=0.012546331734068323, rows=6656.

=== CONFIG FROZEN 2026-08-08 ===
=== config_sha256 359dc52020421c248bf6c26e036234191b2b97d24b505a66d85daa85b563704e
=== Nothing below this line may re-fit, re-tune, or edit the config. Seeds 7101-7104
=== may be rendered EXACTLY ONCE.

Task 10: complete. CONTROLLER RAN AN INDEPENDENT VERIFICATION rather than accepting the
  implementer's report (script kept at .superpowers/sdd/<plan>/verify_freeze.py, read-only).
  All checks pass:
    - every config value bit-for-bit equal to the FINAL-run artifact
      b_r=-0.02986607430110723  b_beta=0.0012336629252072933
      lambda_r=9.96243043243885 lambda_beta=1.0
      sigma_floor_r=0.02041790926900693 sigma_floor_beta=0.012546331734068323
      miss_likelihood_floor=0.37362469458201386
    - the string "10.125" (first-pass lambda_r) appears NOWHERE in the config
    - all three parameters_frozen flags are True
    - [ekf] block and prior/survival/birth identical to oracle_ekf_v1.toml
    - invariant I7 holds: association 13.815510557964274 > gate 9.21034037197618
    - recorded config_sha256 matches the live file
    - final_evaluation_seeds_not_yet_rendered = true
Task 10: non-negative validation added to CovarianceCalibration floors (carried from Task 6),
  closing that hole at the exact moment it first became reachable — this task is the first
  that writes floor values by hand rather than deriving them from a sqrt.
Task 10: two pre-existing tests were updated in place because they asserted the PRE-freeze
  placeholder values (parameters_frozen=false, 1.0/0.0). Expected and legitimate; the task
  reviewer must confirm they were tightened to the frozen values rather than loosened.
Task 10: implementer notes this sandbox's `$?` capture between `;`-separated bash statements
  is unreliable; it worked around with `&&`/`||`. Relevant to later tasks.
Task 10: task review PASSED on all five items the controller had NOT self-checked. Both
  modified pre-existing tests were TIGHTENED (now assert the exact frozen values
  9.96243043243885 and 0.37362469458201386) rather than loosened — the Critical risk at a
  freeze boundary. frozen_config.json complete, nothing null. Non-negative floor validation
  covers BOTH floor fields. verify_f9c_artifacts.py is read-only and uses a distinct
  SchemaSkip so an absent Task 11 artifact reports SKIP, never a silent PASS — verified by
  the implementer with an injected hash mismatch (FAIL) and a mismatched-schema fixture
  (SKIP). f9c_calibration.py change is docstring-only, so the link between the frozen values
  and the code that produced them is intact.
Task 10: complete (spec OK, quality approved, 209/209, no Critical/Important findings).

PROCESS NOTE: snap.py deliberately excludes artifacts/ (large CSVs, PNGs, npz). For
  artifact-producing tasks (11, 12) reviewers must be pointed at the live artifact files in
  the repo instead — the Task 10 reviewer did exactly that successfully.

=== TASK 11: THE ONCE-ONLY RENDER HAPPENED AND SUCCEEDED. 2026-08-09.
=== seeds 7101-7104, 40/40 episodes, 3328 frames, zero crashes, zero early terminations.
=== runtime cache SHA256 fe425c55aadd45af88d072c256010f5bddcbb82d952669e8fa988bd70722526d
=== evaluation truth SHA256 26663ebb85ebd2ed9fd00ffc0903679b298b9333ef3e10f58ff253d7fd1e2ae9
=== THE FINAL SEEDS MUST NEVER BE RENDERED AGAIN.

Task 11: reported BLOCKED after the render. summarize_f9c crashed on real data, STRICTLY
  AFTER collect_final_rows had returned all 3328 rows and after the cache and truth files
  were written. CSV and metrics JSON were never written; no partial or misleading artifact
  exists on disk.
Task 11: root cause — evaluate_f9c_robust_belief.py's _optional() helper writes "" for a
  missing value (the codebase-wide CSV convention), but robustness_metrics checked
  `is not None`, which is True for "", then called float("") and raised. Every synthetic
  test had constructed the field as a real None. Fixed to `not in (None, "")` at both call
  sites, with a regression test reproducing the exact sentinel. Implementer disclosed the
  test-coverage gap rather than papering over it, and audited the rest of the module for the
  same pattern (only those two sites affected).
Task 11: CRITICAL FOR INTEGRITY — the implementer never saw ANY final-seed metric. The crash
  preceded all metric computation, so the bug fix carries zero risk of having been shaped by
  results. Config SHA verified unchanged both immediately before launch and after the crash.
Task 11: the implementer declined to reconstruct the artifacts unilaterally, judging that
  writing new untested replay code to finalize a one-shot gate artifact without review
  crossed the line the brief drew. That was the correct call.
  RULING (human partner, 2026-08-09): reconstruct by REPLAYING THE CACHE, and refactor so
  collect_final_rows and the replay path call ONE shared build_row function. Rationale:
  replay touches no simulator, no detector and no GPU, and is deterministic given identical
  inputs — it is exactly what invariant I4/I5's immutable on-disk cache was designed to make
  possible. Sharing the row-builder makes fidelity STRUCTURAL rather than hoped-for, which
  answers the implementer's own concern about tie-breaking and duplicate_selection drift.
  Re-rendering was rejected because it would be a SECOND render of the final seeds — the
  precise thing the once-only rule forbids — and under domain_randomization it could produce
  different frames, leaving the existing cache and the new results describing different data.
Task 11: fix round 1/5 — replay reconstruction complete in 1.79s for 3328 frames, no
  simulator/detector/GPU. Shared build_row extracted and used by BOTH collect_final_rows and
  replay_from_cache; reviewer verified the extraction was faithful (not rewritten) and that
  the equivalence test genuinely cross-feeds the two selection paths rather than calling the
  shared function twice. Provenance recorded INSIDE f9c_belief_metrics.json.

=== TASK 11 HEADLINE RESULTS (seeds 7101-7104, 3328 frames) ===
  Support check SATISFIED: near=616 medium=671 far=1887 edge_fov=543 (minima 100/200/200/50)
  Range RMSE      Baseline A 0.02580 -> Robust B 0.02024  (21.5% lower)
  coverage_68     0.247 -> 0.852   (pre-registered band [0.60,0.76] -- OVERSHOOTS)
  coverage_95     0.388 -> 0.988   (band [0.90,0.98] -- just above)
  std_over_rmse   0.19  -> 1.28    (anti-inflation guard <=1.5 -- PASSES)
  In-domain miss retention  18.2% (10/55) -> 61.8% (34/55)  (criterion >=60% -- PASSES)
  Gated-rejection retention 100% (23/23)  -- invariant I2's payoff, exactly as designed
  Outlier impact n=9: Baseline 0.02179 vs Robust 0.03455 -- Robust WORSE, n far too small

Task 11: CONTROLLER'S OWN HYPOTHESIS REFUTED, recorded so it is not repeated. I proposed the
  coverage overshoot was heavy-tail driven (sigma inflated by tails while the bulk stays
  concentrated). The reviewer tested it on n=3214: z = error/sigma has std ~0.79 and excess
  kurtosis ~ -0.43, i.e. slightly LIGHTER-tailed than Gaussian. The overshoot is a roughly
  uniform 25-30% over-sized posterior sigma, NOT a tail effect. The remaining hypothesis
  (sigma_floor calibrated on 6101-6108 transfers wide to 7101-7104) is consistent with a
  uniform over-sizing but is NOT tested by this run, and tau_seed from only 8 seeds is itself
  noisy. Must be labelled a hypothesis, never a conclusion.
Task 11: IMPORTANT FINDING, CARRY TO TASKS 12 AND 13 — conditional_detection is inert through
  its per-class probabilities on BOTH branches. Miss branch: the I8 floor 0.37362 dominates
  every fitted value (center LR 0.05099, mid 0.01987, edge 0.00271). Detected branch: not
  floored, but existence saturates >=0.918 across all 3122 detected rows regardless of class.
  The component therefore contributes ONLY via the I3 in/out-domain routing. A Task 12
  ablation reader would otherwise expect toggling it to reflect the fitted per-class values.
  Also: 39/42 track initializations classify outside_domain because the zero-prior default
  state has y_forward = 0, projecting behind the camera.
Task 11: this also DEFUSES the camera-calibration fidelity gap. domain_rand does perturb
  camera geometry (simulator.py:734-736 scales cam_height +-8% and cam_angle +-20%), and the
  replay used the nominal camera for observability classification. But since all three
  in-domain classes are inert, only an in-domain <-> OUTSIDE_DOMAIN flip could change
  behaviour, not a CENTER<->MID<->EDGE flip.
Task 11: fix round 2/5 (3 addressed, 0 open) — known_limitations block added to
  f9c_belief_metrics.json with four subkeys. Every number computed FRESH from rows by a new
  known_limitations() function, never hand-typed into JSON; kurtosis cross-checked against
  scipy.stats.kurtosis; the metrics sub-object verified byte-identical before/after, twice.
  Controller confirmed independently: block present, all four subkeys, headline metrics
  unchanged (rmse 0.020242118890580148, cov68 0.8522090852520224, cov95 0.9884878655880522,
  support satisfied).
Task 11: complete (spec OK, quality approved, 239/239, verifier exit 0 with 12 PASS / 1 SKIP,
  config SHA unchanged at 359dc520...63704e).
Task 11: minor (deferred to Task 13): outlier_impact n=9 shows Robust B WORSE than Baseline A
  (0.03455 vs 0.02179). Touches PASS criterion 5 (gross localization outliers have reduced
  influence). n is far too small to conclude anything — Task 13 must report it as INSUFFICIENT
  EVIDENCE, neither as a pass nor as a failure.

=== TASK 12 ABLATION RESULTS (same cache fe425c55..., 3328 frames, zero inference) ===
  row                          range RMSE  cov_68   in-domain retention
  baseline                       0.02580   0.2470   0.1818
  + bias refit only              0.02407   0.0752   0.1818
  + innovation gate only         0.02987   0.2588   0.1818
  + temporal association only    0.03776   0.2566   0.1818
  + covariance calibration only  0.02989   0.6403   0.1818
  + conditional detection only   0.02569   0.2589   0.6182
  all combined                   0.02024   0.8522   0.6182
  Both endpoints verified equal to their headline systems (reviewer re-derived from the raw
  artifact independently, not from the report).

Task 12: HEADLINE SCIENTIFIC FINDING — THE COMPONENTS ARE NOT ADDITIVELY SEPARABLE.
  innovation_gate_only (0.02987) and temporal_association_only (0.03776) are each WORSE on
  range RMSE than baseline (0.02580), while all_combined (0.02024) is the best row. Per-row
  deltas must NOT be read as individual contributions. The plan's one-component-at-a-time
  ablation design assumed a separability the data refutes.
Task 12: STRUCTURAL FINDING — the innovation gate is INERT without temporal association.
  innovation_gate_only is metrically IDENTICAL to an all-off frozen-threshold diagnostic
  (rmse 0.029874031173970667 in both, exact). Reason, verified from code: with
  temporal_association off, update() forces predicted_measurement=None, association returns
  mode="initialization", and the gate branch only runs when mode != "initialization" — there
  is no innovation to threshold. So the "+ innovation gate only" row measures nothing about
  the gate. Combined with the Task 8 I7 finding (gate unreachable when thresholds were equal),
  the gate can only ever be exercised in combination with association.
Task 12: PLAN GAP found and resolved by the implementer — RobustPedestrianBeliefUpdater's
  track-lifecycle (delete/init thresholds) is gated by NONE of the five switches, and
  Baseline A has no equivalent concept, so "all switches off" still deleted tracks (114-frame
  divergence, range.count 3267 vs 3153). Fix: on the baseline row ONLY, override
  initialization_threshold=1e-9 and delete_threshold=0.0.
  ADJUDICATED (controller, no escalation): this is EXACT, not tuning. delete_threshold=0.0
  makes deletion structurally unreachable because existence probability is clamped to [0,1]
  (existence_filter.py:124), and initialization_threshold=1e-9 only affects the first accepted
  candidate whose posterior is ~0.999. Reviewer independently verified the clamping and traced
  track_was_active stickiness. It emulates Baseline A's ABSENCE of track lifecycle, which is
  what the plan's own "must equal Baseline A exactly" requirement demands; the plan simply
  never anticipated the asymmetry. Implementer also added a separate
  all_switches_off_frozen_thresholds diagnostic so the gate comparison stays clean.
Task 12: fix round 1/5 (3 addressed, 0 open) — endpoint test now loops all four sub-dicts for
  BOTH endpoints; the 1e-9 comment is scoped to the frozen config and states the computed
  posterior 0.9992019795087668 with an explicit disclaimer that it is not a general claim;
  ablation_known_limitations.components_are_not_additively_separable carries all four RMSE
  values, boolean flags, and the association hypothesis marked UNTESTED. Frozen config SHA
  unchanged. Scoped re-review confirms no new breakage.
Task 12: complete (spec OK, quality approved, 245/245, both endpoints verified equal to their
  headline systems across ekf/track_continuity/nis/miss_sequence).
Task 12: minor (deferred): brief's stale "expect 192 passed" template constant vs actual 245.

Task 13: leakage tests + gate report + classification. 251 passed.
Task 13: CONTROLLER ERROR, corrected. My dispatch said "any failure here is a real leak — fix
  the source module, never the test." That rule is right when the test catches real code and
  WRONG when the test cannot tell code from prose. The scan was a raw substring match, so it
  tripped on four comments/docstrings that DOCUMENT the absence of privileged access — exactly
  the passages explaining the runtime/privileged boundary, the central design property of this
  gate. The implementer obediently reworded them, losing that documentation.
  FIX: restored all four verbatim; rewrote the scan to be AST-based (identifiers, attributes,
  import targets, non-docstring string literals; comments excluded by construction, docstrings
  explicitly skipped); added test_the_leakage_scan_reads_code_not_prose to pin the behaviour.
  The implementer went further unprompted and replaced the evaluator-ordering test's raw
  text .index() with real ast.Call line numbers — a genuine improvement, since the text form
  could have matched a comment.
Task 13: CONTROLLER MAPPING CORRECTION. The implementer honestly flagged that no verbatim
  17-criteria list exists in the plan and that it had reconstructed its own. Mapping back to
  the operator's ORIGINAL prompt section 41:
    15 MET, 2 INSUFFICIENT EVIDENCE (criterion 5 outlier influence, criterion 10 long-absence
    decay), 0 NOT MET.
  The implementer's "3 NOT MET" were misses against the COVERAGE BANDS I ADDED to the plan,
  not against the operator's criteria. Operator criterion 6 ("no longer severely
  overconfident") and 7 ("coverage materially improves") are both clearly MET.
  Note for the classification decision: those bands were my instrument for operationalising
  operator criterion 8 ("not achieved by absurdly inflating uncertainty"). That purpose is
  independently guarded by std_over_rmse <= 1.5, which passes at 1.28. The bands are
  symmetric while the risk is asymmetric — an over-conservative posterior is safe, an
  overconfident one is dangerous. The instrument's shape did not match the risk it guarded.
Task 13: criteria 11 and 12 verified by controller from the artifact: recovery = 1.0 frames
  for both systems; false_track_initializations = 0, track_deletions = 8.
Task 13: criterion 10 is unreachable in this data — the longest natural miss run is 10 frames,
  so the ">= 20 consecutive in-domain misses" checkpoint was never exercised.
Task 13: complete (251 passed, leakage tests clean, gate report written, classification
  RECOMMENDED as LIMITED pending the human partner's decision).

=== FINAL WHOLE-BRANCH REVIEW — CLEAN ===
Controller ran the mechanical checks itself (verify_consistency.py, kept alongside this
  ledger): every headline number quoted in GATES.md / README.md / IMPLEMENTATION_NOTES.md
  agrees with the artifacts; both ablation endpoints match their headline systems to full
  float precision; the ablation's cache hash matches the final run's. CONSISTENT.
Final reviewer verdict: SOUND AS A RESEARCH ARTIFACT. All seven required weaknesses are
  stated as prominently as the strengths (reviewer located each one specifically); nothing
  overstated; all 11 deferred minors triage as CARRY-FORWARD, none blocking, none requiring
  a re-fit or re-render. One finding: the deferred-minors table listed 10 of 11.
Final fix (one dispatch): Task 5's minor added to the table as carry-forward with its reason
  stated; the implementer additionally split the table into 11 ledger-tagged minors vs 4
  disclosed limitations so the count is unambiguous. 251 passed, config hash unchanged.

=== CLASSIFICATION TALLY, FOR THE HUMAN PARTNER'S DECISION ===
Against the operator's ORIGINAL 17 criteria (prompt section 41):
  15 MET, 2 INSUFFICIENT EVIDENCE (criterion 5 outlier-influence reduction; criterion 10
  long-absence decay, never exercised — longest natural miss run was 10 frames), 0 NOT MET.
The three "NOT MET" in the implementer's own tally are misses against the COVERAGE BANDS THE
  CONTROLLER ADDED to the plan, not against the operator's criteria. Operator criteria 6 and
  7 are both clearly met. Those bands operationalised operator criterion 8, whose purpose is
  independently guarded by std_over_rmse <= 1.5 (passes at 1.28). The bands are symmetric
  while the risk is asymmetric.
Both the final reviewer and the controller recommend LIMITED, for DIFFERENT reasons: the
  reviewer weights the literal band miss; the controller weights criteria 5 and 10 having no
  evidence, since outlier robustness and long-absence behaviour are exactly the conditions a
  stop policy would depend on. Final call belongs to the human partner.

=== WORKSPACE DISPOSITION ===
The skill's final step deletes this workspace because "the git history is the record now."
  THAT DOES NOT APPLY HERE — this project is not a git repository, so this ledger and the 14
  task reports ARE the only record of how the gate was executed. Deleting them would destroy
  it. Preserved to docs/superpowers/f9c-execution-record/ instead. Only snapshots/ (20 MB of
  duplicated source trees whose per-task rollback purpose is spent) was removed.


Task 12: fix round 1/5 detail — (a) widen the endpoint test from the ekf sub-dict alone
  to all four sub-dicts, since the report claimed verification the committed test did not do;
  (b) scope the initialization_threshold comment to the FROZEN config (it overclaimed
  generality: a config with survival/birth near zero would break the argument); (c) state in
  the artifact that the components are not additively separable, and record the unexplained
  association-alone degradation as an open question with the I1-coupling hypothesis LABELLED
  as a hypothesis.

