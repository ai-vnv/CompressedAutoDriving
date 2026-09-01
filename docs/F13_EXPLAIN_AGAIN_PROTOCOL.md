# F13 Explain Again and Compression-Failure Protocol

## Frozen scope

F13 compares the immutable Original Belief-PPO (`02e898ce...c6250`) with the
selected A7 INT8 actor (`f8e4e3ae...b7cbc7e`). Both consume the same normalized
public 29D semantic representation and produce the same physical action
contract. MobileNet, YOLO, belief filters, scenario logic, and normalization
remain unchanged. F12 stays PASS for C4-only deployment; C0--C2 retention is
not reopened.

Historical status is immutable: F11 R002 LIMITED, R002b/R003/R004 PASS, R006
FAILED before intervention, and R007 BLOCKED. F13 is not a recovery of R006.

## Claims and evidence

| Claim | Minimum evidence | Failure interpretation |
|---|---|---|
| Semantic structure survived compression | Valid exact A7-QAT surrogate plus same-state/same-reference Distributional IG | Without a valid surrogate, semantic attribution is UNRESOLVED, not failed |
| Functional sensitivity survived | Direct deployed-INT8 and Original interventions on identical public states | Direction/magnitude drift is functional sensitivity drift |
| C4 behavior survived targeted stress | Same-seed Original/A7 paired C4 episodes | Both failing is base-policy limitation; A7-only repeated failure is compression-related |

## Run order and stop gates

1. **A — Integrity.** Verify both model hashes/contracts, search for the exact
   pre-conversion A7 QAT state, and calibrate new replay tolerances only on F11
   development public states.
2. **B — Attribution.** Run the frozen R004 Distributional IG protocol only if
   A passes. Never approximate a missing surrogate.
3. **C — Counterfactual.** Reuse R003 operators unchanged on the same frozen
   R004 public states, evaluating Original and deployed A7 directly.
4. Freeze an explanation/probe decision artifact.
5. **D — Stress.** Run the two pre-existing C4 sentinels and only functional-
   drift-triggered diagnostics under identical paired seeds. No parameter search.
6. Generate reports/figures, verify artifacts, run the active test suite, stop.

## Replay integrity (new F13 protocol)

Development calibration uses the F11 R002 public trace. The Original replay
tolerance is `max(5e-6, 2 × development maximum absolute replay error)`. A7 is
not compared to an old stored action; its device repeatability tolerance is
`max(1e-7, 2 × development maximum repeat error)`. Sham deltas must remain
within the corresponding frozen tolerance. These rules are fixed before R004
states are evaluated and do not modify historical R006.

## Attribution structure, if unblocked

Use the exact R004 states, reference draws, groups, phases, action units, and
equal 24-reference estimator. Structural preservation is preregistered as at
least 8/10 phase-action cells satisfying: group Spearman >=0.70, group-share
L1 <=0.40, same top group, and top-two Jaccard >=0.50. Overall top-group
preservation must be at least 0.80. Attribution equality is not required.

## Direct counterfactual analysis

Operators remain exactly: `pedestrian_absent`, `stop_absent`, `lane_centered`,
`lane_low_confidence`, `previous_action_neutral`, and `sham`. Primary pairs are
pedestrian/pedestrian-relevant, stop/stop-required, lane/lane-curve, and sham.
Preservation requires mean expected direction, paired direction agreement
>=0.90, normalized mean effect drift <=0.10, and normalized P95 effect drift
<=0.25. Action ranges are 0.4 m/s and 8 rad/s.
All factual and intervened vectors must remain inside the frozen PPO normalized
observation clip `[-3, 3]`; this is distinct from the `[-1, 1]` action bound.

## Explanation-guided and sentinel C4 probes

The fixed exploratory seeds are 179001--179004 and the reserved confirmatory
seeds are 179101--179104. Both policies run identical C4 conditions. No scenario
parameter optimization is allowed. Two pre-existing sentinels always run:

- S1: pedestrian-relevant action saturation;
- S2: stop-satisfied/post-stop action ordering and restart.

Functional probes are additionally triggered only if C shows direction
agreement below 0.90 or normalized mean effect drift above 0.10. A possible
closed-loop compression failure requires an Original-pass/A7-fail differential
on at least two confirmatory seeds. One occurrence is descriptive only.

## Three-axis classification

- Behavioral C4: `PRESERVED` or `DEGRADED`.
- Semantic explanation: `PRESERVED`, `PARTIALLY PRESERVED`, `SHIFTED`, or
  `UNRESOLVED`.
- Counterfactual sensitivity: `PRESERVED`, `PARTIALLY PRESERVED`, `SHIFTED`, or
  `INVALID`.

Overall PASS requires preserved C4 behavior and substantially preserved valid
semantic/counterfactual evidence. Missing valid A7 attribution yields at most
LIMITED even if behavior is preserved. Integrity violation or reproducible
A7-only C4 safety/control failure yields FAILED.

## Runtime boundary

Public 29D is the only policy/intervention input. Privileged simulator truth may
be read only after action for evaluation. BEV/world geometry is post-hoc only.
No training, model selection, repair, reward changes, perception changes, or
re-explanation method search is permitted.
