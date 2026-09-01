# F14 Explainability-Aware Compression Diagnostics Protocol

Status: **FROZEN BEFORE A1–A7 EXPLANATION**  
Date: 2026-08-15, Asia/Jakarta

## Purpose and immutable history

F14 is a new diagnostic extension. It asks how each frozen F12 compression
operation changes the actor mapping, semantic attribution, semantic intervention
sensitivity, action fidelity, and already-recorded closed-loop behavior. It does
not retrain, repair, replace, or select any policy.

Historical results remain immutable: F11 R002 is LIMITED, R002b/R003/R004 are
PASS, R006 is FAILED, and R007 is BLOCKED; F12 is PASS for C4-only deployment;
F13 is LIMITED and its A7 gradient attribution remains UNRESOLVED because the
exact QAT pre-conversion state was not saved. Exact Group Shapley is a new
model-agnostic F14 analysis, not a retroactive fix to F13.

## Verified frozen actors

| ID | Historical operation | Architecture | Precision | SHA256 |
|---|---|---|---|---|
| A0 | Original B-PPO actor | 29→256→256→2 | FP32 | `713d26d93488a17fae246b227e1de38f51501dc87a3d20ac6176036a8a8e64c5` |
| A1 | structured pruning | 29→64→64→2 | FP32 | `6e4ff154a209f44daf5f6ba45415ce47d0d4b60506aed28689e3795113da3904` |
| A2 | pruning + distillation | 29→64→64→2 | FP32 | `fd79dba7c2b4aa63bdcbe0e28f84847e15ea15baba33a8e927e6b6136b18a69f` |
| A3 | INT8 PTQ | 29→256→256→2 | INT8 | `b79c2b4c489826cf7ea4853d7104be1630356deea0611138c9e606d0740b179d` |
| A4 | fake-quant/QAT KD → INT8 | 29→256→256→2 | INT8 | `ad5ba8acd0c86ac315d6b65f9b6a419fa558c14cb405b41090e38eb0ab0f8dba` |
| A5 | pruning + INT8 PTQ | 29→64→64→2 | INT8 | `cf0c093c7523ac1188e0b94f7373277477eed8fac85ca3334498c3528bdf4358` |
| A6 | pruning + distillation + INT8 PTQ | 29→64→64→2 | INT8 | `773ffb9fb5e0cec24b21f892986de27960f32e64477024a41e88b7c953f09aa7` |
| A7 | pruning + distillation + QAT/KD + INT8 | 29→64→64→2 | INT8 | `f8e4e3ae5c43028d7b5d08e64c31d20dcce28153fb102ffac53a3b1c7b7cbc7e` |

These are the paths and identities in the frozen F12 ablation registry. The
Original training checkpoint remains
`02e898ce12d71f97016d50ed8a40574807e6d2fd995fc9f0dcd24f357f2c6250`;
A0 is F12's immutable actor-only extraction and must replay it within the
existing numerical contract. Every actor hash is checked before analysis.

## Explanation boundary and exact 29D partition

The boundary remains `front RGB → MobileNet/YOLO → measurement → belief →
normalized public 29D → actor → physical action`. F14 explains the actor on the
saved public 29D rows. It does not explain RGB, MobileNet, YOLO, world pose, or
privileged simulator state.

The six ordered players form an exact, non-overlapping partition:

- **Lane:** indices 1–5 and 8–9;
- **Ego:** indices 6–7;
- **StopLine:** index 10;
- **Pedestrian:** indices 11–19;
- **Stop:** indices 20–27;
- **PreviousAction:** indices 28–29.

Programmatic validation must prove that all 29 names occur exactly once and
that no privileged name is present.

## Primary method: phase-conditioned exact Group Shapley

There are six semantic players and therefore exactly 64 coalitions. For output
`k ∈ {v_cmd, omega_cmd}` and group `g`, F14 evaluates the standard exact Shapley
sum over all coalitions not containing `g`. Actor outputs are deterministic
physical actions after the frozen clipping/mapping.

For every factual row `x` and complete reference row `r`, each coalition copies
whole factual groups in the coalition and whole reference groups outside it.
All absent groups come from the same row `r`; group-wise independent reference
drawing is forbidden. Every coalition is checked for `[29]` shape, finiteness,
`[-3,3]` bounds, probability ranges, nonnegative standard deviations, and valid
stop-mode one-hot semantics. Schema validity is not claimed to establish full
physical on-manifold realism.

## Development diagnostic set and references

The only calibration/diagnostic source is the frozen F12 development dataset
(`178001–178008`, SHA256
`2cd50fae6bc4ac248f04563763c8c6968cf9429cfebe2e48ed0d19189018d47e`).
Selection is deterministic and public-only: for each of five frozen public
phases, allocate 100 states as evenly as possible across the eight seeds and
select evenly spaced steps within each seed. No action disagreement, outcome,
RGB appearance, explanation, or privileged field is used.

For each factual state, six fixed RNG draws select four complete same-phase
reference rows from four distinct other seeds. The fixed draw seeds are
`2026081511–2026081516`. The resulting 500 factual IDs, references, and all
coalition definitions are persisted before any ablation explanation.

## A0-only reference robustness calibration and threshold freeze

Only A0 is evaluated first. For each phase/action and every pair of the six
draw-level estimates, calculate six-group Spearman, group-share L1, top-group
agreement, top-two Jaccard, and signed agreement. Shapley local accuracy must
satisfy absolute residual ≤ `1e-5`.

Preservation thresholds are mechanical functions of A0 natural reference
variability, not A7 outcomes:

- minimum Spearman = max(`0.50`, A0 pairwise 5th percentile);
- maximum share L1 = min(`0.75`, A0 pairwise 95th percentile + `0.05`);
- minimum top-group agreement = max(`0.50`, A0 pairwise 5th percentile);
- minimum top-two Jaccard = max(`0.50`, A0 pairwise 5th percentile);
- minimum signed agreement = max(`0.50`, A0 pairwise 5th percentile).

The numeric values are written to an immutable threshold artifact before any
A1–A7 Shapley output is generated. If A0 reference robustness is inadequate,
all model-comparison attribution is LIMITED/UNRESOLVED and no reference tuning
is permitted.

The preregistered adequacy gate is: pairwise 5th-percentile Spearman ≥ `0.50`,
median signed agreement ≥ `0.50`, median top-group agreement ≥ `0.50`,
95th-percentile share L1 ≤ `0.75`, and maximum local-accuracy residual ≤
`1e-5`. All conditions must pass.

## Secondary method: semantic counterfactual policy-input intervention

F14 reuses the frozen R003 operators exactly: `pedestrian_absent`,
`stop_absent`, `lane_centered`, `lane_low_confidence`,
`previous_action_neutral`, and identity `sham`. It reports physical `delta-v`
and `delta-omega` for each actor on the same factual rows. These are functional
policy-input sensitivities, not real-world causal effects.

The sham tolerance is `1e-7`. Comparison rules are frozen from the prior
device-aware protocol: paired direction agreement ≥ `0.90`, normalized mean
effect drift ≤ `0.10`, and normalized P95 drift ≤ `0.25`. Primary cells are
pedestrian-absent/pedestrian-relevant/velocity,
stop-absent/stop-required/velocity, and lane-centered/lane-curve/yaw magnitude.

## A0–A7 same-state diagnosis

After threshold freeze, all eight actors receive the same 500 factual rows,
same 24 reference assignments, same 64 coalitions, normalization, and physical
mapping. Per actor F14 stores signed Shapley, absolute contribution, normalized
share, top group/top two, A0 rank/share/direction comparisons, all intervention
effects, and A0 functional drift.

Frozen F12 selection-split action fidelity, closed-loop behavior, and actor
efficiency are reused after hash validation; historical rollouts are not rerun.
Unavailable evidence is encoded as `UNRESOLVED`, never as zero.

## Pruning frontier, failed traces, and retention

Existing P/PD actors at widths 192, 128, 96, and 64 are eligible for a
descriptive lightweight diagnostic using the same 500 states and the first
four-reference draw. This cannot reopen F12 selection.

A failed-ablation event trace requires a saved per-step public 29D trajectory
that can be joined to an objective event chronology. Episode-summary-only CSV
is insufficient. If such evidence is absent, the trace is recorded as
`UNRESOLVED`; no historical simulator rerun is allowed.

Likewise, retention explanation runs only if compatible saved C0–C3 public 29D
rows already exist. Behavioral retention metrics remain visible regardless.
Missing public rows imply semantic retention explanation `UNRESOLVED`, not zero
drift and not a rerender request.

## Final A0 versus A7 re-explanation

After development results are frozen, use the existing R004 4,400 factual rows
and its exact `6 × 4` same-phase cross-seed reference observations. A0 and A7
receive identical coalition inputs. No locked seed is rerendered and no R004 IG
result is changed. This stage answers a new model-agnostic F14 question.

## Failure hierarchy and classifications

- L0: integrity/explanation validity failure;
- L1: semantic Group Shapley drift;
- L2: semantic intervention sensitivity drift;
- L3: action fidelity drift;
- L4: closed-loop control failure;
- L5: retention/generalization failure.

The hierarchy is descriptive and does not assert that L1 caused L4. Final axes
are reported separately: efficiency, C4 behavior, semantic attribution,
counterfactual sensitivity, and retention. A phase/action cell is structurally
preserved only when Spearman, L1, top-group, and top-two thresholds all pass;
8/10 cells are required for overall `PRESERVED`, 5–7 are `PARTIAL`, and fewer
are `SHIFTED`. Counterfactual status is `PRESERVED`, `PARTIAL`, `SHIFTED`, or
`INVALID` according to the three primary cells and sham gate.

## Artifact layout and compute estimate

All new outputs go under
`artifacts/f14_explainability_aware_compression_v1/`, separated into
`integrity`, `calibration`, `diagnostic`, `final`, and `figures`. Machine-readable
CSV/JSON/NPZ manifests bind config, actor, source-state, reference, and code
hashes.

Expected work is about 0.77 million A0 coalition rows for calibration, 6.14
million rows for A0–A7 diagnosis, under 1 million for the lightweight pruning
frontier, and 13.52 million rows for final A0/A7. This is CPU-only actor
inference on saved states; no simulator, GPU perception, training, or external
logging run is required.

## Fail-closed rules

Stop before scientific comparison on actor/hash ambiguity, inexact group
partition, invalid coalition, privileged leakage, A0 replay failure, nonzero
sham, inadequate reference robustness, or unsafe dataset provenance. Do not
change thresholds, phases, references, seeds, actors, or historical artifacts
after seeing ablation results.

