# F15 — Cross-Curriculum Compression Failure Localization and Recovery Protocol

## Freeze boundary and scope

This protocol was written before any F15 seed was rendered or any F15 result was
inspected. F15 is a new experiment namespace. It does not modify, regenerate, repair,
or reinterpret F10–F14 evidence. It does not run attribution, Integrated Gradients,
Group Shapley, or model selection based on explanations.

The scientific order is fixed:

1. verify frozen provenance and the public 29D/action contract;
2. localize retention collapse across historical A0–A7 and the historical pruning
   frontier on new paired seeds;
3. measure same-state action fidelity and preserve objective failure telemetry;
4. freeze `failure_localization_decision.json`;
5. test the smallest recovery intervention first;
6. freeze one candidate before once-only final holdout;
7. evaluate Original versus that candidate on all C0–C4 holdout curricula.

No recovery training may begin before steps 1–4 are complete.

## Repository audit and immutable evidence

The active repository contains `FORMULATION.md`, `GATES.md`,
`IMPLEMENTATION_NOTES.md`, F10 curriculum/configuration code and reports, and the F12
compression reports/artifacts. A root `EXPERIMENT_PLAN.md` is not present; the only
file with that name is an archived F11 reset plan under `refine-logs/`, so it is not an
authoritative F15 plan.

The immutable Original Belief-PPO is:

- checkpoint: `artifacts/f10_ppo_visual_objects_v30/c4/ppo_selected.pt`;
- SHA256: `02e898ce12d71f97016d50ed8a40574807e6d2fd995fc9f0dcd24f357f2c6250`;
- actor: `29 → 256 → 256 → 2`, with `Tanh` hidden activations;
- deterministic output mapping: clip normalized means to `[-1,1]`, then
  `v_cmd=(u_v+1)×0.2 m/s` and `omega_cmd=4u_omega rad/s`.

The F12 A0–A7 registry is
`artifacts/f12_belief_ppo_compression_v1/final/ablation_registry.json` (SHA256
`4160df2cff9162ce89288aa3a405e6f2d8ecf0578e4aa9f365db71e4acdcb91b`).
The pruning-frontier registry is
`artifacts/f12_belief_ppo_compression_v1/pruning/registry.json` (SHA256
`5b1925e3ae73302b5371c7a05a1ea593b2b231eefa5167bd15fb5a87c9dcb910`).
Every registered A0–A7, P192/P128/P96/P64, and PD192/PD128/PD96/PD64 actor exists and
matched its registered SHA256 in the pre-freeze read-only audit.

Historical F12 retention is motivation, not F15 evidence: on its two C0–C3 retention
seeds the selected A7 completed C3 and the final C4 set, but completed 0/2 in C0, C1,
and C2. F15 does not rerun or alter those rows.

## Curriculum definitions

F15 resolves all environments from `configs/f10_ppo_visual_objects_v30.toml` through
the existing `load_ppo_curriculum_protocol` and `PPOCurriculumEnvironment` code.

| Curriculum | Resolved environment | Public task content | Horizon |
|---|---|---|---:|
| C0 | `small_loop` | lane driving, no pedestrian or stop task | 1900 |
| C1 | `experiment_loop` | experiment-loop lane generalization | 2700 |
| C2 | `experiment_loop_duckie_crossing` | lane + crossing pedestrian | 2700 |
| C3 | `experiment_loop_stop_only` | lane + stop task | 2700 |
| C4 | `experiment_loop_combined` | lane + pedestrian + stop | 4200 |

Scenario paths, detector/belief configuration, route data, and map settings remain the
resolved F10 values; F15 introduces no environment modification.

## Public observation and privileged boundary

Every actor consumes the same normalized public 29D vector in the exact frozen order.
The 29 entries are produced by the unchanged visual perception/belief pipeline and
partition into Lane, Ego, StopLine, Pedestrian, Stop, and PreviousAction fields.
F15 saves both physical public fields and normalized inputs.

Simulator truth is never an actor, distillation, or same-state fidelity input. If
recorded, it is written to a separate evaluation-only structure after the action
boundary. The primary telemetry explicitly distinguishes `policy_public` from
`evaluation_only` fields.

## New disjoint seed allocation

The `180xxx` blocks below were absent from active configs, reports, scripts, tests, and
seed metadata in the pre-freeze audit. The `179xxx` block was rejected because F13 had
already allocated it.

| Purpose | Seeds | Access rule |
|---|---|---|
| failure localization | 180001–180008 | same eight seeds for every model in each curriculum |
| recovery dataset | 180101–180108 | public A0 trajectories only; training/rehearsal |
| recovery selection | 180201–180208 | paired development selection, never final evidence |
| once-only final holdout | 180301–180308 | unopened until candidate claim exists |

The same integer seed is intentionally used across C0–C4 because each curriculum owns
a separate resolved environment/scenario; comparisons within a curriculum are paired.

## Frozen definition of competence and collapse

A compressed candidate receives `PASS` for a curriculum only when:

1. Original A0 passes that curriculum's applicable absolute F10 gate on the paired F15
   seeds;
2. the candidate also passes the absolute F10 gate; and
3. the candidate stays within the frozen F12 relative behavior margins against A0.

If A0 fails the absolute gate, compression retention is `UNRESOLVED` for that
curriculum. This prevents a base-policy limitation from being labeled a compression
failure. `FAIL` therefore means a tested compression-associated loss relative to an A0
competence demonstrated on the same seed block.

Absolute thresholds are copied verbatim into the F15 config from the resolved F10 v30
protocol. Relative margins are copied from F12, with invalid-pose increase explicitly
bounded at 0.125 because invalid pose is a required F15 collapse outcome. No threshold
may change after F15 results are opened.

`First collapse stage` is the earliest actual construction transition for which the
predecessor is `PASS` and the successor is `FAIL`. The final historical path is treated
as A0 Original → A1 Pruning Only → A2 Pruning + KD → A7 Final INT8. A6 is the parallel
A2→PTQ diagnostic; A3/A4 and A5 are parallel branches and are never presented as a
single fictitious linear training history.

## Localization matrix and pruning-width diagnosis

All immutable A0–A7 actors run on localization seeds for C0–C4. Historical pruning-only
and pruning+KD checkpoints at 192, 128, 96, and 64 run on exactly the same seeds.
Identical actor hashes are evaluated once and reused, not described as independent
replicates.

Primary closed-loop metrics are completion, progress, collision, unsafe episode,
minimum pedestrian clearance, lane failure, invalid pose, stop violation, stop
completion, restart, timeout, mean velocity, mean absolute yaw command, stationary
fraction, and termination reason.

## Same-state action fidelity

For each curriculum, only A0 closed-loop trajectories define the offline factual input
set. The exact normalized public 29D rows are replayed through every actor. This isolates
actor mapping from trajectory feedback. Per action and curriculum F15 reports MAE,
RMSE, median, P95, P99, maximum absolute error, signed bias, Pearson, Spearman,
saturation disagreement, and omega sign disagreement above the frozen 0.2 rad/s
deadband. The F12 numerical fidelity thresholds are reused without modification.

Same-state fidelity is not an explanation method and is reported separately from
closed-loop retention.

## Objective failure evidence

Every localization rollout writes provenance, physical and normalized public 29D,
actor normalized/physical outputs, public driving context, and objective event flags.
Evaluation-only truth, when available, is separate.

For each failing model×curriculum cell, the representative event is frozen as the first
objective failure in the lowest failing seed. The fixed event priority and fixed
90-step pre/45-step post window are in config. Visual output is produced by a
descriptive deterministic same-seed replay after selection; it is checked against the
primary telemetry and is not a new statistical replicate. If exact exogenous equality
cannot be established it is labeled `same-seed paired rollout`, never a causal paired
trajectory.

## Failure-localization freeze

Before training, `failure_localization_decision.json` must exist and hash-bind:

- the cross-curriculum matrix;
- pruning-width matrix;
- same-state fidelity artifact;
- failure-event registry;
- first collapse per curriculum;
- what the evidence does and does not support about width, rehearsal coverage, PTQ,
  QAT, and ordering.

Unavailable evidence is encoded `UNRESOLVED`, never as zero drift or PASS.

## Recovery sequence

The first and smallest intervention keeps the Original teacher, historical 64×64
survivor indices, architecture, Smooth-L1 physical-action loss normalized by
`[0.4,8.0]`, Adam optimizer family, 80 epochs, batch size 512, learning rate 0.001,
weight decay 1e-6, deterministic teacher targets, and state-independent log-std rule.
Only public rehearsal coverage changes: training rows are balanced first across C0–C4
and then by supported public phase within curriculum. No simulator GT action, critic
target, reward optimization, or PPO retraining is used.

The frozen decision tree is:

1. train/evaluate 64×64 multi-curriculum KD in FP32;
2. if it fails any curriculum/fidelity/safety gate, repeat the identical recovery at
   96, then 128, then 192; select the smallest passing width;
3. once FP32 passes all gates, run PTQ calibrated on balanced C0–C4 rows;
4. if PTQ introduces a gate failure, run multi-curriculum fake-quantized QAT+KD and
   convert through the same x86 static INT8 backend;
5. test progressive prune–distill only if direct target-width recovery remains
   insufficient. Each progressive stage must pass before proceeding.

The quantization format remains qint8 per-channel symmetric weights, quint8 per-tensor
affine activations, static x86 quantized Linear operations, and float Tanh boundaries.

## Final selection and once-only holdout

The smallest candidate that passes all five absolute+relative retention gates, frozen
same-state action-fidelity gates, and safety checks is selected. `final_candidate.json`
must record its path, SHA256, architecture, precision, training dataset manifest,
quantization provenance, and rationale before final access.

The holdout runner must fail closed unless both the localization decision and final
candidate exist and match config hashes. It writes a pre-access claim before opening
any 180301–180308 environment. Only A0 and the frozen candidate run. There is no
replacement or rerun after access. Failure is reported as failure.

## Efficiency protocol

The final actor is compared with Original using parameter count, serialized bytes,
logical parameter memory, MACs where supported, and one-thread CPU batch-1 latency
median/P95/P99 and throughput (1,000 warmups; 10,000 timed iterations; five repeats).
Actor-only speedup is never called end-to-end visuomotor speedup.

## Estimated compute and storage

Localization contains 320 A0–A7 episodes (8 actors×5 curricula×8 seeds). The unique
pruning-frontier extension adds 240 episodes after hash-based reuse of A0/P64/PD64.
At maximum configured horizons this is at most about 1.7 million simulator steps before
recovery. Recovery adds 40 dataset episodes and at least 80 selection episodes per
candidate comparison; final adds 80 episodes. Actual wall time depends primarily on
unchanged MobileNet/YOLO inference and Mesa rendering. The eight-seed target is not
silently reduced. A post-freeze smoke timing may refine scheduling, not gates or seeds.

Telemetry is compressed per episode; full video is restricted to every failing
representative and one deterministic success per model×curriculum to control storage.

## Stop rules

Stop before scientific evaluation if any actor hash, registry hash, policy config hash,
29D contract, action map, INT8 invocation, or public/privileged separation fails.
Stop before recovery if localization artifacts are incomplete or the localization
decision is absent. Stop before final if candidate provenance or once-only claim is
missing. Never recover by changing seeds, thresholds, curriculum definitions, pruning
survivors, teacher, or final candidate after results are seen.

## Artifact namespace

All outputs are new under `artifacts/f15_cross_curriculum_recovery_v1/`, with separate
`integrity/`, `localization/`, `telemetry/`, `failure_traces/`, `recovery/`, `final/`,
`figures/`, and `logs/` directories. Required JSON/CSV manifests and reports are those
listed in the user-approved F15 specification. Existing F10–F14 paths are read-only.

## Documented preflight command

From repository root, before localization:

```bash
export PYTHONHASHSEED=0
export PYTHONPATH=src:/home/pannntastic/aivnv/duckie/src
export DUCKIETOWN_HEADLESS=1
export CUDA_VISIBLE_DEVICES=0
/home/pannntastic/aivnv/duckie/.venv/bin/python \
  experiments/run_f15_cross_curriculum_recovery.py verify \
  --config configs/f15_cross_curriculum_recovery_v1.toml
```

The command is read-only and must report the frozen Original/A0–A7/frontier hashes,
resolved C0–C4 protocols, exact 29D/action mapping, environment/framework versions,
seed disjointness, and absence of F15 scientific outputs.
