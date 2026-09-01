# F15 Visual Replay Implementation Amendment

**Status: frozen before any F15 video was rendered.**

This amendment changes the *descriptive visualization mechanism only*. It does not
change models, checkpoints, seeds, curricula, gates, thresholds, localization results,
failure selection, the frozen localization decision, or any scientific conclusion.

## Problem

`experiments/render_f15_failure_traces.py` originally produced visual evidence by
re-executing the policy inside the simulator and asserting that the re-execution
reproduced the recorded normalized actions to within `1e-7`:

```python
if max_action_replay_error > 1.0e-7:
    raise RuntimeError(f"descriptive replay action mismatch: {max_action_replay_error}")
```

That assertion cannot be satisfied by this runtime. The actor itself is deterministic —
`run_f15_cross_curriculum_recovery.py` loads it with `actor.cpu().eval()` — but the
frozen F10 perception front-end is not. `src/duckie_pomdp/perception/lane_rgb_model.py`
resolves its device to `cuda` when available, and no `manual_seed`,
`torch.use_deterministic_algorithms`, or `cudnn.deterministic` setting is applied
anywhere in the F15 runner, the PPO environment, or the PPO protocol. Nondeterministic
cuDNN algorithm selection and non-associative parallel float reduction therefore perturb
the lane-pose estimate, which perturbs the public 29D belief, which perturbs the action,
which — in closed loop — compounds over the episode.

This is a property inherited from the frozen F10/F12 stack. F15 did not introduce it and
does not modify the frozen stack.

## Independent evidence that closed-loop rollouts are not reproducible

Before this amendment, F15 already contained duplicated `(model, curriculum, seed)` cells
produced by superseded pre-shard runs. Comparing those repeats against the authoritative
explicit shards gives a direct measurement of run-to-run variation under identical seeds:

- 150 repeated cells;
- 43 differ numerically;
- 7 differ in an **objective outcome label**, including `A0/c2/180001`, where the Original
  Policy completes in one run and times out in another, and `A6/c2/180002`, where
  `lane_failure` and `invalid_pose` exchange places.

Raw evidence is preserved under
`artifacts/f15_cross_curriculum_recovery_v1/integrity/superseded_pre_shard_localization_csv/`
with `superseded_pre_shard_localization_manifest.json`.

## Change

Rendering no longer re-infers the policy. The simulator is driven by the frozen recorded
action sequence, and every overlaid quantity is read from the primary telemetry.

Old:

```text
same seed -> camera -> perception -> belief -> 29D -> actor recomputes action -> simulator -> render
```

New:

```text
FROZEN PRIMARY TELEMETRY
   recorded normalized action a(t)
              |
   same curriculum + same seed + same initial condition
              |
          simulator  (apply RECORDED action at every step)
              |
           render    (overlay values read from telemetry, not recomputed)
              |
   validate reconstruction against the ORIGINAL recorded telemetry
```

The output is a **telemetry-driven reconstruction of the recorded policy episode**, not a
live policy rerun.

## What this does and does not establish

Recorded-action replay removes the `camera -> MobileNet/YOLO/belief -> actor` path from
the reconstruction, which is the known source of nondeterminism. It does **not** by itself
prove that the reconstruction is exact. Residual sources may remain: internal simulator
numerical state, pedestrian state, hidden environment state, frame-skip state,
floating-point accumulation, physics-engine ordering, or external RNG. The reconstruction
is therefore **verified**, not assumed.

## Frozen acceptance criteria

Three levels must be satisfied before any rendered media may be used as evidence. All
thresholds below were fixed before rendering and are implemented as module constants in
`experiments/render_f15_failure_traces.py`.

**Level 1 — recorded-action integrity.** The trace file SHA256 must equal the value bound
into the frozen failure event, and the replayed action array must be element-wise
identical to the telemetry action array.

**Level 2 — trajectory reproduction.** Maximum absolute per-step progress deviation
between reconstruction and telemetry:

| Tier | Threshold | Justification |
|---|---|---|
| exact | `1e-3` m | indistinguishable from bitwise reproduction |
| tolerant | `0.05` m | about five simulator steps of travel at the `0.4 m/s` action bound, and roughly a quarter of the Duckietown half-lane width, so a deviation inside this band cannot by itself change a lane-departure or invalid-pose verdict |

**Level 3 — failure reproduction.** The first objective failure event of the
reconstruction, computed with the same frozen `first_objective_failure_event` rule used to
build the failure registry, must reproduce the recorded event:

| Tier | Event step | Event labels | Termination |
|---|---|---|---|
| exact | identical | identical | identical |
| tolerant | within ±2 steps | identical | identical |

Event labels must match exactly at both tiers. A reconstruction that places the failure at
a different step with different labels is not a reconstruction of that failure.

For an objectively selected successful episode, Levels 1 and 2 are identical. Level 3
requires the replay to reproduce completion without any objective failure label. The
same exact/tolerant progress thresholds apply; there is no separate, looser success
tolerance.

## Outcome classification

| Status | Meaning |
|---|---|
| `VERIFIED_EXACT` | all three levels pass at the exact tier |
| `VERIFIED_WITHIN_TOLERANCE` | all three levels pass at the tolerant tier |
| `UNRESOLVED` | any level fails |

`UNRESOLVED` does not raise and does not stop the run. The media and the full validation
report are retained under an `unresolved/` subdirectory of that trace so the attempt stays
auditable, and `failure_event.json` records `quarantined: true` with the reason. Such
media must not be presented as evidence in any report or figure. The frozen primary
telemetry remains valid and authoritative regardless of reconstruction status.

## Labelling

Rendered media and every figure or caption derived from it use:

> **Recorded-Action Same-Seed Replay** (equivalently, *Telemetry-Driven Reconstruction*)

Methods text must state:

> The visual reconstruction replayed the frozen physical action sequence from the primary
> F15 telemetry. Policy inference was not rerun during rendering.

`same-seed paired rollout` remains the correct term for the Original-versus-Compressed
pairing, and is used alongside — never instead of — the disclosure that the actions are
recorded actions. The media are never described as live model inference, and never as a
causal paired trajectory.

## Scope of impact

| Item | Affected |
|---|---|
| models, checkpoints, actor hashes | no |
| seeds, curricula, environment configuration | no |
| retention gates and thresholds | no |
| localization episodes and telemetry | no |
| cross-curriculum competence matrix | no |
| pruning-width retention | no |
| same-state action fidelity | no |
| failure selection rule and failure registry | no |
| frozen localization decision | no |
| recovery decision, final candidate, holdout | no |
| descriptive visualization mechanism | **yes** |

Primary evidence for every F15 claim is the frozen telemetry from the original 560
localization episodes. Video is a descriptive reconstruction of an already-recorded
failure and is never a new statistical replicate.

## Outcome — recorded-action replay was itself rejected

This section records results. It does not alter any criterion above; every threshold was
fixed before rendering and none was changed afterwards.

Recorded-action replay was executed against the frozen failure events. The frozen
criteria rejected it:

| Event | L2 max progress error | Recorded event step | Replay event step | Labels | Status |
|---|---:|---:|---:|---|---|
| `A1/c0/180002` | `2.96e-08` m | 362 | 218 | identical | `UNRESOLVED` |
| `A1/c1/180002` | `7.14e-09` m | 237 | 66 | identical | `UNRESOLVED` |

The result is sharply split. **Level 2 passed by an enormous margin** — trajectory
reproduction was accurate to about 30 nanometres over the replayed steps, roughly five
orders of magnitude tighter than the `1e-3` exact tier. Driving the simulator with the
recorded action sequence does reproduce the recorded path.

**Level 3 failed decisively.** The replayed episodes terminated far earlier than the
recorded ones (219 of 363 steps, and 67 of 238) and raised their first objective failure
hundreds of steps early. The failure *labels* matched exactly in both cases, so the
replay reaches the same kind of failure — but not at the recorded time. Rendering then
crashed on the third event with `paired failure window contains no aligned frames`,
because the early termination left the 90-before/45-after extraction window empty.

A secondary anomaly is recorded but not explained: Level 1 reported
`replayed_actions_identical_to_telemetry: false` even though the simulator was fed the
telemetry array directly. This is flagged as unresolved rather than attributed to a cause.

Two consequences follow. First, the earlier claim in this document that only the
`camera -> perception -> belief -> actor` path carries nondeterminism is **not supported**:
some residual episode-level state also fails to reproduce, and the conservative wording in
"What this does and does not establish" was the correct posture. Second, camera-based
visual evidence at the recorded failure time cannot be produced by this runtime at all.

Visual evidence is therefore produced by a third mechanism, `direct_frozen_telemetry_timeline`
(`experiments/render_f15_telemetry_diagnostics.py`), which plots the frozen telemetry
directly and includes no camera or simulator frames. Each event records
`camera_or_simulator_frames_included: false` with `reason_camera_replay_absent`. Because
it only draws numbers that are already in the primary telemetry, it cannot disagree with
the competence matrix. The cost is that prompt requirement §14's front-camera side-by-side
is **not satisfied**, and this must be reported as an unmet deliverable rather than
presented as achieved.

## Preserved records

- `artifacts/f15_cross_curriculum_recovery_v1/failure_traces/*/*/*/failure_event.json`
  — the recorded-action replay validation reports that caused this mechanism to be
  rejected in turn (see Outcome below).
- `artifacts/f15_cross_curriculum_recovery_v1/logs/render_visual_evidence*.log`
  — the three recorded-action render attempts, retained including the crash.
- `artifacts/f15_cross_curriculum_recovery_v1/integrity/superseded_pre_shard_localization_csv/`
  — repeated-cell evidence of closed-loop irreproducibility.
- `artifacts/f15_cross_curriculum_recovery_v1/integrity/rejected_preliminary_duplicate_replay/`
  — earlier rejected replay attempt, retained.

Nothing in this amendment permits deleting a prior log or artifact.
