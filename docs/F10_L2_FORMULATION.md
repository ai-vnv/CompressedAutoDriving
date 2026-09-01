# F10-L2 Formulation — `experiment_loop` Transfer Curriculum

Status: frozen before transfer training.

## Question and scope

F10-L2 asks whether the selected F10-L1 lane controller can be transferred to
a different closed-loop map containing both left and right turns. The target
map is Gym-Duckietown's native `experiment_loop`: no intersections, no target
objects, and no stop/pedestrian logic. This remains a lane-control curriculum,
not the full POMDP policy.

The frozen F10-L1 checkpoint is the only initialization:

```text
artifacts/f10_l1/sac_lane_baseline.pt
SHA256 7d492fbff98fca9200266743151c849dd323a7f3259425e4d13eaa3a0ac32f72
```

F10-L1 artifacts are never overwritten. F10-L2 writes only under
`artifacts/f10_l2/` and a separate W&B group.

## Why `experiment_loop`

`small_loop_cw` would mostly mirror the already learned geometry.
`experiment_loop` is a more informative next step: it is still a controlled
closed track, but combines left and right curves. A real-simulator reference
probe from tile `(1, 0)` completed a `7.408 m` lap in 1,705 steps. The frozen
machine-readable witness is `artifacts/f10_l2/map_probe.json`. The F10-L1 SAC
checkpoint failed zero-shot by touching the yellow-line proxy; the retained
reward audit records this outcome in both probe episodes. Transfer training is
therefore necessary rather than cosmetic.

## Runtime boundary

The observation ordering, normalization, action mapping, differential-drive
adapter, reward decomposition, and safety semantics are unchanged from F10-L1.

```text
six agent-visible lane/motion values
    -> fixed normalization
    -> warm-started canonical SAC
    -> normalized [-1, 1]^2 action
    -> existing SACActionMapper
    -> PolicyAction(v_cmd, omega_cmd)
    -> existing differential-drive adapter
    -> real Gym-Duckietown experiment_loop
```

World pose, path length, lap state, and yellow clearance are reward/evaluation
only. The policy never receives them.

## Transfer semantics

Actor, twin critics, targets, entropy state, and optimizer state are loaded
from the hash-verified step-50,000 F10-L1 checkpoint. The replay buffer is new
and empty because the source replay was not checkpointed. For the first 2,000
transitions, actions come from the stochastic warm-start policy. The first
gradient update occurs immediately after transition 2,000 is inserted, so the
update sees a fully populated 2,000-transition new-map buffer.

This is not training from scratch and not a second checkpoint selection on the
F10-L1 data.

## Map, spawn, lap, and episode

- Map: `experiment_loop`
- Route: closed loop with mixed left/right turns
- Spawn tile: `(1, 0)`
- Local pose: `(0.520, 0, 0.1755)`, heading `pi`, plus seeded small jitter
- Episode horizon: 2,200 steps (`73.33 s`)
- Minimum lap path: `6.80 m`
- Leave radius: `0.35 m`
- Finish radius: `0.11 m`
- Finish heading tolerance: `0.35 rad`

Lap completion is termination. Horizon is truncation. Invalid pose, yellow
crossing, and lane departure remain distinct terminations.

## Seeds

- Training: `16001-16012`
- Development: `17001-17004`
- Final evaluation: `18001-18004`
- Probe-only: `19001`, explicitly excluded from all active splits

All active seeds are disjoint from F0-F10-L1 historical evaluation seeds.
Final seeds cannot be used for reward tuning, normalization, checkpoint
selection, or hyperparameter decisions.

## Budget and telemetry

One declared transfer run uses 40,000 new-map environment steps with
checkpoints at 10k intervals. SAC hyperparameters remain those of F10-L1:
`256x256` ReLU, learning rate `3e-4`, gamma `0.99`, tau `0.005`, batch 256,
replay 100k, and automatic entropy. Training uses CUDA and logs online to
`vnv/DuckiePOMDP`, group `f10-l2-experiment-loop-transfer-v1`.

## Selection and acceptance

Development seeds select the checkpoint using the unchanged safety-first rule:
reject clearly unsafe candidates, then rank by lap success, lower lateral
error, and return. Final evaluation occurs once after selection.

The selected checkpoint must achieve at least 3/4 final laps, zero invalid
pose/yellow crossing/lane departure, mean `|d| <= 0.075 m`, mean episode p95
`|d| <= 0.125 m`, mean `|phi| <= 0.22 rad`, mean actual velocity at least
`0.10 m/s`, and at least 0.50 lap-success gain over random and always-stop.

F10-L2 is classified PASS, LIMITED, or FAILED without post-hoc final tuning.
Passing means mixed-turn lane competence only. No pedestrian/stop curriculum,
explanation, or optimization follows automatically.
