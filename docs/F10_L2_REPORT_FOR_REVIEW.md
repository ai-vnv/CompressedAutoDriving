# F10-L2 Report for Review — `experiment_loop` Transfer

Classification: **PASS**

F10-L2 tests whether the F10-L1 lane checkpoint transfers to a different
closed Gym-Duckietown map containing both left and right turns. The policy was
warm-started from the selected F10-L1 checkpoint, fine-tuned once under the
frozen protocol, selected on development seeds, and evaluated once on disjoint
final seeds.

This remains a lane-control curriculum stage. It does not claim full POMDP
deployment readiness and does not include YOLO, the F9c belief, stop logic, or
pedestrian response.

## Frozen protocol

- Config: `configs/f10_l2_transfer_v1.toml`
- Config SHA256:
  `2827af458fb49d3d56ac6cf1649dfaf5829519892ff75da9cb0a4410a044665f`
- Map: native Gym-Duckietown `experiment_loop`, mixed left/right turns
- Source checkpoint: F10-L1 step 50,000, SHA256
  `7d492fbff98fca9200266743151c849dd323a7f3259425e4d13eaa3a0ac32f72`
- Training seeds: `16001-16012`
- Development seeds: `17001-17004`
- Final seeds: `18001-18004`
- Historical probe seed: `19001` (excluded from all three splits)
- Observation: the same six agent-visible lane/motion values as F10-L1
- Action: the same normalized SAC action to `PolicyAction(v_cmd, omega_cmd)`
  and existing differential-drive adapter
- Budget: 40,000 new-map simulator steps and 38,001 transfer updates
- Checkpoints: every 10,000 steps
- W&B: <https://wandb.ai/vnv/DuckiePOMDP/runs/y0qu681q>

Actor, twin critics, target critics, entropy state, and optimizer states were
restored from F10-L1. The replay buffer was intentionally new because it was
not checkpointed upstream. Its first 2,000 transitions came from the
stochastic warm-start policy; the first update occurred immediately after
transition 2,000 was inserted.

## Pre-training evidence

The real-map reference controller completed a 7.408 m lap in 1,705 steps. The
unmodified F10-L1 checkpoint failed zero-shot by crossing the yellow-line
proxy in both retained reward-audit episodes, demonstrating that transfer was
necessary.

The reward audit then produced:

| Policy | Lap success | Mean return | Outcome |
|---|---:|---:|---|
| Random | 0% | -8.052 | invalid/yellow failures |
| Always stop | 0% | -6.727 | safe timeout |
| Simple controller | 100% | 23.452 | safe laps |
| Source F10-L1 SAC | 0% | 2.846 | 100% yellow crossing |

The online smoke run completed 128 real-simulator steps, 97 transfer updates,
and an exact checkpoint reload. The immutable pre-training gate, source and
config hashes, map probe, seed isolation, W&B destination, and full test suite
were independently audited before launch.

## Training

The declared run completed all 40,000 steps and generated four checkpoints.
Training was non-monotonic: safe laps appeared around steps 7,400-22,000;
checkpoint 30,000 had collapsed into invalid poses; the policy recovered by
the end. This history was preserved without mid-run tuning.

- Completed episodes: 141
- Transfer updates: 38,001
- Total inherited-plus-transfer update count: 86,002
- Final checkpoint reload: exact and verified
- Training manifest SHA256:
  `e57c5dcdfc2363f7af8f330a0eee300754326eca2f0ad7be5ef72548462e6e93`

## Development checkpoint selection

The predeclared rule filtered unsafe checkpoints, then ranked safe candidates
by lap success, lower lane error, and return.

| Step | Lap success | Invalid | Yellow crossing | Mean `|d|` | Decision |
|---:|---:|---:|---:|---:|---|
| 10,000 | 0% | 0% | 0% | 0.04564 m | safe timeout |
| 20,000 | 100% | 0% | 0% | 0.05500 m | safe candidate |
| 30,000 | 0% | 100% | 0% | 0.04662 m | rejected |
| 40,000 | 100% | 0% | 0% | **0.03464 m** | **selected** |

Selected checkpoint:

- `artifacts/f10_l2/sac_lane_transfer_baseline.pt`
- transfer step: 40,000
- SHA256:
  `09a7fbcf411e9174cd36b4f0413b96c1d85496255c14ae1561014c02645e948a`

In this run the safety-selected, best-return, and last checkpoints happen to
contain the same weights. The separate filenames are provenance roles, not
three deployment policies.

## Once-only final evaluation

The selected checkpoint was evaluated once on untouched seeds 18001-18004.

| Metric | Random | Always stop | Simple | Source F10-L1 | Transfer SAC |
|---|---:|---:|---:|---:|---:|
| Lap success | 0% | 0% | 100% | 0% | **100%** |
| Invalid-pose rate | 25% | 0% | 0% | 0% | **0%** |
| Yellow-crossing rate | 25% | 0% | 0% | 100% | **0%** |
| Lane-departure rate | 75% | 0% | 0% | 0% | **0%** |
| Mean return | -7.451 | -5.405 | 23.335 | 2.790 | **32.446** |
| Mean path length | 0.723 m | 0.000 m | 7.404 m | 2.723 m | **7.246 m** |
| Mean absolute lateral error | 0.04355 m | 0.01690 m | 0.05501 m | 0.01154 m | **0.03413 m** |
| Mean episode p95 `|d|` | 0.10342 m | 0.01690 m | 0.10032 m | 0.02580 m | **0.05763 m** |
| Mean absolute heading error | 0.19715 rad | 0.02537 rad | 0.15489 rad | 0.10367 rad | **0.09582 rad** |
| Minimum yellow clearance | -0.00086 m | 0.02727 m | 0.00932 m | -0.00140 m | **0.02574 m** |
| Mean actual velocity | 0.12933 m/s | 0.00000 m/s | 0.13031 m/s | 0.13703 m/s | **0.12980 m/s** |
| Mean `|omega_cmd|` | 2.0373 | 0.0000 | 0.5566 | 1.2418 | **1.7570 rad/s** |
| Mean action change | 0.7999 | 0.0000 | 0.0052 | 0.5583 | **0.7183** |

All ten pre-registered acceptance checks passed. Transfer SAC did not obtain
success by standing still and generalized across the four final spawn jitters.

## Video evidence

`artifacts/f10_l2/sac_lane_transfer_demo.mp4` is a deterministic real-simulator
proof on development seed 17001, rendered only after selection. It completes
one mixed-turn lap in 1,665 steps (55.5 s):

- path: 7.280 m;
- total return: 32.407;
- mean `|d|`: 0.03445 m;
- p95 `|d|`: 0.05732 m;
- mean `|phi|`: 0.09454 rad;
- minimum yellow clearance: 0.02417 m;
- no invalid pose, yellow crossing, or lane departure.

The overlay labels path length and yellow clearance as `EVAL ONLY`; neither is
fed to SAC. Video SHA256:
`7dca49d54f61584895ac85b4fc9e886954327c9f7937899c91eeeed5394effe7`.

## Known limitations

1. The policy consumes simulator-provided lane-relative ego measurements, not
   an RGB lane estimator.
2. It is validated only on `small_loop` and `experiment_loop`, not arbitrary
   maps or intersections.
3. Four final seeds are enough for this narrow reproducibility gate, not broad
   generalization claims.
4. Transfer SAC remains substantially less smooth than the simple controller
   (`0.7183` vs `0.0052` mean action change).
5. Training was unstable in the middle, including a failed 30k checkpoint.
   The final result depends on safety-first checkpoint selection.
6. This checkpoint excludes YOLO/EKF and is not the deployable full-POMDP
   policy.

## Decision

**F10-L2 PASS.** The selected F10-L1 lane policy was successfully transferred
to a new real Gym-Duckietown map with mixed left/right turns. No full-POMDP,
explanation, optimization, or later curriculum stage is started by this
report.

Final active test suite: **419 passed, 0 failed, 0 skipped**. The emitted
warnings are dependency and simulator warnings, not test failures.
