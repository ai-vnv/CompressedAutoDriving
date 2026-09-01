# F10-PPO visual-belief curriculum: C0-C1 report

## Classification

```text
C0 small_loop      : PASS
C1 experiment_loop : PASS
Protocol stop      : after C1
```

No C2, C3, C4, global-final, explanation, or policy optimization was run.

## Runtime policy boundary

Both stages use the same 29-dimensional normalized belief vector. The lane
path is `front_rgb -> MobileNet lane measurement -> lane EKF -> PPO`. Duckie
and stop-sign perception remain in the frozen YOLO/belief runtime, with
neutral hazard semantics in C0/C1. Actor and critic receive no privileged
simulator truth. Action remains one normalized PPO action mapped to
`PolicyAction(v_cmd, omega_cmd)` with `v_cmd` in `[0, 0.4] m/s` and
`omega_cmd` in `[-4, 4] rad/s`.

The lane direction is counter-clockwise.

## Frozen identities

```text
protocol config SHA256 : 7743d6468a8598c8bed8b76dff00f25a2ed40d3d46476dd3bbbd0c582e5c4280
pretraining gate SHA256: c6be6e748ce8f72135dae56a13ca0bb0d18f841c4a136ac2e55ea090d5e54d20
lane model SHA256      : 91d471d5ccf9875012d564fa8937838fd0f95e6e3e6aabaefcad654d9b4bb84f
```

The active pretraining suite passed 531 tests with zero failures, errors, or
skips. The once-only camera-lane closed-loop gate completed 4/4 C0 and 4/4 C1
counter-clockwise laps before PPO training began.

## C0 small_loop

C0 started from random initialization and ran the frozen 61,440-step budget:
60 PPO updates, 180 completed training episodes, six checkpoints, and an exact
checkpoint reload check. Development used seeds 94101-94104. Four of six
checkpoints passed the development gate; the frozen safety/task ranking chose
step 61,440.

```text
selected checkpoint : artifacts/f10_ppo_visual_v9/c0/ppo_selected.pt
checkpoint SHA256   : c8f1fc8b2b2b00ace1c594479edb8801baf53b1c0e06405e86c013ec911db3c8
stage-final seeds   : 94201-94204
```

Stage-final result:

| Policy | Completion | Progress | Invalid pose | Lane failure | Mean return |
| --- | ---: | ---: | ---: | ---: | ---: |
| random | 0/4 | 0.534 m | 50% | 50% | -7.98 |
| always stop | 0/4 | 0.000 m | 0% | 0% | -4.77 |
| simple controller | 3/4 | 4.765 m | 0% | 25% | 21.48 |
| PPO | 3/4 | 7.005 m | 0% | 0% | 27.73 |

C0 passed every frozen skill, safety, and retention-baseline check.

## C1 experiment_loop

C1 inherited the selected C0 actor, critic, optimizer, and fixed observation
normalization. It then ran 61,440 additional environment steps. The manifest
records update count 60 -> 120, 76 C1 training episodes, six checkpoints, and
an exact reload check. Development used seeds 95101-95104. The last four
checkpoints all completed 4/4 laps with zero invalid poses and zero lane
failures; the frozen ranking chose step 51,200 by progress.

```text
selected checkpoint : artifacts/f10_ppo_visual_v9/c1/ppo_selected.pt
checkpoint SHA256   : 0e26ac28d8806140ff9544ecb094c20e850f66f83972544eb0dd8ac9b4d131b2
stage-final seeds   : 95201-95204
```

Stage-final result:

| Policy | Completion | Progress | Invalid pose | Lane failure | Mean return |
| --- | ---: | ---: | ---: | ---: | ---: |
| random | 0/4 | 0.313 m | 50% | 50% | -7.61 |
| always stop | 0/4 | 0.000 m | 0% | 0% | -7.95 |
| simple controller | 4/4 | 7.238 m | 0% | 0% | 38.37 |
| PPO | 3/4 | 6.498 m | 0% | 25% | 20.92 |

C1 meets the pre-registered minimum completion and maximum lane-failure
limits exactly or better. It is a competent baseline, not an optimized policy.

## Catastrophic-forgetting check

On development seeds, the selected C1 checkpoint retained:

| Checkpoint | small_loop | experiment_loop | Invalid/lane failure |
| --- | ---: | ---: | ---: |
| C0 selected | 4/4 | not applicable | 0/0 on C0 |
| C1 selected | 4/4 | 4/4 | 0/0 on both tasks |

The small-loop completion drop after C1 is `0.00`, below the frozen maximum
of `0.25`.

## W&B

Both full runs finished online in `vnv/DuckiePOMDP`, group
`f10-ppo-visual-v9-camera-lane-competence`:

- C0: run `8y8ni2gl`, 61,440 steps, 60 updates.
- C1: run `0ghlzugq`, 61,440 additional steps, 120 cumulative updates.

## Evidence

The authoritative machine-readable results are:

- `artifacts/f10_ppo_visual_v9/c0/development_metrics.json`
- `artifacts/f10_ppo_visual_v9/c0/stage_final_metrics.json`
- `artifacts/f10_ppo_visual_v9/c1/development_metrics.json`
- `artifacts/f10_ppo_visual_v9/c1/retention_metrics.json`
- `artifacts/f10_ppo_visual_v9/c1/stage_final_metrics.json`

Known limitation: C1 PPO achieves 75% rather than 100% completion on the
once-only stage-final split and reaches the allowed 25% lane-failure ceiling.
This is sufficient for the frozen C1 PASS criterion, but should not be
described as optimal or failure-free.
