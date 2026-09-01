# F10-PPO Visual-Lane Belief Curriculum v2

## Scope

This frozen run retrains only C0 and, after a passing C0 gate, C1. C0 uses
`small_loop`; C1 uses `experiment_loop`. Both maps run counter-clockwise from
tile `(1, 0)` with nominal local pose `(0.520, 0, 0.1755)` and heading `pi`.
Pedestrian and stop challenges remain inactive. C2-C4 are outside this run.

## Runtime boundary

The actor and critic receive no simulator lane pose or privileged object
geometry. Each RGB frame is processed by two independent perception paths:

```text
front RGB
  +-- visual lane measurement -> affine calibration -> lane EKF -> LaneBelief
  +-- frozen YOLO11n -> frozen F9c object belief runtime

LaneBelief + ego motion + neutral hazards + previous action -> 29D PPO input
```

YOLO is executed from C0 onward, but C0/C1 expose semantically neutral
pedestrian and stop fields because those curriculum hazards are inactive. This
keeps the observation dimension fixed without allowing irrelevant detections to
change the driving-only tasks.

Privileged simulator state is read only after the policy vector has been
constructed and is used for reward and offline evaluation.

## Frozen 29D observation

The exact ordering is:

1. lane validity probability
2. lane lateral-error mean [m]
3. lane lateral-error standard deviation [m]
4. lane heading-error mean [rad]
5. lane heading-error standard deviation [rad]
6. actual ego linear velocity [m/s]
7. actual ego yaw rate [rad/s]
8. lane-curvature mean [1/m]
9. lane-curvature standard deviation [1/m]
10. stop-line distance [m]
11-19. pedestrian existence and polar kinematic belief
20-24. stop-sign existence/range/bearing belief
25-27. stop mode one-hot (`NONE`, `REQUIRED`, `SATISFIED`)
28. previous commanded linear velocity [m/s]
29. previous commanded angular velocity [rad/s]

Lane entries 1-5 and 8-9 come from front-camera inference. Entries 6-7 are
actual chassis-motion feedback, not policy commands. Entry 10 is neutral at
`2.0 m` in C0/C1. The previous action gives a feed-forward policy one-step
actuation context and remains distinct from actual motion.

The neutral pedestrian and stop beliefs use zero existence probability, far
range means, broad uncertainty, and `StopMode.NONE`; they cannot encode a
zero-range hazard. Fixed physical normalization from
`configs/f10_ppo_visual_v2.toml` is immutable in training and evaluation.

## Visual-lane evidence

The once-only held-out visual-lane validation uses seeds 36501-36502 on both
maps. It passed the frozen measurement gate with:

- 2,400 frames and 2,057 detections (85.71%);
- lane-belief lateral RMSE 0.02689 m;
- lane-belief heading RMSE 0.06736 rad;
- curvature RMSE 0.74959 1/m.

The interval coverage is under-calibrated and curvature remains substantially
weaker than lateral/heading estimation. These are frozen limitations; final
validation seeds are not reused for tuning.

## Action and PPO

The canonical feed-forward PPO uses separate 256x256 Tanh actor/value MLPs.
The normalized action is mapped exactly once to:

```text
v_cmd     in [0.0, 0.4] m/s
omega_cmd in [-4.0, 4.0] rad/s
```

and then enters the existing differential-drive adapter. PPO parameters are
fixed in `configs/f10_ppo_visual_v2.toml`: learning rate 3e-4, rollout 1,024,
batch 256, 10 epochs, gamma 0.99, GAE lambda 0.95, clip 0.20, entropy
coefficient 0.01, value coefficient 0.50, max gradient norm 0.50.

## Seeds and budget

C0 uses training seeds 40001-40012, development seeds 40101-40104, and
once-only stage-final seeds 40201-40204 for 61,440 steps. C1 uses 41001-41012,
41101-41104, and 41201-41204 for 40,960 additional steps. All are disjoint from
historical and global-final seeds. Checkpoints are emitted every 10,240 steps.

C1 may start only from the selected C0 checkpoint after development selection,
C0 retention evaluation, and a passing once-only C0 stage-final gate. C1 must
also re-evaluate C0 and satisfy the pre-registered maximum completion drop of
0.25.

## Acceptance and stopping rule

C0 requires completion >= 0.50, lane-failure <= 0.25, invalid-pose <= 0.25,
mean absolute lateral error <= 0.09 m, and mean progress >= 3.5 m. C1 requires
completion >= 0.50, lane-failure <= 0.25, invalid-pose <= 0.25, and the C0
retention limit above. Safety filtering precedes task/progress/return in
checkpoint selection.

Training stops after C1. No C2, C3, C4, global-final evaluation, explanation,
policy optimization, recurrent policy, or robust-filter modification is part
of this run. The visual-v2 launcher and evaluator reject C2--C4 rather than
relying on operator convention.

## Monitoring and provenance

Online training is logged to `vnv/DuckiePOMDP` under group
`f10-ppo-visual-belief-curriculum-v2`. Each launch is bound to hashes for the
config, lane calibration, YOLO checkpoint, F9c belief config, maps, source
inventory, tests, reward audit, smoke checkpoint reload, environment profile,
W&B destination preflight, and independent follows-document audit. The source
inventory covers the complete `duckie_pomdp` runtime package and is rechecked
before both C0 and C1; the smoke witness is exactly 128 environment steps with
at least two PPO updates.

For native C0/C1 maps, episode resets reuse one Gym-Duckietown simulator and
OpenGL context while changing only the frozen seed/start pose. This prevents
the Mesa/Pyglet context-retention growth observed when rebuilding a simulator
after every short failed episode. A 36-reset RSS audit is part of the immutable
pretraining evidence.
