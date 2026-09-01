# F10-PPO V14 — C2 Retention Rehearsal

V14 addresses the single failed V13 gate: the selected C2 checkpoint passed the
pedestrian development task but catastrophically forgot C1 driving when the
pedestrian belief was neutral. Acceptance thresholds, PPO hyperparameters,
reward, YOLO/F9c, MobileNet lane belief, 29-dimensional policy boundary, action
adapter, and the V9 C1 source checkpoint remain unchanged.

C2 uses one reusable Gym-Duckietown simulator context. Every third training
episode is a no-pedestrian driving rehearsal: the dynamic Duckie is removed
from the simulator object list (therefore from rendering, collision, stepping,
and privileged lookup) and the public pedestrian belief is neutral. The same
Duckie object is reset and re-enabled for the following crossing episode. This
keeps the task switch physical while avoiding Mesa/Pyglet context growth.

## Pre-registered change

Only C2 **training** adds deterministic rehearsal on `experiment_loop`:

- episode index modulo 3 = 0: Duckie crossing left-to-right;
- episode index modulo 3 = 1: Duckie crossing right-to-left;
- episode index modulo 3 = 2: no physical Duckie, neutral pedestrian belief,
  and a loop-wide counter-clockwise start pose.

The no-pedestrian episode still runs front RGB through the frozen MobileNet lane
measurement and lane EKF. YOLO/F9c remain active but receive no Duckie target.
The actor and critic receive the same normalized 29-dimensional representation.
Privileged truth remains confined to reward and evaluation.

C2 development and stage-final remain pedestrian-only at the full crossing
speed/timing. C1 retention remains no-pedestrian `experiment_loop`. Rehearsal
does not alter evaluation trajectories or acceptance thresholds.

## Frozen protocol

- Source: selected V9 C1 checkpoint, SHA256
  `0e26ac28d8806140ff9544ecb094c20e850f66f83972544eb0dd8ac9b4d131b2`.
- Behavior dataset: immutable V13 policy-visible dataset, SHA256
  `1c2301a714daea9e84be903fb9158500c6b9547a47d420f5561cedb2e24c2268`.
- C2 budget: 40,960 steps; checkpoints 0/10,240/20,480/30,720/40,960.
- C2 passes only after development skill, C1 retention, and untouched
  stage-final all pass. C3 is forbidden before that result.

## Launch contract

```bash
export PYTHONHASHSEED=0
export PYTHONPATH=src:/home/pannntastic/aivnv/duckie/src
export DUCKIETOWN_HEADLESS=1
export CUDA_VISIBLE_DEVICES=0
set -o pipefail
/home/pannntastic/aivnv/duckie/.venv/bin/python experiments/train_f10_ppo.py c2 \
  --config configs/f10_ppo_visual_objects_v14.toml \
  --output-dir artifacts/f10_ppo_visual_objects_v14/c2/training \
  --wandb-mode online \
  --source-checkpoint artifacts/f10_ppo_visual_v9/c1/ppo_selected.pt \
  2>&1 | tee artifacts/f10_ppo_visual_objects_v14/c2/training_run.log
```

Stop after C3. C4 remains outside this protocol scope.
