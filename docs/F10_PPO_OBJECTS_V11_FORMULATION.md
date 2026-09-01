# F10-PPO object curriculum v11 — safety correction

## Frozen boundary

V11 resumes the hash-pinned, passing C1 checkpoint with the same 29D policy
observation, PPO architecture, action adapter, MobileNet lane belief, YOLO
detector, metric projection, F9c pedestrian EKF, association, and existence
filter. No privileged object state enters actor or critic input.

The scope remains exactly:

```text
C2: experiment_loop + one true road-crossing Duckie; stop sign absent
C3: experiment_loop + stop sign/line; Duckie absent
STOP after C3
```

Ego travels counter-clockwise. Both Duckie modes traverse the same full path
across the ego route in opposite directions.

## Why V10 did not progress

V10 trained for 40,960 steps but every development checkpoint was ineligible.
Its last checkpoint avoided collision on all four development episodes, yet
only reached mean progress `4.071 m`, with unsafe proximity in two episodes and
lane/invalid-pose failures. A representative right-to-left pre-collision frame
had true range about `0.118 m` while the frozen policy-visible belief reported
about `0.347 m`. This is retained as an observation-pipeline limitation, not
hidden by changing YOLO or EKF.

## Pre-registered V11 correction

The C2 acceptance thresholds are unchanged. V11 changes training only:

1. Safety shaping starts at `0.65 m` instead of `0.35 m`.
2. Proximity weight changes from `-0.08` to `-0.12`.
3. Proximity cost is proportional to ego speed; a correctly stopped ego is
   not repeatedly punished merely because the Duckie is still nearby. The
   living penalty still makes indefinite stopping suboptimal.
4. Collision, lane failure, and invalid pose each cost `-12` terminal reward.
5. C2 receives 61,440 steps.
6. Training episodes use true full crossings at `0.12`, then `0.16`, then
   `0.20 m/s`; phases change after episodes 24 and 48.

Only the training split uses the speed curriculum. Development and stage-final
episodes use the untouched scenario speed of `0.20 m/s`. Left-to-right and
right-to-left modes remain balanced deterministically. The training telemetry
records mode, speed, and phase for every step and episode.

The reward continues to use privileged geometry only inside the environment's
reward/evaluation boundary. The actor and critic receive only the existing 29D
runtime belief vector.

## Gates

C2 must satisfy the existing V10 safety, progress, stationary, retention, and
stage-final criteria on new disjoint seeds. C3 is locked until C2 is `PASS`.
No threshold may be weakened after observing evaluation results.

## Reproducible launch

Run from `/home/pannntastic/aivnv/duckie-pomdp`:

```bash
export PYTHONHASHSEED=0
export PYTHONPATH=src:/home/pannntastic/aivnv/duckie/src
export DUCKIETOWN_HEADLESS=1
export CUDA_VISIBLE_DEVICES=0
set -o pipefail

/home/pannntastic/aivnv/duckie/.venv/bin/python \
  experiments/train_f10_ppo.py c2 \
  --config configs/f10_ppo_visual_objects_v11.toml \
  --output-dir artifacts/f10_ppo_visual_objects_v11/c2/training \
  --wandb-mode online \
  --source-checkpoint artifacts/f10_ppo_visual_v9/c1/ppo_selected.pt \
  2>&1 | tee artifacts/f10_ppo_visual_objects_v11/c2/training_run.log
```

C3 may use `artifacts/f10_ppo_visual_objects_v11/c2/ppo_selected.pt` only after
C2 development, retention, and stage-final all permit progression.
