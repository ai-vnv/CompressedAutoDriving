# F10-PPO V19 — C3 Low-Variance On-Policy Remediation

V18 proved that the 29-dimensional RGB/lane-belief/YOLO/F9c/stop-belief
representation can complete the stop-only task, but development selected its
DAgger-distilled step-zero actor. Every checkpoint after normal PPO updates
failed. V19 is a fail-closed remediation whose selected C3 checkpoint must have
received real on-policy PPO updates.

## Frozen diagnosis

The V18 step-zero policy had behavior-dataset MSE `0.001106` and completed all
four C3 development episodes. At step 10,240 its MSE was `0.007114`; at step
40,960 it was `0.127287`. The successful deterministic policy had mean action
change about `0.0044`, while V18 stochastic training episodes had action change
around `0.35-0.42`. With `log_std=-1.2`, action-space standard deviation was
about `0.30`, which was too aggressive after a competent DAgger warm start.

## Pre-registered remediation

V19 keeps canonical feed-forward PPO, the fixed 29D policy representation,
reward, camera lane belief, YOLO11n, F9c pedestrian belief, stop belief, action
adapter, map, and C3 acceptance thresholds unchanged. It changes only the
curriculum transition and evaluation discipline:

1. load the frozen passing C2 checkpoint and reproduce the frozen V18 DAgger
   warm start;
2. reset Adam after the curriculum transition;
3. set the post-warm-start exploration `log_std` to `-3.0` (approximately
   `0.05` normalized-action standard deviation);
4. train for eight 1,024-step PPO rollouts and save every rollout;
5. evaluate step zero for diagnosis, but require selected step `>=1024`;
6. refuse once-only stage-final evaluation unless development selected an
   eligible updated checkpoint;
7. run C0-C3 retention before consuming stage-final seeds.

No actor anchoring loss, behavior-cloning loss inside PPO, target-ground-truth
input, reward threshold relaxation, or test-seed tuning is introduced.

## Frozen gates

An updated candidate must pass the existing safety filter and C3 stop skill:

```text
invalid pose rate  <= 0.25
lane failure rate  <= 0.25
stop violation     <= 0.25
stop completion    >= 0.50
restart            >= 0.50
selected step      >= 1024
```

The existing C2 collision-retention increase remains `<=0.10`. Stage-final is
run once only after development selection and retention pass. If no updated
candidate is eligible, V19 is FAILED and C4 remains blocked.

## Seeds

```text
training    167001..167012
development 167101..167104
stage-final  167201..167204
```

These are disjoint from V18 and earlier detector, belief, and curriculum
evaluation seeds.

## Launch contract

```bash
export PYTHONHASHSEED=0
export PYTHONPATH=src:/home/pannntastic/aivnv/duckie/src
export DUCKIETOWN_HEADLESS=1
export CUDA_VISIBLE_DEVICES=0
set -o pipefail
/home/pannntastic/aivnv/duckie/.venv/bin/python experiments/train_f10_ppo.py c3 \
  --config configs/f10_ppo_visual_objects_v19.toml \
  --output-dir artifacts/f10_ppo_visual_objects_v19/c3/training \
  --wandb-mode online \
  --source-checkpoint artifacts/f10_ppo_visual_objects_v15/c2/ppo_selected.pt \
  2>&1 | tee artifacts/f10_ppo_visual_objects_v19/c3/training_run.log
```
