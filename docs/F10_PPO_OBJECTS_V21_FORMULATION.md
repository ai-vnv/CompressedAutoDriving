# F10-PPO V21 — C4 Combined Pedestrian + Stop

## C4 detector integration guard

The YOLO checkpoint, camera projection, F9c EKF, and existence-filter
parameters remain unchanged. C4 adds one deterministic adapter-boundary rule:
Duckie detections below confidence `0.40` are not passed to F9c, while stop-sign
detections retain the frozen detector threshold `0.10`. This value was frozen
from the pre-existing F9 calibration-only set: all 1,193 correct selected
Duckies were at least `0.4207`, whereas all three incorrect selections were at
most `0.2351`. It is not fitted on C4 development or final seeds.

## Stop-line route prior

C4 begins on an earlier loop segment than C3. Its agent-visible stop-line
distance is initialized from the configured map centerline arc between the
configured spawn and stop line, then dead-reckoned using measured actual ego
motion. It never reads simulator world pose. A straight-line projection is
invalid here because it would mark the obligation satisfied on the preceding
segment, before the physical line is reached.

The combined scenario uses a one-shot crossing. The Duckie begins moving so it
intersects the counter-clockwise ego route at the actual encounter time, then
is physically removed after reaching the far side. Historical C2 scenarios
keep Gym-Duckietown's repeat-crossing behavior.

V21 starts only from the selected passing V20 C3 checkpoint at PPO step
1,024 (`ef30983…f4918a`).  It keeps the same feed-forward PPO architecture,
the same 29-dimensional policy-visible belief vector, frozen YOLO/F9c/lane
belief stack, physical action adapter, reward decomposition, and
counter-clockwise route.

The C4 real-simulator episode contains both independent hazards:

1. a Duckie moves across the ego route on the eastbound straight, in balanced
   left-to-right and right-to-left episodes;
2. the stop sign and its distinct stop line remain on the later northbound
   straight.

The runtime actor and critic receive only:

```text
front RGB -> visual lane belief + YOLO object measurements -> F9c/stop belief
          -> fixed 29D vector -> PPO actor/critic
```

Privileged truth remains downstream of vector construction and is used only
for reward, termination, and offline evaluation.

## Frozen C4 protocol

```text
training steps       8,192
checkpoint interval  1,024
episode horizon      4,200 (combined task needs both separated interactions)
training seeds       169001..169012
development seeds    169101..169104
stage-final seeds    169201..169204
source checkpoint    V20 C3 selected PPO step 1,024
optimizer transition reset Adam
action log_std        -3.0
learning rate         1e-5
epochs / rollout      2
clip range            0.05
entropy coefficient   0
gradient norm         0.1
target KL             0.01
```

Step zero is diagnostic and cannot be selected.  A candidate must be at least
step 1,024 and pass the pre-registered collision, unsafe proximity, stop,
restart, completion, progress, lane, stationary, and invalid-pose constraints.
Retention must also preserve explicit C0, C1, C2, and C3 task floors.  C4
stage-final is once-only and cannot run before development selection and the
complete retention matrix pass.

## Launch contract

```bash
export PYTHONHASHSEED=0
export PYTHONPATH=src:/home/pannntastic/aivnv/duckie/src
export DUCKIETOWN_HEADLESS=1
export CUDA_VISIBLE_DEVICES=0
set -o pipefail
/home/pannntastic/aivnv/duckie/.venv/bin/python experiments/train_f10_ppo.py c4 \
  --config configs/f10_ppo_visual_objects_v21.toml \
  --output-dir artifacts/f10_ppo_visual_objects_v21/c4/training \
  --wandb-mode online \
  --source-checkpoint artifacts/f10_ppo_visual_objects_v20/c3/ppo_selected.pt \
  2>&1 | tee artifacts/f10_ppo_visual_objects_v21/c4/training_run.log
```

No global-final, explanation, optimization, or post-final retuning is part of
V21.  The experiment stops after C4 stage-final classification.
