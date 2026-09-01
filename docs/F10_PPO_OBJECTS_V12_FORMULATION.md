# F10-PPO object curriculum v12 — safe wait and restart

V12 retains the frozen 29D RGB→MobileNet lane belief + YOLO→F9c pedestrian
belief boundary, counter-clockwise `experiment_loop`, PPO architecture, action
adapter, object separation, and all V10 acceptance thresholds.

V11 was stopped at step 30,720 as an immutable failed attempt: right-to-left
crossings remained 100% collision and the actor alternated extreme steering
and braking despite `P(exists)≈1`. Slower Duckie speeds made the hazard harder
by keeping it in the lane longer, so V12 keeps physical speed at 0.20 m/s.

Training-only right-to-left start delays progress from 0.60 to 1.00 to the full
1.55 seconds after episodes 8 and 16. Both directions still traverse the full
road-crossing path. Development and stage-final use only the frozen full 1.55 s
timing. A one-shot +3 reward is issued only after an encountered pedestrian
clears the 0.65 m safety region without any unsafe-proximity event; it teaches
the complete stop/wait/restart event. Stationary proximity has zero recurring
cost, while moving toward a nearby pedestrian is penalized more strongly.

At the C1→C2 task boundary actor/critic weights are retained, but stale Adam
state is reset and policy log standard deviation is pre-registered at -1.20.
All canonical PPO hyperparameters, including entropy coefficient 0.01, remain
identical to C1. The lower initial action variance limits destructive exploration
without violating the imported policy contract. C3 is locked until
C2 development, retention, and stage-final all PASS. Scope stops after C3.

Run from the repository root with:

```bash
export PYTHONHASHSEED=0
export PYTHONPATH=src:/home/pannntastic/aivnv/duckie/src
export DUCKIETOWN_HEADLESS=1
export CUDA_VISIBLE_DEVICES=0
set -o pipefail
/home/pannntastic/aivnv/duckie/.venv/bin/python \
  experiments/train_f10_ppo.py c2 \
  --config configs/f10_ppo_visual_objects_v12.toml \
  --output-dir artifacts/f10_ppo_visual_objects_v12/c2/training \
  --wandb-mode online \
  --source-checkpoint artifacts/f10_ppo_visual_v9/c1/ppo_selected.pt \
  2>&1 | tee artifacts/f10_ppo_visual_objects_v12/c2/training_run.log
```
