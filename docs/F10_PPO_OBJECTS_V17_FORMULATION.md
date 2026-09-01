# F10-PPO V17 — C3 Balanced Stop-Belief Distillation

V15 C2 remains the frozen predecessor and is imported by checkpoint SHA256.
V16 corrected the C3 roadside geometry and established, on real simulator
rollouts, that the public-belief controller can detect the sign, stop, hold,
restart, and finish without collision or invalid pose.

V17 teaches that behavior to the existing PPO actor without changing the
runtime representation or the PPO algorithm.  Its warm-start dataset has equal
loss mass from two sources:

1. frozen C2 lane/pedestrian behavior, used as a retention anchor;
2. C3 real-runtime observations collected through
   `RGB -> YOLO stop_sign -> metric projection -> stop belief -> 29D vector`,
   labelled by the deterministic public-belief controller.

The dataset contains no privileged evaluation fields.  The short stop/hold
portion is weighted because unweighted lap data would be dominated by ordinary
lane-following frames.  Actor distillation is followed by the unchanged
canonical PPO training budget.  Step zero is retained as an auditable candidate
so PPO updates cannot erase a safer distilled policy without losing development
selection.

The policy input remains 29D for C0–C3.  C3 contains the stop sign only; the
Duckie is physically removed.  YOLO11n, camera projection, F9c pedestrian
belief, stop belief, visual lane belief, action mapping, PPO hyperparameters,
reward, and acceptance thresholds remain frozen.  Privileged simulator truth
is read only after policy construction for reward and evaluation.

V17 uses new disjoint C3 train, development, and stage-final seeds.  The V16
teacher seeds and every historical V15/V16 evaluation seed are excluded.  C4
remains outside this protocol.

## Launch contract

```bash
export PYTHONHASHSEED=0
export PYTHONPATH=src:/home/pannntastic/aivnv/duckie/src
export DUCKIETOWN_HEADLESS=1
export CUDA_VISIBLE_DEVICES=0
set -o pipefail
/home/pannntastic/aivnv/duckie/.venv/bin/python experiments/train_f10_ppo.py c3 \
  --config configs/f10_ppo_visual_objects_v17.toml \
  --output-dir artifacts/f10_ppo_visual_objects_v17/c3/training \
  --wandb-mode online \
  --source-checkpoint artifacts/f10_ppo_visual_objects_v15/c2/ppo_selected.pt \
  2>&1 | tee artifacts/f10_ppo_visual_objects_v17/c3/training_run.log
```
