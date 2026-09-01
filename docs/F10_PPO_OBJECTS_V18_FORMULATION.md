# F10-PPO V18 — C3 Public-Belief DAgger Correction

V15 C2 remains frozen and PASS. V17 demonstrated a C3 behavior-cloning
failure: its step-zero actor matched teacher observations offline but stalled
after its own action changed the next 29D observation. The failure is a
closed-loop distribution shift, not a YOLO, stop-belief, geometry, or reward
failure.

V18 changes training data only. It keeps the same 29D observation, PPO actor
and critic, action adapter, reward, YOLO11n, camera projection, stop belief,
visual lane belief, scenario geometry, and C3 acceptance thresholds.

The warm-start dataset gives equal loss mass to:

1. frozen C2 retention anchors;
2. successful C3 public-belief teacher trajectories;
3. C3 states reached by the failed V17 step-zero actor, relabelled by the
   public-belief teacher.

The failed actor is executed only on V17 training seeds `165001..165004`.
Teacher labels are functions of the normalized 29D policy observation. No
privileged object or lane truth enters the dataset. V18 uses disjoint
`166xxx` train/development/stage-final seeds.

Before substantive PPO training, the distilled step-zero actor must pass a
training-only closed-loop gate on two fresh V18 training seeds. The gate checks
C2 retention plus C3 completion, progress, stop, restart, lane safety, valid
pose, and non-stationary behavior. This prevents another offline-MSE-only
warm-start from consuming the full PPO budget.

Step zero remains a development candidate. Development selection, official C2
retention, and the once-only C3 stage-final follow the frozen safety-first
protocol. C4 remains fail-closed.

## Launch contract

```bash
export PYTHONHASHSEED=0
export PYTHONPATH=src:/home/pannntastic/aivnv/duckie/src
export DUCKIETOWN_HEADLESS=1
export CUDA_VISIBLE_DEVICES=0
set -o pipefail
/home/pannntastic/aivnv/duckie/.venv/bin/python experiments/train_f10_ppo.py c3 \
  --config configs/f10_ppo_visual_objects_v18.toml \
  --output-dir artifacts/f10_ppo_visual_objects_v18/c3/training \
  --wandb-mode online \
  --source-checkpoint artifacts/f10_ppo_visual_objects_v15/c2/ppo_selected.pt \
  2>&1 | tee artifacts/f10_ppo_visual_objects_v18/c3/training_run.log
```
