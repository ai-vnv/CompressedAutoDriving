# F10-PPO V20 — C3 KL-Guarded On-Policy Remediation

V19 was stopped at smoke, before substantive training. Its two 64-step PPO
updates produced approximate KL `0.9847` and `0.6012` with clip fractions
`0.7031` and `0.6719`. Lower action noise alone was therefore insufficient:
the inherited canonical learning rate was too large relative to the narrow
post-DAgger action distribution.

V20 preserves the fixed 29D RGB/lane-belief/YOLO/F9c/stop-belief interface,
the DAgger step-zero actor, environment, reward, acceptance thresholds, action
adapter, and canonical clipped PPO objective. It adds pre-registered C3-only
optimizer settings commonly supported by canonical PPO implementations:

```text
learning rate        1e-5
epochs per rollout   2
clip range           0.05
entropy coefficient  0
gradient norm        0.1
target KL            0.01
post-transition log_std -3.0
```

The target-KL guard stops the remaining minibatches of a rollout update when
the pre-update approximate KL exceeds `1.5 * target_kl`. It is not an outlier
filter and does not inspect privileged state. Checkpointing remains every
1,024 real-simulator steps for eight rollouts. Step zero remains diagnostic and
is ineligible for selection; once-only stage-final remains fail-closed until an
updated development checkpoint and C0-C3 retention pass.

V20 uses new disjoint seeds:

```text
training    168001..168012
development 168101..168104
stage-final  168201..168204
```

No final seed is used for tuning. If no `step>=1024` checkpoint passes, V20 is
FAILED and C4 remains blocked.

## Launch contract

```bash
export PYTHONHASHSEED=0
export PYTHONPATH=src:/home/pannntastic/aivnv/duckie/src
export DUCKIETOWN_HEADLESS=1
export CUDA_VISIBLE_DEVICES=0
set -o pipefail
/home/pannntastic/aivnv/duckie/.venv/bin/python experiments/train_f10_ppo.py c3 \
  --config configs/f10_ppo_visual_objects_v20.toml \
  --output-dir artifacts/f10_ppo_visual_objects_v20/c3/training \
  --wandb-mode online \
  --source-checkpoint artifacts/f10_ppo_visual_objects_v15/c2/ppo_selected.pt \
  2>&1 | tee artifacts/f10_ppo_visual_objects_v20/c3/training_run.log
```
