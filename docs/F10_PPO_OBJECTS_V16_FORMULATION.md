# F10-PPO V16 — C3 Roadside Stop-Sign Geometry Remediation

V15 C2 is frozen **PASS** and imported by checkpoint hash.  V15 C3 is retained
as a failed attempt: all PPO candidates ignored the stop obligation, while the
public-belief reference controller completed the stop but subsequently produced
`invalid_pose`.  The simulator diagnostic at that event reported
`on_drivable=True` and `collision_free=False`; its vehicle footprint overlapped
the stop-sign collision geometry.  This is a scenario-placement defect, not a
YOLO, lane-belief, or EKF defect.

V16 changes only the C3 stop-only scenario geometry:

- the original `experiment_loop` road and counter-clockwise route are unchanged;
- the stop sign is moved from its colliding lane-edge pose to an external
  roadside shoulder, 0.2355 m laterally from the northbound lane centre;
- the stop line remains an independent geometric reference;
- the Duckie is physically absent in C3.

The 29-dimensional policy input, frozen RGB MobileNet lane estimator, lane EKF,
frozen YOLO11n checkpoint, camera projection, F9c pedestrian belief, stop belief,
PPO architecture/hyperparameters, reward, action mapping, and C2/C3 acceptance
thresholds are unchanged.  Runtime C3 remains:

```text
front RGB
  -> YOLO class stop_sign
  -> bbox bottom-centre
  -> frozen camera projection
  -> stop-sign belief + stop obligation state
  -> 29D PPO observation
  -> PolicyAction(v_cmd, omega_cmd)
```

Privileged simulator state remains downstream of policy construction and is used
only for reward and evaluation.

Before PPO training, the real-simulator geometry gate must demonstrate on the
new C3 development seeds that the public-observation reference controller:

1. detects the stop sign through YOLO;
2. completes the required hold;
3. restarts;
4. has zero stop violations;
5. has zero collisions and zero invalid poses.

V16 uses new C3 train/development/stage-final seeds.  It imports the selected
V15 C2 checkpoint and its PASS/retention evidence by hash.  C4 remains forbidden.

## Launch contract

```bash
export PYTHONHASHSEED=0
export PYTHONPATH=src:/home/pannntastic/aivnv/duckie/src
export DUCKIETOWN_HEADLESS=1
export CUDA_VISIBLE_DEVICES=0
set -o pipefail
/home/pannntastic/aivnv/duckie/.venv/bin/python experiments/train_f10_ppo.py c3 \
  --config configs/f10_ppo_visual_objects_v16.toml \
  --output-dir artifacts/f10_ppo_visual_objects_v16/c3/training \
  --wandb-mode online \
  --source-checkpoint artifacts/f10_ppo_visual_objects_v15/c2/ppo_selected.pt \
  2>&1 | tee artifacts/f10_ppo_visual_objects_v16/c3/training_run.log
```
