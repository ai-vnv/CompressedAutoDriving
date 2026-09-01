# F10-PPO object curriculum v10 — `experiment_loop`

## Scope

This protocol imports the frozen passing C1 checkpoint and extends the same
29-dimensional camera/belief-conditioned PPO policy only through:

```text
C2: experiment_loop + one crossing Duckie, stop sign physically hidden
C3: experiment_loop + one stop sign/stop line, Duckie physically hidden
STOP after C3
```

The objects occupy different route locations. The Duckie crosses the
eastbound straight at `(1.4625, 2.1645)` m. The stop line lies later on the
northbound straight at `(2.1645, 1.4625)` m. Their Euclidean separation is
approximately `1.131 m`; route separation is validated from simulator lane
geometry before training.

## Duckie crossing invariant

The configured Duckie path is the world segment
`(1.4625, 1.7900) -> (1.4625, 2.5390)` m (or its reverse). It must intersect
the counter-clockwise ego lane centreline on tile `(2, 3)`, not merely point
toward the road. A real-simulator gate must additionally demonstrate a
temporal conflict opportunity under the frozen C1 driving policy. A spatial
intersection alone is insufficient. Both endpoints must lie beyond the road
body, so completing the path means the Duckie has fully crossed rather than
stopping beside the lane centerline.

## Stop invariant

The sign creates the obligation; the independently configured stop line is
where stopping is judged. A real rendered orientation gate must show the STOP
face toward the northbound approach before C3 training. C2 receives neutral
stop fields. C3 receives neutral pedestrian fields.

## Runtime boundary

```text
front RGB
  -> frozen MobileNet lane measurement -> lane EKF
  -> frozen YOLO stop_sign/duckie detections
  -> frozen metric projection and pedestrian belief / stop belief
  -> same 29D PPO observation
  -> inherited PPO actor/critic
  -> PolicyAction(v_cmd, omega_cmd)
```

Privileged geometry is available only to scenario validation, reward, and
offline evaluation. It is not attached to actor or critic input.

## Gates

1. Geometry: path/route intersection, opposite road sides, object separation.
2. Real simulator: correct visibility isolation, Duckie moves across the lane,
   correct stop-sign face, and C1 creates a genuine temporal conflict.
3. Reward/pretraining: decomposed object reward audit, CUDA smoke, full tests,
   W&B destination, frozen source/evidence hashes.
4. C2: train/dev/stage-final/retention; stop if not PASS.
5. C3: train/dev/stage-final/retention; stop if not PASS.

C4 combined training is deliberately outside this protocol.

## Reproducible local launch

Run these commands from `/home/pannntastic/aivnv/duckie-pomdp` in WSL Bash.
The environment contract requires deterministic Python hashing, the validated
source overlay, headless rendering, and explicit binding to the single local
CUDA device. `tee` preserves the complete launch log while `pipefail` keeps a
Python failure visible to the caller.

```bash
export PYTHONHASHSEED=0
export PYTHONPATH=src:/home/pannntastic/aivnv/duckie/src
export DUCKIETOWN_HEADLESS=1
export CUDA_VISIBLE_DEVICES=0

set -o pipefail
/home/pannntastic/aivnv/duckie/.venv/bin/python \
  experiments/train_f10_ppo.py c2 \
  --config configs/f10_ppo_visual_objects_v10.toml \
  --output-dir artifacts/f10_ppo_visual_objects_v10/c2/training \
  --wandb-mode online \
  --source-checkpoint artifacts/f10_ppo_visual_v9/c1/ppo_selected.pt \
  2>&1 | tee artifacts/f10_ppo_visual_objects_v10/c2/training_run.log
```

C3 is authorized only after C2 development, retention, and stage-final gates
classify C2 as `PASS`. Its source checkpoint is the hash-verified C2 selection:

```bash
set -o pipefail
/home/pannntastic/aivnv/duckie/.venv/bin/python \
  experiments/train_f10_ppo.py c3 \
  --config configs/f10_ppo_visual_objects_v10.toml \
  --output-dir artifacts/f10_ppo_visual_objects_v10/c3/training \
  --wandb-mode online \
  --source-checkpoint artifacts/f10_ppo_visual_objects_v10/c2/ppo_selected.pt \
  2>&1 | tee artifacts/f10_ppo_visual_objects_v10/c3/training_run.log
```

The interrupted first C2 attempt omitted `PYTHONHASHSEED` and explicit CUDA
binding. It is retained only as
`c2/training_attempt1_missing_env_contract` and is excluded from checkpoint
selection and scientific results.
