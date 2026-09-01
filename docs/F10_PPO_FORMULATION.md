# F10-PPO Formulation — Staged Belief-Conditioned PPO

Status: protocol frozen before substantive training. Final holdout results must
not be inspected until C0--C4 development selection has completed.

## Claim and boundary

F10-PPO tests one primary claim: a feed-forward continuous-control PPO policy
conditioned on the explicit runtime belief can acquire driving, pedestrian
response, and stop compliance cumulatively. The actor and critic receive the
same 25-dimensional, policy-visible vector. Neither receives privileged
simulator truth.

Runtime is:

```text
front RGB -> frozen YOLO11n -> calibrated metric observations
          -> frozen F9c pedestrian belief + runtime stop belief
          -> fixed 25D vector -> PPO -> [v_cmd, omega_cmd]
          -> existing differential-drive adapter -> simulator
```

Simulator truth is permitted only after the action at the reward, termination,
and offline evaluation boundary.

## Repository inspection

- `small_loop` is the native 3x3 counter-clockwise loop with left turns.
- `experiment_loop` is the native 4x4 loop with left and right turns.
- `pomdp_v1` provides one stop sign, one explicit stop line, and one Duckie.
- Public pedestrian belief is the frozen F9c nine-moment polar belief.
- Domain contracts already contain `StopSignBelief` and `StopMode`. Before this
  gate, YOLO stop detections were diagnostic only and `StopMode` remained
  `NONE`; F10-PPO therefore adds the missing runtime adapter without using GT.
- Action is the existing `PolicyAction(v_cmd, omega_cmd)` with bounds
  `[0, 0.4] m/s` and `[-4, 4] rad/s`, converted by the existing differential
  drive adapter.
- Stable-Baselines3 is not installed. No dependency is installed silently;
  canonical feed-forward PPO is implemented with the existing PyTorch 2.12.1.
- Existing reward and evaluation code provides lane/lap, pedestrian footprint,
  stop-line, telemetry, checkpoint, and W&B patterns, but no PPO solver exists.

## Policy observation

The ordering is immutable across C0--C4:

| Index | Field | Unit / semantics | Scale |
|---:|---|---|---:|
| 0 | lateral error | m, positive left | 0.25 |
| 1 | heading error | rad, positive left | 0.75 |
| 2 | actual linear velocity | m/s | 0.4 |
| 3 | actual yaw rate | rad/s, positive CCW | 4.0 |
| 4 | road curvature | 1/m | 5.0 |
| 5 | signed stop-line distance | m, positive before line | 2.0 |
| 6--14 | pedestrian existence, polar means/stds/rates | F9c public belief | config |
| 15--19 | stop-sign existence, range/bearing means/stds | runtime visual belief | config |
| 20--22 | stop mode NONE/REQUIRED/SATISFIED | one-hot | 1.0 |
| 23--24 | previous physical action | m/s, rad/s | 0.4, 4.0 |

All features use fixed physical scaling and clipping to `[-3, 3]`. There are
no mutable running statistics.

Neutral pedestrian means `P(exists)=0`, not range zero. Its range is 2 m with
large uncertainty and its rates are zero with large uncertainty. Neutral stop
means `P(sign exists)=0`, `stop_mode=NONE`, and signed line distance 2 m. The
existence field and one-hot mode are the validity masks; neutral values cannot
encode an immediate hazard.

## Stop representation

The stop-sign visual belief uses only the frozen detector outputs and existing
camera projection. The deterministic highest-confidence stop-sign detection is
projected to range/bearing. Misses decay existence; they are never encoded as
zero range. The stop obligation state machine consumes only this belief,
agent-side dead-reckoned signed stop-line distance, measured actual velocity,
and elapsed steps. It transitions:

```text
NONE -> REQUIRED -> SATISFIED
```

It never reads privileged stop-line distance or true sign state. Stop reward
and evaluation may independently use truth after the runtime observation is
built.

## Action

PPO emits a diagonal Gaussian action in normalized coordinates. Training stores
the sampled pre-clipping action and PPO log probability; the environment clips
exactly once to `[-1,1]^2`, then reuses the existing mapper:

```text
a_v=-1 -> v_cmd=0.0 m/s
a_v=+1 -> v_cmd=0.4 m/s
a_w=-1 -> omega_cmd=-4 rad/s
a_w=+1 -> omega_cmd=+4 rad/s
```

## Reward and episodes

One formulation is used for every stage:

```text
r = progress + lane + pedestrian + stop + smoothness + terminal
```

Inactive hazard terms are exactly zero. Progress uses measured forward motion,
lane terms use agent-visible lane pose, smoothness uses current/previous action,
and terminal events use explicit simulator/evaluation outcomes. Privileged
footprints and stop-line truth are used only for pedestrian/stop training
signals and evaluation, never policy input.

Termination is a true success/failure event: lap/route completion, lane/yellow
failure, pedestrian collision, or simulator invalid pose. Horizon is
truncation. `invalid-pose` is reported separately.

## PPO

- actor and value MLP: 256--256, tanh;
- diagonal Gaussian actor with state-independent learned log standard deviation;
- learning rate 3e-4;
- rollout 1024 steps, minibatch 256, 10 epochs;
- gamma 0.99, GAE lambda 0.95;
- clip 0.2, entropy coefficient 0.01, value coefficient 0.5;
- gradient norm 0.5;
- fixed seed 30000; CUDA.

The network shape never changes. Each stage inherits actor, critic, log-std,
optimizer, and immutable normalization from the selected previous checkpoint.
No replay buffer exists in PPO.

## Curriculum and seed isolation

The exact train/development/stage-final seeds are frozen in
`configs/f10_ppo_v1.toml`. C0 uses 300xx, C1 310xx, C2 320xx, C3 330xx, C4
340xx, and global final uses 350xx--354xx. All are pairwise disjoint and exclude
historical detector/F9/F10 seeds.

- C0: native `small_loop`; hazards neutral; visual domain randomization off,
  matching the validated basic-driving reference.
- C1: native `experiment_loop`; hazards neutral; visual domain randomization off.
- C2: `pomdp_v1`; pedestrian active, stop disabled.
- C3: `pomdp_v1`; stop active, pedestrian removed/neutral.
- C4: `pomdp_v1`; both active.

Stage-final and global-final seeds are never used for training, normalization,
checkpoint selection, or reward/PPO tuning.

## Selection and acceptance

At each stage all planned checkpoints are evaluated on development seeds. The
frozen rule is safety filter, then required stage skill, then progress, then
return. The selected checkpoint is then evaluated on all learned tasks using
development seeds (`retention`), and only then is stage-final run exactly once.
Stage-final classification combines safety, current-stage skill, and the
pre-registered retention limit.

Pre-registered gates are the numeric tables in `configs/f10_ppo_v1.toml`:

- C0: completion >= 50%, lane/invalid failure <= 25%, mean |d| <= 0.09 m,
  mean progress >= 3.5 m.
- C1: completion >= 50%, lane/invalid failure <= 25%, and C0 completion drop
  <= 0.25.
- C2: collision <= 10%, unsafe-episode rate <= 50%, progress >= 0.55 m,
  stationary fraction <= 65%, and C1 completion drop <= 0.25.
- C3: stop completion >= 50%, stop violation <= 25%, restart >= 50%, with C2
  collision increase <= 0.10.
- C4: collision <= 10%, stop completion >= 50%, violation <= 25%, progress
  >= 0.55 m.

A FAILED stage stops the curriculum. LIMITED is reported when hard safety still
passes but either current-stage skill or retention does not; LIMITED does not
permit progression in Version 1. Only PASS authorizes the next stage. The next
training command verifies the previous selected checkpoint hash, embedded PPO
architecture/configuration, PASS classification, and retention result.
Thresholds are not changed after final results.

## Evaluation and artifacts

Every stage records per-step rewards, actions, policy-visible observations,
separately named evaluation GT, episode events, dev candidates, selected
checkpoint/hash, stage-final metrics, and retention results. W&B target is
`vnv/DuckiePOMDP`, group `f10-ppo-belief-curriculum-v1`.

After C4, the selected checkpoint is evaluated once on five untouched global
holdouts. No retraining follows. The final report is
`docs/F10_PPO_REPORT_FOR_REVIEW.md`.

For every explicit evaluation seed, pedestrian mode and spawn jitter are pure
functions of that seed's frozen position in its split. They do not depend on
reset order, checkpoint order, or baseline order.

Full C0 training is guarded by a semantic pretraining gate. It verifies the
exact source inventory and hashes plus: reward audit PASS, PPO smoke gradients
and reload, zero-failure active JUnit suite, exact online W&B destination,
CUDA/environment identity, and a fresh independent agent-follows-doc PASS.
