# F10 Formulation — Baseline Visuomotor SAC Policy

Status before training: **frozen protocol**. F10 trains one canonical SAC
baseline. It does not explain, optimize, or compare RL algorithms.

## Frozen upstream boundary

The runtime chain is the already validated F9c chain:

```text
front RGB -> frozen YOLO11n -> calibrated projection
          -> frozen robust pedestrian EKF/existence belief
```

F10 does not change YOLO, camera geometry, association, innovation gating,
measurement calibration, EKF dynamics, or existence filtering. The pinned
files and SHA256 values are in `configs/f10_sac_v1.toml`. Simulator truth is
available only to the reward and evaluator after the agent observation has
been built.

## Policy observation

The policy input is a 17-element `float32` vector in this exact order:

| # | Feature | Unit | Fixed scale |
|---:|---|---|---:|
| 0 | lateral error | m, positive left | 0.25 |
| 1 | heading error | rad, positive CCW/left | 0.75 |
| 2 | actual linear velocity | m/s | 0.4 |
| 3 | actual yaw rate | rad/s, positive CCW | 4.0 |
| 4 | road curvature | 1/m | 5.0 |
| 5 | signed stop-line distance | m; positive before line | 2.0 |
| 6 | pedestrian existence probability | probability | 1.0 |
| 7 | pedestrian range mean | m, object-origin semantics | 2.0 |
| 8 | pedestrian range standard deviation | m | 1.0 |
| 9 | pedestrian bearing mean | rad, positive left | 1.2 |
| 10 | pedestrian bearing standard deviation | rad | 1.0 |
| 11 | pedestrian radial-velocity mean | m/s | 1.0 |
| 12 | pedestrian radial-velocity standard deviation | m/s | 1.0 |
| 13 | pedestrian bearing-rate mean | rad/s | 4.0 |
| 14 | pedestrian bearing-rate standard deviation | rad/s | 4.0 |
| 15 | previous commanded linear velocity | m/s | 0.4 |
| 16 | previous commanded angular velocity | rad/s | 4.0 |

Each raw feature is divided by its fixed physical scale and clipped to
`[-3, 3]`. No final-evaluation statistics are fitted. Existence probability
remains in `[0,1]`. Previous action is included because the comfort reward
depends on action change.

Road curvature and stop-line distance use the existing agent-visible
`RoadMeasurement` contract, but never simulator `cur_pos`/`cur_angle`. The
initial stop distance is a navigation-route prior computed from the configured
spawn tile/local pose, the map's fixed `0.585 m` tile size, and stop line. It
is then dead-reckoned using measured actual ego speed
and measured lane-heading error. The validated F10 approach is straight, so
the route curvature prior is zero. Object pose, object silhouette, true
pedestrian state, GT IoU, and future data are absent. Stop-sign belief is
omitted because the current runtime has no validated stop-sign belief updater.

## Action

The actor emits `[-1,1]^2`. The existing `NormalizedActionScaler` performs
the only policy-to-physical transform:

```text
v_cmd     = (a_v + 1) / 2 * 0.4 m/s
omega_cmd = a_omega * 4.0 rad/s
```

The resulting `PolicyAction` enters the existing
`DifferentialDriveActionAdapter`; SAC never produces wheel commands.

## Reward

The logged decomposition is

```text
r = progress + lane + stop + pedestrian + comfort + terminal
```

- Progress is route-coordinate displacement times `5.0`, plus a `-0.002`
  per-step living cost. Reverse displacement is not hidden.
- Lane cost is a simple quadratic cost in measured lateral and heading error.
- Stop reward uses the signed agent-visible stop-line distance. A valid stop
  requires 12 consecutive steps at or below `0.025 m/s` inside the final
  `0.30 m`; crossing first incurs `-6`. Completion is rewarded once. The
  larger violation cost was fixed after the pre-training audit showed that
  the original `-2` still let reckless constant-forward motion score well.
- Pedestrian safety uses privileged ego-reference-point to pedestrian-footprint
  distance only inside the reward/evaluation boundary. It is an explicit V1
  contact proxy—not robot-footprint-to-pedestrian-footprint clearance. It
  penalizes proximity and unsafe approach; proxy distance below `0.08 m`
  terminates with `-8`.
- Comfort penalizes changes in physical chassis commands and sustained large
  yaw command.
- Terminal reward is `+2` for reaching `1.25 m` progress after completing the
  stop, and `-1` for simulator `invalid-pose`. Invalid pose is reported as its
  own outcome, not called a collision.

There is deliberately no reward term conditioned on belief uncertainty.

## Episode semantics

True termination: geometric pedestrian contact or successful completion.
Simulator `invalid-pose` ends the simulator session and is recorded separately
as `invalid_pose`, with causality left unresolved. Administrative horizon and
simulator `max-steps-reached` are truncations. Every transition records both
flags and a reason.

## Scenario and seed protocol

The validated `pomdp_v1` map is used with stationary, left-to-right, and
right-to-left pedestrian modes in deterministic round-robin order. Training
perturbs longitudinal pose, lateral pose, and heading within the bounds frozen in config. This supplies lane
recovery, moving/turning ego behavior, pedestrian crossing, and stop-line
interaction without new objects or cameras.

- train: 10001–10018
- development: 11001–11006
- final: 12001–12006

Splits are disjoint and exclude all recorded detector/F7/F8/F9/F9c/F9d/demo
seeds. Final seeds cannot affect normalization, reward, hyperparameters, or
checkpoint selection.

## SAC and budget

Stable-Baselines3 is not installed in the validated environment and is not
silently added. F10 therefore uses a standard PyTorch SAC: tanh-squashed
Gaussian actor, two Q networks, two target Q networks, replay buffer, soft
target updates, and automatic entropy tuning. Networks are two ReLU layers of
256 units. Full hyperparameters are frozen in config. The planned budget is
20,000 environment steps with checkpoints every 4,000 steps. This is one
baseline run, not a hyperparameter search.

Training telemetry is mirrored to Weights & Biases entity/project
`vnv/DuckiePOMDP`, group `f10-sac-baseline-v1`, every 10 environment steps and at every episode
boundary. Local CSV/JSON remains the authoritative auditable record. The
smoke test uses W&B offline mode; the baseline run requires a verified online
login before it starts.

## Reward audit and checkpoint rule

Before SAC training, random, always-stop, constant-forward, and a proportional
lane/safety controller run on non-final seeds. Training cannot start until the
audit verifies that safe competent progress beats random and always-stop, and
reckless progress is penalized by collisions/violations.

Development selection is safety-first:

1. reject checkpoints with collision rate above 0.20 or invalid-pose rate
   above 0.25;
2. maximize success rate;
3. then progress;
4. then return.

If none passes the safety filter, select the least unsafe checkpoint only for
diagnosis and classify F10 no higher than LIMITED.

The evaluator implementation and its seed boundary are frozen before full
training. A one-episode evaluator smoke test may use the first development
seed to prove checkpoint loading and the real RGB-to-action evaluation path;
it is not used for checkpoint selection and never touches a final seed.

Full training refuses to start unless `artifacts/f10/pretraining_gate.json`
attests, under the same config SHA256, that the reward audit passed, the SAC
smoke run produced gradient updates and an exactly reloadable checkpoint, the
development evaluator smoke passed, W&B online access targets
`vnv/DuckiePOMDP`, CUDA is available, no W&B credential is stored in the
project, and the complete test suite passed.

After training, all five periodic checkpoints are evaluated on development
seeds. Three stable aliases are then written:

- `last.pt`: the 20,000-step checkpoint;
- `best_return.pt`: the highest development-return checkpoint, diagnostic
  only and possibly unsafe;
- `sac_baseline.pt`: the safety-first selected checkpoint and the only F10
  control checkpoint used for final evaluation, deployment, and F11.

These aliases may have identical SHA256 values when the same checkpoint wins
more than one definition. Final evaluation never repeats selection and refuses
to overwrite an existing result.

## Final metrics and acceptance

The selected checkpoint is evaluated once on final seeds against random,
always-stop, and the deterministic lane/safety controller. Reports include
success, progress, duration, lane errors/departures, pedestrian collision and
clearance, stop completion/violation, velocity, yaw rate, action change,
oscillation, reward components, and scenario breakdown.

Predeclared PASS requirements are: valid leak-free end-to-end execution;
success at least 0.40; mean progress at least 0.75 m; progress gains of at
least 0.40 m over always-stop and 0.15 m over random; collision rate at most
0.20; lane-departure rate at most 60% of random; and at least `0.03 m/s`
lower mean speed in the pedestrian safety region than in clear frames. Failure
of competence with a valid pipeline is LIMITED; an invalid pipeline is FAILED.
