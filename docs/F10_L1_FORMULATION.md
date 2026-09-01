# F10-L1 Formulation — `small_loop` Counter-Clockwise Lane Competence

Status: frozen before substantive SAC training.

## Scope

F10-L1 isolates the first driving skill: complete one counter-clockwise lap of
Gym-Duckietown's native `small_loop`, remain in the right-hand lane, keep the
robot footprint away from the yellow center line, and avoid leaving the road.
There is no stop sign, pedestrian, YOLO, EKF, reward for uncertainty, or policy
explanation in this stage. The completed F10 POMDP baseline is not overwritten.

The installed map itself states that its four left turns form a
counter-clockwise loop. The locked spawn is tile `(1, 0)`, local pose
`(0.520, 0, 0.1755)`, heading `pi`, with small seed-controlled pose jitter.

## Runtime boundary

```text
agent-visible lane/motion observation
    -> LanePolicyObservation (6 values)
    -> fixed physical normalization
    -> canonical SAC
    -> normalized action in [-1, 1]^2
    -> existing SACActionMapper
    -> PolicyAction(v_cmd, omega_cmd)
    -> existing DifferentialDriveActionAdapter
    -> Gym-Duckietown
```

The policy observation never contains world pose, map object truth, lap state,
or reward-only geometry. World pose is read after the action for reward and
evaluation only. `front_rgb` is rendered but not used by this lane-state
curriculum policy; this is not an end-to-end RGB lane detector.

## Observation

Ordering and fixed scales are:

| Index | Feature | Unit | Scale |
|---:|---|---|---:|
| 0 | lateral error `d` | m | 0.15 |
| 1 | heading error `phi` | rad | 0.50 |
| 2 | actual linear velocity | m/s | 0.40 |
| 3 | actual yaw rate | rad/s | 4.0 |
| 4 | previous commanded linear velocity | m/s | 0.40 |
| 5 | previous commanded angular velocity | rad/s | 4.0 |

Each feature is divided by its fixed scale and clipped to `[-3, 3]`. No
statistics are learned from development or final episodes. This deliberately
small six-dimensional input is a curriculum checkpoint, not yet the full
POMDP deployment input.

## Coordinates and yellow-line semantics

The existing convention is retained: `d > 0` is left of the current lane
tangent. At the locked westbound spawn, moving north toward the yellow line
produces `d < 0`; this was verified against real simulator resets.

The right-lane center is `0.117 m` from the yellow line. With an ego half-width
proxy of `0.075 m`, signed footprint-to-yellow clearance is

```text
yellow_clearance_m = 0.117 + d - 0.075 = 0.042 + d.
```

Positive clearance is separated from the line; zero means the footprint proxy
touches it. This proxy is used only for reward/evaluation. The policy sees `d`,
not the derived clearance or a crossing flag.

## Action

SAC emits `[-1, 1]^2`. Exactly one existing mapping produces:

```text
0 <= v_cmd <= 0.4 m/s
-4 <= omega_cmd <= 4 rad/s
```

Positive `omega` is counter-clockwise. There is no second wheel controller.

## Lap and episode semantics

A manual agent-visible P-controller established the reference envelope on the
real simulator: one lap took 1,232 steps (`41.07 s`) and `5.45 m`, with no
termination. The horizon is therefore 1,500 steps (`50 s`).

Lap completion requires all of:

1. leave the start region (`>0.35 m` from the start);
2. accumulate at least `4.50 m` of world-path length;
3. return within `0.11 m` of the start;
4. heading within `0.35 rad` of the start heading.

World pose and the lap gate are reward/evaluation-only. Lap completion is a
true termination. Horizon is a truncation. Simulator invalid-pose, lane
departure (`|d| > 0.150 m`), and yellow-footprint crossing are terminations
with separate reasons.

## Reward

The decomposed reward is

```text
r = r_progress + r_lane + r_yellow + r_comfort + r_living + r_terminal.
```

- `r_progress` rewards nonnegative measured forward motion, capped at the
  `0.20 m/s` target and gated by heading/lane alignment.
- `r_lane` is a simple quadratic penalty on `d` and `phi`.
- `r_yellow` is a quadratic barrier only inside the `0.030 m` warning margin.
- `r_comfort` penalizes command change and excessive yaw command.
- `r_living` prevents standing still from becoming attractive.
- `r_terminal` gives `+10` for a lap and safety penalties for yellow crossing,
  lane departure, or simulator invalid-pose.

Every term and every safety event is logged separately. The reward receives
privileged pose only for progress/lap accounting; none is returned in the
policy observation.

## Seeds and scenarios

- Training: `13001-13012`
- Development: `14001-14004`
- Final: `15001-15004`

They are disjoint from each other and from the recorded historical/F10 seeds.
All use `small_loop`, counter-clockwise, no domain/dynamics randomization, with
seeded small longitudinal/lateral/heading spawn jitter.

## SAC and budget

Canonical PyTorch SAC is reused unchanged: MLP `256x256` ReLU, learning rate
`3e-4`, gamma `0.99`, tau `0.005`, batch `256`, replay `100,000`, learning
starts at `2,000`, automatic entropy with initial alpha `0.20` and target
entropy `-2`. One declared run uses `60,000` environment steps on CUDA, with
checkpoints every `10,000` steps. Training is logged online to
`vnv/DuckiePOMDP`, group `f10-l1-small-loop-ccw-v1`.

## Reproducible pre-training commands

Run from `/home/pannntastic/aivnv/duckie-pomdp` with the validated interpreter:

```bash
export PYGLET_HEADLESS=true DUCKIETOWN_HEADLESS=1 LIBGL_ALWAYS_SOFTWARE=1
export PYTHONPATH=src:/home/pannntastic/aivnv/duckie/src
PY=/home/pannntastic/aivnv/duckie/.venv/bin/python

$PY experiments/audit_f10_l1_reward.py
$PY experiments/train_f10_l1_sac.py --smoke --wandb-mode online
$PY scripts/verify_f10_l1_pretraining.py
$PY -m pytest tests -q --disable-warnings
```

The explicit online smoke is part of the frozen protocol and must resolve to
`https://wandb.ai/vnv/DuckiePOMDP/runs/<run-id>`. Credentials stay in the WSL
user credential store and are never written to the repository.

## Checkpoint selection

Only development seeds select a checkpoint. First reject checkpoints with
invalid-pose, yellow-crossing, or lane-departure rate above `0.25`; then rank
by lap success, lower mean absolute lane error, and return. The final split is
evaluated once after selection.

## Acceptance criteria

On the untouched final split the selected SAC checkpoint must satisfy:

- lap success rate at least `0.75`;
- zero invalid-pose, yellow-crossing, and lane-departure episodes;
- mean `|d| <= 0.075 m`, p95 `|d| <= 0.125 m`;
- mean `|phi| <= 0.20 rad`;
- mean actual velocity at least `0.10 m/s`;
- lap-success gain of at least `0.50` over random and always-stop.

The policy is classified `PASS`, `LIMITED`, or `FAILED` without post-hoc final
seed tuning. Passing F10-L1 means lane competence only; it does not yet mean
the complete YOLO/EKF POMDP policy is deployment-ready.
