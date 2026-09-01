# F10-PPO Visual v4 Codex — Frozen C0 Formulation

## Scope

This run repairs and retrains C0 (`small_loop`) only. C1 may start only after
the C0 development gate passes. C2–C4, explanation, optimization, and policy
changes are outside this run.

## Runtime decision path

```text
front RGB
  -> dual-boundary lane measurement
  -> fixed offline affine calibration
  -> lane EKF using measured ego motion
  -> 29D belief observation
  -> feed-forward canonical PPO
  -> normalized 2D action
  -> PolicyAction(v_cmd, omega_cmd)
  -> existing differential-drive adapter
```

Simulator lane pose is never available to the actor or value function. It is
used only after policy-observation construction by reward and offline
evaluation. The lane runtime accepts front RGB, actual ego motion, and `dt`.

## Lane-estimator repair

The legacy adaptive-bright fallback aliased every bright marking to both
yellow and white and then preferred only the yellow hypothesis. v4 keeps
strict yellow, strict white, and colour-unknown bright paint separate. It
infers both boundaries independently, fuses consistent centrelines, prefers a
strict-colour boundary on conflict, and returns a structural miss when two
adaptive-only hypotheses disagree. Boundary-source diagnostics are telemetry;
they are not PPO inputs.

The affine mean calibration is fitted on whole trajectories with seeds
61101–61108. Seeds 61201–61208 are development-only. The once-only held-out
lane-belief gate uses 61301–61308. All starts face counter-clockwise.

Held-out gate criteria were fixed before the final run:

- detection rate >= 0.80;
- lateral-belief RMSE <= 0.050 m;
- heading-belief RMSE <= 0.150 rad;
- lateral and heading 68% coverage in [0.50, 0.85];
- lateral and heading 95% coverage in [0.85, 1.00].

Curvature remains a camera-derived belief channel but is explicitly not a v4
pass channel; its held-out error is retained as a limitation.

## Policy observation

The observation dimension and order remain exactly the 29 entries frozen in
`configs/f10_ppo_visual_v4_codex.toml`: five lane-belief entries; measured
linear velocity and yaw rate; lane curvature mean/std; stop-line distance;
nine pedestrian-belief entries; five stop-sign-belief entries; three stop-mode
one-hot entries; and previous commanded linear/angular velocity. C0 supplies
neutral pedestrian/stop semantics. Fixed physical scaling is used and cannot
update during evaluation.

## Start distribution and direction

Every C0 start faces the resolved counter-clockwise route direction. Training
uses a deterministic shuffled cycle: every drivable `small_loop` tile is used
exactly once before a new shuffled cycle begins, followed by bounded pose
jitter. This prevents the single-tile exposure failure of v3.

## PPO and action

Canonical project-local PPO is unchanged: two 256-unit tanh layers,
learning rate 3e-4, rollout 1024, batch 256, 10 epochs, gamma 0.99, GAE lambda
0.95, clip 0.20, entropy coefficient 0.01, value coefficient 0.50, gradient
norm 0.50, and seed 62000. Training budget is 61,440 simulator steps.

The actor produces a bounded two-dimensional normalized action. The one
existing mapper converts it to `v_cmd` in [0, 0.4] m/s and `omega_cmd` in
[-4, 4] rad/s. No second wheel controller is introduced.

## Reward and termination

The frozen v3 decomposed reward is retained to avoid confounding the perception
repair with reward redesign: progress, lane, pedestrian, stop, smoothness, and
terminal components. In C0 pedestrian and stop components are zero. Yellow
clearance warning is preventive; shallow contact on curves may recover under
the frozen curve-recovery rule. Invalid pose, failed recovery, lane departure,
or lap completion terminate; horizon ends truncate.

## Seeds and selection

- C0 train: 62001–62012
- C0 development: 62101–62104
- C0 stage-final (untouched until development pass): 62201–62204
- C1 reserved: train 63001–63012, development 63101–63104, stage-final 63201–63204

Checkpoint selection is safety filter -> C0 skill -> progress -> return. C0
passes development only with completion >= 0.50, lane-failure <= 0.25,
invalid-pose <= 0.25, mean absolute lateral error <= 0.09 m, and mean progress
>= 3.5 m. Development evaluates every retained checkpoint on the same fixed
seed-defined trajectories. No stage-final result may be used for tuning.

## Tracking and artifacts

Training logs to W&B `vnv/DuckiePOMDP`, group
`f10-ppo-visual-v4-codex`, plus local immutable CSV/JSON/checkpoints under
`artifacts/f10_ppo_visual_v4_codex/`. Full training requires source-bound
tests, reward audit, CUDA/W&B preflight, smoke gradient/reload evidence, and a
fresh agent-follows-doc PASS.
