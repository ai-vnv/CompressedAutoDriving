# F10-PPO C1 Remediation v5

## Scope and frozen baseline

This gate retries C1 only. The old failed C1 artifacts remain immutable under
artifacts/f10_ppo_visual_v4_codex/c1. The imported predecessor is the passed
C0 checkpoint with SHA256
82cd311d8369b194539d8229d93ff519652121e1de31ef7322574b8f279dcdc2.

The 29-dimensional observation ordering, action mapping, six reward
components, PPO network, PPO hyperparameters, YOLO checkpoint, and F9c belief
stack remain unchanged. Actor and critic weights plus optimizer state are
retained from C0.

## Failure diagnosis

The failed C1 run was numerically stable, but 12/16 development episodes ended
in yellow crossing and 4/16 in invalid pose. C0 retention fell from 75% lap
completion to 0%.

Training telemetry localizes the perception problem:

| geometry | heading RMSE |
| --- | ---: |
| straight | 0.110 rad |
| left curve | 0.265 rad |
| right curve | 0.520 rad |

The old camera-lane affine calibration was fitted on small_loop, whose
counter-clockwise route contains left curves and straights but no right curve.
C1 also remained pinned near one start tile, so its two right curves had poor
training exposure. This is a representation/distribution failure, not PPO
numerical divergence.

## Pre-registered remediation

Only two causal changes are allowed:

1. Fit one fixed camera-lane affine calibration on disjoint small_loop and
   experiment_loop calibration trajectories, balanced by map and turn
   family. Privileged lane pose is an offline target only.
2. Apply the existing deterministic loop-wide start sampler to every native
   closed-loop stage, including all 12 experiment_loop tiles.

No reward relaxation, PPO retuning, or privileged policy input is permitted.
Counter-clockwise direction remains anchored by tile [1, 0], heading pi.

## Lane calibration protocol

- diagnostic-only: 63901-63906 and 63921
- affine calibration: 64001-64008 and 64021-64032
- affine development: 64101-64108 and 64121-64132
- uncertainty development: 64301-64308 and 64321-64332
- once-only lane final: 64401-64408 and 64421-64432

The affine development gate is fixed before execution:

- detection rate >= 0.80
- lateral RMSE <= 0.055 m
- heading RMSE <= 0.160 rad
- right-curve heading RMSE <= 0.220 rad
- right-curve heading RMSE must improve over raw projection

The dynamic final gate additionally requires 68% coverage in [0.50, 0.85]
and 95% coverage in [0.85, 1.00] for lateral and heading.

## C1 training and selection

- training seeds: 65001-65012
- development seeds: 65101-65104
- stage-final seeds: 65201-65204, untouched until development passes
- budget: 61,440 real-simulator steps
- checkpoints: every 10,240 steps
- W&B: vnv/DuckiePOMDP, group f10-ppo-visual-v5-c1-remediation

Every 12 consecutive training episodes begins once on every drivable
experiment_loop tile. Development and stage-final mappings are deterministic
from their explicit seeds.

Checkpoint selection and C1 acceptance remain unchanged:

- completion rate >= 0.50
- lane-failure rate <= 0.25
- invalid-pose rate <= 0.25
- C0 completion drop <= 0.25

If no development checkpoint passes, or C0 retention fails, C1 is FAILED and
stage-final seeds remain unused. C2 is outside this gate.
