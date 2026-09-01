# F10-PPO visual-lane C0-C1 remediation v6

## Frozen diagnosis

The first C1 run was numerically stable but failed every development episode.
Its visual heading RMSE reached 0.520 rad on right curves.  A first
bidirectional affine recalibration reduced near-centre RMSE, but a closed-loop
reward audit exposed the hidden failure: with true heading at +0.29 to +0.69
rad, the public lane belief reported approximately -0.12 to 0.00 rad and the
controller left the map.

The cause is calibration support, not PPO KL divergence or colour loss.  The
old calibration trajectories stayed close to the centreline, so a 3x3 affine
map could not identify the nonlinear relationship between projected boundary
geometry and heading once the ego accumulated a meaningful pose error.

## Pre-registered remediation

1. Collect disjoint pose-excited RGB calibration/development frames on every
   tile of `small_loop` and `experiment_loop`, using lateral offsets and
   heading offsets up to +/-0.30 rad.
2. Fit one fixed degree-2 ridge calibration with the explicit basis
   `[d, phi, kappa, d^2, phi^2, kappa^2, d*phi, d*kappa, phi*kappa]`.
3. Validate the runtime lane belief in closed loop, including the originally
   failing heading sign, before PPO training.
4. Retrain C0 from random initialization because changing the numerical
   meaning of lane-belief inputs invalidates semantic compatibility with the
   old C0 checkpoint.  C1 may inherit only the newly selected C0 checkpoint.
5. Retain loop-wide starts on both native maps.

PPO architecture/hyperparameters, the 29D observation ordering and scales,
action mapping, reward, YOLO, pedestrian belief, and counter-clockwise route
direction remain unchanged.

## Data boundaries

- pose calibration: deterministic 70xxx seeds;
- pose development/model selection: deterministic 71xxx seeds;
- dynamic uncertainty/closed-loop development: 72xxx seeds;
- once-only lane final gate: 73xxx seeds;
- C0 train/dev/final: 74001-74012 / 74101-74104 / 74201-74204;
- C1 train/dev/final: 75001-75012 / 75101-75104 / 75201-75204.

All groups are disjoint.  Lane final, C0 final, and C1 final data are not used
for fitting or checkpoint selection.

## Gates before training

Pose-excited development must satisfy detection >= 0.55, lateral RMSE <=
0.050 m, heading RMSE <= 0.180 rad, excited-heading sign accuracy >= 0.80,
and excited-heading correlation >= 0.70.  A separate dynamic gate must then
demonstrate safe belief-conditioned steering and calibrated uncertainty on
both maps.  Reward audit, reset-memory audit, W&B preflight, PPO smoke,
full tests, and fresh agent-follows-doc audit must all pass before C0.

## Training and stop rule

C0 and C1 each use 61,440 steps and checkpoints every 10,240 steps.  Stage
selection and acceptance thresholds remain inherited and unchanged.  If C0
does not PASS, C1 is not launched.  If C1 development or C0 retention fails,
C1 final seeds remain untouched.  This gate stops after C1.
