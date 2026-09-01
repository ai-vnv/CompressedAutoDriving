# F10-PPO C0-C1 camera-lane remediation v7

## Frozen diagnosis

The previous C1 policy did not fail because PPO diverged: KL stayed small and
explained variance recovered.  It failed because the colour/ground-plane lane
estimator treated a short preview line as the ego-local lane tangent.  On
`experiment_loop` curves this mixed road curvature into heading, including
wrong-signed estimates after a bend.  Affine, quadratic, pose-excited, and
line/arc geometric corrections were retained as failed diagnostics; none met
the pre-registered static plus closed-loop gates.

## Camera-only remediation

Version 7 uses a compact MobileNetV3-small lane-pose regressor:

```text
front RGB
  -> MobileNet lane measurement [d, phi, kappa]
  -> unchanged lane EKF [mean, uncertainty, validity]
  -> fixed 29D PPO belief observation
```

Simulator lane pose is used only as an offline supervised target and held-out
evaluation truth.  Runtime model inference accepts only `HxWx3 uint8` RGB.
Duckie and stop-sign perception remain the frozen YOLO path.  PPO architecture,
reward, action mapping, observation order, and counter-clockwise direction are
unchanged.

## Data and frozen selection

`lane_rgb_v1` contains 600 train, 300 development, and 300 once-only final
frames covering every tile of `small_loop` and `experiment_loop`, 15 combined
lateral/heading poses, domain randomization, and straight/left/right geometry.
Splits use disjoint seeds.  The selected model is development epoch 30 with
SHA256 `d02b182843dd0e0bc3931e1bc36b09aea11aa1ab99ad5456c910fbc77128144d`.
The once-only final gate passed with lateral RMSE 0.01050 m, heading RMSE
0.06681 rad, curvature RMSE 1.31144 m^-1, and heading-sign accuracy 0.99583.

## Remaining gates before PPO

1. Run a disjoint real-simulator closed-loop camera-belief gate on both maps.
2. Run reward, reset-memory, W&B, environment, smoke, and complete test gates.
3. Obtain a fresh agent-follows-doc PASS tied to all current hashes.
4. Train C0 from random initialization; the old policy is semantically
   incompatible with the new lane measurement.
5. Proceed to C1 only from the selected C0 checkpoint after C0 PASS.

No C2-C4 work is authorized in this remediation.
