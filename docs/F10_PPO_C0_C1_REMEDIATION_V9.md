# F10-PPO C0-C1 camera-lane competence remediation v9

V8 passed its frame-level held-out lane gate and completed all four C0
closed-loop development laps, but it completed only one of four C1 laps. The
three failures were `invalid_pose` at positive true lateral errors of roughly
0.11--0.14 m. Development-only traces showed two concrete model failures:

1. the lane model underestimated the positive lateral error by about
   0.06--0.09 m near the road boundary;
2. on one experiment-loop curve it predicted negative heading/curvature while
   evaluation truth was positive, causing the audit controller to steer the
   wrong way.

The V8 training distribution explains both failures. It contained very few
real right-curve frames, its recovery starts used only +/-0.04 m lateral
offset, and horizontal flipping swapped the physical roles of Duckietown's
yellow centreline and white outer boundary.

V9 therefore changes only the camera-lane estimator training data and image
preprocessing:

- the top 25% of RGB is cropped before resize so model capacity focuses on road
  geometry;
- horizontal flipping is disabled because it is not a physically valid lane
  augmentation;
- new counter-clockwise real-simulator images cover every tile, three
  longitudinal locations, lateral offsets through +/-0.09 m, and heading
  offsets through +/-0.35 rad;
- real right-curve tiles receive extra independent domain-randomized samples;
- training metadata is balanced across real left/right/straight frames without
  altering any pixels or labels;
- the existing V8 dynamic train/development trajectories remain available, but
  their already-consumed final split is excluded.

Simulator lane state remains an offline training/evaluation label. Runtime
input remains exactly `front_rgb`; the selected measurement then enters the
unchanged lane EKF and the unchanged 29-dimensional PPO belief observation.

Before viewing the V9 final split, development must pass the global frame gate
plus right-turn heading RMSE <= 0.14 rad and edge-lateral RMSE <= 0.03 m. The
untouched final split uses the same criteria. The selected model must then pass
the pre-registered C0/C1 closed-loop development gate followed by its untouched
closed-loop final seeds. PPO C0 starts from random initialization; C1 may inherit
only a passing selected C0 checkpoint. Reward, action mapping, PPO
hyperparameters, and the 29D ordering remain unchanged.

## Frozen V9 results

The selected camera-lane model is epoch 25 of the single planned V9 run:

```text
checkpoint : artifacts/f10_ppo_visual_v9/lane_rgb_model/best.pt
SHA256     : 91d471d5ccf9875012d564fa8937838fd0f95e6e3e6aabaefcad654d9b4bb84f
runtime    : front_rgb only, crop top 25%, no horizontal flip
```

The once-only held-out frame gate passed. Its global RMSE values were
0.00880 m for lateral error, 0.05350 rad for heading error, and 0.56531 1/m
for curvature. The right-turn heading stratum had 120 samples and
0.04718 rad RMSE; the edge-lateral stratum had 484 samples and 0.00909 m
RMSE. The artifact is
`artifacts/f10_ppo_visual_v9/lane_rgb_final/final_metrics.json`.

The pre-registered closed-loop development and once-only final gates also
passed. Both C0 `small_loop` and C1 `experiment_loop` completed 4/4
counter-clockwise laps in each split, with zero invalid poses and zero lane
failures. Mean absolute true lateral error on the once-only final split was
0.02446 m for C0 and 0.02574 m for C1. The final evidence is
`artifacts/f10_ppo_visual_v9/lane_closed_loop_gate/final_metrics.json`.

These gates validate the camera-derived lane representation, not PPO policy
competence. PPO C0 must still start from random initialization and pass its
own development/stage-final criteria before C1 is allowed to inherit it.
