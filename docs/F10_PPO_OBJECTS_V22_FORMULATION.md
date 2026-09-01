# F10-PPO V22 — C4 Cumulative Public-Belief DAgger

V21 is retained as a negative result: all eight updated checkpoints failed the
combined-task development gate.  The dominant failure was 100% stop violation,
with zero completion and excessive stationary behavior.  V21 stage-final was
not run.

V22 starts again from the frozen passing V20 C3 checkpoint.  Before canonical
PPO updates, its actor is distilled on three training-only roles: the cumulative
C3 behavior anchor, successful C4 teacher trajectories, and states visited by
the failed V21 C4 learner relabelled by the same teacher.  Teacher and DAgger
roles have equal loss mass; the cumulative C3 anchor has three times that mass
to prevent the stop-only skill from being overwritten.  This weighting was
frozen only after the initial equal-mass attempt passed C4 but failed the
training-only C3 retention gate; that failed attempt is archived and excluded.
Both teacher and learner labels consume only the fixed 29D normalized public
belief vector.  No simulator truth is an input or target.

The YOLO weights, camera geometry, F9c updater, visual lane belief,
actor/critic architecture, action adapter, reward, combined scenario, and
acceptance thresholds remain frozen.  A detector-boundary V1 plausibility gate
is added because the new combined audit proved that yellow lane dashes can be
classified as Duckie above confidence 0.40.  At 640x480, all 1,193 correct F9
calibration boxes and all 75 visible C4 audit boxes end at or above image row
232.42, while 51 of 52 absent-object false rows end below row 253.04.  The one
remaining audit row is the boundary frame that still renders the exiting
Duckie before simulator removal.  Therefore Duckie boxes with bottom y > 240
are excluded before F9c.  The filter consumes class and bbox only; no GT,
scenario time, or privileged object existence enters runtime.

```text
front RGB -> visual lane belief + YOLO -> F9c pedestrian/stop belief
          -> 29D vector -> PPO actor and critic
```

The DAgger source uses only V21 training seeds 169001 and 169002.  V22 uses new
disjoint train/dev/final seeds 170001..170204.  Step zero is gated on
training-only C3 retention and C4 competence but remains ineligible for
checkpoint selection.  At least one normal PPO update (step 1024) must pass the
unchanged development gate before retention or once-only stage-final may run.

The conservative on-policy update uses 4,096 steps, checkpoints every 1,024,
learning rate 2e-6, one epoch, clip 0.02, entropy coefficient 0, max gradient
norm 0.05, and target KL 0.005.  This is still canonical feed-forward PPO; the
smaller update protects the already competent distilled behavior rather than
changing the algorithm.

No global-final, explanation, optimizer study, detector retraining, or F11 work
is in scope.  Stop after C4 stage-final classification.
