# F10-PPO V23 — Cumulative policy rehearsal for C4

V22 is retained as a development negative result.  It passed C4 development
with every updated checkpoint and preserved C3, but the selected checkpoint
failed C2 retention: collision and unsafe-episode rates were both 0.50.  No V22
C4 stage-final seed was used.

The failure was traced to the warm-start target, not the POMDP boundary.  The
V22 anchor was an older C3 teacher dataset whose own retained C2 behavior was
already limited; increasing that dataset's weight did not preserve the actual
passing V20 C3 policy.  V23 therefore rehearses the frozen V20 policy itself on
C2 and C3 training trajectories, then adds successful public-belief C4 teacher
trajectories and C4 learner-state DAgger corrections.

```text
C2 V20 policy rehearsal     2x role mass
C3 V20 policy rehearsal     2x role mass
C4 public-belief teacher    1x role mass
C4 DAgger learner states    1x role mass
```

Every target consumes only the fixed 29D normalized public-belief vector.
Runtime remains:

```text
front RGB -> visual lane belief + frozen YOLO/F9c pedestrian/stop belief
          -> 29D vector -> PPO actor and critic -> [v_cmd, omega_cmd]
```

Simulator truth is used only by reward/evaluation.  The bbox-only Duckie image
domain filter from V22 remains frozen.  The step-zero gate is strengthened: it
must pass C2 pedestrian safety/progress, C3 stop/hold/restart, and C4 combined
competence on training-only seeds before substantive PPO is permitted.

The canonical PPO update remains conservative and unchanged from V22: 4,096
steps, checkpoints every 1,024, learning rate 2e-6, one epoch, clip 0.02,
entropy coefficient 0, maximum gradient norm 0.05, and target KL 0.005.  Step
zero remains ineligible; an updated checkpoint must pass development, the full
C0-C4 retention matrix, and once-only C4 stage-final.

V23 uses new disjoint seeds 171001..171204.  No global-final, explanation,
detector retraining, policy optimization study, or F11 work is in scope.  Stop
after C4 stage-final classification.
