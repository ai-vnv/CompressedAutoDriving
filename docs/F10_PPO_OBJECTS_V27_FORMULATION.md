# F10-PPO V27 — Conditional Retention Distillation

## Scope

V27 is a C4-only correction after V25 showed that global behavior cloning
fixed C2 but catastrophically damaged C3/C4.  It does not change the frozen
29-dimensional policy representation, RGB lane estimator, YOLO detector,
F9c pedestrian belief, stop belief, action adapter, PPO architecture, or
reward semantics.

The runtime boundary remains:

```text
front RGB -> lane belief + YOLO/F9c/stop belief -> 29D vector
          -> PPO actor and critic -> [v_cmd, omega_cmd]
```

Privileged truth is used offline by the teacher and for reward/evaluation.
It is absent from the student NPZ and from actor/critic runtime inputs.

## Teacher and retention construction

The frozen V24 teacher dataset supplies C2 correction actions and supervised
value targets.  V27 replaces the C3 and C4 action targets with the exact raw
actor means from the retained V22 C4 checkpoint.  C2, C3, and C4 receive equal
total loss mass.  Thus the teacher may change pedestrian-only behavior while
explicitly rehearsing the already competent stop and combined behavior.

One small development sweep was performed only on historical/training seeds:
1, 2, 4, and 8 distillation epochs.  Eight epochs was selected because the
training-only closed-loop gate passed C2, C3, and C4 simultaneously.  No V27
development or stage-final seed was inspected for this choice.

The resulting step-zero checkpoint is frozen by SHA256.  Substantive training
loads it through the precomputed behavior-checkpoint boundary; it does not
repeat the teacher fit.

## PPO update

Canonical PPO remains unchanged.  V27 performs 4,096 online environment steps
with checkpoints at 1,024-step intervals.  To prevent rapid forgetting, the
C4 update uses learning rate `2e-7`, one epoch, clip range `0.01`, entropy
coefficient `0`, maximum gradient norm `0.02`, and target KL `0.002`.
Development selection requires an updated checkpoint at step >= 1,024.

## Seeds

- train: 174001–174012
- development: 174101–174104
- once-only stage final: 174201–174204

All are disjoint and outside the frozen historical range through 173999.

## Gates

Before substantive training:

1. the step-zero checkpoint must pass 4 C2, 4 C3, and 2 C4 training episodes;
2. reward, scenario, reset-memory, W&B, smoke, and full-test evidence must pass;
3. source/evidence hashes must be frozen;
4. a fresh independent agent-follows-doc audit must pass.

After training, development selection is safety-first.  Retention must pass for
C0–C4 before the C4 stage-final seeds can be run exactly once.  This protocol
stops after C4 and does not run the separate global-final/F11 stages.
