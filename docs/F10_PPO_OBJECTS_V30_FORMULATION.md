# F10-PPO C4 — Existence-Gated Pedestrian Belief

## Objective

Complete C4 with the frozen 29-dimensional belief-conditioned PPO interface:

```text
front RGB -> visual lane belief
front RGB -> frozen YOLO -> F9c pedestrian belief / stop belief
public 29D belief -> PPO actor and critic -> [v_cmd, omega_cmd]
```

Simulator truth is restricted to offline teacher labels, reward, and evaluation.
It is absent from the distilled NPZ, actor/critic input, action selection, and
runtime belief preprocessing.

## Diagnosed failure and frozen correction

The V29 diagnostic reproduced a C4 yellow-line failure after the Duckie had
left the world.  At failure, `P(e_ped)=0.00299`, but the EKF conditional
Gaussian still held the last range/bearing.  That conditional Gaussian is not
meaningful when existence is effectively absent.

V30 therefore freezes this semantic rule before training:

```text
P(e_ped) < 0.4  -> complete pedestrian slice uses neutral absent semantics
P(e_ped) >= 0.4 -> preserve the public F9c Gaussian belief unchanged
```

The vector remains 29D and its ordering/scales do not change.  YOLO, F9c EKF,
lane belief, stop belief, PPO architecture, and physical action mapping remain
unchanged.

The threshold is inherited from the training-only F9 confidence/existence gate
and the V28 belief-gated teacher dataset.  It is not fitted on C4 development or
final seeds.

## Teacher and student

The teacher uses privileged state only offline to label the first guided C4
episode and build behavior targets.  The student checkpoint is obtained by:

1. retaining the V22 C4 actor as the driving/stop baseline;
2. importing the V25 teacher-guided critic;
3. updating only actor first-layer pedestrian columns 10:19;
4. enforcing exact neutral-behavior preservation through bias compensation;
5. training on V28 public-belief observations with C2 correction plus C2/C3/C4
   retention and a neutral counterfactual for every active hazard sample.

The frozen step-zero checkpoint must pass C2, C3, and C4 training-only gates.
PPO on-policy training then starts from it; step zero is ineligible for final
selection, and the first eligible candidate is step 1024.

## Frozen training protocol

- Stage: C4 only; C0-C3 are retained checks.
- Environment steps: 4096.
- Checkpoints: 1024, 2048, 3072, 4096.
- PPO actor/critic: unchanged 29D -> 256 -> 256 architecture.
- C4 PPO override: learning rate 2e-7, one epoch, clip 0.01, entropy 0,
  max gradient norm 0.02, target KL 0.002.
- W&B: `vnv/DuckiePOMDP`, group
  `f10-ppo-visual-objects-v30-c4-existence-gated`.
- Development selects only among updated checkpoints using the frozen safety
  rule.  C2 and C3 retention must pass before stage-final C4 is allowed.
- Stage-final seeds are evaluated once.  No global-final or C5/F11 work occurs.

## Gate criteria

C4 PASS requires, on the once-only stage-final split:

- completion >= 0.50;
- progress >= 4.50 m;
- stop completion >= 0.50;
- restart >= 0.50;
- stop violation <= 0.25;
- collision <= 0.10;
- unsafe episode rate <= 0.50;
- lane failure <= 0.25;
- stationary fraction <= 0.65;
- C2 and C3 retention PASS;
- full test suite PASS.
