# F10-PPO Visual-Lane Curriculum v3 — Curve Recovery Protocol

## Scope

This protocol replaces the failed visual-lane v2 C0 attempt without overwriting
its configuration or artifacts. It is limited to:

```text
C0 small_loop (counter-clockwise)
  -> C1 experiment_loop
  -> STOP for review
```

C1 may start only after the selected C0 checkpoint passes the frozen C0
development and stage-final gates. C2-C4 remain outside this experiment.

## Runtime boundary

The policy input is the same fixed 29-dimensional vector used by visual-lane
v2. Lane state is estimated from the front RGB camera. Duckie and stop-sign
measurements remain YOLO-derived. The policy and value function receive no
simulator geometry or evaluation ground truth.

```text
front RGB
  -> camera lane measurement + lane EKF
  -> YOLO Duckie/stop detections + frozen F9c belief
  -> 29D policy vector
  -> PPO
  -> normalized action
  -> PolicyAction(v_cmd, omega_cmd)
  -> existing differential-drive adapter
```

The observation order and physical normalization scales are frozen in
`configs/f10_ppo_visual_v3.toml`. Pedestrian and stop fields are neutral in C0
and C1. The actor and critic use the same policy-visible vector.

## Direction and action

Both native-map stages start near heading `pi` and drive the route
counter-clockwise. The action remains:

```text
v_cmd     in [0.0, 0.4] m/s
omega_cmd in [-4.0, 4.0] rad/s
```

No wheel controller or action mapping is added.

## Motivation for v3

The v2 development evaluation ended every episode at the first geometrical
yellow-line contact. A retained failure video showed contact during a curve and
a material difference between the camera lane belief and reward-side lane
geometry. Treating every infinitesimal curve contact as irreversible made the
episode gate stricter than the intended behavior.

The v3 change is deliberately confined to reward and termination semantics. It
does not tune the camera estimator on development/final data and does not place
privileged curvature in the policy observation.

## Yellow-line curve-recovery state machine

Reward/evaluation may use true road curvature because it stays outside the
policy boundary. Let:

```text
clearance = lane_center_to_yellow + d_true - ego_half_width
contact   = clearance <= 0
curve     = abs(kappa_true) >= 0.75 1/m
deep      = clearance < -0.035 m
```

The fixed v3 rules are:

1. A shallow contact on a curve is nonterminal and starts a recovery event.
2. The existing continuous yellow proximity/contact penalty remains active.
3. Deep penetration is terminal, including on a curve.
4. Yellow contact on a straight without an active curve-recovery event is
   terminal.
5. After curve contact, at most 15 contact frames are tolerated while leaving
   the curve.
6. Recovery succeeds only after 3 consecutive contact-free frames.
7. If the grace window expires while contact persists, the episode terminates
   as `yellow_recovery_failed`.
8. Lap completion is forbidden while recovery is still pending.

There is no recovery bonus. This prevents reward farming: the desired behavior
is to avoid contact where possible and return cleanly to the lane after an
unavoidable shallow curve contact.

Legacy configurations omit or disable this state machine, preserving their
original immediate-contact termination behavior.

## Reward

The six-component public PPO reward remains:

```text
r = progress + lane + pedestrian + stop + smoothness + terminal
```

All v2 weights remain frozen. Only yellow contact termination changes as
specified above. In C0/C1, pedestrian and stop terms are exactly zero.

Telemetry records per step:

```text
yellow_contact
yellow_recovery_started
yellow_recovery_active
yellow_recovered
reward components
policy-visible 29D vector
evaluation-only true curvature and lane geometry
```

Evaluation-only values are namespaced separately and never enter PPO.

## PPO and training budget

The PPO algorithm and architecture are unchanged:

```text
actor/critic MLP : [256, 256], tanh
rollout           : 1024 steps
batch             : 256
epochs            : 10
learning rate     : 3e-4
gamma             : 0.99
GAE lambda        : 0.95
clip              : 0.20
entropy coeff     : 0.01
value coeff       : 0.50
max grad norm     : 0.50
```

C0 trains for 61,440 environment steps from random initialization. C1 trains
for 40,960 steps and must inherit the selected passing C0 checkpoint, including
optimizer state. Training is logged online to `vnv/DuckiePOMDP` under the v3
group.

## Seeds and contamination control

The v3 C0, C1, later reserved stages, and global-final seeds are new and
disjoint from previous protocol files. Exact allocations are frozen in the v3
config. V2 development outcomes are motivation only; no v2 development or
stage-final seeds are reused for v3 checkpoint selection.

## Gates

Before full C0 training, the current config/source inventory must have:

- reward audit PASS;
- real-simulator reset/memory audit PASS;
- PPO gradient/checkpoint smoke PASS;
- full test suite PASS;
- W&B destination preflight PASS;
- environment/CUDA evidence PASS;
- fresh agent-follows-doc audit PASS.

After C0 training, checkpoints are evaluated only on C0 development seeds.
The pre-registered safety-first rule and C0 acceptance thresholds in the config
select a checkpoint. Stage-final evaluation is once-only. C1 is not launched if
C0 is `LIMITED` or `FAILED`.

After C1, the selected checkpoint is re-evaluated on small_loop and must satisfy
the configured retention tolerance. The experiment stops after the C1 report.

## Scientific interpretation

This protocol does not declare yellow lines irrelevant. It distinguishes:

```text
brief shallow curve contact + verified recovery
```

from:

```text
deep crossing, straight crossing, or failure to recover
```

This matches the requested driving behavior while keeping a falsifiable safety
gate and preserving the camera-belief POMDP boundary.
