# F11 PPO Explanation Formulation

## Scope

F11 explains exactly the frozen C4 belief-conditioned PPO checkpoint. It does
not retrain PPO, alter YOLO/MobileNet/EKF, run the global-final holdout, or
optimize the policy.

Frozen inputs:

- policy config SHA256: `85e2cbd321e2db53de270e3c0b885a723137d17e94af3640ed5a8a9f917fe829`;
- selected checkpoint SHA256: `02e898ce12d71f97016d50ed8a40574807e6d2fd995fc9f0dcd24f357f2c6250`;
- actor and critic input: the same normalized 29-dimensional public belief
  vector used at runtime;
- physical action: `v_cmd in [0, 0.4] m/s`, `omega_cmd in [-4, 4] rad/s`.

## Two recorded explanation ideas

### Method 1 — Integrated Gradients (execute now)

Apply Integrated Gradients (IG) to the PPO MLP input. Explain three scalar
targets independently:

1. deterministic physical `v_cmd`;
2. deterministic physical `omega_cmd`;
3. critic value `V(b_t)`.

For target `F`, input `x`, and baseline `x'`:

```text
IG_i(x) = (x_i - x'_i) integral_0^1 dF(x' + alpha(x-x')) / dx_i d alpha
```

The numerical integral uses 128 trapezoidal path intervals. The baseline for
every frame is the normalized public observation at the start of that same
episode. This baseline is on-manifold, deterministic, uses no privileged
truth, and makes the interpretation explicit: which belief changes since the
episode start explain the current action/value?

Qualitative output is a feature-by-time signed attribution heatmap for the
pre-registered first explanation seed (`176001`), aligned with action and
public pedestrian/stop-belief events.

Quantitative output includes:

- mean absolute attribution and attribution share per feature and semantic
  group;
- frequency with which each feature appears in the top five;
- adjacent-frame Spearman stability of absolute attribution rankings;
- IG completeness residual;
- deletion faithfulness: replace IG-ranked features with the episode-start
  baseline and compare action change against bottom-ranked and 16 deterministic
  random rankings.

Attribution is computed on normalized network inputs. Action-target IG is
reported in the corresponding physical output units. Attribution does not
claim that a feature is causally important merely because its gradient is
large; deletion faithfulness is included as a limited diagnostic.

### Method 2 — Semantic counterfactual belief ablation (record only)

In a later gate, intervene on meaningful belief groups—such as neutralizing
pedestrian belief, neutralizing stop belief, increasing uncertainty, or
neutralizing lane belief—and measure action and closed-loop behavior changes.
This is intentionally not executed in the present task, so Method 1 results
cannot be post-hoc tuned against Method 2.

## Runtime and leakage boundary

Explanation trajectories use the normal C4 chain:

```text
front RGB
  -> MobileNet lane measurement and lane belief
  -> frozen YOLO object detections
  -> frozen pedestrian/stop belief updates
  -> normalized 29D public vector
  -> frozen PPO actor and critic
```

The explanation dataset stores only the policy-visible normalized and physical
fields, deterministic policy outputs, and runtime event labels derived from
those public fields. It must not store `evaluation_gt`, simulator world pose,
true object state, GT boxes, collision geometry, or reward-only truth.

## Explanation-only seeds

Seeds `176001-176004` are disjoint from training, development, stage-final,
global-final, detector, and historical F9/F10 seeds. They are explanation-only:
they cannot be used for training, checkpoint selection, reward tuning, or
normalization.

## Pre-registered acceptance checks

Method 1 is valid only if:

1. the config and checkpoint hashes match before and after analysis;
2. the PPO parameter tensors are byte-identical before and after IG;
3. all four explanation seeds produce finite 29D observations and outputs;
4. no privileged fields enter the saved explanation trajectory;
5. median absolute IG completeness residual is at most `1e-4` and P99 is at
   most `1e-3` for every target;
6. figures are generated from saved machine-readable artifacts in both PNG
   and PDF form;
7. faithfulness is reported honestly even if IG-ranked deletion does not beat
   random deletion.

Failure of the faithfulness comparison classifies Method 1 as `LIMITED`, not
as a reason to tune the checkpoint or explanation seeds.

## Spatial BEV rendering of Method 1

The pre-registered qualitative seed `176001` is also replayed once to place the
already-computed actor attributions on the `experiment_loop` bird's-eye map.
The replay must reproduce every frozen public observation and deterministic
action within `1e-6`; otherwise rendering stops.

World pose is deliberately stored in a second artifact and is read only after
the actor has selected its action. It is used solely to join `(seed, step)` to
a map location. It is never appended to the 29D policy vector, used to select
an action, or supplied to Integrated Gradients. The BEV therefore answers
"where on the route was this public-belief attribution observed?" without
changing what the policy knew.

The four panels show:

1. signed total IG for physical `v_cmd`;
2. the semantic group with the largest absolute `v_cmd` attribution;
3. signed total IG for physical `omega_cmd`;
4. the semantic group with the largest absolute `omega_cmd` attribution.

The Duckie crossing path, stop line, and stop-sign location are evaluation-only
map references. The map uses world `(x,z)` in metres with `z` increasing down,
matching the simulator BEV orientation and counter-clockwise route.
