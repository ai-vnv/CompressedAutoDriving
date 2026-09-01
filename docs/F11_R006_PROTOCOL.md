# F11 R006 — Once-Only Confirmatory Semantic Action Intervention Protocol

## Scientific question

R006 tests whether the frozen PPO actor is functionally dependent on the same
semantic public inputs validated in R003 and attributed in R004. It is an
offline, paired policy-input experiment:

`factual public 29D -> frozen PPO` versus
`semantically intervened public 29D -> frozen PPO`.

The allowed wording is **confirmatory counterfactual policy dependence**. R006
does not establish a causal effect in the simulator or physical world; that
requires the separately gated closed-loop R007 experiment.

## Frozen data and actor boundary

- Frozen PPO C4 checkpoint: SHA256
  `02e898ce12d71f97016d50ed8a40574807e6d2fd995fc9f0dcd24f357f2c6250`.
- Source: the already frozen R004 public-only trace from locked seeds
  `177101–177108`.
- Exactly the same 4,400 stride-selected states used by R004 are evaluated.
- The simulator and locked seeds are **not rendered again**.
- Actor targets are deterministic physical `v_cmd` and `omega_cmd`; critic
  value is descriptive only.
- No GT, privileged state, world pose, bbox, or IoU enters the intervention or
  PPO evaluation.

## Frozen semantic interventions

R006 reuses the R003 operators unchanged:

1. `pedestrian_absent`: replace the complete pedestrian 9D tuple with the
   runtime neutral-absence tuple.
2. `stop_absent`: replace stop-line/sign semantics with the neutral sign tuple,
   neutral stop-line, and one-hot stop mode `NONE`.
3. `lane_centered`: set only lane lateral and heading means to zero while
   preserving validity, uncertainty, and curvature.
4. `lane_low_confidence`: apply the R003-frozen validity and uncertainty bounds.
5. `previous_action_neutral`: set only the two previous-action fields to zero.
6. `sham`: bitwise-identical actor input.

No feature-specific operator was added after inspecting R004. The last two
non-sham auxiliary interventions are reported descriptively and are not used to
rescue a failed confirmatory gate.

## Frozen phases and hypotheses

Phase assignment is copied exactly from R004 and uses only public fields.

### H-pedestrian

On `pedestrian_relevant`, `pedestrian_absent` must increase commanded velocity:

- mean `delta v >= 0.10 m/s`;
- seed-cluster-bootstrap 95% CI lower bound `> 0`;
- at least 75% of paired deltas are positive.

Across all non-pedestrian phases, mean absolute `delta v <= 0.01 m/s`.
Pedestrian `delta omega` has no preregistered sign because left/right geometry
can reverse the appropriate direction. Its magnitude and public-bearing strata
are reported without becoming a gate.

### H-stop

On `stop_required`, `stop_absent` must increase commanded velocity:

- mean `delta v >= 0.05 m/s`;
- seed-cluster-bootstrap 95% CI lower bound `> 0`;
- at least 75% of paired deltas are positive.

`pedestrian_relevant` is the frozen stop negative-control phase, with mean
absolute `delta v <= 0.02 m/s`. Nominal and curve phases are not stop-negative
controls because a visible sign/approach context may legitimately be public
there.

### H-lane

On `lane_curve`, `lane_centered` must produce:

- mean absolute `delta omega >= 0.10 rad/s`;
- seed-bootstrap lower bound on mean absolute `delta omega >= 0.10 rad/s`;
- effect at least 1.5 times its `pedestrian_relevant` negative-control effect.

Signed yaw is descriptive because counter-clockwise route geometry and lane
error sign can cause cancellation.

### Sham

The maximum absolute effect across velocity, yaw, and critic must remain at or
below `1e-7`.

## Statistical protocol

- Pairing unit: the same factual state before and after one semantic operator.
- Uncertainty unit: locked evaluation seed, not individual frame.
- Bootstrap: 2,000 equal-seed cluster resamples, seed `2026081606`, 95% CI.
- All six interventions and all five public phases are reported.
- Primary claims use physical action deltas; critic deltas are appendix data.
- Thresholds cannot change after the R006 once-only claim is written.

## Classification

- `PASS`: all structural integrity and confirmatory criteria pass.
- `LIMITED`: runtime integrity is intact but one or more scientific effect gates
  fail. Thresholds are not weakened and there is no R006b.
- `FAILED`: source/hash/schema/replay/model-integrity boundary fails.

The claim file is written before the frozen R004 trace is opened. An existing
R006 output directory prevents rerun. R007 remains blocked and must not be
started automatically.

## Documented invocation

From repository root:

```bash
export PYTHONHASHSEED=0
export PYTHONPATH=src:/home/pannntastic/aivnv/duckie/src
export CUDA_VISIBLE_DEVICES=0

/home/pannntastic/aivnv/duckie/.venv/bin/python \
  experiments/run_f11_r006_once.py \
  --config configs/f11_ppo_explanation_r006_v1.toml \
  --mode preflight

set -o pipefail
/home/pannntastic/aivnv/duckie/.venv/bin/python \
  experiments/run_f11_r006_once.py \
  --config configs/f11_ppo_explanation_r006_v1.toml \
  --mode once \
  2>&1 | tee artifacts/f11_ppo_explanation_v2_r006_once.log
```

The second command is once-only and may be issued only after preflight and
independent agent-follows-doc review of this frozen protocol.
