# F11 R002b — Phase-Conditioned Distributional IG Protocol

Status before execution: **PREREGISTERED / DEVELOPMENT ONLY**.

R001 remains PASS, historical fixed-reference R002 remains LIMITED, and R003
remains PASS. R004 and seeds `177101–177108` remain locked while R002b runs.

## Scientific question

R002b asks whether group-level actor attribution is stable across independent
Monte-Carlo draws from one semantically defined reference distribution. It
does not search for a fourth preferred baseline and does not replace or delete
the negative R002 result.

## Frozen target

The explained functions are the deterministic PPO actor distribution means
mapped to physical `v_cmd_mps` and `omega_cmd_rad_s`. The PPO, MobileNet lane
estimator, YOLO detector, belief updaters, normalization, 29D ordering, and
checkpoint are unchanged.

## Reference distribution

Only the existing public development trace from seeds `177001–177004` is used.
For every factual sample, each reference is sampled from:

1. the same public phase under the frozen amended phase definition;
2. a different development seed/trajectory;
3. the exact normalized public 29D actor-input space.

No privileged truth, simulator pose, bounding box, IoU, or locked seed is used.
Six deterministic independent draws are used. Each draw averages exact path IG
over four independently sampled cross-seed references per factual input. This
is a Monte-Carlo distributional/multi-reference IG estimate. Completeness is
checked against `F(x) - mean(F(reference))` for each draw.

## Primary analysis

The primary statistics are the six semantic groups:

`Lane | Ego | StopLine | Pedestrian | Stop | PreviousAction`.

Individual 29D feature results remain secondary. Agreement is evaluated among
the six independent draws from the same reference protocol, using the same
point thresholds as R002:

- median group Spearman at least `0.5`;
- signed-group agreement at least `0.6`;
- top-group agreement at least `0.33`;
- median group-share L1 at most `0.75`;
- completeness median at most `1e-4` and P99 at most `1e-3`.

Cross-seed bootstrap 95% confidence intervals are reported with the development
seed as the resampling unit. They are uncertainty diagnostics and are not used
to weaken the preregistered point thresholds.

## Decision rule

- If every frozen gate passes: `R002b=PASS`; R004 may be unlocked but is not
  executed automatically.
- Otherwise: `R002b=LIMITED`; no R002c is permitted, IG becomes secondary
  baseline-sensitive evidence, and semantic interventions become the primary
  explanation path.

The experiment refuses to overwrite existing R002b artifacts and never has a
mode that collects or opens locked evaluation episodes.

