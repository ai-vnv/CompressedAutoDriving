# F11 R004 — Once-Only Locked Actor Attribution Protocol

Status before execution: **FROZEN / READY — NOT RUN**.

R001 PASS, historical fixed-reference R002 LIMITED, R002b PASS, and R003 PASS
are frozen. This document and its configuration must be finalized before any
seed `177101–177108` is opened.

## Final estimator

R004 explains the deterministic PPO actor means mapped to physical
`v_cmd_mps` and `omega_cmd_rad_s`. For every factual state, draw `d` averages
four exact path-IG estimates:

`IG_draw_d(x) = mean_{j=1..4} IG(x; reference_{d,j})`.

The final attribution is fixed as:

`IG_final(x) = mean_{d=1..6} IG_draw_d(x)`.

Thus all 24 references contribute equally. No draw selection, median-draw
choice, voting, or post-result estimator change is permitted.

## Locked reference sampling

Factual and reference states both come only from locked seeds `177101–177108`.
For a factual state from seed `i`, every reference must:

1. have the same public phase;
2. come from a seed other than `i`;
3. be a normalized public 29D state;
4. use one of four distinct reference seeds within each draw.

Six deterministic draw seeds are frozen in the configuration. Sampling is
without replacement within each draw. Development references are forbidden.

If a factual phase has fewer than four other locked seeds with that phase,
that phase and R004 are classified `LIMITED`. There is no fallback, pooling
across phases, reuse of development references, sampling with replacement, or
threshold adjustment.

## Once-only boundary

The executable creates `once_only_launch_claim.json` before the first locked
environment reset. If the R004 directory or launch claim already exists, it
refuses to run. A technical preflight may validate code/checkpoint contracts,
but cannot instantiate the environment with locked seeds or write R004 data.

The trace stores only public policy information and RGB hashes. Privileged
truth may not enter or be stored in collection, reference construction,
attribution, or actor evaluation.

## Frozen gates

The R002b thresholds remain unchanged:

- completeness median <= `1e-4` and P99 <= `1e-3`;
- median six-group Spearman >= `0.5`;
- signed-group agreement >= `0.6`;
- top-group agreement >= `0.33`;
- median group-share L1 <= `0.75`.

Every required public phase must have sufficient sample and cross-seed support.
Cross-seed bootstrap 95% intervals use locked evaluation seed as the cluster
unit. Failed thresholds produce `LIMITED`; there is no “almost pass”.

## Reporting semantics

Primary results are six semantic groups:

`Lane | Ego | StopLine | Pedestrian | Stop | PreviousAction`.

Individual 29D features are secondary. R004 establishes holdout attribution
and its numerical stability, not causal correctness. R003/R006 provide
action-level intervention evidence, while R007 is required for closed-loop
behavioral claims.

R004 stops after producing and verifying its final artifacts. R006/R007 are not
started automatically.

