# F11 R006 — Confirmatory Semantic Intervention Report

## Decision

`R006 = FAILED` at its preregistered factual-action replay integrity gate.

No semantic intervention result was produced, R006 must not be rerun under the
frozen protocol, and R007 remains blocked.

## What executed

The independent preflight passed after two initial audit divergences were fixed:
all bootstrap calculations use seed `2026081606`, and locked arrays cannot be
opened or hashed before the once-only claim. A second fresh agent-follows-doc
audit returned PASS.

The one permitted R006 invocation then:

1. wrote `once_only_launch_claim.json` before opening the R004 trace;
2. hash-verified the frozen R004 public trace and 4,400 sample indices;
3. loaded the unchanged C4 PPO checkpoint;
4. replayed factual deterministic physical actions;
5. stopped before applying or evaluating any semantic intervention because the
   frozen replay tolerance was exceeded.

The simulator was not constructed or rendered. R004 and its locked trajectories
were not rerun.

## Exact failure

The frozen maximum replay tolerance was `1e-6`. Across 4,400 states:

| Diagnostic | Result |
|---|---:|
| Maximum velocity error | `1.31e-7 m/s` |
| Maximum yaw error | `1.85e-6 rad/s` |
| P99 maximum action error | `1.13e-6` |
| Rows above tolerance | `119 / 4400` (`2.70%`) |
| Worst state | seed `177106`, step `1956` |

The magnitude is consistent with a small CUDA floating-point replay difference,
not a changed policy: velocity remains well inside the gate and yaw exceeds it
by only `8.48e-7 rad/s`. Nevertheless, this interpretation cannot override the
threshold after locked execution. The correct classification is `FAILED`, not
PASS or LIMITED.

## Consequences

- No `pedestrian_absent`, `stop_absent`, `lane_centered`, or auxiliary holdout
  effect may be claimed from R006.
- The R003 development intervention evidence remains development-only.
- R004 attribution remains PASS and unchanged.
- R007 closed-loop behavioral intervention remains blocked.
- There is no automatic R006b and no threshold relaxation.

A future recovery requires explicit review and a newly named protocol. It must
justify its replay tolerance independently, retain this failed attempt, and must
not present itself as the original preregistered R006.

## Integrity and tests

- Once-only config SHA256:
  `d56b9f4a320736d8cf061791ffcc89fdb69d0074f1f729c22be7c550e510e9f2`
- Claim SHA256:
  `5c8e379d2e38c4aec682c83a3e068e0935b52c9015a954081ef1765008aec660`
- Failure marker SHA256:
  `b5449427b498a9da811da2ad2845371add50d4c2381755db5834ea0a7d6243af`
- One-shot log SHA256:
  `2a14ea9a7493f9e19ea40e5265d9c668288fcddeca66424107042006ade70f0d`
- Full active suite: `664 passed, 0 failed, 0 skipped` with 426 warnings.
- No R007 artifact exists.

## Read-only verification

From repository root:

```bash
PYTHONHASHSEED=0 \
PYTHONPATH=src:/home/pannntastic/aivnv/duckie/src \
CUDA_VISIBLE_DEVICES=0 \
/home/pannntastic/aivnv/duckie/.venv/bin/python \
  experiments/verify_f11_r006_failure.py \
  --config configs/f11_ppo_explanation_r006_v1.toml
```

Expected result: exit zero while reporting `classification=FAILED`, the frozen
`1e-6` tolerance, measured maximum replay error above that threshold, no
scientific intervention artifact, no rerun permission, and no R007.
