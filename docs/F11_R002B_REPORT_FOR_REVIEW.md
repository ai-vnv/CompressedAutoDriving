# F11 R002b — Distributional IG Report for Review

## Decision

`R002b = PASS` on development seeds only.

Historical R002 remains `LIMITED`; its negative result is not overwritten.
R001 and R003 remain `PASS`. R004 is now **unlocked but not run**, and locked
seeds `177101–177108` remain unopened.

## Frozen protocol

R002b used phase-conditioned distributional multi-reference IG. For every one
of 2,200 factual public 29D states, each reference came from the same public
phase and a different development trajectory. Six deterministic independent
draws were evaluated; each draw averaged exact path IG over four references per
input. The PPO actor targets remained deterministic physical `v_cmd_mps` and
`omega_cmd_rad_s`.

This was the only R002b protocol. No baseline search, threshold change, PPO
change, perception change, retraining, or locked-seed access occurred.

## Results

| Gate | Frozen threshold | Result | Status |
|---|---:|---:|---|
| Median six-group Spearman | >= 0.5 | 1.000 | PASS |
| Signed-group agreement | >= 0.6 | 0.647 | PASS |
| Top-group agreement | >= 0.33 | 1.000 | PASS |
| Median group-share L1 | <= 0.75 | 0.0313 | PASS |
| Completeness median | <= 1e-4 | max 2.99e-6 | PASS |
| Completeness P99 | <= 1e-3 | max 9.31e-5 | PASS |

There were 600 draw-pair/context comparisons. The fifth percentile group
Spearman was 0.943 and the 95th percentile group-share L1 was 0.0905.

Seed-cluster bootstrap 95% intervals were:

- median Spearman: `[1.000, 1.000]`;
- sign agreement: `[0.629, 0.668]`;
- top-group agreement: `[1.000, 1.000]`;
- median group-share L1: `[0.0273, 0.0348]`.

All five registered public phases were represented. Every sampled reference
matched the factual public phase, came from another development seed, and had
zero self-reference matches.

## Interpretation

R002 showed that three semantically different fixed baselines answer different
questions and yield baseline-sensitive rankings. R002b shows that attribution
is stable across repeated draws from one preregistered distributional-reference
question. These findings are complementary; R002b does not make single-reference
IG baseline-invariant.

The R002b estimand is explicitly within-phase, cross-trajectory attribution.
It must not be described as an absolute causal explanation or as evidence that
the three historical baselines agree. R003 remains the action-level semantic
intervention evidence. Closed-loop behavioral claims still require later gates.

## Reproducibility and leakage boundary

- Development seeds: `177001–177004`.
- Locked seeds: `177101–177108`, unopened.
- Reference tensor: `(6, 4, 2200, 29)`.
- Attribution tensor: `(6, 2, 2200, 29)`.
- Frozen PPO checkpoint SHA256:
  `02e898ce12d71f97016d50ed8a40574807e6d2fd995fc9f0dcd24f357f2c6250`.
- R002b config SHA256:
  `7610fc0d8bdeb5b4df5f4f66c6f3f7587f56d0042af058b6e9663fc847b95925`.
- No privileged/GT/world-pose/bbox/IoU field is stored.
- No R004 artifact exists.

The first full-suite command accidentally included historical `_archive`
tests. Its failing log is retained under `_archive`; the correct active-suite
command was then run explicitly against `tests/` and produced:

`655 passed, 0 failed, 0 skipped, 426 warnings`.

## Read-only verification

From the repository root:

```bash
PYTHONHASHSEED=0 \
PYTHONPATH=src:/home/pannntastic/aivnv/duckie/src \
/home/pannntastic/aivnv/duckie/.venv/bin/python \
experiments/verify_f11_r002b.py \
--config configs/f11_ppo_explanation_r002b_v1.toml
```

Expected result: verifier exit code zero, `classification=PASS`, 2,200 samples,
six draws, four references per input, no locked seeds, no R004, and 655 active
tests passing.

## Gate status

`R001 PASS -> R002 LIMITED (historical) -> R002b PASS -> R003 PASS`.

R004 is eligible for a separately reviewed, once-only locked-seed execution
using this R002b protocol unchanged. R004 was not started by this run.

