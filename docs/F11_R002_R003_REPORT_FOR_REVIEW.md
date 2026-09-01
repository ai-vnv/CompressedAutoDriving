# F11 R002/R003 Development Report

## Gate status

```text
R001  PASS
R002  LIMITED
R003  PASS
R004  BLOCKED
```

No locked evaluation seed was opened. This report contains development evidence
only and does not claim final PPO feature use.

## Data and frozen boundary

- Development seeds: `177001–177004`
- Locked/unopened evaluation seeds: `177101–177108`
- Real visual C4 frames: `8,800`
- Fixed-stride explanation samples: `2,200`
- PPO checkpoint SHA256: `02e898ce12d71f97016d50ed8a40574807e6d2fd995fc9f0dcd24f357f2c6250`
- Actor targets: physical `v_cmd` and `omega_cmd` derived from deterministic
  distribution means before sampling
- Primary groups: `Lane | Ego | StopLine | Pedestrian | Stop | PreviousAction`
- Privileged truth stored: no

The five public phases have support after the development-only phase freeze:

| Public phase | Samples |
|---|---:|
| nominal | 524 |
| lane curve | 589 |
| pedestrian relevant | 476 |
| stop required | 456 |
| stop satisfied near stop line | 155 |

The initial phase pilot revealed that persistent `stop_mode_satisfied` swallowed
the post-stop route. Before inspecting attribution, the definition was frozen to
require `abs(stop_line_distance_m) <= 0.5`; the lane-curve threshold was frozen
at `abs(kappa) >= 1.5`. The raw public trace was not recollected. The amendment
is recorded in `r002/phase_definition_amendment.json`.

## R002 — baseline robustness

R002 is **LIMITED**. IG numerical completeness passes for all combinations of:

```text
episode-reset baseline
public-median baseline
semantic-neutral-hazard baseline
×
v_cmd / omega_cmd
```

All completeness medians are below `2.27e-5`; all P99 values are below
`1.54e-4`, comfortably inside the frozen `1e-4` median and `1e-3` P99 limits.

However, the semantic-group interpretation changes materially with baseline:

| Robustness statistic | Result | Frozen requirement | Gate |
|---|---:|---:|---|
| Median pairwise group Spearman | 0.143 | >= 0.50 | FAIL |
| Mean signed-group agreement | 0.552 | >= 0.60 | FAIL |
| Top-group pair agreement | 0.300 | >= 0.33 | FAIL |
| Median group-share L1 distance | 0.896 | <= 0.75 | FAIL |

Therefore no baseline is selected as the preferred explanation, and final R004
attribution remains blocked. Completeness alone is not treated as sufficient.

## R003 — semantic intervention operators

R003 is **PASS** as an operator-validation gate. Every intervention:

- produces finite, schema-valid 29D inputs within frozen normalization bounds;
- changes only registered semantic fields;
- uses complete neutral tuples instead of arbitrary all-zero deletion;
- consumes only saved public policy vectors;
- leaves locked evaluation seeds unopened.

The sham intervention is bitwise identical at the actor input and causes exactly
zero action/value change.

Development-only directional diagnostics include:

| Intervention / public phase | Mean delta v_cmd | Mean abs delta omega_cmd |
|---|---:|---:|
| pedestrian absent / pedestrian relevant | +0.306 m/s | 1.804 rad/s |
| stop absent / stop required | +0.219 m/s | 1.203 rad/s |
| stop absent / stop satisfied | +0.155 m/s | 0.797 rad/s |
| lane centered / lane curve | +0.011 m/s | 0.377 rad/s |
| sham / nominal | 0 | 0 |

These values are **semantic intervention evidence / counterfactual policy
dependence on development data**, not real-world causal claims and not final
holdout results.

Two failed R003 audit attempts are retained under `_archive/`: one exposed a
float32 sham round-trip and one exposed an inverted negative boolean in the gate
classifier. Both were audit-code defects; neither changed policy, data,
thresholds, or scientific results.

## Regression and decision

Full suite: **651 passed, 0 failed, 0 skipped** (426 warnings).

R003 is ready for later use, but the overall unlock condition is not met because
R002 is LIMITED. Do not run R004, inspect locked seeds, select a favorable
baseline, or weaken the registered agreement thresholds. The next action must be
a protocol review of baseline-sensitive IG, not final attribution.

## Artifacts

- `artifacts/f11_ppo_explanation_v2/r002/development_trace.npz`
- `artifacts/f11_ppo_explanation_v2/r002/integrated_gradients.npz`
- `artifacts/f11_ppo_explanation_v2/r002/baseline_robustness.json`
- `artifacts/f11_ppo_explanation_v2/r002/group_attribution_development.csv`
- `artifacts/f11_ppo_explanation_v2/r002/phase_definition_amendment.json`
- `artifacts/f11_ppo_explanation_v2/r003/semantic_interventions.npz`
- `artifacts/f11_ppo_explanation_v2/r003/intervention_effects.csv`
- `artifacts/f11_ppo_explanation_v2/r003/intervention_validation.json`
- `artifacts/f11_ppo_explanation_v2/r002_r003_full_tests.log`

Read-only verifier:

```bash
export PYTHONHASHSEED=0
export PYTHONPATH=src:/home/pannntastic/aivnv/duckie/src
/home/pannntastic/aivnv/duckie/.venv/bin/python \
  experiments/verify_f11_r002_r003.py \
  --config configs/f11_ppo_explanation_development_v2.toml
```
