# F11 R004 — Once-Only Final Actor Attribution Report

## Decision

`R004 = PASS` on the once-only locked evaluation seeds `177101–177108`.

R001 PASS, historical R002 LIMITED, R002b PASS, and R003 PASS remain unchanged.
R006/R007 were not started.

## Execution boundary

The once-only claim was written before the first locked reset. The frozen C4
runtime then produced 17,600 public deployment frames through:

`RGB -> MobileNet lane belief + YOLO/F9c object beliefs -> public 29D -> PPO`.

Attribution used 4,400 stride-selected factual states. For each factual state,
six draws each selected four same-phase references from four distinct other
locked seeds. The final estimator is the exact equal mean of all six draw means,
equivalent to 24 equally weighted reference IG estimates. No development
reference or fallback was used.

All five phases occurred in all eight locked seeds, so the preregistered
insufficient-support rule was not triggered.

## Frozen gate results

| Gate | Threshold | Locked result | Status |
|---|---:|---:|---|
| Median six-group Spearman | >= 0.5 | 1.000 | PASS |
| Signed-group agreement | >= 0.6 | 0.661 | PASS |
| Top-group agreement | >= 0.33 | 1.000 | PASS |
| Median group-share L1 | <= 0.75 | 0.0294 | PASS |
| Final completeness median, velocity | <= 1e-4 | 4.60e-7 | PASS |
| Final completeness P99, velocity | <= 1e-3 | 3.59e-5 | PASS |
| Final completeness median, yaw | <= 1e-4 | 4.35e-6 | PASS |
| Final completeness P99, yaw | <= 1e-3 | 8.30e-5 | PASS |

There were 1,200 draw-pair/context comparisons. Fifth-percentile Spearman was
0.943 and 95th-percentile share L1 was 0.0846.

## Primary holdout attribution

Values below are mean absolute six-group shares. They answer which public
groups contributed to the deterministic actor outputs relative to the frozen
phase-conditioned reference distribution.

### Linear velocity

| Rank | Group | Overall share | Seed-bootstrap 95% CI |
|---:|---|---:|---:|
| 1 | Lane | 0.315 | [0.308, 0.323] |
| 2 | Stop | 0.237 | [0.234, 0.241] |
| 3 | Pedestrian | 0.191 | [0.188, 0.195] |
| 4 | PreviousAction | 0.124 | [0.124, 0.125] |
| 5 | Ego | 0.069 | [0.068, 0.070] |
| 6 | StopLine | 0.063 | [0.062, 0.064] |

### Angular velocity

| Rank | Group | Overall share | Seed-bootstrap 95% CI |
|---:|---|---:|---:|
| 1 | Lane | 0.533 | [0.531, 0.535] |
| 2 | Pedestrian | 0.196 | [0.192, 0.200] |
| 3 | Stop | 0.147 | [0.144, 0.150] |
| 4 | PreviousAction | 0.045 | [0.0447, 0.0457] |
| 5 | Ego | 0.0448 | [0.0442, 0.0454] |
| 6 | StopLine | 0.0334 | [0.0328, 0.0340] |

## Phase-specific findings

The phase-conditioned results are more informative than the global average:

- `pedestrian_relevant`: Pedestrian contributes `0.881` of velocity share and
  `0.903` of yaw share.
- `stop_required`: Stop contributes `0.396` of velocity share; Lane contributes
  `0.437` and Stop `0.376` of yaw share.
- `lane_curve`: Lane contributes `0.429` of velocity share and `0.768` of yaw
  share.
- `stop_satisfied`: Lane contributes `0.650` of velocity share and `0.885` of
  yaw share.
- In nominal/curve/stop phases where the pedestrian tuple is neutral, its
  attribution share is zero under this reference protocol.

These are attribution statements, not claims of real-world causality. Overall
signed totals can cancel across left/right steering and braking/acceleration;
the preregistered primary statistic is therefore absolute group share. R003 and
future R006/R007 remain necessary for intervention and behavioral evidence.

## Integrity

- Locked seeds were rendered once in one process.
- Trace: `(17600, 29)` public states.
- References: `(6, 4, 4400, 29)`.
- Draw attribution: `(6, 2, 4400, 29)`.
- Final attribution: `(2, 4400, 29)`.
- Every reference is same-phase, non-self, cross-seed, and uses four distinct
  reference seeds per draw.
- Final tensor equals the exact float32 mean of all six draw tensors.
- No privileged/GT/world-pose/bbox/IoU schema is stored.
- PPO, MobileNet, YOLO, belief code, checkpoint, and thresholds were unchanged.
- Active suite: `660 passed, 0 failed, 0 skipped` with 426 warnings.
- R006/R007 artifacts do not exist.

## Read-only verification

From repository root:

```bash
PYTHONHASHSEED=0 \
PYTHONPATH=src:/home/pannntastic/aivnv/duckie/src \
/home/pannntastic/aivnv/duckie/.venv/bin/python \
experiments/verify_f11_r004.py \
--config configs/f11_ppo_explanation_r004_v1.toml
```

Expected result: exit zero, `classification=PASS`, 17,600 trace rows, 4,400
attribution samples, 24 effective references, no R006/R007, and 660 active
tests passing.

## Gate status

`R004 PASS`. R006 is eligible for a separate preregistered run, but was not
started automatically.

