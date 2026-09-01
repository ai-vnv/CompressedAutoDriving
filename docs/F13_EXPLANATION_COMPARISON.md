# F13 Explanation Comparison

## Frozen actors and boundary

The comparison keeps the Original Belief-PPO
(`02e898ce...f2c6250`, 29-256-256-2 FP32) and selected A7
(`f8e4e3ae...b7cbc7e`, 29-64-64-2 static INT8) immutable. Both consume the
same normalized public 29D representation and retain the same physical action
mapping.

## Gradient-attribution status

**A7 Distributional IG is BLOCKED.** F12 converted the QAT model directly to
the deployable INT8 graph and persisted the INT8 actor plus loss history, but
not the exact pre-conversion fake-quantized model state. The history is not a
model checkpoint. No approximate or dequantized surrogate was created.

Consequently:

- Original R004 attribution remains valid and unchanged;
- A7 attribution shares, ranking, Spearman, L1 drift, and top-group
  preservation are **UNRESOLVED**;
- unavailable attribution is never encoded as zero;
- F13-B figures explicitly display the blocked branch.

This is an evidence limitation, not evidence that A7 lacks semantic structure.

## Direct deployed-INT8 counterfactual comparison

The six frozen R003 operators were evaluated on the 4,400 sampled public R004
states. Original replay maximum error was `1.43e-6`, below the new frozen F13
limit `5e-6`; repeated A7 inference differed by exactly zero. Sham effects were
exactly zero for both actors.

Primary comparisons:

| Intervention / public phase | N | Original mean effect | A7 mean effect | Direction agreement | Normalized mean drift | Result |
|---|---:|---:|---:|---:|---:|---|
| pedestrian absent / pedestrian relevant, delta-v | 956 | +0.30367 m/s | +0.27486 m/s | 1.000 | 0.0720 | preserved |
| stop absent / stop required, delta-v | 909 | +0.21883 m/s | +0.17703 m/s | 1.000 | 0.1045 | shifted (limit 0.10) |
| lane centered / lane curve, delta-omega | 1,171 | mean abs. 0.3774 rad/s | mean abs. 0.2183 rad/s | descriptive sign agreement 0.847 | mean-absolute drift 0.0199 of range | preserved by frozen magnitude rule |

An auxiliary result also shows reduced pedestrian-intervention yaw magnitude:
mean delta-omega changed from `+1.8899` to `+0.9163 rad/s` (normalized mean
drift `0.1217`). This is reported as functional drift, not as a preregistered
directional failure.

## Classification

- Semantic explanation structure: **UNRESOLVED**
- Counterfactual functional sensitivity: **PARTIALLY PRESERVED**

R006 remains historically FAILED and was not repaired. This is a new F13,
device-specific replay protocol.
