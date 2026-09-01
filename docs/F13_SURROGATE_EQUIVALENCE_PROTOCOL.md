# F13 A7-QAT Surrogate Equivalence Protocol

## Purpose

The deployable A7 actor is an x86 static-INT8 TorchScript model. Standard
gradient Integrated Gradients is not run directly on its quantized kernels.
F13 may attribute A7 only if the exact fake-quantized QAT state immediately
preceding the frozen INT8 conversion is present and strongly provenance-bound.

## Fail-closed provenance gate

The candidate must be an immutable artifact from the same F12 A7 run and prove:

1. the 29→64→64→2 architecture and unchanged public 29D ordering;
2. the same F12 config and QAT seed;
3. direct use as the source of the frozen A7 conversion;
4. a stored state hash, creation record, and conversion-source link;
5. no reconstruction, dequantization, refit, or post-hoc surrogate creation.

Loss history alone is not a model state. A dequantized INT8 actor is not the
required pre-conversion state. If no exact artifact passes, A7 gradient
attribution is `BLOCKED`; counterfactual and C4 diagnostics continue.

## Development-only numerical gate

If and only if provenance passes, compare the exact QAT state against deployed
A7 on the frozen F12 development public-state dataset, before reading any A7
attribution. Physical-action requirements are frozen as:

| Metric | v_cmd | omega_cmd |
|---|---:|---:|
| MAE maximum | 0.002 m/s | 0.030 rad/s |
| P99 absolute error maximum | 0.010 m/s | 0.100 rad/s |
| Maximum absolute error | 0.050 m/s | 0.500 rad/s |
| Pearson minimum | 0.999 | 0.999 |
| Spearman minimum | 0.995 | 0.995 |

Additional gates are at least 0.995 agreement for action-bound saturation and
omega sign above the frozen 0.2 rad/s deadband. Every supported public phase
must contain at least 100 rows. All requirements must pass.

## Frozen interpretation

- Provenance absent or ambiguous: attribution `BLOCKED`.
- Provenance valid but numerical equivalence fails: attribution `BLOCKED`.
- Both pass: call the model only `A7-QAT attribution surrogate`, never the
  deployed actor; direct INT8 remains primary for counterfactual analysis.

No threshold may be changed after inspecting F11 R004 attribution comparisons.

