# F16 Quantization-Scheme Scope Limitation

This note was written after completion of the preregistered determinism preflight and
the primary-media integrity gate, and **before training or evaluation of any F16
Direct/Progressive scientific candidate**.

It does not alter the frozen F16 candidate matrix, optimization sequences, widths, gates,
seeds, training data, quantization backend, or candidate selection rules. `docs/F16_PROTOCOL.md`
(SHA256 `629cd8285a8665285ce9dda3473cd61c26b555351bb5bb0820291b0122715b81`) and
`configs/f16_sequence_int8_recovery_v1.toml` (SHA256
`b08282ec0545180d435459924c353515f839f541dd7da329f2b273d12b2d0fba`) remain unmodified.

## Scope

F16 varies only:

1. **optimization sequence** — Direct, Progressive;
2. **target width** — 64, 96, 128, 192;
3. **quantization/recovery route** — FP32, PTQ INT8, and QAT+KD INT8 when the
   preregistered condition triggers it.

The quantization *representation* itself remains fixed to the frozen x86 static INT8
procedure: qint8 per-channel symmetric weights, quint8 per-tensor affine activations,
static observers, float Tanh boundaries.

F16 can therefore determine whether cross-curriculum INT8 retention changes with
optimization sequence, width, PTQ/QAT recovery, or their tested interactions.

F16 **cannot** determine whether a failure is specifically caused by:

- activation quantization granularity;
- per-tensor versus per-channel activation scaling;
- observer choice;
- calibration algorithm;
- quantization scale resolution;
- mixed-precision allocation;
- alternative INT8 backend behaviour;
- weight-only versus weight-and-activation quantization;
- or any other quantization representation not varied in F16.

## What the F15 evidence does and does not motivate

F15 recorded that both PTQ and QAT+KD kept same-state action MAE comparatively small
while Pearson and Spearman correlation degraded past their gates — the failing fidelity
checks were rank-order checks, not magnitude checks. This indicates that the **relative
structure of the action mapping changed**, not merely that magnitude error grew.

**The F15 correlation/rank-order degradation motivates quantization representation as a
future hypothesis, but does not identify activation granularity as its mechanism.**

Many factors can jointly produce that pattern: quantization, saturation, activation
scale, network sensitivity, width, and closed-loop error accumulation. No single one of
them is isolated by the F15 data.

## Interpretation rule

If an FP32 candidate passes C0–C4 but the matched INT8 candidate fails, the strongest
permitted conclusion is:

> "INT8 conversion under the frozen F16 quantization procedure is associated with a new
> retention failure at this width and sequence."

If Direct and Progressive differ at a matched endpoint, F16 may support an
optimization-sequence effect **under the tested protocol**.

If increasing width restores INT8 retention, F16 may support width-dependent quantization
robustness **under the tested procedure**.

If all tested widths and sequences fail after INT8 conversion, F16 must report:

> "No tested combination of width and optimization sequence preserved full C0–C4
> retention under the frozen INT8 procedure."

It must **not** conclude that:

- INT8 precision is intrinsically insufficient;
- per-tensor activation quantization caused the failure;
- steering resolution is definitively too coarse;
- or that another untested quantization scheme would solve the problem.

Those mechanisms require a separate controlled experiment.

## Follow-up trigger

A subsequent experiment may investigate quantization-scheme granularity **only if** F16
leaves the INT8 failure unresolved. Such a study must be preregistered separately and must
not retroactively change F16.

A negative F16 result is therefore not a failure of F16. "Neither tested width nor
optimization sequence recovered INT8 retention" is a clean, well-controlled finding, and
it is precisely what would justify varying the quantization representation next with a
single manipulated variable.
