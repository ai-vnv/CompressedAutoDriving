# F18 — FP16 Control Protocol

**Frozen before any F18 result is produced or seen.**

## Question

Does reducing floating-point precision from FP32 to FP16 preserve C0–C4 competence on the
fixed recovered checkpoint, before any further INT8 effort is spent?

## Scope — deliberately minimal

One new candidate only: **A3 → FP16**. No KD retraining, no width change, no width sweep,
no pruning-schedule or training-seed study. Comparators are already-measured F17 results
on the identical block: A3 (FP32) and A6 (INT8 PTQ). Neither is re-run.

- Parent checkpoint: the F17 anchor
  `artifacts/f15_cross_curriculum_recovery_v1/recovery/fp32/w64/actor_multicurriculum_kd_fp32.pt`
  (SHA256 `64c84cd0…`), weights cast to float16, nothing else changed.
- Evaluation: seeds 180201–180208, `cuda_strict_deterministic` environment backend,
  frozen F17 gates, A0 reference reused from F17 (identical block and backend; its
  episode CSV hash is recorded).
- Primary RGB evidence via the validated ring-buffer recorder for every objective failure.
- Holdout 180301–180308 stays sealed.

## FP16 validity gate — frozen before the probe runs

The candidate may be *called* FP16 only if the execution is genuinely reduced-precision:

- **V1** — the half-precision actor executes forward on the target device without dtype
  errors, with float16 inputs.
- **V2** — the output tensor dtype is float16.
- **V3** — outputs differ from the FP32 parent in a manner consistent with fp16 rounding
  (weights and per-layer activations genuinely stored/consumed at 16 bits). An
  accumulation-dtype probe (a sum construction that fp16 accumulation loses and fp32
  accumulation keeps) characterizes whether accumulation is fp16 or wider. **Wider
  accumulation with fp16 weights/activations is standard fp16 inference semantics and is
  acceptable, but must be labelled "FP16 storage/arithmetic with FP32 accumulation".**
- **STOP condition** — if the backend cannot execute half natively and merely casts the
  whole computation back to FP32 with no genuine 16-bit representation anywhere (outputs
  bit-identical to FP32), F18 stops and reports that finding. It is not called an FP16
  deployment.
- **Latency honesty** — if FP16 is not faster than FP32 on this CPU backend, the report
  says so; the candidate is then a numerical-precision/storage result, not a
  latency-deployment result.

## Determinism addendum

Before the evaluation block, the FP16 configuration must reproduce bit-exactly across 3
repeats on 2 preregistered cells (c0 and c3, first seed), under the same criteria as the
frozen F16 gate and the F17 INT8 addendum. FAIL bars the evaluation.

## Measurements

1. closed-loop C0–C4 under frozen gates (+ failure phenotype);
2. same-state fidelity vs Original on identical 29D rows;
3. serialized checkpoint bytes and logical parameter memory (params × 2 vs × 4);
4. actor-only CPU latency median/P95/P99 and throughput, frozen `[benchmark]`
   parameters (1 thread, batch 1, 1000 warmup, 10000 iters, 5 repeats), FP32 vs FP16.

## Decision rule — frozen

- FP16 **PASS** C0–C4 → conclude that reduced floating-point precision preserves
  competence on this checkpoint while the tested INT8 procedure does not; report the
  size/latency trade-off; **STOP** (no automatic further sweeps).
- FP16 **FAIL** → document curriculum and failure phenotype; **STOP**.

Either way, whether INT8 remains worth pursuing is reported as a recommendation, not
acted on.

## Claim limits

- No claim beyond this checkpoint, this block, this backend.
- "FP16 preserves competence" is licensed only with the validity gate's actual
  characterization attached (e.g. FP32 accumulation).
- No deployment-readiness claim.
