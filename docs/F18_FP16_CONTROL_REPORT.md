# F18 FP16 Control — Final Report

**Classification: `FP16_PRESERVES_COMPETENCE`. Stage 1 complete, stopped as the frozen
decision rule requires. No further sweep started. Holdout 180301–180308 still sealed.**

Scope was one new candidate. The A0 reference and the A3/A6 comparator rows were reused
from F17 on the identical block and backend; both were verified to reproduce F17's verdicts
and completion rates exactly. Total new compute: 40 episodes plus two preregistered probes.

## Preconditions, frozen before any curriculum result

**FP16 validity gate — PASS.** Half forward executes natively; output dtype is `float16`;
every parameter is 2 bytes; outputs differ from FP32 (max |Δ| = 0.00199 over 512 probe rows)
so the silent-upcast STOP condition did not trigger. A discriminating probe (sum of N ones
with N exactly representable in fp16, where sequential fp16 accumulation would saturate at
2048) returned N exactly, so accumulation is wider than fp16. The licensed label for every
FP16 claim below is therefore:

> **FP16 weights and activations with FP32-wide accumulation (standard fp16 inference
> semantics), FP32 I/O boundary.**

**FP16 determinism addendum — PASS.** Bit-identical actions, progress, episode length,
completion, failure labels and termination reason across 3 repeats on c0 and c3, under the
same criteria as the frozen F16 gate and the F17 INT8 addendum.

## Closed-loop retention

| Candidate | Prec | C0 | C1 | C2 | C3 | C4 |
|---|---|---|---|---|---|---|
| A3 anchor | FP32 | PASS | PASS | PASS | PASS | PASS |
| **F16H** | **FP16** | **PASS** | **PASS** | **PASS** | **PASS** | **PASS** |
| A6 (F17) | INT8 | PASS | PASS | PASS | FAIL | FAIL |

FP16 is behaviourally near-indistinguishable from its FP32 parent: mean progress differs by
0.001–0.008 m, completion rates are identical on all five curricula, stop-violation is 0.000
everywhere, and the single non-completing C1 episode is the *same* episode (seed 180206) that
the FP32 anchor also does not complete — an inherited property of the anchor, not a new
FP16 failure. That cell still passes. No new objective failure appeared on any curriculum.

## Same-state fidelity vs Original

| | C0 | C1 | C2 | C3 | C4 | gate result |
|---|---|---|---|---|---|---|
| A3 FP32 ω MAE / Spearman | .032/.9997 | .038/.9996 | .034/.9961 | .020/.9976 | .022/.9860 | PASS ×5 |
| **F16H FP16** | .032/.9997 | .038/.9996 | .034/.9961 | .020/.9976 | .023/.9859 | **PASS ×5** |
| A6 INT8 | .163/.9632 | .152/.9826 | .092/.9778 | .086/.9609 | .087/.9450 | FAIL 4/5 |

FP16 tracks the FP32 anchor to three or four decimals on every metric. **F16H is the first
candidate in this project to pass the frozen behaviour gates and the frozen fidelity gates
on all five curricula simultaneously.**

## Cost

| | Serialized file | Logical parameter memory | Latency median | P95 | P99 | vs FP32 |
|---|---|---|---|---|---|---|
| A3 FP32 | 29 295 B | 24 840 B (6 210 × 4) | 19.37 µs | 29.64 | 45.32 | 1.00× |
| **F16H FP16** | **15 865 B** | **12 420 B (6 210 × 2)** | 24.49 µs | 39.37 | 63.24 | **0.79× (21 % slower)** |
| A6 INT8 | 34 088 B | not comparable | 12.89 µs | 17.06 | 32.41 | 1.50× |

Two results are reported exactly as measured, not as hoped:

- **FP16 is slower, not faster,** on this x86 CPU backend — there is no native half compute
  path being used, so conversion overhead is paid without compensation. Under the protocol's
  latency-honesty clause, F16H is a **numerical-precision and memory result, not a
  latency-deployment result**.
- **The INT8 file is larger than the FP32 file** (34 088 > 29 295 B) because it is a traced
  TorchScript graph whose metadata dominates weight savings on a 6 210-parameter actor. Only
  logical parameter memory is comparable across precisions, and INT8's is not extractable
  from the traced graph, so it is left blank rather than estimated.

## Interpretation, conservatively

Licensed by this evidence:

> **Reducing floating-point precision to FP16 (with FP32-wide accumulation) preserved
> cross-curriculum competence on this fixed recovered checkpoint, while the tested INT8
> procedure did not.** The C3/C4 retention failure is therefore not a generic consequence of
> reducing numeric precision; it is specific to the tested integer-quantization procedure at
> this width.

This composes with the F17 control (A4: PTQ of the unpruned Original passes all five) into a
consistent picture: the failure needs *both* narrow width *and* integer quantization. Neither
precision reduction alone (F16H) nor quantization alone (A4) reproduces it.

Not licensed: any claim beyond this checkpoint, this evaluation block, this backend; any
deployment-readiness claim; any claim that FP16 would hold on another training realization
(F16 showed the recovery itself is training-realization sensitive).

## Eligibility and the holdout

The frozen selection rule requires INT8, so an FP16 candidate **cannot** be eligible by
construction. Eligibility is reported unchanged rather than redefined after seeing results:
`eligible = false` for every member, and **180301–180308 remain sealed**.

Whether the deployment-precision requirement should be amended from "INT8" to "INT8 or FP16"
is a decision for the investigator, and under the project's freeze discipline it must be
taken **prospectively** — an amendment hash-bound before the holdout is opened — never as a
post-hoc relaxation that lets an already-seen result through.

## Recommendation on pursuing INT8

Deployment case: **weak on this evidence.** The actor is 6 210 parameters. FP16 already
halves parameter memory (12.4 KB) with zero retention loss, and INT8's only measured
advantage is actor latency (12.9 µs vs 19.4 µs). Against a measured simulator step cost of
~37 800 µs, the actor accounts for well under 0.1 % of step time — the perception stage
dominates, so shaving 6.5 µs off the actor cannot matter for end-to-end throughput. INT8 is
worth pursuing for deployment only if a target platform *requires* integer-only inference
(no fp16 path), which is a hardware-requirements question this experiment cannot answer.

Scientific case: **still open and now cleanly motivated.** F18 sharpens F17's finding rather
than closing it — the width × integer-quantization interaction remains unexplained, and the
untested variable is the quantization representation itself (activation granularity,
calibration), scoped in `docs/F16_QUANTIZATION_SCOPE_LIMITATION.md`. Stage 2 (2–3 PTQ
ablations, ~2 h) would answer *why*, not merely *whether*.

These are separate motives and should be decided separately. Per the frozen decision rule,
nothing further was started.

## Artefacts

- `configs/f18_fp16_control_v1.toml` (gates/seeds/fidelity copied programmatically verbatim
  from the frozen F17 config; equality asserted at freeze time)
- `artifacts/f18_fp16_control_v1/integrity/` — `fp16_validity_gate.json`,
  `fp16_determinism_addendum.json`, `protocol_manifest.json`
- `artifacts/f18_fp16_control_v1/results/` — `pathway_results.csv`,
  `same_state_fidelity.csv`, `pathway_summary.json`, `precision_benchmark.json`,
  `f18_outcome.json`
- `artifacts/f18_fp16_control_v1/figures/12_precision_control_fp32_fp16_int8.png/.pdf`
- `artifacts/f18_fp16_control_v1/candidates/actor_fp16.pt` (parent SHA `64c84cd0…`, not
  retrained, width unchanged)
