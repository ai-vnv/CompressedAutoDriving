# F17 Comparison-Interpretation Amendment

**Frozen before any F17 comparison was interpreted.** At the time of writing only the A0
reference was being evaluated; A0 is the gate reference and licenses no comparison claim.

This amendment constrains **interpretation wording only**. It changes no gate, threshold,
seed, checkpoint, pathway definition, or eligibility rule.
`docs/F17_PROTOCOL.md` (SHA256 `0d8aa9e991db1deb2a22f3fdf91902862131d5dc61307068a14d375cd3dbde12`)
and `configs/f17_optimization_method_order_v1.toml` (SHA256
`8fe36a61f60d6794aa94013dbeb731495bc7bc7aada089ce26d9262919bfdbce`) remain unmodified.

## Clarification 1 — A5 versus A6 is a placement comparison, not a factorial order proof

| Pathway | Construction |
|---|---|
| A5 | prune → PTQ |
| A6 | prune → balanced C0–C4 KD → PTQ |

What differs between them is **the presence of balanced distillation before
quantization**, not merely the ordering of a fixed set of operations. A6 contains a stage
A5 does not contain.

The licensed question is therefore:

> **Does inserting balanced distillation before PTQ preserve more cross-curriculum
> competence than quantizing the pruned actor directly?**

If A6 outperforms A5, the supported conclusion is:

> Balanced distillation before PTQ was beneficial under the tested pathway.

**Not** supported: that all permutations of pruning, distillation, and quantization have a
determined ranking, or that "optimization order matters" as a general law. F17 evaluates a
small set of practically valid pathways, not a complete factorial design over operation
orderings.

## Clarification 2 — A6 versus A8 compares alternative quantization routes, not a repair

| Pathway | Construction | Parent |
|---|---|---|
| A6 | prune → balanced KD → **PTQ** | A3 anchor |
| A8 | prune → balanced KD → **balanced QAT+KD** → INT8 | A3 anchor |

Both branch from the same recovered FP32 anchor A3. **A8 does not continue training from
the already-converted A6 INT8 graph.** It is a different quantization route taken from the
same FP32 parent.

The licensed question is therefore:

> **Can a QAT+KD quantization route preserve or restore cross-curriculum retention relative
> to the PTQ route?**

Permitted wording if A8 succeeds where A6 fails:

> Under the tested procedures, the QAT+KD quantization route preserved retention that the
> PTQ route did not, taken from the same recovered FP32 parent.

**Not** permitted:

- "QAT repaired the failed PTQ model." A8 never trains the A6 INT8 graph.
- "QAT fixes quantization damage in general." Only two fixed procedures from one fixed
  parent were compared.

## What remains unchanged and is the strength of the design

A3 → A6 stays the clean quantization test, because the parent is a single fixed checkpoint
that has already been verified to PASS all five curricula on the identical deterministic
block (40 episodes, seeds 180201–180208). If that fixed checkpoint shows PASS→FAIL after
conversion on the same block, the supported wording is:

> INT8 conversion under the frozen PTQ procedure was associated with a new retention
> failure for this fixed recovered FP32 checkpoint.

No training-realization variability enters that statement, because no training occurs
anywhere in the F17 primary comparison — every pathway member is a pre-existing frozen
checkpoint.

A0 → A4 remains the quantization-only control. If quantizing the unpruned Original already
degrades a curriculum, quantization alone can produce retention failure without pruning. If
A4 is largely intact while A6 is not, the interaction with the compressed and recovered
actor is the more relevant factor rather than INT8 precision as such.

## Reporting requirement

Every F17 comparison in the reports and figures must state its licensed question in the
wording fixed above. The generic phrase "does optimization order matter" is not used as a
caption for A5 versus A6.
