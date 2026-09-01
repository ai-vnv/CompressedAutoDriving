# F14 A0 Reference Calibration

Classification: **PASS**

This development-only stage used 500 public 29D states (100 per frozen phase),
six independent draws, and four complete same-phase cross-seed reference rows
per draw. No A1–A7 Shapley result was inspected before this result and threshold
freeze.

## A0 self-consistency

- group Spearman: median `1.000000`, P05 `0.942857`;
- group-share L1: median `0.031378`, P95 `0.080886`;
- top-group agreement median: `1.000000`;
- top-two Jaccard P05: `0.633333`;
- signed agreement median: `0.800000`;
- maximum exact-Shapley local-accuracy residual: `2.384e-07`.

## Frozen preservation thresholds

- minimum group Spearman: `0.942857`;
- maximum group-share L1: `0.130886`;
- minimum top-group agreement: `1.000000`;
- minimum top-two Jaccard: `0.633333`;
- minimum signed agreement: `0.600000`;
- structurally preserved phase/action cells: at least `8` of `10`.

These values are frozen before A1–A7 evaluation and will not be weakened after
compression results are observed.
