# F13 Compression-Induced Failure-Mode Report

## Frozen probe

The diagnostic used four fixed exploratory C4 seeds (`179001-179004`). Original
and A7 ran the same stage, seed, scenario generator, initial conditions,
pedestrian trajectory, and stop configuration through the unchanged visual
pipeline. No parameter search was performed. Evaluation truth was read only
after policy action.

An initial launcher attempt accidentally left two concurrent WSL child
processes and produced duplicate rows. That whole attempt is retained under
`_failed_concurrent_launch_attempt_*` and excluded. The accepted run contains
exactly eight unique `(policy, seed)` rows.

## Paired C4 results

| Metric | Original | A7 |
|---|---:|---:|
| completion | 4/4 | 4/4 |
| collision episodes | 0/4 | 0/4 |
| unsafe episodes | 0/4 | 0/4 |
| minimum pedestrian clearance | 0.4961 m | 0.4904 m |
| stop completion | 4/4 | 4/4 |
| stop violation | 0/4 | 0/4 |
| restart | 4/4 | 4/4 |
| lane failure | 0/4 | 0/4 |
| invalid pose | 0/4 | 0/4 |

Pedestrian-phase action saturation increased by `0.0307`, below the frozen
`0.05` trigger. There was no Original-pass/A7-fail episode. The reserved
confirmatory seeds (`179101-179104`) therefore remained unopened.

## Failure hierarchy

- **Level 1 attribution drift:** unresolved because A7 IG is blocked.
- **Level 2 functional sensitivity drift:** supported for stop-release
  magnitude; auxiliary pedestrian-yaw sensitivity is also reduced.
- **Level 3 action drift:** not triggered in the accepted closed-loop probe.
- **Level 4 closed-loop failure:** not observed.

The result is a latent functional-sensitivity difference without an observed
compression-induced C4 safety/control failure in this limited fixed probe.

## Behavioral classification

**PRESERVED for the tested C4 conditions.** This does not claim universal
robustness outside the frozen grid.
