# Slide Outline — Internship Summary Talk (AI VNV), 15-20 min

Style: unchanged (emerald gradient, Poppins, gold accents; solo author title as
in the tidied deck). Two acts: the journey, then the main project in depth.

## Act I — The journey (6 slides, ~5 min)
| # | Slide | Message | Time |
|---|-------|---------|------|
| 1 | Title | "From Backend Engineer to Closed-Loop V&V Research" + name, AI VnV Research Intern | 0:15 |
| 2 | Starting point | Backend engineer, curious about decision making under uncertainty; month 1 honest: unfamiliar CS research ground, bad performance | 1:00 |
| 3 | The reset | Diagnose the gap, restart from fundamentals: Coursera Decision Making + JuliaAcademy POMDP course | 0:45 |
| 4 | Learning by building | Pluto notebooks hands-on: MDP/POMDP concepts, solver playground, Gallery of POMDPs.jl -> repo Decision-Making-Under-Uncertainty | 1:00 |
| 5 | Project 1: DuckieMDP | MDP formulation in Gym-Duckietown -> preprint "Certified Explanations for MDP Driving Policies" (3 pillars, coupled counterfactuals, 4 policy classes) | 1:15 |
| 6 | Side project: Duckietown.jl | Julia POMDPs.jl reimplementation, validated decision-by-decision vs Python incl. exact RNG streams (bit-for-bit seeded episodes), native renderer + DORASolvers case study | 1:00 |

## Act II — Main project (14 slides, ~11 min)
| # | Slide | Message | Time |
|---|-------|---------|------|
| 7 | Main project | POMDP formulation -> the OJ-ITS paper title; the arc: learn -> formulate -> evaluate | 0:30 |
| 8 | The problem | Aggregate scores hide which behaviors compression removes | 0:45 |
| 9 | Key idea | Acting policies amplify errors -> closed loop; scenario-based assessment reversed (fix scenarios+seeds, vary SUT); 4 questions | 1:00 |
| 10 | The driving task | fig_task: 5 curricula, shared seeds | 0:45 |
| 11 | System under test | fig_pipeline: YOLO+MobileNetV3 -> EKF -> 29-d belief -> actor 73,986 params | 1:00 |
| 12 | How the pipeline works | A0-A9 stages: prune -> distill (2 coverages) -> PTQ/QAT INT8 + FP16; every stage re-evaluated | 1:00 |
| 13 | Protocol | 400 matched episodes, preregistered acceptance, bit-exact backend, 2 axes | 0:45 |
| 14 | Finding 1+2 | Pruning breaks everything (-91.6%); rehearsal coverage decides recovery (C4-only vs balanced) | 1:00 |
| 15 | Finding 3 | fig_matrix: INT8 undoes completed recovery, 8/8 -> 3/8 | 1:00 |
| 16 | Phenotypes | fig_stop: A6 freezes 0.22 m before line; A8 drives through | 1:00 |
| 17 | Finding 4+5 | Controls isolate width x INT8; fidelity does not predict driving (both directions) | 1:00 |
| 18 | Cost | 24.8/12.4/7.6 KB; FP16 +26% latency; actor <0.1% step cost | 0:30 |
| 19 | Takeaways | Judge compressed policies by how they drive, stage by stage | 0:45 |
| 20 | Future works | Wider quantization settings + more training realizations; perception-stack compression under same protocol; transfer to other stacks; Julia-native V&V line via Duckietown.jl | 1:00 |

## Act III — Closing (2 slides, ~1.5 min)
| # | Slide | Message | Time |
|---|-------|---------|------|
| 21 | What the internship built | backend -> RL/POMDP research: 3 repos, 2 papers, 1 Julia package; the lesson: fundamentals first, then honest evaluation | 0:45 |
| 22 | Thank you | Repos: CompressedAutoDriving, DuckieMDP, Decision-Making-Under-Uncertainty, Duckietown.jl | 0:15 |

Total ~17:30 (fits 15-20 min)
