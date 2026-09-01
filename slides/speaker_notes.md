# Speaker Notes (quick cues) — Internship Summary

Full script: `TALK_SCRIPT.md`. Cues also embedded in `presentation.pptx`.

## Act I — The journey
1. **Title** — one sentence: the whole internship, backend -> submitted V&V paper.
2. **Starting point** — honest: month 1 bad performance; gap = fundamentals. [1:00]
3. **Reset** — Coursera + JuliaAcademy; rule: rebuild every concept in code. [0:45]
4. **Learning by building** — Pluto notebooks, solver playground, Gallery of
   POMDPs.jl; public repo. [1:00]
5. **DuckieMDP** — certified explanations, 3 pillars; counterfactuals share
   exogenous noise -> exact; 4 policy classes; arXiv preprint. [1:15]
6. **Duckietown.jl** — POMDPs.jl port; validated decision-by-decision incl.
   exact RNG streams -> bit-for-bit; native renderer + DORA lap. [1:00]

## Act II — Main project (OJ-ITS)
7. **Divider** — POMDP goes from learning object to instrument. [0:30]
8. **Problem** — aggregate scores hide which behaviors die. [0:45]
9. **Key idea** — feedback loop compounds drift; assessment reversed:
   scenarios+seeds fixed, SUT varies; 4 questions. [1:00]
10. **Task** — 5 curricula, 8 shared seeds; stop tasks are the hard ones. [0:45]
11. **SUT** — YOLO + MobileNetV3 -> EKF -> 29-d belief -> 73,986-param actor. [1:00]
12. **Pipeline** — A1 -> A2/A3 -> A6/A8 -> A9; rest controls; every stage
    re-evaluated. [1:00]
13. **Protocol** — 400 matched episodes; preregistered; bit-exact; 2 axes. [0:45]
14. **F1+F2** — pruning (-91.6%) fails all; rehearsal data decides recovery. [1:00]
15. **F3** — matrix; INT8 re-breaks C3-C4; 8/8 -> 3/8. [1:00]
16. **Phenotypes** — A6 parks 0.22 m forever; A8 drives through. [1:00]
17. **F4+F5** — width x INT8 interaction; fidelity misleads both directions. [1:00]
18. **Cost** — FP16 memory move (+26% latency); actor <0.1% step cost. [0:45]
19. **Takeaways** — judge by how it drives, stage by stage. [0:45]
20. **Future works** — wider settings; compress perception; transfer protocol;
    Julia-native V&V line. [0:45]

## Act III — Closing
21. **Reflection** — 4 repos, 2 papers, 1 Julia package; fundamentals first. [0:30]
22. **Thanks** — 4 repos on screen. [0:15]
