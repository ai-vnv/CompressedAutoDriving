# Research Output Manifest

> Auto-maintained by ARIS skills. Tracks all generated artifacts across the research lifecycle.

| Timestamp | Skill | File | Stage | Description |
|-----------|-------|------|-------|-------------|
| 2026-08-14 10:46 | /experiment-plan | refine-logs/EXPERIMENT_PLAN_20260814_104619.md | implementation | Frozen F11 explanation protocol |
| 2026-08-14 10:46 | /experiment-plan | refine-logs/EXPERIMENT_PLAN.md | implementation | Latest F11 explanation protocol |
| 2026-08-14 10:46 | /experiment-plan | refine-logs/EXPERIMENT_TRACKER_20260814_104619.md | implementation | Initial F11 execution tracker |
| 2026-08-14 11:00 | /experiment-plan | refine-logs/EXPERIMENT_TRACKER_20260814_110053.md | implementation | Completed F11 tracker |
| 2026-08-14 11:00 | /experiment-plan | refine-logs/EXPERIMENT_TRACKER.md | implementation | Latest completed F11 tracker |
| 2026-08-14 10:46 | /experiment-plan | docs/F11_PPO_EXPLANATION_FORMULATION.md | implementation | Two registered explanation ideas and frozen Method 1 protocol |
| 2026-08-14 11:00 | /experiment-plan | docs/F11_IG_REPORT_FOR_REVIEW.md | implementation | Integrated Gradients review report |
| 2026-08-14 10:46 | /experiment-plan | configs/f11_ppo_integrated_gradients_v1.toml | implementation | Frozen explanation configuration |
| 2026-08-14 10:50 | /run-experiment | artifacts/f11_ppo_integrated_gradients_v1/policy_trajectory.npz | implementation | Public-only C4 explanation trajectories |
| 2026-08-14 10:50 | /run-experiment | artifacts/f11_ppo_integrated_gradients_v1/trajectory_manifest.json | implementation | Trajectory provenance |
| 2026-08-14 10:50 | /run-experiment | artifacts/f11_ppo_integrated_gradients_v1/collect.log | implementation | Real-simulator collection log |
| 2026-08-14 11:00 | /academic-plotting | artifacts/f11_ppo_integrated_gradients_v1/integrated_gradients.npz | implementation | Per-frame IG arrays |
| 2026-08-14 11:00 | /academic-plotting | artifacts/f11_ppo_integrated_gradients_v1/integrated_gradients_metrics.json | implementation | Completeness, stability, faithfulness, and importance metrics |
| 2026-08-14 11:00 | /academic-plotting | artifacts/f11_ppo_integrated_gradients_v1/feature_attribution_summary.csv | implementation | Per-feature attribution summary |
| 2026-08-14 11:00 | /academic-plotting | artifacts/f11_ppo_integrated_gradients_v1/group_attribution_summary.csv | implementation | Semantic-group attribution summary |
| 2026-08-14 11:00 | /academic-plotting | artifacts/f11_ppo_integrated_gradients_v1/faithfulness.csv | implementation | Top/random/bottom deletion curves |
| 2026-08-14 11:00 | /academic-plotting | artifacts/f11_ppo_integrated_gradients_v1/ig_qualitative_timeline.png | implementation | Raster qualitative attribution timeline |
| 2026-08-14 11:00 | /academic-plotting | artifacts/f11_ppo_integrated_gradients_v1/ig_qualitative_timeline.pdf | implementation | Vector qualitative attribution timeline |
| 2026-08-14 11:00 | /academic-plotting | artifacts/f11_ppo_integrated_gradients_v1/ig_group_importance.png | implementation | Raster global group importance plot |
| 2026-08-14 11:00 | /academic-plotting | artifacts/f11_ppo_integrated_gradients_v1/ig_group_importance.pdf | implementation | Vector global group importance plot |
| 2026-08-14 11:00 | /academic-plotting | artifacts/f11_ppo_integrated_gradients_v1/ig_faithfulness.png | implementation | Raster deletion-faithfulness plot |
| 2026-08-14 11:00 | /academic-plotting | artifacts/f11_ppo_integrated_gradients_v1/ig_faithfulness.pdf | implementation | Vector deletion-faithfulness plot |
| 2026-08-14 11:00 | /run-experiment | artifacts/f11_ppo_integrated_gradients_v1/analyze.log | implementation | Integrated Gradients execution log |
| 2026-08-14 11:00 | /run-experiment | artifacts/f11_ppo_integrated_gradients_v1/full_tests.log | implementation | Full 628-test witness |
| 2026-08-14 11:00 | /run-experiment | artifacts/f11_ppo_integrated_gradients_v1/artifact_manifest.json | implementation | SHA256 manifest and test counts |
| 2026-08-14 11:45 | /academic-plotting | configs/f11_ppo_ig_bev_v1.toml | implementation | Frozen evaluation-only IG-to-BEV configuration |
| 2026-08-14 11:47 | /run-experiment | artifacts/f11_ppo_integrated_gradients_v1/bev/evaluation_only_pose_trace.npz | implementation | Exact seed-176001 pose replay kept outside policy/IG inputs |
| 2026-08-14 11:47 | /academic-plotting | artifacts/f11_ppo_integrated_gradients_v1/bev/ig_bev_action_map.png | implementation | Raster C4 BEV actor-attribution map |
| 2026-08-14 11:47 | /academic-plotting | artifacts/f11_ppo_integrated_gradients_v1/bev/ig_bev_action_map.pdf | implementation | Vector C4 BEV actor-attribution map |
| 2026-08-14 11:47 | /academic-plotting | artifacts/f11_ppo_integrated_gradients_v1/bev/ig_bev_samples.csv | implementation | Spatially aligned actor-IG samples |
| 2026-08-14 11:47 | /run-experiment | artifacts/f11_ppo_integrated_gradients_v1/bev/ig_bev_manifest.json | implementation | BEV hashes, replay equality, and leakage boundary |
| 2026-08-14 14:51 | /experiment-plan | refine-logs/EXPERIMENT_PLAN_20260814_145137.md | implementation | Reset F11 plan grounded in the frozen C4 RGB-to-belief-to-29D runtime |
| 2026-08-14 14:51 | /experiment-plan | refine-logs/EXPERIMENT_PLAN.md | implementation | Latest reset F11 explanation plan |
| 2026-08-14 14:51 | /experiment-plan | refine-logs/EXPERIMENT_TRACKER_20260814_145137.md | implementation | Reset F11 execution tracker |
| 2026-08-14 14:51 | /experiment-plan | refine-logs/EXPERIMENT_TRACKER.md | implementation | Latest reset F11 execution tracker |
| 2026-08-14 15:14 | /run-experiment | configs/f11_ppo_explanation_v2.toml | implementation | Frozen R001 observation-contract configuration |
| 2026-08-14 15:14 | /run-experiment | artifacts/f11_ppo_explanation_v2/r001/contract_audit.json | implementation | R001 PASS contract and replay metrics |
| 2026-08-14 15:14 | /run-experiment | artifacts/f11_ppo_explanation_v2/r001/public_trace.npz | implementation | Public-only 29D deployment trace |
| 2026-08-14 15:14 | /run-experiment | artifacts/f11_ppo_explanation_v2/r001/trace_manifest.json | implementation | R001 artifact hashes and leakage declaration |
| 2026-08-14 15:14 | /run-experiment | artifacts/f11_ppo_explanation_v2/r001/full_tests.log | implementation | Full 640-test regression witness |
| 2026-08-14 15:14 | /run-experiment | docs/F11_R001_OBSERVATION_CONTRACT_REPORT.md | implementation | R001 observation/belief boundary report |
| 2026-08-14 15:14 | /experiment-plan | refine-logs/EXPERIMENT_TRACKER_20260814_151431.md | implementation | Timestamped tracker after R001 PASS |
| 2026-08-14 15:14 | /experiment-plan | refine-logs/EXPERIMENT_TRACKER.md | implementation | Latest F11 v2 tracker after R001 PASS |
| 2026-08-14 15:43 | /run-experiment | configs/f11_ppo_explanation_development_v2.toml | implementation | Frozen development-only R002/R003 protocol |
| 2026-08-14 15:43 | /run-experiment | artifacts/f11_ppo_explanation_v2/r002/development_trace.npz | implementation | 8,800-frame public-only development trace |
| 2026-08-14 15:43 | /run-experiment | artifacts/f11_ppo_explanation_v2/r002/baseline_robustness.json | implementation | R002 LIMITED baseline-agreement result |
| 2026-08-14 15:43 | /run-experiment | artifacts/f11_ppo_explanation_v2/r003/intervention_validation.json | implementation | R003 PASS semantic-operator result |
| 2026-08-14 15:43 | /run-experiment | artifacts/f11_ppo_explanation_v2/r002_r003_full_tests.log | implementation | Full 651-test regression witness |
| 2026-08-14 15:43 | /run-experiment | docs/F11_R002_R003_REPORT_FOR_REVIEW.md | implementation | Development gate report; R004 remains blocked |
| 2026-08-14 15:43 | /experiment-plan | refine-logs/EXPERIMENT_TRACKER_20260814_154307.md | implementation | Timestamped R002/R003 tracker |
| 2026-08-14 15:43 | /experiment-plan | refine-logs/EXPERIMENT_TRACKER.md | implementation | Latest tracker with R002 LIMITED and R003 PASS |
| 2026-08-14 16:10 | /run-experiment | configs/f11_ppo_explanation_r002b_v1.toml | implementation | Preregistered single R002b distributional-reference protocol |
| 2026-08-14 16:10 | /run-experiment | docs/F11_R002B_PROTOCOL.md | implementation | Frozen phase-conditioned multi-reference IG design and unchanged gates |
| 2026-08-14 16:10 | /run-experiment | artifacts/f11_ppo_explanation_v2/r002b/reference_draws.npz | implementation | Six deterministic cross-seed, same-phase public reference draws |
| 2026-08-14 16:10 | /run-experiment | artifacts/f11_ppo_explanation_v2/r002b/distributional_integrated_gradients.npz | implementation | Distributional actor IG for 2,200 development states |
| 2026-08-14 16:10 | /run-experiment | artifacts/f11_ppo_explanation_v2/r002b/distributional_robustness.json | implementation | R002b PASS stability and bootstrap metrics |
| 2026-08-14 16:10 | /run-experiment | artifacts/f11_ppo_explanation_v2/r002b/artifact_manifest.json | implementation | R002b hashes, environment witness, and test provenance |
| 2026-08-14 16:10 | /run-experiment | artifacts/f11_ppo_explanation_v2/r002b_full_tests.log | implementation | Active suite: 655 passed, 0 failed, 0 skipped |
| 2026-08-14 16:10 | /run-experiment | docs/F11_R002B_REPORT_FOR_REVIEW.md | implementation | Development-only R002b PASS report; R004 unlocked but not run |
| 2026-08-14 16:10 | /experiment-plan | refine-logs/EXPERIMENT_TRACKER_20260814_161027.md | implementation | Timestamped tracker after R002b PASS |
| 2026-08-14 16:10 | /experiment-plan | refine-logs/EXPERIMENT_TRACKER.md | implementation | Latest tracker: R002b PASS and R004 ready/not run |
| 2026-08-14 17:09 | /run-experiment | configs/f11_ppo_explanation_r004_v1.toml | implementation | Frozen once-only locked R004 protocol and unchanged gates |
| 2026-08-14 17:09 | /run-experiment | docs/F11_R004_PROTOCOL.md | implementation | Final 24-reference estimator and no-fallback support rule |
| 2026-08-14 17:09 | /run-experiment | artifacts/f11_ppo_explanation_v2/r004/once_only_launch_claim.json | implementation | Claim written before locked-seed access |
| 2026-08-14 17:09 | /run-experiment | artifacts/f11_ppo_explanation_v2/r004/locked_public_trace.npz | implementation | 17,600-frame public-only locked deployment trace |
| 2026-08-14 17:09 | /run-experiment | artifacts/f11_ppo_explanation_v2/r004/reference_draws.npz | implementation | Same-phase four-distinct-cross-seed references |
| 2026-08-14 17:09 | /run-experiment | artifacts/f11_ppo_explanation_v2/r004/distributional_integrated_gradients.npz | implementation | Six draw-level physical actor attributions |
| 2026-08-14 17:09 | /run-experiment | artifacts/f11_ppo_explanation_v2/r004/final_mean_attribution.npz | implementation | Exact equal mean over all 24 effective references |
| 2026-08-14 17:09 | /run-experiment | artifacts/f11_ppo_explanation_v2/r004/final_attribution_metrics.json | implementation | R004 PASS holdout metrics and group attribution |
| 2026-08-14 17:09 | /run-experiment | artifacts/f11_ppo_explanation_v2/r004/artifact_manifest.json | implementation | Frozen R004 source/artifact/test hashes |
| 2026-08-14 17:09 | /run-experiment | artifacts/f11_ppo_explanation_v2/r004_full_tests.log | implementation | Active suite: 660 passed, 0 failed, 0 skipped |
| 2026-08-14 17:09 | /run-experiment | docs/F11_R004_REPORT_FOR_REVIEW.md | implementation | Once-only final actor attribution report |
| 2026-08-14 17:09 | /experiment-plan | refine-logs/EXPERIMENT_TRACKER_20260814_170916.md | implementation | Timestamped tracker after R004 PASS |
| 2026-08-14 17:09 | /experiment-plan | refine-logs/EXPERIMENT_TRACKER.md | implementation | Latest tracker with R004 PASS and R006 ready/not run |
# F11 R006 confirmatory semantic intervention

- `configs/f11_ppo_explanation_r006_v1.toml` — frozen once-only R006 protocol configuration.
- `docs/F11_R006_PROTOCOL.md` — preregistered semantic intervention hypotheses and gates.
- `docs/F11_R006_REPORT_FOR_REVIEW.md` — immutable failed-run report; no scientific intervention result.
- `artifacts/f11_ppo_explanation_v2/r006/` — once-only claim and failure marker.
- `artifacts/f11_ppo_explanation_v2/r006_failure_diagnosis.json` — machine-readable replay near-miss diagnosis.
- `experiments/verify_f11_r006_failure.py` — read-only failed-attempt verifier.

# F13 Explain Again and compression diagnostics

- `configs/f13_explain_compressed_v1.toml` — frozen Original-versus-A7 diagnostic protocol.
- `docs/F13_EXPLAIN_AGAIN_PROTOCOL.md` — preregistered attribution, intervention, and paired-stress design.
- `docs/F13_SURROGATE_EQUIVALENCE_PROTOCOL.md` — exact-QAT provenance and numerical-equivalence gate.
- `docs/F13_EXPLANATION_COMPARISON.md` — blocked A7-IG branch and direct INT8 counterfactual results.
- `docs/F13_FAILURE_MODE_REPORT.md` — paired C4 stress results and failure hierarchy.
- `docs/F13_FINAL_REPORT.md` — final PRESERVED/UNRESOLVED/PARTIALLY-PRESERVED classification.
- `artifacts/f13_explain_compressed_v1/integrity/` — model hashes, surrogate audit, and replay calibration.
- `artifacts/f13_explain_compressed_v1/counterfactual/` — frozen public-state Original/A7 intervention comparison.
- `artifacts/f13_explain_compressed_v1/failure_modes/` — paired exploratory result and unopened-confirmatory marker.
- `artifacts/f13_explain_compressed_v1/figures/` — publication PNG/PDF comparison figures.
- `artifacts/f13_explain_compressed_v1/final/` — classification, registry, figure data, and SHA256 manifest.
- `artifacts/f13_explain_compressed_v1/_failed_concurrent_launch_attempt_20260815_002900/` — excluded duplicate-launch audit trail.
- `experiments/verify_f13_explanation_boundary.py` — frozen actor/surrogate/replay boundary verifier.
- `experiments/run_f13_counterfactual_comparison.py` — direct deployed-INT8 semantic interventions.
- `experiments/run_f13_failure_mode_probe.py` — same-seed C4 paired diagnostics.
- `experiments/generate_f13_figures.py` — deterministic publication figure generator.
- `experiments/verify_f13_artifacts.py` — fail-closed final F13 classifier and manifest builder.

# F14 Explainability-Aware Compression Diagnostics

- `configs/f14_explainability_aware_compression_v1.toml` — frozen A0--A7, public-state, reference, metric, and threshold protocol.
- `docs/F14_PROTOCOL.md` — preregistered exact Group Shapley and semantic-intervention protocol.
- `docs/F14_REFERENCE_CALIBRATION.md` — A0-only reference-robustness result and frozen thresholds.
- `docs/F14_ABLATION_EXPLANATION.md` — same-state A0--A7 semantic/function comparison.
- `docs/F14_FAILURE_MODE_REPORT.md` — descriptive L0--L5 diagnosis, frontier, and unavailable-trace disclosure.
- `docs/F14_FINAL_REEXPLANATION.md` — frozen 4,400-state A0-versus-A7 result.
- `docs/F14_FINAL_REPORT.md` — final multidimensional LIMITED classification.
- `src/duckie_pomdp/explain/group_shapley.py` — exact 64-coalition six-group Shapley engine.
- `src/duckie_pomdp/explain/compression_diagnostics.py` — frozen FP32/INT8 wrappers, public references, metrics, and interventions.
- `experiments/prepare_f14_diagnostic.py` — provenance audit and deterministic 500-state diagnostic-set builder.
- `experiments/calibrate_f14_shapley_references.py` — A0-only robustness calibration and threshold freeze.
- `experiments/run_f14_ablation_explanations.py` — same-state A0--A7 explanations and frozen F12 evidence integration.
- `experiments/analyze_f14_failure_modes.py` — pruning-frontier and descriptive failure hierarchy.
- `experiments/run_f14_final_reexplanation.py` — final A0/A7 R004-state comparison without simulator replay.
- `experiments/generate_f14_figures.py` — deterministic publication PNG/PDF generator.
- `experiments/verify_f14_artifacts.py` — fail-closed final package verifier and manifest writer.
- `tests/test_f14_explainability_aware_compression.py` — 29D, coalition, Shapley, actor, counterfactual, provenance, and immutability tests.
- `artifacts/f14_explainability_aware_compression_v1/` — machine-readable calibration, A0--A7/final comparisons, pruning/failure/retention diagnostics, figures, logs, classification, and SHA256 manifest.
