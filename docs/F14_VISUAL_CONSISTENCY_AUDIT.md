# Audit Konsistensi Visual F14

## Ruang lingkup

Audit ini dibuat untuk package visual human-readable. Ia tidak mengubah report, artifact, threshold, actor, atau status historis F11–F14. Semua figure baru dihasilkan dari source data beku oleh `experiments/generate_f14_explained_figures_id.py`.

## Kesimpulan

**PASS untuk package dokumentasi/visualisasi.** Satu inkonsistensi denominator historis ditemukan dan dijelaskan tanpa menulis ulang report lama. Dataset development dan final dipisahkan. Label figure utama memakai bahasa Inggris dan nama model yang mudah dibaca; ID A0–A7 hanya dipertahankan sebagai metadata/reproducibility.

## 1. Resolusi 1/3 versus 3/8

### Bukti protocol dan code aktif

Primary counterfactual preservation terdiri dari tepat tiga cell:

1. `pedestrian_relevant × pedestrian_absent × v_cmd`;
2. `stop_required × stop_absent × v_cmd`;
3. `lane_curve × lane_centered × omega_cmd`.

`src/duckie_pomdp/explain/compression_diagnostics.py` menghitung klasifikasi primary terhadap tiga cell tersebut dan memeriksa sham sebagai integrity gate terpisah.

`artifacts/f14_explainability_aware_compression_v1/protocol_alignment_amendment.json` menyatakan bahwa summary attempt awal salah karena memperlakukan dua actor outputs untuk beberapa interventions sebagai delapan primary cells. Attempt itu dipertahankan di `_attempt1_counterfactual_scope_error/` untuk audit trail.

### Dampak pada reporting

- `docs/F14_ABLATION_EXPLANATION.md` dan machine-readable corrected metrics memberi **Pruning + PTQ = 1/3 primary counterfactual tests preserved**.
- `docs/F14_FAILURE_MODE_REPORT.md` historis masih memuat “3/8 functional cells”. Nilai ini bukan denominator primary yang valid; report historis tidak diubah karena immutable.
- Figure baru menggunakan **1/3**, bukan 3/8.
- Delapan action/intervention combinations attempt awal boleh disebut auxiliary attempt comparisons hanya dalam penjelasan audit, tidak sebagai primary classification.

**Source of truth baru untuk visual:** `artifacts/f14_explainability_aware_compression_v1/ablation_comparison_metrics.json`.

## 2. Development versus final

| Analisis | Dataset | Models | Tujuan | Dilarang dicampur dengan |
|---|---|---|---|---|
| A0–A7 development diagnosis | 500 states, 100 per phase | semua ablation | memahami operasi compression | final 4.400-state shares |
| Final re-explanation | 4.400 frozen R004 states | Original dan Final INT8 | final shared model-agnostic comparison | development classification counts |

Figure 3–8 memakai development results. Figure 9–10 memakai final 4.400-state results. Caption/subtitle masing-masing menyebut basis data.

## 3. Status mapping yang dipakai

| Dimensi diagnosis | Nilai valid |
|---|---|
| Semantic Attribution | REFERENCE, PRESERVED, PARTIAL, SHIFTED, UNRESOLVED |
| Counterfactual Response | REFERENCE, PRESERVED, PARTIAL, SHIFTED, INVALID/UNRESOLVED |
| Action Fidelity | REFERENCE, PASS, FAIL, UNRESOLVED |
| C4 Behavior | REFERENCE, PRESERVED, NOT PRESERVED, UNRESOLVED |

`PARTIAL` dibuat berbeda warna dari `SHIFTED`. Evidence unavailable tidak pernah diisi nol.

## 4. Terminologi visual

- “confusion matrix” tidak digunakan; figure 8 bernama **Optimization-Stage Failure-Mode Diagnostic Matrix**.
- L1/L2/L3/L4 tidak digunakan sebagai visible column labels.
- “Semantic cells” diganti **Semantic Attribution Preservation**.
- “Functional cells” diganti **Counterfactual Response Preservation**.
- “Fidelity gate” diganti **Action Fidelity to Original Policy**.
- “C4 completion” diganti **C4 Task Completion/Behavior**.
- Main model labels menggunakan nama manusia, bukan A0–A7.

## 5. Batas klaim yang diaudit

- Figure pipeline menyatakan PPO tidak menerima RGB, detector boxes, atau world pose secara langsung.
- Group Shapley disebut relative attribution, bukan causal importance.
- Counterfactual disebut semantic policy-input intervention, bukan world-level causality.
- 3,04× disebut actor-only CPU speedup, bukan end-to-end visuomotor speedup.
- F12 tetap `PASS` untuk C4-only deployment.
- F13 tetap `LIMITED`; blocked gradient attribution tidak diperbaiki secara retrospektif.
- F14 tetap `LIMITED` karena semantic dan functional sensitivity shifted walau fidelity/C4 preserved.
- Retention outside C4 tetap terbatas; semantic retention explanation `UNRESOLVED`.

## 6. Perbedaan yang bukan inkonsistensi

1. **F11 Distributional IG versus F14 Group Shapley.** Keduanya metode dan reference construction yang berbeda; nilai attribution tidak harus identik.
2. **F13 gradient attribution blocked versus F14 Group Shapley available.** F14 adalah extension model-agnostic baru yang memakai forward inference direct; ia tidak memulihkan missing historical QAT surrogate.
3. **Action fidelity FAIL dengan C4 preserved.** Numerical same-state gate dan closed-loop task completion menguji hal berbeda.
4. **Development Original heatmap versus final Original heatmap.** Sampel berbeda (500 versus 4.400), sehingga angka dapat berbeda.

## 7. Pemeriksaan file dan source data

Generator figure membaca:

- `artifacts/f14_explainability_aware_compression_v1/ablation_comparison_metrics.json`;
- `artifacts/f14_explainability_aware_compression_v1/failure_modes/failure_hierarchy.json`;
- `artifacts/f14_explainability_aware_compression_v1/ablation_group_summary.csv`;
- `artifacts/f14_explainability_aware_compression_v1/final_comparison_metrics.json`;
- `artifacts/f14_explainability_aware_compression_v1/final_a0_a7_counterfactuals.csv`;
- `artifacts/f12_belief_ppo_compression_v1/benchmarks/actor_benchmarks.json`.

Setiap figure PNG/PDF memiliki SHA256 dalam `artifacts/f14_explainability_aware_compression_v1/figures_explained_id/figure_source_manifest.json`.

## 8. Dokumentasi yang tidak ditemukan

Root `EXPERIMENT_PLAN.md` tidak ada pada repository aktif. Salinan histori ada di `refine-logs/`, tetapi tidak dipakai sebagai source of truth. F0–F8 direkonstruksi dari `FORMULATION.md`, `GATES.md`, `IMPLEMENTATION_NOTES.md`, source code domain/integration, config, dan artifact yang memang tersedia. Tidak semua tahap awal mempunyai report standalone dengan nama tahap masing-masing.

## 9. Immutability

Package baru menambah:

- satu generator figure;
- direktori `artifacts/f14_explainability_aware_compression_v1/figures_explained_id/`;
- panduan Indonesia;
- glosarium figure;
- audit konsistensi ini.

Ia tidak mengedit frozen F11/F12/F13/F14 reports, registries, model checkpoints, metric CSV/JSON, atau threshold protocol.
