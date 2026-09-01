# Glosarium Figure F14 untuk Pembaca Indonesia

Seluruh figure baru memakai bahasa Inggris agar siap dipakai pada manuscript internasional. Dokumen singkat ini menjelaskan label yang muncul pada figure tanpa mengganti istilah teknisnya.

## Cara membaca empat kolom diagnosis

| Label figure | Arti dalam Bahasa Indonesia | Pertanyaan yang dijawab |
|---|---|---|
| **Semantic Attribution Preservation** | kesamaan pola attribution semantik terhadap Original Policy | Apakah urutan dan share kontribusi enam grup masih serupa? |
| **Counterfactual Response Preservation** | kesamaan respons terhadap perubahan input semantik | Apakah action berubah dalam arah/besar yang serupa ketika konsep input diubah? |
| **Action Fidelity to Original Policy** | kesetiaan output action pada input 29D yang sama | Apakah `v_cmd` dan `omega_cmd` cukup dekat dengan Original? |
| **C4 Task Completion / C4 Behavior** | hasil interaksi closed-loop pada skenario gabungan C4 | Apakah robot menyelesaikan task dan gate keselamatan yang diuji? |

Keempatnya bukan sinonim. Model dapat lulus C4 tetapi gagal action-fidelity threshold; model juga dapat lulus fidelity sementara attribution-nya shifted.

## Model names

| Label utama pada figure | ID teknis | Arti |
|---|---:|---|
| Original Policy | A0 | actor FP32 `29→256→256→2` sebelum optimisasi |
| Pruning Only | A1 | hidden-neuron pruning tanpa recovery |
| Pruning + Knowledge Distillation | A2 | model pruned dilatih meniru Original |
| Post-Training Quantization (PTQ) | A3 | actor original-size dikonversi langsung ke INT8 |
| QAT + Knowledge Distillation | A4 | actor unpruned menjalani fake-quant recovery lalu INT8 |
| Pruning + PTQ | A5 | actor pruned langsung dikuantisasi |
| Pruning + Distillation + PTQ | A6 | actor pruned dipulihkan dalam FP32 lalu PTQ |
| Final INT8: Pruning + Distillation + QAT | A7 | actor 64×64 dengan recovery FP32 dan fake-quant/QAT distillation |

## Status labels

| Label | Makna |
|---|---|
| REFERENCE | Original Policy adalah titik acuan, bukan kandidat yang dinilai terhadap dirinya sendiri |
| PRESERVED | gate yang dibekukan menyatakan pola/response/behavior dipertahankan |
| PARTIAL | sebagian cell primary dipertahankan, tetapi tidak cukup untuk klasifikasi preserved penuh |
| SHIFTED | perubahan melewati batas preservation yang dibekukan |
| PASS | action-fidelity gate lulus |
| FAIL | action-fidelity gate gagal |
| NOT PRESERVED | closed-loop C4 memburuk terhadap syarat yang dibekukan |
| UNRESOLVED | bukti yang valid tidak tersedia; tidak boleh diperlakukan sebagai zero drift |

## Attribution figure terms

| Istilah figure | Penjelasan |
|---|---|
| Group Shapley | metode yang membagi contribution actor output di antara enam semantic groups menggunakan semua 64 coalition |
| Mean absolute Shapley share | bagian relatif dari total absolute attribution; bukan persentase sebab fisik |
| Driving phase | konteks publik: nominal, curve, pedestrian relevant, stop required, atau stop satisfied |
| Phase–action cell | satu phase × satu output; lima phase × dua action = 10 cell |
| Same reference assignments | kedua actor menerima factual/reference coalition vectors yang identik |
| Attribution redistribution | share relatif berpindah antargrup; bukan bukti causal mechanism dunia |

## Counterfactual figure terms

| Istilah figure | Penjelasan |
|---|---|
| Semantic intervention | satu tuple konsep pada input 29D diganti secara terkontrol |
| Pedestrian removed | pedestrian belief diganti neutral-absence tuple yang valid |
| Stop requirement removed | stop semantics diganti sesuai operator `stop_absent` yang dibekukan |
| Lane centered | lane tuple diubah menjadi kondisi center yang valid |
| Mean change in action | action counterfactual dikurangi action factual |
| Functional sensitivity | besarnya respons actor terhadap perubahan input; bukan causal effect dunia nyata |

Tiga primary counterfactual tests adalah pedestrian→`v_cmd`, stop→`v_cmd`, dan lane→`omega_cmd`. Denominator primary selalu **3**. Angka `3/8` pada report attempt awal bukan denominator primary yang valid; lihat `docs/F14_VISUAL_CONSISTENCY_AUDIT.md`.

## Compression terms

| Istilah | Penjelasan singkat |
|---|---|
| Pruning | menghapus hidden neurons sehingga actor dense menjadi lebih kecil; input 29D tidak dihapus |
| Knowledge Distillation (KD) | compressed student meniru deterministic physical actions Original teacher |
| PTQ | konversi model yang sudah selesai dilatih dari FP32 ke INT8 |
| QAT | simulasi quantization selama recovery/training sebelum conversion INT8 |
| INT8 | representasi integer 8-bit untuk deployment |
| Fidelity | kedekatan output optimized actor terhadap Original pada input yang sama |
| Closed-loop | policy bertindak berulang, environment berubah, lalu policy menerima observation berikutnya |

## Dataset labels

- **Development diagnostic: 500 states** — 100 state per phase, dipakai untuk A0–A7 mechanism analysis dan classification awal.
- **Final comparison: 4,400 states** — frozen R004 public states, dipakai hanya untuk Original Policy versus Final INT8 Policy.

Jangan membandingkan angka heatmap dua dataset seolah-olah nilai itu berasal dari sampel yang sama.

## Rujukan

- `docs/PANDUAN_PROYEK_DUCKIE_POMDP_DARI_NOL.md`
- `docs/F14_PROTOCOL.md`
- `docs/F14_REFERENCE_CALIBRATION.md`
- `docs/F14_FINAL_REEXPLANATION.md`
- `artifacts/f14_explainability_aware_compression_v1/figures_explained_id/figure_source_manifest.json`
