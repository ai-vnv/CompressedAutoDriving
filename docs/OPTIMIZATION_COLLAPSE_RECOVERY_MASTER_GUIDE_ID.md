# Panduan Master: Kolaps dan Pemulihan Kompetensi Selama Optimisasi Actor Belief-PPO

**Dokumen definitif proyek — mengintegrasikan F12 (kompresi historis), F15 (lokalisasi
kolaps + pemulihan), F16 (studi robustness sekunder), dan F17 (urutan metode optimisasi
+ kuantisasi).**

Ditujukan bagi pembaca yang memahami konsep teknik/penelitian tetapi ingin penjelasan
langkah-demi-langkah yang jelas. Setiap klaim besar merujuk ke artefak nyata di
repository.

---

## Ringkasan satu paragraf

Actor Belief-PPO asli (29→256→256→2, ~74 ribu parameter) menguasai lima kurikulum
mengemudi C0–C4. Setelah dikompresi (pruning ke 64×64, distilasi, kuantisasi INT8),
model final historis kehilangan C0–C2. Proyek ini menemukan bahwa **kolaps pertama
terjadi pada pruning**; bahwa **cakupan data distilasi menentukan kurikulum mana yang
pulih** (distilasi fokus-C4 hanya memulihkan C3/C4; distilasi seimbang C0–C4 memulihkan
kelimanya); bahwa **pemulihan itu nyata tetapi sensitif terhadap undian pelatihan**;
bahwa **jadwal pruning (Direct vs Progressive) tidak menunjukkan keunggulan yang
stabil**; bahwa **PTQ menjatuhkan kembali C3/C4 pada model pulihan** dan **rute QAT+KD
tidak menyelamatkannya** (dengan fenotipe kegagalan berlawanan); dan — temuan pamungkas —
bahwa **kegagalan INT8 itu adalah efek interaksi antara width sempit dan kuantisasi,
bukan efek kuantisasi semata**, karena PTQ pada model asli tanpa pruning lolos kelima
kurikulum. Tidak ada kandidat INT8 yang memenuhi seluruh gate beku, sehingga holdout
final (seed 180301–180308) **tidak pernah dibuka** dan tetap tersegel untuk percobaan
berikutnya.

---

## Istilah kunci

| Istilah | Arti |
|---|---|
| **Actor width** | lebar hidden layer: 64 / 96 / 128 / 192 / 256 |
| **Pruning schedule** | Direct (pangkas langsung ke width target) vs Progressive (bertahap dengan KD di antaranya) — subjek F16 |
| **Optimization-method order / pathway** | urutan & penempatan pruning, distilasi, PTQ, QAT — subjek F17 |
| **Same-state fidelity** | seberapa mirip aksi model pada input 29D yang identik (offline) |
| **Closed-loop retention** | apakah robot benar-benar menyelesaikan tugasnya di simulator |
| **Backend deterministik** | konfigurasi CUDA yang membuat evaluasi tereproduksi bit-per-bit |

Kedua ukuran itu **harus dipisah**: proyek ini menemukan dissosiasi di **dua arah** —
model yang fidelity-nya lolos tapi perilakunya gagal (A6/A8), dan model yang perilakunya
sempurna tapi fidelity-nya gagal (A4).

---

## A. KOLAPS PERTAMA — pruning

**Tujuan:** menemukan tahap optimisasi pertama tempat tiap kurikulum berubah PASS→FAIL.

**Proses:** seluruh model historis F12 (A0–A7 + frontier width) dievaluasi ulang pada 8
seed baru berpasangan per kurikulum (F15), lalu direplikasi deterministik pada blok
180201–208 (F17).

**Hasil:** untuk **kelima kurikulum**, transisi pertama adalah
`Original → Pruning Only`. Pada blok deterministik F17, pruning-only (A1) gagal semua:
invalid-pose 62–100% di C0–C2/C4, stop-violation 100% di C3.

**Bukti:**
- `artifacts/f15_cross_curriculum_recovery_v1/localization/failure_localization_decision.json`
- `artifacts/f17_optimization_method_order_v1/results/pathway_results.csv` (baris A1)
- Figure: `artifacts/f16_sequence_int8_recovery_v1/figures/01_cross_curriculum_collapse_map.png`

**Boleh diklaim:** "Pruning adalah tahap optimisasi pertama yang teramati tempat setiap
kurikulum berubah dari PASS menjadi FAIL."
**Tidak boleh:** "neuron yang dibuang tidak berguna" atau "pruning penyebab semua
kegagalan berikutnya."

## B. PEMULIHAN PARSIAL HISTORIS — distilasi fokus-C4

Distilasi historis F12 memakai state pengembangan C4. Hasilnya (A2, deterministik):
C3 dan C4 pulih penuh (completion 1.000), tetapi C0–C2 tetap nol — invalid-pose 100%
di C0, lane-failure 100% di C2. Model final INT8 historis (A7) mewarisi pola yang sama.

**Bukti:** baris A2 dan A7 di `pathway_results.csv`;
figure `07_distillation_coverage_recovery.png` di
`artifacts/f17_optimization_method_order_v1/figures/`.

## C. PEMULIHAN MULTI-KURIKULUM — cakupan rehearsal adalah faktornya

**Eksperimen terkontrol inti proyek.** Yang ditahan tetap: parent pruned yang sama, guru
Original yang sama, Smooth-L1 sama, Adam/80 epoch/batch 512/lr 0.001 sama. Yang diubah
**hanya** cakupan data distilasi: dari fokus-C4 menjadi seimbang C0–C4 (62.176 state
publik, tanpa privileged truth; SHA256 `385e2a3a…`).

**Hasil (A2 vs A3, blok deterministik):**

| | C0 | C1 | C2 | C3 | C4 |
|---|---|---|---|---|---|
| A2 fokus-C4 | 0.000 | 0.000 | 0.000 | 1.000 | 1.000 |
| A3 seimbang | **1.000** | **0.875** | **1.000** | 1.000 | 1.000 |

**Boleh diklaim:** "Memperluas cakupan rehearsal dari distribusi fokus-C4 historis ke
cakupan seimbang C0–C4 memulihkan kelima kurikulum pada actor 64×64 FP32 pulihan yang
diuji."
**Tidak boleh:** "balanced KD selalu memulihkan C0–C4" (lihat D).

**Bukti:** checkpoint anchor
`artifacts/f15_cross_curriculum_recovery_v1/recovery/fp32/w64/actor_multicurriculum_kd_fp32.pt`
(SHA256 `64c84cd0…`); baris A3 di `pathway_results.csv`.

## D. KUALIFIKASI ROBUSTNESS — sensitif realisasi pelatihan (F16, sekunder)

Prosedur pemulihan yang identik, hanya berbeda seed distilasi, memberi hasil berbeda:
D96 lolos kelimanya di dua realisasi lalu kolaps total di realisasi ketiga (C3 completion
0.000, stop-violation 1.000). Diagnostik 2×2 backend-matched memisahkan dua efek secara
bersih: **sensitivitas realisasi pelatihan** (D64 mencatat stop-violation 0.500 di kedua
blok evaluasi, checkpoint F15 mencatat 0.000 di keduanya) dan **sensitivitas blok
evaluasi** (checkpoint byte-identik berbalik PASS→FAIL hanya lewat `minimum_clearance`).

**Boleh diklaim:** "Pemulihan menunjukkan sensitivitas realisasi pelatihan."
**Tidak boleh:** "seed pelatihan menyebabkan mekanismenya."

**Bukti:** `artifacts/f16_sequence_int8_recovery_v1/results/model_vs_evalblock_2x2.json`,
`training_realization_results.csv`, `width_results.csv`;
figure `04_training_realization_stability.png`.

## E. JADWAL PRUNING — tidak ada keunggulan stabil (F16, sekunder)

Lima sel width×kurikulum pernah menunjukkan beda vonis Direct vs Progressive; **kelimanya
tidak stabil antar seed pelatihan** (arah berbalik atau hilang). Width juga
**non-monoton**: 64 stabil gagal (6 model independen, mayoritas safety-relevant), 96 dan
128 berubah-ubah antar realisasi, 192 gagal C2.

**Boleh diklaim:** "Tidak ada keunggulan jadwal-pruning yang stabil yang ditetapkan" dan
"tidak teramati hubungan width–retention yang monoton."

**Bukti:** `sequence_classification.json`;
figure `03_direct_vs_progressive_fp32.png`, `05_width_retention_matrix.png`;
penghentian dini tercatat di `docs/F16_DECISIVE_EARLY_STOP.md`.

## F. URUTAN METODE OPTIMISASI — matriks pathway F17

Sembilan pathway, semuanya checkpoint beku yang sudah ada (nol pelatihan baru), blok
deterministik identik 180201–208, gate identik:

| ID | Pathway | Prec | C0 C1 C2 C3 C4 |
|---|---|---|---|
| A0 | Original | FP32 | REF ×5 |
| A1 | prune | FP32 | ✗ ✗ ✗ ✗ ✗ |
| A2 | prune → KD(C4) | FP32 | ✗ ✗ ✗ ✓ ✓ |
| A3 | prune → KD(seimbang) | FP32 | ✓ ✓ ✓ ✓ ✓ |
| A4 | PTQ tanpa pruning | INT8 | ✓ ✓ ✓ ✓ ✓ |
| A5 | prune → PTQ | INT8 | ✗ ✗ ✗ ✗ ✗ |
| A6 | prune → KD(seimbang) → PTQ | INT8 | ✓ ✓ ✓ ✗ ✗ |
| A7 | prune → KD(C4) → PTQ → QAT(C4) | INT8 | ✗ ✗ ✗ ✓ ✓ |
| A8 | prune → KD(seimbang) → QAT(seimbang) | INT8 | ✓ ✓ ✓ ✗ ✗ |

**Perbandingan penempatan (A5 vs A6):** menyisipkan distilasi seimbang sebelum PTQ
mengubah "gagal kelimanya" menjadi "mempertahankan C0–C2 penuh". *Boleh diklaim:*
"Menyisipkan distilasi seimbang sebelum PTQ mempertahankan jauh lebih banyak kompetensi
daripada langsung mengkuantisasi actor pruned, di bawah pathway yang diuji." *Tidak
boleh:* bukti faktorial urutan operasi.

**Bukti:** `artifacts/f17_optimization_method_order_v1/results/` (pathway_results.csv,
pathway_summary.json); figure `06_optimization_method_pathways.png`,
`10_method_order_comparison.png`; wording terkunci di
`docs/F17_COMPARISON_INTERPRETATION_AMENDMENT.md`.

## G. KOLAPS KUANTISASI — PTQ pada checkpoint pulihan yang tetap

Checkpoint anchor byte-identik, seed identik, tanpa pelatihan di mana pun: A3 lolos
kelimanya di FP32; setelah PTQ beku (A6), C3 gagal safety-relevant (completion 0.375,
restart 0.375 — **robot berhenti dengan benar lalu tidak pernah jalan lagi**) dan C4
gagal behavioural (progress).

**Boleh diklaim:** "Konversi INT8 di bawah prosedur PTQ beku terasosiasi dengan kegagalan
retention baru untuk checkpoint FP32 pulihan yang tetap ini."

**Bukti:** baris A3/A6 di `pathway_results.csv`;
figure `08_fp32_to_ptq_transition.png`; addendum determinisme INT8
`integrity/int8_determinism_addendum.json` (bit-eksak, PASS).

## H. PEMULIHAN INT8 — tidak tercapai; dan temuan interaksi

**Rute QAT (A8)** dari parent yang sama juga gagal C3/C4 — dengan fenotipe
**berlawanan**: PTQ membeku di stop line (stop-violation 0.000), QAT menerobos stop
(0.875 di C4). Fidelity A8 lebih baik dari A6 di semua kurikulum, perilakunya lebih
buruk. *Boleh diklaim:* "Di bawah prosedur yang diuji, rute kuantisasi QAT+KD tidak
mempertahankan retention yang hilang pada rute PTQ." *Tidak boleh:* "QAT memperbaiki
model PTQ yang gagal" (A8 tidak pernah melatih graf INT8 A6).

**Temuan pamungkas (A0 vs A4):** PTQ pada Original **tanpa pruning** lolos kelima
kurikulum sebagai INT8. Maka:

> **Kegagalan retention INT8 pada C3/C4 adalah efek interaksi antara width sempit dan
> kuantisasi — bukan efek kuantisasi semata.** Pruning+KD sendirian lolos (A3, FP32);
> kuantisasi sendirian lolos (A4, INT8); kombinasinya yang gagal (A6/A8).

Pola deskriptif tambahan: setiap pathway INT8 mempertahankan subset yang selaras dengan
penekanan distilasinya (fokus-C4 → {C3,C4}; seimbang → {C0,C1,C2}). INT8 *mampu*
melewati setiap kurikulum, tetapi tidak ada pathway width-64 teruji yang mempertahankan
kelimanya sekaligus. Mekanismenya di luar lingkup beku
(`docs/F16_QUANTIZATION_SCOPE_LIMITATION.md`); **representasi kuantisasi** (granularitas
aktivasi, kalibrasi, dsb.) adalah variabel tunggal yang paling beralasan untuk studi
berikutnya.

**Bukti:** figure `09_qat_recovery.png`; baris A4 di `pathway_results.csv`.

## H2. KONTROL PRESISI FP16 (F18) — presisi bukan penyebabnya

Pertanyaan termurah yang belum pernah diuji: apakah *setiap* pengurangan presisi merusak
C3/C4, atau khusus kuantisasi integer? Checkpoint anchor yang sama di-cast ke FP16 (tanpa
retrain, tanpa ubah width), lalu dievaluasi pada blok deterministik yang sama.

| | C0 | C1 | C2 | C3 | C4 | fidelity |
|---|---|---|---|---|---|---|
| A3 FP32 anchor | ✓ | ✓ | ✓ | ✓ | ✓ | lolos 5/5 |
| **F16H FP16** | **✓** | **✓** | **✓** | **✓** | **✓** | **lolos 5/5** |
| A6 INT8 PTQ | ✓ | ✓ | ✓ | ✗ | ✗ | gagal 4/5 |

F16H adalah **kandidat pertama di proyek ini yang lolos gate perilaku DAN gate fidelity
sekaligus di kelima kurikulum**. Perilakunya nyaris tak terbedakan dari induk FP32-nya
(selisih progress 0,001–0,008 m; satu episode C1 yang tidak selesai adalah episode yang
sama dengan induknya, jadi sifat warisan, bukan kegagalan baru).

**Boleh diklaim:** "Pengurangan presisi floating-point ke FP16 (dengan akumulasi selebar
FP32) mempertahankan kompetensi lintas kurikulum pada checkpoint pulihan yang tetap ini,
sementara prosedur INT8 yang diuji tidak. Kegagalan C3/C4 karena itu bukan konsekuensi
umum dari pengurangan presisi numerik."

Digabung dengan kontrol A4 di bagian H, gambarannya konsisten: kegagalan butuh **dua-duanya**
— width sempit *dan* kuantisasi integer.

**Biaya, apa adanya:** file 15.865 B vs 29.295 B; memori parameter logis 12,4 KB vs 24,8 KB
(tepat 2×). Tetapi **FP16 21% lebih LAMBAT** (24,5 µs vs 19,4 µs) di backend CPU x86 ini —
tidak ada jalur komputasi half native yang terpakai. Jadi F16H adalah hasil
presisi/memori, **bukan** hasil latency-deployment. INT8 1,50× lebih cepat (12,9 µs), tapi
filenya justru lebih besar (34.088 B) karena TorchScript traced graph.

**Status kelayakan:** aturan seleksi beku mensyaratkan INT8, sehingga kandidat FP16 tidak
bisa eligible secara konstruksi. Kelayakan dilaporkan tanpa diubah, dan holdout tetap
tersegel. Mengubah syarat presisi deployment adalah keputusan investigator yang harus
diambil **prospektif** (amendment ter-hash sebelum holdout dibuka), bukan pelonggaran
pasca-hasil.

**Bukti:** `docs/F18_FP16_CONTROL_REPORT.md`;
`artifacts/f18_fp16_control_v1/` (integrity, results, figure 12).

## I. MODEL FINAL — tidak ada yang memenuhi syarat; holdout tetap tersegel

Aturan beku: kandidat final harus INT8 **dan** lolos seluruh gate perilaku, fidelity,
safety, provenance. Hasil: **nol dari sembilan pathway eligible.**

- A4 lolos seluruh perilaku tetapi gagal gate fidelity pada komponen korelasi
  Pearson/Spearman di 4/5 kurikulum (mis. Spearman C4 0.9229 vs gate 0.970) — mereplikasi
  temuan F12 historis; MAE-nya sangat kecil. Ini dissosiasi arah kedua: perilaku utuh,
  gate numerik gagal. Gate **tidak diubah**.
- A6/A8 gagal perilaku C3/C4.

Klasifikasi tercatat: `NO_ELIGIBLE_FINAL_CANDIDATE`
(`artifacts/f17_optimization_method_order_v1/results/eligibility_outcome.json`).
**Seed holdout 180301–180308 tidak pernah dibuka** — selama F15, F16, maupun F17 — dan
tetap bernilai penuh untuk percobaan preregistered berikutnya. Tidak ada tuning tambahan
yang dijalankan untuk memaksa PASS.

## J. BUKTI VISUAL

F15 tidak dapat menghasilkan rekaman kamera yang sah karena RGB tidak disimpan saat
rollout ilmiah; percobaan replay ditolak oleh validasinya sendiri
(`docs/F15_VISUAL_REPLAY_IMPLEMENTATION_AMENDMENT.md`). F16/F17 memperbaikinya: RGB
diambil **selama rollout ilmiah primer** melalui ring buffer yang terbukti tidak
mengubah eksekusi policy (delta aksi nol, panjang episode identik).

- F16: `artifacts/f16_sequence_int8_recovery_v1/primary_media/<model>/<cur>/seed_*/`
  (MP4, GIF, contact sheet, JSON ber-hash)
- F17: `artifacts/f17_optimization_method_order_v1/primary_media/<pathway>/<cur>/seed_*/`
- Telemetri per-step: direktori `telemetry/` pada masing-masing namespace
- Label pasangan: **Same-Seed Primary Rollouts** — bukan "causal counterfactual
  trajectory"

## K. KETERBATASAN

1. **Sensitivitas realisasi pelatihan** — vonis retention satu run pelatihan bisa
   berbalik pada seed lain (F16); kesimpulan F17 terkondisi pada satu anchor.
2. **Sensitivitas blok evaluasi** — `minimum_pedestrian_clearance_m` adalah statistik
   minimum antar-episode, elemen paling rapuh dalam set gate; tiga model sehat gagal
   hanya karenanya.
3. **Kerapuhan metrik korelasi** — Pearson/Spearman tidak informatif pada sinyal
   bervarians rendah; kedua arah dissosiasi fidelity-vs-perilaku teramati.
4. **Prosedur kuantisasi tetap** — semua klaim terbatas pada PTQ/QAT x86 statis
   per-tensor yang diuji; representasi lain belum pernah divariasikan.
5. **Generalisasi terbatas** — lima kurikulum ini, simulator ini, 8 seed per sel, satu
   blok evaluasi utama.
6. **Backend non-deterministik historis** — hasil F15 memakai backend lama (noise ±1
   episode/8); semua perbandingan F17 memakai backend deterministik bit-eksak.

---

## Rantai cerita akhir

```
KOMPETENSI ASLI (A0: C0-C4)
   ↓ pruning ke 64                    → KOLAPS PERTAMA (kelima kurikulum)     [A]
   ↓ KD fokus-C4                      → pulih parsial: C3/C4 saja             [B]
   ↓ KD seimbang C0-C4                → PULIH PENUH (FP32, anchor)            [C]
   |    (dengan kualifikasi: sensitif realisasi pelatihan)                    [D]
   |    (jadwal pruning Direct/Progressive: tidak ada keunggulan stabil)      [E]
   ↓ PTQ                              → C3/C4 jatuh lagi (beku di stop)       [G]
   ↓ rute QAT+KD                      → tetap gagal (menerobos stop)          [H]
   |
   KONTROL: PTQ tanpa pruning         → LOLOS SEMUA → interaksi width×INT8    [H]
   KONTROL: FP16 (presisi saja)       → LOLOS SEMUA → presisi bukan penyebab  [H2]
   ↓
   TIDAK ADA KANDIDAT INT8 ELIGIBLE → HOLDOUT TETAP TERSEGEL                  [I]
```

Setiap panah didukung oleh CSV/JSON ber-hash, telemetri per-step, dan (untuk kegagalan
F16/F17) video kamera dari rollout ilmiah aslinya. Hasil negatif pada tahap akhir adalah
hasil yang sah: proyek ini tidak mencari "model yang lolos", melainkan menjawab di mana,
mengapa, dan dengan intervensi apa kompetensi hilang dan pulih — dan jawaban-jawaban itu
kini terdokumentasi dengan bukti yang dapat diaudit.
