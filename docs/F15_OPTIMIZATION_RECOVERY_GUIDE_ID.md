# Panduan Optimisasi dan Recovery Belief-PPO — F15

Panduan ini menjelaskan tahap optimisasi actor dari awal, alasan F15 diperlukan,
cara kompetensi lintas-curriculum diuji, dan bagaimana recovery dilakukan. Dokumen ini
tidak mengubah hasil historis F10–F14. Istilah *Original Policy* berarti actor Belief-PPO
beku dari F10, sedangkan *optimized policy* berarti actor yang struktur atau presisi
angkanya telah diubah.

Empat jenis bukti harus selalu dibedakan:

1. **same-state action fidelity**: input 29D sama, lalu action numerik dibandingkan;
2. **closed-loop behavior**: policy berinteraksi dengan simulator sehingga action ikut
   mengubah state berikutnya;
3. **curriculum retention**: kompetensi C0–C4 tetap ada atau hilang setelah kompresi;
4. **compression efficiency**: jumlah parameter, byte, memori, dan latency actor.

## 1. Tujuan optimisasi actor

**Tujuan.** Membuat actor lebih kecil dan lebih cepat tanpa kehilangan kompetensi
mengemudi yang diwajibkan.

**Input.** Representasi semantik publik 29D yang sama seperti pada F10.

**Diproses bagaimana.** F12 menguji structured pruning, knowledge distillation (KD),
post-training quantization (PTQ), dan quantization-aware training (QAT). F15 tidak
mengoptimisasi perception, belief filter, reward, atau PPO lagi; F15 melokalisasi
kegagalan dan menjalankan recovery terkontrol.

**Output.** Actor terkompresi yang tetap menghasilkan `v_cmd` dan `omega_cmd`.

**Rujukan proyek:**

- `docs/F12_COMPRESSION_PROTOCOL.md`
- `docs/F15_PROTOCOL.md`
- `configs/f15_cross_curriculum_recovery_v1.toml`

## 2. Original actor 29→256→256→2

Original actor menerima 29 angka ter-normalisasi, melewati dua hidden layer berisi
256 neuron dengan aktivasi `Tanh`, lalu menghasilkan dua mean action ter-normalisasi.
Mapping fisiknya adalah:

\[
v_{cmd}=(u_v+1)\times0.2\;\text{m/s},\qquad
\omega_{cmd}=4u_\omega\;\text{rad/s},
\]

setelah output `u` dibatasi ke `[-1,1]`. Karena itu `v_cmd` berada pada 0–0,4 m/s dan
`omega_cmd` pada −4–4 rad/s. Actor mempunyai 73.986 parameter dense. Original
checkpoint tidak dilatih ulang dalam F12 maupun F15.

**Model/checkpoint:**
`artifacts/f10_ppo_visual_objects_v30/c4/ppo_selected.pt`

**SHA256:**
`02e898ce12d71f97016d50ed8a40574807e6d2fd995fc9f0dcd24f357f2c6250`

**Rujukan proyek:**

- `src/duckie_pomdp/control/ppo_protocol.py`
- `src/duckie_pomdp/optimization/actor_compression.py`
- `docs/F12_COMPRESSION_PROTOCOL.md`

## 3. Apa itu pruning?

Pruning adalah mengurangi struktur network. F12 memakai **structured hidden-neuron
pruning**: satu neuron hidden beserta koneksinya dibuang sebagai satu unit. Ini berbeda
dari sekadar mengubah sebagian weight menjadi nol.

Yang dipangkas adalah hidden width 256→192/128/96/64. Yang **tidak** dipangkas adalah
29 dimensi input, dua action output, MobileNet, YOLO, EKF, stop logic, normalisasi, dan
action mapping. Karena input tidak dibuang, setiap student tetap menerima seluruh 29D.

**Output masuk ke mana selanjutnya.** Actor pruned dapat langsung dievaluasi
(*Pruning Only*) atau menjalani KD.

**Rujukan proyek:**

- `src/duckie_pomdp/optimization/actor_compression.py`
- `artifacts/f12_belief_ppo_compression_v1/pruning/registry.json`
- `docs/F12_COMPRESSION_PROTOCOL.md`

## 4. Bagaimana pruning score dihitung?

Untuk hidden neuron `j`, F12 membekukan heuristic berikut:

\[
S_j=\lVert W^{in}_{j,:}\rVert_2+
    \lVert W^{out}_{:,j}\rVert_2+|b_j|.
\]

- `W_in` mengukur besar koneksi yang masuk ke neuron;
- `W_out` mengukur besar koneksi yang meneruskan output neuron;
- `b` adalah bias, yaitu dorongan dasar neuron sebelum kontribusi input;
- L2 norm mencegah weight positif dan negatif saling meniadakan seperti pada rata-rata
  bertanda.

Neuron diranking **per hidden layer**. Skor tertinggi dipertahankan. Jika skor sama,
indeks neuron original yang lebih rendah didahulukan. Semua width historis dibuat
langsung dari Original, bukan berantai dari checkpoint lebih kecil. Skor ini adalah
heuristic konektivitas, bukan bukti bahwa neuron berskor rendah “tidak berguna”.

**Script dan source:**

- `src/duckie_pomdp/optimization/actor_compression.py`
- `experiments/run_f12_compression.py`
- `configs/f12_belief_ppo_compression_v1.toml`

**Artifact survivor:**
`artifacts/f12_belief_ppo_compression_v1/pruning/registry.json`

## 5. Apa itu knowledge distillation?

Knowledge distillation (KD) pada proyek ini adalah supervised imitation terhadap
fungsi Original actor, bukan reinforcement learning ulang.

```text
                       input 29D yang sama
                         /             \
                Original teacher     compressed student
                         |             |
                    [v_T, omega_T] [v_S, omega_S]
                         \             /
                         Smooth-L1 loss
                                |
                       update student saja
```

Teacher selalu beku. Tidak ada reward simulator, critic target, atau ground-truth
action sebagai label. “Knowledge” yang dipindahkan adalah mapping action
`x_29D → [v_cmd, omega_cmd]`, bukan pasangan neuron teacher–student.

**Rujukan proyek:**

- `src/duckie_pomdp/optimization/actor_compression.py`
- `experiments/run_f12_compression.py`
- `docs/F12_COMPRESSION_PROTOCOL.md`

## 6. Bagaimana historical KD dilakukan?

F12 memberikan state publik dari pipeline normal kepada teacher dan student. Student
dioptimisasi dengan Adam dan Smooth-L1 pada action fisik. Agar steering tidak dominan
hanya karena rentangnya lebih besar, error dibagi full action range:

\[
e_v=(v_S-v_T)/0.4,\qquad e_\omega=(\omega_S-\omega_T)/8.
\]

Sampling historis diseimbangkan menurut public driving phase sehingga nominal driving
tidak menenggelamkan pedestrian/stop states. Parameter `log_std` PPO bersifat
state-independent; untuk student FP32 ia disalin, sedangkan deployment fidelity tetap
menilai deterministic mean. Historical F12 terutama memakai state development C4.

**Config:** `configs/f12_belief_ppo_compression_v1.toml`

**Dataset/artifact:**
`artifacts/f12_belief_ppo_compression_v1/datasets/development_public_actor_states.npz`

**Hasil historis:** direct 64×64 pruning merusak fidelity dan C4; KD mengembalikan
fidelity dan C4 pada split F12. Detail angka terdapat di
`docs/F12_COMPRESSION_RESULTS.md` dan
`artifacts/f12_belief_ppo_compression_v1/final/ablation_table.csv`.

## 7. Apa itu PTQ?

**Post-Training Quantization (PTQ)** mengonversi actor yang sudah selesai dilatih dari
FP32 menjadi INT8. Calibration states digunakan untuk mengamati rentang activation,
lalu ditentukan scale dan zero-point. Secara sederhana:

\[
q=\operatorname{round}(x/s)+z,
\qquad \hat{x}=s(q-z).
\]

Beberapa nilai float dapat jatuh ke kode integer yang sama. Itulah quantization error.
PTQ tidak mengadaptasi weight melalui gradient setelah efek rounding diperkenalkan.
Pada implementasi proyek, `Linear` memakai static x86 INT8; weight adalah qint8
per-channel symmetric, activation quint8 per-tensor affine, dan `Tanh` tetap float di
antara quantize/dequantize boundaries.

**Rujukan proyek:**

- `src/duckie_pomdp/optimization/actor_compression.py`
- `experiments/run_f12_compression.py`
- `docs/F12_COMPRESSION_PROTOCOL.md`

## 8. Apa itu QAT?

**Quantization-Aware Training (QAT)** memperlihatkan efek pembulatan/clipping kepada
student selama recovery. Fake quantization mensimulasikan level INT8 pada forward pass,
tetapi masih mempertahankan jalur gradient untuk update weight. Dalam F12 dan F15 QAT
digabung dengan teacher-guided KD:

```text
Original action target
        ↓
fake-quantized student → Smooth-L1 KD → update student
        ↓
conversion → deployable static INT8 actor
```

QAT tidak mengubah definisi action dan bukan PPO retraining.

**Rujukan proyek:**

- `src/duckie_pomdp/optimization/actor_compression.py`
- `experiments/run_f15_recovery.py`
- `configs/f15_cross_curriculum_recovery_v1.toml`

## 9. Pipeline optimisasi historical F12

F12 tidak hanya menguji satu endpoint. Registry memuat delapan ablation:

| Nama manusia | Operasi | Struktur/presisi |
|---|---|---|
| Original Policy | tidak dikompresi | 256×256 FP32 |
| Pruning Only | direct structured pruning | 64×64 FP32 |
| Pruning + Knowledge Distillation | pruning lalu KD | 64×64 FP32 |
| Post-Training Quantization | PTQ pada Original | 256×256 INT8 |
| QAT + Distillation | fake-quant KD lalu INT8 | 256×256 INT8 |
| Pruning + PTQ | pruning langsung lalu PTQ | 64×64 INT8 |
| Pruning + Distillation + PTQ | KD recovery lalu PTQ | 64×64 INT8 |
| Final INT8 Policy | pruning, FP32 KD, QAT/KD, INT8 | 64×64 INT8 |

ID teknis A0–A7 tetap ada dalam registry untuk reproducibility, tetapi nama manusia
dipakai dalam laporan utama.

**Rujukan proyek:**

- `artifacts/f12_belief_ppo_compression_v1/final/ablation_registry.json`
- `docs/F12_COMPRESSION_ABLATION.md`
- `artifacts/f12_belief_ppo_compression_v1/final/model_selection.json`

## 10. Hasil historical final optimized model

Final F12 adalah actor 29→64→64→2 INT8 dengan 6.210 parameter dan SHA256
`f8e4e3ae5c43028d7b5d08e64c31d20dcce28153fb102ffac53a3b1c7b7cbc7e`.
Dalam scope C4 yang ditetapkan, ia mendapat PASS: completion 8/8, tanpa collision,
unsafe episode, stop violation, atau lane failure. Dibanding Original, parameter turun
91,61%, actor file turun 87,69%, dan median latency actor-only CPU membaik sekitar
3,04×. Angka ini **bukan** speedup end-to-end RGB→action karena perception tidak
dikompresi.

**Rujukan proyek:**

- `docs/F12_COMPRESSION_RESULTS.md`
- `artifacts/f12_belief_ppo_compression_v1/final/final_evaluation.json`
- `artifacts/f12_belief_ppo_compression_v1/final/model_selection.json`

## 11. Penemuan masalah C0–C2 collapse

Historical F12 retention memakai dua seed per C0–C3. Original menyelesaikan C0–C4,
tetapi Final INT8 menyelesaikan 0/2 episode pada C0, C1, dan C2; C3 dan C4 tetap
selesai. Dengan bukti itu F12 secara tepat diklasifikasikan **PASS hanya untuk scope
C4**, bukan deployment universal. Dua seed cukup sebagai warning faktual tetapi belum
melokalisasi operasi mana yang pertama memperkenalkan kegagalan.

**Rujukan proyek:**

- `docs/F12_COMPRESSION_RESULTS.md`
- `artifacts/f12_belief_ppo_compression_v1/final/final_evaluation.json`

## 12. Mengapa hasil C4 saja tidak cukup?

C4 menguji kombinasi lane, pedestrian, dan stop. Namun keberhasilan pada satu distribusi
trajectory tidak menjamin mapping policy tetap stabil pada C0–C3. Closed-loop bersifat
dinamis: error action kecil dapat mengubah pose, menghasilkan observation berbeda,
kemudian error berikutnya dapat teramplifikasi. Karena itu F15 menguji semua curriculum
pada seed baru dan sama antarmodel.

## 13. Tujuan F15

F15 menjawab tiga tahap pertanyaan:

1. **localization:** pada operasi mana kompetensi pertama kali berubah PASS→FAIL;
2. **objective diagnosis:** apakah action pada input 29D yang sama sudah menyimpang,
   dan event closed-loop apa yang muncul;
3. **controlled recovery:** apakah perubahan minimum—coverage KD lintas C0–C4—cukup,
   lalu apakah recovery bertahan setelah INT8.

F15 tidak menjalankan attribution, Group Shapley, Integrated Gradients, reward
optimization, atau PPO retraining.

**Rujukan proyek:**

- `docs/F15_PROTOCOL.md`
- `configs/f15_cross_curriculum_recovery_v1.toml`
- `experiments/run_f15_cross_curriculum_recovery.py`

## 14. Cara mencari tahap pertama collapse

**Input.** Delapan historical actor beku, lima curriculum, dan delapan paired seed baru
180001–180008.

**Proses.** Setiap actor menjalankan seed dan environment yang sama dalam sebuah
curriculum. PASS membutuhkan Original lulus gate absolute, kandidat lulus gate absolute,
dan regresi kandidat terhadap Original tetap di dalam margin yang dibekukan. Bila
Original gagal, hasil kandidat adalah UNRESOLVED, bukan compression failure.

**Output.** Matrix 8 model × 5 curriculum dan transisi PASS→FAIL pertama pada jalur
aktual Original→Pruning→Pruning+KD→Final INT8. Branch PTQ/QAT tetap dilaporkan sebagai
branch, bukan dipaksa menjadi riwayat linear palsu.

**Seed/config:** `configs/f15_cross_curriculum_recovery_v1.toml`

**Artifact:**
`artifacts/f15_cross_curriculum_recovery_v1/localization/failure_localization_decision.json`

## 15. Hasil Original versus semua tahap optimisasi

Pada delapan paired seed baru, hasilnya adalah:

| Tahap | C0 | C1 | C2 | C3 | C4 |
|---|---|---|---|---|---|
| Original Policy | REFERENCE | REFERENCE | REFERENCE | REFERENCE | REFERENCE |
| Pruning Only | FAIL | FAIL | FAIL | FAIL | FAIL |
| Pruning + KD | FAIL | FAIL | FAIL | PASS | PASS |
| PTQ | PASS | PASS | PASS | FAIL | PASS |
| QAT + Distillation | PASS | FAIL | FAIL | PASS | PASS |
| Pruning + PTQ | FAIL | FAIL | FAIL | FAIL | FAIL |
| Pruning + KD + PTQ | FAIL | FAIL | FAIL | PASS | FAIL |
| Final INT8 Policy | FAIL | FAIL | FAIL | PASS | PASS |

Original lulus gate absolute pada kelima curriculum. Direct Pruning adalah satu-satunya
operasi pertama yang mengubah semua curriculum dari competence Original menjadi FAIL.
Distillation historis mengembalikan C3/C4, tetapi tidak C0–C2.

**Sumber hasil:**

- `artifacts/f15_cross_curriculum_recovery_v1/localization/cross_curriculum_results.csv`
- `artifacts/f15_cross_curriculum_recovery_v1/localization/matrix_results.json`
- `artifacts/f15_cross_curriculum_recovery_v1/figures/01_cross_curriculum_competence_across_compression_stages.pdf`

## 16. Hasil per curriculum C0–C4

- **C0:** pertama gagal setelah Pruning Only; tidak pulih pada jalur historis final.
- **C1:** pertama gagal setelah Pruning Only; tidak pulih pada jalur historis final.
- **C2:** pertama gagal setelah Pruning Only; tidak pulih pada jalur historis final.
- **C3:** pertama gagal setelah Pruning Only; pulih setelah historical KD dan tetap
  dipertahankan Final INT8.
- **C4:** pertama gagal setelah Pruning Only; pulih setelah historical KD. PTQ pada
  actor recovered sempat gagal pada branch A6, lalu QAT/KD final memulihkannya.

Jadi “first collapse” sama untuk semua curriculum, tetapi recovery historisnya berbeda.

Curriculum aktual yang diuji adalah: C0 `small_loop`; C1 `experiment_loop`; C2
`experiment_loop_duckie_crossing`; C3 `experiment_loop_stop_only`; dan C4
`experiment_loop_combined`. Horizon masing-masing adalah 1.900, 2.700, 2.700, 2.700,
dan 4.200 step.

**Rujukan proyek:**

- `configs/f10_ppo_visual_objects_v30.toml`
- `docs/F10_PPO_CURRICULUM.md`
- `artifacts/f15_cross_curriculum_recovery_v1/figures/02_first_collapse_stage_by_curriculum.pdf`

## 17. Open-loop action fidelity

Same-state fidelity mengambil hanya trajectory public 29D dari Original, lalu memberi
baris yang **identik** kepada setiap actor secara offline. Yang dibandingkan adalah
`v_cmd` dan `omega_cmd`: MAE, RMSE, median, P95, P99, maximum, signed bias, Pearson,
Spearman, omega-sign disagreement di atas deadband 0,2 rad/s, dan saturation
disagreement. Ini bukan explanation; ini diagnosis mapping numerik actor.

Contoh paling besar adalah Pruning Only pada C0: v MAE 0,11880 m/s dan omega MAE
1,04927 rad/s. Historical KD menguranginya menjadi 0,00554 m/s dan 0,31409 rad/s,
tetapi C0 tetap gagal full fidelity gate. Pada C4 actor yang sama mencapai 0,00139 m/s
dan 0,02310 rad/s serta lulus.

Final INT8 lulus fidelity hanya pada C4 (v MAE 0,00235 m/s; omega MAE 0,03958 rad/s).
Pada C0 omega MAE masih 0,26946 rad/s. Ini menunjukkan bahwa fidelity harus dilaporkan
per curriculum, bukan hanya pada C4.

**Artifact:**
`artifacts/f15_cross_curriculum_recovery_v1/localization/open_loop_fidelity_by_curriculum.csv`

## 18. Closed-loop failure

Dalam closed loop, action memengaruhi trajectory dan input berikutnya. F15 menyimpan
completion, progress, collision, unsafe episode, lane failure, invalid pose, stop
violation/completion, restart, clearance, timeout, mean velocity, mean absolute yaw,
stationary fraction, dan termination reason.

Pruning Only menyelesaikan 4/8 episode C0, 4/8 C1, dan 0/8 C2–C4. Setelah historical
KD, C3/C4 kembali 8/8 tetapi C0–C2 tetap 0/8. Final INT8 juga 0/8 pada C0–C2 dan 8/8
pada C3/C4. Failure objektif didominasi lane failure dan invalid pose, bukan collision;
detail episode dan step dipertahankan dalam failure registry.

**Artifact:**
`artifacts/f15_cross_curriculum_recovery_v1/localization/failure_event_registry.csv`

## 19. Perbedaan open-loop versus closed-loop

| Pertanyaan | Open-loop same-state | Closed-loop |
|---|---|---|
| Input antaractor | persis sama | dapat berbeda setelah action pertama |
| Yang diisolasi | mapping 29D→action | policy + feedback environment |
| Bukti utama | error/correlation/sign/saturation | completion/safety/control/progress |
| Klaim | action fidelity | task behavior/retention |

Action fidelity yang baik tidak otomatis menjamin closed-loop sama; sebaliknya task
dapat selesai walaupun error numerik gagal threshold. Keduanya harus dilaporkan.

## 20. Pruning-width diagnosis

F15 menggunakan checkpoint historis yang benar-benar ada pada 192, 128, 96, dan 64,
masing-masing untuk Pruning Only dan Pruning+KD. Missing checkpoint tidak boleh
direkonstruksi. Tujuannya mencari pola hubungan width dengan retention, bukan melakukan
post-hoc model selection terhadap artifact historis.

Tidak ada width historis yang mempertahankan C0–C4 penuh. Pada Pruning Only, width
192×192 dan 96×96 lulus C0/C1 tetapi gagal C2–C4; 128×128 hanya lulus C0; 64×64 gagal
semua. Setelah historical KD, semua width masih gagal C0–C2, walaupun semuanya lulus
C3 dan width 64/96/128 lulus C4. Pola ini tidak monoton sempurna, sehingga belum boleh
disebut universal “capacity threshold”. Ia lebih dahulu mengarahkan F15 untuk menguji
coverage rehearsal dengan width/survivor tetap.

**Rujukan proyek:**

- `artifacts/f12_belief_ppo_compression_v1/pruning/registry.json`
- `artifacts/f15_cross_curriculum_recovery_v1/localization/pruning_width_retention.csv`
- `artifacts/f15_cross_curriculum_recovery_v1/figures/03_pruning_width_vs_curriculum_retention.pdf`

## 21. Bukti foto, GIF, dan video

F15 mempertahankan telemetry primer. Untuk setiap cell gagal dipilih satu failure trace
dengan aturan objektif. Trace frozen tidak menyimpan RGB. Percobaan replay simulator
dengan recorded action juga berhenti sebelum failure window, sehingga media replay itu
diberi status `UNRESOLVED`, dipindahkan ke subdirektori `unresolved/`, dan tidak dipakai
sebagai bukti.

Sebagai pengganti yang tetap setia pada data, 50 failure event dirender langsung dari
telemetry frozen. Setiap event mempunyai video MP4, GIF, contact sheet PNG, CSV, dan JSON.
Visual tersebut memperlihatkan progress, `v_cmd`, `omega_cmd`, dan lane lateral error
Original versus actor terkompresi. Ini adalah visualisasi data episode yang tercatat,
bukan rekonstruksi kamera dan bukan replicate statistik baru.

Contoh:

- `artifacts/f15_cross_curriculum_recovery_v1/failure_telemetry/A1/c0/seed_180002/failure_telemetry_window.mp4`
- `artifacts/f15_cross_curriculum_recovery_v1/failure_telemetry/A1/c0/seed_180002/failure_telemetry_window.gif`
- `artifacts/f15_cross_curriculum_recovery_v1/failure_telemetry/A1/c0/seed_180002/failure_telemetry_contact_sheet.png`
- `artifacts/f15_cross_curriculum_recovery_v1/failure_telemetry/A1/c0/seed_180002/failure_telemetry.csv`
- `artifacts/f15_cross_curriculum_recovery_v1/success_telemetry/A0/c0/seed_180001/success_telemetry_episode.mp4`

Video simulator A7 yang sudah ada tetap dapat dipakai untuk memahami perilaku C4 secara
kualitatif, tetapi bukan bukti F15:

- `artifacts/f12_belief_ppo_compression_v1/final/a7_c4_front_bev.mp4`
- `artifacts/f12_belief_ppo_compression_v1/final/a7_c4_front_bev.json`

**Manifest:**

- `artifacts/f15_cross_curriculum_recovery_v1/failure_telemetry/failure_telemetry_manifest.json`
- `artifacts/f15_cross_curriculum_recovery_v1/success_telemetry/success_telemetry_manifest.json`

## 22. Cara failure event dipilih secara objektif

Aturan dibekukan sebelum hasil dilihat:

1. pilih episode gagal dengan seed terkecil;
2. pilih event objektif pertama pada episode itu;
3. pertahankan semua label yang muncul pada step sama;
4. ekstrak window 90 step sebelum dan 45 step setelah event.

Label meliputi collision, unsafe, stop violation, lane failure, invalid pose, timeout,
dan termination tanpa completion. Frame tidak dipilih karena “dramatis” atau karena
besar action error.

**Config:** `configs/f15_cross_curriculum_recovery_v1.toml`

**Script:** `experiments/render_f15_failure_traces.py`

## 23. Hipotesis incomplete distillation coverage

Historical KD terutama dibangun dari C4 development states. F15 memperlakukan coverage
itu sebagai **hipotesis recoverable**, bukan kesimpulan sebab-akibat. Uji terkontrol
menjaga teacher, survivor indices, width, loss, optimizer family, budget, normalisasi,
dan action target tetap; hanya distribusi rehearsal yang diubah agar mencakup C0–C4.

## 24. Multi-curriculum knowledge distillation

**Tujuan.** Menguji apakah rehearsal coverage lintas-curriculum dapat memulihkan
retention.

**Input.** Public 29D dari Original rollouts pada seed 180101–180108 untuk C0–C4 serta
deterministic physical action Original.

**Diproses bagaimana.** Sampling memberi massa sama pada setiap curriculum, kemudian
massa sama pada setiap public phase yang didukung dalam curriculum. Student memakai
historical survivor indices dan Smooth-L1/Adam/budget yang sama.

**Output.** Recovered FP32 actor pada width yang sedang diuji.

**Output masuk ke mana.** Pertama ke recovery-selection seeds 180201–180208; hanya actor
yang lulus semua behavior dan fidelity gate boleh masuk PTQ.

**Config:** `configs/f15_cross_curriculum_recovery_v1.toml`

**Script/source:**

- `experiments/run_f15_recovery.py`
- `src/duckie_pomdp/optimization/cross_curriculum_recovery.py`

**Dataset manifest:**
`artifacts/f15_cross_curriculum_recovery_v1/recovery/datasets/dataset_manifest.json`

## 25. Hasil recovery 64×64

**BERHASIL.** Student 64×64 hasil multi-curriculum KD lulus **kelima** gate retention,
kelima gate fidelity, dan seluruh cek keselamatan pada seed recovery-selection
180201–180208: `eligible: true`, `all_curricula_behavior_pass: true`,
`fidelity.all_curricula_pass: true`, nol sub-check gagal.

| Kurikulum | Status | Completion pulihan | Completion Original | Progress pulihan (m) | Progress Original (m) |
|---|---|---:|---:|---:|---:|
| C0 | PASS | 1.000 | 1.000 | 5.345 | 5.337 |
| C1 | PASS | 0.875 | 0.750 | 7.245 | 6.454 |
| C2 | PASS | 0.875 | 0.875 | 7.906 | 7.897 |
| C3 | PASS | 1.000 | 0.875 | 7.250 | 6.818 |
| C4 | PASS | 1.000 | 1.000 | 7.209 | 7.200 |

Bandingkan dengan A2 historis yang memakai **width sama, survivor indices sama, loss
sama, optimizer sama, budget sama**, dan hanya berbeda pada cakupan data rehearsal:
completion A2 adalah **0.000** di C0, C1, dan C2.

Fidelity ikut pulih: omega MAE C0 turun dari 0.31409 rad/s (A2) dan 0.26946 rad/s (A7)
menjadi **0.03172 rad/s**, dengan omega sign disagreement 0.000 dan Pearson 0.99962
terhadap gate 0.980.

Karena hanya cakupan data yang diubah sementara kapasitas, loss, optimizer, dan budget
dipertahankan, hasil ini mendukung **cakupan rehearsal yang tidak lengkap** sebagai
faktor yang dapat dipulihkan pada lupa lintas-kurikulum, di bawah protokol yang diuji.

**Artifact:**
`artifacts/f15_cross_curriculum_recovery_v1/recovery/fp32/w64/selection_result.json`

## 26. Capacity experiment jika diperlukan

Bila 64×64 gagal, protokol mencoba 96, lalu 128, lalu 192 dengan dataset, teacher,
loss, optimizer logic, dan gate yang sama. Width terkecil yang lulus semua C0–C4,
fidelity, dan safety dipilih. Width lebih besar **tidak** dijalankan bila 64 sudah lulus.

**TIDAK DIJALANKAN.** Aturan beku `run_larger_width_only_if_64_fails = true` hanya
memicu width 96/128/192 apabila student FP32 64×64 gagal. Student itu justru lulus semua
gate, sehingga cabang ini tidak pernah aktif dan tidak ada model 96/128/192 yang dilatih.

Jawaban untuk pertanyaan "berapa actor terkecil yang mempertahankan C0–C4" karena itu
adalah **64×64 dengan 6.210 parameter** — ukuran yang sama persis dengan A7 historis yang
gagal, hanya dilatih dengan cakupan rehearsal yang benar.

Bukti dari sisi sebaliknya juga sudah ada di tahap lokalisasi: PD192 (43.202 parameter,
tujuh kali lebih besar) tetap mencatat completion 0.000 di C0–C2. Jadi kapasitas bukan
kendala yang mengikat di sini — **6.210 parameter cukup bila rehearsal-nya lengkap,
43.202 parameter tidak cukup bila tidak.**

Catatan penting: F15 **belum menguji** apakah width lebih besar dapat menghasilkan actor
INT8 yang lolos. Itu tetap terbuka, lihat seksi 36.

## 27. PTQ setelah recovery

Setelah FP32 lulus, PTQ memakai calibration set yang diseimbangkan dari C0+C1+C2+C3+C4,
bukan C4 saja. Weight FP32 recovered dibekukan. Perbandingan recovery-selection adalah
Recovered FP32 versus Recovered+PTQ pada seed yang sama.

**GAGAL.** PTQ memakai calibration set seimbang C0–C4 sebanyak 16.384 baris dari dataset
recovery yang sama (`recovery/ptq/w64/conversion.json`), dengan bobot FP32 hasil recovery
dibekukan. Actor INT8 hasilnya **tidak eligible**.

| Kurikulum | FP32 pulihan | + PTQ |
|---|---|---|
| C0 | PASS | PASS |
| C1 | PASS | PASS |
| C2 | PASS | PASS |
| C3 | **PASS** | **FAIL** |
| C4 | **PASS** | **FAIL** |

Completion C3 anjlok dari 1.000 ke **0.375**, mean progress dari 7.250 m ke **3.836 m**.
Completion C4 turun ke 0.875.

Yang menarik: cek fidelity yang jebol adalah **Pearson dan Spearman**, bukan MAE. omega
MAE C0 tercatat 0.16179 rad/s — masih di dalam gate 0.200 — sementara Pearson 0.97780
terhadap gate 0.980. Artinya kuantisasi tidak sekadar menambah error acak pada besaran
perintah setir, melainkan **merusak urutan relatifnya**. Untuk tugas yang menuntut
modulasi halus seperti melambat tepat di stop line lalu jalan lagi, itu fatal.

Pernyataan yang boleh dibuat: *"PTQ memperkenalkan kegagalan retention baru di C3 dan C4
pada prosedur kalibrasi dan width yang diuji."* Bukan: *"kuantisasi merusak retention."*

**Artifact:**
`artifacts/f15_cross_curriculum_recovery_v1/recovery/ptq/`

## 28. QAT setelah recovery

QAT+KD hanya dijalankan bila PTQ mengubah candidate lulus menjadi gagal pada behavior
atau fidelity. Teacher tetap Original dan rehearsal tetap C0–C4-balanced. Bila PTQ
telah lulus, QAT tidak dijalankan sekadar untuk mencari angka yang lebih cantik.

**DIJALANKAN karena PTQ gagal, dan hasilnya JUGA GAGAL.** QAT+KD multi-kurikulum memakai
teacher Original yang sama, rehearsal seimbang C0–C4 yang sama, fake quantization, dan
backend INT8 statis x86 yang sama (SHA256 `c943e34f…`).

QAT **memang lebih baik** dari PTQ pada fidelity:

| | PTQ | QAT+KD |
|---|---:|---:|
| omega MAE C3 | 0.08639 | **0.05796** |
| omega sign disagreement C3 | 0.00172 | **0.00000** |
| Pearson C0 | 0.97780 | **0.98959** |
| Kurikulum lulus fidelity | C2 saja | **C2 dan C3** |

Tetapi perilaku closed-loop tetap jebol:

- **C3**: completion 0.500, dan gate absolut `maximum_stop_violation_rate` **gagal** —
  actor melanggar stop. Ini regresi keselamatan, bukan sekadar penurunan performa.
- **C4**: completion 0.750, gagal pada `completion_rate`, `mean_progress_m`,
  `minimum_clearance`, `restart_rate`, dan `stop_violation_rate`.

Kedua jalur INT8 gagal di tempat yang sama: **C0–C2 selamat, C3–C4 hancur.** Itu justru
kebalikan dari kegagalan historis, dan yang rusak adalah kurikulum yang menuntut presisi.

Kesimpulan yang sah: pada width 64 dengan prosedur kuantisasi yang diuji, QAT+KD
memperbaiki fidelity tetapi **tidak memulihkan** perilaku C3/C4.

**Artifact bila dijalankan:**
`artifacts/f15_cross_curriculum_recovery_v1/recovery/qat/`

## 29. Progressive prune–distill jika diperlukan

Progressive order 256→192→128→96→64 hanya boleh diuji bila direct pruning target-width
ditambah multi-curriculum KD tidak cukup. Setiap tahap harus lulus gate sebelum lanjut.
Eksperimen ini tidak dijalankan bila recovery sederhana sudah memadai.

**TIDAK DIJALANKAN.** Aturan beku
`run_progressive_pruning_only_if_direct_recovery_is_insufficient = true` menyediakan
progressive prune–distill hanya bila recovery langsung pada target width gagal. Recovery
langsung 256→64 ditambah multi-curriculum KD **berhasil** di FP32.

Kegagalan yang tersisa berkaitan dengan **konversi INT8**, bukan dengan pruning.
Progressive pruning menyasar mekanisme yang berbeda, sehingga menjalankannya tidak akan
menjawab pertanyaan yang sedang terbuka.

Konsekuensinya: pertanyaan **"apakah urutan optimisasi penting?"** tetap **UNRESOLVED**
di F15. Tidak ada klaim apa pun yang dibuat tentang perbandingan progressive versus
direct pruning.

## 30. Final optimized actor

Sebelum holdout dibuka, satu checkpoint dibekukan dalam `final_candidate.json` lengkap
dengan SHA256, width, precision, dataset manifest, quantization method, dan rationale.
Tidak ada replacement setelah holdout.

**TIDAK ADA KANDIDAT FINAL YANG DIBEKUKAN.**

`freeze-candidate` mensyaratkan actor INT8 yang dapat di-deploy dengan `eligible: true`.
Tidak satu pun kandidat INT8 memenuhinya:

| Kandidat | Width | INT8 | Behavior lulus semua | Fidelity lulus semua | Eligible |
|---|---:|---|---|---|---|
| Recovered + Multi-Curriculum KD | 64 | tidak | **ya** | **ya** | **ya** |
| Recovered + PTQ | 64 | ya | tidak | tidak | tidak |
| Recovered + Multi-Curriculum QAT+KD | 64 | ya | tidak | tidak | tidak |

Satu-satunya actor yang eligible berformat FP32, sedangkan protokol beku mewajibkan
kandidat final berupa INT8. Mengganti model lain setelah melihat hasil ini **dilarang**
oleh protokol. Karena itu F15 berhenti di sini dan melaporkan apa adanya.

**Artifact:**

- `artifacts/f15_cross_curriculum_recovery_v1/recovery/recovery_decision.json`
  (`outcome: STOPPED_WITHOUT_ELIGIBLE_INT8_CANDIDATE`)
- `artifacts/f15_cross_curriculum_recovery_v1/recovery/recovery_experiments.csv`
- `artifacts/f15_cross_curriculum_recovery_v1/recovery/fp32/w64/actor_multicurriculum_kd_fp32.pt`
  (SHA256 `64c84cd0…`, satu-satunya actor yang lulus semua gate)

## 31. Final C0–C4 results

Final holdout memakai seed 180301–180308 yang tidak dipakai training atau selection.
Claim once-only ditulis sebelum environment final dibuka. Hanya Original dan kandidat
beku yang dievaluasi.

**FINAL HOLDOUT TIDAK DIBUKA.** Karena tidak ada kandidat yang bisa dibekukan, seed
**180301–180308 tetap tersegel** dan tidak pernah dievaluasi. Tidak ada
`final_holdout_claim.json` maupun `final_holdout.json`, dan itu memang seharusnya.

Ini bukan kelalaian melainkan konsekuensi protokol. Membuka holdout tanpa kandidat beku,
atau membukanya dengan model pengganti yang dipilih setelah melihat hasil, akan
menghanguskan nilai ilmiahnya secara permanen. Dengan tetap tersegel, seed itu masih
bisa dipakai untuk percobaan preregistered berikutnya.

Angka lintas-kurikulum terbaik yang tersedia berasal dari **seed selection 180201–180208**,
bukan holdout:

| Actor | C0 | C1 | C2 | C3 | C4 |
|---|---|---|---|---|---|
| Original Policy | REFERENCE | REFERENCE | REFERENCE | REFERENCE | REFERENCE |
| Recovered 64×64 KD (FP32) | PASS | PASS | PASS | PASS | PASS |
| Recovered + PTQ (INT8) | PASS | PASS | PASS | FAIL | FAIL |
| Recovered + QAT+KD (INT8) | PASS | PASS | PASS | FAIL | FAIL |

Angka ini **tidak boleh** dilaporkan seolah sudah lulus validasi akhir.

**Artifact:**

- `artifacts/f15_cross_curriculum_recovery_v1/recovery/selection_fp32_w64_results.json`
- `artifacts/f15_cross_curriculum_recovery_v1/recovery/selection_ptq_w64_results.json`
- `artifacts/f15_cross_curriculum_recovery_v1/recovery/selection_qat_w64_results.json`
- `artifacts/f15_cross_curriculum_recovery_v1/figures/08_final_cross_curriculum_performance.pdf`
  (keterbatasan ini tercetak di badan figure-nya)

## 32. Compression ratio

Original `29→256→256→2` memiliki **73.986** parameter. Semua actor pulihan
`29→64→64→2` memiliki **6.210** parameter.

- Pengurangan parameter: **91,61%**
- Faktor pengecilan: **11,9×**

Angka ini identik untuk ketiga kandidat (FP32, PTQ, QAT) karena arsitekturnya sama; yang
membedakan hanya presisi dan cara pelatihannya.

Parameter reduction berasal dari width actor, bukan penghapusan input semantik.

## 33. File size

| Actor | Ukuran file | Pengurangan vs Original |
|---|---:|---:|
| Original Policy | 299.667 B | — |
| **Recovered 64×64 KD (FP32)** | **29.295 B** | **−90,22%** |
| Recovered + PTQ (INT8) | 34.088 B | −88,62% |
| Recovered + QAT+KD (INT8) | 34.152 B | −88,60% |
| Final INT8 Policy historis (A7) | 36.880 B | −87,69% |

Perhatikan hal yang berlawanan dengan intuisi: checkpoint **FP32 justru paling kecil**
(29.295 B), lebih kecil daripada semua varian INT8. Penyebabnya adalah struktur
penyimpanan — checkpoint INT8 membawa parameter kuantisasi tambahan (scale dan zero-point
per kanal) serta pembungkus modul terkuantisasi. Karena itu ukuran file dan memori
parameter logis dilaporkan terpisah.

Serialized bytes dipengaruhi struktur checkpoint/JIT selain raw parameter memory; oleh
karena itu keduanya dilaporkan terpisah.

## 34. Actor latency

Benchmark memakai CPU, satu thread, batch 1, 1.000 warm-up, 10.000 timed iteration,
dan lima repeat. Median, P95, P99, serta throughput dilaporkan.

| Actor | Median latency | Speedup vs Original | Kurikulum lulus |
|---|---:|---:|---:|
| Original Policy | 40,428 µs | 1,00× | 5 |
| **Recovered 64×64 KD (FP32)** | **35,840 µs** | **1,13×** | **5** |
| Recovered + PTQ (INT8) | 15,984 µs | 2,53× | 3 |
| Recovered + QAT+KD (INT8) | 15,383 µs | 2,63× | 3 |
| Final INT8 Policy historis (A7) | 15,313 µs | 2,64× | 2 |

Inilah harga sebenarnya dari recovery, dan tidak boleh disamarkan. Satu-satunya actor
yang mempertahankan kelima kurikulum berformat FP32, sehingga ia mempertahankan
pengurangan parameter 91,61% dan file 90,22% tetapi **hanya memberi percepatan 1,13×**.

Percepatan 2,5–2,6× seluruhnya milik actor INT8 yang kehilangan C3/C4. Jadi F15 **tidak**
mencapai kompresi dan retention penuh sekaligus dalam bentuk INT8 yang dapat di-deploy;
yang tercapai adalah kompresi parameter dan ukuran file dengan retention penuh di FP32.

Peningkatan ini adalah **actor-only CPU speedup**, bukan RGB→perception→belief→action
speedup.

## 35. Apa yang berhasil dipertahankan?

Yang **berhasil** dipertahankan:

1. **Kompetensi C0–C4 penuh pada 6.210 parameter**, dalam FP32, lulus kelima gate
   retention dan kelima gate fidelity pada seed selection.
2. **Kompresi yang berarti**: 91,61% lebih sedikit parameter, 90,22% lebih kecil di disk.
3. **Fidelity aksi yang jauh lebih baik** dari endpoint historis: omega MAE C0
   0,03172 rad/s versus 0,26946 rad/s pada A7 — perbaikan 8,5×, murni dari cakupan data.
4. **Integritas historis**: seluruh artefak F10–F14 tidak tersentuh dan terverifikasi hash.
5. **Seed holdout tetap tersegel**, sehingga masih bernilai untuk percobaan berikutnya.

Yang **tidak** berhasil dipertahankan:

1. **Retention C3/C4 setelah konversi INT8**, baik lewat PTQ maupun QAT+KD.
2. **Percepatan besar bersamaan dengan retention penuh** — keduanya tidak diperoleh
   sekaligus.
3. **Bukti visual berbasis kamera** pada saat kegagalan, lihat seksi 21.

Kesimpulan hanya berlaku pada curriculum, gate, simulator, dan seed yang diuji.

## 36. Apa yang masih menjadi limitation?

1. **Rollout closed-loop tidak reproducible di runtime ini.** Perception F10 yang beku
   berjalan di kernel CUDA non-deterministik. Dari 150 sel `(model, kurikulum, seed)` yang
   terulang, 43 berbeda secara numerik dan **7 membalik label outcome objektif** —
   termasuk Original Policy sendiri di C2. Lantai noise praktisnya sekitar satu episode
   dari delapan, yaitu **0,125**, yang persis sebesar beberapa margin relatif beku.
   Temuan utama (0/8 versus 8/8) jauh di luar pita itu, tetapi sel yang berselisih satu
   atau dua episode ditandai **tidak konklusif** di
   `docs/F15_FAILURE_LOCALIZATION_REPORT.md`.
2. **Tidak ada final holdout.** Semua angka recovery berasal dari split selection.
   Actor pulihan belum pernah diuji pada seed yang belum dibuka.
3. **Tidak ada bukti visual berbasis kamera.** Telemetri tidak menyimpan RGB, dan replay
   aksi terekam ditolak oleh validasinya sendiri: lintasan tereproduksi sangat baik
   (2,96e-08 m) tetapi terminasi episode tidak, dengan kegagalan pertama meleset 144 dan
   171 step. Media itu dikarantina di `failure_traces/*/unresolved/`.
4. **Width untuk endpoint INT8 belum dicari.** Aturan beku hanya memicu width lebih besar
   bila FP32 64×64 gagal, dan itu tidak terjadi. Apakah 96, 128, atau 192 menghasilkan
   actor INT8 yang lolos **belum diuji**.
5. **Urutan optimisasi belum terjawab.** Progressive prune–distill tidak dijalankan.
6. **Kelemahan baseline yang diwarisi.** `docs/F10_PPO_CURRICULUM.md` mencatat retention
   C1 pada checkpoint C4 sudah terbatas di 25% completion sejak F10, dan C3 lolos memakai
   jaringan hasil distilasi DAgger, bukan PPO reward-only. Original Policy karena itu
   tidak seragam kuat di C0–C4.
7. **Delapan seed per sel, satu blok seed per tahap.** Kesimpulan terikat pada protokol,
   kurikulum, dan simulator ini.

Perbandingan branch melokalisasi association pada prosedur yang diuji, tetapi tidak
membuktikan neuron tertentu menyebabkan failure. Same-seed rollout juga bukan causal
paired trajectory bila equivalence seluruh noise eksogen tidak dapat dibuktikan.

## 37. Apa yang boleh diklaim?

- “Pruning adalah tahap pertama tempat Cx berubah dari PASS menjadi FAIL” bila matrix
  F15 menunjukkan transisi tersebut.
- “Multi-curriculum KD memulihkan retention di bawah protokol yang diuji” bila selection
  dan holdout mendukung.
- “PTQ memperkenalkan PASS→FAIL di Cx pada calibration procedure ini” bila benar.
- “Final recovered actor lulus C0–C4 holdout” hanya bila kelima gate benar-benar lulus.

## 38. Apa yang tidak boleh diklaim?

- “Neuron yang dibuang tidak berguna.”
- “Quantization secara universal menyebabkan catastrophic forgetting.”
- “KD menjamin generalisasi.”
- “64×64 selalu cukup.”
- “C4 success membuktikan kesiapan universal.”
- “Same-seed paired rollout membuktikan trajectory counterfactual kausal.”
- “Actor-only latency adalah end-to-end system latency.”

## 39. Daftar semua artifact

Artifact utama F15:

- `artifacts/f15_cross_curriculum_recovery_v1/model_registry.json`
- `artifacts/f15_cross_curriculum_recovery_v1/protocol_manifest.json`
- `artifacts/f15_cross_curriculum_recovery_v1/seed_manifest.json`
- `artifacts/f15_cross_curriculum_recovery_v1/dataset_manifest.json`
- `artifacts/f15_cross_curriculum_recovery_v1/localization/cross_curriculum_results.csv`
- `artifacts/f15_cross_curriculum_recovery_v1/localization/open_loop_fidelity_by_curriculum.csv`
- `artifacts/f15_cross_curriculum_recovery_v1/localization/pruning_width_retention.csv`
- `artifacts/f15_cross_curriculum_recovery_v1/localization/failure_event_registry.csv`
- `artifacts/f15_cross_curriculum_recovery_v1/localization/failure_localization_decision.json`
- `artifacts/f15_cross_curriculum_recovery_v1/recovery/recovery_experiments.csv`
- `artifacts/f15_cross_curriculum_recovery_v1/recovery/recovery_decision.json`
- `artifacts/f15_cross_curriculum_recovery_v1/final/efficiency_summary.json`

Dua artifact yang **sengaja tidak diproduksi**, karena tidak ada kandidat INT8 yang
eligible sehingga tidak ada yang boleh dibekukan maupun diuji pada holdout:

- `final/final_candidate.json` — tidak dibuat
- `final/final_holdout_claim.json` dan `final/final_holdout.json` — tidak dibuat, seed
  180301–180308 tetap tersegel
- `artifacts/f15_cross_curriculum_recovery_v1/artifact_manifest.json`

## 40. Peta file repository

| Tahap | Tujuan | Input | Output | Config | Script/source | Artifact/report |
|---|---|---|---|---|---|---|
| Original F10 | policy beku | public 29D | action | `configs/f10_ppo_visual_objects_v30.toml` | `src/duckie_pomdp/control/ppo_protocol.py` | `artifacts/f10_ppo_visual_objects_v30/c4/ppo_selected.pt` |
| Historical compression F12 | ablation pruning/KD/PTQ/QAT | Original + C4 states | A0–A7 | `configs/f12_belief_ppo_compression_v1.toml` | `experiments/run_f12_compression.py` | `docs/F12_COMPRESSION_RESULTS.md` |
| F15 audit/protocol | freeze provenance/gate/seed | historical registry | manifests | `configs/f15_cross_curriculum_recovery_v1.toml` | `experiments/run_f15_cross_curriculum_recovery.py` | `docs/F15_PROTOCOL.md` |
| F15 localization | cari first collapse | A0–A7/frontier + seed 180001–8 | competence/fidelity/failure | sama | `experiments/run_f15_cross_curriculum_recovery.py` | `docs/F15_FAILURE_LOCALIZATION_REPORT.md` |
| F15 failure visuals | bukti objektif | frozen event registry | MP4/GIF/PNG | sama | `experiments/render_f15_failure_traces.py` | `artifacts/f15_cross_curriculum_recovery_v1/failure_traces/` |
| F15 multi-curriculum KD | recovery coverage | C0–C4 public 29D + A0 target | recovered FP32 | sama | `experiments/run_f15_recovery.py` | `docs/F15_RECOVERY_REPORT.md` |
| F15 PTQ/QAT | deployable recovery | recovered FP32 | INT8 candidate | sama | `experiments/run_f15_recovery.py` | `artifacts/f15_cross_curriculum_recovery_v1/recovery/` |
| F15 final | once-only C0–C4 validation | Original + frozen candidate | final classification | sama | `experiments/run_f15_recovery.py` | `docs/F15_FINAL_REPORT.md` |
| F15 figures | komunikasi hasil | frozen CSV/JSON | PNG/PDF | — | `experiments/generate_f15_figures.py` | `artifacts/f15_cross_curriculum_recovery_v1/figures/` |
| F15 verification | hash/manifests/tests | seluruh artifact | PASS/FAIL verifier | sama | `experiments/verify_f15_artifacts.py` | `artifacts/f15_cross_curriculum_recovery_v1/artifact_manifest.json` |

Panduan ini harus dibaca bersama report F15 final. Bila suatu experiment tidak
dijalankan karena stop rule, status yang benar adalah **NOT TESTED**, bukan nol atau
PASS.
