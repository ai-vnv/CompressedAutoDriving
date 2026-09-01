# Panduan Proyek Duckietown POMDP dari Nol

Dokumen ini adalah peta belajar dan peta audit proyek. Tujuannya bukan menggantikan kode, melainkan membuat pembaca dapat mengikuti aliran **input → proses → output** tanpa harus membuka Python terlebih dahulu. Istilah Inggris dipertahankan ketika istilah tersebut juga dipakai pada figure dan paper.

> **Batas ilmiah utama:** seluruh kompetensi yang dibahas adalah kompetensi di simulator. PPO tidak menerima RGB, bounding box, pose dunia, atau *Bird's-Eye View* (BEV) secara langsung. PPO menerima vektor semantik publik 29 dimensi yang dibentuk sebelum *privileged simulator state* dibaca untuk evaluasi.

## 1. Gambaran Besar

Proyek ini melatih Duckiebot virtual untuk mengikuti lane, merespons Duckie yang menyeberang, berhenti pada stop sign, lalu melanjutkan perjalanan. Masalahnya adalah robot tidak boleh memakai “jawaban simulator”, misalnya posisi dunia objek yang tepat. Kamera hanya memberi gambar, detector dapat keliru, objek dapat tertutup, dan keadaan berubah dari waktu ke waktu. Karena itu keadaan dunia hanya **sebagian teramati** (*partially observable*).

POMDP (*Partially Observable Markov Decision Process*) adalah kerangka untuk mengambil keputusan ketika state sebenarnya tidak tersedia secara lengkap. Proyek ini mengubah sensor menjadi *measurement*, menggabungkan measurement sepanjang waktu menjadi *belief*, menyusunnya sebagai input 29D, kemudian actor PPO menghasilkan action fisik.

```text
State dunia yang tidak sepenuhnya diketahui
→ sensor
→ measurement
→ belief
→ representasi semantik policy 29D
→ PPO actor
→ action
```

**Rujukan proyek:**

- `FORMULATION.md`
- `GATES.md`
- `src/duckie_pomdp/domain/state.py`
- `src/duckie_pomdp/control/ppo_observation.py`
- `docs/F11_R001_OBSERVATION_CONTRACT_REPORT.md`

## 2. Apa yang Ada di Simulator?

Lingkungan aktif memakai Gym-Duckietown 6.2.0. Di dalamnya ada Duckiebot, lane, map, kamera depan, stop sign, serta Duckie yang dapat menyeberang. Skenario curriculum memakai `small_loop`, `experiment_loop`, dan pada C3/C4 memakai variasi map/scenario stop dan gabungan yang dibekukan di config masing-masing.

Ada dua lapisan informasi yang harus dibedakan:

| Lapisan | Contoh | Boleh masuk PPO? |
|---|---|---:|
| True simulator state | pose dunia robot/objek, collision truth, posisi objek pasti | Tidak |
| Public policy information | hasil lane vision, belief pedestrian/stop, ego motion terukur, previous action | Ya |

True state boleh dibaca **setelah action dibuat** untuk reward, metric, atau gambar evaluasi. Ia tidak boleh menyusup ke observation policy.

**Input:** frame RGB depan dan ego motion terukur.  
**Proses:** simulator → perception → belief → policy.  
**Output:** gerak Duckiebot dan metric evaluasi.

**Rujukan proyek:**

- `src/duckie_pomdp/adapters/gym_duckietown.py`
- `src/duckie_pomdp/domain/state.py`
- `src/duckie_pomdp/domain/observation.py`
- `src/duckie_pomdp/domain/privileged.py`
- `configs/f10_ppo_visual_objects_v30.toml`

## 3. POMDP dalam Proyek Ini

Notasi dasar dapat dipetakan ke implementasi sebagai berikut:

| Simbol | Arti umum | Implementasi proyek |
|---|---|---|
| \(s_t\) | state sebenarnya/latent | state dunia simulator; disembunyikan dari policy |
| \(o_t\) | observation mentah | RGB depan dan ego motion yang tersedia |
| \(z_t\) | measurement | lane pose, deteksi/proyeksi range-bearing |
| \(b_t\) | belief | estimasi lane, pedestrian, stop beserta ketidakpastian |
| \(a_t\) | action | \([v_{cmd},\omega_{cmd}]\) |

Pemetaan implementasinya:

```text
s_t
→ camera/ego observation
→ MobileNet/YOLO measurement
→ EKF dan belief logic
→ representasi policy 29D
→ PPO
→ [v_cmd, omega_cmd]
```

MobileNet dan YOLO adalah fungsi perception yang **diimplementasikan**, bukan nama formal untuk observation kernel POMDP. Hasilnya masih measurement; belief baru muncul setelah measurement digabungkan dengan prediksi waktu dan ketidakpastian.

**Rujukan proyek:**

- `FORMULATION.md`
- `src/duckie_pomdp/domain/observation.py`
- `src/duckie_pomdp/domain/belief.py`
- `src/duckie_pomdp/control/ppo_environment.py`

## 4. Input Mentah: Kamera dan Ego Motion

### Front RGB

**Tujuan:** memberi bukti visual untuk lane, Duckie, dan stop sign.  
**Input:** citra RGB dari kamera depan simulator.  
**Diproses bagaimana:** frame yang sama masuk ke model lane MobileNetV3-small dan detector YOLO11n.  
**Output:** lane measurement serta deteksi objek.  
**Masuk ke mana:** EKF/belief updater, bukan langsung ke PPO.  
**Unit:** nilai piksel; bukan satuan fisik.

### Ego motion

**Tujuan:** memberi konteks gerak aktual.  
**Input:** actual linear velocity dalam m/s dan actual yaw rate dalam rad/s.  
**Diproses bagaimana:** dimasukkan sebagai dua field publik dan juga membantu prediksi belief.  
**Output:** dua dimensi kelompok `Ego`.  
**Perbedaan penting:** actual velocity adalah gerak yang terukur; previous action adalah command yang dikirim sebelumnya. Keduanya tidak sama secara konseptual.

**Rujukan proyek:**

- `src/duckie_pomdp/adapters/gym_duckietown.py`
- `src/duckie_pomdp/control/ppo_environment.py`
- `src/duckie_pomdp/control/ppo_observation.py`

## 5. Perception untuk Lane

```text
RGB → MobileNetV3-small → lane measurement → lane EKF → LaneBelief
```

**Tujuan:** memperkirakan hubungan robot terhadap lane tanpa membaca pose lane dari simulator.  
**Input:** frame RGB depan.  
**Diproses bagaimana:** jaringan `mobilenet_v3_small_lane_pose_v1` menghasilkan measurement lane; runtime lane belief kemudian mempertahankan mean, standard deviation, dan validity.  
**Output:** lateral error, heading error, curvature, ketidakpastian masing-masing, dan validity probability.  
**Masuk ke mana:** tujuh field kelompok `Lane` pada 29D.

Checkpoint lane aktif adalah `artifacts/f10_ppo_visual_v9/lane_rgb_model/best.pt`; model ini tidak diubah oleh F11–F14.

**Rujukan proyek:**

- `src/duckie_pomdp/perception/lane_rgb_model.py`
- `src/duckie_pomdp/control/lane_belief_runtime.py`
- `configs/lane_belief_v8_competence_rgb.toml`
- `artifacts/f10_ppo_visual_v9/lane_rgb_model/best.pt`

## 6. Perception untuk Pedestrian dan Stop Sign

```text
RGB → YOLO11n → bounding box → bottom-center projection → range/bearing measurement
```

YOLO memakai kelas domain `DUCKIE` untuk pedestrian dan `STOP_SIGN` untuk rambu. Titik bawah-tengah bounding box diproyeksikan dengan model kamera menjadi measurement geometri.

### Jalur pedestrian

**Input:** deteksi Duckie.  
**Proses:** confidence gate → proyeksi range/bearing → F9c EKF dan existence filter.  
**Output:** probabilitas keberadaan, mean/std range, bearing, radial velocity, dan bearing rate.  
**Masuk ke mana:** sembilan field kelompok `Pedestrian`.

### Jalur stop sign

**Input:** deteksi stop sign.  
**Proses:** proyeksi dan stop-belief/state logic, dipadukan dengan route stop-line observer.  
**Output:** belief tanda, jarak stop line, dan mode `NONE/REQUIRED/SATISFIED`.  
**Masuk ke mana:** kelompok `StopLine` dan `Stop`.

Checkpoint detector aktif adalah `artifacts/yolo_v1/best.pt`; F12 hanya mengompresi actor PPO, bukan YOLO.

**Rujukan proyek:**

- `src/duckie_pomdp/perception/yolo_measurement.py`
- `src/duckie_pomdp/perception/camera_geometry.py`
- `src/duckie_pomdp/belief/pedestrian_ekf.py`
- `src/duckie_pomdp/control/stop_belief.py`
- `artifacts/yolo_v1/best.pt`

## 7. Apa Itu Measurement?

Measurement adalah satu hasil pengukuran saat ini, bukan kesimpulan akhir tentang dunia. Misalnya YOLO dan proyeksi mengatakan:

> “Pada frame ini, Duckie tampak kira-kira 1,2 m dari robot pada sudut tertentu.”

Nilai itu dapat bising. Bounding box dapat bergeser atau deteksi dapat hilang. Belief updater kemudian menggabungkan:

```text
perkiraan sebelumnya
+ prediksi akibat gerak
+ measurement baru
+ ketidakpastian
→ belief baru
```

**Input:** measurement satu frame.  
**Output:** estimasi temporal yang lebih stabil beserta uncertainty.

**Rujukan proyek:**

- `src/duckie_pomdp/domain/measurement.py`
- `src/duckie_pomdp/evaluation/yolo_measurement.py`
- `artifacts/yolo_measurement_metrics.json`
- `artifacts/yolo_measurement_noise_v1.json`

## 8. Apa Itu EKF dan Belief?

EKF (*Extended Kalman Filter*) dapat dianalogikan dengan penilaian klinis serial:

- **Measurement** = hasil pemeriksaan hari ini.
- **Belief** = perkiraan kondisi setelah mempertimbangkan riwayat, dinamika, hasil baru, dan ketidakpastian.

Belief menyimpan **mean** sebagai estimasi pusat, **standard deviation** sebagai ketidakpastian, dan—untuk pedestrian/stop sign—**existence probability** sebagai keyakinan bahwa objek benar-benar ada. Siklusnya terdiri dari prediction lalu correction.

### Lane belief

Menjaga estimasi lateral error, heading error, curvature, uncertainty, dan validity dari keluaran vision sepanjang waktu.

### Pedestrian belief

Menjaga range, bearing, radial velocity, bearing rate, uncertainty, dan probability of existence. Ketika existence di bawah gate publik yang dibekukan, runtime memakai tuple semantik netral; ia tidak mengartikan semua angka nol sebagai “objek di posisi nol”.

### Perjalanan F9 yang dapat direkonstruksi

| Tahap | Tujuan | Masalah/input | Perubahan/proses | Output dan hasil |
|---|---|---|---|---|
| F9a | Kalibrasi measurement YOLO | bbox dan geometri kamera | estimasi noise serta validasi range/bearing | measurement contract dan artifact kalibrasi tersedia |
| F9b | Hubungkan YOLO ke pedestrian EKF | measurement temporal | prediksi/correction belief | `LIMITED`: overconfidence masih ditemukan |
| F9c | Membuat belief lebih robust | outlier dan coverage F9b | robust updater + existence handling | akurasi/retention membaik, tetapi gate coverage preregistered tetap `LIMITED` |
| F9d | Menutup evidence robustness | absence/yield/outlier cases | audit bukti tambahan | `LIMITED`: dukungan frame outlier minimum tidak terpenuhi, walau absence/yield pass |

Status `LIMITED` tidak boleh diubah menjadi `PASS` hanya karena komponen tersebut kemudian berguna pada curriculum PPO.

**Rujukan proyek:**

- `src/duckie_pomdp/belief/pedestrian_ekf.py`
- `src/duckie_pomdp/belief/robust_updater.py`
- `configs/f9c_robust_belief_v1.toml`
- `docs/superpowers/F9C_REPORT_FOR_REVIEW.md`
- `docs/superpowers/F9D_REPORT_FOR_REVIEW.md`

## 9. Stop Belief dan Stop Mode

Stop handling memakai dua sumber publik: belief rambu dari YOLO/proyeksi dan jarak stop line berdasarkan route observer. Mode stop adalah one-hot:

- `NONE`: tidak ada kewajiban stop aktif;
- `REQUIRED`: policy sedang wajib berhenti;
- `SATISFIED`: stop sudah dipenuhi dan robot boleh restart sesuai state logic.

Phase `stop_satisfied` hanya diberikan ketika bit `SATISFIED` aktif dan jarak absolut ke stop line tidak lebih dari 0,5 m. `nominal` adalah kategori fallback; karena itu field historis tertentu dapat tetap bernilai walau phase utama bukan lagi `stop_required`.

**Input:** belief stop sign + route stop-line context.  
**Output:** satu jarak stop line, lima field belief stop, dan tiga bit mode.  
**Masuk ke mana:** 29D policy input.

**Rujukan proyek:**

- `src/duckie_pomdp/control/stop_belief.py`
- `src/duckie_pomdp/explain/development_protocol.py`
- `configs/f14_explainability_aware_compression_v1.toml`

## 10. Bagaimana Semua Informasi Menjadi 29D?

Nama dan urutan berikut adalah kontrak runtime. Normalisasi adalah `clip(value / scale, -3, 3)`.

| No. | Nama field | Kelompok | Arti sederhana | Sumber | Unit | Scale | Jenis |
|---:|---|---|---|---|---|---:|---|
| 1 | `lane_validity_probability` | Lane | keyakinan lane valid | lane belief | probabilitas | 1 | belief |
| 2 | `lane_lateral_error_mean_m` | Lane | posisi lateral terhadap lane | lane belief | m | 0.25 | belief mean |
| 3 | `lane_lateral_error_std_m` | Lane | ketidakpastian lateral | lane belief | m | 0.25 | belief std |
| 4 | `lane_heading_error_mean_rad` | Lane | selisih arah terhadap lane | lane belief | rad | 0.75 | belief mean |
| 5 | `lane_heading_error_std_rad` | Lane | ketidakpastian arah | lane belief | rad | 0.75 | belief std |
| 6 | `actual_linear_velocity_mps` | Ego | kecepatan aktual | ego motion | m/s | 0.4 | direct/context |
| 7 | `actual_yaw_rate_rad_s` | Ego | laju putar aktual | ego motion | rad/s | 4 | direct/context |
| 8 | `lane_curvature_mean_inv_m` | Lane | kelengkungan lane | lane belief | 1/m | 5 | belief mean |
| 9 | `lane_curvature_std_inv_m` | Lane | ketidakpastian kelengkungan | lane belief | 1/m | 5 | belief std |
| 10 | `stop_line_distance_m` | StopLine | jarak bertanda ke stop line | route observer | m | 2 | public context |
| 11 | `pedestrian_existence_probability` | Pedestrian | peluang Duckie ada | existence filter | probabilitas | 1 | belief |
| 12 | `pedestrian_range_mean_m` | Pedestrian | jarak Duckie | pedestrian EKF | m | 2 | belief mean |
| 13 | `pedestrian_range_std_m` | Pedestrian | ketidakpastian jarak | pedestrian EKF | m | 2 | belief std |
| 14 | `pedestrian_bearing_mean_rad` | Pedestrian | arah relatif Duckie | pedestrian EKF | rad | 1.2 | belief mean |
| 15 | `pedestrian_bearing_std_rad` | Pedestrian | ketidakpastian arah | pedestrian EKF | rad | π | belief std |
| 16 | `pedestrian_radial_velocity_mean_mps` | Pedestrian | gerak mendekat/menjauh | pedestrian EKF | m/s | 1 | belief mean |
| 17 | `pedestrian_radial_velocity_std_mps` | Pedestrian | ketidakpastian radial velocity | pedestrian EKF | m/s | 1 | belief std |
| 18 | `pedestrian_bearing_rate_mean_rad_s` | Pedestrian | perubahan arah relatif | pedestrian EKF | rad/s | 4 | belief mean |
| 19 | `pedestrian_bearing_rate_std_rad_s` | Pedestrian | ketidakpastian bearing rate | pedestrian EKF | rad/s | π | belief std |
| 20 | `stop_sign_existence_probability` | Stop | peluang rambu ada | stop belief | probabilitas | 1 | belief |
| 21 | `stop_sign_range_mean_m` | Stop | jarak rambu | stop belief | m | 2 | belief mean |
| 22 | `stop_sign_range_std_m` | Stop | ketidakpastian jarak | stop belief | m | 2 | belief std |
| 23 | `stop_sign_bearing_mean_rad` | Stop | arah relatif rambu | stop belief | rad | 1.2 | belief mean |
| 24 | `stop_sign_bearing_std_rad` | Stop | ketidakpastian arah rambu | stop belief | rad | π | belief std |
| 25 | `stop_mode_none` | Stop | mode tidak wajib stop | stop state machine | one-hot | 1 | public state |
| 26 | `stop_mode_required` | Stop | mode wajib stop | stop state machine | one-hot | 1 | public state |
| 27 | `stop_mode_satisfied` | Stop | kewajiban stop terpenuhi | stop state machine | one-hot | 1 | public state |
| 28 | `previous_linear_velocity_cmd_mps` | PreviousAction | command maju sebelumnya | previous policy action | m/s | 0.4 | context |
| 29 | `previous_angular_velocity_cmd_rad_s` | PreviousAction | command belok sebelumnya | previous policy action | rad/s | 4 | context |

Enam kelompok membagi semua dimensi tepat sekali: Lane 7, Ego 2, StopLine 1, Pedestrian 9, Stop 8, PreviousAction 2.

29D **bukan** RGB, bukan full simulator state, dan bukan pure probability distribution. Istilah yang tepat adalah **representasi semantik policy 29 dimensi yang dikondisikan oleh belief**.

**Rujukan proyek:**

- `src/duckie_pomdp/control/ppo_observation.py`
- `configs/f10_ppo_visual_objects_v30.toml`
- `artifacts/f11_ppo_explanation_v2/r001/contract_audit.json`
- `artifacts/f14_explainability_aware_compression_v1/integrity/actor_registry_verified.json`

## 11. Apa yang Masuk ke PPO?

**Input:** vektor normalized 29D.  
**Model:** actor MLP (*multilayer perceptron*) dengan dua hidden layer. Original memakai `29 → 256 → 256 → 2` dan aktivasi `tanh`.  
**Output:** deterministic actor mean yang dipetakan ke `v_cmd` dan `omega_cmd` dalam satuan fisik.

- `v_cmd`: perintah kecepatan maju, rentang fisik 0–0,4 m/s.
- `omega_cmd`: perintah kecepatan sudut/belok, rentang fisik −4–4 rad/s.

Previous command adalah output policy pada langkah sebelumnya; actual measured velocity adalah respons gerak yang benar-benar teramati. PPO tidak boleh menganggap keduanya identik.

**Rujukan proyek:**

- `src/duckie_pomdp/control/ppo.py`
- `src/duckie_pomdp/control/ppo_observation.py`
- `src/duckie_pomdp/domain/action.py`
- `artifacts/f10_ppo_visual_objects_v30/c4/ppo_selected.pt`

## 12. Apa Itu PPO?

PPO (*Proximal Policy Optimization*) adalah algoritme reinforcement learning. Secara sederhana:

- **policy**: aturan terpelajar untuk memilih action;
- **actor**: jaringan yang menghasilkan distribusi/action;
- **critic**: jaringan yang memperkirakan nilai state untuk membantu training;
- **reward**: sinyal latihan tentang kemajuan, keselamatan, lane, stop, dan tujuan;
- **episode**: satu rangkaian interaksi dari reset hingga selesai/gagal;
- **training**: bobot diperbarui menggunakan rollout;
- **inference/deployment**: bobot dibekukan dan actor menghasilkan action.

Actor dan critic dipakai saat PPO training, tetapi F12 mengompresi **actor saja** untuk deployment. Critic bukan bagian actor-only INT8 yang diukur.

**Rujukan proyek:**

- `src/duckie_pomdp/control/ppo.py`
- `experiments/train_f10_ppo.py`
- `configs/f10_ppo_visual_objects_v30.toml`
- `docs/F10_PPO_CURRICULUM.md`

## 13. Curriculum C0–C4

Curriculum menambah kesulitan secara bertahap. Definisi berikut direkonstruksi dari config dan laporan aktif, bukan dari asumsi umum.

| Tahap | Tujuan dan skenario | Kapabilitas baru | Status/bukti utama |
|---|---|---|---|
| C0 | basic driving pada `small_loop`, tanpa pedestrian/stop | bergerak dan mengikuti lane dasar | predecessor curriculum artifact |
| C1 | generalisasi pada `experiment_loop`, tanpa pedestrian/stop | menyelesaikan loop baru | frozen predecessor checkpoint |
| C2 | `experiment_loop` dengan Duckie crossing LTR/RTL | respons pedestrian dari YOLO→belief | selected C2 predecessor digunakan C3 |
| C3 | stop scenario, pedestrian tidak aktif | stop, hold, restart | selected C3 checkpoint menjadi sumber C4 |
| C4 | combined pedestrian + stop, LTR/RTL | gabungan lane, crossing, stop, restart | Original Belief-PPO final; designated deployment scope |

Input policy tetap 29D di semua tahap; scenario menentukan semantic evidence yang aktif. C4 disebut designated deployment scenario karena seleksi F12 secara eksplisit membekukan scope tersebut, bukan karena C4 membuktikan generalisasi universal.

**Rujukan proyek:**

- `docs/F10_PPO_CURRICULUM.md`
- `configs/f10_ppo_visual_objects_v30.toml`
- `experiments/train_f10_ppo.py`
- `artifacts/f10_ppo_visual_objects_v30/c4/ppo_selected.pt`

## 14. Original Belief-PPO

Original Belief-PPO adalah checkpoint C4 sebelum compression:

- architecture actor: `29 → 256 → 256 → 2`, FP32;
- checkpoint: `artifacts/f10_ppo_visual_objects_v30/c4/ppo_selected.pt`;
- SHA256: `02e898ce12d71f97016d50ed8a40574807e6d2fd995fc9f0dcd24f357f2c6250`;
- output: physical `v_cmd` dan `omega_cmd`;
- C4 combined behavior menjadi reference untuk F12–F14.

Checkpoint PPO penuh berisi state training/critic; F12 mengekstrak actor-only representation untuk perbandingan deployment. Keduanya tidak boleh disamakan berdasarkan ukuran file saja.

Known retention limitation tetap dilaporkan: competence terpilih adalah C4-only, sedangkan hasil retention luar scope tidak semuanya dipertahankan.

**Rujukan proyek:**

- `artifacts/f10_ppo_visual_objects_v30/c4/ppo_selected.pt`
- `docs/F12_COMPRESSION_PROTOCOL.md`
- `artifacts/f12_belief_ppo_compression_v1/final/model_selection.json`

## 15. Explanation / Explainable RL

Empat pertanyaan ini harus selalu dipisahkan:

| Konsep | Pertanyaan | Input eksperimen | Yang boleh disimpulkan |
|---|---|---|---|
| Attribution | Informasi mana yang relatif berkontribusi? | state yang sama + reference | struktur kontribusi actor relatif terhadap metode/reference |
| Counterfactual sensitivity | Jika satu konsep input diubah, bagaimana action berubah? | semantic intervention terkontrol | ketergantungan fungsional pada input policy |
| Action fidelity | Apakah output optimized actor dekat dengan Original? | input 29D yang sama pada dua actor | kesamaan numerik action |
| Closed-loop behavior | Apakah robot tetap menyelesaikan task? | policy berinteraksi berulang dengan environment | performa dinamis pada skenario yang diuji |

Attribution dapat berubah sementara behavior tetap lulus. Sebaliknya, attribution stabil tidak menjamin closed-loop aman.

**Rujukan proyek:**

- `docs/F11_FINAL_EXPLANATION_SUMMARY.md`
- `docs/F14_PROTOCOL.md`
- `docs/F14_FINAL_REPORT.md`

## 16. Group Shapley

F14 memakai **phase-conditioned exact Group Shapley** agar FP32 dan static INT8 dapat dibandingkan dengan forward inference yang sama, tanpa gradient surrogate.

Enam “pemain” adalah Lane, Ego, StopLine, Pedestrian, Stop, dan PreviousAction. Enam grup menghasilkan \(2^6=64\) kemungkinan coalition. Untuk setiap coalition, grup tertentu berasal dari factual row, sedangkan semua grup yang tidak hadir berasal dari **satu complete same-phase reference row**. Ini mencegah pencampuran tiap grup dari reference berbeda.

Contoh: untuk coalition `{Lane, Pedestrian}`, nilai Lane dan Pedestrian berasal dari factual state; Ego, StopLine, Stop, dan PreviousAction berasal dari satu reference row.

`Mean absolute attribution share = 0.70` berarti 70% dari jumlah absolut attribution pada konstruksi coalition/reference tersebut berada pada grup itu. Ia **tidak** berarti 70% sebab fisik perilaku dunia.

**Rujukan proyek:**

- `src/duckie_pomdp/explain/group_shapley.py`
- `src/duckie_pomdp/explain/group_shapley.py`
- `docs/F14_REFERENCE_CALIBRATION.md`
- `artifacts/f14_explainability_aware_compression_v1/coalition_schema.json`

## 17. Driving Phases

Phase ditentukan hanya dari field publik dengan prioritas yang dibekukan.

| Phase | Arti sederhana | Rule publik utama | Mengapa dipisahkan |
|---|---|---|---|
| Nominal | berkendara biasa | fallback ketika phase lain tidak aktif | baseline lane-following |
| Lane curve | tikungan | `abs(curvature mean) ≥ 1.5 1/m` | steering biasanya lebih relevan |
| Pedestrian relevant | pedestrian dekat dan cukup dipercaya | existence ≥ 0.4 dan range ≤ 1.2 m | mengisolasi konteks Duckie aktif |
| Stop required | kewajiban berhenti aktif | one-hot `stop_mode_required > 0.5` | menguji braking/stop semantics |
| Stop satisfied | stop telah dipenuhi dekat garis | satisfied aktif dan `abs(stop_line_distance) ≤ 0.5 m` | menguji restart/post-stop |

Kombinasi pedestrian+stop memiliki prioritas internal tersendiri, tetapi lima phase di atas adalah taxonomy utama figure F11/F14.

**Rujukan proyek:**

- `src/duckie_pomdp/explain/development_protocol.py`
- `configs/f14_explainability_aware_compression_v1.toml`
- `artifacts/f14_explainability_aware_compression_v1/diagnostic_state_manifest.json`

## 18. Counterfactual Tests

Semantic counterfactual mengubah satu konsep pada 29D memakai tuple valid yang sudah dibekukan:

- `pedestrian_absent`: ganti seluruh sembilan field pedestrian dengan neutral absence;
- `stop_absent`: ganti stop-line/stop tuple sesuai operator valid;
- `lane_centered`: buat lane center yang valid;
- `lane_low_confidence`: turunkan validity dan naikkan uncertainty sesuai batas frozen;
- `previous_action_neutral`: netralkan dua command sebelumnya;
- `sham`: identity control; harus menghasilkan delta nol dalam tolerance device.

### Tiga uji primary

1. pedestrian removed pada `pedestrian_relevant` → respons `v_cmd`;
2. stop requirement removed pada `stop_required` → respons `v_cmd`;
3. lane centered pada `lane_curve` → respons `omega_cmd`.

Klasifikasi utama selalu `x/3`. Angka historis `3/8` pada laporan failure-mode awal berasal dari agregasi attempt pertama yang memperlakukan dua output untuk beberapa operator sebagai delapan cell. Amendment protocol menyatakan denominator itu tidak sesuai primary protocol. Karena itu `Pruning + PTQ` yang benar adalah **1/3 primary tests preserved**; artifact attempt lama dipertahankan hanya sebagai audit trail, bukan sebagai hasil utama.

**Rujukan proyek:**

- `src/duckie_pomdp/explain/development_protocol.py`
- `src/duckie_pomdp/explain/compression_diagnostics.py`
- `artifacts/f14_explainability_aware_compression_v1/protocol_alignment_amendment.json`
- `artifacts/f14_explainability_aware_compression_v1/ablation_comparison_metrics.json`

## 19. Mengapa Model Dioptimisasi?

Actor original cukup kecil, tetapi deployment research tetap perlu mengukur berapa banyak capacity yang dapat dihapus tanpa merusak policy. F12 menguji parameter count, actor bytes, memory, action fidelity, latency CPU actor-only, dan closed-loop C4.

Perception tidak dikompresi. Karena itu percepatan actor **bukan** percepatan penuh `RGB → perception → belief → action`. MobileNet dan YOLO dapat tetap menjadi bottleneck end-to-end.

**Rujukan proyek:**

- `docs/F12_COMPRESSION_PROTOCOL.md`
- `docs/F12_COMPRESSION_RESULTS.md`
- `artifacts/f12_belief_ppo_compression_v1/benchmarks/actor_benchmarks.json`

## 20. Pruning

Structured neuron pruning membuang **seluruh hidden neuron** dan membentuk jaringan dense yang benar-benar lebih kecil. Input 29D tidak dipangkas; semantic group tidak dihapus. F12 membandingkan width 192, 128, 96, dan 64, semuanya dibuat langsung dari Original `256×256`, bukan dipangkas bertahap dari student lain. Target yang akhirnya dipilih adalah `29 → 64 → 64 → 2`.

### Apa yang dihitung satu neuron?

Neuron hidden menghitung kombinasi linear lalu `Tanh`:

\[
z_j=\sum_i w_{ji}x_i+b_j,\qquad h_j=\tanh(z_j).
\]

Weight incoming menentukan seberapa kuat neuron menerima sinyal. Weight outgoing menentukan seberapa kuat hasil neuron diteruskan. Bias \(b_j\) adalah “dorongan dasar” sebelum input ditambahkan. Jika seluruh input nol, pre-activation masih dapat bernilai \(b_j\).

Analogi klinis yang sederhana: dalam model keputusan rujukan, bias menyerupai baseline tendency sebelum red flag atau gejala pasien dimasukkan. Analogi ini hanya menjelaskan fungsi matematis; hidden neuron asli tidak mempunyai interpretasi klinis tunggal.

### Frozen neuron-importance score

Untuk setiap hidden neuron \(j\), implementasi menghitung:

\[
\boxed{
S_j=
\lVert W^{in}_{j,:}\rVert_2+
\lVert W^{out}_{:,j}\rVert_2+
|b_j|
}
\]

Ini **penjumlahan**, bukan rata-rata weight bertanda. L2 norm mencegah weight positif dan negatif saling menghapus. Contoh:

\[
W^{in}=[0.2,-0.3,0.1]
\Rightarrow
\lVert W^{in}\rVert_2=\sqrt{0.2^2+(-0.3)^2+0.1^2}\approx0.374.
\]

Jika incoming norm 0,37, outgoing norm 0,21, dan absolute bias 0,05, maka skor neuron adalah 0,63. Membagi ketiga komponen dengan konstanta tiga akan memberi ranking yang sama, tetapi implementasi sebenarnya menjumlahkan langsung.

Score layer pertama memakai baris `fc1.weight`, kolom terkait pada `fc2.weight`, dan `abs(fc1.bias)`. Score layer kedua memakai baris `fc2.weight`, kolom terkait pada `out.weight`, dan `abs(fc2.bias)`. Ranking dilakukan **terpisah pada setiap layer**. Neuron dengan skor terbesar dipertahankan; jika skor sama, original neuron index yang lebih rendah menang. Exact survivor indices disimpan pada metadata pruning.

```text
previous layer
      │ incoming L2
      ▼
 ┌──────────┐
 │ neuron j │ ← |bias|
 └──────────┘
      │ outgoing L2
      ▼
 next layer
```

Untuk width 64, 64 neuron terbaik dari masing-masing layer dipilih. Weight student kemudian disalin sebagai submatrix yang konsisten: selected rows `fc1`, selected rows dan columns `fc2`, serta selected columns output layer. Output bias Original tetap disalin.

### Input, proses, dan output pruning

**Input:** immutable Original FP32 actor `29→256→256→2`.  
**Proses:** hitung score → stable layer-wise ranking → pilih survivors → salin dense subnetwork.  
**Output:** actor FP32 yang benar-benar lebih kecil serta survivor metadata.  
**Yang tidak berubah:** 29D input, Tanh, two-action output, physical action mapping, perception, dan belief filters.

Score ini adalah **heuristic connectivity magnitude**, bukan oracle yang membuktikan suatu neuron tidak berguna. F12 tidak mengklaim novelty criterion dan tidak memakai F11 attribution untuk memilih neuron.

Pada jalur `Original → Pruning Only`, F14 development diagnostics menemukan:

- Attribution Semantik: `SHIFTED` (0/10 phase–action cells preserved);
- Counterfactual Response: `SHIFTED` (1/3 primary tests preserved);
- Action Fidelity: `FAIL`;
- C4 Behavior: `NOT PRESERVED`.

Frozen F12 selection metrics memperlihatkan kerusakan yang sama secara numerik:

| Model | `v_cmd` MAE | `omega_cmd` MAE | C4 completion |
|---|---:|---:|---:|
| Pruning Only | 0,03504 m/s | 0,35821 rad/s | 0% |
| Pruning + KD | 0,00122 m/s | 0,01908 rad/s | 100% |

Kesimpulan konservatif: direct pruning merusak mapping actor. Ini tidak membuktikan neuron yang dibuang “tidak berguna”; weight yang tersisa sebelumnya dilatih ketika neuron yang kini hilang masih ikut bekerja. Network kecil membutuhkan recovery.

**Rujukan proyek:**

- `src/duckie_pomdp/optimization/actor_compression.py` — `build_pruned_actor()`, `_stable_topk()`
- `configs/f12_belief_ppo_compression_v1.toml` — criterion, bias, tie-break, candidate widths
- `experiments/run_f12_compression.py` — pembuatan seluruh pruning branches dari Original
- `artifacts/f12_belief_ppo_compression_v1/final/ablation_table.csv` — fidelity, C4, size, latency
- `docs/F12_COMPRESSION_ABLATION.md`
- `docs/F14_ABLATION_EXPLANATION.md`

## 21. Knowledge Distillation

Dalam knowledge distillation (KD), Original Policy adalah **teacher** yang dibekukan dan model kecil adalah **student**. KD F12 bukan PPO training ulang: tidak ada reward optimization, critic update, atau ground-truth action dari simulator.

```text
                       SAME NORMALIZED 29D INPUT
                                  │
                    ┌─────────────┴─────────────┐
                    ▼                           ▼
       ORIGINAL ACTOR / TEACHER       COMPRESSED ACTOR / STUDENT
             29→256→256→2                   29→64→64→2
                    ▼                           ▼
             [v_teacher, ω_teacher]      [v_student, ω_student]
                    └──────────── compare ──────┘
                                  ▼
                      normalized Smooth-L1 loss
                                  ▼
                         update STUDENT only
```

### Dari mana data KD berasal?

Development seeds `178001–178008` dikumpulkan melalui deployment pipeline yang tidak berubah:

```text
RGB → MobileNet/YOLO → belief → normalized public 29D → Original actor
```

Dataset menyimpan normalized/physical public 29D, public phase, seed/episode/step, dan deterministic teacher actions. Ia tidak menyimpan privileged truth sebagai training label. F11 locked seeds `177101–177108` dilarang dipakai untuk optimisasi.

**Input KD:** factual public \(x_t^{29D}\).  
**Teacher target:** deterministic physical \([v_T,\omega_T]\).  
**Student prediction:** deterministic physical \([v_S,\omega_S]\).  
**Output:** student weights yang disesuaikan agar \(\pi_S(x)\approx\pi_T(x)\) pada distribusi state terkait.

### Physical action mapping yang ditiru

Actor menghasilkan normalized mean. Implementasi meng-clamp mean ke `[-1,1]`, lalu memetakan:

\[
v=(\mu_v+1)\times0.2\ \text{m/s},
\qquad
\omega=\mu_\omega\times4\ \text{rad/s}.
\]

Jadi teacher dan student dibandingkan dalam unit fisik yang sama.

### Smooth-L1 loss dan normalisasi action range

Rentang fisik kedua output berbeda: velocity memiliki full range 0,4 m/s, sedangkan yaw rate memiliki full range 8 rad/s. F12 membandingkan:

\[
\tilde v=\frac{v}{0.4},\qquad
\tilde\omega=\frac{\omega}{8},
\]

kemudian menghitung `smooth_l1_loss` antara normalized student dan teacher outputs. Ini ekuivalen dengan menilai error relatif terhadap full physical range, sehingga steering tidak mendominasi hanya karena satuannya lebih besar.

Contoh: error velocity 0,02 m/s dan error yaw 0,4 rad/s sama-sama menjadi normalized error 0,05. Smooth-L1 bertindak kuadratik di sekitar error kecil dan lebih linear untuk error besar, sehingga lebih robust daripada pure MSE terhadap outlier error.

Hyperparameter yang dibekukan:

| Parameter | Nilai |
|---|---:|
| Optimizer | Adam |
| Epoch | 80 |
| Batch size | 512 |
| Learning rate | 0,001 |
| Weight decay | 0,000001 |
| Base seed | 2026081401 |
| Loss | normalized physical-action Smooth-L1 |

### Mengapa phase balancing diperlukan?

Nominal driving dapat menghasilkan jauh lebih banyak row daripada stop atau pedestrian event. Implementasi memberi setiap row sampling weight \(1/n_{phase}\), lalu menormalisasikannya menjadi probability. Akibatnya setiap supported phase memperoleh total sampling mass yang seimbang meskipun jumlah row berbeda.

Lima phase adalah `nominal`, `lane_curve`, `pedestrian_relevant`, `stop_required`, dan `stop_satisfied`. Ini menjaga event langka agar tidak tenggelam dalam mini-batch, tetapi tidak menjamin retention pada curriculum lain yang tidak terdapat dalam data C4 tersebut.

### Bagaimana dengan PPO `log_std`?

Original PPO adalah Gaussian stochastic policy dan mempunyai state-independent `log_std`. Deployment fidelity menjelaskan deterministic actor mean. Untuk checkpoint student **FP32**, `log_std` disalin persis dari Original dan tidak menjadi compression target terpisah. Static INT8 artifact adalah actor-only deterministic deployment module; dokumen tidak mengklaim `log_std` ada di file INT8 tersebut.

Pada `Pruning Only → Pruning + Knowledge Distillation`:

- primary counterfactual response pulih menjadi 3/3;
- action fidelity menjadi `PASS`;
- C4 behavior menjadi `PRESERVED`;
- attribution preservation tetap 0/10 (`SHIFTED`).

Distillation menurunkan selection MAE Pruning Only sekitar 96,5% untuk velocity dan 94,7% untuk yaw, serta mengembalikan C4 completion dari 0% menjadi 100%. Hasil explanation tetap tidak kontradiktif: dua jaringan dapat menghasilkan action dan behavior cukup mirip sambil membagi attribution relatif secara berbeda. KD memindahkan **fungsi output teacher**, bukan correspondence satu-per-satu antara teacher neuron dan student neuron.

**Rujukan proyek:**

- `src/duckie_pomdp/optimization/actor_compression.py` — `physical_actions()`, `distill_dense_actor()`
- `experiments/run_f12_compression.py` — data fields, branch seeds, training calls, history output
- `configs/f12_belief_ppo_compression_v1.toml` — seeds, optimizer budget, phase balancing, `log_std`
- `artifacts/f12_belief_ppo_compression_v1/datasets/development_public_actor_states.npz`
- `artifacts/f12_belief_ppo_compression_v1/prune_distill/pd75/distillation_history.json`
- `artifacts/f12_belief_ppo_compression_v1/final/ablation_table.csv`
- `docs/F12_COMPRESSION_PROTOCOL.md` — frozen scientific protocol
- `docs/F12_COMPRESSION_RESULTS.md` — recovery summary
- `docs/F14_ABLATION_EXPLANATION.md`

## 22. Post-Training Quantization (PTQ)

Quantization tidak membuang neuron dan tidak harus mengubah parameter count. Ia mengubah representasi numerik. Secara konseptual nilai float \(x\) dikodekan sebagai integer:

\[
q=\operatorname{round}(x/s)+z,
\qquad
\hat{x}=s(q-z),
\]

dengan scale \(s\), zero-point \(z\), dan kode integer \(q\). Banyak nilai FP32 yang berdekatan dapat jatuh pada kode integer sama; selisih \(x-\hat{x}\) adalah quantization error. Karena output policy kontinu, istilah evaluasi yang dipakai adalah **action fidelity**, bukan classification accuracy.

PTQ mengubah model yang sudah selesai dilatih dari FP32 ke INT8 menggunakan calibration, tanpa gradient recovery setelah conversion.

```text
trained FP32 actor
→ run representative development states
→ observe activation ranges
→ determine quantization parameters
→ convert three Linear layers
→ static INT8 actor
```

### Apa yang benar-benar dikuantisasi?

Frozen backend adalah PyTorch `torch.ao` eager static quantization pada CPU x86:

- Linear weights: `qint8`, per-channel symmetric;
- activations: `quint8`, per-tensor affine;
- tiga `Linear` harus benar-benar menjadi `torch.ao.nn.quantized.Linear`;
- `Tanh` tetap float dengan explicit quantize/dequantize boundaries;
- calibration memakai development public states, maksimal 1.024 row per phase;
- `require_real_int8()` fail-closed jika bukan tepat tiga quantized Linear atau weight bukan `qint8`.

Per-channel weight scale berarti setiap output channel dapat mempunyai scale sendiri. Ini mempertahankan resolusi lebih baik ketika range weight antar-neuron berbeda. Activation memakai satu affine scale/zero-point per tensor. Kode tidak menetapkan klaim tersendiri bahwa bias disimpan sebagai INT8; integritas label INT8 diperiksa dari quantized Linear modules dan qint8 weights.

```text
input float
→ QuantStub → INT8 Linear → DeQuantStub → float Tanh
→ QuantStub → INT8 Linear → DeQuantStub → float Tanh
→ QuantStub → INT8 Linear → DeQuantStub → physical mapping
```

### Pruning versus quantization

| Aspek | Pruning | Quantization |
|---|---|---|
| Yang diubah | struktur jaringan | representasi angka/kernels |
| Neuron dibuang | ya | tidak |
| Parameter count berkurang | ya | tidak harus |
| FP32→INT8 | tidak | ya |
| Risiko utama | kehilangan capacity/mapping | rounding, clipping, calibration error |

Contoh nyata: Original dan PTQ-only sama-sama memiliki 73.986 parameters dan architecture 256×256. PTQ menurunkan actor file 299.667→109.160 bytes dan CPU median latency 42,77→16,63 µs, tetapi tidak mengubah parameter count.

Pada PTQ actor original-size, C4 behavior masih `PRESERVED`, tetapi action fidelity `FAIL`, attribution 7/10 (`PARTIAL`), dan counterfactual 2/3 (`PARTIAL`). Jadi pada data proyek ini efek PTQ tidak dapat diringkas sebagai noise angka kecil saja.

Pada actor yang sudah dipulihkan dengan distillation (`Pruning + Distillation + PTQ`), fidelity dan C4 tetap lulus, tetapi semantic attribution tetap shifted dan counterfactual hanya partial. Efek operasi harus dibaca dalam konteks model asalnya.

**Rujukan proyek:**

- `src/duckie_pomdp/optimization/actor_compression.py` — `QuantizableBeliefActor`, `prepare_ptq()`, `require_real_int8()`
- `experiments/run_f12_compression.py` — `balanced_calibration()` dan A3/A5/A6 PTQ branches
- `configs/f12_belief_ppo_compression_v1.toml` — backend, dtype, granularity, calibration cap
- `artifacts/f12_belief_ppo_compression_v1/final/ablation_registry.json` — frozen actor paths/hashes/backend metadata
- `artifacts/f12_belief_ppo_compression_v1/final/ablation_table.csv` — parameter, bytes, latency, fidelity, C4
- `docs/F12_COMPRESSION_PROTOCOL.md`
- `docs/F12_COMPRESSION_ABLATION.md`
- `docs/F14_ABLATION_EXPLANATION.md`

## 23. Quantization-Aware Training (QAT)

QAT mensimulasikan rounding/clipping quantization selama recovery melalui fake-quantization. Parameter trainable tetap dapat di-update oleh gradient, tetapi forward path “merasakan” level quantized yang kelak dipakai deployment.

```text
FP32/fake-quant student
→ simulate quantize/dequantize effects
→ deterministic physical student action
→ compare with frozen Original teacher action
→ normalized Smooth-L1 KD
→ update fake-quant student
→ convert to deployable static INT8
```

Dengan demikian QAT branch proyek ini selalu **teacher-guided QAT distillation**, bukan PPO/reward training. QAT A4 dimulai dari unpruned Original-size actor. QAT A7 dimulai dari recovered pruned FP32 A2, lalu menjalani fake-quant KD sebelum conversion.

Frozen QAT budget adalah 40 epochs, learning rate 0,0001, Adam dengan weight decay KD yang sama, batch size 512, phase-balanced sampling, dan base seed 2026081402. A4 menggunakan base seed; A7 menggunakan offset satu. Setelah conversion, `require_real_int8()` memastikan deployment artifact berisi tiga static INT8 Linear kernels.

| Aspek | PTQ | QAT |
|---|---|---|
| Kapan efek quantization dikenalkan? | setelah training | selama recovery/training |
| Ada gradient recovery? | tidak | ya, melalui fake-quantized path |
| Output deployment | INT8 | INT8 |
| Tujuan | conversion sederhana | actor belajar menghadapi error quantization |

### Apa yang QAT pulihkan pada proyek ini?

Pada selected pruning pathway, A6 (`Pruning + Distillation + PTQ`) sudah lulus fidelity dan C4. Frozen selection rule hanya memilih A7 jika QAT meningkatkan mean normalized two-action MAE minimal 10% tanpa merusak behavior. A7 memberi improvement 10,654%, sehingga menjadi selected deployment actor.

Pada final 17.600 public states, A7 mencapai `v_cmd` MAE 0,002228 m/s dan `omega_cmd` MAE 0,035594 rad/s, lalu lulus semua overall dan phase-wise fidelity gates. C4 final tetap 8/8 complete tanpa collision, unsafe episode, stop violation, atau lane failure. Namun F14 menunjukkan semantic attribution dan counterfactual sensitivity tetap shifted. Jadi QAT/KD memulihkan deployment fidelity, bukan bukti internal semantic equivalence.

F13 tidak dapat melakukan gradient IG pada A7 karena exact pre-conversion QAT state historis tidak dipersist. F14 tidak “memperbaiki” F13; F14 memakai metode baru yang model-agnostic, Group Shapley, langsung pada forward FP32/INT8.

**Rujukan proyek:**

- `src/duckie_pomdp/optimization/actor_compression.py` — `prepare_qat()`, `convert_qat()`, shared KD loop
- `experiments/run_f12_compression.py` — A4/A7 construction dan frozen QAT selection rule
- `configs/f12_belief_ppo_compression_v1.toml` — QAT budget dan 10% selection margin
- `artifacts/f12_belief_ppo_compression_v1/prune_distill_quant_distill/qat_distillation_history.json`
- `artifacts/f12_belief_ppo_compression_v1/prune_distill_quant_distill/actor_int8.pt`
- `artifacts/f12_belief_ppo_compression_v1/final/model_selection.json`
- `docs/F12_COMPRESSION_RESULTS.md`
- `docs/F13_FINAL_REPORT.md`
- `docs/F14_PROTOCOL.md`

### Ringkasan pipeline optimisasi final

```text
Original Actor
29→256→256→2, FP32, 73,986 parameters
        │
        │ structured neuron pruning
        ▼
Pruned Actor
29→64→64→2, FP32, 6,210 parameters
        │
        │ phase-balanced teacher KD
        ▼
Recovered Pruned Actor
29→64→64→2, FP32
        │
        │ fake-quant QAT + teacher KD
        ▼
Quantization-Robust Student
        │
        │ static INT8 conversion + kernel integrity check
        ▼
Final INT8 Actor
29→64→64→2, 6,210 parameters, 36,880-byte actor file
```

Compression datang dari dua mekanisme berbeda: pruning mengurangi parameters 73.986→6.210; quantization mengurangi precision dan logical parameter memory. Distillation mengajarkan kembali mapping output, tetapi tidak mengubah jumlah parameters. Hasil akhirnya adalah 91,61% parameter reduction, 87,69% actor-file reduction, dan 3,04× actor-only CPU speedup—bukan 3,04× end-to-end perception-to-action speedup.

## 24. Semua Varian Optimisasi Tanpa A-Code sebagai Label Utama

Angka berikut adalah F14 **development diagnostic 500 states**, kecuali efficiency yang berasal dari frozen F12.

| Nama model | Apa yang dilakukan | Arsitektur/precision | Attribution | Counterfactual | Action fidelity | C4 behavior | Interpretasi | ID teknis |
|---|---|---|---|---|---|---|---|---|
| Original Policy | tidak diubah | 256×256 FP32 | 10/10 reference | 3/3 reference | Reference | Reference | acuan | A0 |
| Pruning Only | hidden neurons dipangkas | 64×64 FP32 | 0/10 SHIFTED | 1/3 SHIFTED | FAIL | NOT PRESERVED | direct pruning merusak mapping | A1 |
| Pruning + Knowledge Distillation | pruning lalu recovery teacher | 64×64 FP32 | 0/10 SHIFTED | 3/3 PRESERVED | PASS | PRESERVED | fungsi/behavior pulih, attribution tidak | A2 |
| Post-Training Quantization (PTQ) | original-size langsung INT8 | 256×256 INT8 | 7/10 PARTIAL | 2/3 PARTIAL | FAIL | PRESERVED | behavior lulus walau equivalence gagal | A3 |
| QAT + Distillation | unpruned fake-quant recovery lalu INT8 | 256×256 INT8 | 5/10 PARTIAL | 3/3 PRESERVED | FAIL | PRESERVED | sensitivity pulih; fidelity gate tidak | A4 |
| Pruning + PTQ | pruning lalu langsung INT8 | 64×64 INT8 | 1/10 SHIFTED | 1/3 SHIFTED | FAIL | NOT PRESERVED | PTQ tidak menyelamatkan kerusakan pruning | A5 |
| Pruning + Distillation + PTQ | recovery FP32 lalu PTQ | 64×64 INT8 | 0/10 SHIFTED | 2/3 PARTIAL | PASS | PRESERVED | deployable tetapi semantic drift tetap | A6 |
| Final INT8: Pruning + Distillation + QAT | recovery FP32 dan QAT/KD | 64×64 INT8 | 0/10 SHIFTED | 1/3 SHIFTED | PASS | PRESERVED | selected C4 deployment; bukan semantic equivalent | A7 |

**Rujukan proyek:**

- `artifacts/f14_explainability_aware_compression_v1/ablation_comparison_metrics.json`
- `artifacts/f14_explainability_aware_compression_v1/failure_modes/failure_hierarchy.json`
- `artifacts/f12_belief_ppo_compression_v1/final/model_selection.json`

## 25. Failure Mode per Tahap Optimisasi

### Pruning Only

Attribution shifted, counterfactual shifted, fidelity gagal, dan C4 tidak dipertahankan. Kerusakan terjadi pada mapping actor dan perilaku. Tidak boleh diklaim satu jenis drift sendirian menyebabkan failure.

### Pruning + Knowledge Distillation

Counterfactual, fidelity, dan C4 pulih; attribution belum kembali setara Original. Distillation memulihkan kompetensi fungsional tanpa memulihkan attribution equivalence.

### PTQ

Attribution dan counterfactual partial, fidelity gagal, tetapi C4 preserved. Ini menunjukkan behavioral completion dan numerical equivalence adalah gate berbeda.

### QAT + Distillation

Counterfactual primary preserved dan C4 preserved; attribution partial dan fidelity tetap gagal. Recovery tambahan tidak otomatis memperbaiki semua axis.

### Pruning + PTQ

Attribution/counterfactual shifted, fidelity gagal, C4 tidak preserved. Dibanding Pruning Only, PTQ tidak merescue degradasi yang sudah ada; evidence disebut pruning-dominated, bukan bukti bahwa PTQ tidak memberi efek tambahan sama sekali.

### Pruning + Distillation + PTQ

Fidelity dan C4 preserved, counterfactual partial, attribution shifted. PTQ mempertahankan deployment competence hasil recovery, tetapi tidak semantic equivalence.

### Final INT8

Fidelity dan C4 preserved; attribution dan counterfactual shifted pada diagnostic protocol. Deployment success tidak sama dengan explanation equivalence.

**Rujukan proyek:**

- `docs/F14_FAILURE_MODE_REPORT.md`
- `artifacts/f14_explainability_aware_compression_v1/failure_modes/failure_hierarchy.json`
- `docs/F14_VISUAL_CONSISTENCY_AUDIT.md`

## 26. Final Original vs Final INT8

Final comparison berbeda dari development table. Ia memakai **4.400 frozen R004 public states**, same states, same 24 reference assignments, same coalition vectors, dan action mapping yang sama untuk Original dan Final INT8.

Hasil final:

- semantic attribution: `SHIFTED`, hanya 1/10 phase–action cells preserved;
- counterfactual functional sensitivity: `SHIFTED`, 1/3 primary tests preserved;
- action fidelity: `PASS` pada frozen F12 gate;
- C4 behavior: `PRESERVED` pada tested final scenarios.

Salah satu pola yang terlihat adalah kenaikan relative attribution ke `PreviousAction`, terutama pada beberapa steering contexts; misalnya top group `stop_required × omega_cmd` berpindah dari Lane pada Original ke PreviousAction pada Final INT8. Kalimat yang tepat adalah “relative attribution toward PreviousAction increased”, bukan “model lebih banyak berpikir tentang action sebelumnya”.

**Rujukan proyek:**

- `docs/F14_FINAL_REEXPLANATION.md`
- `artifacts/f14_explainability_aware_compression_v1/final_comparison_metrics.json`
- `artifacts/f14_explainability_aware_compression_v1/final_a0_a7_shapley.csv`
- `artifacts/f14_explainability_aware_compression_v1/final_a0_a7_counterfactuals.csv`

## 27. Hasil Efisiensi

Final INT8 dibanding actor original:

- parameter reduction: sekitar 91,61%;
- actor-file reduction: sekitar 87,69%;
- actor-only CPU median speedup: sekitar 3,04×;
- architecture: 256×256 FP32 → 64×64 static INT8.

Angka latency ini hanya actor batch-1 pada CPU x86. Ia tidak boleh dipresentasikan sebagai 3,04× percepatan sistem visuomotor lengkap karena MobileNet, YOLO, projection, dan belief updater tidak dikompresi.

**Rujukan proyek:**

- `docs/F12_COMPRESSION_RESULTS.md`
- `artifacts/f12_belief_ppo_compression_v1/benchmarks/actor_benchmarks.json`
- `artifacts/f12_belief_ppo_compression_v1/final/model_selection.json`

## 28. Retention Limitation

Scope deployment terpilih adalah C4. Frozen retention evidence melaporkan Final INT8 menyelesaikan C3/C4, tetapi completion C0–C2 adalah 0% pada retention suite yang dipakai. Ini adalah **behavioral retention limitation** dan tetap harus terlihat.

F14 tidak menemukan compatible saved public 29D rows untuk membuat same-state semantic explanation C0–C3 yang sah. Karena itu **semantic retention explanation = UNRESOLVED**. UNRESOLVED lebih ilmiah daripada mengisi bukti yang tidak tersedia dengan “zero drift” atau merender ulang frozen evaluation tanpa izin.

**Rujukan proyek:**

- `docs/F14_FINAL_REPORT.md`
- `artifacts/f14_explainability_aware_compression_v1/retention_semantic_diagnostic.json`
- `docs/F12_COMPRESSION_RESULTS.md`

## 29. Temuan Ilmiah Utama dalam Bahasa Sederhana

1. Original Policy memakai pola informasi semantik yang berbeda di phase berkendara yang berbeda.
2. Aggressive structured pruning tanpa recovery merusak mapping policy.
3. Knowledge distillation memulihkan functional response, action fidelity, dan C4 behavior, tetapi tidak mengembalikan exact attribution equivalence.
4. PTQ dapat disertai semantic/functional drift, bukan hanya perubahan numerik yang tidak bermakna.
5. Final INT8 mempertahankan action fidelity dan behavior C4 pada kondisi yang diuji.
6. Final INT8 tidak semantically atau functionally equivalent dengan Original di bawah frozen F14 diagnostics.
7. Behavioral equivalence tidak sama dengan explanation equivalence.
8. Retention di luar C4 terbatas, dan semantic retention explanation belum tersedia.

## 30. Apa yang Aman dan Tidak Aman Diklaim?

| AMAN DIKLAIM | TIDAK BOLEH DIKLAIM |
|---|---|
| Group Shapley menunjukkan relative contribution pada coalition/reference protocol tertentu. | Shapley membuktikan sebab fisik perilaku. |
| Semantic counterfactual menunjukkan policy-input functional sensitivity. | Menghapus input membuktikan causal effect di dunia nyata. |
| Final INT8 mempertahankan tested C4 behavior. | Final INT8 setara secara universal dengan Original. |
| Relative attribution ke PreviousAction meningkat pada beberapa phase/action. | Model “berpikir” tentang previous action. |
| Distillation disertai recovery fidelity dan C4. | Distillation pasti memulihkan reasoning internal. |
| Pruning Only merusak policy mapping pada protocol ini. | Neuron yang dipangkas tidak berguna. |
| Actor-only CPU inference lebih cepat. | Sistem RGB-to-action lengkap 3,04× lebih cepat. |

## 31. Glosarium

| Istilah | Arti sederhana |
|---|---|
| POMDP | pengambilan keputusan ketika state dunia hanya terlihat sebagian |
| State | keadaan dunia pada satu waktu |
| Observation | informasi yang tersedia bagi sistem pada satu langkah |
| Measurement | estimasi satu-frame dari sensor/perception |
| Belief | estimasi state temporal beserta ketidakpastian |
| EKF | filter rekursif yang memprediksi lalu mengoreksi belief |
| YOLO | detector objek pada gambar |
| MobileNet | jaringan ringan untuk perception; di sini menghasilkan lane measurement |
| PPO | algoritme reinforcement learning yang melatih policy |
| Actor | jaringan yang menghasilkan action policy |
| Critic | jaringan yang memperkirakan nilai dan membantu training |
| Action | command yang dikirim ke robot |
| `v_cmd` | command kecepatan maju dalam m/s |
| `omega_cmd` | command laju belok dalam rad/s |
| Attribution | pembagian relative contribution input terhadap output |
| Group Shapley | attribution eksak atas coalition enam semantic groups |
| Counterfactual | input semantik yang diubah terkontrol untuk melihat respons action |
| Fidelity | kedekatan output optimized actor terhadap Original pada input sama |
| Closed-loop | policy berulang kali bertindak dan menerima state berikutnya dari environment |
| Nominal | berkendara biasa di luar curve/pedestrian/active-stop phase terdaftar |
| Pruning | membuang hidden neurons sehingga jaringan dense lebih kecil |
| Knowledge Distillation | student belajar meniru output teacher |
| FP32 | angka floating point 32-bit |
| INT8 | representasi integer 8-bit untuk deployment |
| PTQ | quantization setelah training selesai |
| QAT | training/recovery dengan simulasi quantization |
| Retention | kemampuan lama yang tetap bertahan setelah tahap baru/optimisasi |
| Ablation | varian eksperimen yang mengisolasi satu kombinasi operasi |
| Checkpoint | file bobot/state model yang dibekukan |
| SHA256 | fingerprint file untuk memeriksa identitas dan integritas |

## 32. Project File Map / Rujukan File

Semua path di tabel ini dikonfirmasi ada pada repository aktif. `EXPERIMENT_PLAN.md` tidak berada di root aktif; histori plan yang tersedia ada di `refine-logs/` dan tidak dipakai sebagai source of truth utama.

| Tahap | Tujuan | Input utama | Output utama | Source code | Config | Artifact/checkpoint | Report/dokumentasi | Status |
|---|---|---|---|---|---|---|---|---|
| F0–F8 foundation | formulasi POMDP, domain, simulator, action/measurement/belief contracts | simulator state dan sensor | public contracts | `src/duckie_pomdp/domain/`, `src/duckie_pomdp/adapters/` | `configs/` | `artifacts/` stage-specific | `FORMULATION.md`, `GATES.md`, `IMPLEMENTATION_NOTES.md` | sesuai gate historis masing-masing |
| F9a | kalibrasi YOLO measurement | detections | calibrated range/bearing noise | `experiments/calibrate_f9_yolo_measurement.py` | config measurement terkait | `artifacts/yolo_measurement_metrics.json` | `GATES.md` | PASS calibration evidence |
| F9b/F9c/F9d | pedestrian belief robustness | YOLO measurement | EKF/existence belief | `src/duckie_pomdp/belief/pedestrian_ekf.py`, `src/duckie_pomdp/belief/robust_updater.py` | `configs/f9c_robust_belief_v1.toml` | F9 evidence under `artifacts/` | `docs/superpowers/F9C_REPORT_FOR_REVIEW.md`, `docs/superpowers/F9D_REPORT_FOR_REVIEW.md` | LIMITED per frozen gates |
| F10 C0–C4 | train curriculum Belief-PPO | public normalized 29D | Original PPO C4 | `experiments/train_f10_ppo.py`, `src/duckie_pomdp/control/ppo.py` | `configs/f10_ppo_visual_objects_v30.toml` | `artifacts/f10_ppo_visual_objects_v30/c4/ppo_selected.pt` | `docs/F10_PPO_CURRICULUM.md` | C4 selected |
| F11 R001 | verify deployment observation/explanation boundary | stored public trace | exact 29D/action replay | F11 verification scripts under `experiments/` | F11 configs under `configs/` | `artifacts/f11_ppo_explanation_v2/r001/` | `docs/F11_R001_OBSERVATION_CONTRACT_REPORT.md` | PASS |
| F11 R002/R002b/R003 | baseline robustness, multi-reference IG, semantic interventions | development public states | robustness/intervention evidence | F11 scripts and `src/duckie_pomdp/explain/` | F11 configs | `artifacts/f11_ppo_explanation_v2/` | `docs/F11_R002_R003_REPORT_FOR_REVIEW.md` | LIMITED/PASS/PASS |
| F11 R004 | final Distributional IG | 4.400 locked public states | frozen original attribution | F11 attribution scripts | frozen R004 protocol | `artifacts/f11_ppo_explanation_v2/r004/` | `docs/F11_R004_REPORT_FOR_REVIEW.md` | PASS |
| F12 | actor compression | Original actor + public optimization states | A0–A7 dan selected A7 | `src/duckie_pomdp/optimization/`, F12 scripts under `experiments/` | `configs/f12_belief_ppo_compression_v1.toml` | `artifacts/f12_belief_ppo_compression_v1/` | `docs/F12_COMPRESSION_PROTOCOL.md`, `docs/F12_COMPRESSION_RESULTS.md`, `docs/F12_COMPRESSION_ABLATION.md` | PASS, C4-only |
| F13 | explain-again attempt dan failure probe | Original + A7 | counterfactual/behavioral diagnostic; gradient blocked | F13 scripts under `experiments/` | `configs/f13_explain_compressed_v1.toml` | `artifacts/f13_explain_compressed_v1/` | `docs/F13_FINAL_REPORT.md` | LIMITED |
| F14 dev | model-agnostic exact Group Shapley semua ablation | same 500 public states | A0–A7 mechanism diagnosis | `src/duckie_pomdp/explain/group_shapley.py`, `src/duckie_pomdp/explain/compression_diagnostics.py` | `configs/f14_explainability_aware_compression_v1.toml` | `artifacts/f14_explainability_aware_compression_v1/` | `docs/F14_ABLATION_EXPLANATION.md`, `docs/F14_FAILURE_MODE_REPORT.md` | LIMITED overall |
| F14 final | Original vs Final INT8 same-state comparison | frozen 4.400 R004 states | final shared Shapley/counterfactual comparison | F14 scripts under `experiments/` | frozen F14 protocol | `artifacts/f14_explainability_aware_compression_v1/final_comparison_metrics.json` | `docs/F14_FINAL_REEXPLANATION.md`, `docs/F14_FINAL_REPORT.md` | semantic/functional SHIFTED; C4 PRESERVED |
| Human-readable package | menjelaskan hasil tanpa mengubah evidence | frozen reports/artifacts | 11 figures + guide | `experiments/generate_f14_explained_figures_id.py` | tidak ada scientific config baru | `artifacts/f14_explainability_aware_compression_v1/figures_explained_id/` | dokumen ini dan `docs/F14_VISUAL_CONSISTENCY_AUDIT.md` | documentation-only |

## 33. Cara Menelusuri Klaim ke Source

Gunakan urutan berikut ketika memeriksa satu klaim:

1. baca report manusia untuk konteks;
2. cari metric exact pada JSON/CSV machine-readable;
3. cocokkan actor/config/source hash pada manifest;
4. periksa code hanya untuk semantics dan perhitungan, bukan untuk menggantikan hasil artifact;
5. jangan mencampur hasil development 500 states dengan final 4.400 states.

Contoh:

> PPO menerima representasi 29D, bukan RGB secara langsung.

**Rujukan proyek:**

- `src/duckie_pomdp/control/ppo_observation.py`
- `src/duckie_pomdp/control/ppo_environment.py`
- `docs/F11_R001_OBSERVATION_CONTRACT_REPORT.md`

> Final INT8 mempertahankan C4 tetapi attribution dan counterfactual sensitivity shifted.

**Rujukan proyek:**

- `artifacts/f14_explainability_aware_compression_v1/final_comparison_metrics.json`
- `docs/F14_FINAL_REPORT.md`

> Angka figure development dan final dapat berbeda karena datasetnya berbeda.

**Rujukan proyek:**

- `artifacts/f14_explainability_aware_compression_v1/diagnostic_state_manifest.json`
- `artifacts/f11_ppo_explanation_v2/r004/locked_trace_manifest.json`
- `docs/F14_VISUAL_CONSISTENCY_AUDIT.md`

---

**Status sejarah yang dipertahankan:** F11 memiliki R002 `LIMITED`, R002b/R003/R004 `PASS`, R006 `FAILED`, dan R007 `BLOCKED`; F12 `PASS` untuk deployment C4-only; F13 `LIMITED`; F14 `LIMITED`. Package ini hanya dokumentasi dan visualisasi—tidak melakukan retraining, rerun simulator, perubahan threshold, atau reinterpretasi hasil beku.
