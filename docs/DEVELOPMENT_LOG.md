# Duckie POMDP — Formulation Scaffold

Repository ini berisi kontrak formulasi, integrasi Gym-Duckietown nyata,
action adapter yang terukur, minimal POMDP scenario, true-state validation,
calibrated ground projection, F5b range calibration, F6 oracle observation,
F7 pedestrian EKF, dan checkpoint detector baseline F8-prep yang auditable.
Evaluasi frozen-test F8a/F8b dan F9 YOLO-to-frozen-EKF juga sudah selesai.
F9b berstatus LIMITED; F9c (robust observation + belief calibration)
menyelesaikan evaluasi final dengan rekomendasi status LIMITED (keputusan
akhir ada pada human partner). Reward logic dan solver RL belum dimulai.

Urutan kerja yang berlaku:

1. Review dan kunci formulasi di `FORMULATION.md`. Scaffold ini sudah `PASSED`.
2. Integrasikan environment nyata dan pisahkan agent/privileged path.
3. Validasi action adapter dan actuator envelope pada simulator nyata.
4. Validasi complete true state dan signed stop-line geometry.
5. Validasi pixel-to-ground projection sebelum detector.
6. Kunci range semantics dan measurement calibration melalui F5b.
7. Validasi F6 oracle measurement model dan F7 belief updater.
8. Dataset detector dan baseline YOLO sudah melewati F8-prep.
9. F8a/F8b sudah mengukur detector dan observation error pada frozen test set.
10. F9a measurement calibration sudah passed; F9b runtime evaluation selesai
    dengan status LIMITED karena posterior range masih overconfident.
11. F9c (robust observation + belief calibration) sudah selesai evaluasi
    final pada seed `7101-7104`: Robust B menurunkan RMSE dan memperbaiki
    miss-retention signifikan dibanding Baseline A, tetapi coverage range
    overshoot pita pre-registered pada sisi konservatif, sehingga
    rekomendasi status adalah LIMITED, bukan CONTROL_READY. Reward dan
    solver tetap belum dimulai; keputusan decision/control menunggu human
    partner meninjau `IMPLEMENTATION_NOTES.md` bagian "F9c gate report".

Policy action sudah dikunci sebagai
`PolicyAction(v_cmd, omega_cmd)`. Konversi menjadi left/right wheel command
adalah tanggung jawab `ActionAdapter`, bukan policy.

Kandidat bounds V1 hasil F2 adalah `v_cmd in [0, 0.4] m/s` dan
`omega_cmd in [-4, 4] rad/s`; nilainya belum permanen. Detail desain dan bukti ada di
`IMPLEMENTATION_NOTES.md`.

Domain contracts mengikuti `docs/POMDP_FORMULATION_SCAFFOLD.md`. Jalankan contract gate dengan:

```bash
export PYTHONPATH=src:/home/pannntastic/aivnv/duckie/src
export DUCKIETOWN_HEADLESS=1
/home/pannntastic/aivnv/duckie/.venv/bin/python -m pytest tests -q
```

Implementasi lama tidak dihapus. Semuanya berada di `_archive/attempt_01/` dan
tidak ikut menjadi bagian dari scaffold aktif.

## Probe

Jalankan F4 state-validation experiment:

```bash
export PYTHONPATH=src:/home/pannntastic/aivnv/duckie/src
export DUCKIETOWN_HEADLESS=1
/home/pannntastic/aivnv/duckie/.venv/bin/python experiments/validate_state.py
```

Jalankan F5 ground-projection validation:

```bash
export PYTHONPATH=src:/home/pannntastic/aivnv/duckie/src
export DUCKIETOWN_HEADLESS=1
/home/pannntastic/aivnv/duckie/.venv/bin/python \
  experiments/validate_ground_projection.py
```

Jalankan F5b origin/surface comparison dan held-out calibration:

```bash
export PYTHONPATH=src:/home/pannntastic/aivnv/duckie/src
export DUCKIETOWN_HEADLESS=1
/home/pannntastic/aivnv/duckie/.venv/bin/python \
  experiments/validate_range_semantics.py
```

`range_m` secara kanonik berarti distance ke object-model origin. Nearest
collision-footprint range disimpan terpisah sebagai privileged validation
quantity. Runtime calibration hanya memakai raw projection dan parameter tetap
di `configs/measurement_model_v1.toml`.

Artefak reproducible:

```text
artifacts/state_validation.csv
artifacts/ground_projection_validation.csv
artifacts/ground_projection_metrics.json
artifacts/range_semantics_validation.csv
artifacts/range_calibration_validation.csv
artifacts/measurement_noise_v1.json
artifacts/oracle_measurement_validation.csv
artifacts/ekf_tracking_validation.csv
artifacts/belief_calibration_metrics.json
artifacts/detection_dataset_v1_manifest.json
artifacts/detection_dataset_v1_stats.json
artifacts/yolo_v1/model_manifest.json
artifacts/yolo_v1/training_metrics.json
artifacts/yolo_v1/test_sanity.json
artifacts/yolo_v1/best.pt
```

Jalankan F6 oracle Monte Carlo validation dari real simulator truth:

```bash
export PYTHONPATH=src:/home/pannntastic/aivnv/duckie/src
export DUCKIETOWN_HEADLESS=1
/home/pannntastic/aivnv/duckie/.venv/bin/python \
  experiments/validate_oracle_measurement.py
```

Jalankan F7 enam-skenario, tiga-mode EKF validation dan Q sensitivity:

```bash
export PYTHONPATH=src:/home/pannntastic/aivnv/duckie/src
export DUCKIETOWN_HEADLESS=1
/home/pannntastic/aivnv/duckie/.venv/bin/python \
  experiments/validate_pedestrian_ekf.py
```

Konfigurasi sintetis oracle, Q, initialization, dan existence filter berada di
`configs/oracle_ekf_v1.toml`. Angka dropout/existence tersebut bukan hasil ukur
YOLO. GT berhenti di `OracleObservationModel`; belief updater hanya menerima
`ObjectMeasurement` dan actual `EgoMotion`.

Jalankan ulang F2 actuator-envelope experiment:

```bash
export PYTHONPATH=src:/home/pannntastic/aivnv/duckie/src
export DUCKIETOWN_HEADLESS=1
/home/pannntastic/aivnv/duckie/.venv/bin/python \
  experiments/characterize_action_envelope.py
```

Hasil machine-readable disimpan di `artifacts/action_envelope.csv`. Setiap
case membedakan 10 transient steps dan 20 steady measurement steps.

Jalankan ulang Gate A0:

```bash
export PYTHONPATH=src:/home/pannntastic/aivnv/duckie/src
export DUCKIETOWN_HEADLESS=1
/home/pannntastic/aivnv/duckie/.venv/bin/python \
  scripts/gate_action_adapter.py
```

Gunakan environment Python simulator yang sudah ada:

```bash
export PYTHONPATH=src:/home/pannntastic/aivnv/duckie/src
export DUCKIETOWN_HEADLESS=1
/home/pannntastic/aivnv/duckie/.venv/bin/python scripts/probe_camera.py
```

Regenerasi dan validasi dataset detector dari simulator nyata:

```bash
export PYTHONPATH=src:/home/pannntastic/aivnv/duckie/src
export DUCKIETOWN_HEADLESS=1
/home/pannntastic/aivnv/duckie/.venv/bin/python \
  experiments/generate_detection_dataset.py
/home/pannntastic/aivnv/duckie/.venv/bin/python \
  experiments/validate_detection_dataset.py
```

Dataset `duckietown_detection_v1` memakai mapping tunggal
`0=stop_sign, 1=duckie`. Split dilakukan per episode/seed/mode, bukan per
frame. RGB/mask privileged hanya digunakan offline untuk membuat bbox; runtime
detector tetap menerima `front_rgb` saja.

Training baseline dan readiness probe:

```bash
export PYTHONPATH=src:/home/pannntastic/aivnv/duckie/src
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONHASHSEED=8123
/home/pannntastic/aivnv/duckie/.venv/bin/python \
  experiments/train_yolo_v1.py
/home/pannntastic/aivnv/duckie/.venv/bin/python \
  scripts/probe_yolo_readiness.py
```

Checkpoint stabil berada di `artifacts/yolo_v1/best.pt`; hash, class mapping,
framework, dataset, dan seed disimpan di
`artifacts/yolo_v1/model_manifest.json`. Probe hanya membuktikan readiness.
Ia bukan pengganti evaluasi F8a/F8b pada test split.

## Frozen F8a/F8b evaluation

Operating point final ditetapkan sebelum membaca hasil test: confidence
`0.10`, NMS IoU `0.70`, match IoU `0.50`, input `480`, dan checkpoint SHA256
`3d4f816d440690493b856d25403a84a3249e4250599319c32569b97cb8d7482c`.
Perintah berikut menjalankan inference final dan karena itu bukan smoke test
harian atau perintah untuk threshold search:

```bash
export PYTHONPATH=src:/home/pannntastic/aivnv/duckie/src
export DUCKIETOWN_HEADLESS=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONHASHSEED=8123
export CUDA_VISIBLE_DEVICES=0
/home/pannntastic/aivnv/duckie/.venv/bin/python \
  experiments/evaluate_yolo_v1.py
```

Verifikasi artefak tanpa menjalankan inference ulang:

```bash
/home/pannntastic/aivnv/duckie/.venv/bin/python \
  scripts/verify_f8_evaluation.py
```

F8a pada 164 held-out images menghasilkan recall `1.0` untuk 62 stop-sign
dan 84 Duckie opportunities. Fixed-threshold precision masing-masing
`0.8857` dan `0.9655`; tidak ada FN. Far stop-sign tetap `N/A` karena split
test tidak mengandung opportunity tersebut. F8b menunjukkan raw YOLO range
RMSE `0.17269 m` dan bearing RMSE `0.05348 rad`. Frozen F5b calibration tidak
dipakai dalam kandidat F9 karena menaikkan range RMSE menjadi `0.17494 m`;
tidak ada calibration baru yang di-fit pada test set.

Artefak utama:

```text
artifacts/yolo_detection_validation.csv
artifacts/yolo_detection_metrics.json
artifacts/yolo_measurement_validation.csv
artifacts/yolo_measurement_metrics.json
artifacts/yolo_measurement_noise_v1.json
artifacts/yolo_error_cases/
configs/measurement_model_yolo_v1.toml
```

`measurement_model_yolo_v1.toml` berstatus candidate untuk F9 dan belum
diterapkan ke EKF F7 yang frozen.

## F9 YOLO-to-EKF evaluation

F9 memakai seed kalibrasi `4101--4104` dan seed evaluasi final
`5101--5104`; keduanya juga terpisah dari seed train/validation/test detector.
Konfigurasi [f9_yolo_ekf_v1.toml](configs/f9_yolo_ekf_v1.toml) membekukan
checkpoint, split, koreksi bias, `R_YOLO`, dan parameter observasi existence
sebelum evaluasi final dibuka. F5b calibration tidak berada di runtime path.

Kalibrasi F9a menghasilkan 1.193 matched measurements dan memilih koreksi
aditif `b_r=-0.0459048 m`, `b_beta=+0.00414568 rad`. Koreksi runtime adalah
`z_corrected=z_raw-b`. Range sigma near/medium/far adalah
`0.003048/0.012491/0.016062 m`; bearing sigma `0.012648 rad`. Estimasi
calibration-only adalah `P_D=0.976678` dan `P_FA=0.000780` (0 event dalam 640
RGB counterfactual dengan Duckie disembunyikan). Counterfactual renderer hanya
ada pada privileged validation port; detector tetap menerima RGB saja.

Final F9b memakai satu inference YOLO per frame dan menjalankan raw/corrected
EKF berdampingan pada 2.172 frame. Analisis artefak dan verifier tidak
menjalankan inference ulang:

```bash
export PYTHONPATH=src:/home/pannntastic/aivnv/duckie/src
/home/pannntastic/aivnv/duckie/.venv/bin/python \
  experiments/analyze_f9_results.py
/home/pannntastic/aivnv/duckie/.venv/bin/python \
  experiments/verify_f9_artifacts.py
```

Corrected current-frame/EKF range RMSE adalah `0.02898/0.02808 m`; bearing
RMSE `0.00662/0.00871 rad`. Jadi EKF hanya memberi perbaikan range kecil dan
justru memperburuk bearing dibanding current frame. Corrected EKF range bias
tetap positif `+0.01879 m`. Range posterior sangat overconfident: coverage
68/95 persen hanya `0.152/0.258`, walau NIS exceedance global `0.0542` dekat
target 0.05. Ini menunjukkan campuran residual temporally correlated/outlier
tidak direpresentasikan baik oleh diagonal Gaussian `R`.

Ada 57 natural misses, 78 duplicate-selection frames, 16 visible localization
mismatches, nol class-negative false alarm, dan nol false-track initialization.
Belief aktif hanya bertahan pada 8/57 missed frames karena real `P_D` yang
tinggi membuat miss menjadi evidence absence yang kuat. Outlier IoU<0.5
mencapai measurement range RMSE `0.1511 m` dan NIS P95 `282.0`, tetapi tidak
menyebabkan numerical filter failure. F9b karena itu berstatus LIMITED, bukan
PASSED untuk control readiness. F7 equations/Q tetap tidak berubah.

Artefak:

```text
artifacts/f9_yolo_measurement_calibration.csv
artifacts/f9_measurement_model.json
artifacts/f9_yolo_ekf_validation.csv
artifacts/f9_belief_metrics.json
artifacts/f9_nis_metrics.json
artifacts/f9_error_cases/
```

## F9c — robust observation and belief calibration

F9c membuat estimator belief pedestrian F9b (berstatus LIMITED) menjadi
robust dan calibrated: bias refit pada seed kalibrasi baru, covariance
inflation + posterior variance floor, innovation gate + temporal
association, dan conditional detection dengan miss-likelihood floor
(invariant I8) — semuanya di sekitar F7 EKF yang tetap frozen byte-for-byte.
Kalibrasi memakai seed `6101-6108` (8 seed); evaluasi final memakai seed
`7101-7104`, dirender **tepat satu kali** (2026-08-09) dan tidak boleh
dirender ulang. `configs/f9c_robust_belief_v1.toml` read-only sejak freeze
Task 10, SHA256 `359dc52020421c248bf6c26e036234191b2b97d24b505a66d85daa85b563704e`.

Verifikasi artefak tanpa render/inference ulang:

```bash
export PYTHONPATH=src:/home/pannntastic/aivnv/duckie/src
/home/pannntastic/aivnv/duckie/.venv/bin/python \
  experiments/verify_f9c_artifacts.py
```

Reproduksi metrik final (replay dari runtime cache yang sudah di-hash-verify;
**tidak** memanggil simulator, detector, atau GPU):

```bash
export PYTHONPATH=src:/home/pannntastic/aivnv/duckie/src
/home/pannntastic/aivnv/duckie/.venv/bin/python \
  experiments/evaluate_f9c_robust_belief.py \
  --config configs/f9c_robust_belief_v1.toml --replay-from-cache
```

Reproduksi ablation tujuh-baris (juga replay-only, invariant I4):

```bash
export PYTHONPATH=src:/home/pannntastic/aivnv/duckie/src
/home/pannntastic/aivnv/duckie/.venv/bin/python \
  experiments/evaluate_f9c_robust_belief.py \
  --config configs/f9c_robust_belief_v1.toml --ablation
```

Jalankan suite penuh termasuk leakage tests F9c:

```bash
export PYTHONPATH=src:/home/pannntastic/aivnv/duckie/src
export DUCKIETOWN_HEADLESS=1
/home/pannntastic/aivnv/duckie/.venv/bin/python -m pytest tests -q
# 251 passed, 0 failed, 0 skipped
```

Headline: Robust B menurunkan range RMSE 21.5% (`0.02580 -> 0.02024 m`) dan
bearing RMSE 14.8% (`0.01590 -> 0.01356 rad`) dibanding Baseline A (F9b
frozen-bias path, tidak dimodifikasi); mempertahankan active belief pada
61.8% (34/55) genuine in-domain miss dibanding Baseline A 18.2% (10/55); dan
mempertahankan active belief pada 100% (23/23) `gated_rejection` frame,
bukti langsung invariant I2 (gate rejection adalah DETEKSI, bukan miss).
Namun range coverage_68/95 (`0.852/0.988`) dan bearing coverage_68 (`0.851`)
overshoot pita pre-registered `[0.60,0.76]`/`[0.90,0.98]` pada sisi
konservatif; hipotesis heavy-tail untuk overshoot ini sudah diuji dan
DITOLAK (kurtosis negatif). Dari 17 kriteria PASS yang pre-specified: 12
met, 3 not met (tiga pita coverage di atas), 2 insufficient evidence
(checkpoint 20-consecutive-miss dan pengurangan outlier impact). Rekomendasi
klasifikasi: **LIMITED** (bukan keputusan final — manusia yang menetapkan
pita akseptansi memutuskan). Detail lengkap, semua 8 pertanyaan eksplisit,
dan alasan klasifikasi ada di `IMPLEMENTATION_NOTES.md` bagian
"F9c gate report".

Artefak:

```text
artifacts/f9c_frozen_config.json
artifacts/f9c_calibration.csv
artifacts/f9c_calibration_metrics.json
artifacts/f9c_runtime_cache.npz
artifacts/f9c_evaluation_truth.npz
artifacts/f9c_validation.csv
artifacts/f9c_belief_metrics.json
artifacts/f9c_nis_metrics.json
artifacts/f9c_ablation_metrics.json
artifacts/f9c_error_cases/
```

## F9d — control-readiness evidence closure

F9d does not change the F9c estimator. It freezes F9c, then closes two
evidence gaps with natural localization-outlier stress and long-absence
stress. Frozen F9d config SHA256:
`7bbe6525c24e294b55a46808301249633236658814e906a68d0d804d5e8a8ca6`.

Read-only verification (safe; no simulator, YOLO, or EKF execution):

```bash
export PYTHONPATH=src:/home/pannntastic/aivnv/duckie/src
/home/pannntastic/aivnv/duckie/.venv/bin/python \
  experiments/verify_f9d_artifacts.py --final
```

This returns 15 PASS, 0 FAIL, 0 SKIP. Final seeds `8201-8204` and
`8301-8304` have already been rendered exactly once and must not be rendered
again. The final outcome is `LIMITED`: long-absence evidence passed, while
the natural-outlier run produced 43 frames against a minimum of 50 despite
favourable descriptive Robust-B accuracy. See
`docs/superpowers/F9D_REPORT_FOR_REVIEW.md`.

Final active regression suite: **351 passed, 0 failed, 0 skipped**. The
264 emitted warnings are dependency deprecations/runtime warnings, not test
failures.

Artifacts:

```text
artifacts/f9d_frozen_config.json
artifacts/f9d_association_diagnostic.json
artifacts/f9d_yield_probe.json
artifacts/f9d_absence_yield_probe.json
artifacts/f9d_outlier_stress.csv
artifacts/f9d_outlier_metrics.json
artifacts/f9d_outlier_runtime_cache.npz
artifacts/f9d_outlier_evaluation_truth.npz
artifacts/f9d_absence_stress.csv
artifacts/f9d_absence_metrics.json
```

## YOLO-to-belief evidence video

`experiments/render_yolo_belief_video.py` renders a demo-only, disjoint seed
through the real runtime chain:

```text
front RGB -> frozen YOLO11n -> metric projection -> frozen F9c robust EKF -> belief
```

The video overlay shows YOLO boxes/bottom-center points, associated metric
measurement, gate decision/NIS, belief mean and uncertainty, rates, and
existence probability. Magenta ground truth is read only after the belief
update and is labelled `GT EVAL ONLY`; it never enters detector, association,
or EKF inputs.

```bash
export PYGLET_HEADLESS=true DUCKIETOWN_HEADLESS=1 LIBGL_ALWAYS_SOFTWARE=1
export PYTHONPATH=src:/home/pannntastic/aivnv/duckie/src
/home/pannntastic/aivnv/duckie/.venv/bin/python \
  experiments/render_yolo_belief_video.py
```

Generated evidence:

```text
artifacts/yolo_belief_demo.mp4
artifacts/yolo_belief_demo.json
artifacts/yolo_belief_demo_preview.png
```

## F10-L1 — `small_loop` counter-clockwise lane curriculum

F10-L1 is a separate first driving curriculum after the full F10 policy was
classified LIMITED. It trains SAC only to complete the native `small_loop`
counter-clockwise while remaining in the right lane, avoiding the yellow
center line, and staying on the road. Its six-dimensional observation uses
agent-visible lane/motion state; it does not use YOLO/EKF and is not a full
POMDP deployment checkpoint.

The selected step-50,000 checkpoint completed 4/4 untouched final laps with
zero invalid pose, yellow crossing, or lane departure. Mean absolute lateral
error was `0.01062 m`; mean episode p95 was `0.02502 m`. Training is available
at <https://wandb.ai/vnv/DuckiePOMDP/runs/z39mxtvl>.

Selected checkpoint and proof:

```text
artifacts/f10_l1/sac_lane_baseline.pt
artifacts/f10_l1/checkpoint_manifest.json
artifacts/f10_l1/final_metrics.json
artifacts/f10_l1/sac_lane_demo.mp4
artifacts/f10_l1/sac_lane_demo.json
docs/F10_L1_REPORT_FOR_REVIEW.md
```

Re-render the development-only proof video without touching the final split:

```bash
export PYGLET_HEADLESS=true DUCKIETOWN_HEADLESS=1
export PYTHONPATH=src:/home/pannntastic/aivnv/duckie/src
/home/pannntastic/aivnv/duckie/.venv/bin/python \
  experiments/render_f10_l1_lane_video.py --device cuda \
  --output artifacts/f10_l1/sac_lane_demo_rerender.mp4
```

## F10-L2 — `experiment_loop` mixed-turn transfer

F10-L2 warm-starts the selected F10-L1 checkpoint and transfers it to the
native `experiment_loop`, which contains both left and right turns. The frozen
40,000-step run is tracked at
<https://wandb.ai/vnv/DuckiePOMDP/runs/y0qu681q>.

Safety-first development selection chose transfer step 40,000. On the
once-only final seeds `18001-18004`, the selected checkpoint completed 4/4
laps with zero invalid pose, yellow crossing, or lane departure. Mean absolute
lateral error was `0.03413 m`, mean episode p95 was `0.05763 m`, and mean
actual velocity was `0.12980 m/s`.

```text
artifacts/f10_l2/sac_lane_transfer_baseline.pt
artifacts/f10_l2/checkpoint_manifest.json
artifacts/f10_l2/development_metrics.json
artifacts/f10_l2/final_metrics.json
artifacts/f10_l2/sac_lane_transfer_demo.mp4
artifacts/f10_l2/sac_lane_transfer_demo.json
docs/F10_L2_REPORT_FOR_REVIEW.md
```

Re-render the development-only proof without touching final seeds:

```bash
export PYGLET_HEADLESS=true DUCKIETOWN_HEADLESS=1
export PYTHONPATH=src:/home/pannntastic/aivnv/duckie/src
/home/pannntastic/aivnv/duckie/.venv/bin/python \
  experiments/render_f10_l2_lane_video.py --device cuda \
  --output artifacts/f10_l2/sac_lane_transfer_demo_rerender.mp4
```

F10-L2 is still a lane-only curriculum. YOLO, F9c belief, stop logic, and
pedestrian response are not part of this checkpoint.

Final F10-L2 regression suite: **419 passed, 0 failed, 0 skipped**.
