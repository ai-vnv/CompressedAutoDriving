# Formulasi Masalah — Version 1

Status: kontrak formulasi, F1–F5b geometry, F6 oracle observation, dan F7
pedestrian EKF sudah diimplementasikan. Detector, reward, dan solver tetap
berada di gate terpisah.

Dokumen rinci yang menjadi sumber formulasi adalah
`POMDP Problem Formulation Scaffold — Gym-Duckietown.md`.

## 1. Tujuan dan scope

Duckiebot harus mengikuti satu rute, berhenti di stop line yang sah, dan
memberi jalan kepada satu Duckie pedestrian. Policy tidak boleh membaca
ground-truth object dari simulator.

Version 1 mencakup:

```text
front RGB camera
one stop sign and one valid stop line
one relevant Duckie pedestrian
effectively observed ego/lane quantities
Cartesian temporal tracking
polar probabilistic belief
continuous chassis command
```

Multi-pedestrian association, side camera, end-to-end RGB policy, intent
prediction, dan full-city navigation tidak termasuk Version 1.

## 2. POMDP

```text
P = <S, A, T, R, Omega, O, gamma>
```

- `S`: true world state, hanya tersedia pada simulator/evaluation boundary.
- `A`: `PolicyAction(v_cmd, omega_cmd)`.
- `T`: simulator dynamics, stop-mode transition, pedestrian motion, dan
  perubahan object state ke ego frame.
- `R`: progress, lane, stop, pedestrian, comfort, dan collision components.
- `Omega`: camera, ego/road measurements, dan metric object measurements.
- `O`: detection/missed-detection serta range/bearing measurement model.
- `gamma`: `0.99`, disimpan di `configs/pomdp_v1.toml`.

## 3. True state

```text
POMDPState
├── EgoState
│   ├── lateral_error_m
│   ├── heading_error_rad
│   ├── linear_velocity_mps       # v_actual
│   └── yaw_rate_rad_s            # omega_actual
├── RoadState
│   ├── curvature_inv_m
│   ├── stop_line_distance_m      # signed distance to stopping point
│   └── stop_mode                 # NONE | REQUIRED | SATISFIED
├── StopSignState
│   └── exists, range_m, bearing_rad  # object-model origin
└── PedestrianState
    └── exists, range_m, bearing_rad, # object-model origin
        radial_velocity_mps, bearing_rate_rad_s
```

`range_m` untuk stop sign dan pedestrian secara kanonik adalah jarak dari
reference point ego ke simulator object-model origin. Nearest collision
footprint adalah quantity privileged terpisah untuk safety/evaluation dan tidak
menjadi alias `range_m`. Stop-sign range juga tetap berbeda dari
`stop_line_distance_m`, yaitu jarak ke lokasi berhenti.

Jika object tidak ada, true object kinematics menggunakan `None`. Kecepatan
pedestrian tidak pernah dimasukkan sebagai one-frame visual measurement.

## 4. Observation boundary

Agent-visible raw input:

```text
SensorObservation
├── front_rgb: uint8[H, W, 3]
├── EgoObservation
│   └── d, phi, v_actual, omega_actual
└── RoadMeasurement (optional pada F1 bila map tidak menyediakan route geometry)
    └── curvature, signed stop-line distance
```

Ego dan road quantities boleh diperlakukan effectively observed pada Version
1. Kelonggaran ini tidak berlaku untuk stop sign atau pedestrian state.

Object perception menghasilkan:

```text
Detection(class, confidence, bounding_box, bottom_center)
  -> ground-plane projection
  -> raw metric range/bearing
  -> fixed offline-fitted range calibration
  -> ObjectMeasurement(detected, confidence, x_left, y_forward, range, bearing)
```

Kalibrasi runtime hanya menerima raw range/bearing dan parameter tetap dari
`configs/measurement_model_v1.toml`. Privileged object pose/footprint tidak
pernah menjadi input runtime.

Missed detection menggunakan `None` untuk seluruh nilai metric dan confidence.
`range=0` atau `bearing=0` tidak boleh dipakai sebagai sentinel.

## 5. Observation model

Model awal difaktorkan sebagai:

```text
O = O_ego O_road O_sign O_ped
```

`O_sign` dan `O_ped` harus membedakan:

- detection probability `P_D(r, beta)`;
- false-positive probability `P_FA`;
- range noise;
- bearing noise;
- missing measurement.

Detector confidence adalah score detector, bukan calibrated existence
probability.

F5b mengunci target range sebagai object origin. Pada held-out real-rendered
samples, residual calibrated range mempunyai bias `0.00036 m`, MAE `0.00515 m`,
dan RMSE `0.00597 m`. Noise range dicatat per distance bin; bearing tetap
identity-calibrated dan noise empirisnya dicatat terpisah.

F6 mengimplementasikan approximation `O(o|s)` ini dalam tiga mode:

```text
oracle_clean   : exact canonical GT within observation domain
oracle_noisy   : F5b residual bias + diagonal Gaussian R(r)
oracle_dropout : oracle_noisy + synthetic Bernoulli misses
```

GT hanya masuk ke `OracleObservationModel`; keluarannya tetap
`ObjectMeasurement`. Bearing Gaussian dinyatakan provisional karena residual
held-out F5b skewed/heavy-tailed. Synthetic dropout tidak diklaim sebagai
performa YOLO.

## 6. Belief

Policy menerima `BeliefState`, bukan `POMDPState`:

```text
BeliefState
├── EgoObservation
├── RoadBelief(curvature, stop-line distance, stop mode)
├── StopSignBelief(P_exist, range mean/std, bearing mean/std)
└── PedestrianBelief(
      P_exist,
      range mean/std,
      bearing mean/std,
      radial-velocity mean/std,
      bearing-rate mean/std,
    )
```

Pedestrian tracker menggunakan Cartesian state internal:

```text
X = [x_left, y_forward, vx, vy]
```

`vx,vy` adalah physical pedestrian world velocity yang diekspresikan pada
orientasi ego saat ini, bukan apparent velocity akibat ego bergerak. Dengan
`A` sebagai rotasi old-ego axes ke new-ego axes dan `t_ego` sebagai actual ego
displacement pada old axes:

```text
p_new = A @ (p_old + v_ped * dt - t_ego)
v_new = A @ v_ped
```

Range-rate dan bearing-rate publik tetap relatif terhadap ego. Karena itu
transformasi polar menggabungkan posterior physical velocity dengan actual
ego translation dan yaw pada timestep tersebut.

Posterior kemudian dikonversi menjadi polar belief. Covariance harus
dipropagasi melalui Jacobian, bukan diganti standard deviation arbitrer.

F7 menggunakan EKF karena observation function polar non-linear:

```text
h(X) = [sqrt(x^2+y^2), atan2(x,y)]
```

Untuk bearing convention `atan2(x_left,y_forward)`, Jacobian bearing adalah
`[y/r^2, -x/r^2, 0, 0]`. Process Q Version 1 adalah diagonal dengan density
`0.001 m/sqrt(s)` untuk posisi dan `0.005 m/s/sqrt(s)` untuk velocity; satu
konfigurasi dipakai pada seluruh skenario.

Belief updater contract:

```text
update(
    previous_belief,
    previous_action,
    ego_motion,
    perception,
    dt_s,
)
```

`previous_action` adalah transition control input. Geometric compensation
memakai actual `EgoMotion(v_actual, omega_actual)`.

## 7. Action and actuation

```text
a_t = [v_cmd, omega_cmd]
```

```text
v_cmd     in [0.0, 0.4] m/s
omega_cmd in [-4.0, 4.0] rad/s
```

`omega_cmd > 0` berarti counter-clockwise. Reverse tidak diizinkan pada
Version 1.

```text
NormalizedPolicyAction
  -> NormalizedActionScaler
  -> PolicyAction(v_cmd, omega_cmd)
  -> DifferentialDriveActionAdapter
  -> WheelCommand(left duty, right duty)
  -> simulator wheel-duty boundary
```

Jangan mengirim hasil adapter kembali ke `DuckietownEnv.step(v, omega)` karena
akan menyebabkan double conversion.

State dan action berbeda:

```text
v_actual != v_cmd
omega_actual != omega_cmd
```

Angka envelope di atas adalah kandidat F2, belum batas permanen. Sweep nyata
dan rationale candidate tersedia di `IMPLEMENTATION_NOTES.md`.

## 8. Transition

Simulator menjadi generative ego transition. Pedestrian motion model awal
adalah constant velocity dengan process noise:

```text
x_next  = x + vx * dt + process_noise
y_next  = y + vy * dt + process_noise
vx_next = vx + process_noise
vy_next = vy + process_noise
```

Previous object belief harus ditransformasikan ke ego frame baru menggunakan
actual translation dan yaw sebelum measurement correction.

Missed measurement menjalankan prediction-only. Position/velocity covariance
bertambah melalui transition dan Q; measurement berikutnya mengontraksikan
posterior. Pedestrian existence probability difilter terpisah dari Gaussian
kinematic state dan tidak pernah disamakan dengan oracle/detector confidence.

## 9. Reward and episode semantics

Scaffold mendeklarasikan, tetapi belum menghitung:

```text
R = progress + lane + stop + pedestrian + comfort + collision
```

Setiap term harus dilog terpisah. Weight, stop dwell time, stop-speed
threshold, route completion, episode horizon, terminated conditions, dan
truncated conditions baru boleh dikunci pada reward/environment gate.

## 10. Privileged boundary

`PrivilegedSimulatorState` berada pada port terpisah dari `AgentEnvironment`.
Ia hanya boleh dipakai untuk label, calibration, oracle baseline, debugging,
dan evaluation. Object origin dan collision footprint tersedia sebagai field
yang berbeda agar semantic comparison tidak mencampur keduanya.

`SensorObservation`, `PerceptionObservation`, `BeliefState`, dan `Transition`
tidak mempunyai field privileged atau ground-truth world pose.

## 11. Coordinate conventions

- Simulator ground plane: `(x, z)` dalam meter.
- Ego ground frame: `(x_left, y_forward)` dalam meter.
- Bearing positif ke kiri.
- Yaw rate positif counter-clockwise.
- Bounding box: `(x_min, y_min, x_max, y_max)`.
- Ground-contact pixel: `((x_min+x_max)/2, y_max)`.

Projection, coordinate transform, oracle observation, dan tracker sudah
melewati gate F5/F6/F7. F9 menghubungkan detector nyata ke tracker melalui
`RGB -> YOLO -> raw projection -> fixed observation preprocessing -> frozen
EKF`. Privileged truth baru dibaca setelah kedua runtime update selesai.

F9 tidak mengubah state, transition, Jacobian, Q, atau public belief contract.
Hasil final menunjukkan range posterior overconfident dan track existence
cepat hilang pada natural misses. Karena itu chain ini valid sebagai measured
baseline, tetapi belum diklaim cukup terkalibrasi untuk reward/control. Robust
observation/filtering harus menjadi gate terpisah jika dipilih setelah review.
