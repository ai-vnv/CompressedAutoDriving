# Implementation Notes — F1 Environment + F2 Action Adapter

Status: F0 tetap utuh. F1/F2 diimplementasikan tanpa perception, belief
filter, reward shaping, atau solver.

## Audit Gym-Duckietown yang aktif

Environment lokal yang benar-benar dipakai saat integrasi:

```text
Python                  3.10.20
gym-duckietown          6.2.0 (source overlay)
Gym                     0.26.2
NumPy                   1.26.4
pyglet                  1.5.27
renderer                Mesa llvmpipe
map                     small_loop
seed                    73
```

Source simulator aktif berada di
`/home/pannntastic/aivnv/duckie/src/gym_duckietown`. Ini berbeda dari optional
dependency `duckietown-gym-daffy==6.1.34` di `pyproject.toml`; karena itu
perintah reproducibility menambahkan source overlay tersebut ke `PYTHONPATH`.

API aktual yang ditemukan:

```text
Simulator.reset(segment=False) -> np.ndarray
Simulator.seed(seed)            # reset tidak menerima seed keyword
DuckietownEnv.step([v, omega])  -> (obs, reward, done, info)
Simulator.step([left, right])   -> (obs, reward, done, info)
```

Meskipun Gym yang terpasang versi 0.26.2, simulator masih memakai API lama
empat elemen dan satu flag `done`. Adapter memetakan `max-steps-reached` menjadi
`truncated`; kode selesai lainnya menjadi `terminated`.

`DuckietownEnv.step` menerima chassis command lalu mengubahnya menjadi duty.
Base `Simulator.step` menerima left/right motor duty dan melakukan clipping
`[-1, 1]`. Karena proyek harus mencatat intermediate conversion dan saturation,
environment adapter memanggil base boundary dengan wheel duty. Mengirim wheel
duty ke `DuckietownEnv.step` akan mengonversi action dua kali.

Ego variables yang tersedia adalah `cur_pos`, `cur_angle`, `timestamp`, dan
`speed`. Actual `v` dan `omega` sengaja tidak diambil dari command maupun dari
`speed`: keduanya diturunkan secara konsisten dari dua pose dan timestamp
berturut-turut. Lane pose tersedia melalui `get_lane_pos2(pos, angle)` dengan
`LanePosition.dist` dan `LanePosition.angle_rad`.

Tidak ada integrasi Q-learning, SARSA, SAC, atau TD3 pada source aktif. Artefak
eksperimen lama hanya ada di `_archive/attempt_01/` dan tidak diubah.

## F1 — environment boundary

Implementasi `adapters/gym_duckietown.py` membuat satu session dengan tiga view:

```text
Gym-Duckietown session
├── GymDuckietownAgentEnvironment
│   └── SensorObservation(front_rgb, ego)
├── GymDuckietownPrivilegedStateSource
│   └── PrivilegedSimulatorState
└── GymDuckietownDiagnosticsSource
    └── conversion/actuator/experiment diagnostics
```

Agent view tidak mempunyai world pose, pedestrian truth, sign truth, atau
object list simulator. Reward pada `Transition` masih zero placeholder;
native simulator reward hanya tersedia pada diagnostics dan tidak dianggap
sebagai reward POMDP.

Actual motion dihitung sebagai:

```text
dt           = timestamp(t+1) - timestamp(t)
delta_yaw    = wrapped(heading(t+1) - heading(t))
v_actual     = pose displacement projected on midpoint heading / dt
omega_actual = delta_yaw / dt
```

Map `small_loop` tidak memiliki `sign_stop`, Duckie pedestrian, atau stop-line
route metadata. Field privileged terkait karena itu `None`, bukan nilai rekaan.
Ini mengungkap implementation conflict kecil terhadap scaffold F0 yang semula
membuat road/stop-line values wajib. Kontrak diubah menjadi optional; model
matematika dan pemisahan observasi/privileged tidak berubah. `collision` juga
optional karena `invalid-pose` tidak membuktikan collision.

## Coordinate dan sign convention

Satu source of truth ada di `domain/coordinates.py`:

```text
world ground plane       (x, z), meter
ego ground frame         (x_left, y_forward), meter
forward positive         sepanjang heading kendaraan
lateral positive         ke kiri kendaraan
yaw/omega positive       counter-clockwise
heading error positive   heading di kiri lane tangent
bearing positive         di kiri ego heading
angle                    radian
linear velocity          meter/second
yaw rate                 radian/second
```

Di representasi `(x,z)` simulator, unit forward vector untuk heading `h` adalah
`(cos(h), -sin(h))`. Integration test nyata memverifikasi command yaw positif
menghasilkan `omega_actual > 0`, dan command negatif menghasilkan kebalikannya.

## F2 — conversion dan saturation

Boundary tetap:

```text
PolicyAction(v_cmd [m/s], omega_cmd [rad/s])
  -> physical wheel angular velocities [rad/s]
  -> calibrated motor duty
  -> explicit clip to simulator limit
  -> WheelCommand(left duty, right duty)
  -> base Simulator.step
```

Untuk wheel separation `L` dan radius `R`:

```text
wheel_rate_left  = (v_cmd - 0.5 * L * omega_cmd) / R
wheel_rate_right = (v_cmd + 0.5 * L * omega_cmd) / R
```

`ActionConversion` menyimpan requested action, physical wheel rates,
unclipped left/right duty, final duty, flag per-wheel saturation, dan aggregate
`saturated`. Reverse dan non-finite action ditolak eksplisit.

## F2 actuator-envelope run

Perintah reproduksi:

```bash
export PYTHONPATH=src:/home/pannntastic/aivnv/duckie/src
export DUCKIETOWN_HEADLESS=1
/home/pannntastic/aivnv/duckie/.venv/bin/python \
  experiments/characterize_action_envelope.py
```

Run deterministik memakai `small_loop`, fixed straight-lane pose, seed 73,
30 Hz, 10 transient steps, dan 20 steady measurement steps. CSV lengkap ada di
`artifacts/action_envelope.csv`.

Ringkasan hasil nyata:

| `v_cmd` | `omega_cmd` | mean `v_actual` | mean `omega_actual` | left duty | right duty |
|---:|---:|---:|---:|---:|---:|
| 0.0 | 0.0 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| 0.2 | 0.0 | 0.1284 | 0.0000 | 0.2329 | 0.2329 |
| 0.4 | 0.0 | 0.2567 | 0.0000 | 0.4659 | 0.4659 |
| 0.5 | 0.0 | 0.3209 | 0.0000 | 0.5823 | 0.5823 |
| 0.2 | -4.0 | 0.1284 | -1.5506 | 0.4705 | -0.0047 |
| 0.2 | +4.0 | 0.1284 | +1.5506 | -0.0047 | 0.4705 |
| 0.4 | -4.0 | 0.2567 | -1.5506 | 0.7035 | 0.2283 |
| 0.4 | +4.0 | 0.2567 | +1.5506 | 0.2283 | 0.7035 |

Semua 14 kasus selesai tanpa clipping, off-road, termination, atau truncation.
Kandidat `v_max=0.4 m/s` dan `omega_max=4.0 rad/s` memiliki peak duty 0.7035,
lebih konservatif daripada linear probe 0.5 m/s. Nilai tersebut tetap berstatus
candidate, bukan batas permanen, sampai skenario kurva/lane-retention yang lebih
panjang tersedia.

## F3 — minimal POMDP scenario

Scenario deterministik dikonfigurasi oleh `configs/scenario_pomdp_v1.toml` dan
map `maps/pomdp_v1.yaml`:

```text
map                 pomdp_v1, 5 x 5 tiles, tile size 0.585 m
route               closed perimeter; straight approach + one T-junction tile
seed                123
ego spawn           world (0.6500, 2.7495), heading 0 rad / east
stop sign           world (1.3455, 2.89575)
stop line point      world (1.1700, 2.7495), normal to route heading 0 rad
pedestrian start     world (1.55025, 2.4219)
default mode         cross_left_to_right, 0.20 m/s, crossing length 0.80 m
```

Map memiliki tepat satu `sign_stop` dan satu dynamic `duckie`; adapter menolak
scenario bila count bukan satu. Mode pedestrian yang didukung adalah
`stationary`, `cross_left_to_right`, dan `cross_right_to_left`.

Gym-Duckietown 6.2.0 menghapus direktori dari absolute custom-map path sebelum
resource lookup. Project-local subclass hanya mengoreksi loader untuk file map
eksternal; built-in map path tetap memakai implementation upstream. Dynamic
Duckie upstream juga menambahkan `vel` sekali per physics step, bukan
`vel * dt`; adapter karena itu mengubah configured m/s menjadi meter/step
dengan frame rate simulator.

## F4 — true-state extraction

Semua state memakai unit formulasi: meter, radian, m/s, rad/s, dan curvature
`1/m`. Curvature dihitung dari first/second derivatives cubic Bezier lane
centerline pada closest route point. Pada straight validation segment,
`kappa=0.0 1/m`.

Stop-line distance menggunakan signed longitudinal projection:

```text
rho_stop = dot(stop_point - ego_position, route_forward_unit)
```

Jadi `rho_stop > 0` sebelum garis, `=0` pada garis, dan `<0` setelah melewati
garis. Ia tidak diturunkan dari range stop sign.

Pedestrian world velocity, `rdot`, dan `betadot` dihitung dari pose history
bertimestamp. Membaca privileged state berkali-kali tidak mengubah estimator.
Hasil controlled cases di `artifacts/state_validation.csv`:

| Case | Representative result |
|---|---|
| stationary ego + pedestrian | `v=0`, `omega=0`, `rdot=0`, `betadot=0` |
| ego forward, pedestrian stationary | `r: 0.9580 -> 0.8665 m`, final `rdot=-0.1283 m/s` |
| ego turns CCW, pedestrian stationary | `beta: 0.3490 -> 0.1982 rad`, final `betadot=-0.4004 rad/s` |
| pedestrian crosses left-to-right | `beta: +0.3490 -> -0.2247 rad` |
| stop-line crossing | `rho_stop: +0.5200 -> -0.0577 m` |

Initial manual geometry checks agree with the locked world-to-ego transform:

```text
sign: r=0.710710 m, beta=-0.207261 rad
ped:  r=0.958004 m, beta=+0.349003 rad
```

## F5 — calibrated ground-plane projection

`perception/camera_geometry.py` builds the same pinhole model used by the
simulator from exposed calibration:

```text
image                 640 x 480 px
vertical FOV          75 deg
camera height         0.108 m
camera pitch          19.15 deg
camera forward offset 0.066 m
distortion            disabled for this deterministic scenario
```

Runtime projection is strictly:

```text
pixel + CameraCalibration
  -> inverse view/projection ray
  -> y=0 ground intersection
  -> GroundPoint(x_left, y_forward)
  -> (range, bearing)
```

It has no object kind, object world pose, or privileged handle. Validation-only
pixels are obtained by rendering an object, hiding only that object, and taking
the bottom-center of the image-difference silhouette. Privileged pose is used
only after projection to calculate error.

Distance bins are `near < 0.55 m`, `medium [0.55, 0.80) m`, and
`far >= 0.80 m`. Horizontal FOV bins use normalized distance from principal
point: center `<1/3`, mid-FOV `<2/3`, and edge-FOV otherwise.

The original F5 run across 108 real rendered samples reported:

| Metric | MAE | RMSE |
|---|---:|---:|
| `x_left` | 0.0932 m | 0.1088 m |
| `y_forward` | 0.3253 m | 0.3363 m |
| range | 0.3419 m | 0.3534 m |
| bearing | 0.00670 rad | 0.00801 rad |

F5b showed this interpretation was incorrect. The replicated view matrix made
`gluLookAt` target ground height even though Gym-Duckietown targets the same
camera height and applies downward pitch separately. This introduced an extra
fictitious pitch. The pre-fix CSV/JSON are retained as
`ground_projection_*_pre_f5b.*`.

After fixing the extrinsic, the same 108-sample F5 run gives raw origin-range
MAE `0.01425 m`, RMSE `0.01698 m`, and bearing RMSE `0.00942 rad`.

## F5b — range semantics and measurement calibration

Every simulator object now exposes two separate privileged references:

```text
range_to_origin_m  = ego reference -> object model origin
range_to_surface_m = ego reference -> nearest collision-footprint boundary
```

The runtime projector receives neither reference. They are read only after
projection for offline comparison. Across 120 real-rendered samples:

| Raw target | Bias | MAE | RMSE | Residual SD |
|---|---:|---:|---:|---:|
| object origin | -0.01420 m | 0.01441 m | 0.01701 m | 0.00941 m |
| nearest footprint | +0.02680 m | 0.02680 m | 0.02855 m | 0.00986 m |

The corrected bottom-center projection therefore estimates model origin more
closely than nearest footprint. Version-1 canonical `range_m` remains
`object_origin`. Collision clearance can later combine origin range with
explicit footprint geometry; it must not silently redefine the state.

Calibration split is trajectory-level, never frame-random:

```text
fit (68): calibration_approach, turn_left, pedestrian_crossing
held out (52): straight_approach, turn_right
```

The offline least-squares correction is:

```text
r_cal = 0.9507847585432267 * r_raw + 0.05181745469768865
```

Held-out origin-range results improve from raw MAE/RMSE
`0.01498/0.01756 m` to calibrated `0.00515/0.00597 m`; calibrated bias is
`0.00036 m`, residual SD `0.00601 m`, skewness `-0.026`, and excess kurtosis
`-0.609`. Empirical range SD is `0.00436 m` near, `0.00556 m` medium, and
`0.00549 m` far using thresholds `0.55/0.80 m`. The near estimate has only six
held-out samples and is retained with its sample count rather than presented as
a high-confidence population value.

Held-out bearing is not corrected: bias `0.00221 rad`, MAE `0.00876 rad`,
RMSE `0.01234 rad`, and SD `0.01226 rad`. Its skewness `-2.05` and excess
kurtosis `5.29` do not support a strong Gaussian claim. Range/bearing residual
correlation is `-0.172` (covariance `-1.269e-05 m rad`), so Version 1 exports a
diagonal covariance while recording the empirical correlation for later audit.

All fixed runtime and future F6/F7 noise parameters are in
`configs/measurement_model_v1.toml`. Privileged GT is used only by the F5b
experiment and never by `MeasurementCalibrator`.

## F6/F7 implementation plan (pre-implementation gate)

Scope is limited to an oracle measurement boundary and a pedestrian EKF. No
detector, learned policy, reward, or multi-object association is introduced.

F6 will use this dependency direction:

```text
PrivilegedSimulatorState
  -> OracleObservationModel (the privileged boundary ends here)
  -> ObjectMeasurement(range, bearing, detected)
  -> BeliefUpdater
```

`oracle_clean` emits exact origin range/bearing. `oracle_noisy` samples the
held-out calibrated residual model from `configs/measurement_model_v1.toml`.
`oracle_dropout` adds explicitly synthetic Bernoulli misses. GT range is not
passed through the raw pixel-range calibration again. Oracle positives use
confidence `1.0` only as a source label, never as existence probability.

F7 keeps the internal posterior over:

```text
X = [x_left, y_forward, v_left, v_forward]
```

Here velocity is the pedestrian's physical world velocity expressed in the
current ego-oriented axes. It is not apparent ego-relative velocity. For a
constant measured ego twist over `dt`, let `A(delta_yaw)` rotate old ego-axis
components into the new ego axes and let `t_ego` be the ego displacement in
the old axes. Prediction is:

```text
p_new = A @ (p_old + velocity_old * dt - t_ego)
v_new = A @ velocity_old
```

The corresponding affine state transition has `A` in the position and
velocity blocks and `dt*A` in the position/velocity cross block. Translation
and rotation use `EgoMotion` derived from simulator pose history. The previous
`PolicyAction` remains present only to preserve the formal POMDP update
contract; it is not substituted for actual motion.

The polar observation is `h(X)=[sqrt(x^2+y^2), atan2(x,y)]`. The unusual
project bearing convention makes the bearing Jacobian row
`[y/r^2, -x/r^2, 0, 0]`; analytical and numerical Jacobians must agree before
the F7 experiment is accepted. Public polar uncertainty will be propagated
from the Cartesian covariance, not selected independently.

Implementation gates:

1. F6 unit/statistical checks and a real-simulator truth trajectory must pass
   before F7 experiment results are accepted.
2. F7 deterministic ego-motion, Jacobian, miss/re-observation, and existence
   tests must pass before running the scenario matrix.
3. One common process-noise setting is selected from a small sensitivity grid;
   it is not tuned per scenario.
4. F6/F7 are marked PASSED only after reproducible CSV/JSON artifacts and the
   full suite pass in the documented local simulator environment.

## F6 — oracle observation results

`perception/oracle_measurement.py` is the only component that accepts
`PrivilegedSimulatorState`. It emits the existing `ObjectMeasurement`; neither
the updater nor public belief imports privileged types. Oracle range starts
from canonical calibrated GT and adds final F5b residuals, so the raw camera
calibration is not applied a second time.

Real Gym-Duckietown truth was sampled at near, medium, and far start poses.
Monte Carlo output has 30,024 rows in
`artifacts/oracle_measurement_validation.csv`:

| Quantity | Configured | Empirical |
|---|---:|---:|
| near range bias | 0.007282 m | 0.007255 m |
| near range SD | 0.004355 m | 0.004398 m |
| medium range bias | -0.001704 m | -0.001673 m |
| medium range SD | 0.005557 m | 0.005398 m |
| far range bias | 0.000971 m | 0.000860 m |
| far range SD | 0.005488 m | 0.005457 m |
| bearing bias | 0.002211 rad | 0.002275 rad |
| bearing SD | 0.012261 rad | 0.012303 rad |
| miss probability | 0.200 | 0.201 |

Clean max absolute range error is zero and bearing numerical error is
`5.6e-17 rad`. Dropout and false-positive parameters are explicitly synthetic;
false positives remain disabled because the single-object contract has no
unambiguous false-object geometry.

## F7 — pedestrian EKF results

One simulator-data collection supplies six controlled trajectories:

```text
stationary pedestrian / stationary ego
stationary pedestrian / moving ego
stationary pedestrian / turning ego
left-to-right crossing
right-to-left crossing
crossing / moving and turning ego
```

The prior right-to-left implementation conflict was exposed here: it reversed
heading but retained the left endpoint as spawn. The adapter now spawns that
mode at the opposite crossing endpoint. A real-simulator regression verifies
negative-to-positive bearing traversal.

The 3x3 shared Q sweep uses the same replayed real trajectories for every
candidate. Version 1 selected:

```text
position process std = 0.001 m/sqrt(s)
velocity process std = 0.005 m/s/sqrt(s)
```

This is one global choice, never a trajectory-specific fit. It reduced both
RMSE and excessive conservatism relative to the initial `0.005/0.050` model.
Initial velocity uncertainty remains `0.35 m/s`, so zero velocity is not
treated as known during initialization.

Across all six trajectories:

| Mode | Observation RMSE r / beta | EKF RMSE r / beta | EKF RMSE rdot / betadot |
|---|---:|---:|---:|
| clean | 0 / ~0 | 0.000002 m / 0.000001 rad | 0.000589 m/s / 0.000590 rad/s |
| noisy | 0.005802 m / 0.012467 rad | 0.002356 m / 0.004926 rad | 0.022521 m/s / 0.038050 rad/s |
| dropout | 0.005459 m / 0.013111 rad | 0.002738 m / 0.005686 rad | 0.020031 m/s / 0.057579 rad/s |

The dropout replay has 448 initialized posterior rows, 356 detections, and an
empirical miss fraction `0.205`. Noisy 68/95 percent coverage is
`0.694/0.958` for range and `0.718/0.971` for bearing. Dropout coverage is
`0.748/0.982` and `0.757/0.975`. Rate intervals remain conservative:
dropout 68-percent coverage is `0.866` for both rates and 95-percent coverage
is `0.998/0.987`. This is reported as remaining calibration headroom, not as a
Gaussian-perfect result.

Existence probability is independent of confidence. Under uninterrupted
detections it rises from `0.976` after the first hit toward one. Under dropout
the global mean is `0.982`; the minimum `0.147` occurs after a rare miss run in
the right-to-left episode and recovers when measurements return.

Ego compensation uses only `v_actual,omega_actual`. In clean controlled runs,
stationary-pedestrian physical velocity RMSE is numerical zero for stationary,
moving, and turning ego. Thus no clean-mode ego/pedestrian motion confusion was
observed. Under noisy/dropout modes, physical velocity error is measurement-
driven; it does not systematically increase for moving/turning ego versus the
stationary-ego dropout realization. Full results, marginal NLL, predicted SD,
coverage, scenario breakdowns, and Q sweep are in
`artifacts/belief_calibration_metrics.json`.

## F8a readiness audit — historical blocker

F8 begins with a hard detector-asset gate. The audit performed on 2026-08-07
covered the active repository and the sibling projects
`/home/pannntastic/aivnv/duckie` and
`/home/pannntastic/aivnv/handson-duckie`. Model, dataset, and label-map searches
excluded virtual environments, package caches, Git metadata, and
`node_modules`.

The audit found no YOLO checkpoint, ONNX/engine export, labeled detection
dataset, or class-name mapping suitable for both required classes
`stop_sign` and `duckie`. All local `.pt` files inspected by location and
surrounding project metadata are RL policy checkpoints or DINO feature/policy
checkpoints, not object detectors. The archived F0 attempt also records the
same prerequisite explicitly: YOLO requires weights and a labeled/evaluation
dataset.

The active simulator Python environment reports:

```text
Python       3.10.20
PyTorch      2.12.1+cu130
CUDA         available, one device
GPU          NVIDIA GeForce RTX 4060 Laptop GPU
Ultralytics  not installed
```

At the time of the audit the GPU had `4717 / 8188 MiB` allocated, so it was
not considered free for an experiment under the local compute policy. This is
secondary to the missing checkpoint: installing Ultralytics alone would not
produce a detector capable of recognizing Duckietown-specific objects.

The repository already has framework-independent `Detection`, `BoundingBox`,
and `ObjectClass` contracts plus the `ObjectDetector` port. The readiness
probe was run exactly against the documented expected path and returned:

```json
{
  "ultralytics_installed": false,
  "weights": "weights/detector.pt",
  "weights_exist": false,
  "ready_for_inference_implementation": false
}
```

Accordingly, at the time of that audit no generic COCO checkpoint was
substituted, no package or model was downloaded, no training was started, and
no F8 measurement artifact was fabricated. The missing prerequisites recorded
there are resolved by F8-prep below. F6/F7 source, configuration, and artifacts
remain unchanged.

## F8-prep — simulator dataset and baseline YOLO

The historical blocker above was removed by generating an auditable detector
dataset from real Gym-Duckietown 6.2.0 frames. The canonical detector mapping
is defined once in `domain/detection.py` and is exactly:

```text
0 = stop_sign
1 = duckie
```

Annotation uses an object-specific difference between visible and temporarily
hidden simulator RGB renders. This produces a tight box around the pixels the
network actually sees. It is an offline privileged annotation path; the
runtime `ObjectDetector.detect(rgb)` signature accepts only a uint8 RGB image.
Native simulator segmentation was not used because the current 6.2.0
`render_obs(segment=True)` object-mesh path raises an unhashable-list error.
The same visible/hidden rendering method already validated during F5 was
therefore reused instead of creating an unrelated bbox approximation.

The first visual QA exposed that the map's stop sign showed its rear face at
the original 90-degree rotation. A deterministic rotation sweep showed the
complete red STOP face at 180 degrees. `maps/pomdp_v1.yaml` now uses 180
degrees, the dataset was regenerated, and a real-render regression requires a
substantial red-pixel population inside the stop-sign bbox.

Visibility requires at least 24 changed pixels and a 5x5-pixel bbox. At most
one image border may be touched. Truncated stop signs require 100 pixels of
height and truncated Duckies require 30 pixels. Capture occurs at most every
six simulator steps and requires 0.03 m ego translation, limiting temporal
duplicates. The split unit is an episode identified by disjoint seed and
pedestrian mode; no frame, seed, or episode is shared across train, validation,
and test.

Dataset `duckietown_detection_v1` contains:

| Split | Images | Stop boxes | Duckie boxes | Negative frames |
|---|---:|---:|---:|---:|
| train | 471 | 166 | 252 | 215 |
| validation | 168 | 59 | 85 | 81 |
| test | 164 | 62 | 84 | 80 |

Across all splits there are 287 stop-sign and 421 Duckie boxes. Duckie boxes
cover 165 near, 153 medium, and 103 far opportunities; stop-sign boxes cover
173 near and 114 medium opportunities. The fixed semantically valid sign
placement does not produce a far-bin stop-sign sample, which is a known V1
dataset limitation rather than a hidden extrapolation claim. FOV coverage is
`center/mid/edge = 150/177/94` for Duckie and `79/152/56` for stop sign.
Duckie trajectory counts are 134 stationary, 136 left-to-right, and 151
right-to-left. Automated QA found zero split leakage, zero duplicate image
hashes across splits, and produced 18 deterministic box overlays.

The training environment is specified in `configs/yolo_env_v1.json`:
Ultralytics 8.4.116, Torch 2.12.1+cu130, Torchvision 0.27.1+cu130, and an RTX
4060 Laptop GPU. YOLO11n was initialized from generic pretrained weights and
fine-tuned only on the two project classes. This is transfer initialization,
not substitution of generic COCO predictions. One seed-8123 run used 40
epochs, 480-pixel input, batch 8, AdamW selected by the trainer, and the
conservative augmentations in `configs/yolo_train_v1.toml`.

Development-only validation metrics are precision `0.96088`, recall
`0.98774`, mAP50 `0.99243`, and mAP50-95 `0.94575`. These are validation
diagnostics, not final F8a test claims. The stable checkpoint is
`artifacts/yolo_v1/best.pt`, SHA256
`3d4f816d440690493b856d25403a84a3249e4250599319c32569b97cb8d7482c`.
Its internal mapping is exactly `0=stop_sign,1=duckie`. A limited post-selection
sanity check detected each target in all eight selected untouched-test examples
per class, and the independent readiness probe loaded the checkpoint and ran
front-RGB inference successfully. Full test suite: 89 passed, 0 failed, 0
skipped. F5b/F6/F7 implementation and calibration parameters remain frozen.

At the end of F8-prep, F8a and F8b were READY, not PASSED. Full detector
performance and metric range/bearing characterization were intentionally not
started during that earlier gate; the final evaluation is documented below.

## F8a — frozen-test detector evaluation

The checkpoint and split remained frozen. Before inference, the evaluator
verified SHA256
`3d4f816d440690493b856d25403a84a3249e4250599319c32569b97cb8d7482c`,
the dataset-manifest hash recorded by model provenance, and checkpoint mapping
`0=stop_sign,1=duckie`. The pre-specified operating point was confidence
`0.10`, NMS IoU `0.70`, match IoU `0.50`, `imgsz=480`, `max_det=300`, CUDA
device 0. Ultralytics' standard PR curve used a `0.001` score floor only to
compute AP; it did not change the fixed operating-point counts. All 164 image
inferences completed before privileged `objects.csv` was read. GT therefore
participated only in offline matching.

Final held-out results are:

| Class | Visible N | TP | FP | FN | Precision | Recall | F1 | mAP50 | mAP50-95 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| stop_sign | 62 | 62 | 8 | 0 | 0.88571 | 1.00000 | 0.93939 | 0.99388 | 0.95539 |
| duckie | 84 | 84 | 3 | 0 | 0.96552 | 1.00000 | 0.98246 | 0.99488 | 0.91309 |

Global mAP50 is `0.99438` and mAP50-95 is `0.93424`. Supported range/FOV
strata all had recall `1.0`: stop sign near/medium `40/22` and
center/mid/edge `18/34/10`; Duckie near/medium/far `31/34/19` and
center/mid/edge `36/31/17`. There is no far stop-sign sample, so its result is
explicitly `N/A`, not extrapolated.

There are 102 stop-sign-negative frames: seven false boxes occurred in six,
giving `0.06863` FP per negative frame and event probability `0.05882`.
Duckie has zero false boxes in 80 negative frames. The three total Duckie FPs
and one remaining stop-sign FP are duplicate/unmatched detections on positive
frames. Visual audit shows several stop-sign negative-frame FPs are genuine
severely truncated sign pixels that failed the frozen dataset eligibility
threshold, rather than a hallucinated sign. They remain counted; no difficult
case was removed. The corrected 180-degree map orientation shows the front
red STOP face in final audit images, so the historical reversed-sign defect is
not present in this evaluation.

Mean IoU is `0.94301` for stop signs and `0.94203` for Duckies. Median IoU is
`0.96945/0.95897`; mean bottom-center error is `2.357/1.023 px` respectively.
Confidence is associated with localization in pixel space: correlation with
IoU is `0.897/0.630`, and with absolute bottom-center error is
`-0.815/-0.579`. TP mean confidence is `0.917/0.922`; FP mean confidence is
`0.414/0.495`, although individual high-confidence truncated/duplicate cases
exist.

## F8b — YOLO metric measurement characterization

The one runtime path is `RGB -> frozen YOLO -> bbox bottom-center -> existing
CalibratedGroundProjector -> (x_left,y_forward) -> (range,bearing)`. Projection
does not accept object GT. All 146 matched boxes projected successfully. Range
continues to mean object-model-origin distance, and bearing remains positive
left with radians.

| Group | N | Raw range bias / MAE / RMSE / SD (m) | F5b range bias / MAE / RMSE / SD (m) | Bearing bias / MAE / RMSE / SD (rad) |
|---|---:|---:|---:|---:|
| all | 146 | 0.10414 / 0.10680 / 0.17269 / 0.13822 | 0.12245 / 0.12245 / 0.17494 / 0.12538 | -0.00688 / 0.04222 / 0.05348 / 0.05322 |
| stop_sign | 62 | 0.05824 / 0.06416 / 0.08989 / 0.06903 | 0.08311 / 0.08311 / 0.10252 / 0.06051 | -0.02200 / 0.04591 / 0.06069 / 0.05702 |
| duckie | 84 | 0.13803 / 0.13827 / 0.21416 / 0.16474 | 0.15148 / 0.15148 / 0.21316 / 0.15087 | 0.00428 / 0.03950 / 0.04747 / 0.04756 |

Raw conditional range RMSE by GT range is near `0.04208 m` (`N=71`), medium
`0.16262 m` (`N=56`), and far `0.38024 m` (`N=19`, Duckie only). Bearing RMSE
is `0.06164/0.04314/0.04794 rad`. By FOV, raw range RMSE is
center/mid/edge `0.14580/0.21770/0.06806 m`; bearing RMSE is
`0.02208/0.05521/0.08459 rad`, with counts `54/65/27`. Detection probability
must be read alongside these conditional errors; the frozen split happened to
have no misses in supported strata.

The accurately-localized F5b baseline is range RMSE `0.005967 m` and bearing
RMSE `0.012343 rad`. Raw YOLO projection is therefore `28.94x` worse in range
and `4.33x` worse in bearing. Applying the frozen F5b linear calibration to
YOLO boxes increases global range RMSE by `1.31%`, from `0.17269` to
`0.17494 m`. No YOLO-specific correction was fitted on test. The separate F9
candidate consequently selects raw projected range, records its measured
bias, and leaves F5b calibration disabled.

The candidate global measurement parameters are range bias `+0.10414 m`,
`sigma_r=0.13822 m`, bearing bias `-0.00688 rad`, and
`sigma_beta=0.05322 rad`. Its diagonal covariance is
`R=diag(0.0191051 m^2, 0.00283261 rad^2)`. Class/range-specific candidates and
their sample counts are exported because range error is strongly
distance-dependent. Residual range-bearing correlation is only `0.06057`
with covariance `0.0004042 m.rad`; this does not justify an off-diagonal term.

Global raw-range residuals are a poor Gaussian approximation (skew `2.289`,
excess kurtosis `5.539`); bearing is a reasonable approximation (skew
`0.362`, excess kurtosis `0.177`). The deterministic outliers show that a
small vertical bottom-point error near the horizon can cause a large range
error despite high IoU and confidence. Confidence-to-absolute-error
correlation is `+0.058` for raw range and `-0.246` for bearing. Thus confidence
does describe bbox quality, but does not yet justify confidence-conditioned
metric covariance; range and geometry bin are more informative.

For this split, bbox localization/projection sensitivity is the dominant
scientific failure: there were no misses, while false positives affect the
discrete observation path but cannot explain the `28.94x` conditional range
degradation. F8b exports a candidate only. F6 oracle behavior, F7 Q,
initialization, existence filter, ego-motion compensation, and covariance
transforms remain unchanged; YOLO has not been connected to the EKF.
The final full repository suite reports 94 passed, 0 failed, and 0 skipped.

## F9a/F9b — frozen YOLO measurement to frozen EKF

F9 uses three non-overlapping data domains. Detector train/validation/test
seeds remain `1101--1106`, `2101--2102`, and `3101--3102`. Measurement
calibration uses `4101--4104`; final belief evaluation uses `5101--5104`.
`load_f9_protocol(..., require_frozen=True)` rejects overlapping seed sets,
checkpoint hash changes, F7 EKF changes, unfrozen parameters, or a calibration
artifact hash mismatch.

The first pre-freeze calibration audit exposed only one class-negative frame,
which made the Beta-prior `P_FA` estimate meaningless. A second audit showed
that a reversed-camera negative scenario was not representative. Neither run
opened final seeds. The accepted protocol instead measures false alarms from
the same simulator scenes rendered through the privileged validation port
with the Duckie hidden. Detector inference still has signature `detect(rgb)`;
the counterfactual render method is absent from the agent environment.

The accepted calibration artifact contains 1,904 rows: 1,264 natural-scene
opportunities and 640 controlled negative RGB frames. There are 1,221 eligible
visible opportunities and 1,193 IoU>=0.5 selected measurements. Matched
support is 42 near, 239 medium, 912 far and 435/629/129 center/mid/edge. The
frozen additive model is:

```text
r_corrected    = r_raw - (-0.0459048047 m)
beta_corrected = wrap(beta_raw - 0.0041456789 rad)
```

After bias removal, calibration residual SD is `0.015812 m` globally and
`0.012648 rad` for bearing. The predeclared heteroscedastic rule selected
range bins because all bins have at least 30 samples and max/min sigma ratio
is `5.270`: near/medium/far sigma is
`0.0030476/0.0124911/0.0160619 m`. Bearing remains one global sigma. The
range-bearing correlation `-0.2897` is recorded but the pre-specified V1
covariance stays diagonal. Range and especially bearing retain heavy tails;
the Gaussian model is explicitly provisional.

Calibration-only detection estimates use Jeffreys Beta(0.5,0.5) posterior
means: `P_D=0.9766776` from 1193/1221 and `P_FA=0.0007800` from 0/640. These
replace only observation parameters in the frozen existence algorithm.
Prior, survival, birth, EKF Q, initialization, ego transform, Jacobians, and
polar covariance propagation remain byte-for-byte equivalent to F7; the F7
config SHA256 remains
`a4815c8d0e17f1868d51619ae51d2183c72832a022edce88aa3c10302594d701`.

The final run processes six scenarios for each of four unseen seeds, 2,172
frames total. Runtime order is strict: image inference, single-Duckie
highest-confidence selection, raw projection, fixed bias preprocessing, and
both EKF updates complete before privileged state/silhouette evaluation. Stop
sign detections are logged but never sent to the pedestrian EKF. Raw and
corrected filters see identical frames, actions, actual ego motion, misses,
and frozen Q/R except for the fixed additive preprocessing.

Global final results:

| Path | range bias / MAE / RMSE (m) | bearing bias / MAE / RMSE (rad) |
|---|---:|---:|
| current raw YOLO | -0.02662 / 0.02664 / 0.03430 | 0.00392 / 0.00633 / 0.00769 |
| raw YOLO -> EKF | -0.02770 / 0.02778 / 0.03478 | 0.00385 / 0.00719 / 0.00959 |
| current corrected YOLO | 0.01929 / 0.02284 / 0.02898 | -0.00023 / 0.00502 / 0.00662 |
| corrected YOLO -> EKF | 0.01879 / 0.02317 / 0.02808 | -0.00026 / 0.00564 / 0.00871 |

The calibration-only correction materially reduces range RMSE but does not
generalize to zero bias: it flips final bias from `-0.02662` to `+0.01929 m`.
Temporal filtering improves corrected range by only 3.1% versus the corrected
current frame, and it worsens bearing by 31.5%. Rate RMSE is `0.02591 m/s`
and `0.03695 rad/s` for the corrected EKF.

Corrected posterior coverage (68/95%) is range `0.152/0.258`, bearing
`0.598/0.785`, range-rate `0.668/0.859`, and bearing-rate `0.805/0.915`.
Range is severely overconfident. NIS has mean `1.843`, median `0.0703`, P95
`6.614`, and `5.42%` above the 2D chi-square 95% threshold. The superficially
reasonable exceedance fraction coexists with poor marginal coverage because
residuals are temporally correlated and contain localized heavy-tail bursts.
The maximum corrected NIS is `310.49`.

There are 57 natural misses in visible frames (recall `0.9737`), organized in
mean/max runs `7.125/10` frames. Active existence belief survives only 8/57
missed frames; re-detection restores it in one frame. Five frames naturally
leave the observation domain, where existence decays from `0.823` to
approximately `0.00012`. There are no class-negative false alarms or false
track initializations in final data.

Highest-confidence selection sees 78 multiplicity events. Their corrected
measurement/belief range RMSE is `0.04998/0.04653 m`, compared with
`0.02786/0.02716 m` on single-or-missed frames. Sixteen visible detections
fall below IoU 0.5; measurement range RMSE there is `0.15106 m`, belief RMSE
`0.05607 m`, and NIS P95 `282.03`. The filter remains numerically stable, but
these bursts are the exact empirical motivation for a future robust
observation/gating review. Confidence correlation is `-0.319` with absolute
range error, `-0.049` with absolute bearing error, and `-0.469` with NIS; this
is useful diagnostic evidence but not enough to introduce post-hoc
confidence-conditioned R in F9.

No final stratum contains near-range truth (`N=0`); medium/far counts are
319/1853. This limitation is reported rather than extrapolated. Corrected
range RMSE is `0.02227/0.02897 m` for medium/far. Edge-FOV corrected range and
bearing RMSE is `0.03572 m/0.01696 rad` (`N=82`), worse than center/mid.

F9a is PASSED. F9b is LIMITED: the real chain, accuracy evaluation, NIS,
misses, and leakage gates are complete, but interval calibration and miss
continuity are not adequate for an unqualified move to decision/control. No
robust gating, filter redesign, reward, or RL was implemented.

The final repository suite reports 108 passed, 0 failed, and 0 skipped. The
read-only F9 verifier also passes all seed, frame-matrix, checkpoint,
calibration, baseline-hash, structural-miss, and error-overlay checks.

## F9c

F9c makes the F9b pedestrian-belief estimator robust and well-calibrated. Task
1 builds only the guard rails — `configs/f9c_robust_belief_v1.toml` and
`src/duckie_pomdp/evaluation/f9c_protocol.py` — so that every later F9c task
runs inside a loader that makes it impossible to accidentally edit frozen F7
physics or read frozen earlier-split data. No estimator code was written in
this task; `robust_observation.*` switches and `covariance_calibration` /
`measurement_model` / `conditional_detection` parameters remain unfit
(`parameters_frozen = false`) placeholders for Tasks 2-10.

**Freeze-boundary table** — what `f9c_protocol._validate` enforces on every
load of `configs/f9c_robust_belief_v1.toml`:

| Frozen from F7 (`oracle_ekf_v1.toml`) — must match exactly | Unfrozen for F9c |
| --- | --- |
| `[ekf]` block (process noise, initial velocity std, minimum range, covariance floor, clean sigmas) | `measurement_model.*` (range/bearing bias) |
| `existence.prior_probability` | `covariance_calibration.*` (range/bearing scale, posterior floors) |
| `existence.survival_probability` | `conditional_detection.*` (per-FOV-class detection probability, false-positive rate) |
| `existence.birth_probability` | `existence.detection_probability` — **intentionally** unfrozen; F9c's conditional-detection work supersedes the single scalar P_D inherited from F9. This is a deliberate omission, not an oversight, and is called out with an inline comment in both the config and `_validate`. |

A config that edits any left-column value fails to load with a `ValueError`
containing the literal substring `frozen F7` (EKF block) or naming the
specific existence key (prior/survival/birth probability).

**Seed allocation** — written before any 7101-series frame was rendered:

- Calibration seeds: `6101-6108` (8 seeds, up from F9's 4, to shrink the bias
  standard error that broke F9b's transfer — see the Task 1 planning note on
  `SE(b̂) ≈ τ̂/√n_seeds`).
- Final-evaluation seeds: `7101-7104` (4 seeds).
- Forbidden seeds (every earlier split, enforced disjoint by `_validate`):
  `1101-1106`/`2101-2102`/`3101-3102` (YOLO detector train/val/test, read from
  `artifacts/detection_dataset_v1_manifest.json`), `4101-4104`/`5101-5104`
  (F9/F9b calibration/final-evaluation seeds).

**Pre-specified acceptance bands** (`[acceptance]` in the config, loaded into
`AcceptanceBands`), written before any 7101-series frame was rendered and
never to be adjusted after seeing final-evaluation results:

- Marginal 68% coverage in `[0.60, 0.76]`; 95% coverage in `[0.90, 0.98]`.
- `max_std_over_rmse = 1.5` (posterior std may not exceed 1.5x the empirical
  RMSE — the other direction of the F9b overconfidence problem).
- `max_rmse_ratio_vs_baseline = 1.15` (F9c may not be worse than Baseline A by
  more than 15% RMSE).
- Minimum support per stratum (`[minimum_support]`): near ≥100, medium ≥200,
  far ≥200, edge-FOV ≥50 samples, so no acceptance claim rests on an
  underpowered stratum.

**Tests:** `tests/test_f9c_protocol.py` (6 tests, all passing) checks seed
disjointness against every earlier split, frozen-`[ekf]` equality with F7,
frozen survival/birth equality, rejection of an edited
`position_process_std_m_per_sqrt_s`, the pre-specified acceptance bands, and
that all five `robust_observation` ablation switches default to enabled. Full
repository suite: 114 passed, 0 failed, 0 skipped (108 pre-existing + 6 new).

### Task 2 — final-evaluation near-range scenarios

F9b's final stratum had `N=0` near-range truth because the only near-range
placements (`calibration_near_stationary`, `calibration_medium_stationary`,
`ego_start_x_offset_m` 0.50/0.25) were flagged `use_for_final_evaluation =
false`. Task 2 replaces those two calibration-only entries in
`configs/f9c_robust_belief_v1.toml` with four scenarios usable by **both**
calibration and final evaluation:

| Scenario | pedestrian_mode | linear_velocity_mps | steps | ego_start_x_offset_m |
| --- | --- | ---: | ---: | ---: |
| `approach_near_stationary_ego` | stationary | 0.0 | 60 | 0.50 |
| `approach_medium_stationary_ego` | stationary | 0.0 | 60 | 0.25 |
| `approach_near_moving_ego` | stationary | 0.20 | 90 | 0.30 (traverses into near) |
| `cross_near_left_to_right` | cross_left_to_right | 0.0 | 110 | 0.40 |

The six original F9b scenarios (`stationary_ped_stationary_ego`,
`stationary_ped_moving_ego`, `stationary_ped_turning_ego`,
`cross_left_to_right`, `cross_right_to_left`,
`crossing_moving_turning_ego`) are unchanged byte-for-byte; they remain
Baseline A's control trajectories.

**Resolved ambiguity between the brief's test and its config (ruling
recorded):** running the brief's own Step 1 test verbatim against the
brief's own Step 3 TOML initially appeared contradictory, because
`approach_near_moving_ego`'s `ego_start_x_offset_m = 0.30` falls under a
naive `>= 0.35` near-range filter. The human ruling is that the **test's
membership predicate**, not the config value, was wrong: 0.30 is
deliberate — the scenario starts at the medium bin and drives in at
0.20 m/s for 90 steps specifically so it *traverses* into near range ("sweeps
range continuously downward, which is what makes the near bin a traversed
regime rather than a single static pose"). Classifying near-range membership
by start offset alone is the wrong criterion for a scenario whose entire
purpose is to move through the bin; raising the start offset to compensate
would have shortened the traversal and reduced the medium-range frames the
scenario contributes.

The config keeps `ego_start_x_offset_m = 0.30` for `approach_near_moving_ego`
(reverted, no inline comment). The test's near-range predicate was replaced
with a `reaches_near(spec)` helper that counts a scenario as near either by
`ego_start_x_offset_m >= 0.35` (starts close) or by `linear_velocity_mps >
0.0 and ego_start_x_offset_m >= 0.25` (starts at medium and drives in). All
other scenario values are unchanged from the brief.

**Step 5 dry-run** (arithmetic only, no simulator/render invoked, per the
Task 2 assignment's explicit deferral):

```python
from pathlib import Path
from duckie_pomdp.evaluation.f9c_protocol import load_f9c_protocol
protocol = load_f9c_protocol(Path("configs/f9c_robust_belief_v1.toml"))
final = [s for s in protocol.scenarios if s.use_for_final_evaluation]
print(sum(s.steps + 1 for s in final) * len(protocol.final_evaluation_seeds), "final frames")
```

Result: **3468 final frames** (10 final-evaluation scenarios x 4 seeds
`7101-7104`, `sum(steps+1)` = 867 frames/seed). Restricting to the scenarios
that reach near range under the same predicate the test uses —
`ego_start_x_offset_m >= 0.35`, or `linear_velocity_mps > 0.0 and
ego_start_x_offset_m >= 0.25` for a scenario that traverses in
(`approach_near_stationary_ego`, `approach_near_moving_ego` at 0.30 m
traversing, `cross_near_left_to_right`) gives 1052 frames across those 4
seeds (263/seed of near-reaching placement frames), well above the
proportional `minimum_support.near / len(final_evaluation_seeds) = 100 / 4 =
25`-per-seed floor. This is a scenario-count proxy, not an actual
`distance_bin` count — real per-frame near/medium/far classification depends
on simulated ego/pedestrian trajectories and the true `range_m` at each
frame, including the continuously-decreasing range in
`approach_near_moving_ego`, which cannot be computed without rendering.
Actual `distance_bin` verification against this dry-run proxy is deferred to
Task 9, the first task that renders calibration seeds (`6101-6108`) through
the (not-yet-existing) Task 11 collector; per the Task 2 assignment, no
simulator run or frame render was attempted in this task, and none of
`7101-7104` was touched.

`tests/test_f9c_protocol.py` gained one test,
`test_f9c_scenario_matrix_supports_near_range_final_evaluation`, verifying
`>=2` near-range-reaching final-evaluation scenarios (a scenario reaches near
range either by starting close, `ego_start_x_offset_m >= 0.35`, or by
starting at medium range and driving in,
`linear_velocity_mps > 0.0 and ego_start_x_offset_m >= 0.25`), `>=2`
near-range-reaching calibration scenarios, and at least one near, final,
positive-velocity (approaching) scenario. Full repository suite: 115 passed,
0 failed, 0 skipped (114 pre-existing + 1 new).

### Task 10 — freeze the configuration

Point of no return for gate F9c. Task 9's **final** calibration re-run
(`artifacts/f9c_calibration_metrics.json` — the fixed-scenario, floored,
false-positive-corrected re-run; NOT the crashed-scenario/pre-floor numbers
in task-9-report.md's earlier sections) was transcribed programmatically
(via `repr()` on values read straight from the JSON, never by hand) into
`configs/f9c_robust_belief_v1.toml`'s `[measurement_model]`,
`[covariance_calibration]`, and `[conditional_detection]` sections, and all
three `parameters_frozen` flags were set `true`:

| Section.key | Frozen value |
| --- | ---: |
| `measurement_model.bias_model` | `"global_additive"` |
| `measurement_model.range_bias_m` | `-0.02986607430110723` |
| `measurement_model.bearing_bias_rad` | `0.0012336629252072933` |
| `covariance_calibration.range_scale` (λ_r) | `9.96243043243885` |
| `covariance_calibration.bearing_scale` (λ_β) | `1.0` |
| `covariance_calibration.range_posterior_floor_m` | `0.02041790926900693` |
| `covariance_calibration.bearing_posterior_floor_rad` | `0.012546331734068323` |
| `conditional_detection.detection_probability_center` | `0.9490486257928118` |
| `conditional_detection.detection_probability_mid_fov` | `0.9801336146272855` |
| `conditional_detection.detection_probability_edge_fov` | `0.997211155378486` |
| `conditional_detection.detection_probability_outside_domain` | `0.5586734693877551` |
| `conditional_detection.false_positive_probability` | `0.00078003120124805` (unchanged; already F9b's frozen value) |
| `conditional_detection.miss_likelihood_floor` | `0.37362469458201386` |

Every value was re-read from the frozen config and compared bit-for-bit
(`==`, not `pytest.approx`) against the calibration artifact; all 13 match
exactly. **`lambda_r` is confirmed `9.96243043243885`, not the `10.125`
first-pass number** that appears in `task-9-report.md`'s pre-floor,
crashed-scenario section.

**Frozen config SHA256: `359dc52020421c248bf6c26e036234191b2b97d24b505a66d85daa85b563704e`.**

`artifacts/f9c_frozen_config.json` records `config_sha256`,
`checkpoint_sha256`, `calibration_artifact_sha256`, `frozen_f7_config_sha256`,
`calibration_seeds`, `final_evaluation_seeds`, the full fitted-parameter set,
the pre-specified acceptance bands and minimum-support floors, an ISO
timestamp, and the literal `"final_evaluation_seeds_not_yet_rendered": true`.
`load_f9c_protocol(..., require_frozen=True)` now loads without raising and
reports the same hash.

**Freeze did not perturb frozen F7 physics**: `[ekf]` and the three frozen
`[existence]` keys (`prior_probability`, `survival_probability`,
`birth_probability`) remain byte-identical to `configs/oracle_ekf_v1.toml`.
**Invariant I7** still holds: association gate `13.815510557964274` >
innovation gate `9.21034037197618`.

**Carried-forward review item closed**: `CovarianceCalibration.__post_init__`
(`src/duckie_pomdp/belief/covariance_calibration.py`) now rejects a negative
`range_posterior_floor_m` or `bearing_posterior_floor_rad` with a
`ValueError` explaining that a negative floor would *shrink* reported
uncertainty in `floor_polar_standard_deviation`'s quadrature sum instead of
inflating it — closing the hole Task 6's review flagged, at the point this
task first writes floor values into config by hand. Two new tests cover
both fields.

**Two pre-existing tests were updated in lockstep with the freeze** (not
added; their assertions targeted the pre-freeze placeholder values that no
longer exist once `parameters_frozen = true`):
`test_f9c_protocol.py::test_load_robust_observation_config_builds_a_coordinator_config`
(`range_scale` `1.0` → `9.96243043243885`) and
`test_f9c_covariance_calibration.py::test_load_miss_likelihood_floor_reads_the_frozen_fitted_value`
(renamed from `..._defaults_to_a_no_op`; `0.0` → `0.37362469458201386`).

**`experiments/verify_f9c_artifacts.py`** (new, modeled on
`experiments/verify_f9_artifacts.py`) is a read-only verifier with two
operating modes:

- Runs *now* (Task 10 state): confirms the frozen config loads under
  `require_frozen=True`, cross-checks all 4 hashes named in
  `f9c_frozen_config.json`, re-derives all 13 fitted parameters against
  `f9c_calibration_metrics.json` bit-for-bit, confirms `[ekf]`/existence
  byte-identity with F7, invariant I7, and that the F5b/F6/F7 upstream
  frozen-baseline hashes are unchanged.
- Degrades gracefully for Task 11's not-yet-existing artifacts
  (`f9c_validation.csv`, `f9c_belief_metrics.json`, `f9c_nis_metrics.json`,
  `f9c_error_cases/`): each dependent check reports `SKIP` with an explicit
  "Task 11 has not produced this yet" message rather than crashing. Once
  Task 11 exists, the same script re-derives seed/scenario/frame-matrix
  completeness, config-hash consistency, miss-row geometry, and (on a
  schema-tolerant best-effort basis, since Task 11's exact CSV column names
  do not exist yet to pin down) range RMSE/coverage directly from the CSV,
  cross-checked against the belief-metrics JSON's own reported numbers. A
  genuinely unmatched schema reports `SKIP` (a distinct `SchemaSkip`
  exception, never conflated with `FAIL`); a genuine numeric mismatch
  reports `FAIL` and a non-zero exit. Verified against synthetic
  Task-11-shaped fixtures (created and deleted within this task, never
  touching seeds 7101-7104) exercising both the all-present-but-differently-named-columns
  path (SKIP, not FAIL, exit 0) and an injected hash mismatch (FAIL, exit
  non-zero).

Full repository suite: **209 passed, 0 failed, 0 skipped** (207 pre-existing
+ 2 new: the two negative-floor validation tests; the two pre-existing tests
above were modified in place, not added).

**From this point `configs/f9c_robust_belief_v1.toml` is read-only** until
F9c reports. Files changed: `configs/f9c_robust_belief_v1.toml`,
`src/duckie_pomdp/belief/covariance_calibration.py`,
`src/duckie_pomdp/evaluation/f9c_calibration.py` (docstring only),
`tests/test_f9c_covariance_calibration.py`, `tests/test_f9c_protocol.py`.
Files created: `artifacts/f9c_frozen_config.json`,
`experiments/verify_f9c_artifacts.py`.

### Task 9 addendum — calibration seeds vs. the plan's F9a-derived predictions

The ledger records that this comparison was supposed to land here and instead
only reached `task-9-report.md`. Folded in now, verbatim from the **final**
(fixed-scenario, floored, false-positive-corrected) calibration re-run —
never the crashed-scenario first pass:

| Quantity | Predicted from F9a (k=4 seeds) | Observed on 6101–6108 (k=8 seeds, final run) | Assessment |
|---|---|---|---|
| Bias model | not predicted | `global_additive` (per-bin LOSO **−0.89%**, far below the +10% bar) | Clean, unambiguous |
| `b_r` | F9b frozen: −0.0459 m | **−0.02986607 m** | Different scenario/seed mix; expected to differ |
| `b_β` | F9a: +0.00415 rad | **+0.00123366 rad** | Small either way |
| τ̂_seed,range vs τ̂_episode,range | seed materially larger | **0.018416 m vs 0.005906 m (3.12x)** | Structural prediction **CONFIRMED** on 8 independent seeds |
| σ̂_w,range | ≈0.0074 m | **0.008084 m** | 9% above; close |
| σ_floor,r | 0.015–0.018 m | **0.020418 m** | 13% above the top of the band |
| τ̂_episode,bearing vs τ̂_seed,bearing | episode materially larger | **0.011139 rad vs 0.005311 rad (2.10x)** | Structural prediction **CONFIRMED** |
| σ̂_w,bearing | ≈0.0046 rad | **0.009178 rad** | ~100% above — the largest deviation in the table; attributed to a broader calibration scenario mix (turning-ego, moving-crossing) that F9a's narrower set never exercised |
| σ_floor,β | 0.012–0.016 rad | **0.012546 rad** | Inside the band |
| λ_r | 3–8 | **9.96243043243885** | 25% above the top of the band |
| λ_β | not separately predicted | **1.0** (no inflation needed) | Base bearing R already sufficient |
| P_D^eff(EDGE_FOV) vs P_D^eff(CENTER) | EDGE_FOV materially below CENTER | EDGE_FOV **0.9972** vs CENTER **0.9490** — inverted | Structure inverts; traced to a range confound (CENTER frames are 78% far-range, EDGE_FOV frames skew near/medium) rather than a genuine edge-of-frame detection advantage; within any one distance bin, detection rate is flat (≥0.946) across FOV position |

**The two structural predictions that motivated the nested (seed → episode\|seed)
variance estimator both replicate cleanly on 8 independent calibration
seeds**: range offset is seed-carried, bearing offset is episode-carried.
Every magnitude deviation above is reported, not adjusted toward the
F9a-derived band — per the plan's own instruction not to retune a fit to
match a prediction.

### Task 11 — final evaluation on seeds 7101–7104 (headline)

**The render happened exactly once, 2026-08-09**: 40/40 episodes (4 seeds x
10 final-evaluation scenarios), 3,328 frames, zero crashes, zero early
terminations. Post-render metrics code crashed on a `_optional()`-CSV-empty-string
bug that no synthetic test exercised (see the Task 11 report); the human
partner ruled that the missing artifacts be reconstructed by **replaying**
the already-written, hash-verified runtime cache/evaluation-truth pair
through a refactored shared row-builder — never by re-rendering. Both
reconstruction runs reproduced byte-identical output. Seeds 7101–7104 were
never rendered a second time.

- **Runtime-cache SHA256**: `fe425c55aadd45af88d072c256010f5bddcbb82d952669e8fa988bd70722526d`
- **Evaluation-truth SHA256**: `26663ebb85ebd2ed9fd00ffc0903679b298b9333ef3e10f58ff253d7fd1e2ae9`
- **Frozen config SHA256** (reconfirmed unchanged before and after every fix
  round): `359dc52020421c248bf6c26e036234191b2b97d24b505a66d85daa85b563704e`

**Support check (first thing checked, before any accuracy metric, per the
plan's Step 5):**

| Bin | Count | Minimum | Margin |
|---|---:|---:|---:|
| near | 616 | 100 | 6.2x |
| medium | 671 | 200 | 3.4x |
| far | 1887 | 200 | 9.4x |
| edge_fov | 543 | 50 | 10.9x |

`support_check.satisfied == true`. `CONTROL_READY` is not excluded on this
basis alone.

**Baseline A vs Robust B, full headline table:**

| Metric | Baseline A | Robust B | Band / guard | In band? |
|---|---:|---:|---|---|
| Range bias (signed, m) | +0.016263 | +0.001531 | — | Robust B ~10.6x closer to zero |
| Range MAE (m) | 0.019645 | 0.017292 | — | |
| Range RMSE (m) | 0.025796 | 0.020242 | ≤1.15x Baseline (0.029665) | **met**, ratio 0.785 |
| Bearing bias (signed, rad) | −0.002203 | +0.000395 | — | |
| Bearing MAE (rad) | 0.009278 | 0.008500 | — | |
| Bearing RMSE (rad) | 0.015904 | 0.013556 | ≤1.15x Baseline (0.018290) | **met**, ratio 0.852 |
| Range-rate RMSE (m/s) | 0.018436 | 0.019561 | — (Q frozen from F7; not targeted) | Robust B slightly *worse* |
| Bearing-rate RMSE (rad/s) | 0.037537 | 0.038648 | — (same) | Robust B slightly *worse* |
| Range coverage_68 | 0.2470 | 0.8522 | [0.60, 0.76] | **not met** (overshoots) |
| Range coverage_95 | 0.3881 | 0.9885 | [0.90, 0.98] | **not met** (0.0085 over top) |
| Bearing coverage_68 | 0.4536 | 0.8513 | [0.60, 0.76] | **not met** (overshoots) |
| Bearing coverage_95 | 0.6957 | 0.9403 | [0.90, 0.98] | **met** |
| Range coverage_error_68 / _95 | 0.433 / 0.562 | 0.172 / 0.038 | (informational) | Robust B closer on both |
| Bearing coverage_error_68 / _95 | 0.226 / 0.254 | 0.171 / 0.010 | (informational) | Robust B closer on both |
| Range mean marginal NLL | 31.085 | −2.439 | lower is better | Robust B far better |
| Bearing mean marginal NLL | 1.136 | −2.881 | lower is better | Robust B far better |
| Range mean_predicted_std (m) | 0.004933 | 0.025883 | — | |
| Range std_over_rmse | 0.191 | 1.279 | ≤1.5 | **met**, comfortably |
| Bearing std_over_rmse | 0.315 | 1.009 | ≤1.5 | **met**, comfortably |

**τ̂/σ̂_w and the posterior floors that follow from them** — see the Task 9
addendum table above; the frozen floors are `σ_floor,r = 0.02041790926900693 m`,
`σ_floor,β = 0.012546331734068323 rad`.

**Natural misses maintained / duplicate handling / outlier handling:**

- **Natural misses** (detector genuinely missed, pedestrian GT-visible):
  55 frames identical for both systems (same rendered frames, same detector
  inference). Baseline A retains an active belief on 10/55 (18.2%); Robust B
  on 34/55 (**61.8%**) — more than triples retention on the identical miss
  set. `in_domain_control_readiness.under_powered = false` (55 ≥ the 20-frame
  power floor).
- **Duplicate handling**: `duplicate_frames = 84` (frames where more than one
  raw candidate was present, i.e. `duplicate_selection = true`).
  `wrong_association_events = 2` (frames where temporal association picked a
  *different* candidate than highest-confidence selection would have, and
  that pick's GT IoU was below 0.5 — i.e. association's deviation from
  highest-confidence was itself wrong on 2 of the 84 duplicate frames).
- **Outlier handling** (`outlier_impact`, n=9 GT-labelled localization
  mismatches — `eligible_visible AND detector_detected AND NOT
  selected_correct_iou50`, independent of which system's gate accepted
  anything): raw measurement RMSE over these frames is 0.171 m. Baseline A
  belief RMSE 0.02179 m; **Robust B belief RMSE 0.03455 m — worse than
  Baseline A on this specific 9-frame subset.** See Finding 1 below; this is
  reported as insufficient evidence in the wrong direction, not explained
  away.

**Miss breakdown (invariant I2/I3, per-class, never pooled):**

| Class | Frames | Active belief retained | Retention |
|---|---:|---:|---:|
| `detector_miss_in_domain` | 55 | 34 | **61.8%** (primary control-readiness criterion) |
| `detector_miss_outside_domain` | 0 | 0 | n/a — no frames this run |
| `gated_rejection` | 23 | 23 | **100%** — invariant I2's payoff: under the pre-I2 design every one of these would have scored as an existence miss |

`gate_accept_reject`: 3,033 accepted / 23 rejected (99.2% accept rate).

**False tracks / deletions / recovery:**

- `false_track_initializations = 0` (both systems)
- `track_deletions = 8` (Robust B; Baseline A has no track-lifecycle concept
  and never un-initializes once corrected)
- `recoveries = 3` (re-initializations after a track previously existed and
  was lost, excluding each episode's first-ever init)
- `mean_frames_to_recover_after_redetection = 1.0` for **both** systems
  (≤2-frame recovery band, met)
- Natural miss-run checkpoints (Robust B, 11 genuine runs, mean/median
  length 5.0, max 10): at length 1, 90.9% still active, 100% eventual
  recovery; at length 5, 0% still "active" by the 0.5 threshold at that exact
  frame but 100% still recover (mean 1.0 frames after the miss ends); at
  length 10 (the single longest run), existence has decayed to 0.0095 and
  still recovers. **No run this evaluation reached the pre-specified 20-frame
  checkpoint** — the ≥20-consecutive-miss / P(e)<0.10 criterion has no data
  to evaluate against; see the classification section.

**NIS diagnostics, accepted vs rejected candidates, separately** (computed
directly from `f9c_validation.csv`'s `robust_b_gate_nis`/`robust_b_gate_decision`
columns for this report):

| Population | n | mean | median | min | max |
|---|---:|---:|---:|---:|---:|
| Accepted (Robust B) | 3,035 | 0.239 | 0.010 | — | 9.012 |
| Rejected (Robust B) | 23 | 11.309 | 10.703 | 9.482 | 13.643 |
| Baseline A (no gate) | 3,083 | 1.542 | 0.045 | — | — |

Every rejected NIS lies strictly between the innovation-gate threshold
(9.21034037197618) and the association gate (13.815510557964274) — exactly
invariant I7's design: association's looser gate lets these candidates
through to the innovation gate, which then correctly rejects them. Robust B
has **0.0%** of accepted-population frames above the chi-square(2) 95%
threshold, vs Baseline A's 3.3% (Baseline A has no gate at all).

**Predicted-observability vs GT FOV-region confusion, final seeds** (computed
fresh from `f9c_validation.csv`'s `robust_b_observability_class`/`fov_region`
columns for this report — not previously computed for the final seeds in any
prior task artifact; Task 9's confusion matrix covers the calibration seeds
only):

```
predicted \ GT     center   mid_fov   edge_fov   outside
center             1083     64        0          0
mid_fov            159      1247      84         11
edge_fov           0        50        369        79
outside_domain     3        25        90         64
```

Diagonal agreement: center 1083/1245 = 87.0%, mid_fov 1247/1386 = 90.0%,
edge_fov 369/543 = 68.0%, outside 64/154 = 41.6%. This is noticeably noisier
than the calibration-seed matrix (all >94%), especially for edge_fov and
outside. The most likely contributor is the disclosed camera-calibration
approximation in the reconstruction path (`replay_from_cache` classifies
observability using gym-duckietown's **nominal** camera constants, because
per-episode `domain_randomization` camera perturbation (cam_height ±8%,
cam_angle/cam_fov_y ±20%) was never captured in the runtime cache by design
and cannot be reconstructed from it) — this is a plausible explanation, not
a proven one; it was not tested directly against this specific breakdown. Per
Task 11's own finding, this classification feeds *only* Robust B's
existence-filter step, and `conditional_detection`'s per-class magnitudes are
inert (floored on the miss branch, saturated on the detected branch), so a
CENTER↔MID_FOV↔EDGE_FOV confusion has no behavioural consequence; only an
in-domain↔OUTSIDE_DOMAIN flip could matter, and outside-domain is exactly
where agreement is worst (41.6%) — consistent with, though not proof of, the
camera-approximation hypothesis.

### Task 12 — ablation (same runtime cache, zero inference)

**Runtime-cache SHA256, shared by the headline run and the ablation
(invariant I4):** `fe425c55aadd45af88d072c256010f5bddcbb82d952669e8fa988bd70722526d`
— re-verified by `_load_cache_and_truth` on every `run_ablation` call,
including against the hardcoded default expected values from Task 11.

| Row | Range RMSE (m) | coverage_68 | coverage_95 | In-domain retention |
|---|---:|---:|---:|---:|
| baseline (== Baseline A exactly) | 0.02580 | 0.2470 | 0.3881 | 0.1818 |
| + bias refit only | 0.02407 | 0.0752 | 0.2277 | 0.1818 |
| + innovation gate only | 0.02987 | 0.2588 | 0.4075 | 0.1818 |
| + temporal association only | 0.03776 | 0.2566 | 0.4063 | 0.1818 |
| + covariance calibration only | 0.02989 | 0.6403 | 0.9175 | 0.1818 |
| + conditional detection only | 0.02569 | 0.2589 | 0.4060 | 0.6182 |
| all combined (== Robust B exactly) | 0.02024 | 0.8522 | 0.9885 | 0.6182 |

**Headline scientific finding: the components are not additively
separable.** `innovation_gate_only` (0.02987) and `temporal_association_only`
(0.03776) are each *worse* than `baseline` (0.02580) on range RMSE, while
`all_combined` (0.02024) is the best row. Per-row deltas must not be read as
individual component contributions.

**Structural finding: the innovation gate is inert without temporal
association.** With `temporal_association` off, `update()` forces
`predicted_measurement=None`, association unconditionally returns
`mode="initialization"`, and the gate branch only runs when
`mode != "initialization"` — there is no innovation to threshold.
`innovation_gate_only` is metrically *identical* (`range.rmse =
0.029874031173970667` exactly) to an all-off frozen-threshold diagnostic
built specifically to isolate this. The gate can only ever be exercised in
combination with association, so a reader must not compare
`innovation_gate_only` directly against `baseline` to judge the gate alone.

**`conditional_detection`'s in-domain-retention jump (0.18 → 0.62) is real,
but comes from invariant I3's routing plus the I8 miss-likelihood floor, not
the fitted per-class detection probabilities** — see Finding 4 below.

### Task 13 — leakage tests, gate report, and classification

**Leakage tests** (`tests/test_f9c_leakage.py`, 6 tests). Covers the
task-list-mandated six runtime modules (the plan's original five plus
`bias_correction.py`, added since the runtime coordinator applies the frozen
F9c bias stage as a named runtime stage — Task 3b/Task 8):

```
src/duckie_pomdp/belief/innovation_gate.py
src/duckie_pomdp/belief/bias_correction.py
src/duckie_pomdp/belief/measurement_association.py
src/duckie_pomdp/belief/covariance_calibration.py
src/duckie_pomdp/belief/observability.py
src/duckie_pomdp/belief/robust_updater.py
```

A raw whole-file substring version of this scan tripped on three passages
on its first run — all of them comments/docstrings *documenting the absence*
of privileged access (the exact invariant the scan exists to check), not
actual code references. The first attempt at a fix reworded those passages
to dodge the substring match; that was a documentation regression, corrected
in fix round 1 (below) by restoring the original wording and rewriting the
scan itself to be AST-based — it inspects identifiers, attribute names,
import targets, and non-docstring string-literal values, and explicitly
skips docstrings, so comments (never present in the AST) and docstrings are
excluded by construction rather than by wording around them. A dedicated
test, `test_the_leakage_scan_reads_code_not_prose`, pins this: a synthetic
module whose only occurrence of a forbidden token is in a comment/docstring
must pass; a synthetic module with a genuine attribute reference (e.g.
`observation.privileged.read()`) must be caught. The evaluator-ordering test
(`test_the_evaluator_steps_both_beliefs_before_reading_privileged_truth`)
uses the same code-only principle: it locates the real `ast.Call` nodes for
`integration.privileged.read()`/`baseline_updater.update(...)`/
`robust_updater.update(...)` and compares their line numbers, rather than
doing a raw text `.index()` that a docstring quoting the call text could
satisfy first.

No import, type reference, GT column name, or IoU/silhouette computation
exists anywhere in the six runtime modules — every occurrence of a forbidden
token in these files is prose explaining the absence of exactly what the
scan checks for, and all such prose is intact in its original wording.

`test_f9b_frozen_artifacts_are_untouched` was verified against the **live**
file before being trusted: `sha256sum artifacts/f9_measurement_model.json`
independently reproduces
`eb09ea6c64b6cbf3306057092e254a0e049776b38581e5b873a8ef9e2e91b278` exactly —
the F9b artifact is untouched.

Full repository suite: **251 passed, 0 failed, 0 skipped** (245 + 6 leakage
tests).

---

## F9c gate report

**Reproduction:**

```bash
wsl.exe -d Ubuntu-Baru -- bash -lc 'cd /home/pannntastic/aivnv/duckie-pomdp && \
  export PYTHONPATH=src:/home/pannntastic/aivnv/duckie/src && \
  export DUCKIETOWN_HEADLESS=1 && \
  /home/pannntastic/aivnv/duckie/.venv/bin/python -m pytest tests -q'
# 251 passed, 0 failed, 0 skipped

wsl.exe -d Ubuntu-Baru -- bash -lc 'cd /home/pannntastic/aivnv/duckie-pomdp && \
  /home/pannntastic/aivnv/duckie/.venv/bin/python experiments/verify_f9c_artifacts.py'
# exit 0, {"PASS": 12, "SKIP": 1}
```

The calibration and final-evaluation renders themselves must **never** be
re-run: calibration seeds 6101–6108 were rendered once during Task 9 (the
final, internally-consistent re-run); final-evaluation seeds 7101–7104 were
rendered exactly once during Task 11 and reconstructed thereafter by
`--replay-from-cache`/`--ablation` only, both of which perform zero
inference and zero rendering (invariant I4).

**Headline numbers:**

```text
config_sha256      359dc52020421c248bf6c26e036234191b2b97d24b505a66d85daa85b563704e
calibration seeds  6101-6108 (6,656 rows, 80 episodes)
final seeds        7101-7104 (3,328 frames, 40/40 episodes, rendered EXACTLY ONCE)
runtime cache       fe425c55aadd45af88d072c256010f5bddcbb82d952669e8fa988bd70722526d
evaluation truth    26663ebb85ebd2ed9fd00ffc0903679b298b9333ef3e10f58ff253d7fd1e2ae9

gate            hard reject, chi-square 2-DOF 99% = 9.21034037197618
association     minimum-NIS, chi-square 99.9% = 13.815510557964274 (deliberately looser, invariant I7)
bias             global_additive; b_r = -0.02986607430110723, b_beta = 0.0012336629252072933
                 (per-bin LOSO improved only -0.89%, far below the +10% pre-specified bar)
covariance       lambda_r = 9.96243043243885, lambda_beta = 1.0
floors           sigma_floor_r = 0.02041790926900693, sigma_floor_beta = 0.012546331734068323,
                 from a nested (seed, episode) variance fit on 6101-6108; range offset is
                 seed-carried (variance 3.39e-4 vs 3.49e-5, ~9.7x), bearing offset is
                 episode-carried (variance 1.24e-4 vs 2.82e-5, ~4.4x)
miss floor       LR_floor = 0.37362469458201386 = LR_nominal ** (1/L_mean),
                 L_mean = 4.0333 measured on 6101-6108, LR_nominal = 0.018858
```

See the Task 9 addendum above for the full τ̂/σ̂_w variance-component table,
and the Task 11 section above for the complete Baseline A vs Robust B table,
miss breakdown, NIS diagnostics, and observability confusion matrix.

### Findings, stated plainly (Part 3)

**1. Outlier impact is INSUFFICIENT EVIDENCE, not a pass.** n=9 GT-labelled
localization-mismatch frames; Baseline A belief RMSE 0.02179 m vs Robust B
0.03455 m — **Robust B is worse** on that tiny sample. This touches the
outlier-impact PASS criterion directly. It is reported as insufficient
evidence in the wrong direction: n=9 is far too small to draw a general
conclusion either way, but the measured direction does not support a claim
that robust observation handling reduced outlier impact, and that criterion
is **not** counted as met below.

**2. Two coverage bands were missed, on the conservative side — and a third
is also outside once bearing is checked.** `coverage_68 = 0.852` (range)
against the pre-registered `[0.60, 0.76]` band, and `coverage_95 = 0.988`
(range) against `[0.90, 0.98]` — the pre-registered bands were **not met**.
Bearing coverage_68 (0.8513) is **also** outside its `[0.60, 0.76]` band;
bearing coverage_95 (0.9403) **is** inside `[0.90, 0.98]`. Separately,
`std_over_rmse` passes the ≤1.5 anti-inflation guard comfortably on both
axes (range 1.279, bearing 1.009), so the improvement was **not** bought by
absurd inflation. Both facts are stated; neither is hidden behind the other.

**3. The heavy-tail explanation for the coverage overshoot is refuted.**
Measured `z = (robust_b_belief_range_m - gt_range_m) / robust_b_belief_range_std_m`
over n=3,214 rows gives std ≈ 0.7899 and excess (Fisher) kurtosis ≈ −0.4266 —
slightly **lighter**-tailed than Gaussian, independently cross-checked
against `scipy.stats.kurtosis`. A heavy-tail explanation for the overshoot is
therefore refuted, not merely unsupported. The remaining hypothesis — that
`sigma_floor_r`/`sigma_floor_beta`, calibrated from `tau_seed`/`tau_episode`
estimated on 8 calibration seeds, transfer wide when applied unchanged to 4
different final-evaluation seeds — is consistent with the data (a fairly
uniform ~25–30% over-sized posterior sigma across the bulk of the
distribution) but is explicitly **untested by this run** and is labelled a
hypothesis, never a conclusion.

**4. `conditional_detection` is inert through its per-class probabilities,
on both branches.** Miss branch: the I8 floor (0.37362) dominates every
in-domain class's implied miss likelihood ratio (center 0.0510, mid_fov
0.0199, edge_fov 0.0028 — all far below the floor); OUTSIDE_DOMAIN never
reaches this branch at all (invariant I3). Detected branch: not floored, but
existence saturates ≥0.918 across all 3,122 detected rows regardless of
class (per-class minimums 0.969/0.918/0.9999/0.9986 for
center/mid_fov/edge_fov/outside_domain). The component therefore contributes
to Robust B's behaviour **only** via invariant I3's in-domain vs
OUTSIDE_DOMAIN routing — toggling it should not be read as testing the
fitted per-class magnitudes. Separately: 39/42 track initializations
classify OUTSIDE_DOMAIN, because the zero-prior default state
(`range_mean_m=0`, `bearing_mean_rad=0` → `y_forward=0`) fires
`PredictedObservabilityModel.classify`'s `y_forward <= 0.0` branch before any
real track exists.

**5. The components are not additively separable.** `innovation_gate_only`
(0.02987 m) and `temporal_association_only` (0.03776 m) are each worse on
range RMSE than `baseline` (0.02580 m), while `all_combined` (0.02024 m) is
the best of the seven ablation rows. Per-row deltas must not be read as
individual, additive component contributions.

**6. The innovation gate is structurally inert without temporal
association.** With `temporal_association` off, `update()` forces
`predicted_measurement=None`; `MeasurementAssociator.associate` then
unconditionally returns `mode="initialization"`, and the gate branch (steps
7–8 of `update()`) only ever runs when `mode != "initialization"`. There is
no innovation to threshold, so `innovation_gate_only` is metrically identical
(exactly, `range.rmse = 0.029874031173970667`) to a frozen-threshold, all-off
diagnostic built specifically to prove this. The gate can only be exercised
in combination with association.

**7. Reproducibility evidence is n=2 full calibration runs.** Both under
`domain_randomization = true`, agreeing to ~0.16% (`lambda_r` 9.977928850799799
→ 9.96243043243885 between the two back-to-back re-runs after the I8/scenario
fix round). The frozen values are one sample from a distribution, not a
deterministic constant — n=2 is evidence of short-run stability, not proof of
tight convergence.

**8. The P_FA scare.** A self-fit false-alarm rate of 31.2% was diagnosed as
an artefact, not a genuine false-positive problem: all 65 detector-flagged
frames among the 209 "not `eligible_visible`" frames carry a real GT range —
the pedestrian was physically present in every one, just outside the
conservative `eligible_visible` silhouette-visibility rule. These are correct
detections of a real pedestrian, not false alarms. The frozen F9b P_FA
(`0.00078003120124805`, measured properly from counterfactual renders with
the Duckie hidden — a genuine negative) was used for the recommended
`LR_floor` instead. Recorded here because the raw 31.2% figure looks
alarming out of context, and a future reader deserves the resolution rather
than re-discovering it.

**Triage of deferred minors from the ledger.** The ledger tags exactly
**11** items `minor (deferred)` across Tasks 1–12. All 11 are listed below,
in ledger (task) order, each with an explicit carry-forward/closed
disposition. None of them block the classification below.

| Item (task) | Disposition |
|---|---|
| `_scenario_spec` duplication in `f9c_protocol.py` (Task 1) | Defensible (F9b's `f9_protocol.py` is frozen); does not block |
| Brief's `load_scenario` reference vs actual `scenario_for` (Task 1) | Harmless naming mismatch in a doc reference; does not block |
| `from_config` silently accepts `model="identity"` with stray nonzero bias fields (Task 3b) | `identity()` is test-only and never appears in an ablation config; does not block |
| `AssociationConfig.initialization_rule` stored but never dispatched/validated (Task 4) | Inert field; a typo would silently no-op forever but nothing currently sets it to anything but the one supported value; does not block |
| `nis is not None` guard unreachable in the temporal branch (Task 4) | Dead defensiveness; does not block |
| `duckie_detections` re-filters by `ObjectClass.DUCKIE` independently of `select_single_duckie`'s own internal filter (Task 5) | **Deliberate carry-forward, not an oversight.** Introduced under Task 5's own constraint that the frozen Baseline-A selection path must not be rerouted through new code — the candidate-emitting loop needed its own class filter rather than reusing (and thereby coupling to) `select_single_duckie`'s internal one. No reported number in this gate report depends on it; does not block |
| Floor non-negative validation missing at definition time (Task 6) | **Closed** at Task 10 — validation now exists on `CovarianceCalibration.__post_init__`; not open |
| Two extra `ValueError` guards beyond the brief (Task 6) | Harmless; does not block |
| "Existence below `initialization_threshold`" reset branch untested (Task 8) | Test-coverage gap, hand-verified only; does not affect any frozen or reported number; does not block, but worth a future regression test |
| `RobustStepRecord.nis` is `None` on association-internal rejection even though `association.candidate_nis` exists (Task 8) | Diagnostic-field completeness gap only; does not affect any reported metric; does not block |
| Stale "expect 192 passed" template constant in the Task 12 brief vs actual 245 | Cosmetic; does not block |

**Four further disclosed limitations from the ledger** (not tagged `minor
(deferred)` — reported findings/concerns in their own right, included here
for completeness rather than counted toward the 11 above):

| Item (task) | Disposition |
|---|---|
| Joint 2-DOF NIS median at fit (0.505) far under the chi-square(2) target (1.386) (Task 9) | Disclosed V1 design limitation of a single pooled λ — the far bin (58% of the fitting set) dominates the fit; the plan specifies exactly one λ_r/λ_β, so this is not a defect to fix within F9c v1; does not block, but should inform any V2 design discussion |
| σ̂_w,bearing ~100% above the F9a-derived band (Task 9) | Disclosed, attributed to a broader calibration scenario mix; does not block |
| `innovation_gate_only` vs `baseline` table-reading trap (Task 12) | Addressed directly in Finding 6/this report by presenting the clean frozen-threshold-diagnostic comparison rather than the raw table row; does not block |
| Camera-calibration approximation in the cache-replay path (Task 11) | Scoped to Robust B's existence-filter observability classification only; per Finding 4, in-domain classes are inert to this misclassification, so only an in-domain↔OUTSIDE_DOMAIN flip could matter; plausible but unproven contributor to the noisier final-seed confusion matrix above; does not block |

### PASS-criteria classification (Part 4)

The plan's "Global Constraints" section pre-registers acceptance bands,
guards, an existence-retention criterion, a recovery bound, a false-track
bound, and per-stratum minimum support — 16 discrete numeric checks — plus
one further criterion named explicitly in this task's own brief (localization-
outlier impact reduction), for **17 pre-specified PASS criteria** in total.
No single verbatim numbered list of "17" exists in the plan text itself; this
numbering is this report's own enumeration of every quantitative
pre-registration in the plan's Global Constraints plus the brief's outlier
question, cited against its source line so it can be checked independently.

| # | Criterion (source) | Measured | Verdict |
|---|---|---|---|
| 1 | Range coverage_68 ∈ [0.60, 0.76] | 0.8522 | **NOT MET** — overshoots |
| 2 | Bearing coverage_68 ∈ [0.60, 0.76] | 0.8513 | **NOT MET** — overshoots |
| 3 | Range coverage_95 ∈ [0.90, 0.98] | 0.9885 | **NOT MET** — 0.0085 over the top |
| 4 | Bearing coverage_95 ∈ [0.90, 0.98] | 0.9403 | **MET** |
| 5 | Range anti-inflation guard: std_over_rmse ≤ 1.5 | 1.279 | **MET** |
| 6 | Bearing anti-inflation guard: std_over_rmse ≤ 1.5 | 1.009 | **MET** |
| 7 | Range accuracy guard: Robust RMSE ≤ 1.15x Baseline | 0.020242 ≤ 0.029665 | **MET** (ratio 0.785) |
| 8 | Bearing accuracy guard: Robust RMSE ≤ 1.15x Baseline | 0.013556 ≤ 0.018290 | **MET** (ratio 0.852) |
| 9 | Existence primary: `detector_miss_in_domain` retention ≥ 0.60 | 0.618 (34/55) | **MET** — margin is narrow (1.8 points) but the criterion is a clean ≥, and 55 frames clears the 20-frame power floor |
| 10 | Existence secondary: after ≥20 consecutive predicted in-domain misses, P(e) < 0.10 | longest genuine run this evaluation = 10 frames | **INSUFFICIENT EVIDENCE** — never reached; the run that got closest (length 10) decayed existence to 0.0095, well under 0.10, but that is 10 frames of evidence toward a 20-frame criterion, not a test of it |
| 11 | Recovery: mean frames to reactivate after re-detection ≤ 2 | 1.0 (both systems) | **MET** |
| 12 | False tracks: `false_track_initializations` ≤ 1 | 0 | **MET** |
| 13 | Minimum support, near ≥ 100 | 616 | **MET** (6.2x) |
| 14 | Minimum support, medium ≥ 200 | 671 | **MET** (3.4x) |
| 15 | Minimum support, far ≥ 200 | 1887 | **MET** (9.4x) |
| 16 | Minimum support, edge_fov ≥ 50 | 543 | **MET** (10.9x) |
| 17 | Outlier-impact reduction (Task 13 brief, "did robust observation handling reduce localization-outlier impact?") | n=9, Baseline 0.02179 m vs Robust 0.03455 m | **INSUFFICIENT EVIDENCE, adverse direction** — not counted as met; n is far too small to be a settled conclusion either way, but the measured direction does not support the claim |

**Tally: 12 MET, 3 NOT MET, 2 INSUFFICIENT EVIDENCE** (criteria 1, 2, 3 not
met; criteria 10, 17 unproven either way).

**Near-range gate (plan-mandated, checked separately from the 17 above):**
near-range support is 616 ≥ 100, so `CONTROL_READY` is **not** excluded on
this basis. This does not by itself make the gate `CONTROL_READY` — it only
removes the one condition the plan says makes `CONTROL_READY` unavailable
regardless of every other metric.

### The eight questions (Task 11's brief, answered against the final data)

1. **Did robust observation handling reduce localization-outlier impact?**
   No conclusion supported by n=9 — the point estimate moved the wrong way
   (Robust B worse). Insufficient evidence.
2. **Did temporal association improve duplicate frames?** `duplicate_frames
   = 84`, `wrong_association_events = 2` — association resolves the large
   majority of duplicate-candidate frames to a correct pick (98% of 84), but
   the ablation shows `temporal_association_only` is *worse* than baseline on
   range RMSE in isolation (Finding 5/6) — association's net benefit to
   headline accuracy is realized only in combination with the other
   components, not as a standalone improvement.
3. **Did range uncertainty become realistically calibrated?** Not within the
   pre-registered bands: range coverage_68/95 both overshoot (Finding 2).
   Substantially *less* miscalibrated than Baseline A's severe
   under-coverage (0.247/0.388), and not achieved by absurd inflation
   (Finding 2, std_over_rmse guard), but "realistically calibrated" against
   the numbers fixed before this run is not established.
4. **Did the conditional detection model improve belief through natural
   misses?** Yes, materially (in-domain retention 18.2% → 61.8%), but per
   Finding 4 this is entirely the I8 miss-likelihood floor plus invariant
   I3's routing, not the fitted per-class `P_D^eff` values, which are inert
   on this run's data.
5. **Did separating detection evidence from kinematic acceptance prevent the
   gate from worsening existence collapse?** Yes, and it is directly
   quantified: 23 `gated_rejection` frames, **100%** (23/23) retained an
   active belief. Under the pre-invariant-I2 design, every one of those 23
   frames would have been scored as an existence miss instead.
6. **Was RMSE materially worsened to achieve calibration?** No — range RMSE
   improved 21.5% and bearing RMSE improved 14.8%; the accuracy guard
   (criteria 7–8) is met comfortably in both axes.
7. **Is EKF + robust observation handling sufficient for Version-1 POMDP?**
   Mixed. It fixes Baseline A's severe under-coverage and materially
   improves miss-continuity and accuracy, but overshoots three of four
   coverage bands on the conservative side and has no settled evidence on
   outlier robustness or the 20-consecutive-miss criterion.
8. **Is the system control-ready?** See the classification recommendation
   below — this is presented as evidence for the human partner's decision,
   not resolved unilaterally here.

### Recommended classification: `LIMITED`

**Reasoning.** `CONTROL_READY` is not warranted: 3 of 17 pre-specified
criteria are clearly not met (all three are coverage-band misses, criteria
1–3), and a further 2 are unproven rather than passed (criteria 10 and 17).
A configuration that misses its own pre-registered calibration bands on the
primary metric (range) cannot be called control-ready by the plan's own
rubric, regardless of how large the accuracy and miss-continuity gains are.

`FAILED` is not warranted either. Every miss is in the conservative
direction — Robust B is *over*-confident about its own uncertainty in the
wrong direction (believes itself less certain than it is), never
under-confident, which the plan itself treats as the safer failure mode for
a downstream collision-avoidance consumer, and this is independently
confirmed (Finding 2's anti-inflation guard passes cleanly, so the
over-coverage is not runaway or degenerate). 12 of 17 criteria are cleanly
met, including both accuracy guards, both anti-inflation guards, the primary
existence-retention criterion, recovery, false-tracks, and all four support
minima. The heavy-tail explanation for the miss was actively tested and
refuted rather than left as an untested excuse (Finding 3), and the
remaining hypothesis (floor transfer from 8 calibration seeds to 4 different
final seeds) is stated as a hypothesis, not asserted as settled — this is
the behavior of a system whose remaining problem is diagnosed, not one whose
behavior is unexplained or uncontrolled.

`LIMITED` reflects both: materially better than F9b (which was itself
`LIMITED` on far worse numbers — coverage 0.152/0.258 vs this run's
0.852/0.988, and 8/57 = 14.0% pooled miss retention vs this run's 61.8%
in-domain), while still short of the numeric bar this plan pre-registered
for `CONTROL_READY` on 3 of 17 criteria.

**Strongest argument for `LIMITED` over `CONTROL_READY`**: the plan
pre-registered exact coverage bands *before* this run specifically so that a
result could not be rounded into a pass after the fact ("never to be
adjusted after seeing final-evaluation results") — range coverage_68 misses
its band by 0.092 (12% relative), which is not a rounding-distance miss.

**Strongest argument against `LIMITED`, for the human partner's
consideration**: every miss is conservative, not dangerous, and the plan's
own guard for detecting a "cheating" over-conservative fit
(`std_over_rmse ≤ 1.5`) passes with margin (1.279, 1.009) — an argument could
be made that a pre-registered symmetric band is the wrong tool for scoring
an asymmetric-risk safety filter that overshoots only on the safe side. This
report does not make that argument on the human partner's behalf; it is
surfaced here because the classification is a recommendation, and the bands
themselves were set by the human partner, who is positioned to decide
whether a conservative miss should be treated the same as a dangerous one
for this specific gate.

**This is a recommendation, not the final word.** The full evidence above —
all 17 criteria, all 8 findings, the deferred-minors triage, and the
ablation/NIS/confusion-matrix detail — is presented so the human partner's
classification decision does not require re-deriving anything.

**STOP.** No stop logic, reward, or SAC begins after this report, per the
plan's explicit instruction.

## F9d

F9c was classified `LIMITED` on two unevidenced claims: gross
localization-outlier robustness, and long-absence existence decay. F9d
exists only to collect that evidence. It adds no estimator capability and,
critically, **F9c is frozen for the whole of F9d** — every parameter is
imported and hash-verified, never re-fitted. A disappointing F9d result is
itself the finding; re-fitting in response would destroy the only thing this
gate is for.

### Task 1 — protocol with frozen-F9c import guards

Added `src/duckie_pomdp/evaluation/f9d_protocol.py`
(`F9dProtocol`/`load_f9d_protocol`) and `configs/f9d_evidence_closure_v1.toml`,
modeled on `f9c_protocol.py`/`f9_protocol.py` and reusing their `sha256` and
scenario/detector dataclasses by import (nothing copied).

**F9c hash assertion.** `load_f9d_protocol` reads `[provenance].f9c_config`
and `[provenance].f9c_config_sha256` from the F9d config, resolves the F9c
config path relative to the F9d config's own directory, and recomputes
`sha256(f9c_config_path)`. If that does not equal the pinned
`f9c_config_sha256`, it raises `ValueError` containing the literal substring
`frozen F9c` and refuses to load. The pinned value —

```
359dc52020421c248bf6c26e036234191b2b97d24b505a66d85daa85b563704e
```

— matches `sha256sum configs/f9c_robust_belief_v1.toml` on disk both before
and after this task; `f9c_robust_belief_v1.toml` and `f9c_protocol.py` were
not modified.

**No estimator parameters in the F9d config.** `configs/f9d_evidence_closure_v1.toml`
defines only `[provenance]`, `[split]`, `[minima]`, `[criteria]`, and
`[artifacts]`. It contains none of `measurement_model`,
`covariance_calibration`, `conditional_detection`, `innovation_gate`,
`association`, `ekf`, or `existence` — `_validate` also asserts this
programmatically and raises `ValueError` if any of those sections appear.
Scenario matrices are deliberately absent at this stage; Tasks 3/4 add them
after measuring real yields.

**Read-only accessor, not a copy.** `f9c_parameters(protocol)` calls
`load_f9c_protocol(protocol.f9c_config_path, require_frozen=True)` fresh on
every call — it re-parses the TOML from disk each time rather than caching
any field, so there is no F9d-owned copy of an F9c parameter that could ever
drift from the file on disk. `F9dProtocol` and the `F9cProtocol` it returns
are both frozen dataclasses.

**Seed disjointness.** F9d's three seed bands (`development_seeds`
8101–8108, `outlier_final_seeds` 8201–8204, `absence_final_seeds`
8301–8304) are asserted pairwise disjoint, and `forbidden_seeds` folds in
every earlier F7/F8/F9/F9b/F9c seed (1101–1106, 2101–2102, 3101–3102,
4101–4104, 5101–5104, F9c's calibration seeds 6101–6108, and F9c's
final-evaluation seeds 7101–7104) so F9d cannot silently reuse data any
earlier gate touched.

**`outlier_support_satisfied`.** Implements the pre-registered three-way
support rule: `frames >= minimum_outlier_frames` AND
`events >= minimum_outlier_events` AND `seeds >= minimum_outlier_seeds`,
and additionally short-circuits to `False` whenever
`frames < insufficient_outlier_frames`, regardless of the other two —
50 frames from two long bursts on one seed is not the evidence this gate
needs.

TDD: `tests/test_f9d_protocol.py` written first and confirmed to fail with
`ModuleNotFoundError: No module named 'duckie_pomdp.evaluation.f9d_protocol'`
before any implementation existed; all 6 tests pass after implementation.
Full suite: 257 passed (251 baseline + 6 new).

### Tasks 2–5 — diagnostic, evidence probes, B3, and freeze

Task 2 replayed the frozen 3,328-frame F9c cache only. C1's selection-scale
hypothesis was refuted (four outliers under both lambda values), while the
separate abstention hypothesis was supported (42 no-selection frames at
lambda=1 vs 22 frozen). C2 tied 12–12 paired wins; min-NIS was not inferior
under the registered conjunctive rule, but produced 2 vs 1 outliers.

Task 3 development seeds projected 119.5 natural outlier frames, 68.5 events,
and all four final seeds with events, authorising freeze. Task 4 separately
validated B1, detector-boundary B2, and genuine-removal B3. Development
projection cleared support for each kind; B2 had zero GT-invisible dropout
frames. Real-simulator B3 tests proved RGB disappearance and privileged
existence disappearance on the same first absent frame. Full active suite at
the Task 4 gate was 331/331.

Task 5 froze `configs/f9d_evidence_closure_v1.toml` at SHA256
`7bbe6525c24e294b55a46808301249633236658814e906a68d0d804d5e8a8ca6`.
The verifier has graceful development mode and strict final mode; a unit
test proves strict mode fails when an artifact is missing.

### Tasks 6–8 — final evidence and classification

F9d-A rendered seeds 8201–8204 exactly once: 7,658 rows, 43 natural outlier
frames, 29 events, all four seeds. The pre-registered 50-frame minimum did
not pass, so the result is `INSUFFICIENT_EVIDENCE`. Descriptively, Robust B
range RMSE was `0.01648 m` vs Baseline A `0.06827 m` (ratio `0.241`), with
lower bias and transient error. Forty-six early invalid-pose terminations
are retained as evidence, not excluded.

F9d-B rendered seeds 8301–8304 exactly once: 7,260 rows and no warnings.
B1/B2/B3 support was 20/15, 24/12, and 16/12 runs at >=20/>=40 frames,
respectively. B1's prediction-only recurrence matched to `4.44e-16` after a
metrics-only fix excluded positive detections at the GT visibility boundary;
the CSV was reused and no render/inference/filter reran. B2 reached mean
`P(e)=0.002989` at frame 20 in all 24 runs and recovered one frame after
detection returned. B3 deleted all 16 genuinely removed tracks.

Strict artifact verification passed 15/15 with no skip. Leakage/freeze tests
passed. Because CONTROL_READY requires both A and B, F9d is **LIMITED**:
B passed, A remains under-supported, and C is diagnostic-only. Full report:
`docs/superpowers/F9D_REPORT_FOR_REVIEW.md`.

Final active repository suite: **351 passed, 0 failed, 0 skipped** with 264
dependency/runtime warnings. The suite was invoked through the documented
Duckietown virtual environment and scoped to `tests/`; archived historical
attempts are not part of the active test contract.

## YOLO-to-belief evidence video

A reproducible presentation-only renderer now demonstrates the complete real
runtime path without changing F9c: front RGB, frozen YOLO11n, the existing
camera projector, robust association/gate, frozen EKF kinematics, and public
pedestrian belief. Demo seed `9101` is disjoint from calibration/evaluation
seeds. Privileged truth is read after `updater.update()` and is used only for
the explicitly labelled magenta evaluation overlay and manifest errors.

The generated MP4 contains 121 frames at 15 FPS (`8.07 s`, `1000x480`). In
241 simulator frames it recorded 231 Duckie detections, 238 active-track
frames, 226 accepted updates, 5 gate/association rejections, and 7 duplicate
detection frames. Demo-only belief RMSE was `0.01178 m` range and
`0.02238 rad` bearing. Provenance, frozen hashes, counts, and the MP4 SHA256
are stored in `artifacts/yolo_belief_demo.json`.

## F10-L1 — staged `small_loop` counter-clockwise lane training

The first post-F10 curriculum was deliberately narrowed to lane competence.
The policy receives six agent-visible ego/lane values and uses the existing
normalized-action to `PolicyAction` to wheel-command chain. YOLO/F9c and all
pedestrian/stop logic are unchanged and absent from this stage. Yellow-line
clearance, path length, and lap geometry are reward/evaluation-only.

The reward audit, online W&B smoke (`iz0pipsf`), CUDA witness, exact checkpoint
reload, source/config hashes, disjoint seeds, and focused tests all passed
before the one declared 60,000-step training run. The official online run is
`z39mxtvl` at `vnv/DuckiePOMDP`; it completed 58,001 gradient updates and 118
episodes. A terminal-wrapper-limited step-1,000 process is explicitly archived
as aborted and does not contribute evidence.

Development-only safety-first selection compared all six 10k checkpoints and
selected step 50,000 (SHA256 `7d492fbf...2f72`). Step 60,000 remains the last
checkpoint for audit only. On untouched final seeds 15001-15004, selected SAC
completed 4/4 laps with zero invalid pose, yellow crossing, lane departure, or
timeout. Mean absolute lateral error was `0.01062 m`, mean episode p95 `|d|`
was `0.02502 m`, mean actual velocity was `0.13976 m/s`, and return was
`30.973`. All ten pre-registered acceptance checks passed.

The real-simulator proof `artifacts/f10_l1/sac_lane_demo.mp4` uses development
seed 14001 after selection. It completes a 5.03 m lap in 35.77 s with mean
`|d|=0.01140 m`, p95 `|d|=0.02602 m`, and no safety event. Its evaluation-only
overlay is explicitly labelled. Final classification is **PASS** for lane
competence only; it is not a full POMDP deployment checkpoint. Full active
suite: **411 passed, 0 failed, 0 skipped**.

## F10-L2 — `experiment_loop` mixed-turn transfer

F10-L2 reused the F10-L1 six-state observation, action mapper, reward, and SAC
implementation without adding privileged inputs. It restored the selected
F10-L1 step-50,000 actor, critics, target critics, entropy state, and optimizer
states, then collected a new replay buffer on `experiment_loop`. Training,
development, final, and historical probe seeds are disjoint.

The reward audit established that the simple controller can complete the new
map while the source F10-L1 SAC crosses the yellow line. After an online W&B
smoke (`3rlwg0tv`), the one declared 40,000-step transfer run (`y0qu681q`)
completed 38,001 updates and saved four checkpoints. Training was
non-monotonic: the step-30,000 candidate failed all four development episodes
with invalid poses, while step 40,000 recovered.

The frozen safety-first selection chose step 40,000, SHA256
`09a7fbcf...948a`. On once-only final seeds 18001-18004, transfer SAC completed
4/4 laps with zero invalid pose, yellow crossing, lane departure, or timeout.
Mean `|d|` was `0.03413 m`, mean episode p95 `|d|` was `0.05763 m`, mean
actual velocity was `0.12980 m/s`, and mean return was `32.446`. The source
F10-L1 SAC completed 0/4 and crossed yellow in 4/4; the simple controller
completed 4/4 with return `23.335` and mean `|d|=0.05501 m`.

The proof video `artifacts/f10_l2/sac_lane_transfer_demo.mp4` completes a
7.28 m mixed-turn lap in 55.5 s without a safety event. The final
classification is **PASS** for cross-map lane transfer only. It remains a
lane-state curriculum checkpoint and does not include YOLO/F9c, pedestrian,
or stop behaviour. Full active suite: **419 passed, 0 failed, 0 skipped**.
# F10-PPO visual-lane v3 curve-recovery result (2026-08-11)

- Added an isolated `f10_ppo_visual_v3.toml` protocol; v2 artifacts/config were
  retained unchanged.
- Yellow contact is now recoverable only for shallow curve contact. Deep
  penetration, standalone straight contact, and recovery timeout remain
  terminal. Three clear frames are required before recovery completes.
- True curvature is consumed only by reward/evaluation. The PPO actor and critic
  still receive the same 29D visual-lane/YOLO/belief vector.
- Pretraining evidence passed: reward audit, 36-reset memory audit, 128-step PPO
  smoke, W&B preflight, independent audit, and 472 active tests.
- C0 full training completed 61,440 steps and 60 updates in W&B run `oog0l05m`.
  There were zero training lap completions.
- Development evaluated all six checkpoints on seeds `47101-47104`. No
  checkpoint was eligible. Step 40,960 was retained diagnostically only:
  completion 25%, progress 2.139 m, lane failure 0%, invalid pose 75%.
- Per the frozen STOP rule, stage-final seeds were not used and C1 was not
  started. See `docs/F10_PPO_VISUAL_RECOVERY_REPORT_FOR_REVIEW.md`.
