# Third-party components and model weights

The MIT license in `LICENSE` covers the source code, configurations, and
documentation authored in this repository. Some tracked model weights are
derived from third-party projects and carry the terms of their upstream
project. Those terms are not superseded by the MIT license.

## Model weights in `models/`

| File | Derived from | Upstream license | Terms that apply |
|---|---|---|---|
| `yolo11n_duckietown_best.pt` | Ultralytics YOLO11 (`yolo11n.pt`), fine-tuned on Duckietown frames | **AGPL-3.0** (Ultralytics offers a separate commercial license) | Treat this file as AGPL-3.0. If you need different terms, retrain the detector from a permissively licensed base, or obtain an Ultralytics commercial license. |
| `mobilenetv3_lane_pose_best.pt` | torchvision MobileNetV3-Small backbone, fine-tuned for lane pose regression | BSD-3-Clause (torchvision) | BSD-3-Clause attribution applies to the backbone. |
| `actor_A0.pt` … `actor_A9.pt` | Trained from scratch in this work (belief-state PPO, then compressed) | — | MIT, as in `LICENSE`. |

The actor checkpoints, which are the study object of the paper, contain no
third-party weights. The perception checkpoints are inputs to the pipeline and
are provided so the closed-loop evaluation can be reproduced.

## Software dependencies

| Project | Role | License |
|---|---|---|
| [Gym-Duckietown](https://github.com/duckietown/gym-duckietown) | driving simulator | see upstream repository |
| [Ultralytics](https://github.com/ultralytics/ultralytics) | YOLO11 detector, training and inference | AGPL-3.0 |
| [PyTorch](https://pytorch.org) / torchvision | networks, eager-mode quantization | BSD-3-Clause |
| [Stable-Baselines3](https://github.com/DLR-RM/stable-baselines3) | PPO implementation | MIT |

Pinned versions are in `pyproject.toml` and `constraints.txt`.
