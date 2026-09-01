from pathlib import Path
from types import MappingProxyType

import numpy as np

from duckie_pomdp.control.ppo_protocol import (
    load_ppo_curriculum_protocol,
    pretraining_source_paths,
)
from duckie_pomdp.evaluation.privileged_c4_teacher import PrivilegedC4Teacher


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/f10_ppo_visual_objects_v24.toml"
V25_CONFIG = ROOT / "configs/f10_ppo_visual_objects_v25.toml"


def test_v24_dataset_keeps_privileged_truth_out_of_student_npz() -> None:
    protocol = load_ppo_curriculum_protocol(CONFIG)
    warm = protocol.raw["behavior_warm_start"]["c4"]
    dataset = (CONFIG.parent / warm["dataset"]).resolve()
    with np.load(dataset) as data:
        assert set(data.files) == {
            "observations",
            "actions",
            "weights",
            "value_targets",
            "value_weights",
        }
        assert data["observations"].shape == (26822, 29)
        assert np.count_nonzero(data["value_weights"]) == 1947
        assert all(np.all(np.isfinite(data[key])) for key in data.files)


def test_v24_teacher_consumes_truth_not_student_observation() -> None:
    teacher = PrivilegedC4Teacher()
    assert not hasattr(teacher, "observation")
    info = MappingProxyType(
        {
            "evaluation_gt": MappingProxyType(
                {
                    "lane_lateral_error_m": 0.0,
                    "lane_heading_error_rad": 0.0,
                    "road_curvature_inv_m": 0.0,
                    "pedestrian_exists": True,
                    "pedestrian_range_m": 0.6,
                    "pedestrian_bearing_rad": 0.0,
                    "pedestrian_radial_velocity_mps": -0.1,
                    "stop_line_distance_m": 1.0,
                }
            ),
            "stop_completed": False,
        }
    )
    action = teacher.act(info)
    assert action.shape == (2,)
    assert action[0] == -1.0


def test_v24_protocol_freezes_guided_builder_and_disjoint_seeds() -> None:
    protocol = load_ppo_curriculum_protocol(CONFIG)
    assert len(protocol.observation_order) == 29
    assert protocol.raw["behavior_warm_start"]["c4"]["critic_enabled"] is True
    assert "experiments/build_ppo_c4_privileged_guidance_v24.py" in set(
        pretraining_source_paths(protocol)
    )
    stage = protocol.stage("c4")
    assert stage.training_seeds == tuple(range(172001, 172013))
    assert stage.development_seeds == tuple(range(172101, 172105))
    assert stage.stage_final_seeds == tuple(range(172201, 172205))


def test_v25_initializes_from_audited_c4_descendant() -> None:
    protocol = load_ppo_curriculum_protocol(V25_CONFIG)
    warm = protocol.raw["behavior_warm_start"]["c4"]
    assert warm["initialize_from_checkpoint_sha256"] == warm[
        "learner_checkpoint_sha256"
    ]
    assert protocol.stage("c4").training_seeds == tuple(range(173001, 173013))
