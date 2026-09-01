from pathlib import Path
from types import SimpleNamespace

from duckie_pomdp.control.ppo_protocol import load_ppo_curriculum_protocol
from duckie_pomdp.control.stop_belief import RuntimeStopBeliefUpdater
from duckie_pomdp.domain.detection import BoundingBox, Detection, ObjectClass
from duckie_pomdp.domain.observation import EgoObservation
from duckie_pomdp.domain.state import StopMode


CONFIG = Path(__file__).resolve().parents[1] / "configs" / "f10_ppo_v1.toml"


class Projector:
    def project_raw(self, detection):
        del detection
        return SimpleNamespace(raw_polar=SimpleNamespace(range_m=0.8, bearing_rad=-0.1))


def _detection(confidence=0.9):
    return Detection(ObjectClass.STOP_SIGN, confidence, BoundingBox(10, 10, 30, 50))


def test_stop_obligation_uses_detection_and_agent_motion_only():
    protocol = load_ppo_curriculum_protocol(CONFIG)
    updater = RuntimeStopBeliefUpdater(protocol, Projector(), active=True)
    updater.reset()
    ego = EgoObservation(0.0, 0.0, 0.0, 0.0)
    step = updater.update((_detection(),), stop_line_distance_m=0.8, ego=ego)
    assert step.belief.existence_probability > 0.7
    assert step.mode is StopMode.REQUIRED
    for _ in range(protocol.raw["stop_belief"]["hold_steps"]):
        step = updater.update((_detection(),), stop_line_distance_m=0.1, ego=ego)
    assert step.mode is StopMode.SATISFIED


def test_inactive_stop_updater_is_semantically_neutral():
    protocol = load_ppo_curriculum_protocol(CONFIG)
    updater = RuntimeStopBeliefUpdater(protocol, Projector(), active=False)
    step = updater.update((_detection(),), stop_line_distance_m=0.0, ego=EgoObservation(0,0,0,0))
    assert step.belief.existence_probability == 0.0
    assert step.belief.range_mean_m > 0.0
    assert step.mode is StopMode.NONE

