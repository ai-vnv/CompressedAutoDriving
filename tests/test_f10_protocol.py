from pathlib import Path

import pytest

from duckie_pomdp.control.f10_protocol import load_f10_protocol


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "f10_sac_v1.toml"


def test_f10_protocol_pins_upstream_and_action_bounds() -> None:
    protocol = load_f10_protocol(CONFIG)
    assert protocol.action_bounds == (0.0, 0.4, -4.0, 4.0)
    assert protocol.detector_checkpoint_sha256 == "3d4f816d440690493b856d25403a84a3249e4250599319c32569b97cb8d7482c"
    assert protocol.belief_config_sha256 == "359dc52020421c248bf6c26e036234191b2b97d24b505a66d85daa85b563704e"


def test_f10_seed_splits_are_disjoint_and_historical_seeds_are_excluded() -> None:
    split = load_f10_protocol(CONFIG).seeds
    train, dev, final, historical = map(
        set,
        (split.training, split.development, split.final_evaluation, split.historical_evaluation),
    )
    assert train.isdisjoint(dev)
    assert train.isdisjoint(final)
    assert dev.isdisjoint(final)
    assert (train | dev | final).isdisjoint(historical)


def test_f10_observation_order_has_one_positive_scale_per_feature() -> None:
    protocol = load_f10_protocol(CONFIG)
    assert len(protocol.observation_order) == 17
    assert len(protocol.observation_scales) == 17
    assert all(value > 0.0 for value in protocol.observation_scales)


def test_f10_protocol_detects_split_leakage(tmp_path: Path) -> None:
    text = CONFIG.read_text(encoding="utf-8").replace(
        "development = [11001,", "development = [10001,"
    )
    path = tmp_path / "f10.toml"
    path.write_text(text, encoding="utf-8")
    with pytest.raises(ValueError, match="split leakage"):
        load_f10_protocol(path, require_frozen=False)
