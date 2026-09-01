"""F10-PPO Visual-Lane v4, Task 1: lane-belief uncertainty recalibration + gate.

Covers ``duckie_pomdp.control.lane_belief_uncertainty``: the calibration map
that widens the EKF's overconfident reported sigma, the coverage scoring it
is checked against, and the pre-registered ``[uncertainty_gate]`` that v3 was
missing. The mean estimate (point extractor / bias calibration) is untouched
by this task and is asserted as such below.

Fix round 1 adds the runtime wiring: the calibration must actually reach the
belief that ``ppo_observation.py`` feeds into the policy, gated behind
``[v4_changes].belief_uncertainty_refit`` so a config without that flag (all
of v3) reproduces the raw EKF sigma bit-for-bit.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10 runtime used by Gym-Duckietown.
    import tomli as tomllib

import numpy as np
import pytest

from duckie_pomdp.control.lane_belief_uncertainty import (
    LaneUncertaintyCalibration,
    apply_calibration_to_lane_belief,
    coverage_report,
    evaluate_uncertainty_gate,
    lateral_overcoverage_disclosure,
    load_lane_uncertainty_calibration,
    load_validation_errors_and_sigmas,
    resolve_runtime_calibration,
)
from duckie_pomdp.control.lane_belief_runtime import (
    VisualLaneBeliefRuntime,
    _resolve_filter_config_path,
)
from duckie_pomdp.control.ppo_protocol import (
    load_ppo_curriculum_protocol,
    pretraining_source_paths,
)
from duckie_pomdp.domain.belief import LaneBelief
from duckie_pomdp.domain.measurement import GroundPoint
from duckie_pomdp.perception.camera_geometry import (
    CalibratedGroundProjector,
    CameraCalibration,
)

ROOT = Path(__file__).resolve().parents[1]
VALIDATION_CSV = ROOT / "artifacts" / "visual_lane" / "lane_belief_final_validation.csv"
V1_CONFIG = ROOT / "configs" / "lane_belief_v1.toml"
V2_CONFIG = ROOT / "configs" / "lane_belief_v2.toml"
V3_PROTOCOL_CONFIG = ROOT / "configs" / "f10_ppo_visual_v3.toml"

# The plan's pre-registered, verified-not-invented calibration.
HEADING_FLOOR_RAD = 0.051
HEADING_SCALE = 1.0
LATERAL_FLOOR_M = 0.0
LATERAL_SCALE = 4.0

# The pre-v4 effective floor observed in the raw (uncalibrated) belief --
# see the task brief: min reported heading sigma in the validation set.
PRE_V4_EFFECTIVE_HEADING_FLOOR_RAD = 0.0197


def _load() -> dict:
    return load_validation_errors_and_sigmas(VALIDATION_CSV)


def test_floor_and_scale_only_ever_widen_the_interval() -> None:
    """A calibration must never report less uncertainty than the raw belief."""
    cal = LaneUncertaintyCalibration(0.051, 1.0, 0.0, 4.0)
    h, l = cal.apply(0.0197, 0.0089)
    assert h >= 0.0197
    assert l >= 0.0089


def test_calibration_rejects_a_scale_below_one() -> None:
    """A scale < 1.0 could narrow the interval below the raw belief; reject it."""
    with pytest.raises(ValueError):
        LaneUncertaintyCalibration(0.051, 0.9, 0.0, 4.0)


def test_heading_floor_lands_the_measured_validation_set_in_band() -> None:
    """Load the real validation CSV, apply the calibration, and assert coverage.

    This is the whole point of the task -- if it does not reproduce, the
    numbers in the plan are wrong and the task must stop, not adjust the band.
    """
    data = _load()
    cal = LaneUncertaintyCalibration(
        HEADING_FLOOR_RAD, HEADING_SCALE, LATERAL_FLOOR_M, LATERAL_SCALE
    )
    calibrated_heading_sigma = np.array(
        [cal.apply(s, 0.0)[0] for s in data["heading"]["sigma"]]
    )
    report = coverage_report(data["heading"]["error"], calibrated_heading_sigma)

    assert 0.60 <= report["coverage_68"] <= 0.76
    assert 0.90 <= report["coverage_95"] <= 0.98
    assert report["sigma_over_rmse"] <= 1.5

    # Pinned to the plan's reproduced numbers (see task-1-report.md).
    assert report["coverage_68"] == pytest.approx(0.602, abs=0.01)
    assert report["coverage_95"] == pytest.approx(0.925, abs=0.01)
    assert report["sigma_over_rmse"] == pytest.approx(0.76, abs=0.01)


def test_lateral_scale_lands_coverage_68_but_overcovers_95() -> None:
    """The plan's lateral scale puts cov68 in band; cov95 over-covers by design."""
    data = _load()
    cal = LaneUncertaintyCalibration(
        HEADING_FLOOR_RAD, HEADING_SCALE, LATERAL_FLOOR_M, LATERAL_SCALE
    )
    calibrated_lateral_sigma = np.array(
        [cal.apply(0.0, s)[1] for s in data["lateral"]["sigma"]]
    )
    report = coverage_report(data["lateral"]["error"], calibrated_lateral_sigma)

    assert 0.60 <= report["coverage_68"] <= 0.76
    assert report["coverage_95"] > 0.98  # over-covers -- not gated, see disclosure
    assert report["coverage_68"] == pytest.approx(0.738, abs=0.01)
    assert report["coverage_95"] == pytest.approx(1.000, abs=0.01)
    assert report["sigma_over_rmse"] == pytest.approx(1.33, abs=0.01)


def test_lateral_overcoverage_is_disclosed_not_hidden() -> None:
    """The metrics artifact must carry lateral_overcoverage_disclosure with the
    z-quantiles, and lateral cov95 must NOT be presented as passing a band.
    """
    data = _load()
    disclosure = lateral_overcoverage_disclosure(
        data["lateral"]["error"], data["lateral"]["sigma"]
    )
    assert disclosure["z_quantiles_raw_belief"]["p50"] == pytest.approx(3.01, abs=0.05)
    assert disclosure["z_quantiles_raw_belief"]["p95"] == pytest.approx(4.90, abs=0.05)
    assert disclosure["z_quantiles_raw_belief"]["p99"] == pytest.approx(5.31, abs=0.05)
    assert "not" in disclosure["claim"].lower()
    assert "gate" in disclosure["claim"].lower()

    # The gate itself must never receive a lateral_coverage_95 band to check.
    gate_config = tomllib.loads(V2_CONFIG.read_text())["uncertainty_gate"]
    assert "lateral_coverage_95_band" not in gate_config


def test_the_gate_rejects_a_calibration_that_understates_uncertainty() -> None:
    """Feed the pre-v4 floor (0.0197) and assert the gate FAILS on cov68.
    A gate that has never been seen to fail is not a gate.
    """
    data = _load()
    gate_config = tomllib.loads(V2_CONFIG.read_text())["uncertainty_gate"]

    undercalibrated = LaneUncertaintyCalibration(
        PRE_V4_EFFECTIVE_HEADING_FLOOR_RAD, 1.0, 0.0, 1.0
    )
    heading_sigma = np.array(
        [undercalibrated.apply(s, 0.0)[0] for s in data["heading"]["sigma"]]
    )
    lateral_sigma = np.array(
        [undercalibrated.apply(0.0, s)[1] for s in data["lateral"]["sigma"]]
    )
    heading_report = coverage_report(data["heading"]["error"], heading_sigma)
    lateral_report = coverage_report(data["lateral"]["error"], lateral_sigma)

    result = evaluate_uncertainty_gate(heading_report, lateral_report, gate_config)

    assert result["pass"] is False
    assert result["checks"]["heading_coverage_68_in_band"] is False


def test_the_gate_accepts_the_fitted_v4_calibration() -> None:
    """The chosen v4 calibration must actually pass the gate it was fit to."""
    data = _load()
    gate_config = tomllib.loads(V2_CONFIG.read_text())["uncertainty_gate"]
    cal = LaneUncertaintyCalibration(
        HEADING_FLOOR_RAD, HEADING_SCALE, LATERAL_FLOOR_M, LATERAL_SCALE
    )
    heading_sigma = np.array(
        [cal.apply(s, 0.0)[0] for s in data["heading"]["sigma"]]
    )
    lateral_sigma = np.array(
        [cal.apply(0.0, s)[1] for s in data["lateral"]["sigma"]]
    )
    heading_report = coverage_report(data["heading"]["error"], heading_sigma)
    lateral_report = coverage_report(data["lateral"]["error"], lateral_sigma)

    result = evaluate_uncertainty_gate(heading_report, lateral_report, gate_config)

    assert result["pass"] is True


def test_mean_estimates_are_untouched() -> None:
    """Applying the calibration must not alter heading_mean or lateral_mean."""
    cal = LaneUncertaintyCalibration(0.051, 1.0, 0.0, 4.0)
    heading_mean = -0.0597
    lateral_mean = 0.0412
    heading_std, lateral_std = cal.apply(0.02, 0.01)
    # apply() only ever touches std -- means are simply never passed in or out.
    assert heading_mean == -0.0597
    assert lateral_mean == 0.0412
    assert isinstance(heading_std, float)
    assert isinstance(lateral_std, float)


# ---------------------------------------------------------------------------
# Fix round 1: the runtime wiring (belief-reporting boundary + the
# [v4_changes].belief_uncertainty_refit switch).
# ---------------------------------------------------------------------------


def _belief(
    *, heading_std: float = 0.02, lateral_std: float = 0.01
) -> LaneBelief:
    return LaneBelief(
        validity_probability=0.9,
        lateral_error_mean_m=0.0412,
        lateral_error_std_m=lateral_std,
        heading_error_mean_rad=-0.0597,
        heading_error_std_rad=heading_std,
        curvature_mean_inv_m=0.5,
        curvature_std_inv_m=0.3,
    )


def _projector() -> CalibratedGroundProjector:
    return CalibratedGroundProjector(
        CameraCalibration(
            image_width_px=640,
            image_height_px=480,
            vertical_fov_deg=75.0,
            camera_height_m=0.108,
            camera_pitch_deg=19.15,
            camera_forward_offset_m=0.066,
        )
    )


def _synthetic_lane_image(
    *, lateral_error_m: float, heading_error_rad: float, curvature_inv_m: float
) -> np.ndarray:
    image = np.zeros((480, 640, 3), dtype=np.uint8)
    projector = _projector()
    c1 = np.tan(heading_error_rad)
    c2 = 0.5 * curvature_inv_m * (1.0 + c1 * c1) ** 1.5
    for forward in np.linspace(0.10, 1.10, 250):
        centre = lateral_error_m + c1 * forward + c2 * forward**2
        for boundary, colour in (
            (centre + 0.117, np.array([220, 180, 10], dtype=np.uint8)),
            (centre - 0.117, np.array([210, 210, 210], dtype=np.uint8)),
        ):
            try:
                pixel = projector.ground_to_pixel(GroundPoint(boundary, forward))
            except ValueError:
                continue
            column, row = int(round(pixel.x_px)), int(round(pixel.y_px))
            if 2 <= column < 638 and 2 <= row < 478:
                image[row - 1 : row + 2, column - 2 : column + 3] = colour
    return image


def test_apply_calibration_to_lane_belief_none_is_identity() -> None:
    """calibration=None (the off/v3 path) returns the belief unchanged."""
    belief = _belief()
    assert apply_calibration_to_lane_belief(belief, None) is belief


def test_apply_calibration_to_lane_belief_widens_only_reported_std() -> None:
    belief = _belief(heading_std=0.02, lateral_std=0.005)
    cal = LaneUncertaintyCalibration(0.051, 1.0, 0.0, 4.0)
    widened = apply_calibration_to_lane_belief(belief, cal)
    assert widened.heading_error_std_rad == pytest.approx(0.051)  # floored
    assert widened.lateral_error_std_m == pytest.approx(0.02)  # 4x scaled
    # everything else, including both means, is untouched.
    assert widened.heading_error_mean_rad == belief.heading_error_mean_rad
    assert widened.lateral_error_mean_m == belief.lateral_error_mean_m
    assert widened.validity_probability == belief.validity_probability
    assert widened.curvature_mean_inv_m == belief.curvature_mean_inv_m
    assert widened.curvature_std_inv_m == belief.curvature_std_inv_m


def test_load_lane_uncertainty_calibration_v1_has_none_v2_has_the_table() -> None:
    assert load_lane_uncertainty_calibration(V1_CONFIG) is None
    calibration = load_lane_uncertainty_calibration(V2_CONFIG)
    assert calibration is not None
    assert calibration.heading_floor_rad == pytest.approx(0.051)
    assert calibration.lateral_scale == pytest.approx(4.0)


@pytest.mark.parametrize(
    "protocol_raw",
    [
        {},
        {"v4_changes": {}},
        {"v4_changes": {"belief_uncertainty_refit": False}},
        {"v4_changes": {"start_randomisation": True}},  # a different switch is on
    ],
)
def test_resolve_runtime_calibration_defaults_off(protocol_raw: dict) -> None:
    """Every v3-shaped protocol (no table, or the flag absent/false) is off."""
    assert resolve_runtime_calibration(protocol_raw, V2_CONFIG) is None


def test_resolve_runtime_calibration_on_reads_the_table() -> None:
    protocol_raw = {"v4_changes": {"belief_uncertainty_refit": True}}
    calibration = resolve_runtime_calibration(protocol_raw, V2_CONFIG)
    assert calibration is not None
    assert calibration.heading_floor_rad == pytest.approx(0.051)


def test_resolve_runtime_calibration_on_without_table_raises() -> None:
    """Flag says on, but the referenced config has no [lane_belief_uncertainty]
    table -- this must fail loudly, not silently resolve to None (that would
    be an "on" flag that changes nothing, the exact bug this task closes).
    """
    protocol_raw = {"v4_changes": {"belief_uncertainty_refit": True}}
    with pytest.raises(RuntimeError, match="belief_uncertainty_refit"):
        resolve_runtime_calibration(protocol_raw, V1_CONFIG)


def test_resolve_filter_config_path_v1_is_unchanged() -> None:
    assert _resolve_filter_config_path(V1_CONFIG) == V1_CONFIG


def test_resolve_filter_config_path_v2_chains_to_v1() -> None:
    assert _resolve_filter_config_path(V2_CONFIG) == V1_CONFIG


def test_resolve_filter_config_path_rejects_a_drifted_source(tmp_path: Path) -> None:
    wrong_sha256 = "0" * 64
    broken = tmp_path / "lane_belief_v2_broken.toml"
    broken.write_text(
        'source_config = "lane_belief_v1.toml"\n'
        f'source_config_sha256 = "{wrong_sha256}"\n'
    )
    # tmp_path has no lane_belief_v1.toml next to it -- FileNotFoundError.
    with pytest.raises(FileNotFoundError):
        _resolve_filter_config_path(broken)

    sibling = tmp_path / "lane_belief_v1.toml"
    sibling.write_text(V1_CONFIG.read_text())
    with pytest.raises(RuntimeError, match="drifted"):
        _resolve_filter_config_path(broken)


def test_runtime_flag_off_reproduces_v3_sigma_bit_for_bit() -> None:
    """The central claim of the fix: with the calibration resolved to None
    (flag off), a runtime pointed at the v2 extension config reports byte-
    identical belief to a runtime pointed directly at v1 -- both the source
    -config chaining and the calibration gate are no-ops when off.
    """
    image_sequence = [
        _synthetic_lane_image(
            lateral_error_m=0.02, heading_error_rad=0.05, curvature_inv_m=0.3
        ),
        _synthetic_lane_image(
            lateral_error_m=0.04, heading_error_rad=0.15, curvature_inv_m=0.8
        ),
    ]

    runtime_v1 = VisualLaneBeliefRuntime(_projector(), config_path=V1_CONFIG)
    runtime_v2_off = VisualLaneBeliefRuntime(
        _projector(), config_path=V2_CONFIG, uncertainty_calibration=None
    )

    from duckie_pomdp.domain.state import EgoMotion

    step_v1 = runtime_v1.reset(image_sequence[0])
    step_v2 = runtime_v2_off.reset(image_sequence[0])
    assert step_v1.belief == step_v2.belief

    step_v1 = runtime_v1.update(
        image_sequence[1], actual_ego_motion=EgoMotion(0.2, 0.1), dt_s=1.0 / 30.0
    )
    step_v2 = runtime_v2_off.update(
        image_sequence[1], actual_ego_motion=EgoMotion(0.2, 0.1), dt_s=1.0 / 30.0
    )
    assert step_v1.belief == step_v2.belief


def test_runtime_flag_on_widens_the_reported_belief_std() -> None:
    calibration = load_lane_uncertainty_calibration(V2_CONFIG)
    image = _synthetic_lane_image(
        lateral_error_m=0.02, heading_error_rad=0.05, curvature_inv_m=0.3
    )

    runtime_off = VisualLaneBeliefRuntime(_projector(), config_path=V2_CONFIG)
    runtime_on = VisualLaneBeliefRuntime(
        _projector(), config_path=V2_CONFIG, uncertainty_calibration=calibration
    )

    off_belief = runtime_off.reset(image).belief
    on_belief = runtime_on.reset(image).belief

    # Same means (the point extractor/filter is identical either way).
    assert on_belief.heading_error_mean_rad == off_belief.heading_error_mean_rad
    assert on_belief.lateral_error_mean_m == off_belief.lateral_error_mean_m
    # Reported std only ever widens.
    assert on_belief.heading_error_std_rad >= off_belief.heading_error_std_rad
    assert on_belief.lateral_error_std_m >= off_belief.lateral_error_std_m


def test_v3_pretraining_source_paths_are_unaffected_by_v2_support() -> None:
    """Loading v3 must not pick up lane_belief_v2.toml as a frozen source --
    v3's already-recorded pretraining gate evidence must stay valid.
    """
    protocol = load_ppo_curriculum_protocol(V3_PROTOCOL_CONFIG, require_frozen=True)
    sources = pretraining_source_paths(protocol)
    assert "configs/lane_belief_v1.toml" in sources
    assert "configs/lane_belief_v2.toml" not in sources


def test_pretraining_source_paths_include_the_full_chain_for_a_v2_pointing_protocol(
    tmp_path: Path,
) -> None:
    """A protocol whose provenance points lane_belief_config at v2 must pick
    up BOTH v2 (the extension actually named) and v1 (what it chains to,
    and what the runtime filter tables actually load from) as frozen
    sources -- the allowlist accepting v2 must not weaken what is checked.
    """
    v2_sha256 = hashlib.sha256(V2_CONFIG.read_bytes()).hexdigest()
    text = V3_PROTOCOL_CONFIG.read_text().replace(
        'lane_belief_config = "lane_belief_v1.toml"\n'
        'lane_belief_config_sha256 = '
        '"41df028bf6290eef80cc5aab8114a7b7ccb0c4b19db6c3b7919a0ac68d552272"',
        f'lane_belief_config = "lane_belief_v2.toml"\n'
        f'lane_belief_config_sha256 = "{v2_sha256}"',
    )
    assert 'lane_belief_config = "lane_belief_v2.toml"' in text, (
        "the v3 config text this test patches has drifted from what it expects"
    )
    # Must live inside configs/ -- every other provenance path in the file
    # (scenario, pomdp_map, action_config, ...) is resolved relative to the
    # config file's own directory.
    scratch = ROOT / "configs" / "_scratch_v4_provenance_probe.toml"
    scratch.write_text(text)
    try:
        protocol = load_ppo_curriculum_protocol(scratch, require_frozen=True)
        sources = pretraining_source_paths(protocol)
        assert "configs/lane_belief_v1.toml" in sources
        assert "configs/lane_belief_v2.toml" in sources
    finally:
        scratch.unlink()
