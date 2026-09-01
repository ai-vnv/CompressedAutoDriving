from __future__ import annotations

import pytest

from duckie_pomdp.adapters.differential_drive import (
    DifferentialDriveActionAdapter,
    NormalizedActionScaler,
    PolicyActionBounds,
)
from duckie_pomdp.domain.action import NormalizedPolicyAction, PolicyAction


def test_conversion_matches_duckietown_equations() -> None:
    adapter = DifferentialDriveActionAdapter()

    result = adapter.convert(
        PolicyAction(
            linear_velocity_mps=0.12,
            angular_velocity_rad_s=0.8,
        )
    )

    assert result.wheel_command.left == pytest.approx(0.0922431866)
    assert result.wheel_command.right == pytest.approx(0.1872816212)
    assert result.wheel_angular_velocity.left_rad_s == pytest.approx(
        (0.12 - 0.5 * 0.102 * 0.8) / 0.0318
    )
    assert not result.saturated


def test_positive_angular_velocity_drives_right_wheel_faster() -> None:
    adapter = DifferentialDriveActionAdapter()

    positive = adapter.to_wheels(PolicyAction(0.1, 0.5))
    negative = adapter.to_wheels(PolicyAction(0.1, -0.5))

    assert positive.right > positive.left
    assert negative.right < negative.left


def test_zero_action_produces_zero_wheel_actuation() -> None:
    wheels = DifferentialDriveActionAdapter().to_wheels(PolicyAction(0.0, 0.0))

    assert wheels.left == pytest.approx(0.0)
    assert wheels.right == pytest.approx(0.0)


def test_straight_action_produces_symmetric_wheel_actuation() -> None:
    wheels = DifferentialDriveActionAdapter().to_wheels(PolicyAction(0.2, 0.0))

    assert wheels.left == pytest.approx(wheels.right)
    assert wheels.left > 0.0


def test_saturation_is_explicit() -> None:
    result = DifferentialDriveActionAdapter().convert(PolicyAction(2.0, 0.0))

    assert result.saturated
    assert result.left_saturated
    assert result.right_saturated
    assert result.unclipped_wheel_command.left > 1.0
    assert result.wheel_command.left == 1.0
    assert result.wheel_command.right == 1.0


@pytest.mark.parametrize(
    ("normalized", "expected_linear", "expected_angular"),
    [
        (NormalizedPolicyAction(-1.0, -1.0), 0.0, -2.0),
        (NormalizedPolicyAction(0.0, 0.0), 0.2, 0.0),
        (NormalizedPolicyAction(1.0, 1.0), 0.4, 2.0),
    ],
)
def test_normalized_action_scaling_has_no_reverse(
    normalized: NormalizedPolicyAction,
    expected_linear: float,
    expected_angular: float,
) -> None:
    scaler = NormalizedActionScaler(PolicyActionBounds(0.4, 2.0))

    action = scaler.to_policy_action(normalized)

    assert action.linear_velocity_mps == pytest.approx(expected_linear)
    assert action.angular_velocity_rad_s == pytest.approx(expected_angular)


def test_reverse_command_is_rejected() -> None:
    with pytest.raises(ValueError, match="does not allow reverse"):
        PolicyAction(-0.01, 0.0)


@pytest.mark.parametrize("invalid", [float("nan"), float("inf"), float("-inf")])
@pytest.mark.parametrize("field", ["linear", "angular"])
def test_non_finite_policy_action_is_rejected(field: str, invalid: float) -> None:
    linear = invalid if field == "linear" else 0.1
    angular = invalid if field == "angular" else 0.0

    with pytest.raises(ValueError, match="must be finite"):
        PolicyAction(linear, angular)
