from __future__ import annotations

from dataclasses import fields
from inspect import signature
from typing import get_type_hints

from duckie_pomdp.domain.belief import BeliefState
from duckie_pomdp.domain.measurement import PerceptionObservation
from duckie_pomdp.domain.observation import SensorObservation
from duckie_pomdp.domain.privileged import PrivilegedSimulatorState
from duckie_pomdp.domain.transition import Transition
from duckie_pomdp.ports.belief_updater import BeliefUpdater
from duckie_pomdp.ports.environment import AgentEnvironment, PrivilegedStateSource


def test_agent_visible_contracts_do_not_reference_privileged_state() -> None:
    agent_types = (
        SensorObservation,
        PerceptionObservation,
        BeliefState,
        Transition,
    )

    for contract in agent_types:
        annotations = str(get_type_hints(contract))
        assert "PrivilegedSimulatorState" not in annotations
        assert all("privileged" not in field.name for field in fields(contract))
        assert all("ground_truth" not in field.name for field in fields(contract))


def test_privileged_access_is_not_part_of_agent_environment() -> None:
    agent_annotations = str(get_type_hints(AgentEnvironment.step))
    privileged_annotations = get_type_hints(PrivilegedStateSource.read)

    assert "PrivilegedSimulatorState" not in agent_annotations
    assert privileged_annotations["return"] is PrivilegedSimulatorState


def test_transition_has_no_untyped_info_leakage_channel() -> None:
    assert "info" not in {field.name for field in fields(Transition)}


def test_belief_updater_requires_command_motion_and_perception() -> None:
    parameters = list(signature(BeliefUpdater.update).parameters)

    assert parameters == [
        "self",
        "previous_belief",
        "previous_action",
        "ego_motion",
        "perception",
        "dt_s",
    ]
