from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from exo_bench.agent import (
    ACTION_SCHEMA,
    OBSERVATION_SCHEMA,
    ScriptedBaselineAgent,
    SimulatedTimeAgentScheduler,
    load_runtime_contract,
    safe_stop_action,
    validate_action,
    validate_observation,
    verify_agent_artifacts,
)
from exo_bench.paths import AGENT_CONTRACTS_ROOT


def observation() -> dict:
    return {
        "schema_version": OBSERVATION_SCHEMA,
        "simulation_time_s": 1.0,
        "base_position_m": {"x": 0.0, "y": 0.0, "z": 1.05},
        "heading_yaw_rad": 0.0,
        "planar_velocity_mps": {"x": 0.0, "y": 0.0},
        "task": {
            "id": "AGENT-001@1.0.0",
            "goal_position_m": {"x": 3.0, "y": 1.0},
            "goal_heading_rad": 0.3217505543966422,
            "target_radius_m": 0.75,
            "heading_tolerance_rad": 0.35,
        },
        "remaining_time_s": 34.0,
        "previous_applied_action": safe_stop_action(),
    }


def test_agent_contract_artifacts_are_json_and_hash_bound() -> None:
    for path in AGENT_CONTRACTS_ROOT.glob("*.json"):
        assert json.loads(path.read_text(encoding="utf-8"))
    hashes = verify_agent_artifacts()
    assert set(hashes) == {
        "agent_manifest_sha256",
        "observation_sha256",
        "action_sha256",
        "runtime_sha256",
    }
    assert all(len(value) == 64 for value in hashes.values())


def test_observation_contract_rejects_low_level_or_nonfinite_fields() -> None:
    assert validate_observation(observation())["schema_version"] == OBSERVATION_SCHEMA
    raw = observation() | {"qpos": [0.0] * 17}
    with pytest.raises(ValueError, match="fields"):
        validate_observation(raw)
    raw = observation()
    raw["heading_yaw_rad"] = math.nan
    with pytest.raises(ValueError, match="finite"):
        validate_observation(raw)


def test_action_boundary_distinguishes_requested_and_clamped_applied_actions() -> None:
    requested = {
        "schema_version": ACTION_SCHEMA,
        "type": "velocity",
        "forward_mps": 4.0,
        "lateral_mps": -2.0,
        "yaw_rate_rps": 3.0,
    }
    result = validate_action(requested, load_runtime_contract())
    assert result.requested_action == requested
    assert result.applied_action == {
        "schema_version": ACTION_SCHEMA,
        "type": "velocity",
        "forward_mps": 0.75,
        "lateral_mps": -0.25,
        "yaw_rate_rps": 0.6,
    }
    assert result.clamped and not result.rejected


@pytest.mark.parametrize("invalid", [None, {}, {"schema_version": ACTION_SCHEMA, "type": "velocity", "forward_mps": math.nan, "lateral_mps": 0.0, "yaw_rate_rps": 0.0}])
def test_invalid_or_nonfinite_actions_reject_to_safe_stop(invalid: object) -> None:
    result = validate_action(invalid, load_runtime_contract())
    assert result.rejected
    assert result.requested_action is None
    assert result.applied_action == safe_stop_action()


def test_simulated_time_scheduler_records_latency_without_advancing_observation_time() -> None:
    ticks = iter((100_000_000, 107_500_000))
    scheduler = SimulatedTimeAgentScheduler(ScriptedBaselineAgent(), clock_ns=lambda: next(ticks))
    decision = scheduler.decide(observation())
    assert decision.simulation_time == 1.0
    assert decision.wall_latency_ms == 7.5
    assert scheduler.steps_per_decision(0.02) == 10
    assert decision.applied_action["type"] == "velocity"


def test_scripted_agent_uses_only_the_public_serializable_observation() -> None:
    public_observation = json.loads(json.dumps(observation()))
    action = ScriptedBaselineAgent().act(public_observation)
    assert set(action) == {"schema_version", "type", "forward_mps", "lateral_mps", "yaw_rate_rps"}
    assert action["forward_mps"] > 0
