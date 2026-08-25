from __future__ import annotations

import math

import pytest

from exo_bench.agent import safe_stop_action
from exo_bench.live import LiveMessageBuffer, parse_live_message, serialize_live_message


def frame() -> dict:
    action = safe_stop_action()
    observation = {
        "schema_version": "exo.agent.observation.v1",
        "simulation_time_s": 0.0,
        "base_position_m": {"x": 0.0, "y": 0.0, "z": 1.05},
        "heading_yaw_rad": 0.0,
        "planar_velocity_mps": {"x": 0.0, "y": 0.0},
        "task": {
            "id": "AGENT-001@1.0.0",
            "goal_position_m": {"x": 3.0, "y": 0.0},
            "goal_heading_rad": 0.0,
            "target_radius_m": 0.75,
            "heading_tolerance_rad": 0.35,
        },
        "remaining_time_s": 35.0,
        "previous_applied_action": None,
    }
    return {
        "schema_version": "exo.live.v1",
        "type": "frame",
        "sequence": 0,
        "simulation_time": 0.02,
        "qpos": [0.0] * 17,
        "qvel": [0.0] * 16,
        "body_poses": [0.0] * 91,
        "task": {"phase": "navigate"},
        "observation": observation,
        "requested_action": action,
        "applied_action": action,
        "agent_latency_ms": 0.5,
        "telemetry": {"speed_mps": 0.0},
        "events": [],
    }


def test_live_protocol_round_trips_canonical_visualization_frames() -> None:
    payload = serialize_live_message(frame())
    assert parse_live_message(payload) == frame()


def test_live_protocol_rejects_nonfinite_or_wrong_width_state() -> None:
    invalid = frame()
    invalid["qpos"] = [0.0] * 16
    with pytest.raises(ValueError, match="17 finite"):
        serialize_live_message(invalid)
    invalid = frame()
    invalid["agent_latency_ms"] = math.nan
    with pytest.raises(ValueError, match="nonfinite"):
        serialize_live_message(invalid)


def test_live_message_buffer_is_append_only_and_has_no_control_input() -> None:
    bridge = LiveMessageBuffer()
    bridge.hello({
        "schema_version": "exo.live.v1",
        "type": "hello",
        "run": {"browser_control_allowed": False},
    })
    bridge.frame(frame())
    bridge.complete({
        "schema_version": "exo.live.v1",
        "type": "complete",
        "sequence": 1,
        "simulation_time": 0.02,
        "result": {"passed": True},
    })
    messages, complete = bridge.snapshot()
    assert len(messages) == 3
    assert complete
    assert not hasattr(bridge, "command")
