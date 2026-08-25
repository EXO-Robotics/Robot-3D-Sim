from __future__ import annotations

import json
import zipfile
from pathlib import Path

import numpy as np
import pytest

from exo_bench.agent_runner import AgentBenchmarkRunner
from exo_bench.live import LiveMessageBuffer, parse_live_message
from exo_bench.recorder import inspect_replay


@pytest.mark.parametrize("seed", range(5))
def test_agent_001_scripted_scenario_matrix_passes(seed: int) -> None:
    result = AgentBenchmarkRunner.from_aliases().run(seed)
    assert result.passed, result.to_dict()
    assert result.reason == "complete"
    assert result.falls == 0
    assert result.collisions is None
    assert result.final_position_error_m <= 0.75
    assert result.final_heading_error_rad <= 0.35
    assert result.final_speed_mps <= 0.18
    assert result.rejected_action_count == 0


@pytest.mark.parametrize("seed", range(5))
def test_agent_001_is_exact_same_runtime_deterministic(seed: int) -> None:
    runner = AgentBenchmarkRunner.from_aliases()
    first = runner.run(seed)
    repeated = runner.run(seed)
    assert first.determinism_digest == repeated.determinism_digest
    first_stable = first.to_dict()
    repeated_stable = repeated.to_dict()
    for value in (first_stable, repeated_stable):
        value.pop("agent_wall_latency_ms")
    assert first_stable == repeated_stable


def test_agent_exorun_contains_strict_v2_agent_and_body_pose_payloads(tmp_path: Path) -> None:
    path = tmp_path / "agent.exorun"
    result = AgentBenchmarkRunner.from_aliases().run(0, record_path=path)
    summary = inspect_replay(path, require_archive_receipt=True)
    assert result.passed
    assert summary.manifest["schema_version"] == "exo.run.v2"
    assert summary.manifest["determinism"]["digest_schema"] == "exo.agent-episode-digest.v1"
    assert summary.manifest["determinism"]["wall_clock_agent_latency_excluded"] is True
    assert summary.verified_files == 15
    with zipfile.ZipFile(path) as archive:
        replay_index = json.loads(archive.read("replay/index.json"))
        agent_index = json.loads(archive.read("agent/index.json"))
        assert replay_index["schema_version"] == "exo.replay.v2"
        assert replay_index["arrays"]["body_poses.f32"] == [summary.samples, 91]
        assert agent_index["decisions"] == result.agent_decision_count


def test_agent_replay_rejects_nonfinite_recorded_state_even_with_rehashed_payload(tmp_path: Path) -> None:
    source = tmp_path / "source.exorun"
    broken = tmp_path / "broken.exorun"
    AgentBenchmarkRunner.from_aliases().run(0, record_path=source)
    with zipfile.ZipFile(source) as original:
        payloads = {info.filename: original.read(info.filename) for info in original.infolist()}
    qpos = np.frombuffer(payloads["replay/qpos.f32"], dtype="<f4").copy()
    qpos[0] = np.nan
    payloads["replay/qpos.f32"] = qpos.tobytes()
    manifest = json.loads(payloads["manifest.json"])
    import hashlib
    manifest["contents"]["replay/qpos.f32"] = hashlib.sha256(payloads["replay/qpos.f32"]).hexdigest()
    payloads["manifest.json"] = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode()
    with zipfile.ZipFile(broken, "w") as archive:
        for name, payload in payloads.items():
            archive.writestr(name, payload)
    with pytest.raises(ValueError, match="nonfinite"):
        inspect_replay(broken)


def test_agent_replay_rejects_missing_agent_decision_payload(tmp_path: Path) -> None:
    source = tmp_path / "source.exorun"
    broken = tmp_path / "broken.exorun"
    AgentBenchmarkRunner.from_aliases().run(0, record_path=source)
    with zipfile.ZipFile(source) as original, zipfile.ZipFile(broken, "w") as changed:
        for info in original.infolist():
            if info.filename != "agent/decisions.jsonl":
                changed.writestr(info, original.read(info.filename))
    with pytest.raises(ValueError, match="missing required entries"):
        inspect_replay(broken)


def test_agent_run_emits_read_only_versioned_live_frames_and_events() -> None:
    bridge = LiveMessageBuffer()
    result = AgentBenchmarkRunner.from_aliases().run(0, live_sink=bridge)
    payloads, complete = bridge.snapshot()
    messages = [parse_live_message(payload) for payload in payloads]
    frames = [message for message in messages if message["type"] == "frame"]
    assert result.passed and complete
    assert messages[0]["type"] == "hello"
    assert messages[0]["run"]["browser_control_allowed"] is False
    assert messages[-1]["type"] == "complete"
    assert any(event["type"] == "agent_decision" for frame in frames for event in frame["events"])
    assert any(event["type"] == "agent_course_completed" for frame in frames for event in frame["events"])
