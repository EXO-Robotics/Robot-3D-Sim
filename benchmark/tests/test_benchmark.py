from pathlib import Path

import pytest

from exo_bench.recorder import inspect_replay
from exo_bench.runner import BenchmarkRunner


@pytest.mark.parametrize("course", ["BASIC-001", "BASIC-002", "BASIC-003", "BASIC-004"])
def test_basic_course_passes_without_trainer_actuators(course: str) -> None:
    result = BenchmarkRunner.from_aliases("h1", "baseline", course).run(seed=42)
    assert result.passed, result.to_dict()
    assert result.falls == 0


def test_basic_004_is_deterministic() -> None:
    runner = BenchmarkRunner.from_aliases("h1", "baseline", "BASIC-004")
    first = runner.run(seed=42)
    second = runner.run(seed=42)
    assert first.determinism_digest == second.determinism_digest
    assert first.to_dict() | {"determinism_digest": "ignored"} == second.to_dict() | {"determinism_digest": "ignored"}


def test_exorun_contains_verified_trace_and_recorded_state(tmp_path: Path) -> None:
    path = tmp_path / "result.exorun"
    result = BenchmarkRunner.from_aliases("h1", "baseline", "BASIC-003").run(seed=42, record_path=path)
    summary = inspect_replay(path)
    assert result.passed
    assert summary.metrics["passed"] is True
    assert summary.manifest["determinism_digest"] == result.determinism_digest
    assert summary.verified_files == 11

