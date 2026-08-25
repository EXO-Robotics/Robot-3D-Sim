from pathlib import Path
import zipfile

import pytest

from exo_bench.recorder import inspect_replay
from exo_bench.runner import BenchmarkRunner


@pytest.mark.parametrize("course", ["BASIC-001", "BASIC-002", "BASIC-003", "BASIC-004"])
def test_basic_course_passes_without_trainer_actuators(course: str) -> None:
    result = BenchmarkRunner.from_aliases("h1", "baseline", course).run(seed=42)
    assert result.passed, result.to_dict()
    assert result.falls == 0
    assert result.collisions is None
    assert result.reset_profile == "exo.reset-perturbation.basic.v1"


@pytest.mark.parametrize("course", ["BASIC-001", "BASIC-002", "BASIC-003", "BASIC-004"])
def test_each_basic_course_is_exact_same_runtime_deterministic(course: str) -> None:
    runner = BenchmarkRunner.from_aliases("h1", "baseline", course)
    first = runner.run(seed=42)
    second = runner.run(seed=42)
    assert first.determinism_digest == second.determinism_digest
    assert first.to_dict() | {"determinism_digest": "ignored"} == second.to_dict() | {"determinism_digest": "ignored"}


def test_seeded_reset_profile_changes_the_initial_perturbation() -> None:
    runner = BenchmarkRunner.from_aliases("h1", "baseline", "BASIC-001")
    first = runner.run(seed=0)
    repeated = runner.run(seed=0)
    different = runner.run(seed=1)
    assert first.reset_perturbation == repeated.reset_perturbation
    assert first.reset_perturbation != different.reset_perturbation


def test_exorun_contains_verified_trace_and_recorded_state(tmp_path: Path) -> None:
    path = tmp_path / "result.exorun"
    result = BenchmarkRunner.from_aliases("h1", "baseline", "BASIC-003").run(seed=42, record_path=path)
    summary = inspect_replay(path, require_archive_receipt=True)
    assert result.passed
    assert summary.metrics["passed"] is True
    assert summary.manifest["determinism_digest"] == result.determinism_digest
    assert summary.verified_files == 11
    assert summary.archive_receipt_verified is True
    assert result.archive_sha256 == summary.archive_sha256
    assert path.with_name(path.name + ".sha256").exists()


def test_exorun_archive_bytes_are_reproducible(tmp_path: Path) -> None:
    runner = BenchmarkRunner.from_aliases("h1", "baseline", "BASIC-001")
    first = tmp_path / "first.exorun"
    second = tmp_path / "second.exorun"
    first_result = runner.run(seed=3, record_path=first)
    second_result = runner.run(seed=3, record_path=second)
    assert first_result.archive_sha256 == second_result.archive_sha256
    assert first.read_bytes() == second.read_bytes()


def test_replay_rejects_an_incomplete_archive(tmp_path: Path) -> None:
    source = tmp_path / "source.exorun"
    broken = tmp_path / "broken.exorun"
    BenchmarkRunner.from_aliases("h1", "baseline", "BASIC-001").run(seed=0, record_path=source)
    with zipfile.ZipFile(source, "r") as original, zipfile.ZipFile(broken, "w") as changed:
        for info in original.infolist():
            if info.filename != "trace/actions.f32":
                changed.writestr(info, original.read(info.filename))
    with pytest.raises(ValueError, match="missing required entries"):
        inspect_replay(broken)


def test_replay_rejects_a_mismatched_external_archive_receipt(tmp_path: Path) -> None:
    path = tmp_path / "result.exorun"
    BenchmarkRunner.from_aliases("h1", "baseline", "BASIC-001").run(seed=0, record_path=path)
    path.with_name(path.name + ".sha256").write_text(f"{'0' * 64}  {path.name}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="does not match"):
        inspect_replay(path, require_archive_receipt=True)
