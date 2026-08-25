from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any

from .agent import (
    ACTION_SCHEMA,
    OBSERVATION_SCHEMA,
    RUNTIME_SCHEMA,
    ScriptedBaselineAgent,
    SimulatedTimeAgentScheduler,
    load_runtime_contract,
    safe_stop_action,
    validate_action,
    validate_observation,
    verify_agent_artifacts,
)
from .agent_runner import AgentBenchmarkRunner, AgentRunResult
from .hashing import sha256_file
from .paths import AGENT_QUALIFICATION_SPEC_PATH, QUALIFICATION_SPEC_PATH, REPOSITORY_ROOT
from .provenance import git_provenance, runtime_provenance
from .recorder import inspect_replay


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _gate(gate_id: str, passed: bool, evidence: dict[str, Any]) -> dict[str, Any]:
    return {"id": gate_id, "status": "PASS" if passed else "FAIL", "evidence": evidence}


def _stable_result(result: AgentRunResult) -> dict[str, Any]:
    value = result.to_dict()
    for key in (
        "agent_wall_latency_ms",
        "replay_path",
        "archive_sha256",
        "archive_receipt_path",
    ):
        value.pop(key, None)
    return value


class AgentQualifier:
    def __init__(self) -> None:
        self.spec = _read_json(AGENT_QUALIFICATION_SPEC_PATH)
        self.foundation_receipt_path = (
            AGENT_QUALIFICATION_SPEC_PATH.parent / self.spec["foundation"]["receipt"]
        ).resolve()
        self.foundation_receipt = _read_json(self.foundation_receipt_path)
        self.runner = AgentBenchmarkRunner.from_aliases()

    def qualify(self, evidence_dir: Path | None = None) -> dict[str, Any]:
        source = git_provenance()
        runtime = runtime_provenance()
        foundation = self._foundation_evidence()
        contract_gate = self.contract_integrity()
        boundary_gate = self.public_boundary()
        scheduler_gate = self.scheduler_gate()
        matrix_gate, determinism_gate = self.scenario_matrix()
        if evidence_dir is None:
            with tempfile.TemporaryDirectory(prefix="exo-agent-qualification-") as directory:
                replay_gate = self.replay_integrity(Path(directory))
        else:
            replay_gate = self.replay_integrity(evidence_dir)
        gates = [contract_gate, boundary_gate, scheduler_gate, matrix_gate, determinism_gate, replay_gate]
        runtime_pinned = (
            runtime["torch_num_threads"] == 1
            and runtime["torch_num_interop_threads"] == 1
            and runtime["torch_deterministic_algorithms"] is True
        )
        qualified = (
            foundation["qualified"]
            and foundation["receipt_hash_matches"]
            and all(gate["status"] == "PASS" for gate in gates)
            and runtime_pinned
            and not source["dirty"]
        )
        return {
            "schema_version": "exo.agent-qualification.v1",
            "qualification_spec": {
                "id": self.spec["spec_id"],
                "version": self.spec["version"],
                "sha256": sha256_file(AGENT_QUALIFICATION_SPEC_PATH),
            },
            "foundation": foundation,
            "candidate": self.spec["candidate"],
            "exo_bench": source,
            "environment": runtime,
            "artifacts": {
                **self.runner.artifact_hashes,
                **self.runner.agent_hashes,
                "agent_course_sha256": sha256_file(self.runner.artifacts.course_path),
                "foundation_spec_sha256": sha256_file(QUALIFICATION_SPEC_PATH),
            },
            "determinism": self.spec["determinism"],
            "gates": gates,
            "foundation_checks": {
                "foundation_receipt_hash_matches": foundation["receipt_hash_matches"],
                "foundation_receipt_qualified": foundation["qualified"],
                "source_worktree_clean": not source["dirty"],
                "runtime_determinism_controls_pinned": runtime_pinned,
                "all_agent_gates_pass": all(gate["status"] == "PASS" for gate in gates),
                "external_model_api_count": 0,
            },
            "qualified": qualified,
        }

    def _foundation_evidence(self) -> dict[str, Any]:
        expected = self.spec["foundation"]["receipt_sha256"]
        actual = sha256_file(self.foundation_receipt_path)
        candidate = self.foundation_receipt.get("candidate", {})
        expected_candidate = {
            "robot": self.spec["foundation"]["robot"],
            "controller": self.spec["foundation"]["controller"],
        }
        return {
            "receipt": self.foundation_receipt_path.relative_to(REPOSITORY_ROOT).as_posix(),
            "receipt_sha256": actual,
            "expected_receipt_sha256": expected,
            "receipt_hash_matches": actual == expected,
            "qualified": self.foundation_receipt.get("qualified") is True and candidate == expected_candidate,
            "candidate": candidate,
        }

    def contract_integrity(self) -> dict[str, Any]:
        hashes = verify_agent_artifacts()
        runtime = load_runtime_contract()
        manifest = self.runner.agent_manifest
        passed = (
            manifest["contracts"]["observation"]["id"] == OBSERVATION_SCHEMA
            and manifest["contracts"]["action"]["id"] == ACTION_SCHEMA
            and manifest["contracts"]["runtime"]["id"] == RUNTIME_SCHEMA
            and runtime["schema_version"] == RUNTIME_SCHEMA
            and all(len(value) == 64 for value in hashes.values())
        )
        return _gate("AQ-001-contract-integrity", passed, {"hashes": hashes, "runtime": runtime})

    def public_boundary(self) -> dict[str, Any]:
        manifest = self.runner.agent_manifest
        scenario = self.runner.artifacts.course["scenario_set"]["scenarios"][0]
        observation = {
            "schema_version": OBSERVATION_SCHEMA,
            "simulation_time_s": 0.0,
            "base_position_m": {"x": 0.0, "y": 0.0, "z": 1.05},
            "heading_yaw_rad": float(scenario["start_heading_rad"]),
            "planar_velocity_mps": {"x": 0.0, "y": 0.0},
            "task": {
                "id": "AGENT-001@1.0.0",
                "goal_position_m": {
                    "x": float(scenario["target_position_m"][0]),
                    "y": float(scenario["target_position_m"][1]),
                },
                "goal_heading_rad": float(scenario["goal_heading_rad"]),
                "target_radius_m": float(self.runner.artifacts.course["success"]["target_radius_m"]),
                "heading_tolerance_rad": float(self.runner.artifacts.course["success"]["heading_tolerance_rad"]),
            },
            "remaining_time_s": 35.0,
            "previous_applied_action": None,
        }
        validate_observation(observation)
        action = ScriptedBaselineAgent().act(observation)
        validation = validate_action(action, load_runtime_contract())
        forbidden_observation_fields = {"qpos", "qvel", "joints", "torques", "mujoco", "environment"}
        passed = (
            not forbidden_observation_fields.intersection(observation)
            and not validation.rejected
            and manifest["interface"]["environment_internals"] is False
            and manifest["interface"]["low_level_joint_or_torque_control"] is False
            and manifest["network_access"] is False
        )
        return _gate(
            "AQ-002-public-boundary",
            passed,
            {
                "observation_fields": sorted(observation),
                "action_fields": sorted(action),
                "environment_internals": False,
                "low_level_joint_or_torque_control": False,
                "network_access": False,
            },
        )

    def scheduler_gate(self) -> dict[str, Any]:
        ticks = iter((1_000_000_000, 1_004_000_000))
        scheduler = SimulatedTimeAgentScheduler(ScriptedBaselineAgent(), clock_ns=lambda: next(ticks))
        steps = scheduler.steps_per_decision(0.02)
        scenario = self.runner.artifacts.course["scenario_set"]["scenarios"][0]
        observation = {
            "schema_version": OBSERVATION_SCHEMA,
            "simulation_time_s": 1.25,
            "base_position_m": {"x": 0.0, "y": 0.0, "z": 1.05},
            "heading_yaw_rad": 0.0,
            "planar_velocity_mps": {"x": 0.0, "y": 0.0},
            "task": {
                "id": "AGENT-001@1.0.0",
                "goal_position_m": {"x": 3.0, "y": 0.0},
                "goal_heading_rad": float(scenario["goal_heading_rad"]),
                "target_radius_m": 0.75,
                "heading_tolerance_rad": 0.35,
            },
            "remaining_time_s": 33.75,
            "previous_applied_action": safe_stop_action(),
        }
        decision = scheduler.decide(observation)
        passed = (
            steps == 10
            and decision.simulation_time == 1.25
            and decision.wall_latency_ms == 4.0
            and scheduler.runtime["scheduler"] == "simulated_time_paused"
        )
        return _gate(
            "AQ-003-simulated-time-scheduler",
            passed,
            {
                "decision_interval_s": scheduler.decision_interval_s,
                "policy_control_period_s": 0.02,
                "low_level_steps_per_decision": steps,
                "measured_fixture_wall_latency_ms": decision.wall_latency_ms,
                "wall_latency_affects_physics": False,
            },
        )

    def scenario_matrix(self) -> tuple[dict[str, Any], dict[str, Any]]:
        cases: list[dict[str, Any]] = []
        for seed in self.spec["scenario_seeds"]:
            first = self.runner.run(int(seed))
            repeated = self.runner.run(int(seed))
            deterministic = (
                first.determinism_digest == repeated.determinism_digest
                and _stable_result(first) == _stable_result(repeated)
            )
            case_passed = first.passed and repeated.passed and deterministic
            cases.append(
                {
                    "seed": seed,
                    "status": "PASS" if case_passed else "FAIL",
                    "episode_passed": first.passed and repeated.passed,
                    "reason": first.reason,
                    "deterministic": deterministic,
                    "digest": first.determinism_digest,
                    "repeat_digest": repeated.determinism_digest,
                    "simulated_completion_time_s": first.simulated_completion_time_s,
                    "path_length_m": first.path_length_m,
                    "final_position_error_m": first.final_position_error_m,
                    "final_heading_error_rad": first.final_heading_error_rad,
                    "final_speed_mps": first.final_speed_mps,
                    "final_window_average_speed_mps": first.final_window_average_speed_mps,
                    "falls": first.falls,
                    "energy_joules": first.energy_joules,
                    "agent_decision_count": first.agent_decision_count,
                    "clamped_action_count": first.clamped_action_count,
                    "rejected_action_count": first.rejected_action_count,
                    "scenario": first.scenario,
                    "reset_perturbation": first.reset_perturbation,
                }
            )
        matrix_passed = len(cases) == len(self.spec["scenario_seeds"]) and all(
            case["status"] == "PASS" for case in cases
        )
        deterministic = all(case["deterministic"] for case in cases)
        return (
            _gate(
                "AQ-004-scenario-matrix",
                matrix_passed,
                {"course": "AGENT-001@1.0.0", "case_count": len(cases), "cases": cases},
            ),
            _gate(
                "AQ-005-exact-same-runtime-determinism",
                deterministic,
                {
                    "class": "EXACT-SAME-RUNTIME",
                    "digest_schema": "exo.agent-episode-digest.v1",
                    "repeat_each": True,
                    "wall_clock_agent_latency_excluded": True,
                    "digests": [case["digest"] for case in cases],
                },
            ),
        )

    def replay_integrity(self, evidence_dir: Path) -> dict[str, Any]:
        evidence_dir.mkdir(parents=True, exist_ok=True)
        replay_path = evidence_dir / "AGENT-001-seed-0.exorun"
        result = self.runner.run(0, record_path=replay_path)
        summary = inspect_replay(replay_path, require_archive_receipt=True)
        passed = (
            result.passed
            and summary.manifest["schema_version"] == "exo.run.v2"
            and summary.manifest["determinism_digest"] == result.determinism_digest
            and summary.verified_files == 15
            and summary.archive_receipt_verified
            and summary.metrics["agent_decision_count"] == result.agent_decision_count
        )
        return _gate(
            "AQ-006-replay-integrity",
            passed,
            {
                "path": replay_path.relative_to(REPOSITORY_ROOT).as_posix()
                if replay_path.is_relative_to(REPOSITORY_ROOT)
                else replay_path.name,
                "schema_version": summary.manifest["schema_version"],
                "archive_sha256": summary.archive_sha256,
                "archive_receipt": replay_path.with_name(replay_path.name + ".sha256").name,
                "archive_receipt_verified": summary.archive_receipt_verified,
                "verified_payloads": summary.verified_files,
                "samples": summary.samples,
                "agent_decisions": result.agent_decision_count,
                "digest": result.determinism_digest,
            },
        )


def write_agent_qualification(report: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, allow_nan=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
