from __future__ import annotations

import copy
import json
import math
import tempfile
from pathlib import Path
from typing import Any

import mujoco
import numpy as np

from .course import angle_delta
from .environment import NativeH1Environment
from .hashing import sha256_file
from .paths import QUALIFICATION_SPEC_PATH, REPOSITORY_ROOT
from .policy import TorchScriptVelocityPolicy
from .provenance import git_provenance, manifest_hashes, runtime_provenance
from .recorder import inspect_replay
from .registry import ArtifactRegistry
from .runner import BenchmarkRunner, RunResult


def _gate(gate_id: str, passed: bool, evidence: dict) -> dict:
    return {"id": gate_id, "status": "PASS" if passed else "FAIL", "evidence": evidence}


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _relative_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(REPOSITORY_ROOT).as_posix()
    except ValueError:
        return path.name


def _comparable_result(result: RunResult) -> dict[str, Any]:
    value = result.to_dict()
    for key in ("replay_path", "archive_sha256", "archive_receipt_path"):
        value.pop(key, None)
    return value


class ChassisQualifier:
    def __init__(self, robot: str = "h1", controller: str = "baseline") -> None:
        self.registry = ArtifactRegistry()
        self.artifacts = self.registry.resolve(robot, controller, "BASIC-001")
        self.runner = BenchmarkRunner(self.artifacts)
        self.scene_path = self.artifacts.robot_dir / self.artifacts.robot["model"]["entrypoint"]
        self.policy_path = self.artifacts.controller_dir / self.artifacts.controller["policy"]["path"]
        self.spec = _read_json(QUALIFICATION_SPEC_PATH)
        candidate = self.spec["candidate"]
        if candidate != {
            "robot": self.artifacts.robot_id,
            "controller": self.artifacts.controller_id,
        }:
            raise ValueError("Qualification specification is not bound to the resolved artifact pair")
        matrix = self.spec["basic_matrix"]
        if matrix["reset_profile"] != self.artifacts.reset_profile_id:
            raise ValueError("Qualification specification is not bound to the resolved reset profile")

    def qualify(self, evidence_dir: Path | None = None) -> dict:
        source = git_provenance()
        runtime = runtime_provenance()
        gates = [
            self.model_integrity(),
            self.passive_physics(),
            self.policy_stand(),
            self.velocity_tracking(),
            self.stop(),
            self.turn(),
            self.perturbation_recovery(),
        ]
        matrix = self.basic_matrix()
        if evidence_dir is None:
            with tempfile.TemporaryDirectory(prefix="exo-qualification-") as temporary:
                replay = self.replay_integrity(Path(temporary))
        else:
            replay = self.replay_integrity(evidence_dir)
        course_hashes: dict[str, str] = {}
        for reference in self.spec["basic_matrix"]["courses"]:
            course_artifacts = self.registry.resolve(
                self.artifacts.robot_id,
                self.artifacts.controller_id,
                reference,
                self.artifacts.reset_profile_id,
            )
            course_hashes[reference] = sha256_file(course_artifacts.course_path)
        manifests = manifest_hashes(
            self.artifacts.robot_dir / "manifest.json",
            self.artifacts.controller_dir / "manifest.json",
        )
        runtime_pinned = (
            runtime["torch_num_threads"] == 1
            and runtime["torch_num_interop_threads"] == 1
            and runtime["torch_deterministic_algorithms"] is True
        )
        all_gates_pass = all(gate["status"] == "PASS" for gate in gates)
        qualified = (
            all_gates_pass
            and matrix["status"] == "PASS"
            and replay["status"] == "PASS"
            and not source["dirty"]
            and runtime_pinned
        )
        return {
            "schema_version": "exo.qualification.v1",
            "qualification_spec": {
                "id": self.spec["spec_id"],
                "version": self.spec["version"],
                "sha256": sha256_file(QUALIFICATION_SPEC_PATH),
            },
            "candidate": {
                "robot": self.artifacts.robot_id,
                "controller": self.artifacts.controller_id,
            },
            "exo_bench": source,
            "environment": runtime,
            "artifacts": {
                **manifests,
                **self.runner.artifact_hashes,
                "source_commit": self.artifacts.robot["source"]["commit"],
            },
            "reset_profile": {
                "id": self.artifacts.reset_profile_id,
                "sha256": self.runner.artifact_hashes["reset_profile_sha256"],
            },
            "courses": course_hashes,
            "determinism": self.spec["determinism"],
            "collision_metric": self.spec["collision_metric"],
            "gates": gates,
            "basic_matrix": matrix,
            "replay_integrity": replay,
            "foundation_checks": {
                "source_worktree_clean": not source["dirty"],
                "runtime_determinism_controls_pinned": runtime_pinned,
                "all_chassis_gates_pass": all_gates_pass,
                "all_basic_matrix_cases_pass": matrix["status"] == "PASS",
                "replay_and_external_archive_receipt_pass": replay["status"] == "PASS",
            },
            "qualified": qualified,
        }

    def model_integrity(self) -> dict:
        model = mujoco.MjModel.from_xml_path(str(self.scene_path))
        free_joints = int(np.count_nonzero(model.jnt_type == mujoco.mjtJoint.mjJNT_FREE))
        actuator_names = [
            mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, index) or ""
            for index in range(model.nu)
        ]
        forbidden = set(self.artifacts.robot["model"]["trainer_actuators_forbidden"])
        trainer_actuators = sorted(forbidden.intersection(actuator_names))
        finite_mass = bool(np.isfinite(model.body_mass[1:]).all() and np.all(model.body_mass[1:] > 0))
        finite_inertia = bool(np.isfinite(model.body_inertia[1:]).all() and np.all(model.body_inertia[1:] > 0))
        expected_actuators = int(self.artifacts.robot["model"]["expected_actuators"])
        expected_nq = int(self.artifacts.robot["model"]["expected_nq"])
        expected_nv = int(self.artifacts.robot["model"]["expected_nv"])
        passed = (
            free_joints == 1
            and model.nu == expected_actuators
            and model.nq == expected_nq
            and model.nv == expected_nv
            and not trainer_actuators
            and finite_mass
            and finite_inertia
        )
        return _gate(
            "CQ-001",
            passed,
            {
                "free_joints": free_joints,
                "actuators": model.nu,
                "expected_actuators": expected_actuators,
                "nq": model.nq,
                "expected_nq": expected_nq,
                "nv": model.nv,
                "expected_nv": expected_nv,
                "trainer_actuators": trainer_actuators,
                "finite_positive_mass": finite_mass,
                "finite_positive_inertia": finite_inertia,
            },
        )

    def passive_physics(self) -> dict:
        environment = self._environment()
        initial = environment.reset(seed=0)
        state = initial
        steps = int(5.0 / environment.timestep)
        for _ in range(steps):
            state = environment.passive_step()
            if state.fallen:
                break
        displacement = float(np.linalg.norm(state.base_position - initial.base_position))
        passed = state.fallen and displacement > 0.05
        return _gate(
            "CQ-002",
            passed,
            {
                "fallen": state.fallen,
                "fall_time_seconds": state.time,
                "base_displacement_meters": displacement,
                "trainer_forces": 0,
            },
        )

    def policy_stand(self) -> dict:
        course = copy.deepcopy(self.artifacts.course)
        course["limits"]["time_seconds"] = 30.0
        course["success"]["minimum_upright_seconds"] = 30.0
        trials = [self.runner.run(seed=seed, course_override=course) for seed in range(10)]
        passed = all(trial.passed for trial in trials)
        return _gate(
            "CQ-003",
            passed,
            {
                "trials": 10,
                "falls": sum(trial.falls for trial in trials),
                "nan_trials": sum(trial.reason == "invalid_state" for trial in trials),
                "digests": [trial.determinism_digest for trial in trials],
                "reset_profile": self.artifacts.reset_profile_id,
            },
        )

    def velocity_tracking(self) -> dict:
        trials = [self._velocity_trial(target) for target in (0.25, 0.5, 0.75)]
        passed = all(not trial["fallen"] and trial["rmse_mps"] <= 0.45 for trial in trials)
        return _gate("CQ-004", passed, {"trials": trials, "rmse_limit_mps": 0.45})

    def stop(self) -> dict:
        artifacts = self.registry.resolve("h1", "baseline", "BASIC-003")
        result = BenchmarkRunner(artifacts).run(seed=42)
        return _gate("CQ-005", result.passed, result.to_dict())

    def turn(self) -> dict:
        trials = [self._turn_trial(target) for target in (math.pi / 2, -math.pi / 2, math.pi)]
        passed = all(not trial["fallen"] and trial["error_radians"] <= 0.2 for trial in trials)
        return _gate("CQ-006", passed, {"trials": trials, "error_limit_radians": 0.2})

    def perturbation_recovery(self) -> dict:
        limits = self.spec["disturbance_rejection"]
        environment = self._environment()
        policy = self._policy()
        state = environment.reset(seed=7, reset_profile=self.artifacts.reset_profile)
        policy.reset()
        zero = np.zeros(3, dtype=np.float32)

        def advance_zero():
            observation = policy.observation(state.time, state.qpos, state.qvel, zero)
            _, motor_command = policy.act(observation)
            return environment.advance(motor_command)

        for _ in range(round(float(limits["settle_seconds"]) / environment.control_period)):
            state = advance_zero()
        before = state.base_position.copy()
        push = tuple(float(value) for value in limits["push_newtons"])
        environment.apply_push(push)
        fell_during_trial = False
        try:
            for _ in range(round(float(limits["push_seconds"]) / environment.control_period)):
                state = advance_zero()
                fell_during_trial = fell_during_trial or state.fallen
        finally:
            environment.clear_push()
        force_cleared = bool(np.allclose(environment.data.xfrc_applied[environment.pelvis_body_id], 0.0))
        final_window: list[tuple[float, np.ndarray]] = []
        recovery_steps = round(float(limits["recovery_seconds"]) / environment.control_period)
        final_window_start = state.time + float(limits["recovery_seconds"]) - float(limits["final_window_seconds"])
        for _ in range(recovery_steps):
            state = advance_zero()
            fell_during_trial = fell_during_trial or state.fallen
            if state.time >= final_window_start:
                final_window.append((state.time, state.base_position[:2].copy()))
        if len(final_window) >= 2:
            elapsed = final_window[-1][0] - final_window[0][0]
            positions = np.asarray([sample[1] for sample in final_window])
            path_distance = float(np.sum(np.linalg.norm(np.diff(positions, axis=0), axis=1)))
            net_drift = float(np.linalg.norm(positions[-1] - positions[0]))
            average_speed = path_distance / elapsed
        else:
            net_drift = float("inf")
            average_speed = float("inf")
        finite_state = bool(np.isfinite(state.qpos).all() and np.isfinite(state.qvel).all())
        disturbance_displacement = float(np.linalg.norm(state.base_position - before))
        passed = (
            finite_state
            and not fell_during_trial
            and state.height >= float(limits["minimum_final_height_m"])
            and state.tilt <= float(limits["maximum_final_tilt_rad"])
            and average_speed <= float(limits["maximum_final_window_average_speed_mps"])
            and force_cleared
        )
        return _gate(
            "CQ-007",
            passed,
            {
                "name": "perturbation recovery / disturbance rejection",
                "settle_seconds": limits["settle_seconds"],
                "push_newtons": list(push),
                "push_seconds": limits["push_seconds"],
                "recovery_seconds": limits["recovery_seconds"],
                "disturbance_displacement_meters": disturbance_displacement,
                "finite_state": finite_state,
                "fallen": fell_during_trial,
                "final_height_m": state.height,
                "minimum_final_height_m": limits["minimum_final_height_m"],
                "final_tilt_rad": state.tilt,
                "maximum_final_tilt_rad": limits["maximum_final_tilt_rad"],
                "final_window_average_speed_mps": average_speed,
                "final_window_net_drift_meters": net_drift,
                "maximum_final_window_average_speed_mps": limits[
                    "maximum_final_window_average_speed_mps"
                ],
                "external_force_cleared": force_cleared,
            },
        )

    def basic_matrix(self) -> dict:
        specification = self.spec["basic_matrix"]
        cases: list[dict[str, Any]] = []
        for course in specification["courses"]:
            runner = BenchmarkRunner.from_aliases(
                self.artifacts.robot_id,
                self.artifacts.controller_id,
                course,
                self.artifacts.reset_profile_id,
            )
            for seed in specification["seeds"]:
                first = runner.run(seed=int(seed))
                repeated = runner.run(seed=int(seed))
                result_match = _comparable_result(first) == _comparable_result(repeated)
                deterministic = first.determinism_digest == repeated.determinism_digest and result_match
                passed = first.passed and repeated.passed and deterministic
                cases.append(
                    {
                        "course": course,
                        "seed": seed,
                        "status": "PASS" if passed else "FAIL",
                        "episode_passed": first.passed and repeated.passed,
                        "deterministic": deterministic,
                        "digest": first.determinism_digest,
                        "repeat_digest": repeated.determinism_digest,
                        "reason": first.reason,
                        "time_seconds": first.time_seconds,
                        "progress_meters": first.progress_meters,
                        "falls": first.falls,
                        "collisions": first.collisions,
                        "energy_joules": first.energy_joules,
                        "peak_torque_nm": first.peak_torque_nm,
                        "reset_perturbation": first.reset_perturbation,
                    }
                )
        passed = len(cases) == 20 and all(case["status"] == "PASS" for case in cases)
        return {
            "status": "PASS" if passed else "FAIL",
            "courses": specification["courses"],
            "seeds": specification["seeds"],
            "reset_profile": self.artifacts.reset_profile_id,
            "repeat_each": specification["repeat_each"],
            "case_count": len(cases),
            "passed_cases": sum(case["status"] == "PASS" for case in cases),
            "cases": cases,
        }

    def replay_integrity(self, evidence_dir: Path) -> dict:
        evidence_dir.mkdir(parents=True, exist_ok=True)
        path = evidence_dir / "BASIC-004-seed-0.exorun"
        artifacts = self.registry.resolve("h1", "baseline", "BASIC-004")
        result = BenchmarkRunner(artifacts).run(seed=0, record_path=path)
        summary = inspect_replay(path, require_archive_receipt=True)
        passed = (
            result.passed
            and summary.metrics["passed"] is True
            and summary.manifest["determinism_digest"] == result.determinism_digest
            and summary.verified_files == 11
            and summary.archive_receipt_verified
        )
        return {
            "status": "PASS" if passed else "FAIL",
            "path": _relative_path(path),
            "archive_receipt_path": _relative_path(path.with_name(path.name + ".sha256")),
            "archive_sha256": summary.archive_sha256,
            "size_bytes": path.stat().st_size,
            "verified_payloads": summary.verified_files,
            "samples": summary.samples,
            "archive_receipt_verified": summary.archive_receipt_verified,
            "determinism_digest": result.determinism_digest,
        }

    def _velocity_trial(self, target: float) -> dict:
        environment = self._environment()
        policy = self._policy()
        state = environment.reset(seed=42, reset_profile=self.artifacts.reset_profile)
        policy.reset()
        command = np.asarray([target, 0.0, 0.0], dtype=np.float32)
        samples: list[float] = []
        for _ in range(int(8.0 / environment.control_period)):
            observation = policy.observation(state.time, state.qpos, state.qvel, command)
            _, motor = policy.act(observation)
            state = environment.advance(motor)
            if state.time >= 2.0:
                samples.append(float(state.qvel[0]))
            if state.fallen:
                break
        rmse = float(np.sqrt(np.mean((np.asarray(samples) - target) ** 2))) if samples else float("inf")
        return {
            "target_mps": target,
            "rmse_mps": rmse,
            "mean_mps": float(np.mean(samples)) if samples else 0.0,
            "fallen": state.fallen,
            "energy_joules": state.energy_joules,
        }

    def _turn_trial(self, target: float) -> dict:
        environment = self._environment()
        policy = self._policy()
        state = environment.reset(seed=42, reset_profile=self.artifacts.reset_profile)
        policy.reset()
        previous_yaw = state.yaw
        accumulated = 0.0
        rate = 0.5 if target > 0 else -0.5
        for _ in range(int(15.0 / environment.control_period)):
            command = np.asarray([0.0, 0.0, rate], dtype=np.float32)
            observation = policy.observation(state.time, state.qpos, state.qvel, command)
            _, motor = policy.act(observation)
            state = environment.advance(motor)
            accumulated += angle_delta(state.yaw, previous_yaw)
            previous_yaw = state.yaw
            if abs(accumulated) >= abs(target) - 0.08 or state.fallen:
                break
        return {
            "target_radians": target,
            "actual_radians": accumulated,
            "error_radians": abs(abs(target) - abs(accumulated)),
            "fallen": state.fallen,
        }

    def _environment(self) -> NativeH1Environment:
        return NativeH1Environment(self.scene_path, self.artifacts.robot, self.artifacts.controller)

    def _policy(self) -> TorchScriptVelocityPolicy:
        return TorchScriptVelocityPolicy(self.policy_path, self.artifacts.controller)


def write_qualification(report: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
