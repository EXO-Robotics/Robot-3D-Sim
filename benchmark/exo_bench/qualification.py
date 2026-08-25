from __future__ import annotations

import copy
import json
import math
import platform
from pathlib import Path

import mujoco
import numpy as np

from .course import angle_delta
from .environment import NativeH1Environment
from .policy import TorchScriptVelocityPolicy
from .registry import ArtifactRegistry
from .runner import BenchmarkRunner


def _gate(gate_id: str, passed: bool, evidence: dict) -> dict:
    return {"id": gate_id, "status": "PASS" if passed else "FAIL", "evidence": evidence}


class ChassisQualifier:
    def __init__(self, robot: str = "h1", controller: str = "baseline") -> None:
        self.artifacts = ArtifactRegistry().resolve(robot, controller, "BASIC-001")
        self.runner = BenchmarkRunner(self.artifacts)
        self.scene_path = self.artifacts.robot_dir / self.artifacts.robot["model"]["entrypoint"]
        self.policy_path = self.artifacts.controller_dir / self.artifacts.controller["policy"]["path"]

    def qualify(self) -> dict:
        gates = [
            self.model_integrity(),
            self.passive_physics(),
            self.policy_stand(),
            self.velocity_tracking(),
            self.stop(),
            self.turn(),
            self.perturbation(),
        ]
        return {
            "schema_version": "exo.qualification.v1",
            "candidate": f"{self.artifacts.robot_id}+{self.artifacts.controller_id}",
            "source_commit": self.artifacts.robot["source"]["commit"],
            "artifact_hashes": self.runner.artifact_hashes,
            "runtime": {
                "mujoco_version": mujoco.__version__,
                "python_version": platform.python_version(),
                "platform": platform.platform(),
            },
            "gates": gates,
            "qualified": all(gate["status"] == "PASS" for gate in gates),
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
        trials = [self.runner.run(seed=seed, course_override=course, joint_noise=0.002) for seed in range(10)]
        passed = all(trial.passed for trial in trials)
        return _gate(
            "CQ-003",
            passed,
            {
                "trials": 10,
                "falls": sum(trial.falls for trial in trials),
                "nan_trials": sum(trial.reason == "invalid_state" for trial in trials),
                "digests": [trial.determinism_digest for trial in trials],
            },
        )

    def velocity_tracking(self) -> dict:
        trials = [self._velocity_trial(target) for target in (0.25, 0.5, 0.75)]
        passed = all(not trial["fallen"] and trial["rmse_mps"] <= 0.45 for trial in trials)
        return _gate("CQ-004", passed, {"trials": trials, "rmse_limit_mps": 0.45})

    def stop(self) -> dict:
        artifacts = ArtifactRegistry().resolve("h1", "baseline", "BASIC-003")
        result = BenchmarkRunner(artifacts).run(seed=42)
        return _gate("CQ-005", result.passed, result.to_dict())

    def turn(self) -> dict:
        trials = [self._turn_trial(target) for target in (math.pi / 2, -math.pi / 2, math.pi)]
        passed = all(not trial["fallen"] and trial["error_radians"] <= 0.2 for trial in trials)
        return _gate("CQ-006", passed, {"trials": trials, "error_limit_radians": 0.2})

    def perturbation(self) -> dict:
        environment = self._environment()
        policy = self._policy()
        state = environment.reset(seed=7)
        policy.reset()
        zero = np.zeros(3, dtype=np.float32)
        for _ in range(int(2.0 / environment.control_period)):
            obs = policy.observation(state.time, state.qpos, state.qvel, zero)
            _, motor = policy.act(obs)
            state = environment.advance(motor)
        before = state.base_position.copy()
        environment.apply_push((120.0, 0.0, 0.0))
        for _ in range(5):
            obs = policy.observation(state.time, state.qpos, state.qvel, zero)
            _, motor = policy.act(obs)
            state = environment.advance(motor)
        environment.clear_push()
        displacement = float(np.linalg.norm(state.base_position - before))
        passed = displacement > 0.001 and np.isfinite(state.qpos).all()
        return _gate(
            "CQ-007",
            passed,
            {
                "push_newtons": [120.0, 0.0, 0.0],
                "duration_seconds": 0.1,
                "base_displacement_meters": displacement,
                "finite_state": bool(np.isfinite(state.qpos).all()),
                "fallen": state.fallen,
            },
        )

    def _velocity_trial(self, target: float) -> dict:
        environment = self._environment()
        policy = self._policy()
        state = environment.reset(seed=42)
        policy.reset()
        command = np.asarray([target, 0.0, 0.0], dtype=np.float32)
        samples: list[float] = []
        for _ in range(int(8.0 / environment.control_period)):
            obs = policy.observation(state.time, state.qpos, state.qvel, command)
            _, motor = policy.act(obs)
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
        state = environment.reset(seed=42)
        policy.reset()
        previous_yaw = state.yaw
        accumulated = 0.0
        rate = 0.5 if target > 0 else -0.5
        for _ in range(int(15.0 / environment.control_period)):
            command = np.asarray([0.0, 0.0, rate], dtype=np.float32)
            obs = policy.observation(state.time, state.qpos, state.qvel, command)
            _, motor = policy.act(obs)
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
