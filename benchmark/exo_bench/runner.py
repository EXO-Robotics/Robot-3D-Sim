from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .course import CourseRuntime
from .environment import NativeH1Environment
from .policy import TorchScriptVelocityPolicy
from .provenance import git_provenance, runtime_provenance
from .recorder import EpisodeRecorder
from .registry import ArtifactRegistry, ResolvedArtifacts


@dataclass(frozen=True)
class RunResult:
    robot: str
    controller: str
    course: str
    seed: int
    passed: bool
    reason: str
    time_seconds: float
    progress_meters: float
    falls: int
    collisions: int | None
    energy_joules: float
    peak_torque_nm: float
    final_speed_mps: float
    stop_average_speed_mps: float | None
    determinism_digest: str
    replay_path: str | None
    archive_sha256: str | None
    archive_receipt_path: str | None
    reset_profile: str
    reset_perturbation: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class BenchmarkRunner:
    def __init__(self, artifacts: ResolvedArtifacts) -> None:
        self.artifacts = artifacts
        self.artifact_hashes = ArtifactRegistry.verify_hashes(artifacts)

    @classmethod
    def from_aliases(
        cls,
        robot: str,
        controller: str,
        course: str,
        reset_profile: str = "basic",
    ) -> "BenchmarkRunner":
        return cls(ArtifactRegistry().resolve(robot, controller, course, reset_profile))

    def run(
        self,
        seed: int,
        record_path: Path | None = None,
        course_override: dict | None = None,
    ) -> RunResult:
        course_spec = course_override or self.artifacts.course
        scene = self.artifacts.robot_dir / self.artifacts.robot["model"]["entrypoint"]
        policy_path = self.artifacts.controller_dir / self.artifacts.controller["policy"]["path"]
        environment = NativeH1Environment(
            scene,
            self.artifacts.robot,
            self.artifacts.controller,
        )
        policy = TorchScriptVelocityPolicy(policy_path, self.artifacts.controller)
        initial_state = environment.reset(seed=seed, reset_profile=self.artifacts.reset_profile)
        policy.reset()
        course = CourseRuntime.create(course_spec, initial_state)
        recorder = EpisodeRecorder(
            qpos_width=environment.model.nq,
            qvel_width=environment.model.nv,
            observation_width=policy.observation_size,
            action_width=policy.action_size,
        )
        done = False
        passed = False
        reason = "not_started"
        state = initial_state
        while not done:
            velocity_command = course.command(state)
            observation = policy.observation(
                state.time,
                state.qpos,
                state.qvel,
                velocity_command,
            )
            action, motor_command = policy.act(observation)
            state = environment.advance(motor_command)
            recorder.capture(state, observation, action, velocity_command, course.phase)
            done, passed, reason = course.evaluate(state)
        recorder.events.extend(course.events)
        falls = 1 if reason == "fall" else 0
        digest = recorder.determinism_digest()
        metrics = {
            "passed": passed,
            "reason": reason,
            "time_seconds": state.time,
            "progress_meters": course.progress(state),
            "falls": falls,
            "collisions": None,
            "energy_joules": state.energy_joules,
            "peak_torque_nm": state.peak_torque,
            "final_speed_mps": state.planar_speed,
            "stop_average_speed_mps": course.stop_average_speed_mps,
        }
        archive_sha256: str | None = None
        archive_receipt_path: str | None = None
        if record_path is not None:
            run_manifest = {
                "robot": {
                    "id": self.artifacts.robot_id,
                    "source_commit": self.artifacts.robot["source"]["commit"],
                    "tree_sha256": self.artifact_hashes["model_tree_sha256"],
                },
                "controller": {
                    "id": self.artifacts.controller_id,
                    "source_commit": self.artifacts.controller["source"]["commit"],
                    "policy_sha256": self.artifact_hashes["policy_sha256"],
                },
                "course": {
                    "id": course.reference,
                    "sha256": self.artifact_hashes["course_sha256"] if course_override is None else "override",
                },
                "engine": {
                    "name": "mujoco",
                    "timestep": environment.timestep,
                    "control_period": environment.control_period,
                    "solver": int(environment.model.opt.solver),
                    "integrator": int(environment.model.opt.integrator),
                },
                "exo_bench": git_provenance(),
                "environment": runtime_provenance(),
                "seed": seed,
                "reset_profile": {
                    "id": self.artifacts.reset_profile_id,
                    "sha256": self.artifact_hashes["reset_profile_sha256"],
                    "applied": environment.last_reset_perturbation,
                },
                "sensor_profile": "state-privileged-v1",
            }
            archive_sha256, receipt_path = recorder.write(record_path, run_manifest, metrics)
            archive_receipt_path = str(receipt_path)
        return RunResult(
            robot=self.artifacts.robot_id,
            controller=self.artifacts.controller_id,
            course=course.reference,
            seed=seed,
            passed=passed,
            reason=reason,
            time_seconds=state.time,
            progress_meters=course.progress(state),
            falls=falls,
            collisions=None,
            energy_joules=state.energy_joules,
            peak_torque_nm=state.peak_torque,
            final_speed_mps=state.planar_speed,
            stop_average_speed_mps=course.stop_average_speed_mps,
            determinism_digest=digest,
            replay_path=str(record_path) if record_path is not None else None,
            archive_sha256=archive_sha256,
            archive_receipt_path=archive_receipt_path,
            reset_profile=self.artifacts.reset_profile_id,
            reset_perturbation=environment.last_reset_perturbation,
        )
