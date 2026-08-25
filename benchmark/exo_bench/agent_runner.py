from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from .agent import (
    SCRIPTED_BASELINE_ID,
    Agent,
    AgentDecision,
    ScriptedBaselineAgent,
    SimulatedTimeAgentScheduler,
    action_command,
    load_runtime_contract,
    verify_agent_artifacts,
)
from .agent_course import AgentNavigationCourse
from .environment import NativeH1Environment, SimulationState
from .hashing import sha256_file
from .paths import AGENTS_ROOT
from .policy import TorchScriptVelocityPolicy
from .provenance import git_provenance, runtime_provenance
from .recorder import EpisodeRecorder
from .registry import ArtifactRegistry, ResolvedArtifacts


class LiveFrameSink(Protocol):
    def hello(self, message: dict[str, Any]) -> None: ...

    def frame(self, message: dict[str, Any]) -> None: ...

    def complete(self, message: dict[str, Any]) -> None: ...


@dataclass(frozen=True)
class AgentRunResult:
    robot: str
    controller: str
    agent: str
    course: str
    seed: int
    passed: bool
    reason: str
    simulated_completion_time_s: float
    path_length_m: float
    progress_m: float
    final_position_error_m: float
    final_heading_error_rad: float
    final_speed_mps: float
    final_window_average_speed_mps: float | None
    falls: int
    collisions: int | None
    energy_joules: float
    peak_torque_nm: float
    agent_decision_count: int
    requested_action_count: int
    applied_action_count: int
    clamped_action_count: int
    rejected_action_count: int
    agent_wall_latency_ms: dict[str, float]
    determinism_digest: str
    replay_path: str | None
    archive_sha256: str | None
    archive_receipt_path: str | None
    scenario: dict[str, Any]
    reset_profile: str
    reset_perturbation: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class AgentBenchmarkRunner:
    def __init__(
        self,
        artifacts: ResolvedArtifacts,
        agent: Agent | None = None,
        agent_id: str = SCRIPTED_BASELINE_ID,
    ) -> None:
        if artifacts.course.get("schema_version") != "exo.agent-course.v1":
            raise ValueError("AgentBenchmarkRunner requires an EXO Agent course")
        self.artifacts = artifacts
        self.artifact_hashes = ArtifactRegistry.verify_hashes(artifacts)
        self.agent_id = agent_id
        self.agent = agent or ScriptedBaselineAgent()
        if self.agent.agent_id != agent_id:
            raise ValueError("Agent identity does not match the requested immutable manifest")
        self.agent_hashes = verify_agent_artifacts(agent_id)
        self.agent_manifest_path = AGENTS_ROOT / agent_id / "manifest.json"
        self.agent_manifest = json.loads(self.agent_manifest_path.read_text(encoding="utf-8"))
        self.runtime_contract = load_runtime_contract()

    @classmethod
    def from_aliases(
        cls,
        robot: str = "h1",
        controller: str = "baseline",
        course: str = "agent-navigation",
        reset_profile: str = "basic",
        agent: Agent | None = None,
        agent_id: str = SCRIPTED_BASELINE_ID,
    ) -> "AgentBenchmarkRunner":
        artifacts = ArtifactRegistry().resolve(robot, controller, course, reset_profile)
        return cls(artifacts, agent=agent, agent_id=agent_id)

    def run(
        self,
        seed: int,
        record_path: Path | None = None,
        live_sink: LiveFrameSink | None = None,
    ) -> AgentRunResult:
        scene = self.artifacts.robot_dir / self.artifacts.robot["model"]["entrypoint"]
        policy_path = self.artifacts.controller_dir / self.artifacts.controller["policy"]["path"]
        environment = NativeH1Environment(scene, self.artifacts.robot, self.artifacts.controller)
        policy = TorchScriptVelocityPolicy(policy_path, self.artifacts.controller)
        state = environment.reset(seed=seed, reset_profile=self.artifacts.reset_profile)
        scenario = self._scenario(seed)
        state = environment.add_scenario_start_heading(float(scenario["start_heading_rad"]))
        course = AgentNavigationCourse.create(self.artifacts.course, seed, state)
        scheduler = SimulatedTimeAgentScheduler(self.agent, self.runtime_contract)
        scheduler.reset()
        policy.reset()
        low_level_steps = scheduler.steps_per_decision(environment.control_period)
        recorder = EpisodeRecorder(
            qpos_width=environment.model.nq,
            qvel_width=environment.model.nv,
            observation_width=policy.observation_size,
            action_width=policy.action_size,
        )
        previous_applied: dict[str, Any] | None = None
        decisions: list[AgentDecision] = []
        sequence = 0
        live_course_event_index = 0
        done = False
        passed = False
        reason = "not_started"
        if live_sink is not None:
            live_sink.hello(self._live_hello(seed, scenario, environment))

        while not done:
            observation = course.observation(state, previous_applied)
            decision = scheduler.decide(observation)
            decisions.append(decision)
            recorder.record_agent_decision(observation, decision.record())
            decision_event = {
                "t": state.time,
                "type": "agent_decision",
                "decision": len(decisions),
                "requested_action": decision.requested_action,
                "applied_action": decision.applied_action,
                "clamped": decision.clamped,
                "rejected": decision.rejected,
            }
            recorder.events.append(decision_event)
            previous_applied = decision.applied_action
            velocity_command = np.asarray(action_command(decision.applied_action), dtype=np.float32)
            for low_level_step in range(low_level_steps):
                policy_observation = policy.observation(
                    state.time,
                    state.qpos,
                    state.qvel,
                    velocity_command,
                )
                low_level_action, motor_command = policy.act(policy_observation)
                state = environment.advance(motor_command)
                body_poses = environment.visual_body_poses()
                recorder.capture(
                    state,
                    policy_observation,
                    low_level_action,
                    velocity_command,
                    course.phase,
                    body_poses=body_poses,
                )
                done, passed, reason = course.evaluate(state)
                if live_sink is not None:
                    frame_events: list[dict[str, Any]] = []
                    if low_level_step == 0:
                        frame_events.append(decision_event)
                    frame_events.extend(course.events[live_course_event_index:])
                    live_course_event_index = len(course.events)
                    live_sink.frame(
                        self._live_frame(
                            sequence,
                            state,
                            body_poses,
                            course,
                            decision,
                            frame_events,
                        )
                    )
                sequence += 1
                if done:
                    break

        recorder.events.extend(course.events)
        digest = recorder.determinism_digest()
        latency_values = [decision.wall_latency_ms for decision in decisions]
        latency = {
            "total": float(sum(latency_values)),
            "mean": float(sum(latency_values) / len(latency_values)) if latency_values else 0.0,
            "max": float(max(latency_values, default=0.0)),
        }
        metrics = {
            "passed": passed,
            "reason": reason,
            "simulated_completion_time_s": state.time,
            "path_length_m": course.path_length_m,
            "progress_m": course.progress(state),
            "final_position_error_m": course.position_error(state),
            "final_heading_error_rad": course.heading_error(state),
            "final_speed_mps": state.planar_speed,
            "final_window_average_speed_mps": course.final_window_average_speed_mps,
            "falls": 1 if reason == "fall" else 0,
            "collisions": None,
            "energy_joules": state.energy_joules,
            "peak_torque_nm": state.peak_torque,
            "agent_decision_count": len(decisions),
            "requested_action_count": sum(decision.requested_action is not None for decision in decisions),
            "applied_action_count": len(decisions),
            "clamped_action_count": sum(decision.clamped for decision in decisions),
            "rejected_action_count": sum(decision.rejected for decision in decisions),
            "agent_wall_latency_ms": latency,
        }
        archive_sha256: str | None = None
        archive_receipt_path: str | None = None
        if record_path is not None:
            archive_sha256, receipt_path = recorder.write(
                record_path,
                self._run_manifest(seed, scenario, environment),
                metrics,
            )
            archive_receipt_path = str(receipt_path)

        result = AgentRunResult(
            robot=self.artifacts.robot_id,
            controller=self.artifacts.controller_id,
            agent=self.agent_id,
            course=course.reference,
            seed=seed,
            passed=passed,
            reason=reason,
            simulated_completion_time_s=state.time,
            path_length_m=course.path_length_m,
            progress_m=course.progress(state),
            final_position_error_m=course.position_error(state),
            final_heading_error_rad=course.heading_error(state),
            final_speed_mps=state.planar_speed,
            final_window_average_speed_mps=course.final_window_average_speed_mps,
            falls=1 if reason == "fall" else 0,
            collisions=None,
            energy_joules=state.energy_joules,
            peak_torque_nm=state.peak_torque,
            agent_decision_count=len(decisions),
            requested_action_count=metrics["requested_action_count"],
            applied_action_count=len(decisions),
            clamped_action_count=metrics["clamped_action_count"],
            rejected_action_count=metrics["rejected_action_count"],
            agent_wall_latency_ms=latency,
            determinism_digest=digest,
            replay_path=str(record_path) if record_path is not None else None,
            archive_sha256=archive_sha256,
            archive_receipt_path=archive_receipt_path,
            scenario=scenario,
            reset_profile=self.artifacts.reset_profile_id,
            reset_perturbation=environment.last_reset_perturbation,
        )
        if live_sink is not None:
            live_sink.complete(
                {
                    "schema_version": "exo.live.v1",
                    "type": "complete",
                    "sequence": sequence,
                    "simulation_time": state.time,
                    "result": result.to_dict(),
                }
            )
        return result

    def _scenario(self, seed: int) -> dict[str, Any]:
        scenarios = self.artifacts.course["scenario_set"]["scenarios"]
        matches = [value for value in scenarios if int(value["seed"]) == seed]
        if len(matches) != 1:
            raise ValueError(f"AGENT-001 has no unique scenario for seed {seed}")
        return matches[0]

    def _run_manifest(
        self,
        seed: int,
        scenario: dict[str, Any],
        environment: NativeH1Environment,
    ) -> dict[str, Any]:
        return {
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
            "agent": {
                "id": self.agent_id,
                "manifest_sha256": self.agent_hashes["agent_manifest_sha256"],
                "contracts": {
                    "observation": {"id": "exo.agent.observation.v1", "sha256": self.agent_hashes["observation_sha256"]},
                    "action": {"id": "exo.agent.action.v1", "sha256": self.agent_hashes["action_sha256"]},
                    "runtime": {"id": "exo.agent.runtime.v1", "sha256": self.agent_hashes["runtime_sha256"]},
                },
            },
            "course": {
                "id": f"{self.artifacts.course['course_id']}@{self.artifacts.course['version']}",
                "sha256": sha256_file(self.artifacts.course_path),
                "scenario_set": self.artifacts.course["scenario_set"]["id"],
                "scenario": scenario,
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
            "sensor_profile": "agent-privileged-navigation-v1",
            "scheduler": self.runtime_contract,
            "visual_replay": {
                "body_pose_contract": "exo.visual-body-poses.v1",
                "link_order": [
                    "pelvis", "torso", "head", "left_upper_arm", "left_forearm", "right_upper_arm", "right_forearm",
                    "left_thigh", "left_shin", "left_foot", "right_thigh", "right_shin", "right_foot"
                ],
                "pose_layout": ["px", "py", "pz", "qw", "qx", "qy", "qz"],
                "source": "native MuJoCo forward kinematics recorded during the authoritative run",
            },
        }

    def _live_hello(
        self,
        seed: int,
        scenario: dict[str, Any],
        environment: NativeH1Environment,
    ) -> dict[str, Any]:
        return {
            "schema_version": "exo.live.v1",
            "type": "hello",
            "run": {
                "robot": self.artifacts.robot_id,
                "controller": self.artifacts.controller_id,
                "agent": self.agent_id,
                "course": "AGENT-001@1.0.0",
                "seed": seed,
                "scenario": scenario,
                "authoritative": "native exo-bench",
                "browser_control_allowed": False,
                "control_period_s": environment.control_period,
            },
        }

    @staticmethod
    def _live_frame(
        sequence: int,
        state: SimulationState,
        body_poses: np.ndarray,
        course: AgentNavigationCourse,
        decision: AgentDecision,
        events: list[dict[str, Any]],
    ) -> dict[str, Any]:
        return {
            "schema_version": "exo.live.v1",
            "type": "frame",
            "sequence": sequence,
            "simulation_time": state.time,
            "qpos": state.qpos.tolist(),
            "qvel": state.qvel.tolist(),
            "body_poses": body_poses.tolist(),
            "task": {
                "course_id": course.reference,
                "phase": course.phase,
                "goal_position_m": course.goal_position.tolist(),
                "progress_m": course.progress(state),
            },
            "observation": decision.observation,
            "requested_action": decision.requested_action,
            "applied_action": decision.applied_action,
            "agent_latency_ms": decision.wall_latency_ms,
            "telemetry": {
                "position_m": state.base_position.tolist(),
                "speed_mps": state.planar_speed,
                "heading_rad": state.yaw,
                "energy_joules": state.energy_joules,
                "falls": 1 if state.fallen else 0,
                "position_error_m": course.position_error(state),
            },
            "events": events,
        }
