from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Protocol

from .hashing import sha256_file
from .paths import AGENT_CONTRACTS_ROOT, AGENTS_ROOT


OBSERVATION_SCHEMA = "exo.agent.observation.v1"
ACTION_SCHEMA = "exo.agent.action.v1"
RUNTIME_SCHEMA = "exo.agent.runtime.v1"
SCRIPTED_BASELINE_ID = "exo.agent.scripted-baseline.v0.1"


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _finite_number(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def _velocity_action(forward: float, lateral: float, yaw_rate: float) -> dict[str, Any]:
    return {
        "schema_version": ACTION_SCHEMA,
        "type": "velocity",
        "forward_mps": float(forward),
        "lateral_mps": float(lateral),
        "yaw_rate_rps": float(yaw_rate),
    }


def safe_stop_action() -> dict[str, Any]:
    return _velocity_action(0.0, 0.0, 0.0)


def validate_observation(observation: object) -> dict[str, Any]:
    if not isinstance(observation, dict):
        raise ValueError("Agent observation must be an object")
    required = {
        "schema_version",
        "simulation_time_s",
        "base_position_m",
        "heading_yaw_rad",
        "planar_velocity_mps",
        "task",
        "remaining_time_s",
        "previous_applied_action",
    }
    if set(observation) != required:
        raise ValueError("Agent observation fields do not match exo.agent.observation.v1")
    if observation["schema_version"] != OBSERVATION_SCHEMA:
        raise ValueError("Unsupported Agent observation schema")
    for field in ("simulation_time_s", "heading_yaw_rad", "remaining_time_s"):
        if not _finite_number(observation[field]):
            raise ValueError(f"Agent observation {field} must be finite")
    if float(observation["simulation_time_s"]) < 0 or float(observation["remaining_time_s"]) < 0:
        raise ValueError("Agent observation times must be nonnegative")
    if abs(float(observation["heading_yaw_rad"])) > math.pi:
        raise ValueError("Agent observation heading_yaw_rad must be in [-pi, pi]")
    for field, axes in (("base_position_m", ("x", "y", "z")), ("planar_velocity_mps", ("x", "y"))):
        value = observation[field]
        if not isinstance(value, dict) or set(value) != set(axes):
            raise ValueError(f"Agent observation {field} has invalid fields")
        if not all(_finite_number(value[axis]) for axis in axes):
            raise ValueError(f"Agent observation {field} must be finite")
    task = observation["task"]
    task_fields = {"id", "goal_position_m", "goal_heading_rad", "target_radius_m", "heading_tolerance_rad"}
    if not isinstance(task, dict) or set(task) != task_fields or task["id"] != "AGENT-001@1.0.0":
        raise ValueError("Agent observation task is invalid")
    goal = task["goal_position_m"]
    if not isinstance(goal, dict) or set(goal) != {"x", "y"} or not all(_finite_number(goal[a]) for a in ("x", "y")):
        raise ValueError("Agent observation goal_position_m is invalid")
    for field in ("goal_heading_rad", "target_radius_m", "heading_tolerance_rad"):
        if not _finite_number(task[field]):
            raise ValueError(f"Agent observation task.{field} must be finite")
    if abs(float(task["goal_heading_rad"])) > math.pi:
        raise ValueError("Agent observation task.goal_heading_rad must be in [-pi, pi]")
    if float(task["target_radius_m"]) <= 0:
        raise ValueError("Agent observation task.target_radius_m must be positive")
    if not 0 < float(task["heading_tolerance_rad"]) <= math.pi:
        raise ValueError("Agent observation task.heading_tolerance_rad must be in (0, pi]")
    previous = observation["previous_applied_action"]
    if previous is not None:
        validation = validate_action(previous, load_runtime_contract())
        if validation.rejected or validation.clamped:
            raise ValueError("Agent observation previous_applied_action is not a valid applied action")
    return observation


@dataclass(frozen=True)
class ActionValidation:
    requested_action: dict[str, Any] | None
    applied_action: dict[str, Any]
    clamped: bool
    rejected: bool
    issues: tuple[str, ...]

    def to_record(self) -> dict[str, Any]:
        return {
            "requested_action": self.requested_action,
            "applied_action": self.applied_action,
            "validation": {
                "clamped": self.clamped,
                "rejected": self.rejected,
                "issues": list(self.issues),
            },
        }


def validate_action(action: object, runtime: dict[str, Any]) -> ActionValidation:
    fields = {"schema_version", "type", "forward_mps", "lateral_mps", "yaw_rate_rps"}
    if not isinstance(action, dict) or set(action) != fields:
        return ActionValidation(None, safe_stop_action(), False, True, ("malformed_action",))
    if action.get("schema_version") != ACTION_SCHEMA or action.get("type") != "velocity":
        return ActionValidation(None, safe_stop_action(), False, True, ("unsupported_action_contract",))
    numeric_fields = ("forward_mps", "lateral_mps", "yaw_rate_rps")
    if not all(_finite_number(action[field]) for field in numeric_fields):
        return ActionValidation(None, safe_stop_action(), False, True, ("nonfinite_action",))
    requested = _velocity_action(
        float(action["forward_mps"]),
        float(action["lateral_mps"]),
        float(action["yaw_rate_rps"]),
    )
    applied = dict(requested)
    issues: list[str] = []
    for field in numeric_fields:
        lower, upper = (float(value) for value in runtime["action_limits"][field])
        value = float(requested[field])
        bounded = min(upper, max(lower, value))
        applied[field] = bounded
        if bounded != value:
            issues.append(f"clamped:{field}")
    return ActionValidation(requested, applied, bool(issues), False, tuple(issues))


def load_runtime_contract() -> dict[str, Any]:
    runtime = _read_json(AGENT_CONTRACTS_ROOT / f"{RUNTIME_SCHEMA}.json")
    if runtime.get("schema_version") != RUNTIME_SCHEMA or runtime.get("scheduler") != "simulated_time_paused":
        raise ValueError("Agent runtime contract is invalid")
    interval = runtime.get("decision_interval_s")
    if not _finite_number(interval) or float(interval) <= 0:
        raise ValueError("Agent decision interval must be finite and positive")
    for field in ("forward_mps", "lateral_mps", "yaw_rate_rps"):
        bounds = runtime.get("action_limits", {}).get(field)
        if not isinstance(bounds, list) or len(bounds) != 2 or not all(_finite_number(value) for value in bounds):
            raise ValueError(f"Agent runtime bounds are invalid for {field}")
        if float(bounds[0]) > float(bounds[1]):
            raise ValueError(f"Agent runtime bounds are reversed for {field}")
    return runtime


def verify_agent_artifacts(agent_id: str = SCRIPTED_BASELINE_ID) -> dict[str, str]:
    manifest_path = AGENTS_ROOT / agent_id / "manifest.json"
    manifest = _read_json(manifest_path)
    if manifest.get("agent_id") != agent_id or manifest.get("network_access") is not False:
        raise ValueError("Agent manifest identity or network boundary is invalid")
    hashes = {"agent_manifest_sha256": sha256_file(manifest_path)}
    for label, contract in manifest["contracts"].items():
        path = (AGENTS_ROOT / agent_id / contract["path"]).resolve()
        actual = sha256_file(path)
        if actual != contract["sha256"]:
            raise ValueError(f"Agent {label} contract hash mismatch")
        hashes[f"{label}_sha256"] = actual
    return hashes


class Agent(Protocol):
    agent_id: str

    def reset(self) -> None: ...

    def act(self, observation: dict[str, Any]) -> object: ...


class ScriptedBaselineAgent:
    agent_id = SCRIPTED_BASELINE_ID

    def reset(self) -> None:
        return None

    def act(self, observation: dict[str, Any]) -> dict[str, Any]:
        # This validation is intentionally the only boundary the scripted agent sees.
        obs = validate_observation(observation)
        position = obs["base_position_m"]
        goal = obs["task"]["goal_position_m"]
        dx = float(goal["x"]) - float(position["x"])
        dy = float(goal["y"]) - float(position["y"])
        distance = math.hypot(dx, dy)
        target_radius = float(obs["task"]["target_radius_m"])
        heading = float(obs["heading_yaw_rad"])

        if distance > target_radius * 0.72:
            desired_heading = math.atan2(dy, dx)
            error = angle_delta(desired_heading, heading)
            yaw_rate = max(-0.6, min(0.6, 1.4 * error))
            alignment = max(0.0, math.cos(error))
            if abs(error) > 0.55:
                forward = 0.0
            else:
                forward = min(0.55, max(0.12, 0.5 * distance)) * alignment
            return _velocity_action(forward, 0.0, yaw_rate)

        goal_heading = float(obs["task"]["goal_heading_rad"])
        heading_error = angle_delta(goal_heading, heading)
        planar_velocity = obs["planar_velocity_mps"]
        speed = math.hypot(float(planar_velocity["x"]), float(planar_velocity["y"]))
        if abs(heading_error) > float(obs["task"]["heading_tolerance_rad"]) * 0.55 and speed < 0.28:
            return _velocity_action(0.0, 0.0, max(-0.45, min(0.45, 1.2 * heading_error)))
        return safe_stop_action()


def angle_delta(target: float, current: float) -> float:
    return math.atan2(math.sin(target - current), math.cos(target - current))


@dataclass(frozen=True)
class AgentDecision:
    simulation_time: float
    observation: dict[str, Any]
    requested_action: dict[str, Any] | None
    applied_action: dict[str, Any]
    clamped: bool
    rejected: bool
    issues: tuple[str, ...]
    wall_latency_ms: float

    def record(self) -> dict[str, Any]:
        return {
            "simulation_time": self.simulation_time,
            "observation": self.observation,
            "requested_action": self.requested_action,
            "applied_action": self.applied_action,
            "validation": {
                "clamped": self.clamped,
                "rejected": self.rejected,
                "issues": list(self.issues),
            },
            "wall_latency_ms": self.wall_latency_ms,
        }

    def deterministic_record(self) -> dict[str, Any]:
        value = self.record()
        value.pop("wall_latency_ms")
        return value


class SimulatedTimeAgentScheduler:
    def __init__(
        self,
        agent: Agent,
        runtime: dict[str, Any] | None = None,
        clock_ns: Callable[[], int] = time.perf_counter_ns,
    ) -> None:
        self.agent = agent
        self.runtime = runtime or load_runtime_contract()
        self.clock_ns = clock_ns
        self.decision_interval_s = float(self.runtime["decision_interval_s"])

    def reset(self) -> None:
        self.agent.reset()

    def steps_per_decision(self, physics_control_period_s: float) -> int:
        ratio = self.decision_interval_s / physics_control_period_s
        steps = round(ratio)
        if steps <= 0 or not math.isclose(ratio, steps, rel_tol=0.0, abs_tol=1e-9):
            raise ValueError("Agent decision interval must be an exact multiple of the policy control period")
        return steps

    def decide(self, observation: dict[str, Any]) -> AgentDecision:
        validated_observation = validate_observation(observation)
        simulation_time = float(validated_observation["simulation_time_s"])
        started = self.clock_ns()
        requested = self.agent.act(validated_observation)
        finished = self.clock_ns()
        validation = validate_action(requested, self.runtime)
        return AgentDecision(
            simulation_time=simulation_time,
            observation=validated_observation,
            requested_action=validation.requested_action,
            applied_action=validation.applied_action,
            clamped=validation.clamped,
            rejected=validation.rejected,
            issues=validation.issues,
            wall_latency_ms=max(0.0, (finished - started) / 1_000_000.0),
        )


def action_command(action: dict[str, Any]) -> tuple[float, float, float]:
    return (
        float(action["forward_mps"]),
        float(action["lateral_mps"]),
        float(action["yaw_rate_rps"]),
    )


def decision_dict(decision: AgentDecision) -> dict[str, Any]:
    return asdict(decision)
