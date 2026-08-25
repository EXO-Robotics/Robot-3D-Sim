from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .agent import angle_delta
from .environment import SimulationState


@dataclass
class AgentNavigationCourse:
    specification: dict[str, Any]
    scenario: dict[str, Any]
    initial_position: np.ndarray
    phase: str = "navigate"
    path_length_m: float = 0.0
    last_position: np.ndarray | None = None
    hold_started_at: float | None = None
    stop_window: list[tuple[float, np.ndarray]] = field(default_factory=list)
    final_window_average_speed_mps: float | None = None
    events: list[dict[str, Any]] = field(default_factory=list)

    @classmethod
    def create(
        cls,
        specification: dict[str, Any],
        seed: int,
        initial_state: SimulationState,
    ) -> "AgentNavigationCourse":
        if specification.get("schema_version") != "exo.agent-course.v1" or specification.get("task") != "agent_navigation":
            raise ValueError("Unsupported Agent course specification")
        scenarios = specification["scenario_set"]["scenarios"]
        matches = [scenario for scenario in scenarios if int(scenario["seed"]) == seed]
        if len(matches) != 1:
            raise ValueError(f"AGENT-001 has no unique scenario for seed {seed}")
        runtime = cls(
            specification=specification,
            scenario=matches[0],
            initial_position=initial_state.base_position[:2].copy(),
            last_position=initial_state.base_position[:2].copy(),
        )
        runtime.events.append(
            {
                "t": initial_state.time,
                "type": "agent_course_started",
                "course": runtime.reference,
                "scenario_seed": seed,
                "target_position_m": runtime.goal_position.tolist(),
                "goal_heading_rad": runtime.goal_heading,
            }
        )
        return runtime

    @property
    def reference(self) -> str:
        return f"{self.specification['course_id']}@{self.specification['version']}"

    @property
    def goal_position(self) -> np.ndarray:
        return np.asarray(self.scenario["target_position_m"], dtype=np.float64)

    @property
    def goal_heading(self) -> float:
        return float(self.scenario["goal_heading_rad"])

    @property
    def time_limit(self) -> float:
        return float(self.specification["limits"]["time_seconds"])

    def observation(
        self,
        state: SimulationState,
        previous_applied_action: dict[str, Any] | None,
    ) -> dict[str, Any]:
        success = self.specification["success"]
        return {
            "schema_version": "exo.agent.observation.v1",
            "simulation_time_s": state.time,
            "base_position_m": {
                "x": float(state.base_position[0]),
                "y": float(state.base_position[1]),
                "z": float(state.base_position[2]),
            },
            "heading_yaw_rad": state.yaw,
            "planar_velocity_mps": {
                "x": float(state.qvel[0]),
                "y": float(state.qvel[1]),
            },
            "task": {
                "id": self.reference,
                "goal_position_m": {"x": float(self.goal_position[0]), "y": float(self.goal_position[1])},
                "goal_heading_rad": self.goal_heading,
                "target_radius_m": float(success["target_radius_m"]),
                "heading_tolerance_rad": float(success["heading_tolerance_rad"]),
            },
            "remaining_time_s": max(0.0, self.time_limit - state.time),
            "previous_applied_action": previous_applied_action,
        }

    def update_path(self, state: SimulationState) -> None:
        position = state.base_position[:2]
        if self.last_position is not None:
            self.path_length_m += float(np.linalg.norm(position - self.last_position))
        self.last_position = position.copy()

    def evaluate(self, state: SimulationState) -> tuple[bool, bool, str]:
        self.update_path(state)
        if state.fallen:
            self.events.append({"t": state.time, "type": "agent_course_failed", "reason": "fall"})
            return True, False, "fall"
        if state.time >= self.time_limit:
            self.events.append({"t": state.time, "type": "agent_course_failed", "reason": "timeout"})
            return True, False, "timeout"

        success = self.specification["success"]
        position_error = self.position_error(state)
        heading_error = self.heading_error(state)
        in_goal_geometry = (
            position_error <= float(success["target_radius_m"])
            and heading_error <= float(success["heading_tolerance_rad"])
        )
        if in_goal_geometry:
            if self.hold_started_at is None:
                self.hold_started_at = state.time
                self.phase = "hold"
                self.events.append({"t": state.time, "type": "target_hold_started"})
            hold_seconds = float(success["stop_hold_seconds"])
            self.stop_window.append((state.time, state.base_position[:2].copy()))
            cutoff = state.time - hold_seconds
            self.stop_window = [sample for sample in self.stop_window if sample[0] >= cutoff]
            if self.stop_window and state.time - self.stop_window[0][0] >= hold_seconds - 1e-6:
                positions = np.asarray([sample[1] for sample in self.stop_window])
                elapsed = self.stop_window[-1][0] - self.stop_window[0][0]
                distance = float(np.sum(np.linalg.norm(np.diff(positions, axis=0), axis=1)))
                self.final_window_average_speed_mps = distance / elapsed
                if (
                    self.final_window_average_speed_mps <= float(success["final_speed_mps"])
                    and state.planar_speed <= float(success["final_speed_mps"])
                ):
                    self.phase = "complete"
                    self.events.append(
                        {
                            "t": state.time,
                            "type": "agent_course_completed",
                            "final_window_average_speed_mps": self.final_window_average_speed_mps,
                        }
                    )
                    return True, True, "complete"
        elif self.hold_started_at is not None:
            self.events.append({"t": state.time, "type": "target_hold_reset"})
            self.hold_started_at = None
            self.stop_window.clear()
            self.phase = "navigate"
        return False, False, "running"

    def position_error(self, state: SimulationState) -> float:
        return float(np.linalg.norm(state.base_position[:2] - self.goal_position))

    def heading_error(self, state: SimulationState) -> float:
        return abs(angle_delta(self.goal_heading, state.yaw))

    def progress(self, state: SimulationState) -> float:
        start_error = float(np.linalg.norm(self.initial_position - self.goal_position))
        return start_error - self.position_error(state)

    def phase_event(self, time_s: float, decision_count: int) -> dict[str, Any]:
        return {
            "t": time_s,
            "type": "agent_decision",
            "decision": decision_count,
            "phase": self.phase,
        }
