from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .environment import SimulationState


def angle_delta(current: float, origin: float) -> float:
    return math.atan2(math.sin(current - origin), math.cos(current - origin))


@dataclass
class CourseRuntime:
    specification: dict
    initial_position: np.ndarray
    initial_yaw: float
    phase: str = "active"
    phase_origin: np.ndarray | None = None
    target_yaw: float | None = None
    stop_started_at: float | None = None
    stop_window: list[tuple[float, np.ndarray]] = field(default_factory=list)
    stop_average_speed_mps: float | None = None
    events: list[dict] = field(default_factory=list)

    @classmethod
    def create(cls, specification: dict, initial_state: SimulationState) -> "CourseRuntime":
        runtime = cls(
            specification=specification,
            initial_position=initial_state.base_position.copy(),
            initial_yaw=initial_state.yaw,
        )
        task = specification["task"]
        if task == "turn_walk_stop":
            runtime.phase = "turn"
            runtime.target_yaw = initial_state.yaw + float(specification["success"]["turn_radians"])
        elif task == "walk_stop":
            runtime.phase = "walk"
        elif task == "walk":
            runtime.phase = "walk"
        elif task == "stand":
            runtime.phase = "stand"
        else:
            raise ValueError(f"Unsupported course task: {task}")
        runtime.events.append({"t": initial_state.time, "type": "course_started", "phase": runtime.phase})
        return runtime

    @property
    def reference(self) -> str:
        return f"{self.specification['course_id']}@{self.specification['version']}"

    def command(self, state: SimulationState) -> np.ndarray:
        task = self.specification["task"]
        if task == "stand":
            return np.asarray(self.specification["command"]["velocity"], dtype=np.float32)
        if task == "walk":
            return np.asarray(self.specification["command"]["velocity"], dtype=np.float32)
        if task == "walk_stop":
            if self.phase == "walk" and self._straight_progress(state) >= self._minimum_progress():
                self._transition("stop", state)
            return (
                np.asarray(self.specification["command"]["velocity"], dtype=np.float32)
                if self.phase == "walk"
                else np.zeros(3, dtype=np.float32)
            )
        if task == "turn_walk_stop":
            if self.phase == "turn" and self._turn_complete(state):
                self.phase_origin = state.base_position.copy()
                self._transition("walk", state)
            if self.phase == "walk" and self._heading_progress(state) >= self._minimum_progress():
                self._transition("stop", state)
            if self.phase == "turn":
                return np.asarray(self.specification["command"]["turn"], dtype=np.float32)
            if self.phase == "walk":
                return np.asarray(self.specification["command"]["walk"], dtype=np.float32)
            return np.zeros(3, dtype=np.float32)
        raise AssertionError("Unreachable course task")

    def evaluate(self, state: SimulationState) -> tuple[bool, bool, str]:
        if state.fallen:
            return True, False, "fall"
        if state.time >= float(self.specification["limits"]["time_seconds"]):
            task = self.specification["task"]
            if task == "stand" and state.time >= float(self.specification["success"]["minimum_upright_seconds"]):
                return True, True, "complete"
            return True, False, "timeout"
        task = self.specification["task"]
        if task == "walk" and self._straight_progress(state) >= self._minimum_progress():
            return True, True, "complete"
        if task in {"walk_stop", "turn_walk_stop"} and self.phase == "stop":
            success = self.specification["success"]
            hold_seconds = float(success["stop_hold_seconds"])
            self.stop_window.append((state.time, state.base_position[:2].copy()))
            cutoff = state.time - hold_seconds
            self.stop_window = [sample for sample in self.stop_window if sample[0] >= cutoff]
            if self.stop_window and state.time - self.stop_window[0][0] >= hold_seconds - 1e-6:
                elapsed = state.time - self.stop_window[0][0]
                displacement = float(np.linalg.norm(state.base_position[:2] - self.stop_window[0][1]))
                average_speed = displacement / elapsed
                self.stop_average_speed_mps = average_speed
                if average_speed <= float(success["stop_speed_mps"]):
                    progress = self._straight_progress(state) if task == "walk_stop" else self._heading_progress(state)
                    error = abs(progress - self._minimum_progress())
                    if error <= float(success["stop_zone_tolerance_meters"]):
                        return True, True, "complete"
                    return True, False, "missed_stop_zone"
        return False, False, "running"

    def progress(self, state: SimulationState) -> float:
        if self.specification["task"] == "turn_walk_stop" and self.phase != "turn":
            return self._heading_progress(state)
        return self._straight_progress(state)

    def _minimum_progress(self) -> float:
        return float(self.specification["success"].get("minimum_progress_meters", 0.0))

    def _straight_progress(self, state: SimulationState) -> float:
        return float(state.base_position[0] - self.initial_position[0])

    def _heading_progress(self, state: SimulationState) -> float:
        origin = self.phase_origin if self.phase_origin is not None else self.initial_position
        heading = self.target_yaw if self.target_yaw is not None else self.initial_yaw
        direction = np.asarray([math.cos(heading), math.sin(heading)], dtype=np.float64)
        return float(np.dot(state.base_position[:2] - origin[:2], direction))

    def _turn_complete(self, state: SimulationState) -> bool:
        requested = float(self.specification["success"]["turn_radians"])
        tolerance = float(self.specification["success"]["turn_tolerance_radians"])
        actual = angle_delta(state.yaw, self.initial_yaw)
        return actual >= requested - tolerance if requested >= 0 else actual <= requested + tolerance

    def _transition(self, phase: str, state: SimulationState) -> None:
        if self.phase == phase:
            return
        previous = self.phase
        self.phase = phase
        self.stop_started_at = None
        self.stop_window.clear()
        self.stop_average_speed_mps = None
        self.events.append({"t": state.time, "type": "phase_changed", "from": previous, "to": phase})
