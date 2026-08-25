from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class MotorCommand:
    q_des: np.ndarray
    dq_des: np.ndarray
    kp: np.ndarray
    kd: np.ndarray
    tau_ff: np.ndarray


class TrustedActuatorBoundary:
    """Robot-owned enforcement for direct-joint PD plus feedforward commands."""

    def __init__(self, actuator_manifest: dict, joint_count: int) -> None:
        if actuator_manifest["mode"] != "pd_ff":
            raise ValueError("Unsupported actuator mode")
        if actuator_manifest["transmission"] != "direct_joint":
            raise ValueError("This runtime currently supports only direct-joint transmissions")
        self.joint_count = joint_count
        self.torque_limits = np.asarray(actuator_manifest["torque_limits"], dtype=np.float64)
        self.kp_range = np.asarray(actuator_manifest["kp_range"], dtype=np.float64)
        self.kd_range = np.asarray(actuator_manifest["kd_range"], dtype=np.float64)
        if self.torque_limits.shape != (joint_count,):
            raise ValueError("Torque-limit count does not match actuator count")

    def resolve(self, command: MotorCommand, q: np.ndarray, dq: np.ndarray) -> np.ndarray:
        vectors = (command.q_des, command.dq_des, command.kp, command.kd, command.tau_ff, q, dq)
        if any(np.asarray(vector).shape != (self.joint_count,) for vector in vectors):
            raise ValueError("Motor command shape does not match robot actuator count")
        if not all(np.isfinite(vector).all() for vector in vectors):
            raise ValueError("Motor command contains a non-finite value")
        kp = np.clip(command.kp, self.kp_range[0], self.kp_range[1])
        kd = np.clip(command.kd, self.kd_range[0], self.kd_range[1])
        tau_ff = np.clip(command.tau_ff, -self.torque_limits, self.torque_limits)
        generalized_torque = kp * (command.q_des - q) + kd * (command.dq_des - dq) + tau_ff
        return np.clip(generalized_torque, -self.torque_limits, self.torque_limits)

