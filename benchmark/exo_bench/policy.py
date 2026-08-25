from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import torch

from .actuator import MotorCommand


def gravity_orientation(quaternion_wxyz: np.ndarray) -> np.ndarray:
    qw, qx, qy, qz = quaternion_wxyz
    return np.asarray(
        [
            2 * (-qz * qx + qw * qy),
            -2 * (qz * qy + qw * qx),
            1 - 2 * (qw * qw + qz * qz),
        ],
        dtype=np.float32,
    )


class TorchScriptVelocityPolicy:
    def __init__(self, policy_path: Path, manifest: dict) -> None:
        self.manifest = manifest
        policy_spec = manifest["policy"]
        runtime = manifest["runtime"]
        self.observation_size = int(policy_spec["observation_size"])
        self.action_size = int(policy_spec["action_size"])
        self.default_angles = np.asarray(runtime["default_angles"], dtype=np.float64)
        self.kp = np.asarray(runtime["kp"], dtype=np.float64)
        self.kd = np.asarray(runtime["kd"], dtype=np.float64)
        self.command_scale = np.asarray(runtime["command_scale"], dtype=np.float32)
        self.angular_velocity_scale = float(runtime["angular_velocity_scale"])
        self.joint_position_scale = float(runtime["joint_position_scale"])
        self.joint_velocity_scale = float(runtime["joint_velocity_scale"])
        self.action_scale = float(runtime["action_scale"])
        self.gait_period = float(runtime["gait_period"])
        self.model = torch.jit.load(str(policy_path), map_location="cpu")
        self.model.eval()
        self.previous_action = np.zeros(self.action_size, dtype=np.float32)

    def reset(self) -> None:
        self.previous_action.fill(0)

    def observation(
        self,
        simulation_time: float,
        qpos: np.ndarray,
        qvel: np.ndarray,
        command: np.ndarray,
    ) -> np.ndarray:
        obs = np.zeros(self.observation_size, dtype=np.float32)
        qj = (qpos[7:] - self.default_angles) * self.joint_position_scale
        dqj = qvel[6:] * self.joint_velocity_scale
        phase = (simulation_time % self.gait_period) / self.gait_period
        obs[0:3] = qvel[3:6] * self.angular_velocity_scale
        obs[3:6] = gravity_orientation(qpos[3:7])
        obs[6:9] = command * self.command_scale
        obs[9 : 9 + self.action_size] = qj
        obs[9 + self.action_size : 9 + 2 * self.action_size] = dqj
        obs[9 + 2 * self.action_size : 9 + 3 * self.action_size] = self.previous_action
        obs[-2:] = [math.sin(2 * math.pi * phase), math.cos(2 * math.pi * phase)]
        return obs

    def act(self, observation: np.ndarray) -> tuple[np.ndarray, MotorCommand]:
        with torch.inference_mode():
            tensor = torch.from_numpy(observation).unsqueeze(0)
            action = self.model(tensor).detach().cpu().numpy().squeeze().astype(np.float32)
        if action.shape != (self.action_size,) or not np.isfinite(action).all():
            raise ValueError("Baseline policy returned an invalid action")
        self.previous_action = action.copy()
        q_des = action.astype(np.float64) * self.action_scale + self.default_angles
        command = MotorCommand(
            q_des=q_des,
            dq_des=np.zeros(self.action_size, dtype=np.float64),
            kp=self.kp,
            kd=self.kd,
            tau_ff=np.zeros(self.action_size, dtype=np.float64),
        )
        return action, command

