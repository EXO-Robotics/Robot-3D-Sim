from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mujoco
import numpy as np

from .actuator import MotorCommand, TrustedActuatorBoundary
from .policy import gravity_orientation


@dataclass(frozen=True)
class SimulationState:
    time: float
    qpos: np.ndarray
    qvel: np.ndarray
    base_position: np.ndarray
    yaw: float
    planar_speed: float
    height: float
    tilt: float
    fallen: bool
    contacts: np.ndarray
    energy_joules: float
    peak_torque: float


def yaw_from_quaternion(quaternion_wxyz: np.ndarray) -> float:
    w, x, y, z = quaternion_wxyz
    return math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


class NativeH1Environment:
    def __init__(self, scene_path: Path, robot_manifest: dict, controller_manifest: dict) -> None:
        self.model = mujoco.MjModel.from_xml_path(str(scene_path))
        self.data = mujoco.MjData(self.model)
        self.robot_manifest = robot_manifest
        self.controller_manifest = controller_manifest
        self.model.opt.timestep = float(robot_manifest["model"]["physics_timestep"])
        self.joint_count = int(robot_manifest["model"]["expected_actuators"])
        self.default_angles = np.asarray(controller_manifest["runtime"]["default_angles"], dtype=np.float64)
        self.boundary = TrustedActuatorBoundary(robot_manifest["actuator"], self.joint_count)
        self.decimation = int(controller_manifest["policy"]["control_decimation"])
        self.energy_joules = 0.0
        self.peak_torque = 0.0
        self.last_torque = np.zeros(self.joint_count, dtype=np.float64)
        self.last_reset_perturbation: dict[str, Any] = {
            "base_xy_m": [0.0, 0.0],
            "base_yaw_rad": 0.0,
            "joint_position_rad": [0.0] * self.joint_count,
        }
        self.floor_geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
        self.foot_body_ids = np.asarray(
            [
                mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "left_ankle_link"),
                mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "right_ankle_link"),
            ],
            dtype=np.int32,
        )
        self.pelvis_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")

    @property
    def timestep(self) -> float:
        return float(self.model.opt.timestep)

    @property
    def control_period(self) -> float:
        return self.timestep * self.decimation

    def reset(self, seed: int, reset_profile: dict | None = None) -> SimulationState:
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[7:] = self.default_angles
        if reset_profile is not None:
            xy_limit = self._nonnegative_limit(reset_profile, "base_xy_uniform_m")
            yaw_limit = self._nonnegative_limit(reset_profile, "base_yaw_uniform_rad")
            joint_limit = self._nonnegative_limit(reset_profile, "joint_position_uniform_rad")
            rng = np.random.default_rng(seed)
            xy = rng.uniform(-xy_limit, xy_limit, 2)
            yaw = float(rng.uniform(-yaw_limit, yaw_limit))
            joint_offsets = rng.uniform(-joint_limit, joint_limit, self.joint_count)
            self.data.qpos[:2] += xy
            yaw_quaternion = np.asarray(
                [math.cos(yaw / 2.0), 0.0, 0.0, math.sin(yaw / 2.0)],
                dtype=np.float64,
            )
            rotated = np.empty(4, dtype=np.float64)
            mujoco.mju_mulQuat(rotated, yaw_quaternion, self.data.qpos[3:7].copy())
            self.data.qpos[3:7] = rotated
            self.data.qpos[7:] += joint_offsets
            self.last_reset_perturbation = {
                "base_xy_m": xy.tolist(),
                "base_yaw_rad": yaw,
                "joint_position_rad": joint_offsets.tolist(),
            }
        else:
            self.last_reset_perturbation = {
                "base_xy_m": [0.0, 0.0],
                "base_yaw_rad": 0.0,
                "joint_position_rad": [0.0] * self.joint_count,
            }
        self.data.ctrl[:] = 0
        self.energy_joules = 0.0
        self.peak_torque = 0.0
        self.last_torque.fill(0)
        mujoco.mj_forward(self.model, self.data)
        return self.state()

    @staticmethod
    def _nonnegative_limit(profile: dict, key: str) -> float:
        value = float(profile[key])
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"Reset profile {key} must be finite and nonnegative")
        return value

    def advance(self, command: MotorCommand) -> SimulationState:
        for _ in range(self.decimation):
            q = np.asarray(self.data.qpos[7:], dtype=np.float64)
            dq = np.asarray(self.data.qvel[6:], dtype=np.float64)
            torque = self.boundary.resolve(command, q, dq)
            self.data.ctrl[:] = torque
            self.last_torque = torque.copy()
            self.peak_torque = max(self.peak_torque, float(np.max(np.abs(torque))))
            self.energy_joules += float(np.sum(np.abs(torque * dq))) * self.timestep
            mujoco.mj_step(self.model, self.data)
        return self.state()

    def passive_step(self, steps: int = 1) -> SimulationState:
        self.data.ctrl[:] = 0
        for _ in range(steps):
            mujoco.mj_step(self.model, self.data)
        return self.state()

    def apply_push(self, force_xyz: tuple[float, float, float]) -> None:
        self.data.xfrc_applied[self.pelvis_body_id, :3] = force_xyz

    def clear_push(self) -> None:
        self.data.xfrc_applied[self.pelvis_body_id, :] = 0

    def state(self) -> SimulationState:
        qpos = np.array(self.data.qpos, dtype=np.float64, copy=True)
        qvel = np.array(self.data.qvel, dtype=np.float64, copy=True)
        projected_gravity = gravity_orientation(qpos[3:7])
        tilt = math.acos(float(np.clip(-projected_gravity[2], -1.0, 1.0)))
        height = float(qpos[2])
        fallen = height < 0.62 or tilt > 1.0 or not np.isfinite(qpos).all() or not np.isfinite(qvel).all()
        return SimulationState(
            time=float(self.data.time),
            qpos=qpos,
            qvel=qvel,
            base_position=qpos[:3].copy(),
            yaw=yaw_from_quaternion(qpos[3:7]),
            planar_speed=float(np.linalg.norm(qvel[:2])),
            height=height,
            tilt=tilt,
            fallen=fallen,
            contacts=self._foot_contacts(),
            energy_joules=self.energy_joules,
            peak_torque=self.peak_torque,
        )

    def _foot_contacts(self) -> np.ndarray:
        forces = np.zeros(2, dtype=np.float32)
        if self.floor_geom_id < 0:
            return forces
        for contact_index in range(self.data.ncon):
            contact = self.data.contact[contact_index]
            geom1 = int(contact.geom1)
            geom2 = int(contact.geom2)
            if geom1 != self.floor_geom_id and geom2 != self.floor_geom_id:
                continue
            robot_geom = geom2 if geom1 == self.floor_geom_id else geom1
            body_id = int(self.model.geom_bodyid[robot_geom])
            matches = np.where(self.foot_body_ids == body_id)[0]
            if not len(matches):
                continue
            force = np.zeros(6, dtype=np.float64)
            mujoco.mj_contactForce(self.model, self.data, contact_index, force)
            forces[int(matches[0])] += abs(float(force[0]))
        return forces
