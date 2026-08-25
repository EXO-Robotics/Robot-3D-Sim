from __future__ import annotations

import hashlib
import json
import platform
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import mujoco
import numpy as np

from . import __version__
from .environment import SimulationState


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")


def _float32_bytes(rows: list[np.ndarray], width: int) -> bytes:
    if not rows:
        return np.empty((0, width), dtype="<f4").tobytes()
    return np.asarray(rows, dtype="<f4").reshape(len(rows), width).tobytes(order="C")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


@dataclass
class EpisodeRecorder:
    qpos_width: int
    qvel_width: int
    observation_width: int
    action_width: int
    observations: list[np.ndarray] = field(default_factory=list)
    actions: list[np.ndarray] = field(default_factory=list)
    commands: list[np.ndarray] = field(default_factory=list)
    qpos: list[np.ndarray] = field(default_factory=list)
    qvel: list[np.ndarray] = field(default_factory=list)
    contacts: list[np.ndarray] = field(default_factory=list)
    times: list[np.ndarray] = field(default_factory=list)
    events: list[dict] = field(default_factory=list)

    def capture(
        self,
        state: SimulationState,
        observation: np.ndarray,
        action: np.ndarray,
        command: np.ndarray,
        phase: str,
    ) -> None:
        self.observations.append(np.array(observation, dtype=np.float32, copy=True))
        self.actions.append(np.array(action, dtype=np.float32, copy=True))
        self.commands.append(np.array(command, dtype=np.float32, copy=True))
        self.qpos.append(np.array(state.qpos, dtype=np.float32, copy=True))
        self.qvel.append(np.array(state.qvel, dtype=np.float32, copy=True))
        self.contacts.append(np.array(state.contacts, dtype=np.float32, copy=True))
        self.times.append(np.asarray([state.time], dtype=np.float32))
        if not self.events or self.events[-1].get("phase") != phase:
            self.events.append({"t": state.time, "type": "replay_phase", "phase": phase})

    def determinism_digest(self) -> str:
        digest = hashlib.sha256()
        for rows, width in (
            (self.times, 1),
            (self.qpos, self.qpos_width),
            (self.qvel, self.qvel_width),
            (self.actions, self.action_width),
        ):
            digest.update(_float32_bytes(rows, width))
        return digest.hexdigest()

    def write(
        self,
        output_path: Path,
        run_manifest: dict,
        metrics: dict,
    ) -> None:
        trace_index = {
            "schema_version": "exo.trace.v1",
            "dtype": "float32-le",
            "samples": len(self.times),
            "arrays": {
                "times.f32": [len(self.times), 1],
                "observations.f32": [len(self.observations), self.observation_width],
                "actions.f32": [len(self.actions), self.action_width],
                "commands.f32": [len(self.commands), 3],
                "contacts.f32": [len(self.contacts), 2],
            },
        }
        replay_index = {
            "schema_version": "exo.replay.v1",
            "dtype": "float32-le",
            "samples": len(self.qpos),
            "arrays": {
                "qpos.f32": [len(self.qpos), self.qpos_width],
                "qvel.f32": [len(self.qvel), self.qvel_width],
            },
            "rendering": "authoritative recorded state; physics re-simulation is not required",
        }
        payloads = {
            "metrics.json": _json_bytes(metrics),
            "trace/index.json": _json_bytes(trace_index),
            "trace/times.f32": _float32_bytes(self.times, 1),
            "trace/observations.f32": _float32_bytes(self.observations, self.observation_width),
            "trace/actions.f32": _float32_bytes(self.actions, self.action_width),
            "trace/commands.f32": _float32_bytes(self.commands, 3),
            "trace/contacts.f32": _float32_bytes(self.contacts, 2),
            "trace/events.jsonl": b"".join(
                json.dumps(event, sort_keys=True).encode("utf-8") + b"\n" for event in self.events
            ),
            "replay/index.json": _json_bytes(replay_index),
            "replay/qpos.f32": _float32_bytes(self.qpos, self.qpos_width),
            "replay/qvel.f32": _float32_bytes(self.qvel, self.qvel_width),
        }
        manifest = {
            **run_manifest,
            "schema_version": "exo.run.v1",
            "runner": {
                "name": "exo-bench",
                "version": __version__,
                "mujoco_version": mujoco.__version__,
                "python": platform.python_version(),
                "platform": platform.platform(),
            },
            "determinism_digest": self.determinism_digest(),
            "contents": {name: _sha256_bytes(data) for name, data in sorted(payloads.items())},
        }
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(output_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
            archive.writestr("manifest.json", _json_bytes(manifest))
            for name, data in payloads.items():
                archive.writestr(name, data)


@dataclass(frozen=True)
class ReplaySummary:
    path: Path
    manifest: dict
    metrics: dict
    verified_files: int


def inspect_replay(path: Path) -> ReplaySummary:
    with zipfile.ZipFile(path, "r") as archive:
        names = set(archive.namelist())
        required = {"manifest.json", "metrics.json", "trace/index.json", "replay/index.json"}
        missing = required - names
        if missing:
            raise ValueError(f"Replay is missing required entries: {sorted(missing)}")
        manifest = json.loads(archive.read("manifest.json"))
        metrics = json.loads(archive.read("metrics.json"))
        verified = 0
        for name, expected in manifest.get("contents", {}).items():
            if name not in names:
                raise ValueError(f"Replay content is missing: {name}")
            actual = _sha256_bytes(archive.read(name))
            if actual != expected:
                raise ValueError(f"Replay content hash mismatch: {name}")
            verified += 1
        trace_index = json.loads(archive.read("trace/index.json"))
        replay_index = json.loads(archive.read("replay/index.json"))
        if trace_index["samples"] != replay_index["samples"]:
            raise ValueError("Trace and replay sample counts differ")
    return ReplaySummary(path=path, manifest=manifest, metrics=metrics, verified_files=verified)
