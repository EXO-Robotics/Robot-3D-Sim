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
from .hashing import require_sha256, sha256_file


DIGEST_SCHEMA = "exo.episode-digest.v2"
REQUIRED_PAYLOADS = {
    "metrics.json",
    "trace/index.json",
    "trace/times.f32",
    "trace/observations.f32",
    "trace/actions.f32",
    "trace/commands.f32",
    "trace/contacts.f32",
    "trace/events.jsonl",
    "replay/index.json",
    "replay/qpos.f32",
    "replay/qvel.f32",
}


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _float32_bytes(rows: list[np.ndarray], width: int) -> bytes:
    if not rows:
        return np.empty((0, width), dtype="<f4").tobytes()
    return np.asarray(rows, dtype="<f4").reshape(len(rows), width).tobytes(order="C")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _canonical_events(events: list[dict]) -> list[dict]:
    return sorted(
        events,
        key=lambda event: (
            float(event.get("t", 0.0)),
            str(event.get("type", "")),
            _canonical_json_bytes(event),
        ),
    )


def _update_digest(digest: Any, label: str, payload: bytes) -> None:
    encoded_label = label.encode("utf-8")
    digest.update(len(encoded_label).to_bytes(4, "big"))
    digest.update(encoded_label)
    digest.update(len(payload).to_bytes(8, "big"))
    digest.update(payload)


def _write_zip_payload(archive: zipfile.ZipFile, name: str, payload: bytes) -> None:
    info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
    info.compress_type = zipfile.ZIP_DEFLATED
    info.external_attr = 0o100644 << 16
    archive.writestr(info, payload, compress_type=zipfile.ZIP_DEFLATED, compresslevel=6)


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
        _update_digest(digest, "schema", DIGEST_SCHEMA.encode("utf-8"))
        for label, rows, width in (
            ("times", self.times, 1),
            ("observations", self.observations, self.observation_width),
            ("actions", self.actions, self.action_width),
            ("commands", self.commands, 3),
            ("contacts", self.contacts, 2),
            ("qpos", self.qpos, self.qpos_width),
            ("qvel", self.qvel, self.qvel_width),
        ):
            _update_digest(digest, label, _float32_bytes(rows, width))
        event_payload = b"\n".join(_canonical_json_bytes(event) for event in _canonical_events(self.events))
        if event_payload:
            event_payload += b"\n"
        _update_digest(digest, "canonical_events", event_payload)
        return digest.hexdigest()

    def write(
        self,
        output_path: Path,
        run_manifest: dict,
        metrics: dict,
    ) -> tuple[str, Path]:
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
        canonical_events = _canonical_events(self.events)
        payloads = {
            "metrics.json": _json_bytes(metrics),
            "trace/index.json": _json_bytes(trace_index),
            "trace/times.f32": _float32_bytes(self.times, 1),
            "trace/observations.f32": _float32_bytes(self.observations, self.observation_width),
            "trace/actions.f32": _float32_bytes(self.actions, self.action_width),
            "trace/commands.f32": _float32_bytes(self.commands, 3),
            "trace/contacts.f32": _float32_bytes(self.contacts, 2),
            "trace/events.jsonl": b"".join(_canonical_json_bytes(event) + b"\n" for event in canonical_events),
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
            "determinism": {
                "class": "EXACT-SAME-RUNTIME",
                "digest_schema": DIGEST_SCHEMA,
            },
            "contents": {name: _sha256_bytes(data) for name, data in sorted(payloads.items())},
        }
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(output_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
            _write_zip_payload(archive, "manifest.json", _json_bytes(manifest))
            for name, data in payloads.items():
                _write_zip_payload(archive, name, data)
        archive_sha256 = sha256_file(output_path)
        receipt_path = output_path.with_name(output_path.name + ".sha256")
        receipt_path.write_text(f"{archive_sha256}  {output_path.name}\n", encoding="utf-8")
        return archive_sha256, receipt_path


@dataclass(frozen=True)
class ReplaySummary:
    path: Path
    manifest: dict
    metrics: dict
    verified_files: int
    samples: int
    archive_sha256: str
    archive_receipt_verified: bool


def _validate_array(
    archive: zipfile.ZipFile,
    names: set[str],
    directory: str,
    filename: str,
    shape: object,
    samples: int,
) -> None:
    if (
        not isinstance(shape, list)
        or len(shape) != 2
        or not all(isinstance(value, int) and value >= 0 for value in shape)
        or shape[0] != samples
        or shape[1] <= 0
    ):
        raise ValueError(f"Invalid array shape for {directory}/{filename}: {shape}")
    path = f"{directory}/{filename}"
    if path not in names:
        raise ValueError(f"Replay content is missing: {path}")
    expected_bytes = shape[0] * shape[1] * 4
    if archive.getinfo(path).file_size != expected_bytes:
        raise ValueError(f"Replay array byte length mismatch: {path}")


def _verify_archive_receipt(path: Path, archive_sha256: str) -> bool:
    receipt_path = path.with_name(path.name + ".sha256")
    if not receipt_path.exists():
        return False
    parts = receipt_path.read_text(encoding="utf-8").strip().split()
    if len(parts) != 2 or parts[1] != path.name:
        raise ValueError("Archive SHA-256 receipt is malformed or names a different file")
    expected = require_sha256(parts[0], "archive_sha256")
    if expected != archive_sha256:
        raise ValueError("Archive SHA-256 receipt does not match the .exorun file")
    return True


def inspect_replay(path: Path, require_archive_receipt: bool = False) -> ReplaySummary:
    archive_sha256 = sha256_file(path)
    archive_receipt_verified = _verify_archive_receipt(path, archive_sha256)
    if require_archive_receipt and not archive_receipt_verified:
        raise ValueError("Replay is missing its external .sha256 receipt")
    with zipfile.ZipFile(path, "r") as archive:
        names = set(archive.namelist())
        required = {"manifest.json", *REQUIRED_PAYLOADS}
        missing = required - names
        if missing:
            raise ValueError(f"Replay is missing required entries: {sorted(missing)}")
        unexpected = names - required
        if unexpected:
            raise ValueError(f"Replay contains entries outside exo.run.v1: {sorted(unexpected)}")
        manifest = json.loads(archive.read("manifest.json"))
        if manifest.get("schema_version") != "exo.run.v1":
            raise ValueError(f"Unsupported replay schema: {manifest.get('schema_version')}")
        require_sha256(manifest.get("determinism_digest"), "determinism_digest")
        metrics = json.loads(archive.read("metrics.json"))
        contents = manifest.get("contents")
        if not isinstance(contents, dict) or set(contents) != REQUIRED_PAYLOADS:
            raise ValueError("Replay manifest must hash the complete exo.run.v1 payload set")
        verified = 0
        for name, expected_value in contents.items():
            expected = require_sha256(expected_value, f"contents.{name}")
            if name not in names:
                raise ValueError(f"Replay content is missing: {name}")
            actual = _sha256_bytes(archive.read(name))
            if actual != expected:
                raise ValueError(f"Replay content hash mismatch: {name}")
            verified += 1
        trace_index = json.loads(archive.read("trace/index.json"))
        replay_index = json.loads(archive.read("replay/index.json"))
        expected_index_schemas = {"trace": "exo.trace.v1", "replay": "exo.replay.v1"}
        for label, index in (("trace", trace_index), ("replay", replay_index)):
            if index.get("schema_version") != expected_index_schemas[label]:
                raise ValueError(f"Unsupported {label} index schema")
            if index.get("dtype") != "float32-le" or not isinstance(index.get("samples"), int):
                raise ValueError(f"Invalid {label} index")
        if trace_index["samples"] != replay_index["samples"]:
            raise ValueError("Trace and replay sample counts differ")
        for filename, shape in trace_index.get("arrays", {}).items():
            _validate_array(archive, names, "trace", filename, shape, trace_index["samples"])
        for filename, shape in replay_index.get("arrays", {}).items():
            _validate_array(archive, names, "replay", filename, shape, replay_index["samples"])
        expected_trace = {"times.f32", "observations.f32", "actions.f32", "commands.f32", "contacts.f32"}
        expected_replay = {"qpos.f32", "qvel.f32"}
        if set(trace_index.get("arrays", {})) != expected_trace or set(replay_index.get("arrays", {})) != expected_replay:
            raise ValueError("Replay indexes do not declare the complete exo.run.v1 array set")
        for line in archive.read("trace/events.jsonl").splitlines():
            event = json.loads(line)
            if _canonical_json_bytes(event) != line:
                raise ValueError("Replay events are not in canonical JSON form")
    return ReplaySummary(
        path=path,
        manifest=manifest,
        metrics=metrics,
        verified_files=verified,
        samples=trace_index["samples"],
        archive_sha256=archive_sha256,
        archive_receipt_verified=archive_receipt_verified,
    )
