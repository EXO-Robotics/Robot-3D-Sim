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
AGENT_DIGEST_SCHEMA = "exo.agent-episode-digest.v1"
V1_REQUIRED_PAYLOADS = {
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
V2_REQUIRED_PAYLOADS = {
    *V1_REQUIRED_PAYLOADS,
    "agent/index.json",
    "agent/observations.jsonl",
    "agent/decisions.jsonl",
    "replay/body_poses.f32",
}


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, allow_nan=False, indent=2, sort_keys=True) + "\n").encode("utf-8")


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
    body_poses: list[np.ndarray] = field(default_factory=list)
    agent_observations: list[dict[str, Any]] = field(default_factory=list)
    agent_decisions: list[dict[str, Any]] = field(default_factory=list)

    def capture(
        self,
        state: SimulationState,
        observation: np.ndarray,
        action: np.ndarray,
        command: np.ndarray,
        phase: str,
        body_poses: np.ndarray | None = None,
    ) -> None:
        self.observations.append(np.array(observation, dtype=np.float32, copy=True))
        self.actions.append(np.array(action, dtype=np.float32, copy=True))
        self.commands.append(np.array(command, dtype=np.float32, copy=True))
        self.qpos.append(np.array(state.qpos, dtype=np.float32, copy=True))
        self.qvel.append(np.array(state.qvel, dtype=np.float32, copy=True))
        self.contacts.append(np.array(state.contacts, dtype=np.float32, copy=True))
        self.times.append(np.asarray([state.time], dtype=np.float32))
        if body_poses is not None:
            pose = np.asarray(body_poses, dtype=np.float32)
            if pose.shape != (91,) or not np.isfinite(pose).all():
                raise ValueError("Visual body poses must contain 91 finite float values")
            self.body_poses.append(pose.copy())
        if not self.events or self.events[-1].get("phase") != phase:
            self.events.append({"t": state.time, "type": "replay_phase", "phase": phase})

    @property
    def is_agent_run(self) -> bool:
        return bool(self.agent_observations or self.agent_decisions or self.body_poses)

    def record_agent_decision(self, observation: dict[str, Any], decision: dict[str, Any]) -> None:
        # Round-trip through canonical JSON to fail closed on nonfinite or
        # non-serializable values and detach the scientific record from callers.
        canonical_observation = json.loads(_canonical_json_bytes(observation))
        canonical_decision = json.loads(_canonical_json_bytes(decision))
        self.agent_observations.append(canonical_observation)
        self.agent_decisions.append(canonical_decision)

    def determinism_digest(self) -> str:
        digest = hashlib.sha256()
        schema = AGENT_DIGEST_SCHEMA if self.is_agent_run else DIGEST_SCHEMA
        _update_digest(digest, "schema", schema.encode("utf-8"))
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
        if self.is_agent_run:
            _update_digest(digest, "body_poses", _float32_bytes(self.body_poses, 91))
            observation_payload = b"\n".join(
                _canonical_json_bytes(observation) for observation in self.agent_observations
            )
            deterministic_decisions = []
            for decision in self.agent_decisions:
                value = dict(decision)
                value.pop("wall_latency_ms", None)
                deterministic_decisions.append(value)
            decision_payload = b"\n".join(
                _canonical_json_bytes(decision) for decision in deterministic_decisions
            )
            _update_digest(digest, "agent_observations", observation_payload)
            _update_digest(digest, "agent_decisions_without_wall_latency", decision_payload)
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
        is_agent_run = self.is_agent_run
        if is_agent_run and len(self.body_poses) != len(self.qpos):
            raise ValueError("Agent replay must record one body pose row per physical sample")
        if is_agent_run and len(self.agent_observations) != len(self.agent_decisions):
            raise ValueError("Agent replay observation and decision counts differ")
        replay_index = {
            "schema_version": "exo.replay.v2" if is_agent_run else "exo.replay.v1",
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
        if is_agent_run:
            replay_index["arrays"]["body_poses.f32"] = [len(self.body_poses), 91]
            payloads["replay/index.json"] = _json_bytes(replay_index)
            payloads["replay/body_poses.f32"] = _float32_bytes(self.body_poses, 91)
            agent_index = {
                "schema_version": "exo.agent.trace.v1",
                "observations": len(self.agent_observations),
                "decisions": len(self.agent_decisions),
                "observation_file": "observations.jsonl",
                "decision_file": "decisions.jsonl",
            }
            payloads["agent/index.json"] = _json_bytes(agent_index)
            payloads["agent/observations.jsonl"] = b"".join(
                _canonical_json_bytes(observation) + b"\n" for observation in self.agent_observations
            )
            payloads["agent/decisions.jsonl"] = b"".join(
                _canonical_json_bytes(decision) + b"\n" for decision in self.agent_decisions
            )
        determinism = {
            "class": "EXACT-SAME-RUNTIME",
            "digest_schema": AGENT_DIGEST_SCHEMA if is_agent_run else DIGEST_SCHEMA,
        }
        if is_agent_run:
            determinism["wall_clock_agent_latency_excluded"] = True
        manifest = {
            **run_manifest,
            "schema_version": "exo.run.v2" if is_agent_run else "exo.run.v1",
            "runner": {
                "name": "exo-bench",
                "version": __version__,
                "mujoco_version": mujoco.__version__,
                "python": platform.python_version(),
                "platform": platform.platform(),
            },
            "determinism_digest": self.determinism_digest(),
            "determinism": determinism,
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


def _strict_json_bytes(payload: bytes, label: str) -> Any:
    try:
        return json.loads(
            payload,
            parse_constant=lambda value: (_ for _ in ()).throw(ValueError(f"Nonfinite JSON value in {label}: {value}")),
        )
    except (json.JSONDecodeError, UnicodeDecodeError) as error:
        raise ValueError(f"Malformed JSON in {label}") from error


def _validate_finite_arrays(archive: zipfile.ZipFile, directory: str, index: dict[str, Any]) -> None:
    for filename in index["arrays"]:
        values = np.frombuffer(archive.read(f"{directory}/{filename}"), dtype="<f4")
        if not np.isfinite(values).all():
            raise ValueError(f"Replay array contains nonfinite data: {directory}/{filename}")


def inspect_replay(path: Path, require_archive_receipt: bool = False) -> ReplaySummary:
    archive_sha256 = sha256_file(path)
    archive_receipt_verified = _verify_archive_receipt(path, archive_sha256)
    if require_archive_receipt and not archive_receipt_verified:
        raise ValueError("Replay is missing its external .sha256 receipt")
    with zipfile.ZipFile(path, "r") as archive:
        names = set(archive.namelist())
        if "manifest.json" not in names:
            raise ValueError("Replay is missing required entries: ['manifest.json']")
        manifest = _strict_json_bytes(archive.read("manifest.json"), "manifest.json")
        schema_version = manifest.get("schema_version")
        if schema_version == "exo.run.v1":
            required_payloads = V1_REQUIRED_PAYLOADS
        elif schema_version == "exo.run.v2":
            required_payloads = V2_REQUIRED_PAYLOADS
        else:
            raise ValueError(f"Unsupported replay schema: {schema_version}")
        required = {"manifest.json", *required_payloads}
        missing = required - names
        if missing:
            raise ValueError(f"Replay is missing required entries: {sorted(missing)}")
        unexpected = names - required
        if unexpected:
            raise ValueError(f"Replay contains entries outside {schema_version}: {sorted(unexpected)}")
        require_sha256(manifest.get("determinism_digest"), "determinism_digest")
        metrics = _strict_json_bytes(archive.read("metrics.json"), "metrics.json")
        contents = manifest.get("contents")
        if not isinstance(contents, dict) or set(contents) != required_payloads:
            raise ValueError(f"Replay manifest must hash the complete {schema_version} payload set")
        verified = 0
        for name, expected_value in contents.items():
            expected = require_sha256(expected_value, f"contents.{name}")
            if name not in names:
                raise ValueError(f"Replay content is missing: {name}")
            actual = _sha256_bytes(archive.read(name))
            if actual != expected:
                raise ValueError(f"Replay content hash mismatch: {name}")
            verified += 1
        trace_index = _strict_json_bytes(archive.read("trace/index.json"), "trace/index.json")
        replay_index = _strict_json_bytes(archive.read("replay/index.json"), "replay/index.json")
        expected_index_schemas = {
            "trace": "exo.trace.v1",
            "replay": "exo.replay.v2" if schema_version == "exo.run.v2" else "exo.replay.v1",
        }
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
        if schema_version == "exo.run.v2":
            expected_replay.add("body_poses.f32")
        if set(trace_index.get("arrays", {})) != expected_trace or set(replay_index.get("arrays", {})) != expected_replay:
            raise ValueError("Replay indexes do not declare the complete exo.run.v1 array set")
        _validate_finite_arrays(archive, "trace", trace_index)
        _validate_finite_arrays(archive, "replay", replay_index)
        for line in archive.read("trace/events.jsonl").splitlines():
            event = _strict_json_bytes(line, "trace/events.jsonl")
            if _canonical_json_bytes(event) != line:
                raise ValueError("Replay events are not in canonical JSON form")
        if schema_version == "exo.run.v2":
            from .agent import load_runtime_contract, validate_action, validate_observation

            agent_index = _strict_json_bytes(archive.read("agent/index.json"), "agent/index.json")
            if agent_index.get("schema_version") != "exo.agent.trace.v1":
                raise ValueError("Unsupported Agent trace index schema")
            observation_lines = archive.read("agent/observations.jsonl").splitlines()
            decision_lines = archive.read("agent/decisions.jsonl").splitlines()
            if agent_index.get("observations") != len(observation_lines) or agent_index.get("decisions") != len(decision_lines):
                raise ValueError("Agent trace counts do not match agent/index.json")
            decoded: dict[str, list[dict[str, Any]]] = {"observations": [], "decisions": []}
            for label, lines in (("observations", observation_lines), ("decisions", decision_lines)):
                for line in lines:
                    value = _strict_json_bytes(line, f"agent/{label}.jsonl")
                    if _canonical_json_bytes(value) != line:
                        raise ValueError(f"Agent {label} are not in canonical JSON form")
                    if not isinstance(value, dict):
                        raise ValueError(f"Agent {label} entries must be objects")
                    decoded[label].append(value)
            runtime_contract = load_runtime_contract()
            for index, observation in enumerate(decoded["observations"]):
                validate_observation(observation)
                decision = decoded["decisions"][index]
                required_decision_fields = {
                    "simulation_time", "observation", "requested_action", "applied_action", "validation", "wall_latency_ms"
                }
                if set(decision) != required_decision_fields or decision["observation"] != observation:
                    raise ValueError("Agent decision does not bind its indexed public observation")
                if (
                    not isinstance(decision["simulation_time"], (int, float))
                    or isinstance(decision["simulation_time"], bool)
                    or not np.isfinite(float(decision["simulation_time"]))
                    or not isinstance(decision["wall_latency_ms"], (int, float))
                    or isinstance(decision["wall_latency_ms"], bool)
                    or not np.isfinite(float(decision["wall_latency_ms"]))
                    or float(decision["wall_latency_ms"]) < 0
                ):
                    raise ValueError("Agent decision time or wall latency is invalid")
                requested = decision["requested_action"]
                if requested is not None and validate_action(requested, runtime_contract).requested_action is None:
                    raise ValueError("Agent decision requested action is malformed")
                applied_validation = validate_action(decision["applied_action"], runtime_contract)
                if applied_validation.rejected or applied_validation.clamped:
                    raise ValueError("Agent decision applied action is outside the trusted boundary")
                validation = decision["validation"]
                if not isinstance(validation, dict) or set(validation) != {"clamped", "rejected", "issues"}:
                    raise ValueError("Agent decision validation receipt is malformed")
    return ReplaySummary(
        path=path,
        manifest=manifest,
        metrics=metrics,
        verified_files=verified,
        samples=trace_index["samples"],
        archive_sha256=archive_sha256,
        archive_receipt_verified=archive_receipt_verified,
    )
