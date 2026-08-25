from __future__ import annotations

import json
import math
import re
import threading
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any


LIVE_SCHEMA = "exo.live.v1"
_LOCAL_ORIGIN = re.compile(r"^http://(?:127\.0\.0\.1|localhost)(?::\d+)?$")


def _finite(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def _require_finite_tree(value: object, path: str) -> None:
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return
    if isinstance(value, (int, float)):
        if not _finite(value):
            raise ValueError(f"EXO live message contains nonfinite data at {path}")
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            _require_finite_tree(item, f"{path}[{index}]")
        return
    if isinstance(value, dict):
        for key, item in value.items():
            _require_finite_tree(item, f"{path}.{key}")
        return
    raise ValueError(f"EXO live message contains an unsupported value at {path}")


def validate_live_message(message: object) -> dict[str, Any]:
    if not isinstance(message, dict) or message.get("schema_version") != LIVE_SCHEMA:
        raise ValueError("Unsupported EXO live protocol message")
    message_type = message.get("type")
    if message_type == "hello":
        if set(message) != {"schema_version", "type", "run"}:
            raise ValueError("EXO live hello fields are incomplete or unexpected")
        if not isinstance(message.get("run"), dict) or message["run"].get("browser_control_allowed") is not False:
            raise ValueError("EXO live hello must declare a visualization-only run")
    elif message_type == "frame":
        from .agent import load_runtime_contract, validate_action, validate_observation

        required = {
            "schema_version", "type", "sequence", "simulation_time", "qpos", "qvel", "body_poses", "task",
            "observation", "requested_action", "applied_action", "agent_latency_ms", "telemetry", "events",
        }
        if set(message) != required:
            raise ValueError("EXO live frame fields are incomplete or unexpected")
        if not isinstance(message["sequence"], int) or message["sequence"] < 0:
            raise ValueError("EXO live frame sequence is invalid")
        if not _finite(message["simulation_time"]) or not _finite(message["agent_latency_ms"]):
            raise ValueError("EXO live frame time or latency is nonfinite")
        if float(message["simulation_time"]) < 0 or float(message["agent_latency_ms"]) < 0:
            raise ValueError("EXO live frame time and latency must be nonnegative")
        for field, width in (("qpos", 17), ("qvel", 16), ("body_poses", 91)):
            values = message[field]
            if not isinstance(values, list) or len(values) != width or not all(_finite(value) for value in values):
                raise ValueError(f"EXO live frame {field} must contain {width} finite values")
        if not isinstance(message["events"], list):
            raise ValueError("EXO live frame events must be an array")
        validate_observation(message["observation"])
        runtime = load_runtime_contract()
        requested = message["requested_action"]
        if requested is not None and validate_action(requested, runtime).requested_action is None:
            raise ValueError("EXO live requested action is malformed")
        applied = validate_action(message["applied_action"], runtime)
        if applied.rejected or applied.clamped:
            raise ValueError("EXO live applied action is outside the trusted boundary")
        _require_finite_tree(message["task"], "frame.task")
        _require_finite_tree(message["telemetry"], "frame.telemetry")
        _require_finite_tree(message["events"], "frame.events")
    elif message_type == "complete":
        if set(message) != {"schema_version", "type", "sequence", "simulation_time", "result"}:
            raise ValueError("EXO live completion fields are incomplete or unexpected")
        if not isinstance(message.get("sequence"), int) or not _finite(message.get("simulation_time")):
            raise ValueError("EXO live completion metadata is invalid")
        if not isinstance(message.get("result"), dict):
            raise ValueError("EXO live completion is missing the authoritative result")
        _require_finite_tree(message["result"], "complete.result")
    elif message_type == "error":
        if set(message) != {"schema_version", "type", "message"}:
            raise ValueError("EXO live error fields are incomplete or unexpected")
        if not isinstance(message.get("message"), str):
            raise ValueError("EXO live error message is invalid")
    else:
        raise ValueError(f"Unsupported EXO live message type: {message_type}")
    return message


def serialize_live_message(message: dict[str, Any]) -> str:
    validate_live_message(message)
    return json.dumps(message, allow_nan=False, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def parse_live_message(payload: str | bytes) -> dict[str, Any]:
    try:
        value = json.loads(
            payload,
            parse_constant=lambda token: (_ for _ in ()).throw(ValueError(f"Nonfinite live JSON: {token}")),
        )
    except (json.JSONDecodeError, UnicodeDecodeError) as error:
        raise ValueError("Malformed EXO live JSON") from error
    validate_live_message(value)
    if serialize_live_message(value) != (payload.decode("utf-8") if isinstance(payload, bytes) else payload):
        raise ValueError("EXO live message is not canonical JSON")
    return value


class LiveMessageBuffer:
    """Append-only visualization sink; it has no command/control input surface."""

    def __init__(self) -> None:
        self._messages: list[str] = []
        self._condition = threading.Condition()
        self._complete = False

    def hello(self, message: dict[str, Any]) -> None:
        self._append(message)

    def frame(self, message: dict[str, Any]) -> None:
        self._append(message)

    def complete(self, message: dict[str, Any]) -> None:
        self._append(message)
        with self._condition:
            self._complete = True
            self._condition.notify_all()

    def snapshot(self) -> tuple[list[str], bool]:
        with self._condition:
            return list(self._messages), self._complete

    def wait_after(self, index: int, timeout: float = 1.0) -> tuple[list[str], bool]:
        with self._condition:
            if len(self._messages) <= index and not self._complete:
                self._condition.wait(timeout)
            return list(self._messages[index:]), self._complete

    def _append(self, message: dict[str, Any]) -> None:
        payload = serialize_live_message(message)
        with self._condition:
            self._messages.append(payload)
            self._condition.notify_all()


class _LiveHandler(BaseHTTPRequestHandler):
    server_version = "exo-bench-live/1"

    def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
        if self.path not in {"/", "/events"}:
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        origin = self.headers.get("Origin")
        if origin is not None and not _LOCAL_ORIGIN.fullmatch(origin):
            self.send_error(HTTPStatus.FORBIDDEN, "EXO live accepts local viewer origins only")
            return
        if self.path == "/":
            payload = b"EXO Bench live visualization bridge v1\nGET /events for server-sent events.\n"
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "text/plain; charset=utf-8")
            self.send_header("Content-Length", str(len(payload)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(payload)
            return

        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Connection", "close")
        if origin is not None:
            self.send_header("Access-Control-Allow-Origin", origin)
            self.send_header("Vary", "Origin")
        self.end_headers()
        bridge: LiveMessageBuffer = self.server.bridge  # type: ignore[attr-defined]
        index = 0
        try:
            while True:
                messages, complete = bridge.wait_after(index)
                for payload in messages:
                    self.wfile.write(b"data: " + payload.encode("utf-8") + b"\n\n")
                    self.wfile.flush()
                    index += 1
                if complete and not messages:
                    self.close_connection = True
                    return
        except (BrokenPipeError, ConnectionResetError):
            return

    def log_message(self, format: str, *args: object) -> None:
        return None


class LocalLiveServer:
    def __init__(self, port: int, bridge: LiveMessageBuffer | None = None) -> None:
        if port < 1 or port > 65535:
            raise ValueError("Live port must be between 1 and 65535")
        self.bridge = bridge or LiveMessageBuffer()
        self.server = ThreadingHTTPServer(("127.0.0.1", port), _LiveHandler)
        self.server.bridge = self.bridge  # type: ignore[attr-defined]
        self.thread = threading.Thread(target=self.server.serve_forever, name="exo-live", daemon=True)

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.server.server_port}/events"

    def start(self) -> None:
        self.thread.start()

    def close(self) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=5.0)

    def __enter__(self) -> "LocalLiveServer":
        self.start()
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.close()
