import {
  assertAgentObservation,
  assertAgentVelocityAction,
  type AgentObservation,
  type AgentVelocityAction,
  type ReplayEvent,
} from "../replay/ExoRun";

export const EXO_LIVE_SCHEMA = "exo.live.v1" as const;

export interface LiveHello extends Record<string, unknown> {
  schema_version: typeof EXO_LIVE_SCHEMA;
  type: "hello";
  run: Record<string, unknown> & { browser_control_allowed: false };
}

export interface LiveFrame extends Record<string, unknown> {
  schema_version: typeof EXO_LIVE_SCHEMA;
  type: "frame";
  sequence: number;
  simulation_time: number;
  qpos: number[];
  qvel: number[];
  body_poses: number[];
  task: Record<string, unknown>;
  observation: AgentObservation;
  requested_action: AgentVelocityAction | null;
  applied_action: AgentVelocityAction;
  agent_latency_ms: number;
  telemetry: Record<string, unknown>;
  events: ReplayEvent[];
}

export interface LiveComplete extends Record<string, unknown> {
  schema_version: typeof EXO_LIVE_SCHEMA;
  type: "complete";
  sequence: number;
  simulation_time: number;
  result: Record<string, unknown>;
}

export interface LiveError extends Record<string, unknown> {
  schema_version: typeof EXO_LIVE_SCHEMA;
  type: "error";
  message: string;
}

export type LiveMessage = LiveHello | LiveFrame | LiveComplete | LiveError;

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function finiteNumber(value: unknown, path: string, minimum = -Infinity): asserts value is number {
  if (typeof value !== "number" || !Number.isFinite(value) || value < minimum) {
    throw new Error(`${path} must be a finite number${Number.isFinite(minimum) ? ` >= ${minimum}` : ""}`);
  }
}

function finiteArray(value: unknown, path: string, exactLength?: number): asserts value is number[] {
  if (!Array.isArray(value) || value.length === 0 || (exactLength !== undefined && value.length !== exactLength)) {
    throw new Error(`${path} must be ${exactLength === undefined ? "an array" : `an array of length ${exactLength}`}`);
  }
  value.forEach((item, index) => finiteNumber(item, `${path}[${index}]`));
}

function finiteTree(value: unknown, path: string): void {
  if (typeof value === "number") return finiteNumber(value, path);
  if (Array.isArray(value)) return value.forEach((item, index) => finiteTree(item, `${path}[${index}]`));
  if (isRecord(value)) Object.entries(value).forEach(([key, item]) => finiteTree(item, `${path}.${key}`));
}

function observation(value: unknown): asserts value is AgentObservation {
  assertAgentObservation(value, "frame.observation");
}

function velocityAction(value: unknown, path: string): asserts value is AgentVelocityAction {
  assertAgentVelocityAction(value, path);
}

function eventList(value: unknown): asserts value is ReplayEvent[] {
  if (!Array.isArray(value)) throw new Error("frame.events must be an array");
  value.forEach((event, index) => {
    if (!isRecord(event) || typeof event.type !== "string" || event.type.length === 0) {
      throw new Error(`frame.events[${index}] is invalid`);
    }
    finiteTree(event, `frame.events[${index}]`);
  });
}

export function parseLiveMessage(input: string | unknown): LiveMessage {
  let value: unknown = input;
  if (typeof input === "string") {
    try {
      value = JSON.parse(input);
    } catch (error) {
      throw new Error("Live message is not valid JSON", { cause: error });
    }
  }
  if (!isRecord(value) || value.schema_version !== EXO_LIVE_SCHEMA) {
    throw new Error(`Unsupported live schema: ${isRecord(value) ? String(value.schema_version) : "missing"}`);
  }
  if (value.type === "hello") {
    const keys = new Set(["schema_version", "type", "run"]);
    if (Object.keys(value).length !== keys.size || Object.keys(value).some((key) => !keys.has(key))) {
      throw new Error("Live hello fields are incomplete or unexpected");
    }
    if (!isRecord(value.run) || value.run.browser_control_allowed !== false) {
      throw new Error("Live hello must declare browser_control_allowed false");
    }
    finiteTree(value, "hello");
    return value as LiveHello;
  }
  if (value.type === "error") {
    const keys = new Set(["schema_version", "type", "message"]);
    if (Object.keys(value).length !== keys.size || Object.keys(value).some((key) => !keys.has(key))) {
      throw new Error("Live error fields are incomplete or unexpected");
    }
    if (typeof value.message !== "string" || value.message.length === 0) throw new Error("Live error message is missing");
    finiteTree(value, "error");
    return value as LiveError;
  }
  if (value.type === "complete") {
    const keys = new Set(["schema_version", "type", "sequence", "simulation_time", "result"]);
    if (Object.keys(value).length !== keys.size || Object.keys(value).some((key) => !keys.has(key))) {
      throw new Error("Live completion fields are incomplete or unexpected");
    }
    if (!Number.isInteger(value.sequence) || (value.sequence as number) < 0) throw new Error("Live completion sequence is invalid");
    finiteNumber(value.simulation_time, "complete.simulation_time", 0);
    if (!isRecord(value.result)) throw new Error("Live completion result must be an object");
    finiteTree(value, "complete");
    return value as LiveComplete;
  }
  if (value.type !== "frame") throw new Error(`Unsupported live message type: ${String(value.type)}`);
  const frameKeys = new Set(["schema_version", "type", "sequence", "simulation_time", "qpos", "qvel", "body_poses", "task", "observation", "requested_action", "applied_action", "agent_latency_ms", "telemetry", "events"]);
  if (Object.keys(value).length !== frameKeys.size || Object.keys(value).some((key) => !frameKeys.has(key))) {
    throw new Error("Live frame fields are incomplete or unexpected");
  }
  if (!Number.isInteger(value.sequence) || (value.sequence as number) < 0) throw new Error("frame.sequence must be a nonnegative integer");
  finiteNumber(value.simulation_time, "frame.simulation_time", 0);
  finiteArray(value.qpos, "frame.qpos", 17);
  finiteArray(value.qvel, "frame.qvel", 16);
  finiteArray(value.body_poses, "frame.body_poses", 91);
  validatePoseQuaternions(value.body_poses, "frame.body_poses");
  if (!isRecord(value.task)) throw new Error("frame.task must be an object");
  if (!isRecord(value.telemetry)) throw new Error("frame.telemetry must be an object");
  finiteTree(value.task, "frame.task");
  finiteTree(value.telemetry, "frame.telemetry");
  observation(value.observation);
  if (value.requested_action !== null) velocityAction(value.requested_action, "frame.requested_action");
  velocityAction(value.applied_action, "frame.applied_action");
  finiteNumber(value.agent_latency_ms, "frame.agent_latency_ms", 0);
  eventList(value.events);
  return value as unknown as LiveFrame;
}

function validatePoseQuaternions(poses: number[], path: string): void {
  for (let link = 0; link < 13; link += 1) {
    const offset = link * 7;
    const norm = Math.hypot(poses[offset + 3] ?? 0, poses[offset + 4] ?? 0, poses[offset + 5] ?? 0, poses[offset + 6] ?? 0);
    if (norm < 0.95 || norm > 1.05) throw new Error(`${path} link ${link} quaternion is not normalized`);
  }
}

export function requireLocalEventSourceUrl(value: string, pageOrigin = window.location.origin): URL {
  let url: URL;
  try {
    url = new URL(value, pageOrigin);
  } catch (error) {
    throw new Error("Live stream URL is invalid", { cause: error });
  }
  if (url.protocol !== "http:" && url.protocol !== "https:") throw new Error("Live stream URL must use HTTP(S)");
  if (url.hostname !== "127.0.0.1" && url.hostname !== "localhost" && url.hostname !== "[::1]") {
    throw new Error("Live visualization only accepts localhost streams");
  }
  return url;
}
