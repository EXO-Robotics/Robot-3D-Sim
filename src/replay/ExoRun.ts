import { unzipSync } from "fflate";

export type ExoRunSchemaVersion = "exo.run.v1" | "exo.run.v2";

export interface ExoRunManifest {
  schema_version: ExoRunSchemaVersion;
  robot: { id: string; tree_sha256: string; [key: string]: unknown };
  controller: { id: string; policy_sha256: string; [key: string]: unknown };
  course: { id: string; sha256: string; [key: string]: unknown };
  engine?: Record<string, unknown>;
  exo_bench: { commit: string; dirty: boolean; source_tree_sha256: string };
  environment: Record<string, unknown>;
  seed: number;
  reset_profile: { id: string; sha256: string; applied: Record<string, unknown> };
  determinism: { class: "EXACT-SAME-RUNTIME"; digest_schema: string };
  determinism_digest: string;
  contents: Record<string, string>;
  [key: string]: unknown;
}

export interface ArrayIndex {
  schema_version: string;
  dtype: "float32-le";
  samples: number;
  arrays: Record<string, [number, number]>;
  [key: string]: unknown;
}

export interface AgentTraceIndex {
  schema_version: "exo.agent.trace.v1";
  observations: number;
  decisions: number;
  observation_file: "observations.jsonl";
  decision_file: "decisions.jsonl";
  [key: string]: unknown;
}

export interface AgentObservation extends Record<string, unknown> {
  schema_version: "exo.agent.observation.v1";
  simulation_time_s: number;
}

export interface AgentVelocityAction extends Record<string, unknown> {
  schema_version: "exo.agent.action.v1";
  type: "velocity";
  forward_mps: number;
  lateral_mps: number;
  yaw_rate_rps: number;
}

export interface AgentDecision extends Record<string, unknown> {
  simulation_time: number;
  observation: AgentObservation;
  requested_action: AgentVelocityAction | null;
  applied_action: AgentVelocityAction;
  validation: { clamped: boolean; rejected: boolean; issues: string[]; [key: string]: unknown };
  wall_latency_ms: number;
}

export interface ReplayEvent extends Record<string, unknown> {
  t?: number;
  simulation_time?: number;
  type: string;
}

export interface LoadedExoRun {
  manifest: ExoRunManifest;
  metrics: Record<string, unknown>;
  trace: {
    index: ArrayIndex;
    times: Float32Array;
    observations: Float32Array;
    actions: Float32Array;
    commands: Float32Array;
    contacts: Float32Array;
    events: ReplayEvent[];
  };
  replay: {
    index: ArrayIndex;
    qpos: Float32Array;
    qvel: Float32Array;
    bodyPoses: Float32Array | null;
  };
  agent: {
    index: AgentTraceIndex;
    observations: AgentObservation[];
    decisions: AgentDecision[];
  } | null;
}

const decoder = new TextDecoder("utf-8", { fatal: true });
const sha256Pattern = /^[a-f0-9]{64}$/;
const v1Payloads = [
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
] as const;
const v2OnlyPayloads = [
  "agent/index.json",
  "agent/observations.jsonl",
  "agent/decisions.jsonl",
  "replay/body_poses.f32",
] as const;

function requiredPayloads(version: ExoRunSchemaVersion): Set<string> {
  return new Set(version === "exo.run.v2" ? [...v1Payloads, ...v2OnlyPayloads] : v1Payloads);
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function requiredFile(files: Record<string, Uint8Array>, name: string): Uint8Array {
  const value = files[name];
  if (!value) throw new Error(`EXO run is missing ${name}`);
  return value;
}

function decodeUtf8(files: Record<string, Uint8Array>, name: string): string {
  try {
    return decoder.decode(requiredFile(files, name));
  } catch (error) {
    throw new Error(`${name} is not valid UTF-8`, { cause: error });
  }
}

function parseJson<T>(files: Record<string, Uint8Array>, name: string): T {
  try {
    return JSON.parse(decodeUtf8(files, name)) as T;
  } catch (error) {
    throw new Error(`${name} is not valid JSON`, { cause: error });
  }
}

function parseJsonLines<T>(files: Record<string, Uint8Array>, name: string): T[] {
  return decodeUtf8(files, name)
    .split(/\r?\n/u)
    .filter((line) => line.length > 0)
    .map((line, index) => {
      try {
        return JSON.parse(line) as T;
      } catch (error) {
        throw new Error(`${name} line ${index + 1} is not valid JSON`, { cause: error });
      }
    });
}

async function sha256(value: Uint8Array): Promise<string> {
  const digest = await crypto.subtle.digest("SHA-256", Uint8Array.from(value));
  return Array.from(new Uint8Array(digest), (byte) => byte.toString(16).padStart(2, "0")).join("");
}

function assertFiniteTree(value: unknown, path: string): void {
  if (typeof value === "number") {
    if (!Number.isFinite(value)) throw new Error(`${path} contains a nonfinite number`);
    return;
  }
  if (Array.isArray(value)) {
    value.forEach((item, index) => assertFiniteTree(item, `${path}[${index}]`));
    return;
  }
  if (isRecord(value)) Object.entries(value).forEach(([key, item]) => assertFiniteTree(item, `${path}.${key}`));
}

function assertExactKeys(actual: Iterable<string>, expected: Set<string>, message: string): void {
  const values = new Set(actual);
  if (values.size !== expected.size || [...values].some((name) => !expected.has(name))) throw new Error(message);
}

function validateManifest(value: unknown): asserts value is ExoRunManifest {
  if (!isRecord(value) || (value.schema_version !== "exo.run.v1" && value.schema_version !== "exo.run.v2")) {
    const version = isRecord(value) ? String(value.schema_version) : "missing";
    throw new Error(`Unsupported EXO run schema: ${version}`);
  }
  if (!isRecord(value.contents)) throw new Error("EXO run manifest contents must be an object");
  for (const key of ["robot", "controller", "course", "exo_bench", "reset_profile", "determinism"] as const) {
    if (!isRecord(value[key])) throw new Error(`EXO run manifest ${key} must be an object`);
  }
  const robot = value.robot as Record<string, unknown>;
  const controller = value.controller as Record<string, unknown>;
  const course = value.course as Record<string, unknown>;
  const exoBench = value.exo_bench as Record<string, unknown>;
  if (typeof robot.id !== "string" || typeof controller.id !== "string" || typeof course.id !== "string") {
    throw new Error("EXO run manifest robot, controller, and course IDs are required");
  }
  if (typeof exoBench.commit !== "string" || typeof exoBench.source_tree_sha256 !== "string" || typeof exoBench.dirty !== "boolean") {
    throw new Error("EXO run source provenance is invalid");
  }
  if (!Number.isInteger(value.seed)) throw new Error("EXO run manifest seed must be an integer");
  if (typeof value.determinism_digest !== "string" || !sha256Pattern.test(value.determinism_digest)) {
    throw new Error("EXO run determinism digest is not a concrete SHA-256");
  }
  for (const [name, digest] of Object.entries(value.contents)) {
    if (typeof digest !== "string" || !sha256Pattern.test(digest)) throw new Error(`EXO run manifest hash is invalid: ${name}`);
  }
  assertFiniteTree(value, "manifest.json");
}

function validateArrayIndex(value: unknown, label: "trace" | "replay", version: ExoRunSchemaVersion): asserts value is ArrayIndex {
  if (!isRecord(value) || value.dtype !== "float32-le" || !Number.isInteger(value.samples) || (value.samples as number) < 0 || !isRecord(value.arrays)) {
    throw new Error(`Invalid ${label} index`);
  }
  const expectedSchema = label === "trace" ? "exo.trace.v1" : version === "exo.run.v2" ? "exo.replay.v2" : "exo.replay.v1";
  if (value.schema_version !== expectedSchema) throw new Error(`Unsupported ${label} index schema: ${String(value.schema_version)}`);
  for (const [name, shape] of Object.entries(value.arrays)) {
    if (!Array.isArray(shape) || shape.length !== 2 || !shape.every((item) => Number.isInteger(item) && item >= 0)) {
      throw new Error(`Invalid array shape for ${label}/${name}`);
    }
    if (shape[0] !== value.samples || shape[1] <= 0) throw new Error(`Invalid array shape for ${label}/${name}`);
  }
}

function floatArray(files: Record<string, Uint8Array>, path: string, shape: [number, number]): Float32Array {
  const bytes = requiredFile(files, path);
  const expectedBytes = shape[0] * shape[1] * Float32Array.BYTES_PER_ELEMENT;
  if (bytes.byteLength !== expectedBytes) throw new Error(`${path} has ${bytes.byteLength} bytes; expected ${expectedBytes}`);
  const values = new Float32Array(Uint8Array.from(bytes).buffer);
  for (let index = 0; index < values.length; index += 1) {
    if (!Number.isFinite(values[index])) throw new Error(`${path} contains a nonfinite value at index ${index}`);
  }
  return values;
}

function indexedFloatArray(files: Record<string, Uint8Array>, index: ArrayIndex, directory: string, filename: string): Float32Array {
  const shape = index.arrays[filename];
  if (!shape) throw new Error(`${directory}/index.json does not declare ${filename}`);
  return floatArray(files, `${directory}/${filename}`, shape);
}

function validateBodyPoseQuaternions(poses: Float32Array): void {
  for (let offset = 0; offset < poses.length; offset += 7) {
    const norm = Math.hypot(poses[offset + 3] ?? 0, poses[offset + 4] ?? 0, poses[offset + 5] ?? 0, poses[offset + 6] ?? 0);
    if (norm < 0.95 || norm > 1.05) throw new Error(`replay/body_poses.f32 quaternion is not normalized at link pose ${offset / 7}`);
  }
}

function validateVelocityAction(value: unknown, path: string): asserts value is AgentVelocityAction {
  if (!isRecord(value) || value.schema_version !== "exo.agent.action.v1" || value.type !== "velocity") {
    throw new Error(`${path} is not an exo.agent.action.v1 velocity action`);
  }
  if (Object.keys(value).length !== 5) throw new Error(`${path} contains unexpected fields`);
  for (const key of ["forward_mps", "lateral_mps", "yaw_rate_rps"] as const) {
    if (typeof value[key] !== "number" || !Number.isFinite(value[key])) throw new Error(`${path}.${key} must be finite`);
  }
}

function validateObservation(value: unknown, path: string): asserts value is AgentObservation {
  if (!isRecord(value) || value.schema_version !== "exo.agent.observation.v1") {
    throw new Error(`${path} is not an exo.agent.observation.v1 observation`);
  }
  assertExactKeys(
    Object.keys(value),
    new Set(["schema_version", "simulation_time_s", "base_position_m", "heading_yaw_rad", "planar_velocity_mps", "task", "remaining_time_s", "previous_applied_action"]),
    `${path} fields do not match exo.agent.observation.v1`,
  );
  for (const key of ["simulation_time_s", "heading_yaw_rad", "remaining_time_s"] as const) {
    if (typeof value[key] !== "number" || !Number.isFinite(value[key])) throw new Error(`${path}.${key} must be finite`);
  }
  if ((value.simulation_time_s as number) < 0 || (value.remaining_time_s as number) < 0) throw new Error(`${path} time values must be nonnegative`);
  if (Math.abs(value.heading_yaw_rad as number) > Math.PI) throw new Error(`${path}.heading_yaw_rad must be in [-pi,pi]`);
  validateAxisRecord(value.base_position_m, ["x", "y", "z"], `${path}.base_position_m`);
  validateAxisRecord(value.planar_velocity_mps, ["x", "y"], `${path}.planar_velocity_mps`);
  if (!isRecord(value.task)) throw new Error(`${path}.task must be an object`);
  assertExactKeys(
    Object.keys(value.task),
    new Set(["id", "goal_position_m", "goal_heading_rad", "target_radius_m", "heading_tolerance_rad"]),
    `${path}.task fields are invalid`,
  );
  if (value.task.id !== "AGENT-001@1.0.0") throw new Error(`${path}.task.id must be AGENT-001@1.0.0`);
  validateAxisRecord(value.task.goal_position_m, ["x", "y"], `${path}.task.goal_position_m`);
  for (const key of ["goal_heading_rad", "target_radius_m", "heading_tolerance_rad"] as const) {
    if (typeof value.task[key] !== "number" || !Number.isFinite(value.task[key])) throw new Error(`${path}.task.${key} must be finite`);
  }
  if (Math.abs(value.task.goal_heading_rad as number) > Math.PI) throw new Error(`${path}.task.goal_heading_rad must be in [-pi,pi]`);
  if ((value.task.target_radius_m as number) <= 0) throw new Error(`${path}.task.target_radius_m must be positive`);
  if ((value.task.heading_tolerance_rad as number) <= 0 || (value.task.heading_tolerance_rad as number) > Math.PI) {
    throw new Error(`${path}.task.heading_tolerance_rad must be in (0,pi]`);
  }
  if (value.previous_applied_action !== null) validateVelocityAction(value.previous_applied_action, `${path}.previous_applied_action`);
  assertFiniteTree(value, path);
}

export function assertAgentObservation(value: unknown, path = "agent observation"): asserts value is AgentObservation {
  validateObservation(value, path);
}

export function assertAgentVelocityAction(value: unknown, path = "agent action"): asserts value is AgentVelocityAction {
  validateVelocityAction(value, path);
}

function validateAxisRecord(value: unknown, axes: string[], path: string): void {
  if (!isRecord(value)) throw new Error(`${path} must be an object`);
  assertExactKeys(Object.keys(value), new Set(axes), `${path} axes are invalid`);
  for (const axis of axes) {
    if (typeof value[axis] !== "number" || !Number.isFinite(value[axis])) throw new Error(`${path}.${axis} must be finite`);
  }
}
function validateDecision(value: unknown, path: string): asserts value is AgentDecision {
  if (!isRecord(value) || typeof value.simulation_time !== "number" || !Number.isFinite(value.simulation_time)) {
    throw new Error(`${path}.simulation_time must be finite`);
  }
  assertExactKeys(
    Object.keys(value),
    new Set(["simulation_time", "observation", "requested_action", "applied_action", "validation", "wall_latency_ms"]),
    `${path} fields are incomplete or unexpected`,
  );
  if (value.simulation_time < 0) throw new Error(`${path}.simulation_time must be nonnegative`);
  validateObservation(value.observation, `${path}.observation`);
  if (value.requested_action !== null) validateVelocityAction(value.requested_action, `${path}.requested_action`);
  validateVelocityAction(value.applied_action, `${path}.applied_action`);
  if (
    !isRecord(value.validation)
    || typeof value.validation.clamped !== "boolean"
    || typeof value.validation.rejected !== "boolean"
    || !Array.isArray(value.validation.issues)
  ) {
    throw new Error(`${path}.validation is invalid`);
  }
  assertExactKeys(Object.keys(value.validation), new Set(["clamped", "rejected", "issues"]), `${path}.validation fields are invalid`);
  if (!value.validation.issues.every((issue) => typeof issue === "string")) throw new Error(`${path}.validation.issues must contain strings`);
  if ((value.requested_action === null) !== value.validation.rejected) {
    throw new Error(`${path}.requested_action nullability does not match validation.rejected`);
  }
  if (value.observation.simulation_time_s !== value.simulation_time) {
    throw new Error(`${path}.observation does not bind the decision simulation time`);
  }
  if (typeof value.wall_latency_ms !== "number" || !Number.isFinite(value.wall_latency_ms) || value.wall_latency_ms < 0) {
    throw new Error(`${path}.wall_latency_ms must be a nonnegative finite number`);
  }
  assertFiniteTree(value, path);
}

function parseEvents(files: Record<string, Uint8Array>): ReplayEvent[] {
  const events = parseJsonLines<unknown>(files, "trace/events.jsonl");
  events.forEach((event, index) => {
    if (!isRecord(event) || typeof event.type !== "string" || event.type.length === 0) {
      throw new Error(`trace/events.jsonl line ${index + 1} is not a valid event`);
    }
    assertFiniteTree(event, `trace/events.jsonl[${index}]`);
  });
  return events as ReplayEvent[];
}

function parseAgent(files: Record<string, Uint8Array>): NonNullable<LoadedExoRun["agent"]> {
  const index = parseJson<unknown>(files, "agent/index.json");
  if (
    !isRecord(index)
    || index.schema_version !== "exo.agent.trace.v1"
    || !Number.isInteger(index.observations)
    || (index.observations as number) < 0
    || !Number.isInteger(index.decisions)
    || (index.decisions as number) < 0
    || index.observation_file !== "observations.jsonl"
    || index.decision_file !== "decisions.jsonl"
  ) throw new Error("agent/index.json is invalid");
  assertExactKeys(
    Object.keys(index),
    new Set(["schema_version", "observations", "decisions", "observation_file", "decision_file"]),
    "agent/index.json contains unexpected fields",
  );
  const observations = parseJsonLines<unknown>(files, "agent/observations.jsonl");
  const decisions = parseJsonLines<unknown>(files, "agent/decisions.jsonl");
  if (observations.length !== index.observations || decisions.length !== index.decisions || observations.length !== decisions.length) {
    throw new Error("Agent trace counts do not match agent/index.json");
  }
  observations.forEach((observation, itemIndex) => validateObservation(observation, `agent/observations.jsonl[${itemIndex}]`));
  decisions.forEach((decision, itemIndex) => validateDecision(decision, `agent/decisions.jsonl[${itemIndex}]`));
  decisions.forEach((decision, itemIndex) => {
    if (canonicalJson((decision as AgentDecision).observation) !== canonicalJson(observations[itemIndex])) {
      throw new Error(`agent/decisions.jsonl[${itemIndex}].observation does not match its indexed observation`);
    }
  });
  return { index: index as unknown as AgentTraceIndex, observations: observations as AgentObservation[], decisions: decisions as AgentDecision[] };
}

function canonicalJson(value: unknown): string {
  if (Array.isArray(value)) return `[${value.map((item) => canonicalJson(item)).join(",")}]`;
  if (isRecord(value)) {
    return `{${Object.keys(value).sort().map((key) => `${JSON.stringify(key)}:${canonicalJson(value[key])}`).join(",")}}`;
  }
  return JSON.stringify(value) ?? "null";
}

export async function loadExoRun(archive: ArrayBuffer | Uint8Array): Promise<LoadedExoRun> {
  const input = archive instanceof Uint8Array ? archive : new Uint8Array(archive);
  let files: Record<string, Uint8Array>;
  try {
    files = unzipSync(input);
  } catch (error) {
    throw new Error("EXO run is not a readable ZIP archive", { cause: error });
  }
  const manifestValue = parseJson<unknown>(files, "manifest.json");
  validateManifest(manifestValue);
  const manifest = manifestValue;
  const required = requiredPayloads(manifest.schema_version);
  assertExactKeys(Object.keys(manifest.contents), required, `EXO run manifest does not hash the complete ${manifest.schema_version} payload set`);
  assertExactKeys(Object.keys(files), new Set(["manifest.json", ...required]), `EXO run archive contains missing or unexpected ${manifest.schema_version} entries`);
  for (const [name, expected] of Object.entries(manifest.contents)) {
    const actual = await sha256(requiredFile(files, name));
    if (actual !== expected) throw new Error(`EXO run integrity failure: ${name}`);
  }

  const metrics = parseJson<Record<string, unknown>>(files, "metrics.json");
  if (!isRecord(metrics)) throw new Error("metrics.json must be an object");
  assertFiniteTree(metrics, "metrics.json");
  const traceIndexValue = parseJson<unknown>(files, "trace/index.json");
  const replayIndexValue = parseJson<unknown>(files, "replay/index.json");
  validateArrayIndex(traceIndexValue, "trace", manifest.schema_version);
  validateArrayIndex(replayIndexValue, "replay", manifest.schema_version);
  const traceIndex = traceIndexValue;
  const replayIndex = replayIndexValue;
  if (traceIndex.samples !== replayIndex.samples) throw new Error("EXO trace and replay sample counts differ");

  const expectedTraceArrays = new Set(["times.f32", "observations.f32", "actions.f32", "commands.f32", "contacts.f32"]);
  const expectedReplayArrays = new Set(manifest.schema_version === "exo.run.v2"
    ? ["qpos.f32", "qvel.f32", "body_poses.f32"]
    : ["qpos.f32", "qvel.f32"]);
  assertExactKeys(Object.keys(traceIndex.arrays), expectedTraceArrays, "Trace index does not declare the complete array set");
  assertExactKeys(Object.keys(replayIndex.arrays), expectedReplayArrays, "Replay index does not declare the complete array set");
  if (traceIndex.arrays["times.f32"]?.[1] !== 1) throw new Error("trace/times.f32 must have shape [samples,1]");
  if (traceIndex.arrays["commands.f32"]?.[1] !== 3) throw new Error("trace/commands.f32 must have shape [samples,3]");
  if (traceIndex.arrays["contacts.f32"]?.[1] !== 2) throw new Error("trace/contacts.f32 must have shape [samples,2]");
  if (manifest.schema_version === "exo.run.v2" && replayIndex.arrays["body_poses.f32"]?.[1] !== 91) {
    throw new Error("replay/body_poses.f32 must have shape [samples,91]");
  }

  const times = indexedFloatArray(files, traceIndex, "trace", "times.f32");
  for (let index = 1; index < times.length; index += 1) {
    if ((times[index] ?? 0) < (times[index - 1] ?? 0)) throw new Error("trace/times.f32 is not monotonic");
  }
  const bodyPoses = manifest.schema_version === "exo.run.v2"
    ? indexedFloatArray(files, replayIndex, "replay", "body_poses.f32")
    : null;
  if (bodyPoses) validateBodyPoseQuaternions(bodyPoses);
  return {
    manifest,
    metrics,
    trace: {
      index: traceIndex,
      times,
      observations: indexedFloatArray(files, traceIndex, "trace", "observations.f32"),
      actions: indexedFloatArray(files, traceIndex, "trace", "actions.f32"),
      commands: indexedFloatArray(files, traceIndex, "trace", "commands.f32"),
      contacts: indexedFloatArray(files, traceIndex, "trace", "contacts.f32"),
      events: parseEvents(files),
    },
    replay: {
      index: replayIndex,
      qpos: indexedFloatArray(files, replayIndex, "replay", "qpos.f32"),
      qvel: indexedFloatArray(files, replayIndex, "replay", "qvel.f32"),
      bodyPoses,
    },
    agent: manifest.schema_version === "exo.run.v2" ? parseAgent(files) : null,
  };
}
