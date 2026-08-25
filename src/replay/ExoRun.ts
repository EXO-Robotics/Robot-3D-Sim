import { unzipSync } from "fflate";

export interface ExoRunManifest {
  schema_version: "exo.run.v1";
  robot: { id: string; tree_sha256: string };
  controller: { id: string; policy_sha256: string };
  course: { id: string; sha256: string };
  seed: number;
  determinism_digest: string;
  contents: Record<string, string>;
}

interface ArrayIndex {
  schema_version: string;
  dtype: "float32-le";
  samples: number;
  arrays: Record<string, [number, number]>;
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
  };
  replay: {
    index: ArrayIndex;
    qpos: Float32Array;
    qvel: Float32Array;
  };
}

const decoder = new TextDecoder();

function requiredFile(files: Record<string, Uint8Array>, name: string): Uint8Array {
  const value = files[name];
  if (!value) throw new Error(`EXO run is missing ${name}`);
  return value;
}

function parseJson<T>(files: Record<string, Uint8Array>, name: string): T {
  return JSON.parse(decoder.decode(requiredFile(files, name))) as T;
}

async function sha256(value: Uint8Array): Promise<string> {
  const owned = Uint8Array.from(value);
  const digest = await crypto.subtle.digest("SHA-256", owned);
  return Array.from(new Uint8Array(digest), (byte) => byte.toString(16).padStart(2, "0")).join("");
}

function floatArray(files: Record<string, Uint8Array>, path: string, shape: [number, number]): Float32Array {
  const bytes = requiredFile(files, path);
  const expectedBytes = shape[0] * shape[1] * Float32Array.BYTES_PER_ELEMENT;
  if (bytes.byteLength !== expectedBytes) {
    throw new Error(`${path} has ${bytes.byteLength} bytes; expected ${expectedBytes}`);
  }
  const owned = Uint8Array.from(bytes);
  return new Float32Array(owned.buffer);
}

function indexedFloatArray(
  files: Record<string, Uint8Array>,
  index: ArrayIndex,
  directory: string,
  filename: string,
): Float32Array {
  const shape = index.arrays[filename];
  if (!shape) throw new Error(`${directory}/index.json does not declare ${filename}`);
  return floatArray(files, `${directory}/${filename}`, shape);
}

export async function loadExoRun(archive: ArrayBuffer | Uint8Array): Promise<LoadedExoRun> {
  const input = archive instanceof Uint8Array ? archive : new Uint8Array(archive);
  const files = unzipSync(input);
  const manifest = parseJson<ExoRunManifest>(files, "manifest.json");
  if (manifest.schema_version !== "exo.run.v1") {
    throw new Error(`Unsupported EXO run schema: ${manifest.schema_version}`);
  }
  for (const [name, expected] of Object.entries(manifest.contents)) {
    const actual = await sha256(requiredFile(files, name));
    if (actual !== expected) throw new Error(`EXO run integrity failure: ${name}`);
  }
  const traceIndex = parseJson<ArrayIndex>(files, "trace/index.json");
  const replayIndex = parseJson<ArrayIndex>(files, "replay/index.json");
  if (traceIndex.dtype !== "float32-le" || replayIndex.dtype !== "float32-le") {
    throw new Error("Unsupported EXO run numeric encoding");
  }
  if (traceIndex.samples !== replayIndex.samples) {
    throw new Error("EXO trace and replay sample counts differ");
  }
  return {
    manifest,
    metrics: parseJson<Record<string, unknown>>(files, "metrics.json"),
    trace: {
      index: traceIndex,
      times: indexedFloatArray(files, traceIndex, "trace", "times.f32"),
      observations: indexedFloatArray(files, traceIndex, "trace", "observations.f32"),
      actions: indexedFloatArray(files, traceIndex, "trace", "actions.f32"),
      commands: indexedFloatArray(files, traceIndex, "trace", "commands.f32"),
      contacts: indexedFloatArray(files, traceIndex, "trace", "contacts.f32"),
    },
    replay: {
      index: replayIndex,
      qpos: indexedFloatArray(files, replayIndex, "replay", "qpos.f32"),
      qvel: indexedFloatArray(files, replayIndex, "replay", "qvel.f32"),
    },
  };
}

