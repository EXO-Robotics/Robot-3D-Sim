import { zipSync } from "fflate";
import { describe, expect, it } from "vitest";
import { loadExoRun, type ExoRunSchemaVersion } from "./ExoRun";

const encoder = new TextEncoder();

async function sha256(value: Uint8Array): Promise<string> {
  const digest = await crypto.subtle.digest("SHA-256", Uint8Array.from(value));
  return Array.from(new Uint8Array(digest), (byte) => byte.toString(16).padStart(2, "0")).join("");
}

function float(values: number[]): Uint8Array {
  return new Uint8Array(new Float32Array(values).buffer);
}

function canonicalJson(value: unknown): string {
  if (Array.isArray(value)) return `[${value.map(canonicalJson).join(",")}]`;
  if (typeof value === "object" && value !== null) {
    const record = value as Record<string, unknown>;
    return `{${Object.keys(record).sort().map((key) => `${JSON.stringify(key)}:${canonicalJson(record[key])}`).join(",")}}`;
  }
  return JSON.stringify(value);
}

async function fixture(
  version: ExoRunSchemaVersion = "exo.run.v1",
  mutate?: (payloads: Record<string, Uint8Array>) => void,
  omitManifestHash?: string,
  tamperAfterSign?: (payloads: Record<string, Uint8Array>) => void,
): Promise<Uint8Array> {
  const v2 = version === "exo.run.v2";
  const payloads: Record<string, Uint8Array> = {
    "metrics.json": encoder.encode('{"passed":true,"collisions":null}\n'),
    "trace/index.json": encoder.encode(JSON.stringify({
      schema_version: "exo.trace.v1",
      dtype: "float32-le",
      samples: 1,
      arrays: {
        "times.f32": [1, 1],
        "observations.f32": [1, 2],
        "actions.f32": [1, 1],
        "commands.f32": [1, 3],
        "contacts.f32": [1, 2],
      },
    })),
    "trace/times.f32": float([0.02]),
    "trace/observations.f32": float([1, 2]),
    "trace/actions.f32": float([0.5]),
    "trace/commands.f32": float([0.5, 0, 0]),
    "trace/contacts.f32": float([10, 12]),
    "trace/events.jsonl": encoder.encode(`${canonicalJson({ t: 0.02, type: "course_started" })}\n`),
    "replay/index.json": encoder.encode(JSON.stringify({
      schema_version: v2 ? "exo.replay.v2" : "exo.replay.v1",
      dtype: "float32-le",
      samples: 1,
      arrays: v2
        ? { "qpos.f32": [1, 2], "qvel.f32": [1, 1], "body_poses.f32": [1, 91] }
        : { "qpos.f32": [1, 2], "qvel.f32": [1, 1] },
    })),
    "replay/qpos.f32": float([1, 2]),
    "replay/qvel.f32": float([3]),
  };
  if (v2) {
    const bodyPose = Array.from({ length: 13 }, (_, index) => [index, 0, 1, 1, 0, 0, 0]).flat();
    const observation = {
      schema_version: "exo.agent.observation.v1",
      simulation_time_s: 0.02,
      base_position_m: { x: 0, y: 0, z: 1 },
      heading_yaw_rad: 0,
      planar_velocity_mps: { x: 0, y: 0 },
      task: {
        id: "AGENT-001@1.0.0",
        goal_position_m: { x: 2, y: 0 },
        goal_heading_rad: 0,
        target_radius_m: 0.35,
        heading_tolerance_rad: 0.2,
      },
      remaining_time_s: 20,
      previous_applied_action: null,
    };
    const action = { schema_version: "exo.agent.action.v1", type: "velocity", forward_mps: 0.5, lateral_mps: 0, yaw_rate_rps: 0 };
    const decision = {
      simulation_time: 0.02,
      observation,
      requested_action: action,
      applied_action: action,
      validation: { clamped: false, rejected: false, issues: [] },
      wall_latency_ms: 0.1,
    };
    payloads["replay/body_poses.f32"] = float(bodyPose);
    payloads["agent/index.json"] = encoder.encode(JSON.stringify({
      schema_version: "exo.agent.trace.v1",
      observations: 1,
      decisions: 1,
      observation_file: "observations.jsonl",
      decision_file: "decisions.jsonl",
    }));
    payloads["agent/observations.jsonl"] = encoder.encode(`${canonicalJson(observation)}\n`);
    payloads["agent/decisions.jsonl"] = encoder.encode(`${canonicalJson(decision)}\n`);
  }
  mutate?.(payloads);
  const contents = Object.fromEntries(
    await Promise.all(Object.entries(payloads).map(async ([name, value]) => [name, await sha256(value)])),
  );
  if (omitManifestHash) delete contents[omitManifestHash];
  tamperAfterSign?.(payloads);
  const manifest = {
    schema_version: version,
    robot: { id: "exo.h1-locomotion.v1", tree_sha256: "a".repeat(64) },
    controller: { id: "exo.h1.velocity.v1", policy_sha256: "b".repeat(64) },
    course: { id: v2 ? "AGENT-001@1.0.0" : "BASIC-001@1.0.0", sha256: "c".repeat(64) },
    exo_bench: { commit: "e".repeat(40), dirty: false, source_tree_sha256: "f".repeat(64) },
    environment: { python: "3.12.11" },
    seed: 42,
    reset_profile: { id: "exo.reset-perturbation.basic.v1", sha256: "9".repeat(64), applied: {} },
    determinism: { class: "EXACT-SAME-RUNTIME", digest_schema: "exo.episode-digest.v2" },
    determinism_digest: "d".repeat(64),
    contents,
  };
  return zipSync({ "manifest.json": encoder.encode(JSON.stringify(manifest)), ...payloads });
}

describe("EXO run browser loader", () => {
  it("keeps strict v1 replay compatibility", async () => {
    const run = await loadExoRun(await fixture());
    expect(run.manifest.robot.id).toBe("exo.h1-locomotion.v1");
    expect(Array.from(run.trace.commands)).toEqual([0.5, 0, 0]);
    expect(run.replay.bodyPoses).toBeNull();
    expect(run.agent).toBeNull();
  });

  it("loads a v2 agent trace and authoritative body-link poses", async () => {
    const run = await loadExoRun(await fixture("exo.run.v2"));
    expect(run.agent?.decisions).toHaveLength(1);
    expect(run.agent?.decisions[0]?.applied_action.forward_mps).toBe(0.5);
    expect(run.replay.bodyPoses).toHaveLength(91);
    expect(run.trace.events[0]?.type).toBe("course_started");
  });

  it("rejects a payload changed after the manifest was signed", async () => {
    await expect(loadExoRun(await fixture("exo.run.v1", undefined, undefined, (payloads) => {
      payloads["replay/qpos.f32"] = float([9, 9]);
    }))).rejects.toThrow("integrity failure");
  });

  it("rejects a v2 manifest that omits an agent payload hash", async () => {
    await expect(loadExoRun(await fixture("exo.run.v2", undefined, "agent/decisions.jsonl")))
      .rejects.toThrow("complete exo.run.v2 payload set");
  });

  it("rejects a malformed v2 body-pose dimension", async () => {
    await expect(loadExoRun(await fixture("exo.run.v2", (payloads) => {
      const index = JSON.parse(new TextDecoder().decode(payloads["replay/index.json"])) as { arrays: Record<string, number[]> };
      index.arrays["body_poses.f32"] = [1, 90];
      payloads["replay/index.json"] = encoder.encode(JSON.stringify(index));
    }))).rejects.toThrow("shape [samples,91]");
  });

  it("rejects nonfinite replay arrays even when their hash is valid", async () => {
    await expect(loadExoRun(await fixture("exo.run.v2", (values) => {
      const poses = Array.from({ length: 91 }, () => 0);
      poses[8] = Number.NaN;
      values["replay/body_poses.f32"] = float(poses);
    }))).rejects.toThrow("nonfinite value");
  });

  it("rejects a decision that does not bind its indexed public observation", async () => {
    await expect(loadExoRun(await fixture("exo.run.v2", (payloads) => {
      const decision = JSON.parse(new TextDecoder().decode(payloads["agent/decisions.jsonl"])) as {
        observation: { base_position_m: { x: number } };
      };
      decision.observation.base_position_m.x = 0.25;
      payloads["agent/decisions.jsonl"] = encoder.encode(`${JSON.stringify(decision)}\n`);
    }))).rejects.toThrow("does not match its indexed observation");
  });
});

export { fixture as createExoRunFixture };
