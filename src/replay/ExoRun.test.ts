import { zipSync } from "fflate";
import { describe, expect, it } from "vitest";
import { loadExoRun } from "./ExoRun";

const encoder = new TextEncoder();

async function sha256(value: Uint8Array): Promise<string> {
  const digest = await crypto.subtle.digest("SHA-256", Uint8Array.from(value));
  return Array.from(new Uint8Array(digest), (byte) => byte.toString(16).padStart(2, "0")).join("");
}

async function fixture(tamper = false, omitManifestHash = false): Promise<Uint8Array> {
  const float = (values: number[]) => new Uint8Array(new Float32Array(values).buffer);
  const payloads: Record<string, Uint8Array> = {
    "metrics.json": encoder.encode('{"passed":true}\n'),
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
    "trace/events.jsonl": new Uint8Array(),
    "replay/index.json": encoder.encode(JSON.stringify({
      schema_version: "exo.replay.v1",
      dtype: "float32-le",
      samples: 1,
      arrays: { "qpos.f32": [1, 2], "qvel.f32": [1, 1] },
    })),
    "replay/qpos.f32": float([1, 2]),
    "replay/qvel.f32": float([3]),
  };
  const contents = Object.fromEntries(
    await Promise.all(Object.entries(payloads).map(async ([name, value]) => [name, await sha256(value)])),
  );
  if (omitManifestHash) delete contents["trace/actions.f32"];
  if (tamper) payloads["replay/qpos.f32"] = float([9, 9]);
  const manifest = {
    schema_version: "exo.run.v1",
    robot: { id: "exo.h1-locomotion.v1", tree_sha256: "a".repeat(64) },
    controller: { id: "exo.h1.velocity.v1", policy_sha256: "b".repeat(64) },
    course: { id: "BASIC-001@1.0.0", sha256: "c".repeat(64) },
    exo_bench: { commit: "e".repeat(40), dirty: false, source_tree_sha256: "f".repeat(64) },
    environment: { python: "3.12.11" },
    seed: 42,
    reset_profile: {
      id: "exo.reset-perturbation.basic.v1",
      sha256: "9".repeat(64),
      applied: {},
    },
    determinism: { class: "EXACT-SAME-RUNTIME", digest_schema: "exo.episode-digest.v2" },
    determinism_digest: "d".repeat(64),
    contents,
  };
  return zipSync({ "manifest.json": encoder.encode(JSON.stringify(manifest)), ...payloads });
}

describe("EXO run browser loader", () => {
  it("loads an integrity-checked scientific trace and visual replay", async () => {
    const run = await loadExoRun(await fixture());
    expect(run.manifest.robot.id).toBe("exo.h1-locomotion.v1");
    expect(Array.from(run.trace.commands)).toEqual([0.5, 0, 0]);
    expect(Array.from(run.replay.qpos)).toEqual([1, 2]);
  });

  it("rejects a payload changed after the manifest was signed", async () => {
    await expect(loadExoRun(await fixture(true))).rejects.toThrow("integrity failure");
  });

  it("rejects a manifest that omits a required payload hash", async () => {
    await expect(loadExoRun(await fixture(false, true))).rejects.toThrow("complete v1 payload set");
  });
});
