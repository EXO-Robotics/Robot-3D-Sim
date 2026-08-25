import { describe, expect, it } from "vitest";
import type { AgentDecision } from "../replay/ExoRun";
import { benchmarkModeTemplate, findDecisionAtTime, findSampleIndex } from "./BenchmarkViewer";

describe("EXO Bench browser mode", () => {
  it("exposes distinct replay, upload, localhost, agent, and provenance surfaces", () => {
    const html = benchmarkModeTemplate();
    expect(html).toContain("EXO BENCH");
    expect(html).toContain("bench-upload");
    expect(html).toContain("127.0.0.1:8765/events");
    expect(html).toContain("RECORDED BODY STATE");
    expect(html).toContain("Browser physics disabled");
    expect(html).toContain("bench-provenance");
  });

  it("selects authoritative samples and decisions at or before the playhead", () => {
    expect(findSampleIndex(new Float32Array([0, 0.5, 1, 1.5]), 0.9)).toBe(1);
    const decisions = [0, 0.5, 1].map((simulationTime) => ({ simulation_time: simulationTime })) as AgentDecision[];
    expect(findDecisionAtTime(decisions, 0.9)?.simulation_time).toBe(0.5);
    expect(findDecisionAtTime(decisions, -1)).toBeNull();
  });
});
