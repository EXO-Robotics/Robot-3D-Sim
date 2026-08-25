import { describe, expect, it } from "vitest";
import { parseLiveMessage, requireLocalEventSourceUrl } from "./LiveProtocol";

function frame(): Record<string, unknown> {
  return {
    schema_version: "exo.live.v1",
    type: "frame",
    sequence: 4,
    simulation_time: 1.25,
    qpos: Array.from({ length: 17 }, () => 0),
    qvel: Array.from({ length: 16 }, () => 0),
    body_poses: Array.from({ length: 13 }, () => [0, 0, 1, 1, 0, 0, 0]).flat(),
    task: { phase: "navigate" },
    observation: {
      schema_version: "exo.agent.observation.v1",
      simulation_time_s: 1.25,
      base_position_m: { x: 0, y: 0, z: 1 },
      heading_yaw_rad: 0,
      planar_velocity_mps: { x: 0.7, y: 0 },
      task: {
        id: "AGENT-001@1.0.0",
        goal_position_m: { x: 2, y: 0 },
        goal_heading_rad: 0,
        target_radius_m: 0.35,
        heading_tolerance_rad: 0.2,
      },
      remaining_time_s: 18.75,
      previous_applied_action: null,
    },
    requested_action: { schema_version: "exo.agent.action.v1", type: "velocity", forward_mps: 0.8, lateral_mps: 0, yaw_rate_rps: 0.1 },
    applied_action: { schema_version: "exo.agent.action.v1", type: "velocity", forward_mps: 0.8, lateral_mps: 0, yaw_rate_rps: 0.1 },
    agent_latency_ms: 0.12,
    telemetry: { speed_mps: 0.7 },
    events: [{ type: "agent_decision", t: 1.25 }],
  };
}

describe("EXO live protocol", () => {
  it("parses a complete visualization-only frame", () => {
    const message = parseLiveMessage(JSON.stringify(frame()));
    expect(message.type).toBe("frame");
    if (message.type === "frame") expect(message.body_poses).toHaveLength(91);
  });

  it("rejects a wrong pose dimension and nonfinite telemetry", () => {
    const badShape = frame();
    badShape.body_poses = [1, 2, 3];
    expect(() => parseLiveMessage(badShape)).toThrow("length 91");
    const badTelemetry = frame();
    badTelemetry.telemetry = { speed_mps: Number.POSITIVE_INFINITY };
    expect(() => parseLiveMessage(badTelemetry)).toThrow("finite number");
  });

  it("keeps live connections localhost-only", () => {
    expect(requireLocalEventSourceUrl("http://127.0.0.1:8765/events", "https://example.test").port).toBe("8765");
    expect(() => requireLocalEventSourceUrl("https://remote.example/events", "https://example.test"))
      .toThrow("localhost");
  });
});
