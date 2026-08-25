# EXO Bench architecture freeze

The product has two explicit modes.

| Mode | Runtime | Purpose | Evidence |
|---|---|---|---|
| Assisted Lab | Browser MuJoCo WASM | Interactive training, visual debugging and public GitHub Pages demo | Assisted motion only |
| EXO Bench | Native headless MuJoCo | Deterministic episodes, scoring, provenance and replay | Authoritative benchmark results after qualification |

The Assisted Lab physics baseline remains frozen. EXO Bench does not import its
root translation, height, yaw, pitch or roll trainer actuators.

## Divisions

- **EXO Agent:** every entrant uses the same qualified locomotion policy and
  emits standardized velocity commands. v0.1 implements this division with a
  public privileged-navigation observation and `ScriptedBaselineAgent`.
- **EXO Control:** an entrant supplies standardized motor commands behind the
  trusted robot-specific actuator boundary.
- **EXO End-to-End:** reserved for a later milestone in which entrants supply
  perception, planning and locomotion.

Results from these divisions must never share one leaderboard.

## Timing

The standard reasoning track advances simulated time only after a valid action
is available. Model latency is recorded separately. A future real-time track
will continue stepping physics and enforce wall-clock decision deadlines.

The implemented `exo.agent.runtime.v1` scheduler produces an observation at
simulated time `t`, pauses physics during the agent call, validates the
high-level action, and advances the unchanged locomotion policy for the declared
simulated decision interval. The browser has no official-run control path.

## Agent v0.1 data flow

```text
ScriptedBaselineAgent
  -> exo.agent.observation.v1 / exo.agent.action.v1
  -> exo.agent.runtime.v1 simulated-time scheduler
  -> trusted bounded velocity command
  -> qualified exo.h1.velocity.v1 policy
  -> qualified exo.h1-locomotion.v1 native MuJoCo
  -> exo.run.v2 + read-only live visualization frames
```

The browser animates historical runs from recorded 13-link body poses; it does
not re-run benchmark physics. Local frames use the versioned `exo.live.v1` SSE
protocol on loopback and are visualization only.

## Qualification

Qualification binds a robot model and policy together:

1. CQ-001 model integrity
2. CQ-002 passive fall under gravity
3. CQ-003 30-second policy stand across ten seeds
4. CQ-004 velocity tracking
5. CQ-005 walk and stop
6. CQ-006 turns
7. CQ-007 perturbation recovery / disturbance rejection

Foundation v1 additionally requires BASIC-001 through BASIC-004 across five
versioned seeded reset perturbations, with every case repeated exactly under the
same pinned runtime. Only a pair passing every gate and every matrix case can
receive a stable alias.

The v1 determinism class is `EXACT-SAME-RUNTIME`; cross-platform bitwise
determinism is not claimed. Collision counts remain unavailable (`null`) until
a course provides an explicit allowed/task/violation contact classifier.
