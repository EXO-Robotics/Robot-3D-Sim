# EXO Bench

EXO Bench is the native, authoritative benchmark runtime that complements the
browser-based Assisted Lab. It runs a pinned free-base Unitree H1 model and a
matching pinned locomotion policy in native MuJoCo. EXO Agent v0.1 adds a
scripted navigation agent over a public high-level contract. No external model
API, secret, or remote inference service is part of this milestone.

## Setup

Use the project-local Python environment:

```bash
uv sync --frozen --extra dev --python 3.12
```

## Run

```bash
uv run exo-bench run \
  --course basic-walk-stop \
  --robot h1 \
  --controller baseline \
  --seed 42 \
  --reset-profile basic \
  --record result.exorun \
  --verify-determinism
```

Inspect and integrity-check the recorded run:

```bash
uv run exo-bench replay result.exorun --require-archive-receipt
```

Run chassis qualification:

```bash
uv run exo-bench qualify \
  --output benchmark/qualification/exo-h1-locomotion-foundation-v1.qualification.json \
  --evidence-dir benchmark/qualification/evidence
```

Run the first Agent benchmark and write a strict v2 replay:

```bash
uv run exo-bench agent run \
  --course agent-navigation \
  --agent exo.agent.scripted-baseline.v0.1 \
  --seed 0 \
  --record AGENT-001-seed-0.exorun \
  --verify-determinism
```

Run Agent v0.1 qualification:

```bash
uv run exo-bench agent qualify \
  --output benchmark/qualification/agent-v0.1/qualification.json \
  --evidence-dir benchmark/qualification/evidence
```

For localhost visualization, open the browser app in EXO Bench Live mode, then
run:

```bash
uv run exo-bench agent run \
  --seed 0 \
  --record live-agent.exorun \
  --live-port 8765
```

The read-only Server-Sent Events endpoint is
`http://127.0.0.1:8765/events`. It accepts no browser control messages. The
native runner remains authoritative.

Qualification fails closed when the source worktree is dirty. The committed
receipt binds the frozen implementation commit; it is generated only after the
implementation and qualification specification have been committed.

## Evidence boundary

An artifact alias such as `h1` is only a user-facing convenience. It resolves
to exact manifests under `benchmark/robots` and `benchmark/controllers`.
Qualification applies to the model-policy pair, not to either artifact alone.
The stable pair is `exo.h1-locomotion.v1` plus `exo.h1.velocity.v1`. The `h1`
and `baseline` aliases are conveniences and are not result identities.

The v1 Unitree source model actuates ten leg joints. Its upper body is rigidly
represented in the pelvis model. Passing this milestone therefore proves an
unassisted free-base locomotion benchmark foundation, not full 19-DoF H1
whole-body control, manipulation, sim-to-real transfer, or robot-hardware
validation.

## EXO Agent v0.1 boundary

The Agent division owns three hash-bound contracts under
`benchmark/agent_contracts`: `exo.agent.observation.v1` exposes privileged
navigation state only; `exo.agent.action.v1` permits bounded forward, lateral,
and yaw velocity commands; and `exo.agent.runtime.v1` freezes the paused
simulated-time scheduler.

Malformed or nonfinite actions deterministically become a safe stop. Finite
out-of-range actions are clamped, with requested and applied values recorded
separately. Agents never receive MuJoCo objects, low-level joint state, torque
control, or the qualified policy observation. `ScriptedBaselineAgent` uses only
the serialized public observation.

## Replay contract

`.exorun` is a ZIP-compatible research artifact containing:

- immutable robot, policy, engine, course and seed metadata;
- scientific trace arrays for observations, actions and commands;
- authoritative recorded `qpos`/`qvel` state for visualization;
- events and final metrics;
- a SHA-256 for every payload.

Every generated archive also has an external `result.exorun.sha256` sidecar.
Internal hashes protect the declared payloads; the external hash binds the ZIP
archive as a whole. The v1 parser requires the complete payload set, exact
array dimensions, and no unexpected ZIP entries. Agent runs use the strict
additive `exo.run.v2` contract: every v1 payload plus Agent observations,
requested/applied decisions with latency, and recorded 13-link body poses.

All numeric arrays use row-major little-endian float32. Their dimensions are
declared in the corresponding `index.json`. A viewer must play recorded state;
it must not require re-running physics to display an old result.

The canonical episode digest covers time, observations, actions, commands,
contacts, `qpos`, `qvel`, and key-sorted canonical events. Its claim is
`EXACT-SAME-RUNTIME`: identical source, artifacts, frozen dependency lock,
runtime, architecture, seed, and reset profile must reproduce the exact digest.
Cross-platform bitwise equality is not claimed.

Agent v2 digests additionally bind public Agent observations, decisions with
wall-clock latency removed, and recorded body poses. Latency remains in the
archive and metrics but cannot change standard-track physics or the digest.

## Foundation v1 qualification

The qualification specification is
`benchmark/qualification/foundation-v1.spec.json`. Completion requires:

- CQ-001 through CQ-007, including four seconds of unforced recovery after the
  CQ-007 push;
- BASIC-001 through BASIC-004 across seeds 0–4 using the versioned conservative
  reset-perturbation profile;
- a repeated identical run and digest for all 20 matrix cases;
- concrete model, policy, manifest, course, reset-profile, lock, source-tree,
  qualification-spec, and source-commit provenance;
- pinned single-thread deterministic Torch execution and recorded Python,
  MuJoCo, NumPy, Torch, platform, machine, and processor values;
- strict replay parsing plus the external complete-archive SHA-256 receipt;
- `collisions: null` until a course uses a real contact classifier. Zero is
  reserved for a measured zero count.

This qualifies only the stated locomotion foundation. The separate Agent v0.1
qualification binds AGENT-001, its five explicit scenarios, the scripted agent,
the public contracts, scheduler semantics, exact-same-runtime digests, and a
strict v2 replay. Neither qualification claims obstacle collision semantics,
vision, external model behavior, whole-body control, hardware transfer, or a
public leaderboard.

## Third-party artifacts

The H1 model, meshes, URDF, deployment configuration and pretrained policy are
copied without modification from `unitreerobotics/unitree_rl_gym` commit
`276801e46c5d433564f24658bac64f254b7d2d4b`. They remain under Unitree's
BSD-3-Clause license at `benchmark/third_party/unitree_rl_gym/LICENSE`.
