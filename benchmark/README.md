# EXO Bench

EXO Bench is the native, authoritative benchmark runtime that complements the
browser-based Assisted Lab. It runs a pinned free-base Unitree H1 model and a
matching pinned locomotion policy in native MuJoCo. No model API, secret, or
network service is part of this milestone.

## Setup

Use the project-local Python environment:

```bash
uv sync --extra dev --python 3.12
```

## Run

```bash
uv run exo-bench run \
  --course basic-walk-stop \
  --robot h1 \
  --controller baseline \
  --seed 42 \
  --record result.exorun \
  --verify-determinism
```

Inspect and integrity-check the recorded run:

```bash
uv run exo-bench replay result.exorun
```

Run chassis qualification:

```bash
uv run exo-bench qualify \
  --output benchmark/qualification/results/unitree-h1.json
```

## Evidence boundary

An artifact alias such as `h1` is only a user-facing convenience. It resolves
to exact manifests under `benchmark/robots` and `benchmark/controllers`.
Qualification applies to the model-policy pair, not to either artifact alone.

The v1 Unitree source model actuates ten leg joints. Its upper body is rigidly
represented in the pelvis model. Passing this milestone therefore proves an
unassisted free-base locomotion benchmark foundation, not full 19-DoF H1
whole-body control, manipulation, sim-to-real transfer, or robot-hardware
validation.

## Replay contract

`.exorun` is a ZIP-compatible research artifact containing:

- immutable robot, policy, engine, course and seed metadata;
- scientific trace arrays for observations, actions and commands;
- authoritative recorded `qpos`/`qvel` state for visualization;
- events and final metrics;
- a SHA-256 for every payload.

All numeric arrays use row-major little-endian float32. Their dimensions are
declared in the corresponding `index.json`. A viewer must play recorded state;
it must not require re-running physics to display an old result.

## Third-party artifacts

The H1 model, meshes, URDF, deployment configuration and pretrained policy are
copied without modification from `unitreerobotics/unitree_rl_gym` commit
`276801e46c5d433564f24658bac64f254b7d2d4b`. They remain under Unitree's
BSD-3-Clause license at `benchmark/third_party/unitree_rl_gym/LICENSE`.

