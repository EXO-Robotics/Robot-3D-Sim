# H1 // Humanoid Open Field

A client-side humanoid robotics sandbox powered by MuJoCo WebAssembly and Three.js. The robot can explore a large X/Y field, steer around physical obstacles, follow an editable autonomous controller, and expose live engineering telemetry.

This browser experience is now explicitly the **Assisted Lab**. The repository
also contains the native **EXO Bench** foundation for free-base, unassisted,
replayable embodied-AI trials. See [`benchmark/README.md`](benchmark/README.md)
and [`docs/EXO_BENCH_ARCHITECTURE.md`](docs/EXO_BENCH_ARCHITECTURE.md).

## Open the site

On macOS, double-click **`Open Robot Field.command`**.

The launcher starts the required local web server and opens `http://127.0.0.1:5173/`. Keep its Terminal window open while using the simulator.

The site cannot run by double-clicking `index.html`: browsers block the module/WASM loading path when it is opened as a local `file://` document.

Terminal alternatives:

```bash
npm install
npm run open
```

Or start without automatically opening a browser:

```bash
npm run dev
```

## Open-field controls

- Click **Enter Field** to start physics.
- Use **W/A/S/D** or the on-screen drive pad to override the autonomous controller.
- `W` and `S` move forward/backward; `A` and `D` steer.
- Switch between follow, free-orbit, and overhead cameras.
- Toggle `Δ` for center-of-mass and foot-force debug overlays.
- Power Off disables every MuJoCo actuator and lets the robot collapse under gravity.

## What is implemented

- MuJoCo 3.12 running locally through the canonical single-threaded `@mujoco/mujoco` WASM package
- Original primitive-geometry humanoid with 20 generalized DOF, 14 programmable joint servos, IMU/foot sensing, and explicit trainer assists
- X/Y navigation and yaw steering across a 120 × 120 m field
- Physical crates, a low wall, gate, ramp, and navigation beacons
- Exploration timing, odometer, reset, pause, heading, world coordinates, and simulation speed controls
- Worker-hosted JavaScript controller with copied sensor snapshots, command validation, and an 80 ms watchdog
- Bounded telemetry for speed, body pitch, energy, actuator load, and foot contact force
- Relative Vite asset paths plus a GitHub Pages deployment workflow

## Controller API

The editor defines `control(robot, dt)` and returns joint position targets plus optional field commands:

```js
function control(robot, dt) {
  return {
    drive: 1.0,
    turn: 0.25,
    targets: {
      left_hip_pitch: -0.2,
      right_hip_pitch: 0.1,
    },
  };
}
```

`robot` contains simulation time, X/Y position, velocity, yaw, IMU state, foot contact forces, and copied joint state. Targets and field commands are clamped at the host boundary.

## Verify

```bash
npm test
npm run build
```

## Publish with GitHub Pages

The production build uses relative asset URLs, so it works from a repository URL such
as `https://username.github.io/Robot-3D-Sim/` without hard-coding the repository name.
The workflow at `.github/workflows/pages.yml` tests, builds, and deploys `dist` whenever
`main` is pushed.

1. Create a GitHub repository and push this project to its `main` branch.
2. In **Settings → Pages**, select **GitHub Actions** as the publishing source.
3. Run **Deploy simulator to GitHub Pages** or push another commit to `main`.
4. Open the URL reported by the workflow's `github-pages` deployment.

## Visual model

The MuJoCo XML is a frozen physics baseline. The visible robot is a separate
GLB bound to the 13 named rigid bodies. Collision hulls stay hidden unless
debug mode is on.

- Contract: [`docs/VISUAL_MODEL_CONTRACT.md`](docs/VISUAL_MODEL_CONTRACT.md)
- Physics freeze: [`docs/PHYSICS_BASELINE.md`](docs/PHYSICS_BASELINE.md)
- Runtime asset: `public/models/aion-h1s.glb`

## Blender integration

The project-scoped official Blender Labs MCP setup is documented in
[`docs/BLENDER_MCP.md`](docs/BLENDER_MCP.md). It connects to Blender 5.1+
through a localhost-only bridge. Rebuild the visual GLB with
`tools/blender/build_aion_visual.py` after the add-on is listening on
`127.0.0.1:9876`.

## Evidence boundary

This is real rigid-body/contact simulation, but the first open-field model remains an assisted trainer. Root pitch, roll, height, and planar pace are stabilized so it is immediately navigable. It is not evidence of unassisted 3D balance, dynamically generated running, terrain perception, reinforcement learning, hardware transfer, or physical-robot validation.

The Assisted Lab remains frozen as an interactive trainer. Authoritative
free-root benchmark work happens in EXO Bench and does not reuse its trainer
actuators or claim that the browser controller is unassisted locomotion.
