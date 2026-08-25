# EXO Bench web viewer

The existing browser simulator has two intentionally separate experiences:

- **Assisted Lab** runs the interactive WebAssembly trainer.
- **EXO Bench** visualizes native benchmark output. It does not score runs, advance MuJoCo, or send control commands.

## Replay on GitHub Pages

Open **EXO Bench** and choose **Load sample** to load the bundled
`replays/AGENT-001-seed-0.exorun` asset relative to the Pages base path. You can
also choose **Upload .exorun**; the
archive stays in the browser and is not uploaded to a server.

The viewer verifies every payload SHA-256, required entry, declared array shape,
byte length, and finite numeric value before presenting the run. `exo.run.v1`
archives remain readable for scientific data and provenance. Authoritative robot
animation requires `exo.run.v2` and its recorded `replay/body_poses.f32`; the
browser never re-runs physics to reconstruct a benchmark episode.

## Local replay development

```bash
npm install
npm run dev -- --host 127.0.0.1
```

Open the printed local URL and switch to **EXO Bench**.

## Live localhost visualization

Start the web app locally, then run a native Agent episode with the visualization
bridge enabled:

```bash
.venv/bin/exo-bench agent run \
  --course agent-navigation \
  --seed 0 \
  --live-port 8765
```

In **EXO Bench**, keep `http://127.0.0.1:8765/events` and choose **Connect**.
The versioned `exo.live.v1` EventSource stream is append-only and
visualization-only. The UI has no benchmark control endpoint, so reconnects,
rendering speed, and browser interaction cannot affect the authoritative run.

Use the replay transport for play/pause, scrubbing, reset, and 0.25x–4x playback.
Camera controls offer follow, overhead, and free orbit views. Agent observation,
requested/applied action, wall latency, telemetry, canonical events, final
metrics, and provenance remain visible beside the recorded robot state.
