import { BenchmarkScene, type BenchmarkCameraMode } from "./BenchmarkScene";
import { parseLiveMessage, requireLocalEventSourceUrl, type LiveFrame } from "./LiveProtocol";
import {
  loadExoRun,
  type AgentDecision,
  type AgentObservation,
  type AgentVelocityAction,
  type LoadedExoRun,
  type ReplayEvent,
} from "../replay/ExoRun";

export const SAMPLE_REPLAY_PUBLIC_PATH = "replays/AGENT-001-seed-0.exorun";
const BODY_POSE_WIDTH = 91;

export function findSampleIndex(times: Float32Array, simulationTime: number): number {
  if (times.length === 0) return 0;
  let low = 0;
  let high = times.length - 1;
  while (low <= high) {
    const middle = (low + high) >>> 1;
    if ((times[middle] ?? 0) <= simulationTime) low = middle + 1;
    else high = middle - 1;
  }
  return Math.max(0, Math.min(times.length - 1, high));
}

export function findDecisionAtTime(decisions: AgentDecision[], simulationTime: number): AgentDecision | null {
  let selected: AgentDecision | null = null;
  for (const decision of decisions) {
    if (decision.simulation_time > simulationTime) break;
    selected = decision;
  }
  return selected;
}

export class BenchmarkViewer {
  private scene: BenchmarkScene | null = null;
  private run: LoadedExoRun | null = null;
  private eventSource: EventSource | null = null;
  private animationId = 0;
  private previousWallTime = performance.now();
  private playhead = 0;
  private playbackSpeed = 1;
  private playing = false;
  private started = false;
  private sampleAttempted = false;
  private renderedSample = -1;
  private liveSequence = -1;
  private liveEvents: ReplayEvent[] = [];

  constructor(private readonly host: HTMLElement) {
    this.host.innerHTML = benchmarkModeTemplate();
    this.bindUi();
  }

  async start(): Promise<void> {
    if (!this.started) {
      this.started = true;
      this.setStatus("LOADING VISUAL", "loading");
      try {
        this.scene = await BenchmarkScene.create(this.requireElement("#bench-viewport"));
        this.setStatus("READY", "ready");
        this.previousWallTime = performance.now();
        this.animationId = requestAnimationFrame((time) => this.animate(time));
      } catch (error) {
        this.showError(`Visual model failed: ${errorMessage(error)}`);
        return;
      }
    }
    if (!this.sampleAttempted) {
      this.sampleAttempted = true;
      await this.loadSample(true);
    }
  }

  setVisible(visible: boolean): void {
    this.host.classList.toggle("experience--active", visible);
  }

  async loadArchive(archive: ArrayBuffer | Uint8Array, label = "LOCAL REPLAY"): Promise<void> {
    this.setStatus("VERIFYING", "loading");
    this.clearError();
    try {
      const run = await loadExoRun(archive);
      this.acceptRun(run, label);
    } catch (error) {
      this.setStatus("REJECTED", "error");
      this.showError(`Replay rejected: ${errorMessage(error)}`);
      throw error;
    }
  }

  dispose(): void {
    cancelAnimationFrame(this.animationId);
    this.disconnectLive();
    this.scene?.dispose();
  }

  private bindUi(): void {
    this.requireElement("#bench-sample").addEventListener("click", () => void this.loadSample(false));
    const upload = this.requireInput("#bench-upload");
    upload.addEventListener("change", () => {
      const file = upload.files?.[0];
      if (!file) return;
      void file.arrayBuffer().then((bytes) => this.loadArchive(bytes, file.name)).catch(() => undefined);
      upload.value = "";
    });
    this.requireElement("#bench-play").addEventListener("click", () => this.togglePlayback());
    this.requireElement("#bench-reset").addEventListener("click", () => this.resetPlayback());
    this.requireInput("#bench-timeline").addEventListener("input", (event) => {
      this.playing = false;
      this.playhead = Number((event.currentTarget as HTMLInputElement).value);
      this.renderedSample = -1;
      this.renderCurrentReplay();
      this.syncPlaybackUi();
    });
    this.requireInput("#bench-speed").addEventListener("change", (event) => {
      this.playbackSpeed = Number((event.currentTarget as HTMLSelectElement).value);
    });
    this.host.querySelectorAll<HTMLButtonElement>("[data-bench-camera]").forEach((button) => {
      button.addEventListener("click", () => {
        this.host.querySelectorAll("[data-bench-camera]").forEach((item) => item.classList.remove("bench-icon-button--active"));
        button.classList.add("bench-icon-button--active");
        this.scene?.setCameraMode(button.dataset.benchCamera as BenchmarkCameraMode);
      });
    });
    this.requireElement("#bench-live-toggle").addEventListener("click", () => {
      if (this.eventSource) this.disconnectLive();
      else this.connectLive();
    });
  }

  private async loadSample(quietMissing: boolean): Promise<void> {
    this.setStatus("LOADING SAMPLE", "loading");
    this.clearError();
    try {
      const url = `${import.meta.env.BASE_URL}${SAMPLE_REPLAY_PUBLIC_PATH}`;
      const response = await fetch(url);
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      const run = await loadExoRun(await response.arrayBuffer());
      this.acceptRun(run, "BUNDLED / AGENT-001 / SEED 0");
    } catch (error) {
      this.setStatus("READY", "ready");
      if (!quietMissing) this.showError(`Sample replay unavailable: ${errorMessage(error)}`);
      else {
        this.clearError();
        this.requireElement("#bench-source-note").textContent = "Sample replay is not bundled in this build. Upload a local .exorun or connect localhost.";
      }
    }
  }

  private acceptRun(run: LoadedExoRun, label: string): void {
    this.disconnectLive();
    this.run = run;
    this.liveEvents = [];
    this.renderedSample = -1;
    this.playhead = run.trace.times[0] ?? 0;
    this.playing = false;
    this.requireElement("#bench-source-note").textContent = `${label} · ${run.trace.index.samples.toLocaleString()} verified samples`;
    this.requireElement("#bench-course").textContent = run.manifest.course.id;
    this.requireElement("#bench-schema").textContent = run.manifest.schema_version.toUpperCase();
    this.renderRunIdentity(run);
    this.configureTimeline(run);
    this.renderCurrentReplay();
    const hasPoses = run.replay.bodyPoses !== null;
    this.setStatus(hasPoses ? "REPLAY VERIFIED" : "V1 DATA VERIFIED", hasPoses ? "ready" : "warning");
    if (!hasPoses) {
      this.showError("This exo.run.v1 archive is verified, but it predates authoritative body_poses. Data and provenance are available; robot animation requires exo.run.v2.");
    }
  }

  private configureTimeline(run: LoadedExoRun): void {
    const timeline = this.requireInput("#bench-timeline");
    timeline.min = String(run.trace.times[0] ?? 0);
    timeline.max = String(run.trace.times[run.trace.times.length - 1] ?? 0);
    timeline.step = "any";
    timeline.value = String(this.playhead);
    timeline.disabled = run.trace.times.length === 0;
    this.requireElement("#bench-play").toggleAttribute("disabled", run.trace.times.length === 0);
    this.requireElement("#bench-reset").toggleAttribute("disabled", run.trace.times.length === 0);
  }

  private animate(now: number): void {
    const wallDelta = Math.min(0.1, (now - this.previousWallTime) / 1000);
    this.previousWallTime = now;
    if (this.playing && this.run) {
      const end = this.run.trace.times[this.run.trace.times.length - 1] ?? 0;
      this.playhead = Math.min(end, this.playhead + wallDelta * this.playbackSpeed);
      if (this.playhead >= end) this.playing = false;
      this.renderCurrentReplay();
    }
    this.scene?.render();
    this.syncPlaybackUi();
    this.animationId = requestAnimationFrame((time) => this.animate(time));
  }

  private togglePlayback(): void {
    if (!this.run) return;
    const end = this.run.trace.times[this.run.trace.times.length - 1] ?? 0;
    if (this.playhead >= end) this.playhead = this.run.trace.times[0] ?? 0;
    this.playing = !this.playing;
    this.syncPlaybackUi();
  }

  private resetPlayback(): void {
    this.playing = false;
    this.playhead = this.run?.trace.times[0] ?? 0;
    this.renderedSample = -1;
    this.renderCurrentReplay();
    this.syncPlaybackUi();
  }

  private renderCurrentReplay(): void {
    const run = this.run;
    if (!run || run.trace.times.length === 0) return;
    const sample = findSampleIndex(run.trace.times, this.playhead);
    if (sample === this.renderedSample) return;
    this.renderedSample = sample;
    const simulationTime = run.trace.times[sample] ?? 0;
    this.playhead = simulationTime;
    const bodyPoses = run.replay.bodyPoses;
    if (bodyPoses) this.scene?.updateBodyPoses(bodyPoses.subarray(sample * BODY_POSE_WIDTH, (sample + 1) * BODY_POSE_WIDTH));
    const decision = run.agent ? findDecisionAtTime(run.agent.decisions, simulationTime) : null;
    const observation = decision?.observation ?? closestObservation(run.agent?.observations ?? [], simulationTime);
    this.scene?.setGoal(goalFromObservation(observation));
    this.renderAgent(observation, decision?.requested_action ?? null, decision?.applied_action ?? null, decision?.wall_latency_ms ?? null, decision?.validation.clamped ?? false);
    this.renderTelemetry(run, sample, simulationTime);
    this.renderEvents(run.trace.events, simulationTime);
    this.requireElement("#bench-sim-time").textContent = `${simulationTime.toFixed(3)} S`;
  }

  private renderRunIdentity(run: LoadedExoRun): void {
    const passed = run.metrics.passed === true;
    const result = this.requireElement("#bench-result");
    result.textContent = run.metrics.passed === undefined ? "UNSCORED" : passed ? "PASS" : "FAIL";
    result.className = `bench-result bench-result--${passed ? "pass" : run.metrics.passed === false ? "fail" : "idle"}`;
    const metrics = [
      ["COMPLETION", metric(run.metrics, ["simulated_completion_time_s", "completion_time_seconds", "time_seconds"], "s")],
      ["PATH", metric(run.metrics, ["path_length_m", "path_length_meters", "progress_meters"], "m")],
      ["POSITION ERROR", metric(run.metrics, ["final_position_error_m", "final_position_error_meters"], "m")],
      ["HEADING ERROR", metric(run.metrics, ["final_heading_error_rad", "final_heading_error_radians"], "rad")],
      ["ENERGY", metric(run.metrics, ["energy_joules"], "J")],
      ["FALLS", metric(run.metrics, ["falls"], "")],
    ];
    this.requireElement("#bench-final-metrics").innerHTML = metrics.map(([label, value]) => `
      <div><span>${escapeHtml(label)}</span><strong>${escapeHtml(value)}</strong></div>
    `).join("");
    const manifest = run.manifest;
    this.requireElement("#bench-provenance").innerHTML = [
      ["ROBOT", manifest.robot.id],
      ["CONTROLLER", manifest.controller.id],
      ["AGENT", typeof manifest.agent === "object" && manifest.agent !== null ? String((manifest.agent as Record<string, unknown>).id ?? "—") : "—"],
      ["SEED", String(manifest.seed)],
      ["COMMIT", manifest.exo_bench.commit.slice(0, 12)],
      ["SOURCE TREE", manifest.exo_bench.source_tree_sha256.slice(0, 12)],
      ["EPISODE DIGEST", manifest.determinism_digest.slice(0, 16)],
      ["DETERMINISM", manifest.determinism.class],
    ].map(([label, value]) => `<div><span>${escapeHtml(label)}</span><code>${escapeHtml(value)}</code></div>`).join("");
  }

  private renderAgent(
    observation: AgentObservation | null,
    requested: AgentVelocityAction | null,
    applied: AgentVelocityAction | null,
    latency: number | null,
    clamped: boolean,
  ): void {
    this.requireElement("#bench-observation").innerHTML = observation
      ? summaryRows(observation, ["schema_version", "previous_action"])
      : emptyState("No agent observation at this time.");
    this.requireElement("#bench-requested").innerHTML = actionRows(requested);
    this.requireElement("#bench-applied").innerHTML = actionRows(applied);
    this.requireElement("#bench-latency").textContent = latency === null ? "—" : `${latency.toFixed(3)} MS`;
    this.requireElement("#bench-validation").textContent = clamped ? "CLAMPED" : applied ? "VALID" : "—";
    this.requireElement("#bench-validation").classList.toggle("bench-validation--clamped", clamped);
  }

  private renderTelemetry(run: LoadedExoRun, sample: number, simulationTime: number): void {
    const poses = run.replay.bodyPoses;
    const poseOffset = sample * BODY_POSE_WIDTH;
    const x = poses?.[poseOffset] ?? run.replay.qpos[sample * (run.replay.index.arrays["qpos.f32"]?.[1] ?? 1)] ?? 0;
    const y = poses?.[poseOffset + 1] ?? 0;
    const qvelWidth = run.replay.index.arrays["qvel.f32"]?.[1] ?? 1;
    const vx = run.replay.qvel[sample * qvelWidth] ?? 0;
    const vy = run.replay.qvel[sample * qvelWidth + 1] ?? 0;
    const speed = Math.hypot(vx, vy);
    const contactsOffset = sample * 2;
    const telemetry = [
      ["POSITION", `${x.toFixed(2)}, ${y.toFixed(2)} M`],
      ["PLANAR SPEED", `${speed.toFixed(2)} M/S`],
      ["SIMULATION", `${simulationTime.toFixed(3)} S`],
      ["CONTACT L/R", `${(run.trace.contacts[contactsOffset] ?? 0).toFixed(0)} / ${(run.trace.contacts[contactsOffset + 1] ?? 0).toFixed(0)} N`],
      ["ENERGY FINAL", metric(run.metrics, ["energy_joules"], "J")],
      ["FALLS", metric(run.metrics, ["falls"], "")],
    ];
    this.requireElement("#bench-telemetry").innerHTML = telemetry.map(([label, value]) => `
      <div><span>${escapeHtml(label)}</span><strong>${escapeHtml(value)}</strong></div>
    `).join("");
  }

  private renderEvents(events: ReplayEvent[], simulationTime: number): void {
    const indexed = events.map((event, index) => ({ event, index, time: eventTime(event) }));
    const current = indexed.reduce((selected, item) => item.time <= simulationTime ? item.index : selected, -1);
    const start = Math.max(0, current - 24);
    const visible = indexed.slice(start, start + 64);
    this.requireElement("#bench-events").innerHTML = visible.length === 0 ? emptyState("No events recorded.") : visible.map(({ event, index, time }) => `
      <button class="bench-event ${index === current ? "bench-event--current" : ""}" data-event-time="${time}">
        <time>${time.toFixed(3)}</time><span>${escapeHtml(event.type.replaceAll("_", " "))}</span><small>${escapeHtml(eventDetail(event))}</small>
      </button>
    `).join("");
    this.host.querySelectorAll<HTMLButtonElement>("[data-event-time]").forEach((button) => {
      button.addEventListener("click", () => {
        this.playing = false;
        this.playhead = Number(button.dataset.eventTime);
        this.renderedSample = -1;
        this.renderCurrentReplay();
      });
    });
  }

  private connectLive(): void {
    this.clearError();
    try {
      const url = requireLocalEventSourceUrl(this.requireInput("#bench-live-url").value);
      this.run = null;
      this.playing = false;
      this.liveSequence = -1;
      this.liveEvents = [];
      const source = new EventSource(url);
      this.eventSource = source;
      const onData = (event: Event) => {
        if (!(event instanceof MessageEvent)) return;
        try {
          this.handleLiveMessage(parseLiveMessage(event.data));
        } catch (error) {
          this.showError(`Live frame rejected: ${errorMessage(error)}`);
          this.disconnectLive();
        }
      };
      source.onmessage = onData;
      for (const name of ["hello", "frame", "complete", "error"]) source.addEventListener(name, onData);
      source.onopen = () => this.setStatus("LIVE / LOCALHOST", "live");
      source.onerror = (event) => {
        if (event instanceof MessageEvent) return;
        if (this.eventSource === source) this.setStatus("LIVE RECONNECTING", "warning");
      };
      this.requireElement("#bench-live-toggle").textContent = "DISCONNECT";
      this.requireElement("#bench-source-note").textContent = `Visualization-only stream · ${url.toString()}`;
    } catch (error) {
      this.showError(errorMessage(error));
    }
  }

  private disconnectLive(): void {
    this.eventSource?.close();
    this.eventSource = null;
    const toggle = this.host.querySelector<HTMLElement>("#bench-live-toggle");
    if (toggle) toggle.textContent = "CONNECT";
  }

  private handleLiveMessage(message: ReturnType<typeof parseLiveMessage>): void {
    if (message.type === "hello") {
      this.setStatus("LIVE / LOCALHOST", "live");
      return;
    }
    if (message.type === "error") {
      this.showError(`Runner reported: ${message.message}`);
      return;
    }
    if (message.type === "complete") {
      this.renderLiveMetrics(message.result);
      this.setStatus("LIVE COMPLETE", "ready");
      this.disconnectLive();
      return;
    }
    if (message.sequence <= this.liveSequence) throw new Error(`Out-of-order live sequence ${message.sequence}`);
    this.liveSequence = message.sequence;
    this.scene?.updateBodyPoses(message.body_poses);
    this.scene?.setGoal(goalFromObservation(message.observation));
    this.liveEvents.push(...message.events);
    this.renderAgent(
      message.observation,
      message.requested_action,
      message.applied_action,
      message.agent_latency_ms,
      message.requested_action === null || actionsDiffer(message.requested_action, message.applied_action),
    );
    this.renderLiveTelemetry(message);
    this.renderEvents(this.liveEvents, message.simulation_time);
    this.requireElement("#bench-sim-time").textContent = `${message.simulation_time.toFixed(3)} S`;
    this.requireElement("#bench-course").textContent = stringValue(message.task.course_id) ?? "LIVE RUN";
    this.requireElement("#bench-schema").textContent = "EXO.LIVE.V1";
  }

  private renderLiveTelemetry(frame: LiveFrame): void {
    const body = frame.body_poses;
    const speed = numberValue(frame.telemetry.speed_mps) ?? numberValue(frame.observation.speed_mps);
    const rows = [
      ["POSITION", `${(body[0] ?? 0).toFixed(2)}, ${(body[1] ?? 0).toFixed(2)} M`],
      ["PLANAR SPEED", speed === null ? "—" : `${speed.toFixed(2)} M/S`],
      ["SIMULATION", `${frame.simulation_time.toFixed(3)} S`],
      ["SEQUENCE", String(frame.sequence)],
      ...Object.entries(frame.telemetry).slice(0, 2).map(([key, value]) => [key.replaceAll("_", " ").toUpperCase(), formatValue(value)]),
    ];
    this.requireElement("#bench-telemetry").innerHTML = rows.map(([label, value]) => `<div><span>${escapeHtml(label)}</span><strong>${escapeHtml(value)}</strong></div>`).join("");
  }

  private renderLiveMetrics(metrics: Record<string, unknown>): void {
    const passed = metrics.passed === true;
    const result = this.requireElement("#bench-result");
    result.textContent = metrics.passed === undefined ? "COMPLETE" : passed ? "PASS" : "FAIL";
    result.className = `bench-result bench-result--${passed ? "pass" : metrics.passed === false ? "fail" : "idle"}`;
    this.requireElement("#bench-final-metrics").innerHTML = summaryRows(metrics);
  }

  private syncPlaybackUi(): void {
    const play = this.host.querySelector<HTMLButtonElement>("#bench-play");
    if (play) play.textContent = this.playing ? "PAUSE" : "PLAY";
    const timeline = this.host.querySelector<HTMLInputElement>("#bench-timeline");
    if (timeline && this.run) timeline.value = String(this.playhead);
    const current = this.host.querySelector<HTMLElement>("#bench-time-current");
    if (current) current.textContent = `${this.playhead.toFixed(3)} S`;
    const duration = this.host.querySelector<HTMLElement>("#bench-time-duration");
    if (duration) duration.textContent = `${(this.run?.trace.times[this.run.trace.times.length - 1] ?? 0).toFixed(3)} S`;
  }

  private setStatus(label: string, state: "loading" | "ready" | "live" | "warning" | "error"): void {
    this.requireElement("#bench-status-label").textContent = label;
    this.requireElement("#bench-status-dot").className = `bench-status-dot bench-status-dot--${state}`;
  }

  private showError(message: string): void {
    const error = this.requireElement("#bench-error");
    error.textContent = message;
    error.hidden = false;
  }

  private clearError(): void {
    this.requireElement("#bench-error").hidden = true;
  }

  private requireElement(selector: string): HTMLElement {
    const element = this.host.querySelector<HTMLElement>(selector);
    if (!element) throw new Error(`Missing benchmark UI element: ${selector}`);
    return element;
  }

  private requireInput(selector: string): HTMLInputElement {
    const element = this.host.querySelector<HTMLInputElement>(selector);
    if (!element) throw new Error(`Missing benchmark input: ${selector}`);
    return element;
  }
}

export function benchmarkModeTemplate(): string {
  return `
    <div class="bench-shell">
      <header class="bench-topbar">
        <div class="bench-brand"><span class="bench-brand__mark">EXO</span><div><strong>EXO BENCH</strong><small>AUTHORITATIVE REPLAY</small></div></div>
        <nav class="experience-switch" aria-label="Experience mode">
          <button data-experience="lab">ASSISTED LAB</button><button class="experience-switch__active" data-experience="bench">EXO BENCH</button>
        </nav>
        <div class="bench-run-id"><small>COURSE</small><strong id="bench-course">AGENT-001</strong><span id="bench-result" class="bench-result bench-result--idle">NO RUN</span></div>
        <div class="bench-system"><span id="bench-status-dot" class="bench-status-dot bench-status-dot--loading"></span><span id="bench-status-label">STANDBY</span><strong id="bench-sim-time">0.000 S</strong></div>
      </header>

      <main class="bench-workspace">
        <section class="bench-stage" aria-label="Authoritative robot replay">
          <div id="bench-viewport" class="bench-viewport"></div>
          <div class="bench-stage__meta"><span id="bench-schema">EXO.RUN.V2</span><strong>RECORDED BODY STATE</strong><small>Browser physics disabled for benchmark replay</small></div>
          <div class="bench-camera" aria-label="Replay camera">
            <button data-bench-camera="free" title="Free orbit camera">ORBIT</button>
            <button class="bench-icon-button--active" data-bench-camera="follow" title="Follow robot">FOLLOW</button>
            <button data-bench-camera="overhead" title="Overhead camera">TOP</button>
          </div>
        </section>

        <aside class="bench-agent" aria-label="Agent decision inspector">
          <div class="bench-section-head"><span>AGENT</span><strong>SCRIPTED BASELINE</strong></div>
          <section class="bench-agent__observation"><div class="bench-subhead"><span>OBSERVATION</span><small>PUBLIC CONTRACT</small></div><div id="bench-observation" class="bench-kv">${emptyState("Load a replay or connect localhost.")}</div></section>
          <div class="bench-actions">
            <section><div class="bench-subhead"><span>REQUESTED</span></div><div id="bench-requested" class="bench-action">${actionRows(null)}</div></section>
            <section><div class="bench-subhead"><span>APPLIED</span><small id="bench-validation">—</small></div><div id="bench-applied" class="bench-action">${actionRows(null)}</div></section>
          </div>
          <div class="bench-agent__latency"><span>DECISION WALL LATENCY</span><strong id="bench-latency">—</strong><small>RECORDED · DOES NOT ADVANCE SIM TIME</small></div>
        </aside>

        <section class="bench-telemetry-panel" aria-label="Benchmark telemetry">
          <div class="bench-section-head"><span>TELEMETRY</span><strong>RECORDED STATE</strong></div>
          <div id="bench-telemetry" class="bench-telemetry-grid">${emptyState("No state loaded.")}</div>
          <div id="bench-final-metrics" class="bench-final-metrics"></div>
        </section>

        <section class="bench-events-panel" aria-label="Benchmark event timeline">
          <div class="bench-section-head"><span>EVENTS</span><strong>CANONICAL TRACE</strong></div>
          <div id="bench-events" class="bench-events">${emptyState("No events loaded.")}</div>
        </section>

        <aside class="bench-source" aria-label="Replay and live sources">
          <div class="bench-section-head"><span>DATA SOURCE</span><strong>VISUALIZATION ONLY</strong></div>
          <p id="bench-source-note">Load the bundled replay, upload an .exorun, or view a local native run.</p>
          <div class="bench-source__buttons"><button id="bench-sample">LOAD SAMPLE</button><label>UPLOAD .EXORUN<input id="bench-upload" type="file" accept=".exorun,application/zip" /></label></div>
          <div class="bench-live"><input id="bench-live-url" value="http://127.0.0.1:8765/events" aria-label="Local EventSource URL" /><button id="bench-live-toggle">CONNECT</button></div>
          <p id="bench-error" class="bench-error" hidden></p>
          <details><summary>PROVENANCE</summary><div id="bench-provenance" class="bench-provenance"><span>No verified manifest loaded.</span></div></details>
        </aside>
      </main>

      <footer class="bench-transport">
        <div class="bench-transport__buttons"><button id="bench-reset" disabled>RESET</button><button id="bench-play" class="bench-play" disabled>PLAY</button></div>
        <time id="bench-time-current">0.000 S</time><input id="bench-timeline" type="range" min="0" max="0" value="0" step="any" disabled aria-label="Replay timeline" /><time id="bench-time-duration">0.000 S</time>
        <label>SPEED<select id="bench-speed"><option value="0.25">0.25×</option><option value="0.5">0.5×</option><option value="1" selected>1×</option><option value="2">2×</option><option value="4">4×</option></select></label>
      </footer>
    </div>
  `;
}

function closestObservation(observations: AgentObservation[], simulationTime: number): AgentObservation | null {
  let selected: AgentObservation | null = null;
  for (const observation of observations) {
    if (observation.simulation_time_s > simulationTime) break;
    selected = observation;
  }
  return selected;
}

function goalFromObservation(observation: AgentObservation | null): [number, number] | null {
  if (!observation) return null;
  const direct = observation.goal_position_m ?? observation.goal_position ?? observation.target_position;
  if (Array.isArray(direct) && direct.length >= 2 && direct.every((value) => typeof value === "number" && Number.isFinite(value))) {
    return [direct[0] as number, direct[1] as number];
  }
  const task = observation.task;
  if (typeof task === "object" && task !== null && !Array.isArray(task)) {
    const taskRecord = task as Record<string, unknown>;
    const nested = taskRecord.goal_position_m ?? taskRecord.goal_position;
    if (Array.isArray(nested) && nested.length >= 2 && nested.every((value) => typeof value === "number" && Number.isFinite(value))) {
      return [nested[0] as number, nested[1] as number];
    }
    if (typeof nested === "object" && nested !== null && !Array.isArray(nested)) {
      const goal = nested as Record<string, unknown>;
      if (typeof goal.x === "number" && Number.isFinite(goal.x) && typeof goal.y === "number" && Number.isFinite(goal.y)) return [goal.x, goal.y];
    }
  }
  return null;
}

function actionRows(action: AgentVelocityAction | null): string {
  if (!action) return emptyState("No action");
  return [
    ["FORWARD", `${action.forward_mps.toFixed(3)} M/S`],
    ["LATERAL", `${action.lateral_mps.toFixed(3)} M/S`],
    ["YAW RATE", `${action.yaw_rate_rps.toFixed(3)} RAD/S`],
  ].map(([label, value]) => `<div><span>${label}</span><strong>${value}</strong></div>`).join("");
}

function actionsDiffer(requested: AgentVelocityAction, applied: AgentVelocityAction): boolean {
  return requested.forward_mps !== applied.forward_mps
    || requested.lateral_mps !== applied.lateral_mps
    || requested.yaw_rate_rps !== applied.yaw_rate_rps;
}

function summaryRows(value: Record<string, unknown>, exclude: string[] = []): string {
  const rows: Array<[string, string]> = [];
  const visit = (record: Record<string, unknown>, prefix = "") => {
    for (const [key, item] of Object.entries(record)) {
      if (exclude.includes(key) || rows.length >= 10) continue;
      const label = `${prefix}${key}`;
      if (typeof item === "object" && item !== null && !Array.isArray(item)) visit(item as Record<string, unknown>, `${label}.`);
      else rows.push([label.replaceAll("_", " ").toUpperCase(), formatValue(item)]);
    }
  };
  visit(value);
  return rows.map(([label, item]) => `<div><span>${escapeHtml(label)}</span><strong>${escapeHtml(item)}</strong></div>`).join("");
}

function eventTime(event: ReplayEvent): number {
  return numberValue(event.t) ?? numberValue(event.simulation_time) ?? 0;
}

function eventDetail(event: ReplayEvent): string {
  for (const key of ["phase", "reason", "status", "goal_id"]) {
    const value = stringValue(event[key]);
    if (value) return value.replaceAll("_", " ");
  }
  return "";
}

function metric(metrics: Record<string, unknown>, keys: string[], unit: string): string {
  for (const key of keys) {
    const value = metrics[key];
    if (typeof value === "number" && Number.isFinite(value)) return `${value.toFixed(value >= 100 ? 0 : 3)}${unit ? ` ${unit}` : ""}`;
  }
  return "—";
}

function numberValue(value: unknown): number | null {
  return typeof value === "number" && Number.isFinite(value) ? value : null;
}

function stringValue(value: unknown): string | null {
  return typeof value === "string" && value.length > 0 ? value : null;
}

function formatValue(value: unknown): string {
  if (typeof value === "number") return Number.isInteger(value) ? String(value) : value.toFixed(3);
  if (typeof value === "boolean") return value ? "YES" : "NO";
  if (typeof value === "string") return value;
  if (Array.isArray(value)) return value.map((item) => typeof item === "number" ? item.toFixed(3) : String(item)).join(", ");
  if (value === null || value === undefined) return "—";
  return JSON.stringify(value);
}

function emptyState(message: string): string {
  return `<p class="bench-empty">${escapeHtml(message)}</p>`;
}

function escapeHtml(value: string): string {
  return value.replace(/[&<>'"]/gu, (character) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", "'": "&#39;", '"': "&quot;" })[character] ?? character);
}

function errorMessage(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}
