import { BASELINE_CONTROLLER } from "./control/baseline";
import { PHYSICS_MODEL_XML } from "./model/physicsBaseline";
import { ControllerHost } from "./control/ControllerHost";
import { FieldSession } from "./events/FieldSession";
import { MujocoEngine } from "./physics/MujocoEngine";
import { RobotScene, type CameraMode } from "./renderer/RobotScene";
import type { SensorFrame } from "./sim/types";
import { TelemetryBuffer } from "./telemetry/TelemetryBuffer";
import { BenchmarkViewer } from "./benchmark/BenchmarkViewer";

const CONTROL_DT = 0.01;
const TELEMETRY_DT = 0.04;

export class App {
  private engine: MujocoEngine | null = null;
  private scene: RobotScene | null = null;
  private readonly controller = new ControllerHost();
  private readonly session = new FieldSession();
  private readonly telemetry = new TelemetryBuffer(750);
  private previousFrameTime = performance.now();
  private accumulator = 0;
  private controlAccumulator = 0;
  private telemetryAccumulator = 0;
  private simulationSpeed = 1;
  private countdownToken = 0;
  private lastUiUpdate = 0;
  private powerEnabled = true;
  private animationId = 0;
  private readonly heldControls = new Set<string>();
  private readonly benchmark: BenchmarkViewer;

  constructor(private readonly root: HTMLElement) {
    this.root.innerHTML = this.template();
    this.benchmark = new BenchmarkViewer(this.requireElement("#benchmark-mode"));
    this.bindStaticUi();
  }

  async start(): Promise<void> {
    try {
      this.setBoot("Loading 10 MB physics core", 28);
      this.engine = await MujocoEngine.create(PHYSICS_MODEL_XML);
      this.setBoot("Binding visual model", 62);
      this.scene = await RobotScene.create(this.requireElement("#viewport"), this.engine);
      this.controller.compile(BASELINE_CONTROLLER);
      this.session.ready();
      this.setBoot("Sensors online", 100);
      this.renderModelStats();
      window.setTimeout(() => this.requireElement("#boot").classList.add("boot--hidden"), 420);
      this.previousFrameTime = performance.now();
      this.animationId = requestAnimationFrame((time) => this.animate(time));
    } catch (error) {
      this.fail(`Simulator initialization failed: ${String(error)}`);
    }
  }

  dispose(): void {
    cancelAnimationFrame(this.animationId);
    this.controller.dispose();
    this.scene?.dispose();
    this.engine?.dispose();
    this.benchmark.dispose();
  }

  private animate(now: number): void {
    const engine = this.engine;
    const scene = this.scene;
    if (!engine || !scene) return;
    const wallDt = Math.min(0.05, (now - this.previousFrameTime) / 1000);
    this.previousFrameTime = now;
    const active = this.session.phase === "running" || (!this.powerEnabled && this.session.phase === "ready");
    if (active) this.accumulator = Math.min(0.05, this.accumulator + wallDt * this.simulationSpeed);

    let steps = 0;
    while (active && this.accumulator >= engine.timestep && steps < 25) {
      const frame = engine.sensors();
      this.controlAccumulator += engine.timestep;
      this.telemetryAccumulator += engine.timestep;
      if (this.controlAccumulator >= CONTROL_DT) {
        this.controller.step(this.copyFrame(frame), CONTROL_DT);
        engine.applyTargets(this.controller.targets);
        const command = this.fieldCommand();
        engine.setFieldDrive(command.drive, command.turn);
        this.controlAccumulator %= CONTROL_DT;
      }
      engine.step();
      const updated = engine.sensors();
      if (this.session.phase === "running") {
        this.session.update(updated);
        if (engine.fallen) this.session.fall(updated, engine.energy);
      }
      if (this.telemetryAccumulator >= TELEMETRY_DT) {
        this.captureTelemetry(updated);
        this.telemetryAccumulator %= TELEMETRY_DT;
      }
      this.accumulator -= engine.timestep;
      steps += 1;
    }

    const frame = engine.sensors();
    scene.update(frame);
    if (now - this.lastUiUpdate > 80) {
      this.renderTelemetry(frame);
      this.renderRunState();
      this.lastUiUpdate = now;
    }
    this.animationId = requestAnimationFrame((time) => this.animate(time));
  }

  private bindStaticUi(): void {
    this.root.querySelectorAll<HTMLButtonElement>("[data-experience]").forEach((button) => {
      button.addEventListener("click", () => void this.selectExperience(button.dataset.experience === "bench" ? "bench" : "lab"));
    });
    this.requireElement("#run-button").addEventListener("click", () => this.toggleRun());
    this.requireElement("#reset-button").addEventListener("click", () => this.reset());
    this.requireElement("#power-button").addEventListener("click", () => this.togglePower());
    this.bindDriveControls();
    this.requireElement("#debug-button").addEventListener("click", (event) => {
      const button = event.currentTarget as HTMLButtonElement;
      const active = button.getAttribute("aria-pressed") !== "true";
      button.setAttribute("aria-pressed", String(active));
      button.classList.toggle("tool-button--active", active);
      this.scene?.setDebug(active);
    });
    this.root.querySelectorAll<HTMLButtonElement>("[data-camera]").forEach((button) => {
      button.addEventListener("click", () => {
        this.root.querySelectorAll("[data-camera]").forEach((item) => item.classList.remove("tool-button--active"));
        button.classList.add("tool-button--active");
        this.scene?.setCameraMode(button.dataset.camera as CameraMode);
      });
    });
    this.root.querySelectorAll<HTMLButtonElement>("[data-panel]").forEach((button) => {
      button.addEventListener("click", () => this.selectPanel(button.dataset.panel ?? "lab"));
    });
    this.root.querySelectorAll<HTMLButtonElement>("[data-speed]").forEach((button) => {
      button.addEventListener("click", () => {
        this.simulationSpeed = Number(button.dataset.speed);
        this.root.querySelectorAll("[data-speed]").forEach((item) => item.classList.remove("segment__button--active"));
        button.classList.add("segment__button--active");
      });
    });
    this.requireInput("#strength").addEventListener("input", (event) => {
      const value = Number((event.currentTarget as HTMLInputElement).value);
      this.engine?.setActuatorStrength(value);
      this.requireElement("#strength-value").textContent = `${Math.round(value * 100)}%`;
    });
    this.requireInput("#friction").addEventListener("input", (event) => {
      const value = Number((event.currentTarget as HTMLInputElement).value);
      this.engine?.setGroundFriction(value);
      this.requireElement("#friction-value").textContent = value.toFixed(2);
    });
    this.requireElement("#compile-button").addEventListener("click", () => {
      const source = this.requireElement("#controller-source") as HTMLTextAreaElement;
      this.requireElement("#console-output").textContent = "Compiling controller…";
      this.controller.compile(source.value);
    });
    this.requireElement("#restore-button").addEventListener("click", () => {
      (this.requireElement("#controller-source") as HTMLTextAreaElement).value = BASELINE_CONTROLLER;
      this.controller.compile(BASELINE_CONTROLLER);
    });
    this.controller.addEventListener("compiled", () => {
      this.requireElement("#console-output").textContent = "Controller compiled. Sensor bridge at 100 Hz.";
      this.requireElement("#console-output").classList.remove("console--error");
    });
    this.controller.addEventListener("error", (event) => {
      const detail = (event as CustomEvent<string>).detail;
      this.requireElement("#console-output").textContent = detail;
      this.requireElement("#console-output").classList.add("console--error");
    });
    document.addEventListener("visibilitychange", () => {
      if (document.hidden && this.session.phase === "running") this.session.pause();
    });
    window.addEventListener("beforeunload", () => this.dispose(), { once: true });
  }

  private async toggleRun(): Promise<void> {
    if (!this.engine) return;
    if (this.session.phase === "ready") {
      this.session.countdown();
      const token = ++this.countdownToken;
      const countdown = this.requireElement("#countdown");
      countdown.classList.add("countdown--visible");
      for (const label of ["3", "2", "1", "GO"]) {
        if (token !== this.countdownToken) return;
        countdown.textContent = label;
        await new Promise((resolve) => window.setTimeout(resolve, label === "GO" ? 420 : 650));
      }
      countdown.classList.remove("countdown--visible");
      if (token === this.countdownToken) this.session.start(this.engine.data.time, this.engine.sensors());
    } else if (this.session.phase === "running") {
      this.session.pause();
    } else if (this.session.phase === "paused") {
      this.session.resume(this.engine.data.time, this.engine.sensors());
    } else if (this.session.phase === "fallen") {
      this.reset();
    }
  }

  private reset(): void {
    if (!this.engine) return;
    this.countdownToken += 1;
    this.requireElement("#countdown").classList.remove("countdown--visible");
    this.engine.reset();
    this.engine.setActuationEnabled(this.powerEnabled);
    this.telemetry.clear();
    this.accumulator = 0;
    this.controlAccumulator = 0;
    this.telemetryAccumulator = 0;
    this.heldControls.clear();
    this.syncDriveKeys();
    this.session.reset();
  }

  private togglePower(): void {
    if (!this.engine) return;
    this.powerEnabled = !this.powerEnabled;
    this.engine.setActuationEnabled(this.powerEnabled);
    const button = this.requireElement("#power-button");
    button.textContent = this.powerEnabled ? "POWER ON" : "POWER OFF";
    button.classList.toggle("power--off", !this.powerEnabled);
  }

  private bindDriveControls(): void {
    const keyMap: Record<string, string> = {
      w: "forward", ArrowUp: "forward",
      s: "reverse", ArrowDown: "reverse",
      a: "left", ArrowLeft: "left",
      d: "right", ArrowRight: "right",
    };
    window.addEventListener("keydown", (event) => {
      if (event.target instanceof HTMLTextAreaElement || event.target instanceof HTMLInputElement) return;
      const control = keyMap[event.key] ?? keyMap[event.key.toLowerCase()];
      if (!control) return;
      event.preventDefault();
      this.heldControls.add(control);
      this.syncDriveKeys();
    });
    window.addEventListener("keyup", (event) => {
      const control = keyMap[event.key] ?? keyMap[event.key.toLowerCase()];
      if (!control) return;
      this.heldControls.delete(control);
      this.syncDriveKeys();
    });
    this.root.querySelectorAll<HTMLButtonElement>("[data-drive]").forEach((button) => {
      const control = button.dataset.drive ?? "";
      const press = (event: PointerEvent) => {
        event.preventDefault();
        button.setPointerCapture(event.pointerId);
        this.heldControls.add(control);
        this.syncDriveKeys();
      };
      const release = () => {
        this.heldControls.delete(control);
        this.syncDriveKeys();
      };
      button.addEventListener("pointerdown", press);
      button.addEventListener("pointerup", release);
      button.addEventListener("pointercancel", release);
    });
  }

  private fieldCommand(): { drive: number; turn: number } {
    if (this.heldControls.size === 0) {
      return { drive: this.controller.drive, turn: this.controller.turn };
    }
    const drive = (this.heldControls.has("forward") ? 1.35 : 0) - (this.heldControls.has("reverse") ? 0.85 : 0);
    const turn = (this.heldControls.has("left") ? 1.15 : 0) - (this.heldControls.has("right") ? 1.15 : 0);
    return { drive, turn };
  }

  private syncDriveKeys(): void {
    this.root.querySelectorAll<HTMLElement>("[data-drive]").forEach((button) => {
      button.classList.toggle("drive-key--active", this.heldControls.has(button.dataset.drive ?? ""));
    });
  }

  private captureTelemetry(frame: SensorFrame): void {
    if (!this.engine) return;
    this.telemetry.push({
      time: frame.time,
      distance: this.session.distance,
      speed: Math.max(0, frame.velocity),
      pitch: frame.imu.pitch,
      energy: this.engine.energy,
      leftForce: frame.feet.left,
      rightForce: frame.feet.right,
      drive: this.engine.telemetryDrive(),
    });
  }

  private renderTelemetry(frame: SensorFrame): void {
    if (!this.engine) return;
    this.requireElement("#metric-time").textContent = this.session.elapsed.toFixed(3);
    this.requireElement("#metric-speed").textContent = Math.max(0, frame.velocity).toFixed(2);
    this.requireElement("#metric-distance").textContent = this.session.distance.toFixed(2);
    this.requireElement("#metric-energy").textContent = (this.engine.energy / 1000).toFixed(2);
    this.requireElement("#metric-pitch").textContent = `${(frame.imu.pitch * 57.2958).toFixed(1)}°`;
    this.requireElement("#metric-force").textContent = `${Math.round(frame.feet.left + frame.feet.right)} N`;
    this.requireElement("#coord-x").textContent = frame.position.toFixed(1);
    this.requireElement("#coord-y").textContent = frame.lateral.toFixed(1);
    const heading = ((frame.yaw * 57.2958) % 360 + 360) % 360;
    this.requireElement("#metric-heading").textContent = `${heading.toFixed(0)}°`;
    const drive = this.engine.telemetryDrive();
    this.requireElement("#drive-fill").setAttribute("style", `--drive:${(drive * 100).toFixed(1)}%`);
    this.requireElement("#speed-line").setAttribute("points", this.telemetry.points("speed", 360, 74));
    this.requireElement("#pitch-line").setAttribute("points", this.telemetry.points("pitch", 360, 74));
  }

  private renderRunState(): void {
    const phase = this.session.phase;
    const runButton = this.requireElement("#run-button");
    const labels: Partial<Record<typeof phase, string>> = {
      ready: "ENTER FIELD",
      countdown: "ARMED",
      running: "PAUSE",
      paused: "RESUME",
      fallen: "RESET ROBOT",
    };
    runButton.textContent = labels[phase] ?? "ENTER FIELD";
    this.requireElement("#phase-label").textContent = phase.toUpperCase();
    this.requireElement("#phase-dot").className = `status-dot status-dot--${phase}`;
    const terminal = phase === "fallen";
    this.requireElement("#result-overlay").classList.toggle("result--visible", terminal);
    if (this.session.result) {
      this.requireElement("#result-title").textContent = "FIELD SESSION ENDED";
      this.requireElement("#result-score").textContent = this.session.result.score.toLocaleString("en-US");
      this.requireElement("#result-detail").textContent = `${this.session.result.distance.toFixed(1)} m explored · ${this.session.result.time.toFixed(2)} s · ${(this.session.result.energy / 1000).toFixed(2)} kJ`;
    }
  }

  private renderModelStats(): void {
    if (!this.engine) return;
    this.requireElement("#model-mass").textContent = "49.8 KG";
    this.requireElement("#model-dof").textContent = `${this.engine.model.nv} DOF`;
    this.requireElement("#model-actuators").textContent = `${this.engine.model.nactuator} ACTUATORS`;
  }

  private selectPanel(panel: string): void {
    this.root.querySelectorAll("[data-panel]").forEach((item) => item.classList.toggle("panel-tab--active", (item as HTMLElement).dataset.panel === panel));
    this.requireElement("#lab-panel").classList.toggle("inspector-panel--active", panel === "lab");
    this.requireElement("#code-panel").classList.toggle("inspector-panel--active", panel === "code");
  }

  private async selectExperience(experience: "lab" | "bench"): Promise<void> {
    if (experience === "bench" && this.session.phase === "running") this.session.pause();
    this.requireElement("#assisted-mode").classList.toggle("experience--active", experience === "lab");
    this.benchmark.setVisible(experience === "bench");
    this.root.querySelectorAll<HTMLElement>("[data-experience]").forEach((button) => {
      button.classList.toggle("experience-switch__active", button.dataset.experience === experience);
    });
    if (experience === "bench") await this.benchmark.start();
  }

  private setBoot(label: string, progress: number): void {
    this.requireElement("#boot-label").textContent = label;
    this.requireElement("#boot-progress").setAttribute("style", `--progress:${progress}%`);
  }

  private fail(message: string): void {
    this.requireElement("#boot-label").textContent = message;
    this.requireElement("#boot").classList.add("boot--error");
  }

  private copyFrame(frame: SensorFrame): SensorFrame {
    return structuredClone(frame);
  }

  private requireElement(selector: string): HTMLElement {
    const element = this.root.querySelector<HTMLElement>(selector);
    if (!element) throw new Error(`Missing required element: ${selector}`);
    return element;
  }

  private requireInput(selector: string): HTMLInputElement {
    const input = this.root.querySelector<HTMLInputElement>(selector);
    if (!input) throw new Error(`Missing required input: ${selector}`);
    return input;
  }

  private template(): string {
    return `
      <div id="assisted-mode" class="experience experience--active">
      <div class="app-shell">
        <header class="topbar">
          <div class="brand"><span class="brand__mark">H1</span><span>HUMANOID ROBOT GAMES</span></div>
          <div class="experience-title"><nav class="experience-switch" aria-label="Experience mode"><button class="experience-switch__active" data-experience="lab">ASSISTED LAB</button><button data-experience="bench">EXO BENCH</button></nav><div class="event-title"><span>ASSISTED LAB</span><strong>OPEN FIELD / TRAINER</strong></div></div>
          <div class="system-state"><span id="phase-dot" class="status-dot status-dot--loading"></span><span id="phase-label">LOADING</span><small>MUJOCO 3.12 / WASM</small></div>
        </header>

        <main class="workspace">
          <section class="simulation" aria-label="3D humanoid simulation">
            <div id="viewport" class="viewport"></div>
            <div class="viewport__hud viewport__hud--top">
              <div><small>ROBOT</small><strong>AION // H1-S</strong></div>
              <div class="model-tags"><span id="model-mass">— KG</span><span id="model-dof">— DOF</span><span id="model-actuators">— ACTUATORS</span></div>
            </div>
            <div class="viewport__hud viewport__hud--bottom">
              <div class="field-position"><span>WORLD X <b id="coord-x">0.0</b></span><span>WORLD Y <b id="coord-y">0.0</b></span><span>W/A/S/D TO STEER</span></div>
              <div class="field-bearing"><i></i><span>NAV GRID / 5 M</span></div>
            </div>
            <div class="toolrail" aria-label="View controls">
              <button class="tool-button" data-camera="broadcast" title="Free camera">B</button>
              <button class="tool-button tool-button--active" data-camera="follow" title="Follow camera">F</button>
              <button class="tool-button" data-camera="overhead" title="Overhead camera">T</button>
              <span></span>
              <button id="debug-button" class="tool-button" aria-pressed="false" title="Physics debug">Δ</button>
            </div>
            <div class="drive-pad" aria-label="Robot drive controls">
              <button data-drive="forward" aria-label="Drive forward">W</button>
              <button data-drive="left" aria-label="Turn left">A</button>
              <button data-drive="reverse" aria-label="Drive backward">S</button>
              <button data-drive="right" aria-label="Turn right">D</button>
            </div>
            <div id="countdown" class="countdown">3</div>
            <div id="result-overlay" class="result">
              <small id="result-title">FIELD SESSION ENDED</small>
              <strong id="result-score">0</strong>
              <span>SCORE</span>
              <p id="result-detail">0.0 m explored · 0.00 s · 0.00 kJ</p>
            </div>
          </section>

          <aside class="inspector">
            <nav class="panel-tabs" aria-label="Inspector panels">
              <button class="panel-tab panel-tab--active" data-panel="lab">ASSISTED LAB</button>
              <button class="panel-tab" data-panel="code">CONTROLLER</button>
            </nav>
            <div id="lab-panel" class="inspector-panel inspector-panel--active">
              <div class="mode-note"><span>OPEN-FIELD TRAINER</span><p>Explore freely in X/Y with assisted balance, steering, collision obstacles, and live contact physics.</p></div>
              <section class="control-section">
                <div class="section-label"><span>ACTUATOR STRENGTH</span><output id="strength-value">100%</output></div>
                <input id="strength" type="range" min="0.35" max="1.5" value="1" step="0.05" />
                <div class="range-label"><span>35%</span><span>150%</span></div>
              </section>
              <section class="control-section">
                <div class="section-label"><span>FOOT FRICTION</span><output id="friction-value">1.05</output></div>
                <input id="friction" type="range" min="0.2" max="1.6" value="1.05" step="0.05" />
                <div class="range-label"><span>ICE</span><span>TERRAIN</span></div>
              </section>
              <section class="control-section">
                <div class="section-label"><span>SIMULATION RATE</span><output>FIXED 2 MS</output></div>
                <div class="segment"><button data-speed="0.25">¼×</button><button data-speed="0.5">½×</button><button class="segment__button--active" data-speed="1">1×</button></div>
              </section>
              <section class="sensor-readout">
                <div><span>HEADING</span><strong id="metric-heading">0°</strong></div>
                <div><span>GROUND FORCE</span><strong id="metric-force">0 N</strong></div>
                <div><span>IMU PITCH</span><strong id="metric-pitch">0.0°</strong></div>
                <div class="drive"><span>ACTUATOR LOAD</span><i id="drive-fill"></i></div>
              </section>
            </div>
            <div id="code-panel" class="inspector-panel inspector-panel--code">
              <div class="editor-head"><span>control.js</span><small>100 HZ</small></div>
              <textarea id="controller-source" class="code-editor" spellcheck="false" aria-label="JavaScript robot controller">${BASELINE_CONTROLLER}</textarea>
              <div id="console-output" class="console">Baseline controller ready.</div>
              <div class="editor-actions"><button id="restore-button" class="button button--quiet">RESTORE</button><button id="compile-button" class="button button--accent">COMPILE</button></div>
            </div>
          </aside>
        </main>

        <section class="telemetry" aria-label="Field telemetry">
          <div class="metrics">
            <div><span>TIME / S</span><strong id="metric-time">0.000</strong></div>
            <div><span>SPEED / M·S⁻¹</span><strong id="metric-speed">0.00</strong></div>
            <div><span>EXPLORED / M</span><strong id="metric-distance">0.00</strong></div>
            <div><span>ENERGY / KJ</span><strong id="metric-energy">0.00</strong></div>
          </div>
          <div class="chart"><div class="chart__label"><span>VELOCITY</span><small>LIVE</small></div><svg viewBox="0 0 360 74" preserveAspectRatio="none"><path d="M0 18H360M0 37H360M0 56H360"/><polyline id="speed-line" points=""/></svg></div>
          <div class="chart"><div class="chart__label"><span>BODY PITCH</span><small>RAD</small></div><svg viewBox="0 0 360 74" preserveAspectRatio="none"><path d="M0 18H360M0 37H360M0 56H360"/><polyline id="pitch-line" points=""/></svg></div>
          <div class="run-controls"><button id="power-button" class="button button--power">POWER ON</button><button id="reset-button" class="button button--quiet">RESET</button><button id="run-button" class="button button--accent">ENTER FIELD</button></div>
        </section>
      </div>
      <div id="boot" class="boot"><div class="boot__brand">H1</div><p id="boot-label">Initializing robotics lab</p><div class="boot__track"><i id="boot-progress"></i></div><small>PHYSICS EXECUTES LOCALLY IN YOUR BROWSER</small></div>
      </div>
      <div id="benchmark-mode" class="experience"></div>
    `;
  }
}
