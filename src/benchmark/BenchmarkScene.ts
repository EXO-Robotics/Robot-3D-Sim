import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
import { RoomEnvironment } from "three/examples/jsm/environments/RoomEnvironment.js";
import { VISUAL_LINK_NAMES, visualModelUrl } from "../model/visualContract";
import { HORIZON, STATUS_CYAN } from "../renderer/materials";
import { loadVisualModel, type VisualLinks } from "../renderer/visualBinding";

export type BenchmarkCameraMode = "free" | "follow" | "overhead";

export class BenchmarkScene {
  private readonly scene = new THREE.Scene();
  private readonly camera = new THREE.PerspectiveCamera(38, 1, 0.05, 140);
  private readonly renderer: THREE.WebGLRenderer;
  private readonly controls: OrbitControls;
  private readonly pmrem: THREE.PMREMGenerator;
  private readonly resizeObserver: ResizeObserver;
  private readonly goalMarker: THREE.Group;
  private readonly keyLight: THREE.DirectionalLight;
  private cameraMode: BenchmarkCameraMode = "follow";
  private focus = new THREE.Vector3(0, 0, 0.9);
  private heading = 0;

  static async create(container: HTMLElement): Promise<BenchmarkScene> {
    const visual = await loadVisualModel(visualModelUrl());
    return new BenchmarkScene(container, visual.links);
  }

  private constructor(private readonly container: HTMLElement, private readonly visualLinks: VisualLinks) {
    this.renderer = new THREE.WebGLRenderer({ antialias: true, alpha: false, powerPreference: "high-performance" });
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    this.renderer.setClearColor(HORIZON, 1);
    this.renderer.shadowMap.enabled = true;
    this.renderer.shadowMap.type = THREE.PCFSoftShadowMap;
    this.renderer.outputColorSpace = THREE.SRGBColorSpace;
    this.renderer.toneMapping = THREE.ACESFilmicToneMapping;
    this.renderer.toneMappingExposure = 1.04;
    container.append(this.renderer.domElement);

    this.camera.up.set(0, 0, 1);
    this.camera.position.set(-4.8, -4.4, 2.8);
    this.controls = new OrbitControls(this.camera, this.renderer.domElement);
    this.controls.target.set(0.7, 0, 0.8);
    this.controls.enableDamping = true;
    this.controls.maxPolarAngle = Math.PI * 0.49;
    this.controls.enabled = false;

    this.scene.background = new THREE.Color(HORIZON);
    this.scene.fog = new THREE.Fog(HORIZON, 30, 86);
    this.pmrem = new THREE.PMREMGenerator(this.renderer);
    const environment = new RoomEnvironment();
    this.scene.environment = this.pmrem.fromScene(environment, 0.04).texture;
    environment.dispose();
    this.scene.add(new THREE.HemisphereLight(0xdde2e6, 0x343936, 0.5));
    this.keyLight = new THREE.DirectionalLight(0xfff3e4, 1.5);
    this.keyLight.castShadow = true;
    this.keyLight.shadow.mapSize.set(2048, 2048);
    this.keyLight.shadow.bias = -0.0003;
    this.keyLight.shadow.normalBias = 0.03;
    this.keyLight.shadow.camera.near = 0.5;
    this.keyLight.shadow.camera.far = 36;
    this.keyLight.shadow.camera.left = -8;
    this.keyLight.shadow.camera.right = 8;
    this.keyLight.shadow.camera.top = 8;
    this.keyLight.shadow.camera.bottom = -8;
    this.scene.add(this.keyLight, this.keyLight.target);
    const fill = new THREE.DirectionalLight(0x9eb3c2, 0.32);
    fill.position.set(7, 5, 7);
    this.scene.add(fill);

    this.buildArena();
    for (const name of VISUAL_LINK_NAMES) this.scene.add(this.visualLinks[name]);
    const statusLight = new THREE.PointLight(STATUS_CYAN, 0.55, 2.6, 2);
    this.visualLinks.torso.add(statusLight);
    statusLight.position.set(0.16, 0, 0.34);
    this.goalMarker = this.buildGoalMarker();
    this.scene.add(this.goalMarker);

    this.resizeObserver = new ResizeObserver(() => this.resize());
    this.resizeObserver.observe(container);
    this.resize();
  }

  setCameraMode(mode: BenchmarkCameraMode): void {
    this.cameraMode = mode;
    this.controls.enabled = mode === "free";
  }

  setGoal(position: readonly [number, number] | null): void {
    this.goalMarker.visible = position !== null;
    if (position) this.goalMarker.position.set(position[0], position[1], 0.015);
  }

  updateBodyPoses(poses: ArrayLike<number>): void {
    if (poses.length !== VISUAL_LINK_NAMES.length * 7) throw new Error("Benchmark body pose frame must contain 91 values");
    for (let linkIndex = 0; linkIndex < VISUAL_LINK_NAMES.length; linkIndex += 1) {
      const offset = linkIndex * 7;
      const link = this.visualLinks[VISUAL_LINK_NAMES[linkIndex] ?? "pelvis"];
      link.position.set(poses[offset] ?? 0, poses[offset + 1] ?? 0, poses[offset + 2] ?? 0);
      link.quaternion.set(poses[offset + 4] ?? 0, poses[offset + 5] ?? 0, poses[offset + 6] ?? 0, poses[offset + 3] ?? 1);
      link.quaternion.normalize();
    }
    const pelvis = this.visualLinks.pelvis;
    this.focus.copy(pelvis.position);
    const quaternion = pelvis.quaternion;
    this.heading = Math.atan2(
      2 * (quaternion.w * quaternion.z + quaternion.x * quaternion.y),
      1 - 2 * (quaternion.y * quaternion.y + quaternion.z * quaternion.z),
    );
  }

  render(): void {
    this.updateCamera();
    this.keyLight.position.set(this.focus.x - 6, this.focus.y - 8, this.focus.z + 14);
    this.keyLight.target.position.set(this.focus.x, this.focus.y, 0.2);
    this.keyLight.target.updateMatrixWorld();
    this.controls.update();
    this.renderer.render(this.scene, this.camera);
  }

  dispose(): void {
    this.resizeObserver.disconnect();
    this.controls.dispose();
    this.pmrem.dispose();
    this.scene.traverse((object) => {
      const mesh = object as THREE.Mesh;
      mesh.geometry?.dispose();
      const materials = mesh.material ? (Array.isArray(mesh.material) ? mesh.material : [mesh.material]) : [];
      materials.forEach((material) => material.dispose());
    });
    this.renderer.dispose();
    this.container.replaceChildren();
  }

  private buildArena(): void {
    const ground = new THREE.Mesh(
      new THREE.BoxGeometry(120, 120, 0.05),
      new THREE.MeshStandardMaterial({ color: 0x292d2b, roughness: 0.95, metalness: 0.02 }),
    );
    ground.position.z = -0.035;
    ground.receiveShadow = true;
    this.scene.add(ground, createGrid(1, 55, 0x48514c, 0.09), createGrid(5, 55, 0xa4aea7, 0.22));
    const origin = new THREE.Mesh(
      new THREE.RingGeometry(1.46, 1.54, 64),
      new THREE.MeshStandardMaterial({ color: 0xc8cec8, roughness: 0.74, side: THREE.DoubleSide }),
    );
    origin.position.z = 0.01;
    this.scene.add(origin);
  }

  private buildGoalMarker(): THREE.Group {
    const group = new THREE.Group();
    const ring = new THREE.Mesh(
      new THREE.RingGeometry(0.44, 0.5, 64),
      new THREE.MeshBasicMaterial({ color: STATUS_CYAN, transparent: true, opacity: 0.82, side: THREE.DoubleSide }),
    );
    const inner = new THREE.Mesh(
      new THREE.CircleGeometry(0.44, 64),
      new THREE.MeshBasicMaterial({ color: STATUS_CYAN, transparent: true, opacity: 0.055, side: THREE.DoubleSide }),
    );
    const stem = new THREE.Mesh(
      new THREE.CylinderGeometry(0.006, 0.006, 1.35, 8),
      new THREE.MeshBasicMaterial({ color: STATUS_CYAN, transparent: true, opacity: 0.35 }),
    );
    stem.rotation.x = Math.PI / 2;
    stem.position.z = 0.675;
    group.add(inner, ring, stem);
    group.visible = false;
    return group;
  }

  private updateCamera(): void {
    const forwardX = Math.cos(this.heading);
    const forwardY = Math.sin(this.heading);
    const target = new THREE.Vector3(this.focus.x + forwardX * 0.9, this.focus.y + forwardY * 0.9, 0.82);
    if (this.cameraMode === "follow") {
      this.camera.position.lerp(new THREE.Vector3(
        this.focus.x - forwardX * 4.4 - forwardY * 2.5,
        this.focus.y - forwardY * 4.4 + forwardX * 2.5,
        2.65,
      ), 0.08);
      this.controls.target.lerp(target, 0.1);
      this.camera.lookAt(this.controls.target);
    } else if (this.cameraMode === "overhead") {
      this.camera.position.lerp(new THREE.Vector3(this.focus.x, this.focus.y, 10.5), 0.1);
      this.controls.target.lerp(new THREE.Vector3(this.focus.x, this.focus.y, 0), 0.1);
      this.camera.lookAt(this.controls.target);
    }
  }

  private resize(): void {
    const width = Math.max(1, this.container.clientWidth);
    const height = Math.max(1, this.container.clientHeight);
    this.camera.aspect = width / height;
    this.camera.updateProjectionMatrix();
    this.renderer.setSize(width, height, false);
  }
}

function createGrid(step: number, extent: number, color: number, opacity: number): THREE.LineSegments {
  const positions: number[] = [];
  for (let coordinate = -extent; coordinate <= extent; coordinate += step) {
    positions.push(-extent, coordinate, 0.004, extent, coordinate, 0.004);
    positions.push(coordinate, -extent, 0.004, coordinate, extent, 0.004);
  }
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute("position", new THREE.Float32BufferAttribute(positions, 3));
  return new THREE.LineSegments(geometry, new THREE.LineBasicMaterial({ color, transparent: true, opacity }));
}
