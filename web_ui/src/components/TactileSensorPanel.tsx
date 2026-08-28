import { useEffect, useRef, useState } from "react";
import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
import { STLLoader } from "three/examples/jsm/loaders/STLLoader.js";

import type { TactileSample } from "../ros/types";

interface TactileSensorPanelProps {
  samples?: Array<TactileSample | null>;
  live?: boolean;
}

interface TactileVisual {
  update: (sample: TactileSample | null) => void;
}

const FINGERS = ["Thumb", "Index", "Middle", "Ring", "Pinky"];
const VALUES = [
  ["x", "x", "mm"],
  ["y", "y", "mm"],
  ["fx", "Fx", "N"],
  ["fy", "Fy", "N"],
  ["fz", "Fz", "N"],
] as const;

const SENSOR_ORIGIN_FROM_TIP_BOTTOM_M = 0.0;
const SENSOR_X_SIGN = 1;
const SENSOR_Y_SIGN = 1;

function clamp(value: number, minimum: number, maximum: number) {
  return Math.min(maximum, Math.max(minimum, value));
}

function assetUrl(path: string): string {
  const base = import.meta.env.BASE_URL.endsWith("/")
    ? import.meta.env.BASE_URL
    : `${import.meta.env.BASE_URL}/`;
  return `${base}${path.replace(/^\//, "")}`;
}

function TactileTipCard({ finger, index, sample }: {
  finger: string;
  index: number;
  sample: TactileSample | null;
}) {
  const mountRef = useRef<HTMLDivElement>(null);
  const sampleRef = useRef(sample);
  const visualRef = useRef<TactileVisual | null>(null);
  const [modelState, setModelState] = useState<"loading" | "ready" | "error">("loading");

  sampleRef.current = sample;

  useEffect(() => {
    visualRef.current?.update(sample);
  }, [sample]);

  useEffect(() => {
    const mount = mountRef.current;
    if (!mount) return;

    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0xf5f8f9);
    const camera = new THREE.PerspectiveCamera(34, 1, 0.001, 2);
    camera.position.set(0.045, 0.07, 0.035);
    camera.up.set(0, 0, 1);

    const renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.outputColorSpace = THREE.SRGBColorSpace;
    mount.appendChild(renderer.domElement);

    scene.add(new THREE.HemisphereLight(0xffffff, 0x75818a, 2.2));
    const keyLight = new THREE.DirectionalLight(0xffffff, 2.8);
    keyLight.position.set(0.04, 0.08, 0.08);
    scene.add(keyLight);

    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enablePan = false;
    controls.minDistance = 0.04;
    controls.maxDistance = 0.18;

    const render = () => renderer.render(scene, camera);
    controls.addEventListener("change", render);
    const resize = () => {
      const width = Math.max(1, mount.clientWidth);
      const height = Math.max(1, mount.clientHeight);
      camera.aspect = width / height;
      camera.updateProjectionMatrix();
      renderer.setSize(width, height, false);
      render();
    };
    const observer = new ResizeObserver(resize);
    observer.observe(mount);
    resize();

    new STLLoader().load(
      assetUrl("robot/dg5f_s_description/meshes/dg5fs_right/tactile_tip.STL"),
      (geometry) => {
        geometry.computeVertexNormals();
        geometry.computeBoundingBox();
        const bounds = geometry.boundingBox!;
        const mesh = new THREE.Mesh(
          geometry,
          new THREE.MeshStandardMaterial({
            color: 0xe9eef0,
            roughness: 0.68,
            metalness: 0.04,
            side: THREE.DoubleSide,
          }),
        );
        mesh.rotation.y = -Math.PI / 2;
        mesh.updateMatrixWorld(true);
        mesh.position.sub(new THREE.Box3().setFromObject(mesh).getCenter(new THREE.Vector3()));
        scene.add(mesh);
        mesh.updateMatrixWorld(true);

        const raycaster = new THREE.Raycaster();
        const surfacePoint = (xMm: number, yMm: number) => {
          const localX = clamp(
            bounds.min.x + SENSOR_ORIGIN_FROM_TIP_BOTTOM_M + SENSOR_Y_SIGN * yMm * 0.001,
            bounds.min.x + 0.0002,
            bounds.max.x - 0.0002,
          );
          const localZ = clamp(
            SENSOR_X_SIGN * xMm * 0.001,
            bounds.min.z + 0.0002,
            bounds.max.z - 0.0002,
          );
          const rayOrigin = mesh.localToWorld(new THREE.Vector3(localX, bounds.max.y + 0.02, localZ));
          const rayDirection = new THREE.Vector3(0, -1, 0).transformDirection(mesh.matrixWorld);
          raycaster.set(rayOrigin, rayDirection);
          const hit = raycaster.intersectObject(mesh, false)[0];
          const point = hit
            ? mesh.worldToLocal(hit.point.clone())
            : new THREE.Vector3(localX, bounds.max.y, localZ);
          const normal = hit?.face?.normal.clone().normalize()
            ?? new THREE.Vector3(0, 1, 0);
          if (normal.y < 0) normal.negate();
          point.addScaledVector(normal, 0.00045);
          return { point, normal };
        };

        const origin = surfacePoint(0, 0);
        const originMarker = new THREE.Mesh(
          new THREE.SphereGeometry(0.00065, 12, 8),
          new THREE.MeshBasicMaterial({ color: 0x71838c }),
        );
        originMarker.position.copy(origin.point);
        mesh.add(originMarker);
        mesh.add(new THREE.ArrowHelper(new THREE.Vector3(0, 0, 1), origin.point, 0.007, 0xc84b42, 0.0015, 0.0008));
        mesh.add(new THREE.ArrowHelper(new THREE.Vector3(1, 0, 0), origin.point, 0.009, 0x0b8f8f, 0.0015, 0.0008));

        const contactMarker = new THREE.Mesh(
          new THREE.SphereGeometry(0.00115, 16, 10),
          new THREE.MeshBasicMaterial({ color: 0x08a0a0 }),
        );
        contactMarker.renderOrder = 3;
        mesh.add(contactMarker);

        const forceArrows = [
          new THREE.ArrowHelper(new THREE.Vector3(0, 0, 1), origin.point, 0.005, 0xc84b42, 0.0026, 0.0014),
          new THREE.ArrowHelper(new THREE.Vector3(1, 0, 0), origin.point, 0.005, 0x0b8f8f, 0.0026, 0.0014),
          new THREE.ArrowHelper(new THREE.Vector3(0, 1, 0), origin.point, 0.005, 0x5965bd, 0.0026, 0.0014),
        ];
        forceArrows.forEach((arrow) => mesh.add(arrow));

        const update = (next: TactileSample | null) => {
          contactMarker.visible = next !== null;
          forceArrows.forEach((arrow) => { arrow.visible = false; });
          if (next) {
            const surface = surfacePoint(next.x, next.y);
            contactMarker.position.copy(surface.point);
            const forces = [next.fx, next.fy, next.fz];
            const tangentX = new THREE.Vector3(0, 0, 1).projectOnPlane(surface.normal).normalize();
            const tangentY = new THREE.Vector3(1, 0, 0).projectOnPlane(surface.normal).normalize();
            const axes = [
              tangentX,
              tangentY,
              surface.normal.clone(),
            ];
            forceArrows.forEach((arrow, axis) => {
              const force = forces[axis];
              if (Math.abs(force) <= 0.001) return;
              const length = clamp(Math.abs(force) * 0.006, 0.002, 0.018);
              const direction = axes[axis].multiplyScalar(-Math.sign(force));
              arrow.position.copy(surface.point).addScaledVector(direction, -length);
              arrow.setDirection(direction);
              arrow.setLength(length, 0.0026, 0.0014);
              arrow.visible = true;
            });
          }
          render();
        };

        visualRef.current = { update };
        update(sampleRef.current);
        controls.target.set(0, 0, 0);
        controls.update();
        setModelState("ready");
        render();
      },
      undefined,
      () => setModelState("error"),
    );

    return () => {
      visualRef.current = null;
      observer.disconnect();
      controls.removeEventListener("change", render);
      controls.dispose();
      scene.traverse((object) => {
        const resource = object as THREE.Object3D & {
          geometry?: THREE.BufferGeometry;
          material?: THREE.Material | THREE.Material[];
        };
        resource.geometry?.dispose();
        const materials = resource.material
          ? Array.isArray(resource.material) ? resource.material : [resource.material]
          : [];
        materials.forEach((material) => material.dispose());
      });
      renderer.dispose();
      renderer.domElement.remove();
    };
  }, []);

  return (
    <article className="tactile-finger-card">
      <header>
        <strong>S{index + 1} · {finger}</strong>
        <span>{sample ? "LIVE" : "WAIT DATA"}</span>
      </header>
      <div className="tactile-viewer" ref={mountRef}>
        <span className="tactile-model-state">
          {modelState === "loading" ? "LOADING STL" : modelState === "error" ? "STL ERROR" : ""}
        </span>
        <span className="tactile-axis-legend"><b>x</b><b>y</b><b>Fz</b></span>
      </div>
      <div className="tactile-value-grid">
        {VALUES.map(([key, label, unit]) => (
          <div className={`tactile-value tactile-value-${key}`} key={key}>
            <span>{label}</span>
            <strong>{sample ? sample[key].toFixed(3) : "—"}</strong>
            <small>{unit}</small>
          </div>
        ))}
      </div>
    </article>
  );
}

export function TactileSensorPanel({ samples = [], live = false }: TactileSensorPanelProps) {
  return (
    <section className="panel tactile-panel">
      <header className="panel-head compact-head">
        <div>
          <p className="section-kicker">TACTILE SENSOR</p>
          <h2>Fingertip Contact Preview</h2>
          <p>STL-surface contact points and sensor-local force vectors</p>
        </div>
        <span className={`tactile-status ${live ? "live" : ""}`}>{live ? "LIVE" : "WAIT DATA"}</span>
      </header>
      <div className="tactile-sensor-grid">
        {FINGERS.map((finger, index) => (
          <TactileTipCard finger={finger} index={index} sample={samples[index] ?? null} key={finger} />
        ))}
      </div>
    </section>
  );
}
