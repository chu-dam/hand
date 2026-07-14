import { useEffect, useMemo, useRef, useState } from "react";
import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
import { ColladaLoader } from "three/examples/jsm/loaders/ColladaLoader.js";
import URDFLoader, { type URDFRobot } from "urdf-loader";

import {
  vectorMagnitude,
  type GraspDebugMessage,
  type JointStateMessage,
  type Point3,
} from "../ros/types";

const EXPECTED_JOINTS = new Set(
  Array.from({ length: 5 }, (_, fingerIndex) =>
    Array.from({ length: 4 }, (_, jointIndex) =>
      `joint_${fingerIndex + 1}_${jointIndex + 1}`,
    ),
  ).flat(),
);

const FINGER_COLORS = [0xc84b42, 0x0b8f8f, 0x5965bd, 0xd3920b, 0x7d54a5];
const FORCE_COLOR = 0x0b8f8f;
const MODEL_PACKAGE = "dg5f_s_description";
const MODEL_ROOT = "robot/dg5f_s_description";
const MODEL_PATH = `${MODEL_ROOT}/urdf/dg5fs_left.urdf`;

type ViewerStatus = "loading" | "ready" | "error";

interface ViewerState {
  status: ViewerStatus;
  detail: string;
}

interface HandScene3DProps {
  jointState: JointStateMessage | null;
  debug: GraspDebugMessage | null;
}

interface DebugOverlay {
  fingertips: THREE.Mesh<THREE.SphereGeometry, THREE.MeshBasicMaterial>[];
  forces: THREE.ArrowHelper[];
  geometricCentroid: THREE.Mesh<THREE.SphereGeometry, THREE.MeshBasicMaterial>;
  virtualCentroid: THREE.Mesh<THREE.OctahedronGeometry, THREE.MeshBasicMaterial>;
}

function assetUrl(path: string): string {
  const base = import.meta.env.BASE_URL.endsWith("/")
    ? import.meta.env.BASE_URL
    : `${import.meta.env.BASE_URL}/`;
  return `${base}${path.replace(/^\//, "")}`;
}

function isFinitePoint(point: Point3 | undefined): point is Point3 {
  return Boolean(
    point
      && Number.isFinite(point.x)
      && Number.isFinite(point.y)
      && Number.isFinite(point.z),
  );
}

function applyJointState(robot: URDFRobot, state: JointStateMessage | null): number {
  if (!state) return 0;

  const values: Record<string, number> = {};
  const count = Math.min(state.name.length, state.position.length);

  for (let index = 0; index < count; index += 1) {
    const name = state.name[index];
    const position = state.position[index];
    if (name in robot.joints && Number.isFinite(position)) {
      values[name] = position;
    }
  }

  robot.setJointValues(values);
  robot.updateMatrixWorld(true);
  return Object.keys(values).length;
}

function createDebugOverlay(): DebugOverlay {
  const group = new THREE.Group();
  group.name = "grasp-debug-overlay";

  const fingertips = FINGER_COLORS.map((color, index) => {
    const marker = new THREE.Mesh(
      new THREE.SphereGeometry(0.0038, 18, 12),
      new THREE.MeshBasicMaterial({ color }),
    );
    marker.name = `debug-fingertip-${index + 1}`;
    marker.visible = false;
    marker.renderOrder = 4;
    group.add(marker);
    return marker;
  });

  const forces = FINGER_COLORS.map((_, index) => {
    const arrow = new THREE.ArrowHelper(
      new THREE.Vector3(0, 0, 1),
      new THREE.Vector3(),
      0.01,
      FORCE_COLOR,
      0.004,
      0.0025,
    );
    arrow.name = `debug-total-force-${index + 1}`;
    arrow.visible = false;
    group.add(arrow);
    return arrow;
  });

  const geometricCentroid = new THREE.Mesh(
    new THREE.SphereGeometry(0.005, 20, 14),
    new THREE.MeshBasicMaterial({ color: 0xf1b632 }),
  );
  geometricCentroid.name = "debug-geometric-centroid";
  geometricCentroid.visible = false;
  group.add(geometricCentroid);

  const virtualCentroid = new THREE.Mesh(
    new THREE.OctahedronGeometry(0.006),
    new THREE.MeshBasicMaterial({ color: 0xef4444 }),
  );
  virtualCentroid.name = "debug-virtual-centroid";
  virtualCentroid.visible = false;
  group.add(virtualCentroid);

  return {
    fingertips,
    forces,
    geometricCentroid,
    virtualCentroid,
  };
}

function addOverlayToFrame(frame: THREE.Group, overlay: DebugOverlay) {
  overlay.fingertips.forEach((marker) => frame.add(marker));
  overlay.forces.forEach((arrow) => frame.add(arrow));
  frame.add(overlay.geometricCentroid, overlay.virtualCentroid);
}

function updateDebugOverlay(
  overlay: DebugOverlay,
  debug: GraspDebugMessage | null,
  forceScaleMillimeters: number,
) {
  const frameMatches = debug?.header.frame_id === "link_base";

  overlay.fingertips.forEach((marker, index) => {
    const point = debug?.fingertip_positions[index];
    marker.visible = frameMatches && isFinitePoint(point);
    if (marker.visible && point) marker.position.set(point.x, point.y, point.z);
  });

  overlay.forces.forEach((arrow, index) => {
    const point = debug?.fingertip_positions[index];
    const force = debug?.total_forces[index];
    const magnitude = vectorMagnitude(force);
    const visible = frameMatches
      && isFinitePoint(point)
      && isFinitePoint(force)
      && magnitude > 1e-6;

    arrow.visible = visible;
    if (!visible || !point || !force) return;

    const direction = new THREE.Vector3(force.x, force.y, force.z).normalize();
    const length = Math.min((magnitude * forceScaleMillimeters) / 1_000, 0.12);
    arrow.position.set(point.x, point.y, point.z);
    arrow.setDirection(direction);
    arrow.setLength(
      length,
      Math.min(length * 0.34, 0.012),
      Math.min(length * 0.2, 0.006),
    );
  });

  const showCentroids = frameMatches && debug?.controller_state === "GROPED_GRASP";
  const geometricPoint = debug?.geometric_centroid;
  const virtualPoint = debug?.virtual_centroid;

  overlay.geometricCentroid.visible = showCentroids && isFinitePoint(geometricPoint);
  if (overlay.geometricCentroid.visible && geometricPoint) {
    overlay.geometricCentroid.position.set(
      geometricPoint.x,
      geometricPoint.y,
      geometricPoint.z,
    );
  }

  overlay.virtualCentroid.visible = showCentroids && isFinitePoint(virtualPoint);
  if (overlay.virtualCentroid.visible && virtualPoint) {
    overlay.virtualCentroid.position.set(virtualPoint.x, virtualPoint.y, virtualPoint.z);
  }
}

function disposeObject(root: THREE.Object3D) {
  root.traverse((object) => {
    if (!(object instanceof THREE.Mesh) && !(object instanceof THREE.Line)) return;

    object.geometry?.dispose();
    const materials = Array.isArray(object.material) ? object.material : [object.material];
    materials.forEach((material) => material?.dispose());
  });
}

function errorText(error: unknown): string {
  if (error instanceof Error && error.message) return error.message;
  if (typeof error === "string" && error) return error;
  return "3D 모델을 불러오지 못했습니다.";
}

export function HandScene3D({ jointState, debug }: HandScene3DProps) {
  const mountRef = useRef<HTMLDivElement | null>(null);
  const robotRef = useRef<URDFRobot | null>(null);
  const overlayRef = useRef<DebugOverlay | null>(null);
  const latestJointState = useRef(jointState);
  const latestDebug = useRef(debug);
  const latestForceScale = useRef(16);
  const renderRef = useRef<() => void>(() => undefined);
  const resetViewRef = useRef<() => void>(() => undefined);

  const [forceScale, setForceScale] = useState(16);
  const [viewer, setViewer] = useState<ViewerState>({
    status: "loading",
    detail: "DG5F-S CAD 모델 준비 중",
  });

  const mappedJointCount = useMemo(() => {
    if (!jointState) return 0;
    const observed = new Set<string>();
    jointState.name.forEach((name, index) => {
      const position = jointState.position[index];
      if (EXPECTED_JOINTS.has(name) && Number.isFinite(position)) observed.add(name);
    });
    return observed.size;
  }, [jointState]);

  useEffect(() => {
    latestJointState.current = jointState;
    const robot = robotRef.current;
    if (!robot) return;
    applyJointState(robot, jointState);
    renderRef.current();
  }, [jointState]);

  useEffect(() => {
    latestDebug.current = debug;
    latestForceScale.current = forceScale;
    const overlay = overlayRef.current;
    if (!overlay) return;
    updateDebugOverlay(overlay, debug, forceScale);
    renderRef.current();
  }, [debug, forceScale]);

  useEffect(() => {
    const mount = mountRef.current;
    if (!mount) return;

    let disposed = false;
    let robotForCleanup: URDFRobot | null = null;
    const failedAssets: string[] = [];

    setViewer({ status: "loading", detail: "DG5F-S CAD 모델 준비 중" });

    let renderer: THREE.WebGLRenderer;
    try {
      renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    } catch (error) {
      setViewer({ status: "error", detail: errorText(error) });
      return;
    }

    renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
    renderer.setClearColor(0xf4f8f9, 0);
    renderer.outputColorSpace = THREE.SRGBColorSpace;
    renderer.toneMapping = THREE.ACESFilmicToneMapping;
    renderer.toneMappingExposure = 1.08;
    renderer.domElement.className = "hand-canvas";
    renderer.domElement.setAttribute("aria-label", "실시간 DG5F-S 3D 손 모델");
    mount.appendChild(renderer.domElement);

    const scene = new THREE.Scene();
    const camera = new THREE.PerspectiveCamera(38, 1, 0.001, 10);
    camera.position.set(0.38, 0.2, 0.3);

    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.08;
    controls.screenSpacePanning = true;
    controls.target.set(0, 0.09, 0);
    controls.minDistance = 0.08;
    controls.maxDistance = 1.4;

    scene.add(new THREE.HemisphereLight(0xeefbff, 0x57636a, 2.2));
    const keyLight = new THREE.DirectionalLight(0xffffff, 3.1);
    keyLight.position.set(0.6, 0.8, 0.5);
    scene.add(keyLight);
    const fillLight = new THREE.DirectionalLight(0x8bd5d2, 1.1);
    fillLight.position.set(-0.4, 0.3, -0.5);
    scene.add(fillLight);

    const grid = new THREE.GridHelper(0.42, 21, 0x90a7af, 0xcbd5d9);
    grid.position.y = -0.012;
    const gridMaterials = Array.isArray(grid.material) ? grid.material : [grid.material];
    gridMaterials.forEach((material) => {
      material.transparent = true;
      material.opacity = 0.32;
    });
    scene.add(grid);

    const linkBaseFrame = new THREE.Group();
    linkBaseFrame.name = "link-base-frame";
    linkBaseFrame.rotation.x = -Math.PI / 2;
    scene.add(linkBaseFrame);

    const axes = new THREE.AxesHelper(0.055);
    axes.name = "link-base-axes";
    linkBaseFrame.add(axes);

    const overlay = createDebugOverlay();
    overlayRef.current = overlay;
    addOverlayToFrame(linkBaseFrame, overlay);
    updateDebugOverlay(overlay, latestDebug.current, latestForceScale.current);

    const render = () => renderer.render(scene, camera);
    renderRef.current = render;

    const fitView = () => {
      const robot = robotRef.current;
      if (!robot) return;

      linkBaseFrame.updateMatrixWorld(true);
      const box = new THREE.Box3().setFromObject(robot);
      if (box.isEmpty()) return;

      const sphere = box.getBoundingSphere(new THREE.Sphere());
      const radius = Math.max(sphere.radius, 0.08);
      const verticalFov = THREE.MathUtils.degToRad(camera.fov);
      const distance = (radius / Math.sin(verticalFov / 2)) * 1.18;
      const viewDirection = new THREE.Vector3(1.25, 0.58, 0.92).normalize();

      camera.near = Math.max(radius / 100, 0.001);
      camera.far = Math.max(radius * 30, 5);
      camera.position.copy(sphere.center).addScaledVector(viewDirection, distance);
      camera.updateProjectionMatrix();

      controls.target.copy(sphere.center);
      controls.minDistance = Math.max(radius * 0.7, 0.06);
      controls.maxDistance = Math.max(radius * 8, 0.8);
      controls.update();
      controls.saveState();
      render();
    };

    resetViewRef.current = () => {
      controls.reset();
      render();
    };

    const resize = () => {
      const width = Math.max(mount.clientWidth, 1);
      const height = Math.max(mount.clientHeight, 1);
      renderer.setSize(width, height, false);
      camera.aspect = width / height;
      camera.updateProjectionMatrix();
      render();
    };
    const resizeObserver = new ResizeObserver(resize);
    resizeObserver.observe(mount);
    resize();

    const onContextLost = (event: Event) => {
      event.preventDefault();
      if (!disposed) {
        setViewer({ status: "error", detail: "WebGL 연결이 끊겼습니다. 페이지를 새로고침해 주세요." });
      }
    };
    renderer.domElement.addEventListener("webglcontextlost", onContextLost);

    renderer.setAnimationLoop(() => {
      controls.update();
      render();
    });

    const manager = new THREE.LoadingManager();
    manager.onProgress = (_url, loaded, total) => {
      if (!disposed) {
        setViewer({ status: "loading", detail: `CAD 메시 불러오는 중 · ${loaded}/${total}` });
      }
    };
    manager.onError = (url) => failedAssets.push(url);
    manager.onLoad = () => {
      if (disposed) return;
      window.requestAnimationFrame(() => {
        if (disposed) return;
        fitView();
        if (failedAssets.length > 0) {
          setViewer({
            status: "error",
            detail: `CAD 메시 ${failedAssets.length}개를 불러오지 못했습니다.`,
          });
        } else {
          setViewer({ status: "ready", detail: "실제 JointState 동기화" });
        }
      });
    };

    const loader = new URDFLoader(manager);
    loader.packages = { [MODEL_PACKAGE]: assetUrl(MODEL_ROOT) };
    loader.parseVisual = true;
    loader.parseCollision = false;
    loader.loadMeshCb = (path, loadingManager, _material, done) => {
      const fileLoader = new THREE.FileLoader(loadingManager);
      fileLoader.setResponseType("text");
      fileLoader.load(
        path,
        (data) => {
          try {
            const source = typeof data === "string"
              ? data
              : new TextDecoder().decode(data);

            // The vendor meshes use ROS Z-up coordinates. Telling ColladaLoader
            // that the source is already Y-up prevents its automatic root
            // rotation (and repeated warning); linkBaseFrame converts the robot
            // and debug overlay together afterward.
            const rosFrameSource = source.replace(
              /<up_axis>\s*Z_UP\s*<\/up_axis>/i,
              "<up_axis>Y_UP</up_axis>",
            );
            const colladaLoader = new ColladaLoader(loadingManager);
            const result = colladaLoader.parse(
              rosFrameSource,
              THREE.LoaderUtils.extractUrlBase(path),
            );

            if (!result) {
              done(
                new THREE.Group(),
                new Error(`빈 Collada 모델: ${path}`),
              );
              return;
            }

            done(result.scene);
          } catch (error) {
            done(
              new THREE.Group(),
              new Error(errorText(error)),
            );
          }
        },
        undefined,
        (error) => done(
          new THREE.Group(),
          new Error(errorText(error)),
        ),
      );
    };

    loader.load(
      assetUrl(MODEL_PATH),
      (robot) => {
        if (disposed) {
          disposeObject(robot);
          return;
        }
        robotForCleanup = robot;
        robotRef.current = robot;
        EXPECTED_JOINTS.forEach((jointName) => {
          const joint = robot.joints[jointName];
          if (joint) joint.ignoreLimits = true;
        });
        linkBaseFrame.add(robot);
        applyJointState(robot, latestJointState.current);
        render();
      },
      undefined,
      (error) => {
        if (!disposed) setViewer({ status: "error", detail: errorText(error) });
      },
    );

    return () => {
      disposed = true;
      renderer.setAnimationLoop(null);
      resizeObserver.disconnect();
      controls.dispose();
      renderer.domElement.removeEventListener("webglcontextlost", onContextLost);
      disposeObject(scene);
      renderer.dispose();
      // React StrictMode and Vite HMR can mount this viewer repeatedly during
      // development. Explicitly return the WebGL context instead of waiting for
      // browser garbage collection so later mounts do not exhaust the limit.
      renderer.forceContextLoss();
      if (renderer.domElement.parentNode === mount) mount.removeChild(renderer.domElement);
      if (robotRef.current === robotForCleanup) robotRef.current = null;
      if (overlayRef.current === overlay) overlayRef.current = null;
      renderRef.current = () => undefined;
      resetViewRef.current = () => undefined;
    };
  }, []);

  const frameMatches = !debug || debug.header.frame_id === "link_base";
  const live = viewer.status === "ready" && mappedJointCount === EXPECTED_JOINTS.size;

  return (
    <section className="panel scene-panel">
      <div className="panel-head hand-scene-head">
        <div>
          <p className="section-kicker">CONTROL GEOMETRY</p>
          <h2>실시간 3D 손 모델</h2>
          <p>실제 URDF · JointState 20축 · link_base 기준 계산 힘</p>
        </div>
        <div className="scene-tools">
          <span className={`model-live-badge ${live ? "live" : viewer.status}`}>
            {viewer.status === "error"
              ? "MODEL ERROR"
              : viewer.status === "loading"
                ? "MODEL LOADING"
                : `${mappedJointCount}/${EXPECTED_JOINTS.size} JOINTS`}
          </span>
          <label className="scale-control">
            <span>Force scale</span>
            <input
              type="range"
              min="5"
              max="60"
              step="1"
              value={forceScale}
              onChange={(event) => setForceScale(Number(event.target.value))}
            />
            <strong>{forceScale} mm/N</strong>
          </label>
          <button className="view-reset-button" type="button" onClick={() => resetViewRef.current()}>
            Reset view
          </button>
        </div>
      </div>

      <div className="scene-wrap hand-scene-wrap">
        <div ref={mountRef} className="hand-canvas-mount" />

        {viewer.status !== "ready" && (
          <div className={`model-state-overlay ${viewer.status}`} role="status">
            <span className="model-state-spinner" />
            <strong>{viewer.status === "error" ? "3D 모델 오류" : "3D 모델 로딩 중"}</strong>
            <small>{viewer.detail}</small>
          </div>
        )}

        {viewer.status === "ready" && mappedJointCount < EXPECTED_JOINTS.size && (
          <div className="scene-data-warning">
            JointState 대기 중 · 현재 {mappedJointCount}/{EXPECTED_JOINTS.size}축
          </div>
        )}

        {!frameMatches && (
          <div className="scene-data-warning frame-warning">
            힘 오버레이 숨김 · frame {debug?.header.frame_id || "—"}
          </div>
        )}

        <div className="scene-help">Drag: rotate · Wheel: zoom · Right drag: pan</div>
      </div>

      <div className="scene-legend hand-scene-legend">
        <span><i className="legend-model" />URDF hand · JointState</span>
        <span><i className="legend-dot fingertip" />Debug fingertip</span>
        <span><i className="legend-dot cg" />Geometric centroid</span>
        <span><i className="legend-diamond" />Virtual centroid</span>
        <span><i className="legend-line" />Calculated total force</span>
        <span className="scene-frame-label">frame {debug?.header.frame_id || "link_base"}</span>
      </div>
    </section>
  );
}
