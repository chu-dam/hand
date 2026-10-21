import { useEffect, useMemo, useRef, useState } from "react";
import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
import { ColladaLoader } from "three/examples/jsm/loaders/ColladaLoader.js";
import { STLLoader } from "three/examples/jsm/loaders/STLLoader.js";
import URDFLoader, { type URDFRobot } from "urdf-loader";

import {
  decodeFingerIds,
  vectorMagnitude,
  type GraspDebugMessage,
  type HandSide,
  type JointStateMessage,
  type Point3,
  type RotationMatrix3,
} from "../ros/types";
import { IDENTITY_ROTATION_MATRIX } from "../ros/frames";
import {
  averageCenters,
  estimateContactPolygon3D,
  fitSphereCenterFromContactTriangle,
  sphereEstimationConfigurations,
} from "../ros/sphereGeometry";

const EXPECTED_JOINTS = new Set(
  Array.from({ length: 5 }, (_, fingerIndex) =>
    Array.from({ length: 4 }, (_, jointIndex) =>
      `joint_${fingerIndex + 1}_${jointIndex + 1}`,
    ),
  ).flat(),
);

const FINGER_COLORS = [0xc84b42, 0x0b8f8f, 0x5965bd, 0xd3920b, 0x7d54a5];
const FORCE_COLOR = 0x0b8f8f;
const DEMO_ROTATION_MATRIX: RotationMatrix3 = [
  0.4695, 0, -0.8829,
  0, 1, 0,
  0.8829, 0, 0.4695,
];
const UPWARD_ROTATION_MATRIX: RotationMatrix3 = [
  0, 0, -1,
  0, 1, 0,
  1, 0, 0,
];
const DOWNWARD_ROTATION_MATRIX: RotationMatrix3 = [
  0, 0, 1,
  0, 1, 0,
  -1, 0, 0,
];
const MODEL_PACKAGE = "dg5f_s_description";
const MODEL_ROOT = "robot/dg5f_s_description";
const FINGERTIP_RADIUS_M = 0.009;
const FK_TIP_TO_BALL_CENTER_M = 0.0375 + 0.0090;
const FK_CIRCUMRADIUS_TOLERANCE_M = 0.002;
const BLIND_SPHERE_ADAPTATION_ENABLED = false;

type ViewerStatus = "loading" | "ready" | "error";
interface ViewerState {
  status: ViewerStatus;
  detail: string;
}

interface HandScene3DProps {
  handSide: HandSide;
  jointState: JointStateMessage | null;
  debug: GraspDebugMessage | null;
  handToWorldRotation: RotationMatrix3;
  orientationFromTopic: boolean;
  rotationControlsEnabled: boolean;
  compensationControlsEnabled: boolean;
  onRotationMatrix: (value: number[]) => boolean;
  onCompensationMode: (value: number) => boolean;
  onHumanHandMode: (value: boolean) => boolean;
  onSphereCenterHand: (center: Point3) => boolean;
  onSphereEstimateFailure: (reason: string) => boolean;
}

interface DebugOverlay {
  tipStarts: THREE.Mesh<THREE.SphereGeometry, THREE.MeshBasicMaterial>[];
  fkEndpoints: THREE.Mesh<THREE.SphereGeometry, THREE.MeshBasicMaterial>[];
  fkEndpointSpheres: THREE.Mesh<THREE.SphereGeometry, THREE.MeshPhongMaterial>[];
  forces: THREE.ArrowHelper[];
  geometricCentroid: THREE.Mesh<THREE.SphereGeometry, THREE.MeshBasicMaterial>;
  virtualCentroid: THREE.Mesh<THREE.OctahedronGeometry, THREE.MeshBasicMaterial>;
  fkPreviousCenters: (THREE.Vector3 | null)[];
  fkAverageSphere: THREE.Mesh<THREE.SphereGeometry, THREE.MeshPhongMaterial>;
  fkAverageCenter: THREE.Mesh<THREE.SphereGeometry, THREE.MeshBasicMaterial>;
  contactPolygon: THREE.LineLoop<THREE.BufferGeometry, THREE.LineBasicMaterial>;
  contactPolygonPoints: THREE.Mesh<THREE.SphereGeometry, THREE.MeshBasicMaterial>[];
  contactPolygonCenter: THREE.Mesh<THREE.SphereGeometry, THREE.MeshBasicMaterial>;
  lastFkSphereFailure: string;
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

function millimeters(value: number): string {
  return `${(value * 1000).toFixed(1)} mm`;
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

  const referenceMarkers = (name: string, color: number) => FINGER_COLORS.map((_, index) => {
    const marker = new THREE.Mesh(
      new THREE.SphereGeometry(0.0015, 18, 12),
      new THREE.MeshBasicMaterial({ color, depthTest: false, depthWrite: false }),
    );
    marker.name = `${name}-${index + 1}`;
    marker.visible = false;
    marker.renderOrder = 11;
    group.add(marker);
    return marker;
  });
  const tipStarts = referenceMarkers("physical-tip-start", 0x2563eb);
  const fkEndpoints = referenceMarkers("fk-endpoint", 0xff2d55);
  const fkEndpointSpheres = FINGER_COLORS.map((color, index) => {
    const sphere = new THREE.Mesh(
      new THREE.SphereGeometry(FINGERTIP_RADIUS_M, 24, 16),
      new THREE.MeshPhongMaterial({
        color,
        transparent: true,
        opacity: 0.18,
        depthWrite: false,
        side: THREE.DoubleSide,
      }),
    );
    sphere.name = `debug-fk-fingertip-sphere-${index + 1}`;
    sphere.visible = false;
    sphere.renderOrder = 3;
    group.add(sphere);
    return sphere;
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

  const fkAverageSphere = new THREE.Mesh(
    new THREE.SphereGeometry(0.0375, 40, 28),
    new THREE.MeshPhongMaterial({
      color: 0x7c3aed,
      transparent: true,
      opacity: 0.24,
      depthWrite: false,
      side: THREE.DoubleSide,
    }),
  );
  fkAverageSphere.name = "debug-fk-average-sphere";
  fkAverageSphere.visible = false;
  fkAverageSphere.renderOrder = 4;
  group.add(fkAverageSphere);
  const fkAverageCenter = new THREE.Mesh(
    new THREE.SphereGeometry(0.004, 20, 14),
    new THREE.MeshBasicMaterial({ color: 0x7c3aed, depthTest: false }),
  );
  fkAverageCenter.name = "debug-fk-average-center";
  fkAverageCenter.visible = false;
  fkAverageCenter.renderOrder = 10;
  group.add(fkAverageCenter);

  const contactPolygon = new THREE.LineLoop(
    new THREE.BufferGeometry(),
    new THREE.LineBasicMaterial({ color: 0x00b8d9, depthTest: false }),
  );
  contactPolygon.name = "debug-5f-contact-polygon";
  contactPolygon.visible = false;
  contactPolygon.renderOrder = 12;
  group.add(contactPolygon);
  const contactPolygonPoints = Array.from({ length: 5 }, (_, index) => {
    const marker = new THREE.Mesh(
      new THREE.SphereGeometry(0.002, 16, 10),
      new THREE.MeshBasicMaterial({ color: 0x00b8d9, depthTest: false }),
    );
    marker.name = `debug-5f-contact-point-${index + 1}`;
    marker.visible = false;
    marker.renderOrder = 12;
    group.add(marker);
    return marker;
  });
  const contactPolygonCenter = new THREE.Mesh(
    new THREE.SphereGeometry(0.003, 18, 12),
    new THREE.MeshBasicMaterial({ color: 0x00b8d9, depthTest: false }),
  );
  contactPolygonCenter.name = "debug-5f-contact-polygon-center";
  contactPolygonCenter.visible = false;
  contactPolygonCenter.renderOrder = 12;
  group.add(contactPolygonCenter);

  return {
    tipStarts,
    fkEndpoints,
    fkEndpointSpheres,
    forces,
    geometricCentroid,
    virtualCentroid,
    fkPreviousCenters: Array(6).fill(null),
    fkAverageSphere,
    fkAverageCenter,
    contactPolygon,
    contactPolygonPoints,
    contactPolygonCenter,
    lastFkSphereFailure: "",
  };
}

function applyHandToWorldRotation(
  frame: THREE.Group,
  handToWorldRotation: RotationMatrix3 | null,
) {
  frame.quaternion.identity();
  if (handToWorldRotation !== null) {
    const rotation = new THREE.Matrix4().set(
      handToWorldRotation[0], handToWorldRotation[1], handToWorldRotation[2], 0,
      handToWorldRotation[3], handToWorldRotation[4], handToWorldRotation[5], 0,
      handToWorldRotation[6], handToWorldRotation[7], handToWorldRotation[8], 0,
      0, 0, 0, 1,
    );
    frame.setRotationFromMatrix(rotation);
  }
  frame.updateMatrixWorld(true);
}

function createAxisLabel(label: string, color: number): THREE.Sprite {
  const canvas = document.createElement("canvas");
  canvas.width = 96;
  canvas.height = 96;
  const context = canvas.getContext("2d");
  if (context) {
    context.textAlign = "center";
    context.textBaseline = "middle";
    context.font = "900 58px system-ui, sans-serif";
    context.lineJoin = "round";
    context.lineWidth = 12;
    context.strokeStyle = "rgba(255, 255, 255, 0.96)";
    context.strokeText(label, 48, 50);
    context.fillStyle = `#${color.toString(16).padStart(6, "0")}`;
    context.fillText(label, 48, 50);
  }

  const texture = new THREE.CanvasTexture(canvas);
  texture.colorSpace = THREE.SRGBColorSpace;
  const material = new THREE.SpriteMaterial({
    map: texture,
    transparent: true,
    depthTest: false,
  });
  const sprite = new THREE.Sprite(material);
  sprite.name = `world-axis-${label.toLowerCase()}-label`;
  sprite.scale.setScalar(0.018);
  sprite.renderOrder = 8;
  return sprite;
}

function createWorldAxes(): THREE.Group {
  const length = 0.085;
  const labelOffset = length + 0.012;
  const group = new THREE.Group();
  group.name = "world-axes";

  const axes = new THREE.AxesHelper(length);
  axes.name = "world-axes-lines";
  axes.renderOrder = 7;
  group.add(axes);

  const xLabel = createAxisLabel("X", 0xef4444);
  xLabel.position.set(labelOffset, 0, 0);
  const yLabel = createAxisLabel("Y", 0x22a06b);
  yLabel.position.set(0, labelOffset, 0);
  const zLabel = createAxisLabel("Z", 0x3b82f6);
  zLabel.position.set(0, 0, labelOffset);
  group.add(xLabel, yLabel, zLabel);
  return group;
}

function addOverlayToFrame(frame: THREE.Group, overlay: DebugOverlay) {
  overlay.tipStarts.forEach((marker) => frame.add(marker));
  overlay.fkEndpoints.forEach((marker) => frame.add(marker));
  overlay.fkEndpointSpheres.forEach((sphere) => frame.add(sphere));
  overlay.forces.forEach((arrow) => frame.add(arrow));
  frame.add(
    overlay.geometricCentroid,
    overlay.virtualCentroid,
    overlay.fkAverageSphere,
    overlay.fkAverageCenter,
    overlay.contactPolygon,
    overlay.contactPolygonCenter,
  );
  overlay.contactPolygonPoints.forEach((marker) => frame.add(marker));
}

function updateDebugOverlay(
  overlay: DebugOverlay,
  debug: GraspDebugMessage | null,
  forceScaleMillimeters: number,
  worldOrientationAvailable: boolean,
  robot: URDFRobot | null = null,
  handFrame: THREE.Group | null = null,
  onFkSphereFailure?: (reason: string) => void,
): THREE.Vector3 | null {
  const frameMatches = debug?.header.frame_id === "link_base";
  const showWorldOverlay = frameMatches && worldOrientationAvailable;
  const blindSphereMode = debug?.controller_state === "GROPED_GRASP"
    && debug.grasp_type >= 3;
  const updateBlindSphere = BLIND_SPHERE_ADAPTATION_ENABLED && blindSphereMode
    && (
      !debug.controller_phase.startsWith("blind_")
      || debug.controller_phase === "blind_forward_sphere_estimate"
      || debug.controller_phase === "blind_reverse_sphere_estimate"
      || debug.controller_phase === "blind_direction_sphere_estimate"
    );

  overlay.fkEndpoints.forEach((marker, index) => {
    const point = debug?.fingertip_positions[index];
    marker.visible = showWorldOverlay && isFinitePoint(point);
    if (marker.visible && point) marker.position.set(point.x, point.y, point.z);
  });
  overlay.tipStarts.forEach((marker, index) => {
    const tipLink = robot?.links[`link_${index + 1}_tip`];
    marker.visible = Boolean(showWorldOverlay && tipLink && handFrame);
    if (!marker.visible || !tipLink || !handFrame) return;
    tipLink.updateMatrixWorld(true);
    const worldPoint = tipLink.localToWorld(new THREE.Vector3(0.003, 0, 0));
    marker.position.copy(handFrame.worldToLocal(worldPoint));
  });

  overlay.forces.forEach((arrow, index) => {
    const point = debug?.fingertip_positions[index];
    const force = debug?.total_forces[index];
    const magnitude = vectorMagnitude(force);
    const visible = showWorldOverlay
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

  const showCentroids = showWorldOverlay && debug?.controller_state === "GROPED_GRASP";
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

  const fiveFingerPoints = debug?.controller_state === "GROPED_GRASP"
    && debug.grasp_type === 5
    ? debug.fingertip_positions
      .filter(isFinitePoint)
      .map((point) => new THREE.Vector3(point.x, point.y, point.z))
    : [];
  const contactPolygon = fiveFingerPoints.length === 5
    ? estimateContactPolygon3D(fiveFingerPoints, FINGERTIP_RADIUS_M)
    : null;
  overlay.contactPolygon.visible = Boolean(showWorldOverlay && contactPolygon);
  overlay.fkEndpointSpheres.forEach((sphere, index) => {
    const point = debug?.fingertip_positions[index];
    sphere.visible = Boolean(
      overlay.contactPolygon.visible && isFinitePoint(point),
    );
    if (sphere.visible && point) sphere.position.set(point.x, point.y, point.z);
  });
  overlay.contactPolygonCenter.visible = overlay.contactPolygon.visible;
  overlay.contactPolygonPoints.forEach((marker, index) => {
    marker.visible = overlay.contactPolygon.visible;
    if (contactPolygon) marker.position.copy(contactPolygon.contactPoints[index]);
  });
  if (contactPolygon) {
    overlay.contactPolygon.geometry.setFromPoints(contactPolygon.contactPoints);
    overlay.contactPolygonCenter.position.copy(contactPolygon.center);
  }

  const fkFailures: string[] = [];
  const activeFingerIds = debug
    ? decodeFingerIds(debug.active_finger_ids ?? debug.finger_ids)
    : [];
  const fkConfigurations = sphereEstimationConfigurations(
    debug?.grasp_type ?? 0,
    activeFingerIds,
  );
  const fkSphereCenters = fkConfigurations.map((configuration, index) => {
    const disambiguationPoint = configuration?.disambiguationFinger
      ? debug?.fingertip_positions[configuration.disambiguationFinger - 1]
      : undefined;
    const trianglePoints = configuration.fingerIds
      .map((finger) => debug?.fingertip_positions[finger - 1])
      .filter(isFinitePoint)
      .map((point) => new THREE.Vector3(point.x, point.y, point.z));
    const center = updateBlindSphere && configuration
      ? fitSphereCenterFromContactTriangle(
        trianglePoints,
        FK_TIP_TO_BALL_CENTER_M,
        overlay.fkPreviousCenters[index] ?? new THREE.Vector3(),
        (reason) => { fkFailures[index] = reason; },
        FK_CIRCUMRADIUS_TOLERANCE_M,
        configuration.fingerIds,
        isFinitePoint(disambiguationPoint)
          ? new THREE.Vector3(
            disambiguationPoint.x,
            disambiguationPoint.y,
            disambiguationPoint.z,
          )
          : undefined,
      )
      : null;
    if (center) {
      overlay.fkPreviousCenters[index] = center.clone();
    }
    return center;
  });
  const validFkCenters = fkSphereCenters
    .filter((center): center is THREE.Vector3 => center !== null);
  const contactTriangleCenter = averageCenters(validFkCenters);
  overlay.fkAverageSphere.visible = Boolean(showWorldOverlay && contactTriangleCenter);
  overlay.fkAverageCenter.visible = overlay.fkAverageSphere.visible;
  if (contactTriangleCenter) {
    overlay.fkAverageSphere.position.copy(contactTriangleCenter);
    overlay.fkAverageCenter.position.copy(contactTriangleCenter);
  }
  const fkFailure = fkFailures.find(Boolean) ?? "";
  if (updateBlindSphere && onFkSphereFailure) {
    if (contactTriangleCenter) {
      overlay.lastFkSphereFailure = "";
    } else if (fkFailure && fkFailure !== overlay.lastFkSphereFailure) {
      overlay.lastFkSphereFailure = fkFailure;
      console.warn(`[FK_SPHERE] estimate failed: ${fkFailure}`);
      onFkSphereFailure(fkFailure);
    }
  }
  return contactTriangleCenter;
}

function disposeObject(root: THREE.Object3D) {
  root.traverse((object) => {
    if (object instanceof THREE.Sprite) {
      object.material.map?.dispose();
      object.material.dispose();
      return;
    }
    if (!(object instanceof THREE.Mesh) && !(object instanceof THREE.Line)) return;

    object.geometry?.dispose();
    const materials = Array.isArray(object.material) ? object.material : [object.material];
    materials.forEach((material) => material?.dispose());
  });
}

function errorText(error: unknown): string {
  if (error instanceof Error && error.message) return error.message;
  if (typeof error === "string" && error) return error;
  return "Unable to load the 3D model.";
}

export function HandScene3D({
  handSide,
  jointState,
  debug,
  handToWorldRotation,
  orientationFromTopic,
  rotationControlsEnabled,
  compensationControlsEnabled,
  onRotationMatrix,
  onCompensationMode,
  onHumanHandMode,
  onSphereCenterHand,
  onSphereEstimateFailure,
}: HandScene3DProps) {
  const mountRef = useRef<HTMLDivElement | null>(null);
  const robotRef = useRef<URDFRobot | null>(null);
  const handFrameRef = useRef<THREE.Group | null>(null);
  const overlayRef = useRef<DebugOverlay | null>(null);
  const latestJointState = useRef(jointState);
  const latestDebug = useRef(debug);
  const latestHandToWorldRotation = useRef(handToWorldRotation);
  const latestForceScale = useRef(16);
  const renderRef = useRef<() => void>(() => undefined);
  const resetViewRef = useRef<() => void>(() => undefined);
  const [forceScale, setForceScale] = useState(16);
  const [humanHandMode, setHumanHandMode] = useState(false);
  const [, setContactSphereCenter] = useState<Point3 | null>(null);
  const [viewer, setViewer] = useState<ViewerState>({
    status: "loading",
    detail: "Preparing the DG5F-S CAD model",
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
    latestHandToWorldRotation.current = handToWorldRotation;
    const handFrame = handFrameRef.current;
    if (handFrame) applyHandToWorldRotation(handFrame, handToWorldRotation);
    const overlay = overlayRef.current;
    if (overlay) {
      updateDebugOverlay(
        overlay,
        latestDebug.current,
        latestForceScale.current,
        true,
        robotRef.current,
        handFrameRef.current,
      );
    }
    renderRef.current();
  }, [handToWorldRotation]);

  useEffect(() => {
    latestDebug.current = debug;
    latestForceScale.current = forceScale;
    const overlay = overlayRef.current;
    if (!overlay) return;
    const center = updateDebugOverlay(
      overlay,
      debug,
      forceScale,
      true,
      robotRef.current,
      handFrameRef.current,
      onSphereEstimateFailure,
    );
    if (center) {
      setContactSphereCenter(center);
      onSphereCenterHand(center);
    }
    renderRef.current();
  }, [debug, forceScale, onSphereCenterHand, onSphereEstimateFailure]);

  useEffect(() => {
    const mount = mountRef.current;
    if (!mount) return;

    let disposed = false;
    let robotForCleanup: URDFRobot | null = null;
    const failedAssets: string[] = [];

    setViewer({ status: "loading", detail: "Preparing the DG5F-S CAD model" });

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
    renderer.domElement.setAttribute("aria-label", "Interactive DG5F-S 3D hand viewer");
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

    // ROS uses Z-up while Three.js uses Y-up. The outer group performs only
    // that display conversion. World axes stay fixed in this group; the hand
    // and all link_base debug geometry rotate below it by R_hand_to_world.
    const worldFrame = new THREE.Group();
    worldFrame.name = "ros-world-frame";
    worldFrame.rotation.x = -Math.PI / 2;
    scene.add(worldFrame);
    worldFrame.add(createWorldAxes());

    const handFrame = new THREE.Group();
    handFrame.name = "hand-in-world-frame";
    handFrameRef.current = handFrame;
    applyHandToWorldRotation(handFrame, latestHandToWorldRotation.current);
    worldFrame.add(handFrame);

    const overlay = createDebugOverlay();
    overlayRef.current = overlay;
    addOverlayToFrame(handFrame, overlay);
    updateDebugOverlay(
      overlay,
      latestDebug.current,
      latestForceScale.current,
      true,
      robotRef.current,
      handFrame,
    );

    const render = () => renderer.render(scene, camera);
    renderRef.current = render;

    const fitView = () => {
      const robot = robotRef.current;
      if (!robot) return;

      handFrame.updateMatrixWorld(true);
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
        setViewer({ status: "error", detail: "WebGL context lost. Refresh the page to reconnect." });
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
        setViewer({ status: "loading", detail: `Loading CAD meshes · ${loaded}/${total}` });
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
            detail: `Failed to load ${failedAssets.length} CAD mesh${failedAssets.length === 1 ? "" : "es"}.`,
          });
        } else {
          setViewer({ status: "ready", detail: "Synchronized with live JointState" });
        }
      });
    };

    const loader = new URDFLoader(manager);
    loader.packages = { [MODEL_PACKAGE]: assetUrl(MODEL_ROOT) };
    loader.parseVisual = true;
    loader.parseCollision = false;
    loader.loadMeshCb = (path, loadingManager, material, done) => {
      if (/\.stl(?:$|\?)/i.test(path)) {
        new STLLoader(loadingManager).load(
          path,
          (geometry) => done(new THREE.Mesh(geometry, material)),
          undefined,
          (error) => done(
            new THREE.Group(),
            new Error(errorText(error)),
          ),
        );
        return;
      }

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
            // rotation (and repeated warning); worldFrame converts the ROS
            // world coordinates for display afterward.
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
                new Error(`Empty Collada model: ${path}`),
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
      assetUrl(`${MODEL_ROOT}/urdf/dg5fs_${handSide}.urdf`),
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
        for (let finger = 1; finger <= 5; finger += 1) {
          const tipLink = robot.links[`link_${finger}_tip`];
          if (!tipLink) continue;
          tipLink.children.forEach((visual) => { visual.visible = false; });
          new STLLoader(manager).load(
            assetUrl(`${MODEL_ROOT}/meshes/dg5fs_${handSide}/link_${finger}_tip.STL`),
            (geometry) => {
              if (disposed) {
                geometry.dispose();
                return;
              }
              const visual = new THREE.Mesh(
                geometry,
                new THREE.MeshPhongMaterial({ color: 0xf1f3f2, shininess: 18 }),
              );
              visual.name = `fk-default-tip-${finger}`;
              visual.position.x = 0.003;
              tipLink.add(visual);
              render();
            },
            undefined,
            (error) => failedAssets.push(errorText(error)),
          );
        }
        handFrame.add(robot);
        applyJointState(robot, latestJointState.current);
        updateDebugOverlay(
          overlay,
          latestDebug.current,
          latestForceScale.current,
          true,
          robot,
          handFrame,
        );
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
      if (handFrameRef.current === handFrame) handFrameRef.current = null;
      if (overlayRef.current === overlay) overlayRef.current = null;
      renderRef.current = () => undefined;
      resetViewRef.current = () => undefined;
    };
  }, [handSide]);

  const frameMatches = !debug || debug.header.frame_id === "link_base";
  const live = viewer.status === "ready" && mappedJointCount === EXPECTED_JOINTS.size;
  const contactPolygon = useMemo(() => {
    if (debug?.controller_state !== "GROPED_GRASP" || debug.grasp_type !== 5) return null;
    const points = debug.fingertip_positions
      .filter(isFinitePoint)
      .map((point) => new THREE.Vector3(point.x, point.y, point.z));
    return points.length === 5
      ? estimateContactPolygon3D(points, FINGERTIP_RADIUS_M)
      : null;
  }, [debug]);

  return (
    <section className="panel scene-panel">
      <div className="panel-head hand-scene-head">
        <div>
          <p className="section-kicker">CONTROL GEOMETRY</p>
          <h2>3D Viewer</h2>
          <p>Live URDF · world axes · R_hand_to_world × link_base calculated forces</p>
        </div>
        <div className="scene-tools">
          <span className={`world-frame-badge ${orientationFromTopic ? "ready" : "default"}`}>
            {orientationFromTopic ? "WORLD · TOPIC" : "WORLD · DEFAULT I"}
          </span>
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

        <div className="sphere-estimation-controls">
          <span>HAND MODE</span>
          {([false, true] as const).map((human) => (
            <button
              key={String(human)}
              type="button"
              className={humanHandMode === human ? "active" : ""}
              disabled={!rotationControlsEnabled}
              onClick={() => {
                if (onHumanHandMode(human)) {
                  setHumanHandMode(human);
                }
              }}
            >
              {human ? "HUMAN" : "CURRENT"}
            </button>
          ))}
        </div>

        <div className="scene-orientation-controls">
          <button
            type="button"
            disabled={!rotationControlsEnabled}
            onClick={() => onRotationMatrix([...DEMO_ROTATION_MATRIX])}
          >
            DEMO
          </button>
          <button
            type="button"
            disabled={!rotationControlsEnabled}
            onClick={() => onRotationMatrix([...IDENTITY_ROTATION_MATRIX])}
          >
            RESET
          </button>
          <button
            type="button"
            disabled={!rotationControlsEnabled}
            onClick={() => onRotationMatrix([...UPWARD_ROTATION_MATRIX])}
          >
            UPWARD
          </button>
          <button
            type="button"
            disabled={!rotationControlsEnabled}
            onClick={() => onRotationMatrix([...DOWNWARD_ROTATION_MATRIX])}
          >
            DOWNWARD
          </button>
          {[
            [0, "FRICTION", "FRICTION_ONLY"],
            [1, "GRAVITY", "GRAVITY_ONLY"],
            [2, "GRAVITY + FRICTION", "GRAVITY_FRICTION"],
          ].map(([mode, label, state]) => (
            <button
              key={mode}
              type="button"
              className={`compensation-button ${debug?.controller_state === state ? "active" : ""}`}
              disabled={!compensationControlsEnabled}
              onClick={() => onCompensationMode(Number(mode))}
            >
              {label}
            </button>
          ))}
        </div>

        <div className={`sphere-position-overlay ${contactPolygon ? "live" : "waiting"}`}>
          <span>5F CONTACT POLYGON · BEST-FIT PLANE · TIP R=9 MM</span>
          <div><b>CX</b><strong>{contactPolygon ? millimeters(contactPolygon.center.x) : "—"}</strong></div>
          <div><b>CY</b><strong>{contactPolygon ? millimeters(contactPolygon.center.y) : "—"}</strong></div>
          <div><b>CZ</b><strong>{contactPolygon ? millimeters(contactPolygon.center.z) : "—"}</strong></div>
          <div><b>REQ</b><strong>{contactPolygon ? millimeters(contactPolygon.equivalentRadius) : "—"}</strong></div>
        </div>

        {viewer.status !== "ready" && (
          <div className={`model-state-overlay ${viewer.status}`} role="status">
            <span className="model-state-spinner" />
            <strong>{viewer.status === "error" ? "3D Model Error" : "Loading 3D Model"}</strong>
            <small>{viewer.detail}</small>
          </div>
        )}

        <div className="scene-warning-stack">
          {viewer.status === "ready" && mappedJointCount < EXPECTED_JOINTS.size && (
            <div className="scene-data-warning">
              Waiting for JointState · {mappedJointCount}/{EXPECTED_JOINTS.size} joints mapped
            </div>
          )}

          {!frameMatches && (
            <div className="scene-data-warning frame-warning">
              Force overlay hidden · source frame {debug?.header.frame_id || "—"}
            </div>
          )}
        </div>

        <div className="scene-help">Drag: rotate · Wheel: zoom · Right drag: pan</div>
      </div>

      <div className="scene-legend hand-scene-legend">
        <span><i className="legend-model" />URDF hand · JointState</span>
        <span><i className="legend-dot tip-start" />Physical tip start</span>
        <span><i className="legend-dot fk-endpoint" />FK endpoint</span>
        <span><i className="legend-fingertip-sphere" />9 mm fingertip sphere</span>
        <span><i className="legend-dot fingertip" />Debug fingertip</span>
        <span><i className="legend-dot cg" />Geometric centroid</span>
        <span><i className="legend-contact-polygon" />5F contact polygon</span>
        <span><i className="legend-diamond" />Virtual centroid</span>
        <span><i className="legend-line" />Calculated total force</span>
        <span className="world-axis-legend">
          <b className="axis-x">X</b><b className="axis-y">Y</b><b className="axis-z">Z</b>
          World axes
        </span>
        <span className="scene-frame-label">
          display world · source {debug?.header.frame_id || "link_base"} · orientation {orientationFromTopic ? "topic" : "default I"}
        </span>
      </div>
    </section>
  );
}
