import { useCallback, useEffect, useRef, useState } from "react";
import { Ros, Topic } from "roslib";

import { decodeRotationMatrix, IDENTITY_ROTATION_MATRIX } from "./frames";
import type {
  Float64MultiArrayMessage,
  GraspDebugMessage,
  HandSide,
  JointStateMessage,
  Point3,
  RosConnectionStatus,
  RotationMatrix3,
  TactileSample,
  TipAreaSample,
  Vector3StampedMessage,
} from "./types";

function topicsForHand(side: HandSide) {
  const prefix = side === "right" ? "/dg5f_grasp_control/right" : "/dg5f_grasp_control";
  return {
    jointState: `/dg5f_s_${side}/joint_states`,
    debug: `${prefix}/debug`,
    grasp: side === "right" ? `${prefix}/grasp_type` : "/grasp_type",
    pose: side === "right" ? `${prefix}/pose_type` : "/pose_type",
    alpha: `${prefix}/alpha1_cmd`,
    teaching: `${prefix}/teaching_mode`,
    rotation: `${prefix}/rotation_matrix_cmd`,
    relativeRotationDegrees: `${prefix}/relative_rotation_deg_cmd`,
    continuousRotation: `${prefix}/continuous_rotation_cmd`,
    blindDirectionToggle: `${prefix}/blind_direction_toggle`,
    thumbIndexRingArea: `${prefix}/thumb_index_ring_tip_area`,
    relativeTranslation: `${prefix}/relative_translation_cmd`,
    tactile: `/dg5f_s_${side}/tactile_contacts`,
    tactileContactPoints: `${prefix}/tactile_contact_points`,
  };
}

const RECONNECT_DELAY_MS = 2_000;
const JOINT_RENDER_PERIOD_MS = 33;
const ROTATION_RENDER_PERIOD_MS = 33;

interface CommandMessage {
  data: unknown;
  layout?: {
    dim: never[];
    data_offset: number;
  };
}

type CommandTopic = Topic<CommandMessage>;
type RelativeTranslationTopic = Topic<Vector3StampedMessage>;

interface Publishers {
  grasp: CommandTopic | null;
  pose: CommandTopic | null;
  alpha: CommandTopic | null;
  teaching: CommandTopic | null;
  rotation: CommandTopic | null;
  relativeRotationDegrees: CommandTopic | null;
  continuousRotation: CommandTopic | null;
  blindDirectionToggle: CommandTopic | null;
  relativeTranslation: RelativeTranslationTopic | null;
}

const EMPTY_PUBLISHERS: Publishers = {
  grasp: null,
  pose: null,
  alpha: null,
  teaching: null,
  rotation: null,
  relativeRotationDegrees: null,
  continuousRotation: null,
  blindDirectionToggle: null,
  relativeTranslation: null,
};

function connectionErrorMessage(event: unknown): string {
  if (event instanceof Error && event.message) return event.message;
  if (typeof event === "string" && event) return event;
  return "rosbridge 연결 오류";
}

export function defaultRosbridgeUrl(): string {
  const queryValue = new URLSearchParams(window.location.search).get("rosbridge");
  if (queryValue) return queryValue;
  if (import.meta.env.VITE_ROSBRIDGE_URL) {
    return String(import.meta.env.VITE_ROSBRIDGE_URL);
  }
  const scheme = window.location.protocol === "https:" ? "wss" : "ws";
  const hostname = window.location.hostname || "localhost";
  return `${scheme}://${hostname}:9090`;
}

export function useRosBridge(url: string, handSide: HandSide) {
  const [status, setStatus] = useState<RosConnectionStatus>("connecting");
  const [error, setError] = useState("");
  const [jointState, setJointState] = useState<JointStateMessage | null>(null);
  const [debug, setDebug] = useState<GraspDebugMessage | null>(null);
  const [handToWorldRotation, setHandToWorldRotation] = useState<RotationMatrix3>(
    IDENTITY_ROTATION_MATRIX,
  );
  const [lastJointAt, setLastJointAt] = useState<number | null>(null);
  const [lastDebugAt, setLastDebugAt] = useState<number | null>(null);
  const [lastRotationAt, setLastRotationAt] = useState<number | null>(null);
  const [tactileSamples, setTactileSamples] = useState<TactileSample[]>([]);
  const [tactileContactPoints, setTactileContactPoints] = useState<Point3[]>([]);
  const [thumbIndexRingArea, setThumbIndexRingArea] = useState<TipAreaSample | null>(null);
  const [lastTactileAt, setLastTactileAt] = useState<number | null>(null);
  const [attempt, setAttempt] = useState(0);

  const activeRos = useRef<Ros | null>(null);
  const publishers = useRef<Publishers>({ ...EMPTY_PUBLISHERS });

  useEffect(() => {
    let disposed = false;
    let reconnectTimer: number | undefined;
    let lastJointRenderAt = 0;
    const ros = new Ros();
    const topics = topicsForHand(handSide);
    activeRos.current = ros;

    const clearTelemetry = () => {
      setJointState(null);
      setDebug(null);
      setHandToWorldRotation(IDENTITY_ROTATION_MATRIX);
      setLastJointAt(null);
      setLastDebugAt(null);
      setLastRotationAt(null);
      setTactileSamples([]);
      setLastTactileAt(null);
      setTactileContactPoints([]);
      setThumbIndexRingArea(null);
    };
    const scheduleReconnect = () => {
      if (disposed || reconnectTimer !== undefined) return;
      reconnectTimer = window.setTimeout(() => {
        reconnectTimer = undefined;
        setAttempt((value) => value + 1);
      }, RECONNECT_DELAY_MS);
    };

    clearTelemetry();

    const jointTopic = new Topic<JointStateMessage>({
      ros,
      name: topics.jointState,
      messageType: "sensor_msgs/msg/JointState",
      throttle_rate: JOINT_RENDER_PERIOD_MS,
      queue_length: 1,
      reconnect_on_close: false,
    });
    const debugTopic = new Topic<GraspDebugMessage>({
      ros,
      name: topics.debug,
      messageType: "dg5f_grasp_interfaces/msg/GraspDebug",
      queue_length: 1,
      reconnect_on_close: false,
    });
    const rotationTopic = new Topic<Float64MultiArrayMessage>({
      ros,
      name: topics.rotation,
      messageType: "std_msgs/msg/Float64MultiArray",
      throttle_rate: ROTATION_RENDER_PERIOD_MS,
      queue_length: 1,
      reconnect_on_close: false,
    });
    const tactileTopic = new Topic<Float64MultiArrayMessage>({
      ros,
      name: topics.tactile,
      messageType: "std_msgs/msg/Float64MultiArray",
      throttle_rate: 50,
      queue_length: 1,
      reconnect_on_close: false,
    });

    const commandTopic = (name: string, messageType: string) => new Topic<CommandMessage>({
      ros,
      name,
      messageType,
      queue_size: 1,
      reconnect_on_close: false,
    });

    publishers.current = {
      grasp: commandTopic(topics.grasp, "std_msgs/msg/Int32"),
      pose: commandTopic(topics.pose, "std_msgs/msg/Int32"),
      alpha: commandTopic(topics.alpha, "std_msgs/msg/Float64"),
      teaching: commandTopic(topics.teaching, "std_msgs/msg/Bool"),
      rotation: commandTopic(topics.rotation, "std_msgs/msg/Float64MultiArray"),
      relativeRotationDegrees: commandTopic(topics.relativeRotationDegrees, "std_msgs/msg/Float64"),
      continuousRotation: commandTopic(topics.continuousRotation, "std_msgs/msg/Bool"),
      blindDirectionToggle: commandTopic(topics.blindDirectionToggle, "std_msgs/msg/Int32"),
      relativeTranslation: new Topic<Vector3StampedMessage>({
        ros,
        name: topics.relativeTranslation,
        messageType: "geometry_msgs/msg/Vector3Stamped",
        queue_size: 1,
        reconnect_on_close: false,
      }),
    };

    const onJointState = (message: JointStateMessage) => {
      if (disposed) return;
      const now = Date.now();
      if (now - lastJointRenderAt < JOINT_RENDER_PERIOD_MS) return;
      lastJointRenderAt = now;
      setJointState(message);
      setLastJointAt(now);
    };
    const onDebug = (message: GraspDebugMessage) => {
      if (disposed) return;
      setDebug(message);
      setLastDebugAt(Date.now());
    };
    const onRotationMatrix = (message: Float64MultiArrayMessage) => {
      if (disposed) return;
      const rotation = decodeRotationMatrix(message.data);
      if (rotation === null) return;
      setHandToWorldRotation(rotation);
      setLastRotationAt(Date.now());
    };
    const onTactile = (message: Float64MultiArrayMessage) => {
      if (disposed || message.data.length < 25 || !message.data.slice(0, 25).every(Number.isFinite)) return;
      setTactileSamples(Array.from({ length: 5 }, (_, finger) => {
        const base = finger * 5;
        return {
          x: message.data[base],
          y: message.data[base + 1],
          fx: message.data[base + 2],
          fy: message.data[base + 3],
          fz: message.data[base + 4],
        };
      }));
      setLastTactileAt(Date.now());
    };
    const tactileContactTopic = new Topic<Float64MultiArrayMessage>({ ros, name: topics.tactileContactPoints, messageType: "std_msgs/msg/Float64MultiArray" });
    const thumbIndexRingAreaTopic = new Topic<Float64MultiArrayMessage>({
      ros,
      name: topics.thumbIndexRingArea,
      messageType: "std_msgs/msg/Float64MultiArray",
      queue_length: 1,
      reconnect_on_close: false,
    });
    const onTactileContactPoints = (message: Float64MultiArrayMessage) => {
      if (message.data.length < 15) return;
      setTactileContactPoints(Array.from({ length: 5 }, (_, i) => ({ x: message.data[i * 3], y: message.data[i * 3 + 1], z: message.data[i * 3 + 2] })));
    };
    const onThumbIndexRingArea = (message: Float64MultiArrayMessage) => {
      if (message.data.length < 2 || !message.data.slice(0, 2).every(Number.isFinite)) return;
      setThumbIndexRingArea({ stampSec: message.data[0], areaM2: message.data[1] });
    };

    const onConnection = () => {
      if (disposed) {
        ros.close();
        return;
      }
      setStatus("connected");
      setError("");
      jointTopic.subscribe(onJointState);
      debugTopic.subscribe(onDebug);
      rotationTopic.subscribe(onRotationMatrix);
      tactileTopic.subscribe(onTactile);
      tactileContactTopic.subscribe(onTactileContactPoints);
      thumbIndexRingAreaTopic.subscribe(onThumbIndexRingArea);
    };
    const onError = (event: unknown) => {
      if (disposed) return;
      setStatus("error");
      setError(connectionErrorMessage(event));
      scheduleReconnect();
    };
    const onClose = () => {
      if (disposed) return;
      setStatus("disconnected");
      clearTelemetry();
      scheduleReconnect();
    };

    setStatus("connecting");
    setError("");
    ros.on("connection", onConnection);
    ros.on("error", onError);
    ros.on("close", onClose);
    void ros.connect(url).catch((reason: unknown) => {
      if (disposed) {
        ros.close();
        return;
      }
      setStatus("error");
      setError(connectionErrorMessage(reason));
      scheduleReconnect();
    });

    return () => {
      disposed = true;
      if (reconnectTimer !== undefined) window.clearTimeout(reconnectTimer);
      jointTopic.unsubscribe(onJointState);
      debugTopic.unsubscribe(onDebug);
      rotationTopic.unsubscribe(onRotationMatrix);
      tactileTopic.unsubscribe(onTactile);
      tactileContactTopic.unsubscribe(onTactileContactPoints);
      thumbIndexRingAreaTopic.unsubscribe(onThumbIndexRingArea);
      if (activeRos.current === ros) activeRos.current = null;
      publishers.current = { ...EMPTY_PUBLISHERS };
      ros.close();
    };
  }, [attempt, handSide, url]);

  const publish = useCallback((topic: CommandTopic | null, message: CommandMessage): boolean => {
    const ros = activeRos.current;
    if (topic === null || ros === null || !ros.isConnected) return false;
    topic.publish(message);
    return true;
  }, []);

  const setGraspType = useCallback(
    (value: number) => publish(publishers.current.grasp, { data: value }),
    [publish],
  );
  const setPoseType = useCallback(
    (value: number) => publish(publishers.current.pose, { data: value }),
    [publish],
  );
  const setAlpha1 = useCallback(
    (value: number) => publish(publishers.current.alpha, { data: value }),
    [publish],
  );
  const setTeachingMode = useCallback(
    (value: boolean) => publish(publishers.current.teaching, { data: value }),
    [publish],
  );
  const setRotationMatrix = useCallback(
    (value: number[]) => publish(publishers.current.rotation, {
      layout: { dim: [], data_offset: 0 },
      data: value,
    }),
    [publish],
  );
  const setRelativeRotationDegrees = useCallback(
    (value: number) => publish(publishers.current.relativeRotationDegrees, { data: value }),
    [publish],
  );
  const setContinuousRotation = useCallback(
    (value: boolean) => publish(publishers.current.continuousRotation, { data: value }),
    [publish],
  );
  const toggleBlindDirection = useCallback(() => publish(publishers.current.blindDirectionToggle, { data: 1 }), [publish]);
  const setRelativeTranslationWorld = useCallback((deltaMeters: Point3): boolean => {
    const ros = activeRos.current;
    const topic = publishers.current.relativeTranslation;
    if (topic === null || ros === null || !ros.isConnected) return false;
    topic.publish({
      header: {
        stamp: { sec: 0, nanosec: 0 },
        frame_id: "world",
      },
      vector: {
        x: Number(deltaMeters.x),
        y: Number(deltaMeters.y),
        z: Number(deltaMeters.z),
      },
    });
    return true;
  }, []);
  const reconnect = useCallback(() => setAttempt((value) => value + 1), []);

  return {
    status,
    error,
    jointState,
    debug,
    handToWorldRotation,
    lastJointAt,
    lastDebugAt,
    lastRotationAt,
    tactileSamples,
    tactileContactPoints,
    thumbIndexRingArea,
    lastTactileAt,
    reconnect,
    setGraspType,
    setPoseType,
    setAlpha1,
    setTeachingMode,
    setRotationMatrix,
    setRelativeRotationDegrees,
    setContinuousRotation,
    toggleBlindDirection,
    setRelativeTranslationWorld,
  };
}
