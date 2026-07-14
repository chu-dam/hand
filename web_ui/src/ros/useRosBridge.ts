import { useCallback, useEffect, useRef, useState } from "react";
import { Ros, Topic } from "roslib";

import type {
  GraspDebugMessage,
  JointStateMessage,
  RosConnectionStatus,
} from "./types";

const JOINT_STATE_TOPIC = "/dg5f_s_left/joint_states";
const DEBUG_TOPIC = "/dg5f_grasp_control/debug";
const GRASP_TOPIC = "/grasp_type";
const POSE_TOPIC = "/pose_type";
const ALPHA_TOPIC = "/dg5f_grasp_control/alpha1_cmd";
const TEACHING_TOPIC = "/dg5f_grasp_control/teaching_mode";
const ROTATION_TOPIC = "/dg5f_grasp_control/rotation_matrix_cmd";

const RECONNECT_DELAY_MS = 2_000;
const JOINT_RENDER_PERIOD_MS = 33;

interface CommandMessage {
  data: unknown;
  layout?: {
    dim: never[];
    data_offset: number;
  };
}

type CommandTopic = Topic<CommandMessage>;

interface Publishers {
  grasp: CommandTopic | null;
  pose: CommandTopic | null;
  alpha: CommandTopic | null;
  teaching: CommandTopic | null;
  rotation: CommandTopic | null;
}

const EMPTY_PUBLISHERS: Publishers = {
  grasp: null,
  pose: null,
  alpha: null,
  teaching: null,
  rotation: null,
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

export function useRosBridge(url: string) {
  const [status, setStatus] = useState<RosConnectionStatus>("connecting");
  const [error, setError] = useState("");
  const [jointState, setJointState] = useState<JointStateMessage | null>(null);
  const [debug, setDebug] = useState<GraspDebugMessage | null>(null);
  const [lastJointAt, setLastJointAt] = useState<number | null>(null);
  const [lastDebugAt, setLastDebugAt] = useState<number | null>(null);
  const [attempt, setAttempt] = useState(0);

  const activeRos = useRef<Ros | null>(null);
  const publishers = useRef<Publishers>({ ...EMPTY_PUBLISHERS });

  useEffect(() => {
    let disposed = false;
    let reconnectTimer: number | undefined;
    let lastJointRenderAt = 0;
    const ros = new Ros();
    activeRos.current = ros;

    const clearTelemetry = () => {
      setJointState(null);
      setDebug(null);
      setLastJointAt(null);
      setLastDebugAt(null);
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
      name: JOINT_STATE_TOPIC,
      messageType: "sensor_msgs/msg/JointState",
      throttle_rate: JOINT_RENDER_PERIOD_MS,
      queue_length: 1,
      reconnect_on_close: false,
    });
    const debugTopic = new Topic<GraspDebugMessage>({
      ros,
      name: DEBUG_TOPIC,
      messageType: "dg5f_grasp_interfaces/msg/GraspDebug",
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
      grasp: commandTopic(GRASP_TOPIC, "std_msgs/msg/Int32"),
      pose: commandTopic(POSE_TOPIC, "std_msgs/msg/Int32"),
      alpha: commandTopic(ALPHA_TOPIC, "std_msgs/msg/Float64"),
      teaching: commandTopic(TEACHING_TOPIC, "std_msgs/msg/Bool"),
      rotation: commandTopic(ROTATION_TOPIC, "std_msgs/msg/Float64MultiArray"),
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

    const onConnection = () => {
      if (disposed) {
        ros.close();
        return;
      }
      setStatus("connected");
      setError("");
      jointTopic.subscribe(onJointState);
      debugTopic.subscribe(onDebug);
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
      if (activeRos.current === ros) activeRos.current = null;
      publishers.current = { ...EMPTY_PUBLISHERS };
      ros.close();
    };
  }, [attempt, url]);

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
  const reconnect = useCallback(() => setAttempt((value) => value + 1), []);

  return {
    status,
    error,
    jointState,
    debug,
    lastJointAt,
    lastDebugAt,
    reconnect,
    setGraspType,
    setPoseType,
    setAlpha1,
    setTeachingMode,
    setRotationMatrix,
  };
}
