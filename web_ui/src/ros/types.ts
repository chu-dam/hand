export interface RosTime {
  sec: number;
  nanosec: number;
}

export interface RosHeader {
  stamp: RosTime;
  frame_id: string;
}

export interface Point3 {
  x: number;
  y: number;
  z: number;
}

export interface JointStateMessage {
  header: RosHeader;
  name: string[];
  position: number[];
  velocity: number[];
  effort: number[];
}

export interface GraspDebugMessage {
  header: RosHeader;
  finger_ids: number[] | string;
  fingertip_positions: Point3[];
  geometric_centroid: Point3;
  virtual_centroid: Point3;
  alpha: number[];
  grasp_forces: Point3[];
  rotation_forces: Point3[];
  center_hold_forces: Point3[];
  collision_forces: Point3[];
  total_forces: Point3[];
  controller_torques: number[];
  commanded_efforts: number[];
  grasp_type: number;
  pose_type: number;
  teaching_mode: boolean;
  controller_state: string;
  controller_phase: string;
}

export type RosConnectionStatus =
  | "connecting"
  | "connected"
  | "disconnected"
  | "error";

export function decodeFingerIds(value: number[] | string): number[] {
  if (Array.isArray(value)) {
    return value.map(Number);
  }

  try {
    const binary = window.atob(value);
    return Array.from(binary, (character) => character.charCodeAt(0));
  } catch {
    return [];
  }
}

export function vectorMagnitude(vector: Point3 | undefined): number {
  if (!vector) return 0;
  return Math.hypot(vector.x, vector.y, vector.z);
}

