#!/usr/bin/env python3

import os
from time import sleep, time

import numpy as np
import rclpy
from ament_index_python.packages import get_package_share_directory
from dg5f_grasp_interfaces.msg import GraspDebug
from geometry_msgs.msg import Vector3Stamped
from sensor_msgs.msg import JointState
from std_msgs.msg import Bool, Float64, Float64MultiArray, Int32

from dg5f_grasp_control.config import RuntimeConfig
from dg5f_grasp_control.control_utils import pose_pd, publish_effort, zero_effort
from dg5f_grasp_control.friction import calc_friction, set_hand_side as set_friction_hand_side
from dg5f_grasp_control.grasp_controller import (
    CARD_INDEX_ID,
    CARD_THUMB_ID,
    GraspController,
    PINKY_FINGER_ID,
)
from dg5f_grasp_control.hand_model import (
    FINGER_JOINT_INDEX,
    HAND_JOINT_NAMES,
    JOINT_COUNT,
)
from dg5f_grasp_control.kinematics import set_hand_side
from dg5f_grasp_control.mujoco_gravity import MujocoGravityCompensator
from dg5f_grasp_control.ros_debug import build_grasp_debug_message


FINGER_NAMES = ("thumb", "index", "middle", "ring", "pinky")


def _default_model_path(hand_side):
    if hand_side == "right":
        share_dir = get_package_share_directory("dg5f_s_description")
        return os.path.join(share_dir, "urdf", "dg5fs_right_w_mount.urdf")

    share_dir = get_package_share_directory("dg5f_grasp_control")
    return os.path.join(share_dir, "models", "dg5fs_left_w_mount.xml")


def _declare_and_load_config(node):
    defaults = RuntimeConfig()
    node.declare_parameter("model_xml_path", "")
    for name, value in defaults.__dict__.items():
        node.declare_parameter(name, value)

    cfg = RuntimeConfig(**{
        name: node.get_parameter(name).value
        for name in defaults.__dict__.keys()
    })
    set_hand_side(cfg.hand_side)
    set_friction_hand_side(cfg.hand_side)
    model_path = node.get_parameter("model_xml_path").value
    return cfg, model_path or _default_model_path(cfg.hand_side)


class GraspRealRunner:
    """ROS/hardware adapter for the shared GraspController."""

    def __init__(self, node, cfg, model_xml_path):
        self.node = node
        self.cfg = cfg
        self.model_xml_path = model_xml_path

        self.hand_q = np.zeros(JOINT_COUNT, dtype=np.float64)
        self.hand_qdot = np.zeros(JOINT_COUNT, dtype=np.float64)
        self.got_state = False
        self.last_joint_state_time = None
        self.pending_finger_count = None
        self.pending_pose_type = None
        self.pending_teaching_mode = None
        self.pending_relative_rotation_rad = None
        self.pending_relative_translation_hand = None
        self.pending_continuous_rotation = None
        self.rotation_hand_to_world = np.eye(3, dtype=np.float64)
        self.gravity_in_hand_frame = None
        self.last_debug_publish_time = 0.0

        self.teaching_mode = cfg.start_teaching_mode
        self.teaching_hold_active = False
        self.teaching_hold_pose = np.zeros(JOINT_COUNT, dtype=np.float64)

        self.controller = GraspController(cfg, log=print)
        self.gravity_comp = MujocoGravityCompensator(model_xml_path)
        self.controller.set_tactile_link_pose_provider(
            self.gravity_comp.tactile_link_pose
        )
        self.controller.set_tactile_contact_geometry_provider(
            self.gravity_comp.tactile_contact_geometry
        )

        self.pub = node.create_publisher(Float64MultiArray, cfg.effort_topic, 10)
        self.debug_pub = node.create_publisher(GraspDebug, cfg.debug_topic, 10)
        self.tactile_contact_pub = node.create_publisher(
            Float64MultiArray,
            f"/dg5f_grasp_control/{cfg.hand_side}/tactile_contact_points",
            10,
        )
        node.create_subscription(JointState, cfg.joint_state_topic, self.joint_cb, 10)
        tactile_topic = cfg.tactile_topic or f"/dg5f_s_{cfg.hand_side}/tactile_contacts"
        node.create_subscription(
            Float64MultiArray,
            tactile_topic,
            self.tactile_cb,
            10,
        )
        node.create_subscription(Int32, cfg.command_topic, self.command_cb, 10)
        node.create_subscription(Int32, cfg.pose_topic, self.pose_type_cb, 10)
        node.create_subscription(Float64, cfg.alpha1_topic, self.alpha1_cb, 10)
        node.create_subscription(
            Float64,
            cfg.relative_rotation_deg_topic,
            self.relative_rotation_deg_cb,
            10,
        )
        node.create_subscription(
            Vector3Stamped,
            cfg.relative_translation_topic,
            self.relative_translation_cb,
            10,
        )
        node.create_subscription(
            Bool,
            cfg.teaching_mode_topic,
            self.teaching_mode_cb,
            10,
        )
        node.create_subscription(
            Bool,
            cfg.continuous_rotation_topic,
            self.continuous_rotation_cb,
            10,
        )
        node.create_subscription(
            Int32,
            f"/dg5f_grasp_control/{cfg.hand_side}/blind_direction_toggle",
            self.blind_direction_toggle_cb,
            10,
        )
        node.create_subscription(
            Float64MultiArray,
            cfg.rotation_matrix_topic,
            self.rotation_matrix_cb,
            10,
        )
        node.create_subscription(
            Float64MultiArray,
            f"/dg5f_grasp_control/{cfg.hand_side}/ui_sphere_center_world",
            self.ui_sphere_center_world_cb,
            10,
        )

    def joint_cb(self, msg):
        now = time()
        positions = dict(zip(msg.name, msg.position))
        new_q = self.hand_q.copy()
        for index, name in enumerate(HAND_JOINT_NAMES):
            if name in positions:
                new_q[index] = positions[name]

        velocities = dict(zip(msg.name, msg.velocity))
        qdot_raw = None
        if all(name in velocities for name in HAND_JOINT_NAMES):
            qdot_raw = np.array(
                [velocities[name] for name in HAND_JOINT_NAMES],
                dtype=np.float64,
            )

        if qdot_raw is not None:
            qdot_raw = np.array(
                qdot_raw,
                dtype=np.float64,
            )
            if not np.all(np.isfinite(qdot_raw)):
                qdot_raw = None

        if qdot_raw is not None:
            alpha = float(np.clip(self.cfg.qdot_alpha, 0.0, 1.0))
            self.hand_qdot = (
                (1.0 - alpha) * self.hand_qdot + alpha * qdot_raw
            )
        else:
            self.hand_qdot.fill(0.0)

        self.hand_q = new_q
        self.got_state = True
        self.last_joint_state_time = now

    def tactile_cb(self, msg):
        data = np.asarray(msg.data, dtype=np.float64)
        if data.size < 25 or not np.all(np.isfinite(data[:25])):
            return
        self.controller.set_tactile_contacts(data[:25].reshape(5, 5))

    def ui_sphere_center_world_cb(self, msg):
        data = np.asarray(msg.data, dtype=np.float64)
        if data.size < 3 or not np.all(np.isfinite(data[:3])):
            return
        self.controller.set_ui_sphere_center_world(data[:3])

    def command_cb(self, msg):
        if self.teaching_mode or self.pending_teaching_mode is True:
            self.node.get_logger().warn(
                "Ignore grasp_type command while Teaching Mode is active."
            )
            return

        command = int(msg.data)
        if command < -1 or command > 7:
            self.node.get_logger().warn(
                f"Ignore invalid grasp_type command: {command}. "
                "Use -1, 0, 1, 2, 3, 4, 5, 6, or 7."
            )
            return
        self.pending_finger_count = command

    def pose_type_cb(self, msg):
        if self.teaching_mode or self.pending_teaching_mode is True:
            self.node.get_logger().warn(
                "Ignore pose_type command while Teaching Mode is active."
            )
            return

        pose_type = int(msg.data)
        if pose_type not in self.controller.pose_type_targets:
            valid = ", ".join(map(str, self.controller.pose_type_targets))
            self.node.get_logger().warn(
                f"Ignore invalid pose_type command: {pose_type}. Use {valid}."
            )
            return
        self.pending_pose_type = pose_type

    def teaching_mode_cb(self, msg):
        enable = bool(msg.data)
        self.pending_teaching_mode = enable
        if enable:
            self.pending_relative_rotation_rad = None

    def continuous_rotation_cb(self, msg):
        enable = bool(msg.data)
        if enable and (
            self.teaching_mode or self.pending_teaching_mode is True
        ):
            self.node.get_logger().warn(
                "Ignore continuous rotation while Teaching Mode is active."
            )
            return
        self.pending_continuous_rotation = enable

    def blind_direction_toggle_cb(self, msg):
        if int(msg.data):
            self.controller.request_blind_direction_change()

    def alpha1_cb(self, msg):
        if self.controller.continuous_rotation_active:
            self.node.get_logger().warn(
                "Ignore alpha1 command while continuous rotation is active."
            )
            return
        alpha1 = float(msg.data)
        try:
            self.controller.set_alpha1(alpha1)
        except ValueError as exc:
            self.node.get_logger().warn(f"Ignore invalid alpha1 command: {exc}")
            return
        self.cfg = self.controller.cfg
        self.node.get_logger().info(f"Updated alpha1: {alpha1:.4f}")

    def relative_rotation_deg_cb(self, msg):
        if self.teaching_mode or self.pending_teaching_mode is True:
            self.node.get_logger().warn(
                "Ignore relative rotation command while Teaching Mode is active."
            )
            return
        if self.teaching_hold_active:
            self.node.get_logger().warn(
                "Ignore relative rotation command during Teaching Hold. "
                "Send a grasp_type command first."
            )
            return

        angle_deg = float(msg.data)
        if not np.isfinite(angle_deg) or angle_deg == 0.0:
            self.node.get_logger().warn(
                f"Ignore invalid relative rotation angle: {angle_deg}. "
                "Use a finite, non-zero value in degrees."
            )
            return

        self.pending_relative_rotation_rad = float(np.deg2rad(angle_deg))
        self.pending_relative_translation_hand = None
        self.node.get_logger().info(
            f"Queued relative rotation: {angle_deg:.4f} deg "
            f"({self.pending_relative_rotation_rad:.6f} rad)"
        )

    def relative_translation_cb(self, msg):
        if self.teaching_mode or self.pending_teaching_mode is True:
            self.node.get_logger().warn(
                "Ignore relative translation command while Teaching Mode is active."
            )
            return
        if self.teaching_hold_active:
            self.node.get_logger().warn(
                "Ignore relative translation command during Teaching Hold. "
                "Send a grasp_type command first."
            )
            return

        delta = np.array(
            [msg.vector.x, msg.vector.y, msg.vector.z],
            dtype=np.float64,
        )
        if not np.all(np.isfinite(delta)):
            self.node.get_logger().warn(
                "Ignore relative translation command: vector must be finite."
            )
            return

        frame_id = str(msg.header.frame_id).strip()
        if frame_id == "world":
            delta_hand = self.rotation_hand_to_world.T @ delta
        elif frame_id == self.cfg.debug_frame_id:
            delta_hand = delta
        else:
            self.node.get_logger().warn(
                "Ignore relative translation command: frame_id must be 'world' "
                f"or '{self.cfg.debug_frame_id}', got '{frame_id or '<empty>'}'."
            )
            return

        distance = float(np.linalg.norm(delta_hand))
        maximum = float(self.cfg.relative_translation_max_m)
        if distance <= 0.0 or distance > maximum + 1e-12:
            self.node.get_logger().warn(
                "Ignore relative translation command: norm must be within "
                f"(0, {maximum:.6f}] m, got {distance:.6f} m."
            )
            return

        self.pending_relative_rotation_rad = None
        self.pending_relative_translation_hand = delta_hand.copy()
        self.node.get_logger().info(
            "Queued relative Cartesian translation: "
            f"frame={frame_id}, input_mm={np.round(1000.0 * delta, 3).tolist()}, "
            f"link_base_mm={np.round(1000.0 * delta_hand, 3).tolist()}"
        )

    def rotation_matrix_cb(self, msg):
        values = np.asarray(msg.data, dtype=np.float64)
        if values.size != 9 or not np.all(np.isfinite(values)):
            self.node.get_logger().warn(
                f"Ignore invalid rotation matrix command with {values.size} values. "
                "Send 9 finite values."
            )
            return

        rotation_hand_to_world = values.reshape(3, 3)
        self.rotation_hand_to_world = rotation_hand_to_world.copy()
        self.controller.set_rotation_hand_to_world(rotation_hand_to_world)
        gravity_world = np.array([0.0, 0.0, -9.81], dtype=np.float64)
        self.gravity_in_hand_frame = rotation_hand_to_world.T @ gravity_world
        self.node.get_logger().info(
            "Updated hand gravity vector: "
            f"{np.round(self.gravity_in_hand_frame, 4).tolist()}"
        )

    def _apply_pending_teaching_mode(self):
        if self.pending_teaching_mode is None:
            return

        enable = self.pending_teaching_mode
        self.pending_teaching_mode = None

        if enable:
            if self.teaching_mode:
                return

            self.teaching_mode = True
            self.teaching_hold_active = False
            self.pending_finger_count = None
            self.pending_pose_type = None
            self.pending_relative_rotation_rad = None
            self.pending_relative_translation_hand = None
            self.pending_continuous_rotation = None
            self.controller.cancel_continuous_rotation()
            self.controller.cancel_relative_rotation()
            self.controller.cancel_relative_translation()
            self.controller.cancel_card_grasp()
            print(
                "[TEACHING_MODE] ON -> gravity + friction compensation only; "
                "grasp/pose commands are ignored"
            )
            return

        if not self.teaching_mode:
            return

        self.teaching_mode = False
        self.teaching_hold_pose = self.hand_q.copy()
        self.teaching_hold_active = True
        print(
            "[TEACHING_MODE] OFF -> capture current pose and hold by PD; "
            "waiting for /grasp_type or /pose_type"
        )
        self._print_joint_values(self.teaching_hold_pose, label="TEACHING_HOLD_TARGET")

    def _apply_pending_commands(self, now):
        pose_or_grasp_command_received = (
            self.pending_pose_type is not None
            or self.pending_finger_count is not None
        )
        if pose_or_grasp_command_received and self.teaching_hold_active:
            self.teaching_hold_active = False
            print("[TEACHING_HOLD] released by new grasp/pose command")

        if self.pending_pose_type is not None:
            pose_type = self.pending_pose_type
            self.pending_pose_type = None
            self.controller.apply_pose_type(pose_type, now)

        if self.pending_finger_count is not None:
            command = self.pending_finger_count
            self.pending_finger_count = None
            self.controller.apply_grasp_type(command, now)

        if self.pending_continuous_rotation is not None:
            enable = self.pending_continuous_rotation
            self.pending_continuous_rotation = None
            if enable:
                if self.controller.start_continuous_rotation(now):
                    self.node.get_logger().info(
                        "Continuous rotation sequence started."
                    )
                else:
                    self.node.get_logger().warn(
                        "Continuous rotation requires right-hand "
                        "Pre-rotation or Pre-rotation (Blind Grasping) pose."
                    )
            else:
                self.controller.stop_continuous_rotation(now)

        if self.pending_relative_rotation_rad is not None:
            angle_rad = self.pending_relative_rotation_rad
            self.pending_relative_rotation_rad = None
            angle_deg = float(np.rad2deg(angle_rad))
            try:
                accepted = self.controller.prepare_relative_rotation(angle_rad, now)
            except ValueError as exc:
                self.node.get_logger().warn(
                    f"Relative rotation target rejected: {exc}"
                )
                return
            if accepted:
                self.node.get_logger().info(
                    f"Stored relative rotation target: {angle_deg:.4f} deg"
                )
            else:
                self.node.get_logger().warn(
                    f"Relative rotation target rejected: {angle_deg:.4f} deg"
                )

        if self.pending_relative_translation_hand is not None:
            delta_hand = self.pending_relative_translation_hand.copy()
            self.pending_relative_translation_hand = None
            try:
                accepted = self.controller.prepare_relative_translation(
                    delta_hand,
                    now,
                )
            except ValueError as exc:
                self.node.get_logger().warn(
                    f"Relative translation target rejected: {exc}"
                )
                return
            delta_mm = np.round(1000.0 * delta_hand, 3).tolist()
            if accepted:
                self.node.get_logger().info(
                    "Started relative Cartesian translation: "
                    f"link_base_mm={delta_mm}"
                )
            else:
                self.node.get_logger().warn(
                    "Relative translation target rejected: "
                    f"link_base_mm={delta_mm}"
                )

    def _print_start_info(self):
        print("[START] hand groped grasp with shared controller + gravity + friction compensation")
        print("[XML]", self.model_xml_path)
        print(f"[GRASP_TYPE_TOPIC] {self.cfg.command_topic}")
        print(f"[POSE_TYPE_TOPIC] {self.cfg.pose_topic}")
        print(f"[ALPHA1_TOPIC] {self.cfg.alpha1_topic}")
        print(f"[CONTINUOUS_ROTATION_TOPIC] {self.cfg.continuous_rotation_topic}")
        print(f"[RELATIVE_ROTATION_DEG_TOPIC] {self.cfg.relative_rotation_deg_topic}")
        print(f"[RELATIVE_TRANSLATION_TOPIC] {self.cfg.relative_translation_topic}")
        print(f"[ROTATION_MATRIX_TOPIC] {self.cfg.rotation_matrix_topic}")
        print(f"[TEACHING_MODE_TOPIC] {self.cfg.teaching_mode_topic}")
        print(
            f"[DEBUG_TOPIC] {self.cfg.debug_topic} "
            f"({self.cfg.debug_publish_hz:.1f} Hz, frame={self.cfg.debug_frame_id})"
        )
        print(
            "[GRASP_TYPE] -1=normal, 0=selected pre-grasp, "
            "1=thumb+index, 2=thumb+middle, 3=thumb+index+middle, "
            "4=+ring, 5=+pinky, 6=envelop-grasp, 7=card-grasp"
        )
        print(
            "[POSE_TYPE] 1=normal, 2=default pre-grasp, "
            "3=compact pre-grasp, 4=card pre-grasp"
        )
        print(f"[ALPHA1] {self.cfg.alpha1}")
        print(
            "[POSE_KP/KD/LIMIT] "
            f"{self.cfg.pose_kp}, {self.cfg.pose_kd}, {self.cfg.pose_pd_limit}"
        )

    @staticmethod
    def _print_joint_values(q, label="CURRENT_HAND_JOINT_VALUES"):
        q = np.asarray(q, dtype=np.float64)
        q_rounded = np.round(q, 4)

        print("")
        print("=====================================================")
        print(f"[{label}]")
        print("HAND_TARGET = np.array([")
        for finger_index, finger_name in enumerate(FINGER_NAMES):
            start = finger_index * 4
            values = q_rounded[start:start + 4]
            print(
                f"    {values[0]: .4f}, {values[1]: .4f}, "
                f"{values[2]: .4f}, {values[3]: .4f},    # {finger_name}"
            )
        print("], dtype=np.float64)")
        print("=====================================================")

    def _print_teaching_status(self, gravity, friction, effort, qdot):
        print(
            "[TEACHING_MODE] "
            f"q_max={np.max(np.abs(self.hand_q)):.3f} | "
            f"qdot_max={np.max(np.abs(qdot)):.3f} | "
            f"gc_max={np.max(np.abs(gravity)):.3f} | "
            f"fric_max={np.max(np.abs(friction)):.3f} | "
            f"effort_max={np.max(np.abs(effort)):.3f}"
        )
        self._print_joint_values(self.hand_q)

    def _print_teaching_hold_status(self, hold_err, hold_pd, gravity, friction, effort, qdot):
        print(
            "[TEACHING_HOLD] "
            f"err_max={np.max(np.abs(hold_err)):.4f} | "
            f"qdot_max={np.max(np.abs(qdot)):.4f} | "
            f"pd_max={np.max(np.abs(hold_pd)):.4f} | "
            f"gc_max={np.max(np.abs(gravity)):.3f} | "
            f"fric_max={np.max(np.abs(friction)):.3f} | "
            f"effort_max={np.max(np.abs(effort)):.3f}"
        )

    def _maybe_publish_debug(
        self,
        now,
        output,
        controller_torques,
        commanded_efforts,
        *,
        controller_state=None,
        controller_phase=None,
    ):
        publish_hz = float(self.cfg.debug_publish_hz)
        if publish_hz <= 0.0:
            return
        if now - self.last_debug_publish_time < 1.0 / publish_hz:
            return

        self.last_debug_publish_time = now
        message = build_grasp_debug_message(
            controller=self.controller,
            q=self.hand_q,
            output=output,
            controller_torques=controller_torques,
            commanded_efforts=commanded_efforts,
            stamp=self.node.get_clock().now().to_msg(),
            frame_id=self.cfg.debug_frame_id,
            teaching_mode=self.teaching_mode,
            controller_state=controller_state,
            controller_phase=controller_phase,
        )
        self.debug_pub.publish(message)
        points = Float64MultiArray()
        points.data = [
            float(v)
            for finger in range(1, 6)
            for v in self.controller.tactile_contact_points.get(
                finger, np.zeros(3, dtype=np.float64)
            )
        ]
        self.tactile_contact_pub.publish(points)

    def _print_status(self, output, gravity, friction, effort, qdot):
        if output.state == "GROPED_GRASP":
            pinky_j1 = int(FINGER_JOINT_INDEX[PINKY_FINGER_ID][0])
            pinky_target = output.inactive_pd_target[pinky_j1]
            target_text = "nan" if np.isnan(pinky_target) else f"{pinky_target:.4f}"
            print(
                "[PINKY_J1_DEBUG] "
                f"active_count={output.active_finger_count} | "
                f"use_fingers={output.use_fingers} | "
                f"target={target_text} | "
                f"q={self.hand_q[pinky_j1]:.4f} | "
                f"qdot={qdot[pinky_j1]:.4f} | "
                f"gravity={gravity[pinky_j1]:.4f} | "
                f"friction={friction[pinky_j1]:.4f} | "
                f"inactive_pd={output.inactive_pd[pinky_j1]:.4f} | "
                f"grasp_tau={output.grasp_tau[pinky_j1]:.4f} | "
                f"final_effort={effort[pinky_j1]:.4f}"
            )
            alpha_text = {finger: round(value, 4) for finger, value in output.alpha.items()}
            print(
                f"[{output.state}] t={output.state_elapsed:.2f} | "
                f"alpha={alpha_text} | "
                f"cg={np.round(output.cg, 4)} | "
                f"cv={np.round(output.cv, 4)} | "
                f"grasp_tau_max={np.max(np.abs(output.grasp_tau)):.4f} | "
                f"relative_rotation_phase={output.relative_rotation_phase} | "
                f"relative_target_deg="
                f"{np.degrees(output.relative_rotation_target_rad):.3f} | "
                f"relative_current_deg="
                f"{np.degrees(output.relative_rotation_current_rad):.3f} | "
                f"relative_error_deg="
                f"{np.degrees(output.relative_rotation_error_rad):.3f} | "
                f"rotation_moment="
                f"{output.relative_rotation_command_moment:.4f} | "
                f"translation_phase={output.relative_translation_phase} | "
                f"translation_error_mm="
                f"{np.round(1000.0 * output.relative_translation_error, 3)} | "
                f"translation_force="
                f"{np.round(output.relative_translation_command_force, 3)} | "
                f"translation_tau_max="
                f"{np.max(np.abs(output.translation_torques)):.4f} | "
                f"cv_bias={output.effective_thumb_centroid_bias:.4f} | "
                f"force_balance="
                f"{output.relative_rotation_force_balance_blend:.3f} | "
                f"use_fingers={output.use_fingers} | "
                f"effort_max={np.max(np.abs(effort)):.3f}"
            )
        elif output.state == "CARD_GRASP":
            now = time()
            phase = self.controller.card_phase
            phase_elapsed = (
                0.0
                if self.controller.card_phase_started_at is None
                else max(
                    0.0,
                    now - float(self.controller.card_phase_started_at),
                )
            )
            timeout = {
                "card_floor_contact": self.cfg.card_floor_timeout_sec,
                "card_pinch": self.cfg.card_pinch_timeout_sec,
            }.get(phase)
            required_stable = {
                "card_floor_contact": self.cfg.card_floor_stall_sec,
                "card_pinch": self.cfg.card_pinch_stall_sec,
            }.get(phase)
            force_norms = {
                finger: float(
                    np.linalg.norm(
                        output.total_forces.get(
                            finger,
                            np.zeros(3, dtype=np.float64),
                        )
                    )
                )
                for finger in (CARD_THUMB_ID, CARD_INDEX_ID)
            }
            floor_contact = self.controller.card_floor_contact_detected
            floor_stable = {
                finger: round(value, 3)
                for finger, value in self.controller.card_floor_stable_elapsed_sec.items()
            }
            floor_motion_mm = {
                finger: round(1000.0 * value, 3)
                for finger, value in self.controller.card_floor_motion_m.items()
            }
            print(
                "[CARD_GRASP] "
                f"phase={phase} | "
                f"phase_t={phase_elapsed:.3f}/"
                f"{'hold' if timeout is None else f'{float(timeout):.3f}'}s | "
                f"stable={self.controller.card_stable_elapsed_sec:.3f}/"
                f"{'-' if required_stable is None else f'{float(required_stable):.3f}'}s | "
                "stall_motion_mm="
                f"{1000.0 * self.controller.card_stall_max_motion_m:.3f}/"
                f"{1000.0 * float(self.cfg.card_tip_stall_threshold_m):.3f} | "
                f"floor_contact={floor_contact} | "
                f"floor_stable={floor_stable} | "
                f"floor_motion_mm={floor_motion_mm} | "
                "thumb_j1_hold="
                f"{self.controller.card_thumb_j1_hold_target}/"
                f"{self.controller.card_thumb_j1_hold_error_rad:.4f}rad/"
                f"{self.controller.card_thumb_j1_hold_tau:.4f} | "
                f"F_thumb={force_norms[CARD_THUMB_ID]:.3f}N | "
                f"F_index={force_norms[CARD_INDEX_ID]:.3f}N | "
                f"tau_max={np.max(np.abs(output.tau)):.4f} | "
                f"effort_max={np.max(np.abs(effort)):.4f}"
            )
        elif output.state == "ENVELOP_GRASP":
            info = output.envelop_info
            print(
                f"[{output.state}] t={output.state_elapsed:.2f} | "
                f"tau_level={info.get('tau_level', 0.0):.4f} | "
                f"active_non_thumb={info.get('active_non_thumb', [])} | "
                f"thumb_enabled={info.get('thumb_enabled', False)} | "
                f"active_thumb={info.get('active_thumb', [])} | "
                f"effort_max={np.max(np.abs(effort)):.3f}"
            )
        else:
            print(
                f"[{output.state}] t={output.state_elapsed:.2f} | "
                f"err_max={np.max(np.abs(output.err)):.4f} | "
                f"qdot_max={np.max(np.abs(qdot)):.4f} | "
                f"gc_max={np.max(np.abs(gravity)):.3f} | "
                f"fric_max={np.max(np.abs(friction)):.3f} | "
                f"effort_max={np.max(np.abs(effort)):.3f}"
            )

    def run(self):
        qdot = np.zeros(JOINT_COUNT, dtype=np.float64)
        last_log = 0.0

        self._print_start_info()

        try:
            while rclpy.ok():
                loop_start = time()
                rclpy.spin_once(self.node, timeout_sec=0.0)

                if not self.got_state:
                    publish_effort(self.pub, np.zeros(JOINT_COUNT, dtype=np.float64))
                    sleep(0.05)
                    continue

                now = time()
                qdot = self.hand_qdot.copy()

                self.controller.sync_joint_state(self.hand_q)
                if (
                    self.controller.state == "CARD_GRASP"
                    and (
                        self.last_joint_state_time is None
                        or now - float(self.last_joint_state_time)
                        > float(self.cfg.card_joint_state_timeout_sec)
                    )
                ):
                    self.node.get_logger().error(
                        "CARD grasp stopped: JointState is stale."
                    )
                    self.controller.apply_grasp_type(0, now)
                self._apply_pending_teaching_mode()
                if not self.teaching_mode:
                    self._apply_pending_commands(now)

                gravity = self.gravity_comp.compute(
                    self.hand_q,
                    gravity=self.gravity_in_hand_frame,
                )
                friction = calc_friction(
                    qdot,
                    scale=self.cfg.fric_scale,
                    tanh_k=self.cfg.fric_tanh_k,
                    limit=self.cfg.fric_limit,
                )

                output = None
                hold_pd = np.zeros(JOINT_COUNT, dtype=np.float64)
                hold_err = np.zeros(JOINT_COUNT, dtype=np.float64)

                if self.teaching_mode:
                    # Teaching Mode: remove all grasp/pose control and leave only
                    # gravity and friction compensation for manual hand motion.
                    effort = gravity + friction
                    controller_torques = np.zeros(JOINT_COUNT, dtype=np.float64)
                elif self.teaching_hold_active:
                    # After Teaching Mode is turned off, capture and hold the
                    # taught pose until a new grasp_type or pose_type arrives.
                    hold_pd, hold_err = pose_pd(
                        self.teaching_hold_pose,
                        self.hand_q,
                        qdot,
                        kp=self.cfg.pose_kp,
                        kd=self.cfg.pose_kd,
                        limit=self.cfg.pose_pd_limit,
                    )
                    effort = gravity + friction + hold_pd
                    controller_torques = hold_pd
                else:
                    output = self.controller.step(self.hand_q, qdot, now)
                    effort = gravity + friction + output.tau
                    controller_torques = output.tau

                effort = np.clip(
                    effort,
                    -self.cfg.hand_limit,
                    self.cfg.hand_limit,
                )
                publish_effort(self.pub, effort)

                if self.teaching_mode:
                    debug_state = "TEACHING_MODE"
                    debug_phase = "active"
                elif self.teaching_hold_active:
                    debug_state = "TEACHING_HOLD"
                    debug_phase = "holding"
                else:
                    debug_state = None
                    debug_phase = None
                self._maybe_publish_debug(
                    now,
                    output,
                    controller_torques,
                    effort,
                    controller_state=debug_state,
                    controller_phase=debug_phase,
                )

                if now - last_log >= self.cfg.log_dt:
                    last_log = now
                    if self.teaching_mode:
                        self._print_teaching_status(gravity, friction, effort, qdot)
                    elif self.teaching_hold_active:
                        self._print_teaching_hold_status(
                            hold_err,
                            hold_pd,
                            gravity,
                            friction,
                            effort,
                            qdot,
                        )
                    else:
                        self._print_status(output, gravity, friction, effort, qdot)

                sleep_time = self.cfg.dt - (time() - loop_start)
                if sleep_time > 0.0:
                    sleep(sleep_time)
        finally:
            print("[STOP] zero effort")
            zero_effort(self.pub)


def main(args=None):
    rclpy.init(args=args)
    node = rclpy.create_node("grasp_real")
    try:
        cfg, model_xml_path = _declare_and_load_config(node)
        GraspRealRunner(node, cfg, model_xml_path).run()
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
