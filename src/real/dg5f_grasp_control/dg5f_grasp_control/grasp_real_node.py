import os
from dataclasses import replace
from time import sleep, time

import numpy as np
import rclpy
from ament_index_python.packages import get_package_share_directory
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64, Float64MultiArray, Int32

from dg5f_grasp_control.config import RuntimeConfig
from dg5f_grasp_control.control_utils import pose_pd, publish_effort, zero_effort
from dg5f_grasp_control.friction import calc_friction
from dg5f_grasp_control.grasp_policy import GraspPolicy
from dg5f_grasp_control.hand_model import (
    FINGER_JOINT_INDEX,
    HAND_JOINT_NAMES,
    JOINT_COUNT,
    selected_fingers,
)
from dg5f_grasp_control.mujoco_gravity import MujocoGravityCompensator
from dg5f_grasp_control.poses import HAND_NORMAL_POSE, HAND_PRE_GRASP_POSE

FINGER_SWITCH_VIA_THREE_DELAY = 0.5
ADD_FINGER_PRE_GRASP_DELAY = 0.15
ADD_FINGER_PRE_GRASP_KP_SCALE = 1.4
ADD_FINGER_PRE_GRASP_KD_SCALE = 1.2
ADD_FINGER_BLEND_TIME = 0.7

# finger_count_cmd=6: enveloping grasp.
# Finger ids follow FINGER_JOINT_INDEX: 1=thumb, 2=index, 3=middle, 4=ring, 5=pinky.
# Joint-stage sequence:
#   stage 0: non-thumb 2nd joints
#   stage 1: non-thumb 3rd joints + thumb 3rd joint
#   stage 2: non-thumb 4th joints + thumb 4th joint
ENVELOP_FINGER_ORDER = [2, 3, 4, 5]
ENVELOP_FINGER_TORQUE_LOCAL_JOINTS = [1, 2, 3]  # local joints 2~4 of each non-thumb finger
ENVELOP_THUMB_TORQUE_LOCAL_JOINTS = [2, 3]      # thumb joints 3~4
ENVELOP_NON_THUMB_TAU_SIGN = 1.0
ENVELOP_THUMB_TAU_SIGN = -1.0


def _default_model_path():
    share_dir = get_package_share_directory("dg5f_grasp_control")
    return os.path.join(share_dir, "models", "dg5fs_left_w_mount.xml")


def _declare_and_load_config(node):
    defaults = RuntimeConfig()

    node.declare_parameter("model_xml_path", _default_model_path())

    for name, value in defaults.__dict__.items():
        node.declare_parameter(name, value)

    cfg = RuntimeConfig(**{
        name: node.get_parameter(name).value
        for name in defaults.__dict__.keys()
    })
    model_xml_path = node.get_parameter("model_xml_path").value
    return cfg, model_xml_path


class GraspRealRunner:
    def __init__(self, node, cfg, model_xml_path):
        self.node = node
        self.cfg = cfg
        self.model_xml_path = model_xml_path

        self.hand_q = np.zeros(JOINT_COUNT, dtype=np.float64)
        self.got_state = False
        self.pending_finger_count = None
        self.gravity_in_hand_frame = None
        self.active_finger_count = 0
        self.deferred_finger_count = None
        self.deferred_finger_count_at = None
        self.adding_finger_target_count = None
        self.adding_pre_grasp_fingers = []
        self.adding_ready_at = None
        self.skip_add_pre_grasp_once = False
        self.blending_fingers = []
        self.blend_started_at = None
        self.blend_until = None

        self.envelop_hold_pose = None
        self.envelop_started_at = None
        self.envelop_thumb_enabled = False
        self.envelop_thumb_start_at = None
        self.envelop_last_joint_stall_since = None
        self.envelop_last_info = {}

        initial_count = cfg.use_finger_count if 1 <= cfg.use_finger_count <= 5 else 1
        self.use_fingers = selected_fingers(initial_count)
        self.policy = GraspPolicy(self.use_fingers, cfg)
        self.gravity_comp = MujocoGravityCompensator(model_xml_path)

        self.pub = node.create_publisher(Float64MultiArray, cfg.effort_topic, 10)
        node.create_subscription(JointState, cfg.joint_state_topic, self.joint_cb, 10)
        node.create_subscription(Int32, cfg.command_topic, self.command_cb, 10)
        node.create_subscription(Float64, cfg.alpha1_topic, self.alpha1_cb, 10)
        node.create_subscription(
            Float64MultiArray,
            cfg.rotation_matrix_topic,
            self.rotation_matrix_cb,
            10,
        )

    def joint_cb(self, msg):
        pos = dict(zip(msg.name, msg.position))

        for i, name in enumerate(HAND_JOINT_NAMES):
            if name in pos:
                self.hand_q[i] = pos[name]

        self.got_state = True

    def command_cb(self, msg):
        count = int(msg.data)
        if count < -1 or count > 6:
            self.node.get_logger().warn(
                f"Ignore invalid finger_count command: {count}. "
                "Use -1, 0, 1, 2, 3, 4, 5, or 6."
            )
            return

        self.pending_finger_count = count

    def alpha1_cb(self, msg):
        alpha1 = float(msg.data)
        if alpha1 < 0.0:
            self.node.get_logger().warn(
                f"Ignore invalid alpha1 command: {alpha1}. Use alpha1 >= 0.0."
            )
            return

        self.cfg = replace(self.cfg, alpha1=alpha1)
        self.policy = GraspPolicy(self.use_fingers, self.cfg)
        self.node.get_logger().info(f"Updated alpha1: {alpha1:.4f}")

    def rotation_matrix_cb(self, msg):
        values = np.asarray(msg.data, dtype=np.float64)
        if values.size != 9:
            self.node.get_logger().warn(
                f"Ignore rotation matrix command with {values.size} values. Send 9 values."
            )
            return

        r_hand_to_world = values.reshape(3, 3)
        gravity_world = np.array([0.0, 0.0, -9.81], dtype=np.float64)
        self.gravity_in_hand_frame = r_hand_to_world.T @ gravity_world
        self.node.get_logger().info(
            "Updated hand gravity vector: "
            f"{np.round(self.gravity_in_hand_frame, 4).tolist()}"
        )

    def _inactive_pre_grasp_pd(self, active_fingers, qdot, pre_grasp_fingers=None):
        inactive_pd = np.zeros(JOINT_COUNT, dtype=np.float64)
        active_set = set(active_fingers)
        pre_grasp_set = set(pre_grasp_fingers or [])

        for finger, idxs in FINGER_JOINT_INDEX.items():
            if finger in active_set:
                continue

            idxs = np.asarray(idxs, dtype=int)
            if finger in pre_grasp_set:
                target = HAND_PRE_GRASP_POSE[idxs]
                kp = self.cfg.pose_kp * ADD_FINGER_PRE_GRASP_KP_SCALE
                kd = self.cfg.pose_kd * ADD_FINGER_PRE_GRASP_KD_SCALE
            else:
                target = np.zeros(len(idxs), dtype=np.float64)
                kp = self.cfg.pose_kp
                kd = self.cfg.pose_kd

            pd, _ = pose_pd(
                target,
                self.hand_q[idxs],
                qdot[idxs],
                kp=kp,
                kd=kd,
                limit=self.cfg.pose_pd_limit,
            )
            inactive_pd[idxs] = pd

        return inactive_pd

    def _clear_transition_state(self):
        self.deferred_finger_count = None
        self.deferred_finger_count_at = None
        self.adding_finger_target_count = None
        self.adding_pre_grasp_fingers = []
        self.adding_ready_at = None
        self.skip_add_pre_grasp_once = False
        self.blending_fingers = []
        self.blend_started_at = None
        self.blend_until = None

    def _reset_envelop_grasp(self):
        self.envelop_hold_pose = None
        self.envelop_started_at = None
        self.envelop_thumb_enabled = False
        self.envelop_thumb_start_at = None
        self.envelop_last_joint_stall_since = None
        self.envelop_last_info = {}

    def _start_envelop_grasp(self, now):
        self._clear_transition_state()
        self.envelop_hold_pose = self.hand_q.copy()
        self.envelop_started_at = now
        self.envelop_thumb_enabled = False
        self.envelop_thumb_start_at = None
        self.envelop_last_joint_stall_since = None
        self.envelop_last_info = {
            "active_non_thumb": [],
            "active_thumb": [],
            "thumb_enabled": False,
            "tau_level": float(self.cfg.alpha1 * self.cfg.envelop_tau_scale),
        }

    def _envelop_non_thumb_joint_start_time(self, finger, local_joint):
        # Joint-stage sequencing: all non-thumb 2nd joints start together,
        # then all 3rd joints, then all 4th joints. The `finger` argument is
        # intentionally unused except for API symmetry/readability.
        _ = finger
        joint_order_idx = ENVELOP_FINGER_TORQUE_LOCAL_JOINTS.index(local_joint)
        return self.envelop_started_at + joint_order_idx * self.cfg.envelop_joint_delay

    def _calc_envelop_grasp(self, qdot, now):
        if self.envelop_hold_pose is None or self.envelop_started_at is None:
            self._start_envelop_grasp(now)

        _ = qdot  # The envelop timing is intentionally time-based, not velocity/stall-based.

        tau = np.zeros(JOINT_COUNT, dtype=np.float64)
        pd = np.zeros(JOINT_COUNT, dtype=np.float64)
        hold_mask = np.ones(JOINT_COUNT, dtype=bool)
        active_non_thumb = []
        active_thumb = []

        # Constant torque level for each active envelop-grasp joint.
        # Default is deliberately lower than alpha1/4 because the previous value was too strong on hardware.
        tau_level = float(self.cfg.alpha1 * self.cfg.envelop_tau_scale)
        tau_level = min(tau_level, float(self.cfg.groped_tau_limit))

        # Time-only joint-stage sequencing:
        #   t0 + 0*delay: non-thumb 2nd joints
        #   t0 + 1*delay: non-thumb 3rd joints
        #   t0 + 2*delay: non-thumb 4th joints + thumb 3rd joint
        #   t0 + 3*delay: thumb 4th joint
        # No qdot/stall condition is used for thumb triggering.
        for finger in ENVELOP_FINGER_ORDER:
            idxs = FINGER_JOINT_INDEX[finger]
            for local_joint in ENVELOP_FINGER_TORQUE_LOCAL_JOINTS:
                joint_idx = idxs[local_joint]
                start_t = self._envelop_non_thumb_joint_start_time(finger, local_joint)
                if now >= start_t:
                    hold_mask[joint_idx] = False
                    tau[joint_idx] = float(self.cfg.envelop_non_thumb_tau_sign) * tau_level
                    active_non_thumb.append(joint_idx)

        thumb_idxs = FINGER_JOINT_INDEX[1]
        for order_idx, local_joint in enumerate(ENVELOP_THUMB_TORQUE_LOCAL_JOINTS):
            # Thumb starts one stage later than before:
            # thumb joint 3 starts with the four non-thumb 4th joints,
            # thumb joint 4 starts one joint-delay after that.
            stage_idx = order_idx + 2
            joint_idx = thumb_idxs[local_joint]
            start_t = self.envelop_started_at + stage_idx * self.cfg.envelop_joint_delay
            if now >= start_t:
                hold_mask[joint_idx] = False
                tau[joint_idx] = float(self.cfg.envelop_thumb_tau_sign) * tau_level
                active_thumb.append(joint_idx)

        self.envelop_thumb_enabled = len(active_thumb) > 0

        hold_idxs = np.flatnonzero(hold_mask)
        pd[hold_idxs], err = pose_pd(
            self.envelop_hold_pose[hold_idxs],
            self.hand_q[hold_idxs],
            qdot[hold_idxs],
            kp=self.cfg.pose_kp,
            kd=self.cfg.pose_kd,
            limit=self.cfg.pose_pd_limit,
        )

        self.envelop_last_info = {
            "active_non_thumb": active_non_thumb,
            "active_thumb": active_thumb,
            "thumb_enabled": self.envelop_thumb_enabled,
            "tau_level": tau_level,
        }
        return tau, pd, err

    def run(self):
        prev_q = None
        prev_t = None
        qdot = np.zeros(JOINT_COUNT, dtype=np.float64)

        state = "WAIT_STATE"
        state_start = time()
        last_log = 0.0
        err = np.zeros(JOINT_COUNT, dtype=np.float64)
        grasp_tau = np.zeros(JOINT_COUNT, dtype=np.float64)
        alpha = {}
        cg = np.zeros(3, dtype=np.float64)
        cv = np.zeros(3, dtype=np.float64)

        print("[START] hand groped grasp with gravity + friction compensation")
        print("[XML]", self.model_xml_path)
        print(f"[COMMAND_TOPIC] {self.cfg.command_topic}")
        print(f"[ALPHA1_TOPIC] {self.cfg.alpha1_topic}")
        print(f"[ROTATION_MATRIX_TOPIC] {self.cfg.rotation_matrix_topic}")
        print(
            "[COMMAND] -1=normal, 0=pre-grasp, "
            "1=thumb+index, 2=thumb+middle, 3=thumb+index+middle, "
            "4=+ring, 5=+pinky, 6=envelop-grasp"
        )
        print(f"[INITIAL_FINGER_COUNT] {self.cfg.use_finger_count}")
        print(f"[ALPHA1] {self.cfg.alpha1}")
        print(f"[THUMB_CENTROID_BIAS] {self.cfg.thumb_centroid_bias}")
        print(
            f"[POSE_KP/KD/LIMIT] "
            f"{self.cfg.pose_kp}, {self.cfg.pose_kd}, {self.cfg.pose_pd_limit}"
        )

        try:
            while rclpy.ok():
                loop_t = time()
                rclpy.spin_once(self.node, timeout_sec=0.0)

                if not self.got_state:
                    publish_effort(self.pub, np.zeros(JOINT_COUNT, dtype=np.float64))
                    sleep(0.05)
                    continue

                now = time()

                if prev_q is not None and prev_t is not None:
                    dt = now - prev_t
                    if dt > 1e-6:
                        qdot_raw = (self.hand_q - prev_q) / dt
                        qdot = (1.0 - self.cfg.qdot_alpha) * qdot + self.cfg.qdot_alpha * qdot_raw

                prev_q = self.hand_q.copy()
                prev_t = now

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

                if state == "WAIT_STATE":
                    state = "NORMAL_POSE"
                    state_start = now
                    print("[STATE] WAIT_STATE -> NORMAL_POSE")

                state_elapsed = now - state_start

                if (
                    self.pending_finger_count is None
                    and self.adding_finger_target_count is not None
                    and self.adding_ready_at is not None
                    and now >= self.adding_ready_at
                ):
                    self.pending_finger_count = self.adding_finger_target_count
                    self.blending_fingers = []
                    self.blend_started_at = None
                    self.blend_until = None
                    self.adding_finger_target_count = None
                    self.adding_pre_grasp_fingers = []
                    self.adding_ready_at = None
                    self.skip_add_pre_grasp_once = True

                if (
                    self.pending_finger_count is None
                    and self.deferred_finger_count is not None
                    and self.deferred_finger_count_at is not None
                    and self.adding_finger_target_count is None
                    and now >= self.deferred_finger_count_at
                ):
                    self.pending_finger_count = self.deferred_finger_count
                    self.deferred_finger_count = None
                    self.deferred_finger_count_at = None

                elif (
                    self.pending_finger_count is not None
                    and self.deferred_finger_count is not None
                    and not self.skip_add_pre_grasp_once
                ):
                    self.deferred_finger_count = None
                    self.deferred_finger_count_at = None

                if self.pending_finger_count is not None:
                    requested_count = self.pending_finger_count
                    self.pending_finger_count = None
                    command_count = requested_count

                    if not self.skip_add_pre_grasp_once:
                        self.blending_fingers = []
                        self.blend_started_at = None
                        self.blend_until = None

                    if requested_count == -1:
                        self.active_finger_count = 0
                        self._clear_transition_state()
                        self._reset_envelop_grasp()

                        if state != "NORMAL_POSE":
                            state = "NORMAL_POSE"
                            state_start = now
                            state_elapsed = 0.0
                            print("[COMMAND] finger_count=-1 -> NORMAL_POSE")

                    elif requested_count == 0:
                        self.active_finger_count = 0
                        self._clear_transition_state()
                        self._reset_envelop_grasp()

                        if state != "PRE_GRASP_POSE":
                            state = "PRE_GRASP_POSE"
                            state_start = now
                            state_elapsed = 0.0
                            print("[COMMAND] finger_count=0 -> PRE_GRASP_POSE")

                    elif requested_count == 6:
                        self.active_finger_count = 6
                        self._start_envelop_grasp(now)
                        state = "ENVELOP_GRASP"
                        state_start = now
                        state_elapsed = 0.0
                        print(
                            "[COMMAND] finger_count=6 -> ENVELOP_GRASP "
                            f"tau_per_joint={self.cfg.alpha1 * self.cfg.envelop_tau_scale:.4f}, "
                            f"joint_stage_order=2 -> 3 -> (4+thumb3) -> thumb4, fingers={ENVELOP_FINGER_ORDER}"
                        )

                    else:
                        self._reset_envelop_grasp()

                        if (
                            self.active_finger_count in (1, 2)
                            and requested_count in (1, 2)
                            and requested_count != self.active_finger_count
                        ):
                            command_count = 3
                            self.deferred_finger_count = requested_count
                            self.deferred_finger_count_at = None
                            print(
                                f"[COMMAND] finger_count={requested_count} requested, "
                                f"switch via 3"
                            )

                        prev_fingers = (
                            set(self.use_fingers)
                            if 0 < self.active_finger_count <= 5
                            else set()
                        )
                        target_fingers = selected_fingers(command_count)
                        new_fingers = set(target_fingers)
                        added = sorted(new_fingers - prev_fingers)

                        if (
                            0 < self.active_finger_count <= 5
                            and added
                            and not self.skip_add_pre_grasp_once
                        ):
                            self.adding_finger_target_count = command_count
                            self.adding_pre_grasp_fingers = added
                            self.adding_ready_at = now + ADD_FINGER_PRE_GRASP_DELAY
                            state = "GROPED_GRASP"
                            state_start = now
                            state_elapsed = 0.0
                            print(
                                f"[COMMAND] finger_count={command_count} prepare added fingers "
                                f"{added} at PRE_GRASP for {ADD_FINGER_PRE_GRASP_DELAY:.2f}s"
                            )
                        else:
                            self.skip_add_pre_grasp_once = False
                            self.use_fingers = target_fingers
                            self.policy = GraspPolicy(self.use_fingers, self.cfg)

                            state = "GROPED_GRASP"
                            state_start = now
                            state_elapsed = 0.0
                            self.active_finger_count = command_count

                            removed = sorted(prev_fingers - new_fingers)

                            if (
                                self.deferred_finger_count is not None
                                and self.deferred_finger_count_at is None
                                and command_count == 3
                            ):
                                self.deferred_finger_count_at = now + FINGER_SWITCH_VIA_THREE_DELAY
                                print(
                                    f"[COMMAND] hold finger_count=3 for "
                                    f"{FINGER_SWITCH_VIA_THREE_DELAY:.2f}s before "
                                    f"finger_count={self.deferred_finger_count}"
                                )

                            print(
                                f"[COMMAND] finger_count={command_count} -> "
                                f"GROPED_GRASP, use_fingers={self.use_fingers}, "
                                f"added={added}, removed={removed}"
                            )

                if state == "NORMAL_POSE":
                    pd, err = pose_pd(
                        HAND_NORMAL_POSE,
                        self.hand_q,
                        qdot,
                        kp=self.cfg.pose_kp,
                        kd=self.cfg.pose_kd,
                        limit=self.cfg.pose_pd_limit,
                    )
                    effort = gravity + friction + pd

                elif state == "PRE_GRASP_POSE":
                    pd, err = pose_pd(
                        HAND_PRE_GRASP_POSE,
                        self.hand_q,
                        qdot,
                        kp=self.cfg.pose_kp,
                        kd=self.cfg.pose_kd,
                        limit=self.cfg.pose_pd_limit,
                    )
                    effort = gravity + friction + pd

                elif state == "GROPED_GRASP":
                    grasp_tau, alpha, cg, cv, _ = self.policy.calc_grasp_tau(self.hand_q)

                    blend_pd = np.zeros(JOINT_COUNT, dtype=np.float64)
                    if (
                        self.blending_fingers
                        and self.blend_started_at is not None
                        and self.blend_until is not None
                    ):
                        duration = max(self.blend_until - self.blend_started_at, 1e-6)
                        blend_raw = np.clip((now - self.blend_started_at) / duration, 0.0, 1.0)
                        blend = blend_raw * blend_raw * (3.0 - 2.0 * blend_raw)

                        for finger in list(self.blending_fingers):
                            idxs = np.asarray(FINGER_JOINT_INDEX[finger], dtype=int)
                            pd, _ = pose_pd(
                                HAND_PRE_GRASP_POSE[idxs],
                                self.hand_q[idxs],
                                qdot[idxs],
                                kp=self.cfg.pose_kp,
                                kd=self.cfg.pose_kd,
                                limit=self.cfg.pose_pd_limit,
                            )
                            grasp_tau[idxs] *= blend
                            blend_pd[idxs] = (1.0 - blend) * pd

                        if blend_raw >= 1.0:
                            self.blending_fingers = []
                            self.blend_started_at = None
                            self.blend_until = None

                    inactive_pd = self._inactive_pre_grasp_pd(
                        self.use_fingers,
                        qdot,
                        self.adding_pre_grasp_fingers,
                    )
                    effort = gravity + friction + grasp_tau + inactive_pd + blend_pd

                elif state == "ENVELOP_GRASP":
                    grasp_tau, envelop_pd, err = self._calc_envelop_grasp(qdot, now)
                    effort = gravity + friction + grasp_tau + envelop_pd


                else:
                    effort = gravity + friction

                effort = np.clip(effort, -self.cfg.hand_limit, self.cfg.hand_limit)
                publish_effort(self.pub, effort)

                if now - last_log > self.cfg.log_dt:
                    last_log = now
                    if state == "GROPED_GRASP":
                        alpha_str = {
                            finger: round(float(alpha[finger]), 4)
                            for finger in alpha
                        }
                        print(
                            f"[{state}] "
                            f"t={state_elapsed:.2f} | "
                            f"alpha={alpha_str} | "
                            f"cg={np.round(cg, 4)} | "
                            f"cv={np.round(cv, 4)} | "
                            f"grasp_tau_max={np.max(np.abs(grasp_tau)):.4f} | "
                            f"effort_max={np.max(np.abs(effort)):.3f}"
                        )
                    elif state == "ENVELOP_GRASP":
                        info = self.envelop_last_info
                        print(
                            f"[{state}] "
                            f"t={state_elapsed:.2f} | "
                            f"tau_level={info.get('tau_level', 0.0):.4f} | "
                            f"active_non_thumb={info.get('active_non_thumb', [])} | "
                            f"thumb_enabled={info.get('thumb_enabled', False)} | "
                            f"active_thumb={info.get('active_thumb', [])} | "
                            f"effort_max={np.max(np.abs(effort)):.3f}"
                        )
                    else:
                        print(
                            f"[{state}] "
                            f"t={state_elapsed:.2f} | "
                            f"err_max={np.max(np.abs(err)):.4f} | "
                            f"qdot_max={np.max(np.abs(qdot)):.4f} | "
                            f"gc_max={np.max(np.abs(gravity)):.3f} | "
                            f"fric_max={np.max(np.abs(friction)):.3f} | "
                            f"effort_max={np.max(np.abs(effort)):.3f}"
                        )

                sleep_dt = self.cfg.dt - (time() - loop_t)
                if sleep_dt > 0:
                    sleep(sleep_dt)

        finally:
            print("[STOP] zero effort")
            zero_effort(self.pub)


def main(args=None):
    rclpy.init(args=args)
    node = rclpy.create_node("grasp_real")

    try:
        cfg, model_xml_path = _declare_and_load_config(node)
        runner = GraspRealRunner(node, cfg, model_xml_path)
        runner.run()
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()