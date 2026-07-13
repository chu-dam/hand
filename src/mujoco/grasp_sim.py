#!/usr/bin/env python3

import atexit
import os
import sys
import threading
import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np

# Allow direct execution from the source tree without requiring an installed
# workspace. When the workspace is sourced, the installed package is used.
SOURCE_PACKAGE_ROOT = (
    Path(__file__).resolve().parents[1]
    / "real"
    / "dg5f_grasp_control"
)
if str(SOURCE_PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_PACKAGE_ROOT))

from dg5f_grasp_control.config import RuntimeConfig, load_runtime_config_yaml
from dg5f_grasp_control.grasp_controller import GraspController
from dg5f_grasp_control.hand_model import HAND_JOINT_NAMES, JOINT_COUNT
from dg5f_grasp_control.poses import HAND_NORMAL_POSE, POSE_TYPE_TARGETS

try:
    import rclpy
    from rclpy.executors import SingleThreadedExecutor
    from rclpy.node import Node
    from std_msgs.msg import Float64, Float64MultiArray, Int32
except ImportError:
    rclpy = None
    SingleThreadedExecutor = None
    Node = object
    Float64 = None
    Float64MultiArray = None
    Int32 = None


XML_PATH = Path(__file__).resolve().with_name("dg5fs_left_w_mount.xml")
GRAVITY_WORLD = np.array([0.0, 0.0, -9.81], dtype=np.float64)
DEFAULT_ROTATION_MATRIX = np.eye(3, dtype=np.float64)
ENABLE_STL_SELF_COLLISION = True
LOG_PERIOD = 0.5


def enable_all_stl_mesh_collisions(model):
    """Enable physics collision between all STL/mesh geoms."""
    mesh_geom_ids = np.flatnonzero(model.geom_type == mujoco.mjtGeom.mjGEOM_MESH)
    model.geom_contype[:] = 0
    model.geom_conaffinity[:] = 0

    if ENABLE_STL_SELF_COLLISION and mesh_geom_ids.size > 0:
        model.geom_contype[mesh_geom_ids] = 1
        model.geom_conaffinity[mesh_geom_ids] = 1
        try:
            model.geom_condim[mesh_geom_ids] = np.maximum(
                model.geom_condim[mesh_geom_ids],
                3,
            )
        except Exception:
            pass
    return int(mesh_geom_ids.size)


class GraspSimCommandNode(Node):
    def __init__(self, cfg):
        super().__init__("grasp_sim")
        self._lock = threading.Lock()
        self.pending_grasp_type = None
        self.pending_pose_type = None
        self.pending_alpha1 = None
        self.pending_rotation_matrix = None

        self.create_subscription(Int32, cfg.command_topic, self.grasp_type_cb, 10)
        self.create_subscription(Int32, cfg.pose_topic, self.pose_type_cb, 10)
        self.create_subscription(Float64, cfg.alpha1_topic, self.alpha1_cb, 10)
        self.create_subscription(
            Float64MultiArray,
            cfg.rotation_matrix_topic,
            self.rotation_matrix_cb,
            10,
        )

    def grasp_type_cb(self, msg):
        command = int(msg.data)
        if command < -1 or command > 7:
            self.get_logger().warn(
                f"Ignore invalid grasp_type: {command}. "
                "Use -1, 0, 1, 2, 3, 4, 5, 6, or 7."
            )
            return
        with self._lock:
            self.pending_grasp_type = command
        self.get_logger().info(f"RX grasp_type={command}")

    def pose_type_cb(self, msg):
        pose_type = int(msg.data)
        if pose_type not in POSE_TYPE_TARGETS:
            self.get_logger().warn(
                f"Ignore invalid pose_type: {pose_type}. Use 1, 2, or 3."
            )
            return
        with self._lock:
            self.pending_pose_type = pose_type
        self.get_logger().info(f"RX pose_type={pose_type}")

    def alpha1_cb(self, msg):
        alpha1 = float(msg.data)
        if alpha1 < 0.0 or not np.isfinite(alpha1):
            self.get_logger().warn(
                f"Ignore invalid alpha1: {alpha1}. Use a finite value >= 0."
            )
            return
        with self._lock:
            self.pending_alpha1 = alpha1
        self.get_logger().info(f"RX alpha1={alpha1:.4f}")

    def rotation_matrix_cb(self, msg):
        values = np.asarray(msg.data, dtype=np.float64)
        if values.size != 9 or not np.all(np.isfinite(values)):
            self.get_logger().warn(
                f"Ignore invalid rotation matrix. Expected 9 finite values, got {values.size}."
            )
            return
        with self._lock:
            self.pending_rotation_matrix = values.reshape(3, 3).copy()

    def take_pending(self):
        with self._lock:
            values = (
                self.pending_grasp_type,
                self.pending_pose_type,
                self.pending_alpha1,
                None
                if self.pending_rotation_matrix is None
                else self.pending_rotation_matrix.copy(),
            )
            self.pending_grasp_type = None
            self.pending_pose_type = None
            self.pending_alpha1 = None
            self.pending_rotation_matrix = None
        return values


class GraspSim:
    """MuJoCo adapter for the same GraspController used by real hardware."""

    def __init__(self):
        shared_yaml = (
            Path(__file__).resolve().parents[1]
            / "real"
            / "dg5f_grasp_control"
            / "config"
            / "grasp_real.yaml"
        )
        self.cfg = (
            load_runtime_config_yaml(shared_yaml)
            if shared_yaml.exists()
            else RuntimeConfig()
        )
        self.controller = GraspController(self.cfg, log=print)

        self.model = mujoco.MjModel.from_xml_path(str(XML_PATH))
        self.stl_collision_geom_count = enable_all_stl_mesh_collisions(self.model)
        self.data = mujoco.MjData(self.model)
        self.gravity_data = mujoco.MjData(self.model)
        self.qpos_addr, self.dof_addr = self._get_joint_addresses()

        self.rotation_matrix = DEFAULT_ROTATION_MATRIX.copy()
        self.gravity_in_hand = self.rotation_matrix.T @ GRAVITY_WORLD
        self.model.opt.gravity[:] = self.gravity_in_hand

        self.ros_node = None
        self.ros_executor = None
        self.ros_spin_thread = None
        self.start_time = time.time()
        self.last_print_time = 0.0
        self.last_output = None

        self.set_initial_pose()

    def _get_joint_addresses(self):
        qpos_addr = []
        dof_addr = []
        for name in HAND_JOINT_NAMES:
            joint_id = mujoco.mj_name2id(
                self.model,
                mujoco.mjtObj.mjOBJ_JOINT,
                name,
            )
            if joint_id < 0:
                raise RuntimeError(f"Joint not found in MuJoCo XML: {name}")
            qpos_addr.append(self.model.jnt_qposadr[joint_id])
            dof_addr.append(self.model.jnt_dofadr[joint_id])
        return np.asarray(qpos_addr, dtype=int), np.asarray(dof_addr, dtype=int)

    @property
    def q(self):
        return np.asarray(self.data.qpos[self.qpos_addr], dtype=np.float64)

    @property
    def qdot(self):
        return np.asarray(self.data.qvel[self.dof_addr], dtype=np.float64)

    def set_initial_pose(self):
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[self.qpos_addr] = HAND_NORMAL_POSE
        self.data.qvel[:] = 0.0
        self.data.ctrl[:] = 0.0
        self.data.qfrc_applied[:] = 0.0
        mujoco.mj_forward(self.model, self.data)

    def start_ros(self):
        if rclpy is None:
            print("[WARN] rclpy/std_msgs not found. ROS command topics are disabled.")
            return

        rclpy.init(args=None)
        self.ros_node = GraspSimCommandNode(self.cfg)
        self.ros_executor = SingleThreadedExecutor()
        self.ros_executor.add_node(self.ros_node)
        self.ros_spin_thread = threading.Thread(
            target=self.ros_executor.spin,
            daemon=True,
            name="grasp_sim_ros_spin",
        )
        self.ros_spin_thread.start()
        atexit.register(self.shutdown_ros)

    def shutdown_ros(self):
        if self.ros_executor is not None:
            self.ros_executor.shutdown()
        if self.ros_node is not None:
            self.ros_node.destroy_node()
        if rclpy is not None and rclpy.ok():
            rclpy.shutdown()

    def apply_rotation_matrix(self, rotation_matrix):
        self.rotation_matrix = np.asarray(rotation_matrix, dtype=np.float64).copy()
        self.gravity_in_hand = self.rotation_matrix.T @ GRAVITY_WORLD
        self.model.opt.gravity[:] = self.gravity_in_hand
        print(
            "[COMMAND] rotation matrix updated, gravity_in_hand="
            f"{np.round(self.gravity_in_hand, 4).tolist()}"
        )

    def apply_gravity_compensation(self):
        self.model.opt.gravity[:] = self.gravity_in_hand
        self.gravity_data.qpos[:] = self.data.qpos[:]
        self.gravity_data.qvel[:] = 0.0
        self.gravity_data.qacc[:] = 0.0
        mujoco.mj_forward(self.model, self.gravity_data)
        self.data.qfrc_applied[:] = self.gravity_data.qfrc_bias

    def process_pending_commands(self, now):
        if self.ros_node is None:
            return

        grasp_type, pose_type, alpha1, rotation_matrix = self.ros_node.take_pending()
        if pose_type is not None:
            self.controller.apply_pose_type(pose_type, now)
        if grasp_type is not None:
            self.controller.apply_grasp_type(grasp_type, now)
        if alpha1 is not None:
            self.controller.set_alpha1(alpha1)
            self.cfg = self.controller.cfg
            print(f"[COMMAND] alpha1={alpha1:.4f}")
        if rotation_matrix is not None:
            self.apply_rotation_matrix(rotation_matrix)

    def print_start_info(self):
        print("=" * 80)
        print("[GRASP SIM START - SHARED REAL CONTROLLER]")
        print(f"XML_PATH            : {XML_PATH}")
        print(f"CONTROLLER_MODULE   : {GraspController.__module__}")
        print(f"ALPHA1              : {self.controller.cfg.alpha1}")
        print(f"USE_FINGERS         : {self.controller.use_fingers}")
        print(f"STL_COLLISION_GEOMS : {self.stl_collision_geom_count}")
        print(f"ROS_ENABLED         : {self.ros_node is not None}")
        print(f"GRASP_TYPE_TOPIC    : {self.cfg.command_topic}")
        print(f"POSE_TYPE_TOPIC     : {self.cfg.pose_topic}")
        print(f"ALPHA1_TOPIC        : {self.cfg.alpha1_topic}")
        print(f"ROTATION_TOPIC      : {self.cfg.rotation_matrix_topic}")
        print(f"GRAVITY_IN_HAND     : {np.round(self.gravity_in_hand, 4).tolist()}")
        print(
            "[COMMAND] -1=normal, 0=pre-grasp, 1=thumb+index, "
            "2=thumb+middle, 3=three fingers, 4=four fingers, "
            "5=five fingers, 6=envelop, 7=rotation/transition"
        )
        print("=" * 80)

    def print_status(self, now, output, torque):
        if now - self.last_print_time < LOG_PERIOD:
            return
        self.last_print_time = now

        alpha_text = "{" + ", ".join(
            f"F{finger}:{value:.3f}"
            for finger, value in output.alpha.items()
        ) + "}"
        print(
            f"[{output.state}] "
            f"t={now - self.start_time:6.2f} | "
            f"state_t={output.state_elapsed:5.2f} | "
            f"active_count={output.active_finger_count} | "
            f"use_fingers={output.use_fingers} | "
            f"alpha={alpha_text} | "
            f"tau_max={np.max(np.abs(torque)):.4f} | "
            f"Cg={np.round(output.cg, 4)} | "
            f"Cv={np.round(output.cv, 4)} | "
            f"g7_phase={output.g7_phase}"
        )

    @staticmethod
    def disable_contact_visualization(viewer):
        try:
            viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = False
            viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = False
        except Exception as exc:
            print(f"[WARN] contact visualization option was not changed: {exc}")

    def run(self):
        self.start_ros()
        self.print_start_info()

        with mujoco.viewer.launch_passive(self.model, self.data) as viewer:
            self.disable_contact_visualization(viewer)
            while viewer.is_running():
                loop_start = time.time()
                now = time.time()

                self.controller.sync_joint_state(self.q)
                self.process_pending_commands(now)
                output = self.controller.step(self.q, self.qdot, now)
                self.apply_gravity_compensation()

                torque = np.clip(
                    output.tau,
                    -self.controller.cfg.hand_limit,
                    self.controller.cfg.hand_limit,
                )
                if self.model.nu != JOINT_COUNT:
                    raise RuntimeError(
                        f"Expected {JOINT_COUNT} hand actuators, found {self.model.nu}"
                    )
                self.data.ctrl[:] = torque

                mujoco.mj_step(self.model, self.data)
                viewer.sync()
                self.print_status(now, output, torque)

                sleep_time = self.model.opt.timestep - (time.time() - loop_start)
                if sleep_time > 0.0:
                    time.sleep(sleep_time)


def main():
    GraspSim().run()


if __name__ == "__main__":
    main()
