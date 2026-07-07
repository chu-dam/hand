#!/usr/bin/env python3

import os
from time import sleep, time

import mujoco
import numpy as np
import rbpodo as rb
import rclpy
from std_msgs.msg import Float64MultiArray, Int32


ROBOT_ADDRESS = "169.254.186.20"
XML_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "rb5_850e_payload_1kg.xml",
)

ROTATION_MATRIX_TOPIC = "/dg5f_grasp_control/rotation_matrix_cmd"
FINGER_COUNT_TOPIC = "/dg5f_grasp_control/finger_count_cmd"

PUBLISH_FINGER_COUNT_ON_START = True
FINGER_COUNT_CMD = 1  # 1 = thumb + index

DT = 0.005
LOG_DT = 0.25
T1 = 0.01
T2 = 0.05

RB5_JOINT_NAMES = ["base", "shoulder", "elbow", "wrist1", "wrist2", "wrist3"]
RB5_TAU_LIMIT = np.full(6, 50.0, dtype=np.float64)
MAX_JVEL_DEG_S = 70.0

# From rb5_hand.xml:
# <body name="link_mount" pos="0 -0.0965 0" quat="0.5 0.5 0.5 -0.5">
# This maps the hand root frame, link_mount, into the RB5 link6/TCP frame.
R_LINK_MOUNT_TO_LINK6 = np.array([
    [0.0, 1.0, 0.0],
    [0.0, 0.0, -1.0],
    [-1.0, 0.0, 0.0],
], dtype=np.float64)


def rb_get_joint_position(state):
    if state is None:
        return np.zeros(6, dtype=np.float64)
    return np.array(state.sdata.jnt_ang, dtype=np.float64)


def get_joint_addr(model, joint_names):
    qadr, dadr = [], []

    for name in joint_names:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if jid < 0:
            raise RuntimeError(f"joint not found: {name}")

        qadr.append(model.jnt_qposadr[jid])
        dadr.append(model.jnt_dofadr[jid])

    return np.array(qadr, dtype=int), np.array(dadr, dtype=int)


def connect_rb5():
    robot = rb.Cobot(ROBOT_ADDRESS)
    rc = rb.ResponseCollector()
    robot_data = rb.CobotData(ROBOT_ADDRESS)

    robot.set_operation_mode(rc, rb.OperationMode.Real)
    robot.set_speed_bar(rc, 0.5)
    robot.set_freedrive_mode(rc, on=False)

    return robot, rc, robot_data


def publish_rotation_matrix(pub, rot):
    msg = Float64MultiArray()
    msg.data = rot.reshape(9).astype(float).tolist()
    pub.publish(msg)


def publish_finger_count_once(pub, count):
    msg = Int32()
    msg.data = int(count)
    pub.publish(msg)


def main():
    rclpy.init()
    node = rclpy.create_node("rb5_payload_gc_rotation_pub")

    rot_pub = node.create_publisher(Float64MultiArray, ROTATION_MATRIX_TOPIC, 10)
    finger_pub = node.create_publisher(Int32, FINGER_COUNT_TOPIC, 10)

    try:
        robot, rc, robot_data = connect_rb5()
    except Exception as e:
        print(f"RB5 connection failed: {e}")
        node.destroy_node()
        rclpy.shutdown()
        return

    model = mujoco.MjModel.from_xml_path(XML_PATH)
    data = mujoco.MjData(model)
    rb5_qadr, rb5_dadr = get_joint_addr(model, RB5_JOINT_NAMES)

    tcp_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "tcp")
    if tcp_site_id < 0:
        raise RuntimeError("site not found: tcp")

    gravity_tau = np.zeros(model.nv, dtype=np.float64)

    print("[START] RB5 gravity compensation + tcp rotation matrix publisher")
    print("[XML]", XML_PATH)
    print("[ROTATION_MATRIX_TOPIC]", ROTATION_MATRIX_TOPIC)
    print("[FINGER_COUNT_TOPIC]", FINGER_COUNT_TOPIC)
    print("[PAYLOAD] 1.0 kg sphere at tcp_frame")

    sleep(0.5)
    if PUBLISH_FINGER_COUNT_ON_START:
        publish_finger_count_once(finger_pub, FINGER_COUNT_CMD)
        print(f"[FINGER_COUNT] published once: {FINGER_COUNT_CMD}")

    prev_time = time()
    prev_q_deg = None
    last_log = 0.0

    try:
        while rclpy.ok():
            loop_start = time()
            rclpy.spin_once(node, timeout_sec=0.0)

            state = robot_data.request_data()
            if state is None:
                sleep(DT)
                continue

            if state.sdata.op_stat_collision_occur or state.sdata.op_stat_sos_flag == 4:
                print("[STOP] RB5 safety flag detected")
                break

            now = time()
            loop_dt = now - prev_time
            prev_time = now
            if loop_dt <= 1e-6:
                sleep(DT)
                continue

            q_deg = rb_get_joint_position(state)
            q_rad = np.deg2rad(q_deg)

            if prev_q_deg is None:
                jvel_deg_s = np.zeros(6, dtype=np.float64)
            else:
                jvel_deg_s = (q_deg - prev_q_deg) / loop_dt
            prev_q_deg = q_deg.copy()

            if np.any(np.abs(jvel_deg_s) > MAX_JVEL_DEG_S):
                print(f"[SKIP] joint velocity too high: {np.round(jvel_deg_s, 2)} deg/s")
                sleep(DT)
                continue

            data.qpos[:] = 0.0
            data.qvel[:] = 0.0
            data.qacc[:] = 0.0
            data.qpos[rb5_qadr] = q_rad

            mujoco.mj_forward(model, data)
            mujoco.mj_rne(model, data, 0, gravity_tau)

            rb5_tau = np.clip(gravity_tau[rb5_dadr].copy(), -RB5_TAU_LIMIT, RB5_TAU_LIMIT)
            robot.move_servo_t(rc, rb5_tau, T1, T2, compensation=0)

            r_link6_to_world = data.site_xmat[tcp_site_id].reshape(3, 3).copy()
            r_link_mount_to_world = r_link6_to_world @ R_LINK_MOUNT_TO_LINK6
            publish_rotation_matrix(rot_pub, r_link_mount_to_world)

            if now - last_log >= LOG_DT:
                print(
                    "[GC] "
                    f"q_deg={np.round(q_deg, 2)} | "
                    f"tau={np.round(rb5_tau, 3)} | "
                    f"R_mount_row0={np.round(r_link_mount_to_world[0], 3)}"
                )
                last_log = now

            sleep_dt = DT - (time() - loop_start)
            if sleep_dt > 0:
                sleep(sleep_dt)

    finally:
        print("[STOP] zero RB5 torque")
        try:
            robot.move_servo_t(rc, np.zeros(6, dtype=np.float64), T1, T2, compensation=0)
        except Exception as e:
            print(f"RB5 zero torque failed: {e}")

        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
