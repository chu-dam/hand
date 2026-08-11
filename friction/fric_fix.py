#!/usr/bin/env python3

from time import time, sleep
from datetime import datetime
import csv
import os
from pathlib import Path

import mujoco
import numpy as np

import rclpy
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray


# =====================================================
# Setting
# =====================================================
ROOT_DIR = Path(__file__).resolve().parents[1]
MODEL_PATH = (
    ROOT_DIR
    / "src/vendor/dg5f_s_description/urdf/dg5fs_right_w_mount.urdf"
)
MODEL_MESH_DIR = (
    ROOT_DIR
    / "src/vendor/dg5f_s_description/meshes/dg5fs_right"
)

JOINT_STATE_TOPIC = "/dg5f_s_right/joint_states"
EFFORT_TOPIC = "/dg5f_s_right/effort_controller/commands"
LOG_DIR = ROOT_DIR / "friction_logs" / "right"

DT = 0.005
EFFORT_LIMIT = 7.5
JOINT_STATE_TIMEOUT = 0.25


# =====================================================
# Friction identification setting
# =====================================================
TARGET_JOINT_INDEX = 15

TEST_EFFORTS = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
REPEAT_COUNT = 10

# 측정 전 반대쪽 limit으로 보내는 effort
PREMOVE_EFFORT = 0.30
PREMOVE_MAX_TIME = 8.0
PREMOVE_SETTLE_TIME = 0.2

# 한 방향 측정 최대 시간
MAX_TRIAL_TIME = 8.0

# q가 이 시간 동안 거의 안 변하면 방향 전환
MIN_TRIAL_TIME = 0.8
STOP_HOLD_TIME = 0.6
STOP_Q_CHANGE_THRESH = 0.003   # rad

# joint limit margin
LIMIT_MARGIN_DEG = 5.0

# 나머지 joint 고정용 PD
HOLD_KP = 0.8
HOLD_KD = 0.03
HOLD_PD_LIMIT = 0.25


# =====================================================
# Extra position hold joint
# target joint 측정 중 특정 joint를 0.0 rad로 유지
# 현재 설정:
#   TARGET_JOINT_INDEX = 5
#   LOCK_JOINT_INDEX   = 4
# =====================================================
LOCK_JOINT_ENABLE = True
LOCK_JOINT_INDEX = 12
LOCK_JOINT_TARGET = 0.0#1.5708       # rad

LOCK_KP = 1.2
LOCK_KD = 0.04
LOCK_PD_LIMIT = 0.35

LOCK_TOL = 0.03               # rad
LOCK_TIMEOUT = 5.0


JOINT_NAMES = [
    "joint_1_1", "joint_1_2", "joint_1_3", "joint_1_4",
    "joint_2_1", "joint_2_2", "joint_2_3", "joint_2_4",
    "joint_3_1", "joint_3_2", "joint_3_3", "joint_3_4",
    "joint_4_1", "joint_4_2", "joint_4_3", "joint_4_4",
    "joint_5_1", "joint_5_2", "joint_5_3", "joint_5_4",
]


# =====================================================
# Right-hand motor ranges from dg5fs_right_w_mount.urdf
# index 0 = Motor 1 = joint_1_1
# unit: deg
# =====================================================
JOINT_LIMIT_DEG = np.array([
    [-25.63,  55.07],   # Motor 1
    [-105.0,   0.0],   # Motor 2
    [-90.0,  90.0],   # Motor 3
    [-90.0,  90.0],   # Motor 4

    [-45.0,  12.0],   # Motor 5
    [  0.0, 121.0],   # Motor 6
    [-90.0,  90.0],   # Motor 7
    [-90.0,  90.0],   # Motor 8

    [-30.0,  39.0],   # Motor 9
    [  0.0, 125.0],   # Motor 10
    [-90.0,  90.0],   # Motor 11
    [-90.0,  90.0],   # Motor 12

    [-9.0,  45.0],   # Motor 13
    [  0.0, 121.0],   # Motor 14
    [-90.0,  90.0],   # Motor 15
    [-90.0,  90.0],   # Motor 16

    [  0.0,  95.0],   # Motor 17
    [-5.0,  90.0],   # Motor 18
    [-90.0,  90.0],   # Motor 19
    [-90.0,  90.0],   # Motor 20
], dtype=np.float64)


q_hand = np.zeros(20)
got_state = False
last_joint_state_time = None


def joint_state_cb(msg):
    global q_hand, got_state, last_joint_state_time

    name_to_pos = dict(zip(msg.name, msg.position))
    if not all(name in name_to_pos for name in JOINT_NAMES):
        return

    positions = np.array([name_to_pos[name] for name in JOINT_NAMES])
    if not np.all(np.isfinite(positions)):
        return

    q_hand[:] = positions

    got_state = True
    last_joint_state_time = time()


def load_model():
    assets = {
        path.name: path.read_bytes()
        for path in MODEL_MESH_DIR.glob("*.STL")
    }
    return mujoco.MjModel.from_xml_string(
        MODEL_PATH.read_text(encoding="utf-8"),
        assets,
    )


def get_addr(model):
    qadr = []
    dadr = []

    for name in JOINT_NAMES:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if jid < 0:
            raise RuntimeError(f"joint not found: {name}")

        qadr.append(model.jnt_qposadr[jid])
        dadr.append(model.jnt_dofadr[jid])

    return np.array(qadr), np.array(dadr)


def main():
    global q_hand

    if TARGET_JOINT_INDEX < 0 or TARGET_JOINT_INDEX >= len(JOINT_NAMES):
        raise RuntimeError("TARGET_JOINT_INDEX 범위가 잘못됨")

    if LOCK_JOINT_ENABLE:
        if LOCK_JOINT_INDEX < 0 or LOCK_JOINT_INDEX >= len(JOINT_NAMES):
            raise RuntimeError("LOCK_JOINT_INDEX 범위가 잘못됨")

        if LOCK_JOINT_INDEX == TARGET_JOINT_INDEX:
            raise RuntimeError("LOCK_JOINT_INDEX와 TARGET_JOINT_INDEX가 같으면 안 됩니다.")

    target_joint_name = JOINT_NAMES[TARGET_JOINT_INDEX]
    lock_joint_name = JOINT_NAMES[LOCK_JOINT_INDEX] if LOCK_JOINT_ENABLE else "none"

    raw_min_deg = JOINT_LIMIT_DEG[TARGET_JOINT_INDEX, 0]
    raw_max_deg = JOINT_LIMIT_DEG[TARGET_JOINT_INDEX, 1]

    safe_min_deg = raw_min_deg + LIMIT_MARGIN_DEG
    safe_max_deg = raw_max_deg - LIMIT_MARGIN_DEG

    safe_min_rad = np.deg2rad(safe_min_deg)
    safe_max_rad = np.deg2rad(safe_max_deg)

    if safe_min_rad >= safe_max_rad:
        raise RuntimeError("safe joint range가 잘못됨. LIMIT_MARGIN_DEG 확인 필요.")

    # =====================================================
    # ROS2
    # =====================================================
    rclpy.init()
    node = rclpy.create_node("hand_friction_ident_sweep")

    pub = node.create_publisher(Float64MultiArray, EFFORT_TOPIC, 10)
    node.create_subscription(JointState, JOINT_STATE_TOPIC, joint_state_cb, 10)

    # =====================================================
    # MuJoCo
    # =====================================================
    m = load_model()
    d = mujoco.MjData(m)

    qadr, dadr = get_addr(m)
    G = np.zeros(m.nv)

    # =====================================================
    # CSV
    # =====================================================
    save_dir = LOG_DIR
    os.makedirs(save_dir, exist_ok=True)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = os.path.join(
        save_dir,
        f"hand_friction_sweep_joint{TARGET_JOINT_INDEX}_{target_joint_name}_{stamp}.csv"
    )

    fieldnames = [
        "time",
        "sample_time",
        "trial_id",
        "rep",
        "joint_index",
        "joint_name",
        "effort_mag",
        "direction",
        "test_effort",
        "q",
        "q_deg",
        "qdot",
        "qddot",
        "q_start",
        "q_start_deg",
        "q_delta",
        "safe_min_deg",
        "safe_max_deg",
        "gravity_effort",
        "command_effort",
        "lock_joint_index",
        "lock_joint_name",
        "lock_q",
        "lock_q_deg",
        "lock_target",
        "lock_err",
        "stop_reason",
    ]

    print("[INFO] hand: right")
    print("[INFO] model:", MODEL_PATH)
    print("[INFO] target joint:", TARGET_JOINT_INDEX, target_joint_name)
    print(f"[INFO] raw limit deg : {raw_min_deg:.1f} ~ {raw_max_deg:.1f}")
    print(f"[INFO] safe limit deg: {safe_min_deg:.1f} ~ {safe_max_deg:.1f}")

    if LOCK_JOINT_ENABLE:
        print(
            f"[INFO] lock joint: {LOCK_JOINT_INDEX} {lock_joint_name} "
            f"-> {LOCK_JOINT_TARGET:.3f} rad / {np.rad2deg(LOCK_JOINT_TARGET):.2f} deg"
        )

    print("[INFO] csv:", csv_path)
    print("[WAIT] hand joint state...")

    while rclpy.ok() and not got_state:
        rclpy.spin_once(node, timeout_sec=0.1)
        sleep(0.1)

    sleep(0.5)

    q0 = q_hand[TARGET_JOINT_INDEX]

    if q0 < safe_min_rad or q0 > safe_max_rad:
        print()
        print("[WARN] target joint 현재 위치가 safe range 밖입니다.")
        print(f"[WARN] q_now = {np.rad2deg(q0):.2f} deg")
        print(f"[WARN] safe = {safe_min_deg:.2f} ~ {safe_max_deg:.2f} deg")
        print()

    q_hold = q_hand.copy()

    if LOCK_JOINT_ENABLE:
        q_hold[LOCK_JOINT_INDEX] = LOCK_JOINT_TARGET

    print("[INFO] hold posture captured")
    print(f"[INFO] target initial q: {q0:.4f} rad / {np.rad2deg(q0):.2f} deg")
    print("[START] measurement starts in 2 sec")
    sleep(2.0)

    # =====================================================
    # Derivative variables
    # =====================================================
    prev_time = time()
    prev_q = q_hand.copy()
    prev_qdot = np.zeros(20)

    qdot = np.zeros(20)
    qddot = np.zeros(20)

    def update_derivatives():
        nonlocal prev_time, prev_q, prev_qdot, qdot, qddot

        now = time()
        dt = now - prev_time

        if dt > 1e-6:
            qdot[:] = (q_hand - prev_q) / dt
            qddot[:] = (qdot - prev_qdot) / dt

            prev_q[:] = q_hand
            prev_qdot[:] = qdot
            prev_time = now

        return now

    def compute_gravity():
        d.qpos[:] = 0.0
        d.qvel[:] = 0.0
        d.qacc[:] = 0.0

        d.qpos[qadr] = q_hand

        mujoco.mj_forward(m, d)
        mujoco.mj_rne(m, d, 0, G)

        return G[dadr].copy()

    def publish_effort(test_effort):
        rclpy.spin_once(node, timeout_sec=0.0)
        deadline = time() + JOINT_STATE_TIMEOUT
        while (
            rclpy.ok()
            and (
                last_joint_state_time is None
                or time() - last_joint_state_time > JOINT_STATE_TIMEOUT
            )
            and time() < deadline
        ):
            rclpy.spin_once(node, timeout_sec=DT)

        if (
            last_joint_state_time is None
            or time() - last_joint_state_time > JOINT_STATE_TIMEOUT
        ):
            raise RuntimeError("joint state timeout; stop measurement")
        now = update_derivatives()

        gravity_effort = compute_gravity()
        cmd = gravity_effort.copy()

        # 기본: target joint 제외한 나머지를 q_hold 기준으로 유지
        hold_pd = HOLD_KP * (q_hold - q_hand) - HOLD_KD * qdot
        hold_pd = np.clip(hold_pd, -HOLD_PD_LIMIT, HOLD_PD_LIMIT)

        # 측정 target joint는 hold 제거
        hold_pd[TARGET_JOINT_INDEX] = 0.0

        # lock joint는 별도 PD gain으로 q=LOCK_JOINT_TARGET 유지
        if LOCK_JOINT_ENABLE:
            lock_pd = (
                LOCK_KP * (LOCK_JOINT_TARGET - q_hand[LOCK_JOINT_INDEX])
                - LOCK_KD * qdot[LOCK_JOINT_INDEX]
            )
            hold_pd[LOCK_JOINT_INDEX] = np.clip(
                lock_pd,
                -LOCK_PD_LIMIT,
                LOCK_PD_LIMIT,
            )

        cmd += hold_pd

        # target joint에만 test effort 추가
        cmd[TARGET_JOINT_INDEX] += test_effort

        cmd = np.clip(cmd, -EFFORT_LIMIT, EFFORT_LIMIT)

        msg = Float64MultiArray()
        msg.data = cmd.tolist()
        pub.publish(msg)

        return now, gravity_effort, cmd

    def move_lock_joint_to_target():
        if not LOCK_JOINT_ENABLE:
            return

        print(
            f"[LOCK INIT] move {LOCK_JOINT_INDEX} {lock_joint_name} "
            f"to {LOCK_JOINT_TARGET:.3f} rad / {np.rad2deg(LOCK_JOINT_TARGET):.2f} deg"
        )

        t0 = time()

        while rclpy.ok():
            publish_effort(0.0)

            q_lock = q_hand[LOCK_JOINT_INDEX]
            err = LOCK_JOINT_TARGET - q_lock

            if abs(err) < LOCK_TOL:
                print(
                    f"[LOCK INIT DONE] "
                    f"q={q_lock:.4f} rad / {np.rad2deg(q_lock):.2f} deg | "
                    f"err={err:.4f}"
                )
                break

            if time() - t0 > LOCK_TIMEOUT:
                print(
                    f"[LOCK INIT TIMEOUT] "
                    f"q={q_lock:.4f} rad / {np.rad2deg(q_lock):.2f} deg | "
                    f"err={err:.4f}"
                )
                break

            sleep(DT)

        sleep(0.3)

    def move_to_opposite_side(direction):
        """
        측정 방향의 반대쪽 safe limit으로 먼저 이동.
        direction > 0이면 -방향으로 이동해서 lower limit에서 시작.
        direction < 0이면 +방향으로 이동해서 upper limit에서 시작.
        """
        premove_direction = -direction
        premove_effort = premove_direction * PREMOVE_EFFORT

        t0 = time()
        stall_ref_q = q_hand[TARGET_JOINT_INDEX]
        stall_ref_time = t0
        stop_reason = ""

        print(
            f"[PREMOVE] measurement_dir={direction:+d} | "
            f"premove_effort={premove_effort:+.3f}"
        )

        while rclpy.ok():
            now, _, _ = publish_effort(premove_effort)

            sample_time = now - t0
            q = q_hand[TARGET_JOINT_INDEX]
            q_deg = np.rad2deg(q)

            # 반대쪽 limit 도달
            if premove_direction > 0 and q >= safe_max_rad:
                stop_reason = "upper_limit"
            elif premove_direction < 0 and q <= safe_min_rad:
                stop_reason = "lower_limit"

            # limit에 못 가더라도 q가 더 이상 안 변하면 멈춤
            if not stop_reason and sample_time > MIN_TRIAL_TIME:
                if now - stall_ref_time >= STOP_HOLD_TIME:
                    q_change = abs(q - stall_ref_q)

                    if q_change < STOP_Q_CHANGE_THRESH:
                        stop_reason = "q_not_changing"
                    else:
                        stall_ref_q = q
                        stall_ref_time = now

            if not stop_reason and sample_time > PREMOVE_MAX_TIME:
                stop_reason = "timeout"

            if stop_reason:
                print(
                    f"  -> premove stop: {stop_reason} | "
                    f"q={q:.4f} rad / {q_deg:.2f} deg"
                )
                break

            sleep(DT)

        # 측정 전 짧게 안정화
        t_wait = time()
        while rclpy.ok() and time() - t_wait < PREMOVE_SETTLE_TIME:
            publish_effort(0.0)
            sleep(DT)

    def run_one_direction(writer, trial_id, rep, effort_mag, direction):
        # 측정 시작 전, 반대쪽 끝으로 먼저 이동
        move_to_opposite_side(direction)

        test_effort = direction * effort_mag
        q_start = q_hand[TARGET_JOINT_INDEX]

        t0 = time()
        stop_reason = ""

        stall_ref_q = q_start
        stall_ref_time = t0

        print(
            f"[TRIAL {trial_id}] "
            f"effort={test_effort:+.3f} | "
            f"rep={rep}/{REPEAT_COUNT} | "
            f"start_q={q_start:.4f} rad / {np.rad2deg(q_start):.2f} deg"
        )

        while rclpy.ok():
            now, gravity_effort, cmd = publish_effort(test_effort)

            sample_time = now - t0

            q = q_hand[TARGET_JOINT_INDEX]
            q_deg = np.rad2deg(q)

            qd = qdot[TARGET_JOINT_INDEX]
            qdd = qddot[TARGET_JOINT_INDEX]
            q_delta = q - q_start

            if direction > 0 and q >= safe_max_rad:
                stop_reason = "upper_limit"

            if direction < 0 and q <= safe_min_rad:
                stop_reason = "lower_limit"

            if not stop_reason and sample_time > MIN_TRIAL_TIME:
                if now - stall_ref_time >= STOP_HOLD_TIME:
                    q_change = abs(q - stall_ref_q)

                    if q_change < STOP_Q_CHANGE_THRESH:
                        stop_reason = "q_not_changing"
                    else:
                        stall_ref_q = q
                        stall_ref_time = now

            if not stop_reason and sample_time > MAX_TRIAL_TIME:
                stop_reason = "timeout"

            lock_q = q_hand[LOCK_JOINT_INDEX] if LOCK_JOINT_ENABLE else 0.0
            lock_err = (
                LOCK_JOINT_TARGET - q_hand[LOCK_JOINT_INDEX]
                if LOCK_JOINT_ENABLE else 0.0
            )

            writer.writerow({
                "time": now,
                "sample_time": sample_time,
                "trial_id": trial_id,
                "rep": rep,
                "joint_index": TARGET_JOINT_INDEX,
                "joint_name": target_joint_name,
                "effort_mag": effort_mag,
                "direction": direction,
                "test_effort": test_effort,
                "q": q,
                "q_deg": q_deg,
                "qdot": qd,
                "qddot": qdd,
                "q_start": q_start,
                "q_start_deg": np.rad2deg(q_start),
                "q_delta": q_delta,
                "safe_min_deg": safe_min_deg,
                "safe_max_deg": safe_max_deg,
                "gravity_effort": gravity_effort[TARGET_JOINT_INDEX],
                "command_effort": cmd[TARGET_JOINT_INDEX],
                "lock_joint_index": LOCK_JOINT_INDEX if LOCK_JOINT_ENABLE else -1,
                "lock_joint_name": lock_joint_name,
                "lock_q": lock_q,
                "lock_q_deg": np.rad2deg(lock_q),
                "lock_target": LOCK_JOINT_TARGET if LOCK_JOINT_ENABLE else 0.0,
                "lock_err": lock_err,
                "stop_reason": stop_reason,
            })

            if stop_reason:
                print(
                    f"  -> stop: {stop_reason} | "
                    f"q={q:.4f} rad / {q_deg:.2f} deg | "
                    f"q_delta={q_delta:.4f}, qdot={qd:.4f}"
                )
                break

            sleep(DT)

        # 방향 전환 전 짧게 중력보상만
        t_wait = time()
        while rclpy.ok() and time() - t_wait < 0.15:
            publish_effort(0.0)
            sleep(DT)

    try:
        move_lock_joint_to_target()

        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

            trial_id = 0

            for effort_mag in TEST_EFFORTS:
                print()
                print("====================================")
                print(f"[EFFORT MAG] {effort_mag:.3f}")
                print("====================================")

                for rep in range(1, REPEAT_COUNT + 1):
                    # + direction
                    trial_id += 1
                    run_one_direction(
                        writer=writer,
                        trial_id=trial_id,
                        rep=rep,
                        effort_mag=effort_mag,
                        direction=1,
                    )
                    f.flush()

                    # - direction
                    trial_id += 1
                    run_one_direction(
                        writer=writer,
                        trial_id=trial_id,
                        rep=rep,
                        effort_mag=effort_mag,
                        direction=-1,
                    )
                    f.flush()

            print("[DONE] measurement finished")
            print("[CSV]", csv_path)

    finally:
        print("[STOP] zero effort")

        msg = Float64MultiArray()
        msg.data = np.zeros(20).tolist()
        pub.publish(msg)

        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
