#!/usr/bin/env python3

import os
import time
import atexit
import threading
from enum import Enum

import numpy as np
import mujoco
import mujoco.viewer

try:
    import rclpy
    from rclpy.node import Node
    from rclpy.executors import SingleThreadedExecutor
    from std_msgs.msg import Float64, Int32
except ImportError:
    rclpy = None
    Node = object
    SingleThreadedExecutor = None
    Float64 = None
    Int32 = None


# =====================================================
# XML path
# =====================================================
XML_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "dg5fs_left_w_mount.xml"
)


# =====================================================
# Finger selection
# =====================================================
COMMAND_TOPIC = "/dg5f_grasp_control/finger_count_cmd"
ALPHA1_TOPIC = "/dg5f_grasp_control/alpha1_cmd"

# 손 단독 MuJoCo에서는 real-hand gravity 방향 보정용 회전행렬을
# 외부에서 받지 않고 항등행렬로 고정한다.
ROTATION_MATRIX_IDENTITY = np.eye(3, dtype=np.float64)

# 시작 시 사용할 손가락 명령.
# 0: pre-grasp
# 1: thumb + index
# 2: thumb + middle
# 3: thumb + index + middle
# 4: thumb + index + middle + ring
# 5: all fingers
DEFAULT_FINGER_COMMAND = 3

# 손가락 번호 정의
# 1: thumb
# 2: index
# 3: middle
# 4: ring
# 5: pinky
FINGER_ORDER = [1, 2, 3, 4, 5]

FINGER_SELECTIONS = {
    1: [1, 2],
    2: [1, 3],
    3: [1, 2, 3],
    4: [1, 2, 3, 4],
    5: [1, 2, 3, 4, 5],
}


def selected_fingers(command: int):
    if command not in FINGER_SELECTIONS:
        raise ValueError("finger command must be one of: 1, 2, 3, 4, 5")
    return FINGER_SELECTIONS[command]


FINGER_COMMAND = DEFAULT_FINGER_COMMAND
USE_FINGERS = selected_fingers(FINGER_COMMAND)


# =====================================================
# Control parameters
# =====================================================
# Initial pose sequence
NORMAL_POSE_TIME = 3.0
PRE_GRASP_POSE_TIME = 1.0

# real/dg5f_grasp_control의 RuntimeConfig와 같은 의미의 제어 파라미터.
# MuJoCo XML motor ctrlrange도 [-7.5, 7.5]이므로 최종 effort limit을 동일하게 둔다.
HAND_TAU_LIMIT = 7.5
POSE_KP = 0.4
POSE_KD = 0.05
POSE_PD_LIMIT = 0.25

# Groped grasp policy.
# real 쪽 GraspPolicy는 별도 damping/ramp 없이 alpha1과 Jacobian transpose를 바로 쓴다.
GROPED_FORCE_TARGET = 1.0       # alpha1_cmd로 런타임 변경 가능
GROPED_TAU_LIMIT = 5.0
GROPED_FORCE_DIRECTION_SIGN = 1.0

# 손가락 추가 시 실물 코드와 동일하게 새 손가락만 짧게 pre-grasp 자세로 보낸 뒤
# grasp torque에 포함한다. 전체 손을 초기화하지 않는다.
FINGER_SWITCH_VIA_THREE_DELAY = 0.5
ADD_FINGER_PRE_GRASP_DELAY = 0.15
ADD_FINGER_PRE_GRASP_KP_SCALE = 1.4
ADD_FINGER_PRE_GRASP_KD_SCALE = 1.2

# real/dg5f_grasp_control.grasp_policy와 동일한 per-finger scale/충돌 회피 항.
FINGER_FORCE_SCALE = {
    1: 1.0,
    2: 1.0,
    3: 0.9,
    4: 0.55,
    5: 0.35,
}

COLLISION_AVOID_PAIRS = [
    (3, 4),
    (4, 5),
]

MIN_TIP_DISTANCE = 0.018
COLLISION_REPEL_GAIN = 100.0
COLLISION_REPEL_LIMIT = 0.8

BASE_JOINT_TAU_LIMIT = {
    12: 0.8,
    16: 0.5,
}

# Tesollo Jacobian torque와 MuJoCo actuator 방향이 다르면 여기서 보정한다.
# 우선 전부 +1로 두고, 특정 조인트만 반대로 움직이면 해당 원소를 -1로 바꾼다.
GRASP_TAU_SIGN = np.ones(20, dtype=np.float64)

# Cg에서 엄지 쪽으로 당긴 virtual centroid를 제어 중심으로 사용한다.
# 0.0: 단순 평균 중심 Cg
# 0.3: Cg에서 엄지 방향으로 30% 이동한 Cv
THUMB_CENTROID_BIAS = 0.5

# Tesollo에서 제공한 fingertip FK를 기준으로 grasp Jacobian을 계산한다.
# Jacobian은 FK를 유한차분해서 얻는다. 해석식 직접 이식보다 오타 위험이 낮다.
USE_TESOLLO_KINEMATICS = True
TESOLLO_JAC_EPS = 1e-6

# Test object in the fixed XML.
GRASP_OBJECT_JOINT = "grasp_object_free"
GRASP_OBJECT_POS = np.array([0.0782, -0.0288, 0.1400], dtype=np.float64)
GRASP_OBJECT_QUAT = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)

# contact가 한 번만 튀는 경우를 막기 위한 안정 카운트
CONTACT_STABLE_COUNT = 5

# Contact debug
DEBUG_CONTACTS = True
DEBUG_CONTACT_PRINT_PERIOD = 0.5


# =====================================================
# Joint / body / geom naming
# =====================================================
JOINT_NAMES = [
    "joint_1_1", "joint_1_2", "joint_1_3", "joint_1_4",
    "joint_2_1", "joint_2_2", "joint_2_3", "joint_2_4",
    "joint_3_1", "joint_3_2", "joint_3_3", "joint_3_4",
    "joint_4_1", "joint_4_2", "joint_4_3", "joint_4_4",
    "joint_5_1", "joint_5_2", "joint_5_3", "joint_5_4",
]

FINGER_JOINT_INDEX = {
    1: [0, 1, 2, 3],       # thumb
    2: [4, 5, 6, 7],       # index
    3: [8, 9, 10, 11],     # middle
    4: [12, 13, 14, 15],   # ring
    5: [16, 17, 18, 19],   # pinky
}

FINGER_GRASP_JOINT_INDEX = {
    1: [1, 2, 3],       # thumb curl joints
    2: [5, 6, 7],       # index curl joints
    3: [9, 10, 11],     # middle curl joints
    4: [13, 14, 15],    # ring curl joints
    5: [17, 18, 19],    # pinky curl joints
}

HAND_NORMAL_POSE = np.array([
     0.0288,  0.3665, -0.7102, -0.2981,  # thumb
     0.2559, -0.0094,  0.5128,  0.4074,  # index
    -0.0351,  0.0054,  0.4328,  0.5548,  # middle
    -0.1873, -0.0157,  0.4594,  0.5142,  # ring
    -0.1471, -0.3410, -0.1508,  0.8662,  # pinky
], dtype=np.float64)

HAND_PRE_GRASP_POSE = np.array([
    -0.0431,  1.3725, -0.7102, -0.2981,  # thumb
     0.2559, -0.0094,  0.5128,  0.4074,  # index
    -0.0351,  0.0054,  0.4328,  0.5548,  # middle
    -0.1873, -0.0157,  0.4594,  0.5142,  # ring
    -0.1471, -0.3410, -0.1508,  0.8662,  # pinky
], dtype=np.float64)

TIP_BODY_NAME = {
    1: "link_1_tip",
    2: "link_2_tip",
    3: "link_3_tip",
    4: "link_4_tip",
    5: "link_5_tip",
}

TIP_GEOM_NAME = {
    1: "link_1_tip_geom",
    2: "link_2_tip_geom",
    3: "link_3_tip_geom",
    4: "link_4_tip_geom",
    5: "link_5_tip_geom",
}


# =====================================================
# State machine
# =====================================================
class GraspState(Enum):
    NORMAL_POSE = 0
    PRE_GRASP_POSE = 1
    GROPED_GRASP = 2
    HOLD = 3


# =====================================================
# ROS command interface
# =====================================================
class GraspSimCommandNode(Node):
    def __init__(self):
        super().__init__("grasp_sim")
        self._lock = threading.Lock()
        self.pending_finger_command = None
        self.pending_force_target = None

        self.create_subscription(
            Int32,
            COMMAND_TOPIC,
            self.finger_command_cb,
            10,
        )
        self.create_subscription(
            Float64,
            ALPHA1_TOPIC,
            self.alpha1_cb,
            10,
        )

    def finger_command_cb(self, msg):
        command = int(msg.data)
        if command < 0 or command > 5:
            self.get_logger().warn(
                f"Ignore invalid finger command: {command}. Use 0, 1, 2, 3, 4, or 5."
            )
            return

        with self._lock:
            self.pending_finger_command = command
        self.get_logger().info(f"RX finger_count_cmd={command}")

    def alpha1_cb(self, msg):
        force_target = float(msg.data)
        if force_target < 0.0:
            self.get_logger().warn(
                f"Ignore invalid alpha1 command: {force_target}. Use alpha1 >= 0.0."
            )
            return

        with self._lock:
            self.pending_force_target = force_target
        self.get_logger().info(f"RX alpha1_cmd={force_target:.4f}")

    def take_pending_finger_command(self):
        with self._lock:
            command = self.pending_finger_command
            self.pending_finger_command = None
        return command

    def take_pending_force_target(self):
        with self._lock:
            force_target = self.pending_force_target
            self.pending_force_target = None
        return force_target


# =====================================================
# MuJoCo model
# =====================================================
model = mujoco.MjModel.from_xml_path(XML_PATH)
data = mujoco.MjData(model)
gdata = mujoco.MjData(model)


# =====================================================
# ID utilities
# =====================================================
def joint_id(name: str) -> int:
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
    if jid < 0:
        raise RuntimeError(f"Joint not found: {name}")
    return jid


def optional_joint_id(name: str) -> int:
    return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)


def body_id(name: str) -> int:
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
    if bid < 0:
        raise RuntimeError(f"Body not found: {name}")
    return bid


def geom_id(name: str) -> int:
    gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
    if gid < 0:
        raise RuntimeError(f"Geom not found: {name}")
    return gid


def geom_name(gid: int) -> str:
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, gid)
    return "" if name is None else name


def body_name(bid: int) -> str:
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, bid)
    return "" if name is None else name


QPOS_ADR = np.array([
    model.jnt_qposadr[joint_id(name)] for name in JOINT_NAMES
], dtype=int)

DOF_ADR = np.array([
    model.jnt_dofadr[joint_id(name)] for name in JOINT_NAMES
], dtype=int)

GRASP_OBJECT_JOINT_ID = optional_joint_id(GRASP_OBJECT_JOINT)
GRASP_OBJECT_QPOS_ADR = None
GRASP_OBJECT_DOF_ADR = None

if GRASP_OBJECT_JOINT_ID >= 0:
    GRASP_OBJECT_QPOS_ADR = model.jnt_qposadr[GRASP_OBJECT_JOINT_ID]
    GRASP_OBJECT_DOF_ADR = model.jnt_dofadr[GRASP_OBJECT_JOINT_ID]

TIP_BODY_ID = {
    finger: body_id(TIP_BODY_NAME[finger])
    for finger in FINGER_ORDER
}

TIP_GEOM_ID = {
    finger: geom_id(TIP_GEOM_NAME[finger])
    for finger in FINGER_ORDER
}

TIP_BODY_GEOM_IDS = {
    finger: [
        gid for gid in range(model.ngeom)
        if model.geom_bodyid[gid] == TIP_BODY_ID[finger]
    ]
    for finger in FINGER_ORDER
}


# =====================================================
# Contact utilities
# =====================================================
def detect_finger_contacts():
    """
    tip body에 붙은 geom이 다른 geom과 접촉했는지 확인한다.

    기존처럼 link_X_tip_geom 하나만 보면 실제 collision geom 이름이
    다를 때 접촉을 놓칠 수 있으므로, tip body의 모든 geom을 본다.

    반환:
        contact_now[finger] = True / False
    """
    contact_now = {finger: False for finger in FINGER_ORDER}

    for ci in range(data.ncon):
        c = data.contact[ci]
        g1 = c.geom1
        g2 = c.geom2

        for finger in USE_FINGERS:
            tip_gids = TIP_BODY_GEOM_IDS[finger]

            if g1 in tip_gids or g2 in tip_gids:
                contact_now[finger] = True

    return contact_now


def print_contact_debug():
    """
    MuJoCo contact buffer를 그대로 출력한다.
    data.ncon이 0이면 XML collision/filter 문제일 가능성이 크고,
    ncon은 있는데 finger contact가 0이면 감지 대상 geom 문제다.
    """
    print(f"[CONTACT_DEBUG] ncon={data.ncon}")

    for ci in range(data.ncon):
        c = data.contact[ci]
        g1 = c.geom1
        g2 = c.geom2
        b1 = model.geom_bodyid[g1]
        b2 = model.geom_bodyid[g2]

        print(
            "  "
            f"#{ci:02d} "
            f"{geom_name(g1)}(body={body_name(b1)}) <-> "
            f"{geom_name(g2)}(body={body_name(b2)}) "
            f"dist={c.dist:.6f}"
        )


def print_model_contact_setup():
    print("[TIP CONTACT SETUP]")
    for finger in USE_FINGERS:
        tip_body = TIP_BODY_ID[finger]
        tip_geom = TIP_GEOM_ID[finger]
        body_geom_ids = TIP_BODY_GEOM_IDS[finger]
        body_geom_names = [geom_name(gid) for gid in body_geom_ids]

        print(
            f"  F{finger}: "
            f"tip_body={body_name(tip_body)} | "
            f"declared_tip_geom={geom_name(tip_geom)} "
            f"contype={model.geom_contype[tip_geom]} "
            f"conaffinity={model.geom_conaffinity[tip_geom]} | "
            f"body_geoms={body_geom_names}"
        )


def tesollo_forward_kinematics(q):
    """
    Tesollo 제공 ForwardKinematics()를 Python으로 옮긴 함수.
    반환값은 finger 번호 1~5에 대한 fingertip position, shape=(5, 3).
    단위는 원본 식과 동일하게 meter로 본다.
    """
    s = np.sin(q)
    c = np.cos(q)
    x = np.zeros(15, dtype=np.float64)

    x[0] = (
        -0.0318 * s[1] * s[2] * s[3]
        + 0.0318 * s[1] * c[2] * c[3]
        + 0.0334 * s[1] * c[2]
        + 0.0381 * s[1]
        + 0.0279
    )
    x[1] = (
        0.0318 * (s[0] * s[2] - c[0] * c[1] * c[2]) * c[3]
        + 0.0318 * (s[0] * c[2] + s[2] * c[0] * c[1]) * s[3]
        + 0.0334 * s[0] * s[2]
        - 0.0334 * c[0] * c[1] * c[2]
        - 0.0381 * c[0] * c[1]
        - 0.018
    )
    x[2] = (
        0.0318 * (s[0] * s[2] * c[1] - c[0] * c[2]) * s[3]
        + 0.0318 * (-s[0] * c[1] * c[2] - s[2] * c[0]) * c[3]
        - 0.0334 * s[0] * c[1] * c[2]
        - 0.0381 * s[0] * c[1]
        - 0.0334 * s[2] * c[0]
        + 0.0298
    )

    x[3] = (
        0.01978 * (-s[5] * s[6] + c[5] * c[6]) * s[7]
        + 0.01978 * (s[5] * c[6] + s[6] * c[5]) * c[7]
        + 0.0334 * s[5] * c[6]
        + 0.0334 * s[5]
        + 0.0334 * s[6] * c[5]
        + 0.0143
    )
    x[4] = (
        0.01978 * (s[4] * s[5] * s[6] - s[4] * c[5] * c[6]) * c[7]
        + 0.01978 * (s[4] * s[5] * c[6] + s[4] * s[6] * c[5]) * s[7]
        + 0.0334 * s[4] * s[5] * s[6]
        - 0.0334 * s[4] * c[5] * c[6]
        - 0.0334 * s[4] * c[5]
        - 0.02415 * s[4]
        - 0.028
    )
    x[5] = (
        0.01978 * (-s[5] * s[6] * c[4] + c[4] * c[5] * c[6]) * c[7]
        + 0.01978 * (-s[5] * c[4] * c[6] - s[6] * c[4] * c[5]) * s[7]
        - 0.0334 * s[5] * s[6] * c[4]
        + 0.0334 * c[4] * c[5] * c[6]
        + 0.0334 * c[4] * c[5]
        + 0.02415 * c[4]
        + 0.08365
    )

    x[6] = (
        0.0209 * (-s[10] * s[9] + c[10] * c[9]) * s[11]
        + 0.0209 * (s[10] * c[9] + s[9] * c[10]) * c[11]
        + 0.0334 * s[10] * c[9]
        + 0.0334 * s[9] * c[10]
        + 0.0334 * s[9]
        + 0.0143
    )
    x[7] = (
        0.0209 * (s[10] * s[8] * s[9] - s[8] * c[10] * c[9]) * c[11]
        + 0.0209 * (s[10] * s[8] * c[9] + s[8] * s[9] * c[10]) * s[11]
        + 0.0334 * s[10] * s[8] * s[9]
        - 0.0334 * s[8] * c[10] * c[9]
        - 0.0334 * s[8] * c[9]
        - 0.02415 * s[8]
        - 0.005
    )
    x[8] = (
        0.0209 * (-s[10] * s[9] * c[8] + c[10] * c[8] * c[9]) * c[11]
        + 0.0209 * (-s[10] * c[8] * c[9] - s[9] * c[10] * c[8]) * s[11]
        - 0.0334 * s[10] * s[9] * c[8]
        + 0.0334 * c[10] * c[8] * c[9]
        + 0.0334 * c[8] * c[9]
        + 0.02415 * c[8]
        + 0.08865
    )

    x[9] = (
        0.0209 * (-s[13] * s[14] + c[13] * c[14]) * s[15]
        + 0.0209 * (s[13] * c[14] + s[14] * c[13]) * c[15]
        + 0.0334 * s[13] * c[14]
        + 0.0334 * s[13]
        + 0.0334 * s[14] * c[13]
        + 0.0143
    )
    x[10] = (
        0.0209 * (s[12] * s[13] * s[14] - s[12] * c[13] * c[14]) * c[15]
        + 0.0209 * (s[12] * s[13] * c[14] + s[12] * s[14] * c[13]) * s[15]
        + 0.0334 * s[12] * s[13] * s[14]
        - 0.0334 * s[12] * c[13] * c[14]
        - 0.0334 * s[12] * c[13]
        - 0.02415 * s[12]
        + 0.018
    )
    x[11] = (
        0.0209 * (-s[13] * s[14] * c[12] + c[12] * c[13] * c[14]) * c[15]
        + 0.0209 * (-s[13] * c[12] * c[14] - s[14] * c[12] * c[13]) * s[15]
        - 0.0334 * s[13] * s[14] * c[12]
        + 0.0334 * c[12] * c[13] * c[14]
        + 0.0334 * c[12] * c[13]
        + 0.02415 * c[12]
        + 0.08065
    )

    x[12] = (
        0.0318 * (-s[16] * s[17] * s[18] + c[16] * c[18]) * s[19]
        + 0.0318 * (s[16] * s[17] * c[18] + s[18] * c[16]) * c[19]
        + 0.0334 * s[16] * s[17] * c[18]
        + 0.0272 * s[16] * s[17]
        - 0.02445 * s[16]
        + 0.0334 * s[18] * c[16]
        + 0.013
    )
    x[13] = (
        0.0318 * (s[16] * s[18] - s[17] * c[16] * c[18]) * c[19]
        + 0.0318 * (s[16] * c[18] + s[17] * s[18] * c[16]) * s[19]
        + 0.0334 * s[16] * s[18]
        - 0.0334 * s[17] * c[16] * c[18]
        - 0.0272 * s[17] * c[16]
        + 0.02445 * c[16]
        + 0.01805
    )
    x[14] = (
        -0.0318 * s[18] * s[19] * c[17]
        + 0.0318 * c[17] * c[18] * c[19]
        + 0.0334 * c[17] * c[18]
        + 0.0272 * c[17]
        + 0.0671
    )

    return x.reshape(5, 3)


def tesollo_tip_position(q, finger):
    return tesollo_forward_kinematics(q)[finger - 1].copy()


def tesollo_tip_jacobian(q, finger):
    idxs = FINGER_JOINT_INDEX[finger]
    J = np.zeros((3, 4), dtype=np.float64)

    for col, qidx in enumerate(idxs):
        q_plus = q.copy()
        q_minus = q.copy()
        q_plus[qidx] += TESOLLO_JAC_EPS
        q_minus[qidx] -= TESOLLO_JAC_EPS
        J[:, col] = (
            tesollo_tip_position(q_plus, finger)
            - tesollo_tip_position(q_minus, finger)
        ) / (2.0 * TESOLLO_JAC_EPS)

    return J


def get_tip_position(finger: int) -> np.ndarray:
    """
    제어용 fingertip position.
    USE_TESOLLO_KINEMATICS=True이면 Tesollo FK 기준 position을 쓴다.
    """
    if USE_TESOLLO_KINEMATICS:
        q = data.qpos[QPOS_ADR].copy()
        return tesollo_tip_position(q, finger)

    bid = TIP_BODY_ID[finger]
    return data.xpos[bid].copy()


def get_tip_jacobian(finger: int):
    """
    제어용 fingertip translational Jacobian을 구한다.

    반환:
        J_full: None 또는 3 x nv
        J_finger: 3 x 4
    """
    if USE_TESOLLO_KINEMATICS:
        q = data.qpos[QPOS_ADR].copy()
        return None, tesollo_tip_jacobian(q, finger)

    bid = TIP_BODY_ID[finger]

    jacp = np.zeros((3, model.nv))
    jacr = np.zeros((3, model.nv))

    mujoco.mj_jacBody(model, data, jacp, jacr, bid)

    finger_joint_indices = FINGER_JOINT_INDEX[finger]
    finger_dof_indices = DOF_ADR[finger_joint_indices]

    J_finger = jacp[:, finger_dof_indices]

    return jacp, J_finger


def normalize(v: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    n = np.linalg.norm(v)
    if n < eps:
        return np.zeros_like(v)
    return v / n


def reset_object_pose():
    if GRASP_OBJECT_QPOS_ADR is None:
        return

    data.qpos[GRASP_OBJECT_QPOS_ADR:GRASP_OBJECT_QPOS_ADR + 3] = GRASP_OBJECT_POS
    data.qpos[GRASP_OBJECT_QPOS_ADR + 3:GRASP_OBJECT_QPOS_ADR + 7] = GRASP_OBJECT_QUAT
    data.qvel[GRASP_OBJECT_DOF_ADR:GRASP_OBJECT_DOF_ADR + 6] = 0.0


# =====================================================
# Initial pose
# =====================================================
def set_initial_pose():
    """
    시뮬레이션 시작 상태를 평소 핸드 자세로 둔다.
    준비 자세 이동은 토크 PD가 아니라 qpos 보간으로 처리한다.
    """
    mujoco.mj_resetData(model, data)
    data.qvel[:] = 0.0
    data.ctrl[:] = 0.0
    data.qfrc_applied[:] = 0.0
    data.qpos[QPOS_ADR] = HAND_NORMAL_POSE
    reset_object_pose()
    mujoco.mj_forward(model, data)


# =====================================================
# Pose setter
# =====================================================
def set_hand_pose_kinematic(target_q):
    """
    준비 자세 단계에서 손가락이 튀는 것을 막기 위해 qpos를 직접 세팅한다.
    실제 토크 제어 검증은 GROPED_GRASP 상태에서만 수행한다.
    """
    data.qpos[QPOS_ADR] = target_q
    data.qvel[DOF_ADR] = 0.0
    data.ctrl[:] = 0.0
    data.qfrc_applied[:] = 0.0
    reset_object_pose()
    mujoco.mj_forward(model, data)


def lerp_pose(q0, q1, s):
    s = np.clip(s, 0.0, 1.0)
    return (1.0 - s) * q0 + s * q1


def pose_pd(q_target, q, qdot, kp=POSE_KP, kd=POSE_KD, limit=POSE_PD_LIMIT):
    err = q_target - q
    pd = kp * err - kd * qdot
    return np.clip(pd, -limit, limit), err


def inactive_pre_grasp_pd(active_fingers, qdot, pre_grasp_fingers=None):
    """
    real GraspRealRunner._inactive_pre_grasp_pd와 같은 역할.

    - active_fingers: 현재 grasp torque를 받는 손가락
    - pre_grasp_fingers: 새로 추가되기 직전이라 HAND_PRE_GRASP_POSE로 잠깐 보내는 손가락
    - 나머지 inactive finger는 실물 코드처럼 0 rad 기준으로 PD 유지
    """
    q = data.qpos[QPOS_ADR]
    inactive_pd = np.zeros(20, dtype=np.float64)
    active_set = set(active_fingers)
    pre_grasp_set = set(pre_grasp_fingers or [])

    for finger, idxs in FINGER_JOINT_INDEX.items():
        if finger in active_set:
            continue

        idxs = np.asarray(idxs, dtype=int)
        if finger in pre_grasp_set:
            target = HAND_PRE_GRASP_POSE[idxs]
            kp = POSE_KP * ADD_FINGER_PRE_GRASP_KP_SCALE
            kd = POSE_KD * ADD_FINGER_PRE_GRASP_KD_SCALE
        else:
            target = np.zeros(len(idxs), dtype=np.float64)
            kp = POSE_KP
            kd = POSE_KD

        pd, _ = pose_pd(
            target,
            q[idxs],
            qdot[idxs],
            kp=kp,
            kd=kd,
            limit=POSE_PD_LIMIT,
        )
        inactive_pd[idxs] = pd

    return inactive_pd


# =====================================================
# Gravity compensation
# =====================================================
def apply_gravity_compensation():
    """
    qfrc_applied에 순수 중력보상만 넣는다.
    actuator ctrl에는 넣지 않는다.
    """
    gdata.qpos[:] = data.qpos[:]
    gdata.qvel[:] = 0.0
    gdata.qacc[:] = 0.0
    mujoco.mj_forward(model, gdata)

    data.qfrc_applied[:] = gdata.qfrc_bias


# =====================================================
# Groped grasp controller
# =====================================================
def compute_groped_centroids():
    """
    선택된 fingertip 위치들로 geometric centroid Cg를 만들고,
    교수님 조언대로 제어용 virtual centroid Cv는 엄지 쪽으로 당긴다.
    """
    tip_pos = {
        finger: get_tip_position(finger)
        for finger in USE_FINGERS
    }

    points = np.array([tip_pos[finger] for finger in USE_FINGERS])
    cg = np.mean(points, axis=0)

    if 1 in tip_pos:
        thumb_pos = tip_pos[1]
        cv = cg + THUMB_CENTROID_BIAS * (thumb_pos - cg)
    else:
        cv = cg

    return cg, cv, tip_pos


def compute_groped_alpha_and_forces():
    """
    real/dg5f_grasp_control의 GraspPolicy.calc_alpha_and_forces와 같은 구조.

    alpha1_cmd로 받은 값은 첫 번째 finger의 alpha1로 쓰고,
    나머지 alpha는 fingertip-Cv 거리 비와 힘 평형 조건으로 자동 계산한다.
    """
    tip_pos = {
        finger: get_tip_position(finger)
        for finger in USE_FINGERS
    }

    points = np.array([tip_pos[finger] for finger in USE_FINGERS])
    cg = np.mean(points, axis=0)

    if 1 in tip_pos:
        thumb_pos = tip_pos[1]
        cv = cg + THUMB_CENTROID_BIAS * (thumb_pos - cg)
    else:
        cv = cg

    dist = {}
    fhat = {}
    for finger in USE_FINGERS:
        diff = cv - tip_pos[finger]
        dist[finger] = max(np.linalg.norm(diff), 1e-6)
        fhat[finger] = GROPED_FORCE_DIRECTION_SIGN * diff / dist[finger]

    alpha = {}
    if len(USE_FINGERS) == 2:
        alpha[USE_FINGERS[0]] = GROPED_FORCE_TARGET
        alpha[USE_FINGERS[1]] = GROPED_FORCE_TARGET
        return alpha, fhat, cg, cv, tip_pos

    first_finger = USE_FINGERS[0]
    pivot_finger = USE_FINGERS[-1]
    alpha[first_finger] = GROPED_FORCE_TARGET

    for finger in USE_FINGERS[1:-1]:
        alpha[finger] = dist[first_finger] / dist[finger] * GROPED_FORCE_TARGET

    force_sum = np.zeros(3, dtype=np.float64)
    for finger in USE_FINGERS[:-1]:
        force_sum += alpha[finger] * fhat[finger]

    alpha[pivot_finger] = np.linalg.norm(force_sum)
    return alpha, fhat, cg, cv, tip_pos


def calc_collision_avoidance_forces(tip_pos):
    repel = {
        finger: np.zeros(3, dtype=np.float64)
        for finger in tip_pos
    }

    for finger_a, finger_b in COLLISION_AVOID_PAIRS:
        if finger_a not in tip_pos or finger_b not in tip_pos:
            continue

        diff = tip_pos[finger_a] - tip_pos[finger_b]
        dist = np.linalg.norm(diff)
        if dist < 1e-9 or dist >= MIN_TIP_DISTANCE:
            continue

        direction = diff / dist
        mag = COLLISION_REPEL_GAIN * (MIN_TIP_DISTANCE - dist)
        mag = min(mag, COLLISION_REPEL_LIMIT)

        repel[finger_a] += mag * direction
        repel[finger_b] -= mag * direction

    return repel


def compute_groped_grasp_tau(_elapsed=None):
    """
    real/dg5f_grasp_control.grasp_policy.GraspPolicy.calc_grasp_tau와 같은 구조.

        alpha, fhat, cg, cv, tip_pos = calc_alpha_and_forces(q)
        total_force_i = finger_scale_i * alpha_i * fhat_i + repel_i
        tau_i = J_i^T total_force_i

    손가락 command가 바뀌어도 전체 qpos를 초기화하지 않고,
    현재 q에서 바로 새 USE_FINGERS 집합에 대한 torque만 계산한다.
    """
    tau = np.zeros(20, dtype=np.float64)
    alpha_raw, fhat, cg, cv, tip_pos = compute_groped_alpha_and_forces()
    repel = calc_collision_avoidance_forces(tip_pos)

    for finger in USE_FINGERS:
        _, J_finger = get_tip_jacobian(finger)
        finger_joint_indices = np.asarray(FINGER_JOINT_INDEX[finger], dtype=int)

        grasp_force = FINGER_FORCE_SCALE[finger] * alpha_raw[finger] * fhat[finger]
        total_force = grasp_force + repel[finger]
        tau_finger = J_finger.T @ total_force

        tau[finger_joint_indices] = (
            tau_finger * GRASP_TAU_SIGN[finger_joint_indices]
        )

    tau = np.clip(tau, -GROPED_TAU_LIMIT, GROPED_TAU_LIMIT)

    for joint_idx, limit in BASE_JOINT_TAU_LIMIT.items():
        tau[joint_idx] = np.clip(tau[joint_idx], -limit, limit)

    return tau, alpha_raw


def make_contact_state():
    return (
        {finger: 0 for finger in FINGER_ORDER},
        {finger: False for finger in FINGER_ORDER},
    )


def apply_finger_command(command: int, now: float, current_state: GraspState):
    """
    real GraspRealRunner의 finger_count 처리와 같은 의미로 동작한다.

    중요:
    - command 0만 PRE_GRASP_POSE로 보낸다.
    - command 1~5는 전체 손을 초기화하지 않는다.
    - 현재 자세에서 USE_FINGERS 집합만 바꾸고 GROPED_GRASP torque를 바로 적용한다.
    - 새로 추가된 손가락만 짧게 pre-grasp PD를 받은 뒤 grasp torque에 들어간다.
    """
    global FINGER_COMMAND, USE_FINGERS
    global ACTIVE_FINGER_COUNT
    global deferred_finger_count, deferred_finger_count_at
    global adding_finger_target_count, adding_pre_grasp_fingers, adding_ready_at
    global skip_add_pre_grasp_once

    command_count = command

    if command == 0:
        ACTIVE_FINGER_COUNT = 0
        deferred_finger_count = None
        deferred_finger_count_at = None
        adding_finger_target_count = None
        adding_pre_grasp_fingers = []
        adding_ready_at = None
        skip_add_pre_grasp_once = False

        if current_state != GraspState.PRE_GRASP_POSE:
            print("[COMMAND] finger_count=0 -> PRE_GRASP_POSE")
            return GraspState.PRE_GRASP_POSE, now

        print("[COMMAND] finger_count=0, already PRE_GRASP_POSE")
        return current_state, now

    if (
        ACTIVE_FINGER_COUNT in (1, 2)
        and command in (1, 2)
        and command != ACTIVE_FINGER_COUNT
    ):
        command_count = 3
        deferred_finger_count = command
        deferred_finger_count_at = None
        print(
            f"[COMMAND] finger_count={command} requested, "
            f"switch via 3"
        )

    prev_fingers = set(USE_FINGERS) if ACTIVE_FINGER_COUNT > 0 else set()
    target_fingers = selected_fingers(command_count)
    new_fingers = set(target_fingers)
    added = sorted(new_fingers - prev_fingers)

    if (
        ACTIVE_FINGER_COUNT > 0
        and added
        and not skip_add_pre_grasp_once
    ):
        adding_finger_target_count = command_count
        adding_pre_grasp_fingers = added
        adding_ready_at = now + ADD_FINGER_PRE_GRASP_DELAY
        print(
            f"[COMMAND] finger_count={command_count} prepare added fingers "
            f"{added} at PRE_GRASP for {ADD_FINGER_PRE_GRASP_DELAY:.2f}s"
        )
        return GraspState.GROPED_GRASP, now

    skip_add_pre_grasp_once = False
    USE_FINGERS = target_fingers
    FINGER_COMMAND = command_count
    ACTIVE_FINGER_COUNT = command_count

    adding_finger_target_count = None
    adding_pre_grasp_fingers = []
    adding_ready_at = None

    removed = sorted(prev_fingers - new_fingers)

    if (
        deferred_finger_count is not None
        and deferred_finger_count_at is None
        and command_count == 3
    ):
        deferred_finger_count_at = now + FINGER_SWITCH_VIA_THREE_DELAY
        print(
            f"[COMMAND] hold finger_count=3 for "
            f"{FINGER_SWITCH_VIA_THREE_DELAY:.2f}s before "
            f"finger_count={deferred_finger_count}"
        )

    print(
        f"[COMMAND] finger_count={command_count} -> "
        f"GROPED_GRASP, USE_FINGERS={USE_FINGERS}, "
        f"added={added}, removed={removed}"
    )
    print_model_contact_setup()
    return GraspState.GROPED_GRASP, now


# =====================================================
# Main
# =====================================================
# contact 상태
finger_contact_counter, finger_contact_confirmed = make_contact_state()

# real GraspRealRunner의 finger_count runtime 상태와 같은 의미.
ACTIVE_FINGER_COUNT = 0
deferred_finger_count = None
deferred_finger_count_at = None
adding_finger_target_count = None
adding_pre_grasp_fingers = []
adding_ready_at = None
skip_add_pre_grasp_once = False

state = GraspState.NORMAL_POSE

set_initial_pose()
mujoco.mj_forward(model, data)
start_time = time.time()
state_start_time = start_time
last_print_time = 0.0
last_contact_debug_time = 0.0
ros_node = None
ros_executor = None
ros_spin_thread = None

if rclpy is not None:
    rclpy.init(args=None)
    ros_node = GraspSimCommandNode()
    ros_executor = SingleThreadedExecutor()
    ros_executor.add_node(ros_node)
    ros_spin_thread = threading.Thread(
        target=ros_executor.spin,
        daemon=True,
        name="grasp_sim_ros_spin",
    )
    ros_spin_thread.start()

    def shutdown_ros_node():
        if ros_executor is not None:
            ros_executor.shutdown()
        if ros_node is not None:
            ros_node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

    atexit.register(shutdown_ros_node)
else:
    print("[WARN] rclpy/std_msgs not found. ROS command topics are disabled.")

print("=" * 80)
print("[GRASP SIM START]")
print(f"XML_PATH          : {XML_PATH}")
print(f"FINGER_COMMAND    : {FINGER_COMMAND}")
print(f"ACTIVE_COUNT      : {ACTIVE_FINGER_COUNT}")
print(f"USE_FINGERS       : {USE_FINGERS}")
print(f"ALPHA1            : {GROPED_FORCE_TARGET}")
print(f"GROPED_TAU_LIMIT  : {GROPED_TAU_LIMIT}")
print(f"HAND_TAU_LIMIT    : {HAND_TAU_LIMIT}")
print(f"POSE_KP/KD/LIMIT  : {POSE_KP}, {POSE_KD}, {POSE_PD_LIMIT}")
print(f"ROS_ENABLED       : {ros_node is not None}")
print(f"ROS_DOMAIN_ID     : {os.environ.get('ROS_DOMAIN_ID', '<unset>')}")
print(f"COMMAND_TOPIC     : {COMMAND_TOPIC}")
print(f"ALPHA1_TOPIC      : {ALPHA1_TOPIC}")
print(
    "ROTATION_MATRIX   : "
    f"{ROTATION_MATRIX_IDENTITY.reshape(-1).astype(int).tolist()} "
    "(fixed identity)"
)
print(f"THUMB_CENTROID_BIAS: {THUMB_CENTROID_BIAS}")
print(f"FORCE_DIRECTION_SIGN: {GROPED_FORCE_DIRECTION_SIGN}")
print("AUTO_SEQUENCE     : disabled, real-like topic command mode")
print("[COMMAND] 0=pre-grasp, 1=thumb+index, 2=thumb+middle, 3=thumb+index+middle, 4=+ring, 5=+pinky")
print("=" * 80)
print_model_contact_setup()


with mujoco.viewer.launch_passive(model, data) as viewer:
    try:
        viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = True
        viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = True
    except Exception as e:
        print(f"[WARN] contact visualization option was not enabled: {e}")

    while viewer.is_running():
        loop_t0 = time.time()
        now = time.time()
        elapsed = now - start_time
        state_elapsed = now - state_start_time
        alpha_debug = None

        if ros_node is not None:
            new_force_target = ros_node.take_pending_force_target()
            if new_force_target is not None:
                GROPED_FORCE_TARGET = new_force_target
                print(f"[COMMAND] alpha1={GROPED_FORCE_TARGET:.4f}")

            new_finger_command = ros_node.take_pending_finger_command()
        else:
            new_finger_command = None

        if (
            new_finger_command is None
            and adding_finger_target_count is not None
            and adding_ready_at is not None
            and now >= adding_ready_at
        ):
            new_finger_command = adding_finger_target_count
            adding_finger_target_count = None
            adding_pre_grasp_fingers = []
            adding_ready_at = None
            skip_add_pre_grasp_once = True

        if (
            new_finger_command is None
            and deferred_finger_count is not None
            and deferred_finger_count_at is not None
            and adding_finger_target_count is None
            and now >= deferred_finger_count_at
        ):
            new_finger_command = deferred_finger_count
            deferred_finger_count = None
            deferred_finger_count_at = None

        elif (
            new_finger_command is not None
            and deferred_finger_count is not None
            and not skip_add_pre_grasp_once
        ):
            deferred_finger_count = None
            deferred_finger_count_at = None

        if new_finger_command is not None:
            state, state_start_time = apply_finger_command(
                new_finger_command,
                now,
                state,
            )
            state_elapsed = 0.0
            finger_contact_counter, finger_contact_confirmed = make_contact_state()

        # -------------------------------------------------
        # 1. Gravity compensation
        # -------------------------------------------------
        apply_gravity_compensation()

        # -------------------------------------------------
        # 2. Contact detection
        # -------------------------------------------------
        contact_now = detect_finger_contacts()

        for finger in USE_FINGERS:
            if contact_now[finger]:
                finger_contact_counter[finger] += 1
            else:
                finger_contact_counter[finger] = 0

            if (
                not finger_contact_confirmed[finger]
                and finger_contact_counter[finger] >= CONTACT_STABLE_COUNT
            ):
                finger_contact_confirmed[finger] = True

                print(f"[CONTACT] finger {finger} confirmed")

        num_confirmed = sum(
            1 for finger in USE_FINGERS
            if finger_contact_confirmed[finger]
        )

        # -------------------------------------------------
        # 3. State machine
        # -------------------------------------------------
        q = data.qpos[QPOS_ADR]
        qdot = data.qvel[DOF_ADR]

        if state == GraspState.NORMAL_POSE:
            tau_ctrl, _ = pose_pd(
                HAND_NORMAL_POSE,
                q,
                qdot,
                kp=POSE_KP,
                kd=POSE_KD,
                limit=POSE_PD_LIMIT,
            )

        elif state == GraspState.PRE_GRASP_POSE:
            tau_ctrl, _ = pose_pd(
                HAND_PRE_GRASP_POSE,
                q,
                qdot,
                kp=POSE_KP,
                kd=POSE_KD,
                limit=POSE_PD_LIMIT,
            )

        elif state == GraspState.GROPED_GRASP:
            tau_ctrl, alpha_debug = compute_groped_grasp_tau(state_elapsed)
            tau_ctrl += inactive_pre_grasp_pd(
                USE_FINGERS,
                qdot,
                adding_pre_grasp_fingers,
            )

        elif state == GraspState.HOLD:
            tau_ctrl = np.zeros(20)

        else:
            tau_ctrl = np.zeros(20)

        tau_ctrl = np.clip(tau_ctrl, -HAND_TAU_LIMIT, HAND_TAU_LIMIT)

        # -------------------------------------------------
        # 4. Apply actuator torque
        # -------------------------------------------------
        data.ctrl[:] = tau_ctrl

        # -------------------------------------------------
        # 5. Simulation step
        # -------------------------------------------------
        mujoco.mj_step(model, data)
        viewer.sync()

        # -------------------------------------------------
        # 6. Debug print
        # -------------------------------------------------
        if time.time() - last_print_time > 0.5:
            last_print_time = time.time()
            alpha_now, _, cg, cv, _ = compute_groped_alpha_and_forces()
            alpha_str = "{" + ", ".join(
                f"F{finger}:{alpha_now[finger]:.3f}"
                for finger in USE_FINGERS
            ) + "}"

            contact_str = " ".join(
                f"F{finger}:{int(finger_contact_confirmed[finger])}"
                for finger in USE_FINGERS
            )

            print(
                f"[{state.name}] "
                f"t={elapsed:5.2f}  "
                f"state_t={state_elapsed:5.2f}  "
                f"ncon={data.ncon}  "
                f"active_count={ACTIVE_FINGER_COUNT}  "
                f"contacts={num_confirmed}/{len(USE_FINGERS)}  "
                f"alpha1={GROPED_FORCE_TARGET:.4f}  "
                f"alpha={alpha_str}  "
                f"tau_max={np.max(np.abs(tau_ctrl)):.4f}  "
                f"{contact_str}  "
                f"Cg={np.round(cg, 3)}  "
                f"Cv={np.round(cv, 3)}"
            )

        if (
            DEBUG_CONTACTS
            and time.time() - last_contact_debug_time > DEBUG_CONTACT_PRINT_PERIOD
        ):
            last_contact_debug_time = time.time()
            print_contact_debug()

        # -------------------------------------------------
        # 7. Real-time sync
        # -------------------------------------------------
        dt = model.opt.timestep - (time.time() - loop_t0)
        if dt > 0:
            time.sleep(dt)