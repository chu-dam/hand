#!/usr/bin/env python3

import os
import time
from enum import Enum

import numpy as np
import mujoco
import mujoco.viewer


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
# 사용할 손가락 개수
# 2: thumb + index
# 3: thumb + index + middle
# 4: thumb + index + middle + ring
# 5: all fingers
NUM_USE_FINGERS = 5

# 손가락 번호 정의
# 1: thumb
# 2: index
# 3: middle
# 4: ring
# 5: pinky
FINGER_ORDER = [1, 2, 3, 4, 5]

FINGER_SELECTIONS = {
    2: [1, 2],
    3: [1, 2, 3],
    4: [1, 2, 3, 4],
    5: [1, 2, 3, 4, 5],
}

if NUM_USE_FINGERS not in FINGER_SELECTIONS:
    raise ValueError("NUM_USE_FINGERS must be one of: 2, 3, 4, 5")

USE_FINGERS = FINGER_SELECTIONS[NUM_USE_FINGERS]


# =====================================================
# Control parameters
# =====================================================
# Initial pose sequence
NORMAL_POSE_TIME = 3.0
PRE_GRASP_POSE_TIME = 1.0

# 논문 식 기반 groped grasp:
# tau_i = -D_i qdot_i + alpha_i J_i^T fhat_i
GROPED_DAMPING = 0.01
GROPED_FORCE_TARGET = 0.03       # 처음에는 작게 시작
GROPED_FORCE_RAMP_TIME = 2.0
GROPED_TAU_LIMIT = 0.04
GROPED_FORCE_DIRECTION_SIGN = 1.0

# Tesollo Jacobian torque와 MuJoCo actuator 방향이 다르면 여기서 보정한다.
# 우선 전부 +1로 두고, 특정 조인트만 반대로 움직이면 해당 원소를 -1로 바꾼다.
GRASP_TAU_SIGN = np.ones(20, dtype=np.float64)

# Cg에서 엄지 쪽으로 당긴 virtual centroid를 제어 중심으로 사용한다.
# 0.0: 단순 평균 중심 Cg
# 0.3: Cg에서 엄지 방향으로 30% 이동한 Cv
THUMB_CENTROID_BIAS = 0.3

# Tesollo에서 제공한 fingertip FK를 기준으로 grasp Jacobian을 계산한다.
# Jacobian은 FK를 유한차분해서 얻는다. 해석식 직접 이식보다 오타 위험이 낮다.
USE_TESOLLO_KINEMATICS = True
TESOLLO_JAC_EPS = 1e-6

# Test object in the fixed XML.
GRASP_OBJECT_JOINT = "grasp_object_free"
GRASP_OBJECT_POS = np.array([0.0662, -0.0288, 0.1488], dtype=np.float64)
GRASP_OBJECT_QUAT = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)

# 몇 개 손가락이 닿았는지 확인하기 위한 표시용 조건.
# 제어기는 접촉 전/후 모두 같은 groped grasp 식을 사용한다.
MIN_CONTACT_FINGERS = NUM_USE_FINGERS

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


def compute_groped_grasp_tau(elapsed):
    """
    접촉 전/후를 나누지 않고 같은 식으로 제어한다.

        tau_i = -D_i qdot_i + alpha_i J_i^T fhat_i
        fhat_i = (Cv - Pi) / ||Cv - Pi||

    접촉 전에는 tip들이 Cv 방향으로 모이면서 닫히고,
    접촉 후에는 같은 힘 방향으로 grasp를 유지한다.
    """
    qdot = data.qvel[DOF_ADR]
    tau = np.zeros(20)

    _, cv, tip_pos = compute_groped_centroids()

    force_scale = min(elapsed / GROPED_FORCE_RAMP_TIME, 1.0)
    alpha = GROPED_FORCE_TARGET * force_scale

    for finger in USE_FINGERS:
        p_tip = tip_pos[finger]
        f_hat = GROPED_FORCE_DIRECTION_SIGN * normalize(cv - p_tip)

        _, J_finger = get_tip_jacobian(finger)

        finger_joint_indices = FINGER_JOINT_INDEX[finger]
        grasp_joint_indices = FINGER_GRASP_JOINT_INDEX[finger]
        grasp_cols = [
            finger_joint_indices.index(joint_idx)
            for joint_idx in grasp_joint_indices
        ]

        J_grasp = J_finger[:, grasp_cols]
        tau_grasp = J_grasp.T @ (alpha * f_hat)

        # damping 추가
        grasp_dof_indices = np.array(grasp_joint_indices, dtype=int)
        tau_damp = -GROPED_DAMPING * qdot[grasp_dof_indices]

        tau[grasp_dof_indices] = (
            tau_grasp + tau_damp
        ) * GRASP_TAU_SIGN[grasp_dof_indices]

    tau = np.clip(tau, -GROPED_TAU_LIMIT, GROPED_TAU_LIMIT)

    return tau


# =====================================================
# Main
# =====================================================
# contact 상태
finger_contact_counter = {finger: 0 for finger in FINGER_ORDER}
finger_contact_confirmed = {finger: False for finger in FINGER_ORDER}

state = GraspState.NORMAL_POSE

set_initial_pose()
mujoco.mj_forward(model, data)
start_time = time.time()
state_start_time = start_time
last_print_time = 0.0
last_contact_debug_time = 0.0

print("=" * 80)
print("[GRASP SIM START]")
print(f"XML_PATH          : {XML_PATH}")
print(f"NUM_USE_FINGERS   : {NUM_USE_FINGERS}")
print(f"USE_FINGERS       : {USE_FINGERS}")
print(f"MIN_CONTACT       : {MIN_CONTACT_FINGERS}")
print(f"THUMB_CENTROID_BIAS: {THUMB_CENTROID_BIAS}")
print(f"FORCE_DIRECTION_SIGN: {GROPED_FORCE_DIRECTION_SIGN}")
print(f"NORMAL_POSE_TIME  : {NORMAL_POSE_TIME}")
print(f"PRE_GRASP_TIME    : {PRE_GRASP_POSE_TIME}")
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

        # -------------------------------------------------
        # 1. Gravity compensation
        # -------------------------------------------------
        if state == GraspState.GROPED_GRASP:
            apply_gravity_compensation()
        else:
            data.qfrc_applied[:] = 0.0

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
        if state == GraspState.NORMAL_POSE:
            set_hand_pose_kinematic(HAND_NORMAL_POSE)
            tau_ctrl = np.zeros(20)

            if state_elapsed >= NORMAL_POSE_TIME:
                state = GraspState.PRE_GRASP_POSE
                state_start_time = now
                print("[STATE] NORMAL_POSE -> PRE_GRASP_POSE")

        elif state == GraspState.PRE_GRASP_POSE:
            s = state_elapsed / PRE_GRASP_POSE_TIME
            pre_q = lerp_pose(HAND_NORMAL_POSE, HAND_PRE_GRASP_POSE, s)
            set_hand_pose_kinematic(pre_q)
            tau_ctrl = np.zeros(20)

            if state_elapsed >= PRE_GRASP_POSE_TIME:
                set_hand_pose_kinematic(HAND_PRE_GRASP_POSE)
                finger_contact_counter = {finger: 0 for finger in FINGER_ORDER}
                finger_contact_confirmed = {finger: False for finger in FINGER_ORDER}
                state = GraspState.GROPED_GRASP
                state_start_time = now
                print("[STATE] PRE_GRASP_POSE -> GROPED_GRASP")

        elif state == GraspState.GROPED_GRASP:
            tau_ctrl = compute_groped_grasp_tau(state_elapsed)

        elif state == GraspState.HOLD:
            tau_ctrl = np.zeros(20)

        else:
            tau_ctrl = np.zeros(20)

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
            cg, cv, _ = compute_groped_centroids()

            contact_str = " ".join(
                f"F{finger}:{int(finger_contact_confirmed[finger])}"
                for finger in USE_FINGERS
            )

            print(
                f"[{state.name}] "
                f"t={elapsed:5.2f}  "
                f"state_t={state_elapsed:5.2f}  "
                f"ncon={data.ncon}  "
                f"contacts={num_confirmed}/{MIN_CONTACT_FINGERS}  "
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