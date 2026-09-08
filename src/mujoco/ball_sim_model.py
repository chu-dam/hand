#!/usr/bin/env python3

from pathlib import Path
import time

import mujoco
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
URDF_PATH = (
    ROOT
    / "src/vendor/dg5f_s_description/urdf/dg5fs_right_w_mount.urdf"
)
MESH_DIR = (
    ROOT
    / "src/vendor/dg5f_s_description/meshes/dg5fs_right"
)
SOURCE_PACKAGE = ROOT / "src/real/dg5f_grasp_control"

import sys

if str(SOURCE_PACKAGE) not in sys.path:
    sys.path.insert(0, str(SOURCE_PACKAGE))

from dg5f_grasp_control.config import RuntimeConfig
from dg5f_grasp_control.friction_params_right import (
    HAND_FRIC_B,
    HAND_FRIC_COULOMB_SCALE,
    HAND_FRIC_FC,
)
from dg5f_grasp_control.grasp_controller import GraspController
from dg5f_grasp_control.grasp_policy import polygon_centroid_3d
from dg5f_grasp_control.hand_model import HAND_JOINT_NAMES
from dg5f_grasp_control.kinematics import (
    set_hand_side as set_kinematics_hand_side,
    tip_position,
)
from dg5f_grasp_control.poses import RIGHT_HAND_BLIND_GRASP_PRE_ROTATION_POSE


BALL_RADIUS_M = 0.0375
BALL_MASS_KG = 0.063
BALL_INITIAL_Z_OFFSET_M = 0.003
BALL_CENTER_FINGERS = (1, 2, 3, 4, 5)
HAND_TO_WORLD_ROTATION = np.array(
    [[0.4695, 0.0, -0.8829], [0.0, 1.0, 0.0], [0.8829, 0.0, 0.4695]],
    dtype=np.float64,
)
WORLD_GRAVITY = np.array([0.0, 0.0, -9.81], dtype=np.float64)
POSE_KP = 1.1
POSE_KD = 0.14
POSE_PD_LIMIT_NM = 0.50
EFFORT_LIMIT_NM = 7.5
BLIND_GRASP_DELAY_SEC = 1.0
BLIND_GRASP_ALPHA1_N = 3.0
BALL_RELEASE_MIN_CONTACT_FINGERS = 3
CONTROL_DT_SEC = 0.005
PHYSICS_STEPS_PER_CONTROL = 10
VIEWER_DT_SEC = 1.0 / 60.0
SILICONE_PLA_SLIDING_FRICTION = 2.0
TORSIONAL_FRICTION = 0.005
ROLLING_FRICTION = 0.0001
JOINT_FRICTION_COMPENSATION_FRACTION = 0.10
SOFT_TIP_SOLREF = [0.02, 1.0]
SOFT_TIP_SOLIMP = [0.9, 0.98, 0.002, 0.5, 2.0]
HARD_FINGER_SOLREF = [0.005, 1.0]
HARD_FINGER_SOLIMP = [0.95, 0.99, 0.0005, 0.5, 2.0]
SOFT_TIP_MARGIN_M = 0.0


def _model_assets():
    return {path.name: path.read_bytes() for path in MESH_DIR.glob("*.STL")}


def _sphere_mesh(radius, latitude_count=12, longitude_count=24):
    vertices = [[0.0, 0.0, radius]]
    for latitude in range(1, latitude_count):
        phi = np.pi * latitude / latitude_count
        for longitude in range(longitude_count):
            theta = 2.0 * np.pi * longitude / longitude_count
            vertices.append(
                [
                    radius * np.sin(phi) * np.cos(theta),
                    radius * np.sin(phi) * np.sin(theta),
                    radius * np.cos(phi),
                ]
            )
    vertices.append([0.0, 0.0, -radius])

    faces = []
    first_ring = 1
    for longitude in range(longitude_count):
        next_longitude = (longitude + 1) % longitude_count
        faces.append([0, first_ring + longitude, first_ring + next_longitude])
    for latitude in range(latitude_count - 2):
        ring = first_ring + latitude * longitude_count
        next_ring = ring + longitude_count
        for longitude in range(longitude_count):
            next_longitude = (longitude + 1) % longitude_count
            faces.extend(
                [
                    [ring + longitude, next_ring + longitude, next_ring + next_longitude],
                    [ring + longitude, next_ring + next_longitude, ring + next_longitude],
                ]
            )
    south = len(vertices) - 1
    last_ring = south - longitude_count
    for longitude in range(longitude_count):
        next_longitude = (longitude + 1) % longitude_count
        faces.append([south, last_ring + next_longitude, last_ring + longitude])
    return np.asarray(vertices), np.asarray(faces)


def build_model():
    """Load the right tactile-tip hand and add a free 75 mm ball."""
    set_kinematics_hand_side("right")
    spec = mujoco.MjSpec.from_file(str(URDF_PATH), assets=_model_assets())
    demo_quat = np.empty(4, dtype=np.float64)
    mujoco.mju_mat2Quat(demo_quat, HAND_TO_WORLD_ROTATION.reshape(-1))
    spec.body("link_mount").quat = demo_quat
    ball_vertices, ball_faces = _sphere_mesh(BALL_RADIUS_M)
    spec.add_mesh(
        name="ball_mesh",
        uservert=ball_vertices.reshape(-1),
        userface=ball_faces.reshape(-1),
        smoothnormal=True,
    )
    ball = spec.worldbody.add_body(name="ball")
    ball.add_freejoint(name="ball_free")
    ball.add_geom(
        name="ball_geom",
        type=mujoco.mjtGeom.mjGEOM_MESH,
        meshname="ball_mesh",
        mass=BALL_MASS_KG,
        friction=[SILICONE_PLA_SLIDING_FRICTION, TORSIONAL_FRICTION, ROLLING_FRICTION],
        condim=6,
        priority=1,
        margin=SOFT_TIP_MARGIN_M,
        solref=SOFT_TIP_SOLREF,
        solimp=SOFT_TIP_SOLIMP,
        rgba=[0.95, 0.65, 0.10, 1.0],
    )
    model = spec.compile()
    model.opt.timestep = 0.0005
    model.opt.gravity[:] = WORLD_GRAVITY
    model.opt.enableflags |= int(mujoco.mjtEnableBit.mjENBL_MULTICCD)
    model.opt.solver = mujoco.mjtSolver.mjSOL_NEWTON
    model.opt.iterations = 100
    model.opt.noslip_iterations = 20
    model.opt.impratio = 10.0
    model.dof_damping[:20] = (
        JOINT_FRICTION_COMPENSATION_FRACTION * HAND_FRIC_B
    )
    model.dof_frictionloss[:20] = (
        JOINT_FRICTION_COMPENSATION_FRACTION
        * HAND_FRIC_COULOMB_SCALE
        * HAND_FRIC_FC
    )
    model.dof_armature[:20] = 0.0002
    _configure_collision_filters(model)
    set_ball_suspended(model, True)
    return model


def _configure_collision_filters(model):
    """Collide different fingers and the ball, but not links in one chain."""
    hand_bits = ((1 << 6) - 1) & ~1  # five fingers; fixed palm mounts excluded
    ball_bit = 1 << 6
    ball_geom_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_GEOM, "ball_geom"
    )

    for geom_id in range(model.ngeom):
        if geom_id == ball_geom_id:
            model.geom_contype[geom_id] = ball_bit
            model.geom_conaffinity[geom_id] = hand_bits
            model.geom_friction[geom_id] = [
                SILICONE_PLA_SLIDING_FRICTION,
                TORSIONAL_FRICTION,
                ROLLING_FRICTION,
            ]
            continue

        body_name = mujoco.mj_id2name(
            model,
            mujoco.mjtObj.mjOBJ_BODY,
            int(model.geom_bodyid[geom_id]),
        ) or ""
        finger = 0
        if body_name.startswith("link_"):
            part = body_name.split("_", 2)[1]
            if part.isdigit() and 1 <= int(part) <= 5:
                finger = int(part)
        if finger == 0:
            model.geom_contype[geom_id] = 0
            model.geom_conaffinity[geom_id] = 0
            continue
        own_bit = 1 << finger
        model.geom_contype[geom_id] = own_bit
        model.geom_conaffinity[geom_id] = hand_bits & ~own_bit
        model.geom_solref[geom_id] = HARD_FINGER_SOLREF
        model.geom_solimp[geom_id] = HARD_FINGER_SOLIMP
        mesh_id = int(model.geom_dataid[geom_id])
        if mesh_id >= 0 and mujoco.mj_id2name(
            model, mujoco.mjtObj.mjOBJ_MESH, mesh_id
        ) == "tactile_tip":
            model.geom_friction[geom_id] = [
                SILICONE_PLA_SLIDING_FRICTION,
                TORSIONAL_FRICTION,
                ROLLING_FRICTION,
            ]


def set_ball_suspended(model, suspended):
    ball_body_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_BODY, "ball"
    )
    model.body_gravcomp[ball_body_id] = 1.0 if suspended else 0.0


def hold_ball_pose(model, data, ball_qpos):
    """Pin all six ball DOFs while waiting for grasp activation."""
    ball_joint_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_JOINT, "ball_free"
    )
    qpos_address = int(model.jnt_qposadr[ball_joint_id])
    dof_address = int(model.jnt_dofadr[ball_joint_id])
    data.qpos[qpos_address : qpos_address + 7] = ball_qpos
    data.qvel[dof_address : dof_address + 6] = 0.0
    mujoco.mj_forward(model, data)


def ball_touches_hand(model, data):
    return bool(ball_contact_fingers(model, data))


def ball_contact_fingers(model, data):
    ball_body_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_BODY, "ball"
    )
    fingers = set()
    for contact in data.contact:
        body_1 = int(model.geom_bodyid[contact.geom1])
        body_2 = int(model.geom_bodyid[contact.geom2])
        if (body_1 == ball_body_id) == (body_2 == ball_body_id):
            continue
        hand_body = body_2 if body_1 == ball_body_id else body_1
        body_name = mujoco.mj_id2name(
            model, mujoco.mjtObj.mjOBJ_BODY, hand_body
        ) or ""
        parts = body_name.split("_", 2)
        if len(parts) >= 2 and parts[1].isdigit():
            fingers.add(int(parts[1]))
    return fingers


def _joint_addresses(model):
    qpos_addresses = []
    dof_addresses = []
    for name in HAND_JOINT_NAMES:
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if joint_id < 0:
            raise RuntimeError(f"joint not found: {name}")
        qpos_addresses.append(model.jnt_qposadr[joint_id])
        dof_addresses.append(model.jnt_dofadr[joint_id])
    return np.asarray(qpos_addresses, dtype=int), np.asarray(dof_addresses, dtype=int)


def tactile_tip_center(model, data, finger):
    """Return the tactile STL bounding-box center after fingertip FK."""
    body_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_BODY, f"link_{finger}_4"
    )
    if body_id < 0:
        raise RuntimeError(f"finger body not found: link_{finger}_4")

    for geom_id in np.flatnonzero(model.geom_bodyid == body_id):
        mesh_id = int(model.geom_dataid[geom_id])
        if mesh_id < 0:
            continue
        mesh_name = mujoco.mj_id2name(
            model, mujoco.mjtObj.mjOBJ_MESH, mesh_id
        )
        if mesh_name != "tactile_tip":
            continue

        start = int(model.mesh_vertadr[mesh_id])
        vertices = model.mesh_vert[
            start : start + int(model.mesh_vertnum[mesh_id])
        ]
        local_center = 0.5 * (vertices.min(axis=0) + vertices.max(axis=0))
        rotation = data.geom_xmat[geom_id].reshape(3, 3)
        return data.geom_xpos[geom_id] + rotation @ local_center

    raise RuntimeError(f"tactile_tip mesh not found for finger {finger}")


def reset_at_grasp_pose(model, data):
    """Reset hand and place the ball at the grasp controller's FK centroid."""
    mujoco.mj_resetData(model, data)
    qpos_addresses, _ = _joint_addresses(model)
    data.qpos[qpos_addresses] = (
        RIGHT_HAND_BLIND_GRASP_PRE_ROTATION_POSE
    )
    mujoco.mj_forward(model, data)

    tip_centers = np.asarray(
        [tactile_tip_center(model, data, finger) for finger in BALL_CENTER_FINGERS]
    )
    controller_tips_world = np.asarray(
        [
            HAND_TO_WORLD_ROTATION @ tip_position(data.qpos[qpos_addresses], finger)
            for finger in BALL_CENTER_FINGERS
        ]
    )
    ball_center = polygon_centroid_3d(controller_tips_world)
    ball_center[2] += BALL_INITIAL_Z_OFFSET_M
    ball_joint = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_JOINT, "ball_free"
    )
    ball_qpos = int(model.jnt_qposadr[ball_joint])
    data.qpos[ball_qpos : ball_qpos + 3] = ball_center
    data.qpos[ball_qpos + 3 : ball_qpos + 7] = [1.0, 0.0, 0.0, 0.0]
    mujoco.mj_forward(model, data)
    return ball_center, tip_centers


def apply_gravity_compensated_pose_hold(model, data):
    """Hold the grasp pose with gravity feed-forward and a small pose PD."""
    qpos_addresses, dof_addresses = _joint_addresses(model)
    pose_tau = np.clip(
        POSE_KP * (RIGHT_HAND_BLIND_GRASP_PRE_ROTATION_POSE - data.qpos[qpos_addresses])
        - POSE_KD * data.qvel[dof_addresses],
        -POSE_PD_LIMIT_NM,
        POSE_PD_LIMIT_NM,
    )
    data.qfrc_applied[:] = 0.0
    data.qfrc_applied[dof_addresses] = np.clip(
        data.qfrc_bias[dof_addresses] + pose_tau,
        -EFFORT_LIMIT_NM,
        EFFORT_LIMIT_NM,
    )


def apply_gravity_compensated_grasp(model, data, controller, now):
    qpos_addresses, dof_addresses = _joint_addresses(model)
    output = controller.step(
        data.qpos[qpos_addresses], data.qvel[dof_addresses], now
    )
    data.qfrc_applied[:] = 0.0
    data.qfrc_applied[dof_addresses] = np.clip(
        data.qfrc_bias[dof_addresses] + output.tau,
        -EFFORT_LIMIT_NM,
        EFFORT_LIMIT_NM,
    )
    return output

def main():
    model = build_model()
    data = mujoco.MjData(model)
    center, _ = reset_at_grasp_pose(model, data)
    print(f"ball center world [m]: {np.round(center, 6).tolist()}")
    controller = GraspController(
        RuntimeConfig(
            hand_side="right",
            use_finger_count=5,
            alpha1=BLIND_GRASP_ALPHA1_N,
        ),
        log=print,
    )
    grasp_started = False
    ball_suspended = True
    ball_joint_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_JOINT, "ball_free"
    )
    ball_qpos_address = int(model.jnt_qposadr[ball_joint_id])
    suspended_ball_qpos = data.qpos[
        ball_qpos_address : ball_qpos_address + 7
    ].copy()
    next_control_wall_time = time.perf_counter()
    next_viewer_sync_time = 0.0
    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            if ball_suspended:
                hold_ball_pose(model, data, suspended_ball_qpos)
            if not grasp_started and data.time >= BLIND_GRASP_DELAY_SEC:
                controller.sync_joint_state(data.qpos[:20])
                controller.apply_grasp_type(5, data.time)
                grasp_started = True
                print(f"5-finger blind grasp enabled: alpha1={BLIND_GRASP_ALPHA1_N:.1f} N")
            if (
                grasp_started
                and ball_suspended
                and len(ball_contact_fingers(model, data))
                >= BALL_RELEASE_MIN_CONTACT_FINGERS
            ):
                set_ball_suspended(model, False)
                ball_suspended = False
                print("3+ fingers have contact: ball released")
            if grasp_started:
                apply_gravity_compensated_grasp(
                    model, data, controller, data.time
                )
            else:
                apply_gravity_compensated_pose_hold(model, data)
            for _ in range(PHYSICS_STEPS_PER_CONTROL):
                if ball_suspended:
                    hold_ball_pose(model, data, suspended_ball_qpos)
                mujoco.mj_step(model, data)

            if data.time >= next_viewer_sync_time:
                viewer.sync()
                next_viewer_sync_time = data.time + VIEWER_DT_SEC

            next_control_wall_time += CONTROL_DT_SEC
            remaining = next_control_wall_time - time.perf_counter()
            if remaining > 0.0:
                time.sleep(remaining)
            elif remaining < -CONTROL_DT_SEC:
                next_control_wall_time = time.perf_counter()


if __name__ == "__main__":
    import mujoco.viewer

    main()
