"""Build the ROS debug message without changing the control calculation."""

from typing import Optional

import numpy as np
from dg5f_grasp_interfaces.msg import GraspDebug
from geometry_msgs.msg import Point, Vector3

from dg5f_grasp_control.hand_model import FINGER_COUNT, JOINT_COUNT
from dg5f_grasp_control.kinematics import tip_position


FINGER_IDS = tuple(range(1, FINGER_COUNT + 1))


def _point(values) -> Point:
    values = np.asarray(values, dtype=np.float64)
    return Point(x=float(values[0]), y=float(values[1]), z=float(values[2]))


def _vector(values) -> Vector3:
    values = np.asarray(values, dtype=np.float64)
    return Vector3(x=float(values[0]), y=float(values[1]), z=float(values[2]))


def _vectors_for_all_fingers(values):
    zero = np.zeros(3, dtype=np.float64)
    return [_vector(values.get(finger, zero)) for finger in FINGER_IDS]


def current_grasp_type(controller) -> int:
    """Return the effective high-level grasp command represented by the FSM."""

    if controller.state == "NORMAL_POSE":
        return -1
    if controller.state == "PRE_GRASP_POSE":
        return 0
    return int(controller.active_finger_count)


def build_grasp_debug_message(
    *,
    controller,
    q,
    output,
    controller_torques,
    commanded_efforts,
    stamp,
    frame_id: str,
    teaching_mode: bool = False,
    controller_state: Optional[str] = None,
    controller_phase: Optional[str] = None,
) -> GraspDebug:
    """Convert one control-loop snapshot to the fixed five-finger wire format."""

    q = np.asarray(q, dtype=np.float64)
    controller_torques = np.asarray(controller_torques, dtype=np.float64)
    commanded_efforts = np.asarray(commanded_efforts, dtype=np.float64)
    if q.shape != (JOINT_COUNT,):
        raise ValueError(f"q must have shape ({JOINT_COUNT},)")
    if controller_torques.shape != (JOINT_COUNT,):
        raise ValueError(f"controller_torques must have shape ({JOINT_COUNT},)")
    if commanded_efforts.shape != (JOINT_COUNT,):
        raise ValueError(f"commanded_efforts must have shape ({JOINT_COUNT},)")

    if output is None:
        fingertip_positions = {
            finger: tip_position(q, finger)
            for finger in FINGER_IDS
        }
        alpha = {}
        cg = np.zeros(3, dtype=np.float64)
        cv = np.zeros(3, dtype=np.float64)
        grasp_forces = {}
        translation_forces = {}
        rotation_forces = {}
        center_hold_forces = {}
        collision_forces = {}
        total_forces = {}
    else:
        fingertip_positions = output.fingertip_positions
        alpha = output.alpha
        cg = output.cg
        cv = output.cv
        grasp_forces = output.grasp_forces
        translation_forces = output.translation_forces
        rotation_forces = output.rotation_forces
        center_hold_forces = output.center_hold_forces
        collision_forces = output.collision_forces
        total_forces = output.total_forces

    relative_translation_phase = (
        output.relative_translation_phase
        if output is not None
        else controller.relative_translation_phase
    )
    relative_translation_start = (
        output.relative_translation_start_centroid
        if output is not None
        else controller.relative_translation_start_centroid
    )
    relative_translation_target = (
        output.relative_translation_target_centroid
        if output is not None
        else controller.relative_translation_target_centroid
    )
    relative_translation_delta = (
        output.relative_translation_delta
        if output is not None
        else controller.relative_translation_delta
    )
    relative_translation_error = (
        output.relative_translation_error
        if output is not None
        else controller.relative_translation_error
    )
    relative_translation_velocity = (
        output.relative_translation_centroid_velocity
        if output is not None
        else controller.relative_translation_centroid_velocity
    )
    relative_translation_force = (
        output.relative_translation_command_force
        if output is not None
        else controller.relative_translation_command_force
    )

    if controller_state is None:
        controller_state = output.state if output is not None else controller.state
    if controller_phase is None:
        relative_rotation_phase = (
            output.relative_rotation_phase
            if output is not None
            else controller.relative_rotation_phase
        )
        if relative_translation_phase != "idle":
            controller_phase = relative_translation_phase
        elif relative_rotation_phase != "idle":
            controller_phase = relative_rotation_phase
        else:
            controller_phase = (
                output.g7_phase
                if output is not None
                else controller.grasp_type7_phase
            )

    message = GraspDebug()
    message.header.stamp = stamp
    message.header.frame_id = str(frame_id)
    message.finger_ids = list(FINGER_IDS)
    message.fingertip_positions = [
        _point(
            fingertip_positions[finger]
            if finger in fingertip_positions
            else tip_position(q, finger)
        )
        for finger in FINGER_IDS
    ]
    message.geometric_centroid = _point(cg)
    message.virtual_centroid = _point(cv)
    message.relative_translation_start_centroid = _point(
        relative_translation_start
    )
    message.relative_translation_target_centroid = _point(
        relative_translation_target
    )
    message.relative_translation_delta = _vector(relative_translation_delta)
    message.relative_translation_error = _vector(relative_translation_error)
    message.relative_translation_centroid_velocity = _vector(
        relative_translation_velocity
    )
    message.relative_translation_command_force = _vector(
        relative_translation_force
    )
    message.relative_translation_phase = str(relative_translation_phase)
    message.alpha = [float(alpha.get(finger, 0.0)) for finger in FINGER_IDS]
    message.grasp_forces = _vectors_for_all_fingers(grasp_forces)
    message.translation_forces = _vectors_for_all_fingers(translation_forces)
    message.rotation_forces = _vectors_for_all_fingers(rotation_forces)
    message.center_hold_forces = _vectors_for_all_fingers(center_hold_forces)
    message.collision_forces = _vectors_for_all_fingers(collision_forces)
    message.total_forces = _vectors_for_all_fingers(total_forces)
    message.controller_torques = controller_torques.tolist()
    message.commanded_efforts = commanded_efforts.tolist()
    message.grasp_type = current_grasp_type(controller)
    message.pose_type = int(controller.pose_type)
    message.teaching_mode = bool(teaching_mode)
    message.controller_state = str(controller_state)
    message.controller_phase = str(controller_phase)
    return message
