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
        translation_torques = np.zeros(JOINT_COUNT, dtype=np.float64)
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
        translation_torques = output.translation_torques
        rotation_forces = output.rotation_forces
        center_hold_forces = output.center_hold_forces
        collision_forces = output.collision_forces
        total_forces = output.total_forces

    relative_translation_phase = (
        output.relative_translation_phase
        if output is not None
        else controller.relative_translation_phase
    )
    relative_rotation_phase = (
        output.relative_rotation_phase
        if output is not None
        else controller.relative_rotation_phase
    )
    rotation_start_fingertips = controller.relative_rotation_start_fingertips
    relative_rotation_start_centroid = (
        np.mean(
            np.stack(list(rotation_start_fingertips.values())),
            axis=0,
        )
        if rotation_start_fingertips
        else np.zeros(3, dtype=np.float64)
    )
    relative_rotation_pivot = (
        output.relative_rotation_pivot
        if output is not None
        else controller.relative_rotation_pivot
    )
    relative_rotation_axis = (
        output.relative_rotation_axis
        if output is not None
        else controller.relative_rotation_axis
    )
    relative_rotation_target_rad = (
        output.relative_rotation_target_rad
        if output is not None
        else controller.relative_rotation_target_rad
    )
    relative_rotation_current_rad = (
        output.relative_rotation_current_rad
        if output is not None
        else controller.relative_rotation_current_rad
    )
    relative_rotation_error_rad = (
        output.relative_rotation_error_rad
        if output is not None
        else controller.relative_rotation_error_rad
    )
    relative_rotation_angular_velocity = (
        output.relative_rotation_angular_velocity
        if output is not None
        else controller.relative_rotation_angular_velocity
    )
    relative_rotation_command_moment = (
        output.relative_rotation_command_moment
        if output is not None
        else controller.relative_rotation_command_moment
    )
    relative_rotation_control_mode = (
        output.relative_rotation_control_mode
        if output is not None
        else "idle"
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
    relative_translation_torque_target = (
        output.relative_translation_torque_target
        if output is not None
        else 0.0
    )
    relative_translation_force_scale = (
        output.relative_translation_force_scale
        if output is not None
        else 0.0
    )
    relative_translation_control_mode = (
        output.relative_translation_control_mode
        if output is not None
        else "idle"
    )
    relative_translation_dls_sigma_min = (
        output.relative_translation_dls_sigma_min
        if output is not None
        else 0.0
    )
    relative_translation_dls_condition = (
        output.relative_translation_dls_condition
        if output is not None
        else 0.0
    )
    relative_translation_joint_error = (
        output.relative_translation_joint_error
        if output is not None
        else np.zeros(JOINT_COUNT, dtype=np.float64)
    )
    relative_translation_position_torques = (
        output.relative_translation_position_torques
        if output is not None
        else np.zeros(JOINT_COUNT, dtype=np.float64)
    )
    relative_translation_nullspace_grasp_torques = (
        output.relative_translation_nullspace_grasp_torques
        if output is not None
        else np.zeros(JOINT_COUNT, dtype=np.float64)
    )
    inactive_collision_min_clearance_m = (
        output.inactive_collision_min_clearance_m
        if output is not None
        else controller.inactive_collision_min_clearance_m
    )
    inactive_collision_avoidance_offsets_rad = (
        output.inactive_collision_avoidance_offsets_rad
        if output is not None
        else controller.inactive_collision_avoidance_offsets_rad
    )
    inactive_collision_avoidance_active = (
        output.inactive_collision_avoidance_active
        if output is not None
        else controller.inactive_collision_avoidance_active
    )

    if controller_state is None:
        controller_state = output.state if output is not None else controller.state
    if controller_phase is None:
        continuous_phase = getattr(
            controller,
            "continuous_rotation_phase",
            "idle",
        )
        if continuous_phase != "idle":
            controller_phase = continuous_phase
        elif controller.state == "CARD_GRASP":
            controller_phase = controller.card_phase
        elif relative_translation_phase != "idle":
            controller_phase = relative_translation_phase
        elif relative_rotation_phase != "idle":
            controller_phase = relative_rotation_phase
        else:
            controller_phase = "idle"

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
    message.blind_sphere_estimate_valid = bool(
        controller.blind_sphere_estimate_valid
    )
    message.blind_sphere_center = _point(controller.blind_sphere_center)
    message.blind_sphere_effective_radius_m = float(
        controller.blind_sphere_effective_radius_m
    )
    message.blind_sphere_fit_rms_error_m = float(
        controller.blind_sphere_fit_rms_error_m
    )
    message.blind_four_finger_polygon_area_m2 = float(
        controller.blind_four_finger_polygon_area_m2
    )
    message.relative_rotation_start_centroid = _point(
        relative_rotation_start_centroid
    )
    message.relative_rotation_pivot = _point(relative_rotation_pivot)
    message.relative_rotation_axis = _vector(relative_rotation_axis)
    message.relative_rotation_target_rad = float(relative_rotation_target_rad)
    message.relative_rotation_current_rad = float(relative_rotation_current_rad)
    message.relative_rotation_error_rad = float(relative_rotation_error_rad)
    message.relative_rotation_angular_velocity = float(
        relative_rotation_angular_velocity
    )
    message.relative_rotation_command_moment = float(
        relative_rotation_command_moment
    )
    message.relative_rotation_phase = str(relative_rotation_phase)
    message.relative_rotation_control_mode = str(
        relative_rotation_control_mode
    )
    message.relative_rotation_center_error = _vector(
        np.zeros(3, dtype=np.float64)
    )
    message.relative_rotation_dls_sigma_min = 0.0
    message.relative_rotation_dls_condition = 0.0
    message.relative_rotation_center_joint_error = np.zeros(
        JOINT_COUNT,
        dtype=np.float64,
    ).tolist()
    message.relative_rotation_center_position_torques = np.zeros(
        JOINT_COUNT,
        dtype=np.float64,
    ).tolist()
    message.relative_rotation_nullspace_torques = np.zeros(
        JOINT_COUNT,
        dtype=np.float64,
    ).tolist()
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
    message.relative_translation_torque_target = float(
        relative_translation_torque_target
    )
    message.relative_translation_force_scale = float(
        relative_translation_force_scale
    )
    message.relative_translation_phase = str(relative_translation_phase)
    message.relative_translation_control_mode = str(
        relative_translation_control_mode
    )
    message.relative_translation_dls_sigma_min = float(
        relative_translation_dls_sigma_min
    )
    message.relative_translation_dls_condition = float(
        relative_translation_dls_condition
    )
    message.relative_translation_joint_error = np.asarray(
        relative_translation_joint_error,
        dtype=np.float64,
    ).tolist()
    message.relative_translation_position_torques = np.asarray(
        relative_translation_position_torques,
        dtype=np.float64,
    ).tolist()
    message.relative_translation_nullspace_grasp_torques = np.asarray(
        relative_translation_nullspace_grasp_torques,
        dtype=np.float64,
    ).tolist()
    message.inactive_collision_min_clearance_m = float(
        inactive_collision_min_clearance_m
    )
    message.inactive_collision_avoidance_offsets_rad = np.asarray(
        inactive_collision_avoidance_offsets_rad,
        dtype=np.float64,
    ).tolist()
    message.inactive_collision_avoidance_active = [
        bool(value) for value in inactive_collision_avoidance_active
    ]
    message.alpha = [float(alpha.get(finger, 0.0)) for finger in FINGER_IDS]
    message.grasp_forces = _vectors_for_all_fingers(grasp_forces)
    message.translation_forces = _vectors_for_all_fingers(translation_forces)
    message.rotation_forces = _vectors_for_all_fingers(rotation_forces)
    message.center_hold_forces = _vectors_for_all_fingers(center_hold_forces)
    message.collision_forces = _vectors_for_all_fingers(collision_forces)
    message.total_forces = _vectors_for_all_fingers(total_forces)
    message.translation_torques = np.asarray(
        translation_torques,
        dtype=np.float64,
    ).tolist()
    message.controller_torques = controller_torques.tolist()
    message.commanded_efforts = commanded_efforts.tolist()
    message.grasp_type = current_grasp_type(controller)
    message.pose_type = int(controller.pose_type)
    message.teaching_mode = bool(teaching_mode)
    message.controller_state = str(controller_state)
    message.controller_phase = str(controller_phase)
    return message
