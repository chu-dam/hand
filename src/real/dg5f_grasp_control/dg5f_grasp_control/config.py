from dataclasses import dataclass


@dataclass(frozen=True)
class RuntimeConfig:
    joint_state_topic: str = "/dg5f_s_left/joint_states"
    effort_topic: str = "/dg5f_s_left/effort_controller/commands"
    command_topic: str = "/grasp_type"
    pose_topic: str = "/pose_type"
    alpha1_topic: str = "/dg5f_grasp_control/alpha1_cmd"
    relative_translation_topic: str = "/dg5f_grasp_control/relative_translation_cmd"
    relative_rotation_deg_topic: str = "/dg5f_grasp_control/relative_rotation_deg_cmd"
    rotation_matrix_topic: str = "/dg5f_grasp_control/rotation_matrix_cmd"
    teaching_mode_topic: str = "/dg5f_grasp_control/teaching_mode"
    debug_topic: str = "/dg5f_grasp_control/debug"
    debug_frame_id: str = "link_base"
    debug_publish_hz: float = 20.0

    dt: float = 0.005
    hand_limit: float = 7.5
    log_dt: float = 0.25

    use_finger_count: int = 0

    pose_kp: float = 0.4
    pose_kd: float = 0.05
    pose_pd_limit: float = 0.25

    fric_scale: float = 0.7
    fric_tanh_k: float = 20.0
    fric_limit: float = 0.5
    qdot_alpha: float = 0.5

    alpha1: float = 3.0
    groped_tau_limit: float = 3.0
    groped_force_direction_sign: float = 1.0
    thumb_centroid_bias: float = 0.5
    # Maximum alpha/alpha1 ratio accepted by the regular-grasp proportional
    # distribution and its four/five-contact force-balance correction.
    rotation_force_balance_max_alpha_ratio: float = 10.0
    # On a regular-grasp balance failure, fade the most recent valid bounded
    # Cartesian command to zero instead of switching to the legacy policy.
    force_balance_error_ramp_sec: float = 0.5
    relative_translation_max_m: float = 0.010
    # Cartesian gains retained for virtual-force diagnostics and the
    # zero-resultant fingertip-shape term. Centroid motion itself is controlled
    # by the DLS joint-position gains below.
    relative_translation_kp: float = 600.0
    relative_translation_kd: float = 6.0
    # Relative fingertip-shape stabilization remains active, but centroid
    # position control is applied only along the commanded translation axis.
    relative_translation_shape_kp: float = 120.0
    relative_translation_shape_kd: float = 1.2
    relative_translation_reference_ramp_sec: float = 0.7
    # Hard limits for the diagnostic/shape Cartesian force calculation.
    relative_translation_force_limit: float = 8.50
    relative_translation_per_finger_force_limit: float = 5.50
    relative_translation_velocity_alpha: float = 0.20
    relative_translation_position_tolerance_m: float = 0.0005
    relative_translation_velocity_tolerance_mps: float = 0.003
    relative_translation_settle_sec: float = 0.20
    relative_translation_timeout_sec: float = 3.0
    # Command-axis centroid-position controller. The damped least-squares
    # inverse maps only the scalar error along the requested direction to an
    # active-joint correction; orthogonal centroid motion stays unconstrained.
    relative_translation_dls_damping: float = 0.005
    relative_translation_nullspace_rcond: float = 1e-5
    relative_translation_joint_kp: float = 1.20
    relative_translation_joint_kd: float = 0.06
    relative_translation_joint_correction_limit_rad: float = 0.30
    relative_translation_position_torque_limit: float = 0.30
    relative_translation_nullspace_grasp_gain: float = 1.0
    jacobian_eps: float = 1e-6

    min_tip_distance: float = 0.018
    collision_repel_gain: float = 100.0
    collision_repel_limit: float = 0.8

    # Predictive link-capsule avoidance for fingers that are inactive in a
    # regular grasp. Index/middle/ring shift joint 1; pinky shifts joint 2.
    # The remaining joints continue to hold the selected pre-grasp pose.
    inactive_collision_avoidance_enable: bool = True
    inactive_collision_capsule_radius_m: float = 0.009
    inactive_collision_activation_clearance_m: float = 0.009
    inactive_collision_critical_clearance_m: float = 0.002
    inactive_collision_release_hysteresis_m: float = 0.001
    inactive_collision_max_joint1_offset_rad: float = 0.40
    inactive_collision_joint1_target_rate_radps: float = 1.5
    inactive_collision_prediction_sec: float = 0.25
    inactive_collision_pd_kp: float = 0.5
    inactive_collision_pd_kd: float = 0.10
    inactive_collision_pd_limit: float = 0.25
    inactive_collision_gradient_eps_rad: float = 0.0174533
    inactive_collision_direction_min_delta_m: float = 0.00010
    inactive_collision_joint_limit_margin_rad: float = 0.03
    # Segment 0 is the palm-side proximal segment whose roots are naturally
    # close together. Start at segment 1 to monitor the moving phalanges.
    inactive_collision_first_segment: int = 1

    # Closed-loop relative rotation for regular grasp types 1..5.  At command
    # time C0 and every Pi,0 are frozen.  The controller tracks
    # Pi,d=C0+R(theta_ref)(Pi,0-C0) with a normalized Cartesian position
    # force, adds the ordinary grasp force, and maps the result through each
    # J.T.  Explicit D gain defaults to zero because the real transmission and
    # joint friction already provide substantial physical damping.
    # No centroid/null-space controller is used in this rotation path.
    relative_rotation_max_abs_deg: float = 10.0
    relative_rotation_reference_ramp_sec: float = 0.5
    relative_rotation_position_kp: float = 24.0
    relative_rotation_position_kd: float = 0.0
    relative_rotation_position_error_limit_m: float = 0.025
    relative_rotation_position_tolerance_m: float = 0.002
    relative_rotation_force_limit: float = 10.00
    relative_rotation_radius_min: float = 0.015
    relative_rotation_velocity_alpha: float = 0.20
    relative_rotation_timeout_sec: float = 1.0

    # grasp_type=7 rotation assist based on groped-grasp tangential force.
    # theta is used as a force command scale, not as closed-loop object angle.
    rotation_enable_for_grasp_type7: bool = True
    rotation_theta_rad: float = 0.174533  # +10 deg equivalent. Change sign for opposite direction.
    rotation_gain: float = 0.25
    rotation_force_limit: float = 0.75
    rotation_radius_min: float = 0.035
    # For cleaner in-place rotation, the rotation force can be balanced so its
    # weighted net force becomes zero. Radius compensation can be disabled so
    # fingers close to the centroid do not dominate the motion.
    grasp_type7_rotation_mode: str = "pure_moment"
    grasp_type7_rotation_zero_net_force: bool = True
    grasp_type7_rotation_use_radius_compensation: bool = False
    grasp_type7_rotation_nominal_radius: float = 0.060
    rotation_palm_normal_x: float = -1.0
    rotation_palm_normal_y: float = 0.0
    rotation_palm_normal_z: float = 0.0

    # grasp_type=7 rotation-center hold. Grasp force keeps using the
    # thumb-biased virtual centroid, while rotation and center hold use the
    # unbiased geometric centroid captured at rotation_start.
    grasp_type7_center_hold_enable: bool = True
    grasp_type7_center_hold_gain: float = 1.0
    grasp_type7_center_hold_force_limit: float = 0.10
    grasp_type7_center_hold_project_to_rotation_plane: bool = True

    # grasp_type=7 rotation state detection based on active-finger joint velocity.
    # Rotation starts only after the 4 grasp fingers stay nearly stopped for the
    # hold time. After rotation starts, another near-zero velocity hold marks
    # rotation completion. After that, the first index transition step can run.
    grasp_type7_start_qdot_threshold: float = 0.08
    grasp_type7_start_hold_sec: float = 0.20
    grasp_type7_done_qdot_threshold: float = 0.08
    grasp_type7_done_hold_sec: float = 0.20
    grasp_type7_min_rotation_sec: float = 0.50
    grasp_type7_stop_rotation_when_done: bool = True
    # If true, after index->middle->thumb->ring transition finishes, return to
    # grasp_stabilizing and repeat rotation + transition indefinitely until a
    # new command/pose_type is received.
    grasp_type7_repeat_transition_cycle: bool = True


    # grasp_type=7 first transition step after rotation completion.
    # Current scope: index finger only. After rotation_done, remove index from
    # the active grasp set first (centroid/force redistribution), detach index
    # to pose_type=2 by PD, move index joint_2_1 to 45 deg by PD, then reattach
    # index toward the current centroid by Jacobian-transpose torque.
    grasp_type7_index_transition_enable: bool = True
    grasp_type7_index_pd_tolerance_rad: float = 0.0872665  # 5 deg
    grasp_type7_index_first_joint_target_rad: float = 0.785398  # 45 deg
    grasp_type7_index_attach_force: float = 1.0
    grasp_type7_index_attach_tau_limit: float = 0.8

    # grasp_type=7 second transition step after index reattachment.
    # Remove middle from the active grasp set first, move only middle joint_3_1
    # to 30 deg while holding middle joints 2~4 at pose_type=2, then reattach
    # the middle fingertip toward the current centroid by Jacobian-transpose torque.
    grasp_type7_middle_transition_enable: bool = True
    grasp_type7_middle_first_joint_target_rad: float = 0.523599  # 30 deg
    grasp_type7_middle_attach_force: float = 1.0
    grasp_type7_middle_attach_tau_limit: float = 0.8

    # grasp_type=7 third transition step after middle reattachment.
    # Remove thumb from the active grasp set first, so centroid/force redistributes
    # to the currently contacting index+middle+ring fingers. Then PD-control thumb
    # to [0, 140 deg, 0, pose_type=2 thumb joint_1_4] before reattaching it toward a thumb-biased centroid.
    grasp_type7_thumb_transition_enable: bool = True
    grasp_type7_thumb_joint1_target_rad: float = 0.0
    grasp_type7_thumb_joint2_target_rad: float = 2.443461  # 140 deg
    grasp_type7_thumb_joint3_target_rad: float = 0.0
    grasp_type7_thumb_joint4_target_rad: float = -0.2340
    grasp_type7_thumb_attach_force: float = 1.0
    grasp_type7_thumb_attach_tau_limit: float = 0.8

    # grasp_type=7 fourth transition step after thumb reattachment.
    # Ring is detached while thumb remains in the active grasp set, so the
    # ordinary thumb-biased centroid is used. Move only ring joint_4_1 to 9 deg
    # while holding ring joints 2~4 at pose_type=2, then reattach by J^T torque.
    grasp_type7_ring_transition_enable: bool = True
    grasp_type7_ring_first_joint_target_rad: float = 0.157080  # 9 deg
    grasp_type7_ring_attach_force: float = 1.0
    grasp_type7_ring_attach_tau_limit: float = 0.8

    # Shared attach completion detector for transition fingers.
    grasp_type7_transition_attach_qdot_threshold: float = 0.08
    grasp_type7_transition_attach_hold_sec: float = 0.20
    grasp_type7_transition_attach_min_sec: float = 0.30

    # grasp_type=6: enveloping grasp mode
    # - Time-only joint-stage sequence:
    #   1) index/middle/ring J2 + pinky J3
    #   2) index/middle/ring J3 + pinky J4 + thumb J3
    #   3) index/middle/ring J4 + thumb J4
    # - Previous alpha1*0.25 was too strong on hardware, so the default is reduced.
    envelop_tau_scale: float = 0.10
    envelop_joint_delay: float = 0.20
    envelop_non_thumb_tau_sign: float = 1.0
    envelop_thumb_tau_sign: float = -1.0


def load_runtime_config_yaml(path, node_name="grasp_real"):
    """Load RuntimeConfig values from a ROS 2 parameter YAML file.

    Unknown YAML keys are ignored so older/newer config files remain usable.
    """
    from pathlib import Path

    import yaml

    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as stream:
        document = yaml.safe_load(stream) or {}

    params = document.get(node_name, {}).get("ros__parameters", {})
    defaults = RuntimeConfig()
    values = {
        name: params.get(name, default)
        for name, default in defaults.__dict__.items()
    }
    return RuntimeConfig(**values)
