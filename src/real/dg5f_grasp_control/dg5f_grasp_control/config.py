from dataclasses import dataclass


@dataclass(frozen=True)
class RuntimeConfig:
    hand_side: str = "left"
    start_teaching_mode: bool = False
    joint_state_topic: str = "/dg5f_s_left/joint_states"
    tactile_topic: str = ""
    effort_topic: str = "/dg5f_s_left/effort_controller/commands"
    command_topic: str = "/grasp_type"
    pose_topic: str = "/pose_type"
    alpha1_topic: str = "/dg5f_grasp_control/alpha1_cmd"
    relative_translation_topic: str = "/dg5f_grasp_control/relative_translation_cmd"
    relative_rotation_deg_topic: str = "/dg5f_grasp_control/relative_rotation_deg_cmd"
    continuous_rotation_topic: str = "/dg5f_grasp_control/continuous_rotation_cmd"
    rotation_matrix_topic: str = "/dg5f_grasp_control/rotation_matrix_cmd"
    teaching_mode_topic: str = "/dg5f_grasp_control/teaching_mode"
    debug_topic: str = "/dg5f_grasp_control/debug"
    debug_frame_id: str = "link_base"
    debug_publish_hz: float = 20.0
    dt: float = 0.005
    hand_limit: float = 7.5
    log_dt: float = 0.25

    use_finger_count: int = 0

    pose_kp: float = 0.285
    pose_kd: float = 0.05
    pose_pd_limit: float = 0.25
    pre_rotation_pose_kp: float = 0.60
    pre_rotation_pose_kd: float = 0.085
    pre_rotation_pose_pd_limit: float = 0.50
    pre_rotation_pinky_j1_kp: float = 1.00
    pre_rotation_pinky_j1_kd: float = 0.09
    pre_rotation_pinky_j1_tau_limit: float = 0.60
    blind_grasp_pre_rotation_pose_kp: float = 1.0
    blind_grasp_pre_rotation_pose_kd: float = 5.0

    continuous_rotation_release_deg: float = 15.0
    continuous_rotation_index_ring_release_deg: float = 20.0
    continuous_rotation_ring_j2_release_deg: float = 30.0
    continuous_rotation_thumb_j2_release_deg: float = 20.0
    continuous_rotation_release_sec: float = 0.18
    continuous_rotation_move_sec: float = 0.28
    blind_rotation_grasp_settle_sec: float = 0.50
    blind_pinky_regrasp_sec: float = 0.20
    blind_finger_release_deg: float = 6.0
    blind_middle_release_deg: float = 3.0
    blind_pinky_release_deg: float = 10.0
    blind_pinky_regrasp_j1_target_rad: float = 0.8323
    blind_pinky_regrasp_j2_target_rad: float = 1.2723
    blind_pinky_regrasp_j3_target_rad: float = 0.7594
    blind_pinky_regrasp_j4_target_rad: float = 0.5027
    blind_finger_j1_release_deg: float = 19.0
    blind_thumb_j1_target_rad: float = 0.3241
    blind_thumb_j2_target_rad: float = -1.4177
    blind_thumb_j3_target_rad: float = -0.1822
    blind_thumb_j4_target_rad: float = 1.0472
    blind_reverse_thumb_j1_target_rad: float = 0.3487
    blind_reverse_thumb_j2_target_rad: float = -1.2823
    blind_reverse_thumb_j3_target_rad: float = -0.0682
    blind_reverse_thumb_j4_target_rad: float = 0.8501
    blind_reverse_pinky_release_deg: float = 10.0
    blind_sphere_lift_world_z_threshold_m: float = 0.140
    blind_thumb_lift_j1_target_rad: float = 0.2498
    blind_thumb_lift_j2_target_rad: float = -1.4917
    blind_thumb_lift_j3_target_rad: float = -0.4568
    blind_thumb_lift_j4_target_rad: float = 1.3090
    blind_reverse_thumb_lift_j1_target_rad: float = 0.2719
    blind_reverse_thumb_lift_j2_target_rad: float = -1.3050
    blind_reverse_thumb_lift_j3_target_rad: float = -0.1580
    blind_reverse_thumb_lift_j4_target_rad: float = 1.2034
    blind_sphere_effective_radius_m: float = 0.0375
    blind_sphere_fit_max_error_m: float = 0.008

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
    relative_translation_max_m: float = 0.020
    # Per-fingertip Cartesian PD force added to the ordinary grasp force.
    relative_translation_kp: float = 600.0
    relative_translation_kd: float = 6.0
    relative_translation_reference_ramp_sec: float = 0.7
    # Hard limits for the Cartesian translation force calculation.
    relative_translation_force_limit: float = 8.50
    relative_translation_per_finger_force_limit: float = 5.50
    relative_translation_velocity_alpha: float = 0.20
    relative_translation_position_tolerance_m: float = 0.0005
    relative_translation_velocity_tolerance_mps: float = 0.003
    relative_translation_settle_sec: float = 0.20
    relative_translation_timeout_sec: float = 3.0
    jacobian_eps: float = 1e-6

    min_tip_distance: float = 0.018
    collision_repel_gain: float = 100.0
    collision_repel_limit: float = 0.8

    # Adjacent-joint following for fingers that are inactive in a regular
    # grasp. Index/middle/ring use joint 1; pinky uses joint 2.
    inactive_collision_avoidance_enable: bool = True
    inactive_collision_joint_match_tolerance_rad: float = 0.0349066
    inactive_collision_joint_release_margin_rad: float = 0.0349066
    inactive_collision_pd_kp: float = 0.5
    inactive_collision_pd_kd: float = 0.10
    inactive_collision_pd_limit: float = 0.25
    inactive_collision_joint_limit_margin_rad: float = 0.03

    # Closed-loop relative rotation for regular grasp types 1..5.  At command
    # time C0 and every Pi,0 are frozen.  The controller tracks
    # Pi,d=C0+R(theta_ref)(Pi,0-C0) with a normalized Cartesian position
    # force, adds the ordinary grasp force, and maps the result through each
    # J.T.  Explicit D gain defaults to zero because the real transmission and
    # joint friction already provide substantial physical damping.
    # No centroid/null-space controller is used in this rotation path.
    relative_rotation_reference_ramp_sec: float = 0.5
    relative_rotation_position_kp: float = 24.0
    relative_rotation_position_kd: float = 0.0
    relative_rotation_alpha1_reference: float = 3.0
    relative_rotation_alpha1_double_gain_scale: float = 1.67
    relative_rotation_negative_direction_gain_scale: float = 1.2
    relative_rotation_positive_direction_damping_scale: float = 1.0
    relative_rotation_position_error_limit_m: float = 0.025
    relative_rotation_position_tolerance_m: float = 0.002
    relative_rotation_force_limit: float = 10.00
    relative_rotation_radius_min: float = 0.015
    relative_rotation_velocity_alpha: float = 0.20
    relative_rotation_timeout_sec: float = 1.0


    rotation_palm_normal_x: float = -1.0
    rotation_palm_normal_y: float = 0.0
    rotation_palm_normal_z: float = 0.0

    # grasp_type=7: World -Z floor contact followed by thumb/index pinch hold.
    card_floor_force_n: float = 3.0
    card_floor_hold_force_n: float = 3.0
    card_pinch_force_n: float = 4.0
    card_index_tip_target_deg: float = 80.0
    card_index_tip_kp: float = 1.0
    card_index_tip_kd: float = 0.1
    card_index_tip_tau_limit: float = 0.50
    card_index_tip_return_deg: float = 0.0
    card_index_tip_tolerance_deg: float = 3.0
    card_index_tip_stable_sec: float = 0.10
    card_tip_stall_threshold_m: float = 0.0002
    card_floor_stall_sec: float = 0.20
    card_pinch_stall_sec: float = 0.10
    card_post_pinch_delay_sec: float = 2.0
    card_thumb_j1_hold_kp: float = 0.60
    card_thumb_j1_hold_kd: float = 0.10
    card_thumb_j1_hold_tau_limit: float = 0.40
    card_floor_timeout_sec: float = 2.0
    card_pinch_timeout_sec: float = 1.0
    card_joint_state_timeout_sec: float = 0.10

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
