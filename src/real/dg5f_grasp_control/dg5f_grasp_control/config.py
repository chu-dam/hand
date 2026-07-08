from dataclasses import dataclass


@dataclass(frozen=True)
class RuntimeConfig:
    joint_state_topic: str = "/dg5f_s_left/joint_states"
    effort_topic: str = "/dg5f_s_left/effort_controller/commands"
    command_topic: str = "/dg5f_grasp_control/finger_count_cmd"
    alpha1_topic: str = "/dg5f_grasp_control/alpha1_cmd"
    rotation_matrix_topic: str = "/dg5f_grasp_control/rotation_matrix_cmd"

    dt: float = 0.005
    hand_limit: float = 7.5
    log_dt: float = 0.25

    use_finger_count: int = 0

    normal_pose_time: float = 2.0
    pre_grasp_pose_time: float = 2.0
    groped_grasp_time: float = 20.0

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
    jacobian_eps: float = 1e-6

    min_tip_distance: float = 0.018
    collision_repel_gain: float = 100.0
    collision_repel_limit: float = 0.8

    # finger_count_cmd=6: enveloping grasp mode
    # - Time-only joint-stage sequence:
    #   1) non-thumb 2nd joints
    #   2) non-thumb 3rd joints
    #   3) non-thumb 4th joints + thumb 3rd joint
    #   4) thumb 4th joint
    # - Previous alpha1*0.25 was too strong on hardware, so the default is reduced.
    envelop_tau_scale: float = 0.10
    envelop_joint_delay: float = 0.20
    envelop_non_thumb_tau_sign: float = 1.0
    envelop_thumb_tau_sign: float = -1.0

    # Legacy parameters kept for compatibility with older YAML files.
    # The current finger_count_cmd=6 logic does not use them.
    envelop_finger_delay: float = 0.0
    envelop_thumb_joint_delay: float = 0.20
    envelop_stall_qdot: float = 0.03
    envelop_stall_hold_time: float = 0.25
    envelop_thumb_trigger_after: float = 0.40
    envelop_thumb_force_start_after: float = 1.20