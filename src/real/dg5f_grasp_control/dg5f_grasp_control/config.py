from dataclasses import dataclass


@dataclass(frozen=True)
class RuntimeConfig:
    joint_state_topic: str = "/dg5f_s_left/joint_states"
    effort_topic: str = "/dg5f_s_left/effort_controller/commands"
    command_topic: str = "/grasp_type"
    pose_topic: str = "/pose_type"
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
    # The current grasp_type=6 logic does not use them.
    envelop_finger_delay: float = 0.0
    envelop_thumb_joint_delay: float = 0.20
    envelop_stall_qdot: float = 0.03
    envelop_stall_hold_time: float = 0.25
    envelop_thumb_trigger_after: float = 0.40
    envelop_thumb_force_start_after: float = 1.20

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
