# All positions and Cartesian forces use header.frame_id.
# Arrays indexed by finger_ids always have matching lengths.
std_msgs/Header header

uint8[] finger_ids
geometry_msgs/Point[] fingertip_positions

geometry_msgs/Point geometric_centroid
geometry_msgs/Point virtual_centroid

# Relative rotation about the configured palm-normal axis. The current angle
# is estimated from the fingertip contact constellation, not an object sensor.
# The command-time thumb position is the pivot. Thumb rotation force is zero;
# non-thumb fingers use normalized Cartesian PD target tracking.
geometry_msgs/Point relative_rotation_start_centroid
geometry_msgs/Point relative_rotation_pivot
geometry_msgs/Vector3 relative_rotation_axis
float64 relative_rotation_target_rad
float64 relative_rotation_current_rad
float64 relative_rotation_error_rad
float64 relative_rotation_angular_velocity
float64 relative_rotation_command_moment
string relative_rotation_phase
string relative_rotation_control_mode
# Retained for message compatibility. The DLS/null-space rotation controller
# is no longer active, so the following hierarchy diagnostics are zero.
geometry_msgs/Vector3 relative_rotation_center_error
float64 relative_rotation_dls_sigma_min
float64 relative_rotation_dls_condition
float64[] relative_rotation_center_joint_error
float64[] relative_rotation_center_position_torques
float64[] relative_rotation_nullspace_torques

# Relative task-space centroid target diagnostics in header.frame_id.
geometry_msgs/Point relative_translation_start_centroid
geometry_msgs/Point relative_translation_target_centroid
geometry_msgs/Vector3 relative_translation_delta
geometry_msgs/Vector3 relative_translation_error
geometry_msgs/Vector3 relative_translation_centroid_velocity
geometry_msgs/Vector3 relative_translation_command_force
float64 relative_translation_torque_target
float64 relative_translation_force_scale
string relative_translation_phase
string relative_translation_control_mode
float64 relative_translation_dls_sigma_min
float64 relative_translation_dls_condition
float64[] relative_translation_joint_error
float64[] relative_translation_position_torques
float64[] relative_translation_nullspace_grasp_torques

# Inactive-finger predictive capsule avoidance. Clearance is the minimum
# capsule surface distance for monitored active/inactive adjacent pairs.
# Arrays use fixed finger order 1..5.
float64 inactive_collision_min_clearance_m
float64[] inactive_collision_avoidance_offsets_rad
bool[] inactive_collision_avoidance_active

float64[] alpha

geometry_msgs/Vector3[] grasp_forces
geometry_msgs/Vector3[] translation_forces
geometry_msgs/Vector3[] rotation_forces
geometry_msgs/Vector3[] center_hold_forces
geometry_msgs/Vector3[] collision_forces
# Cartesian policy force sum. During relative rotation this is Fg,i + Fr,i.
geometry_msgs/Vector3[] total_forces

# Controller torque excludes gravity/friction compensation. Commanded effort
# is the final actuator command after compensation and command limiting.
# Translation torque is the exact difference between the active hierarchical
# position/null-space command and the ordinary grasp torque after clipping.
float64[] translation_torques
float64[] controller_torques
float64[] commanded_efforts

int32 grasp_type
int32 pose_type
bool teaching_mode
string controller_state
string controller_phase
