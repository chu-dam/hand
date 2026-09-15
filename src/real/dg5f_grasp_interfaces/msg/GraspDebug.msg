# All positions and Cartesian forces use header.frame_id.
# Arrays indexed by finger_ids always have matching lengths.
std_msgs/Header header

uint8[] finger_ids
uint8[] active_finger_ids
geometry_msgs/Point[] fingertip_positions

geometry_msgs/Point geometric_centroid
geometry_msgs/Point virtual_centroid

# Latest 75 mm sphere estimate from four or more tactile FK contact points.
bool blind_sphere_estimate_valid
geometry_msgs/Point blind_sphere_center
float64 blind_sphere_effective_radius_m
float64 blind_sphere_fit_rms_error_m

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

# Relative task-space centroid target diagnostics in header.frame_id.
geometry_msgs/Point relative_translation_start_centroid
geometry_msgs/Point relative_translation_target_centroid
geometry_msgs/Vector3 relative_translation_delta
geometry_msgs/Vector3 relative_translation_error
geometry_msgs/Vector3 relative_translation_centroid_velocity
geometry_msgs/Vector3 relative_translation_command_force
float64 relative_translation_force_scale
string relative_translation_phase
string relative_translation_control_mode
float64[] relative_translation_position_torques

# Inactive-finger adjacent-joint following diagnostics.
# Arrays use fixed finger order 1..5.
float64[] inactive_collision_avoidance_offsets_rad
bool[] inactive_collision_avoidance_active

float64[] alpha

geometry_msgs/Vector3[] grasp_forces
geometry_msgs/Vector3[] translation_forces
geometry_msgs/Vector3[] rotation_forces
geometry_msgs/Vector3[] collision_forces
# Cartesian policy force sum. During relative rotation this is Fg,i + Fr,i.
geometry_msgs/Vector3[] total_forces

# Controller torque excludes gravity/friction compensation. Commanded effort
# is the final actuator command after compensation and command limiting.
# Translation torque is the difference between the combined Cartesian
# grasp/translation command and the ordinary grasp torque after clipping.
float64[] translation_torques
float64[] controller_torques
float64[] commanded_efforts

int32 grasp_type
int32 pose_type
bool teaching_mode
string controller_state
string controller_phase
