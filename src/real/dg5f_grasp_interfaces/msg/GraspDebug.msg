# All positions and Cartesian forces use header.frame_id.
# Arrays indexed by finger_ids always have matching lengths.
std_msgs/Header header

uint8[] finger_ids
geometry_msgs/Point[] fingertip_positions

geometry_msgs/Point geometric_centroid
geometry_msgs/Point virtual_centroid

float64[] alpha

geometry_msgs/Vector3[] grasp_forces
geometry_msgs/Vector3[] rotation_forces
geometry_msgs/Vector3[] center_hold_forces
geometry_msgs/Vector3[] collision_forces
geometry_msgs/Vector3[] total_forces

# Controller torque excludes gravity/friction compensation. Commanded effort
# is the final actuator command after compensation and command limiting.
float64[] controller_torques
float64[] commanded_efforts

int32 grasp_type
int32 pose_type
bool teaching_mode
string controller_state
string controller_phase
