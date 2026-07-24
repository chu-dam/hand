# dg5f_grasp_control

DG5F-S real-hand grasp controller package.

This package keeps the existing DG5F-S effort controller launch file and moves the custom grasp algorithm from a single experimental script into a ROS 2 Python package.

## Build

Place this package under your workspace `src/` directory, then build:

```bash
cd /home/chu/DG-5F-S
colcon build --symlink-install
source install/setup.bash
```

## Recommended Execution

Terminal 1: start the existing effort controller.

```bash
ros2 launch dg5f_s_driver dg5f_s_left_effort_controller.launch.py
```

Terminal 2: start the grasp controller.

```bash
ros2 launch dg5f_grasp_control grasp_real.launch.py
```

You can also run the node directly:

```bash
ros2 run dg5f_grasp_control grasp_real
```

## One-Command Execution

If the existing `dg5f_s_driver` package is available in the same workspace, this launch starts both the effort controller and the grasp node:

```bash
ros2 launch dg5f_grasp_control grasp_with_effort.launch.py
```

## General grasp force policy (`grasp_type=1~5`)

The ordinary grasp modes always use the geometric centroid as the force
centroid: `Cv = Cg`. Following the corrected interpretation confirmed with the
professor, nominal force magnitude is proportional to fingertip distance from
`Cg`. Finger ID `1` is the thumb, and its magnitude is the user command
`alpha1`.

```text
d_i         = ||Cg - P_i||
alpha_thumb = alpha1
alpha_i     = alpha1 * d_i / d_thumb
fhat_i      = (Cg - P_i) / d_i
```

For a three-finger grasp, `Cg` is the arithmetic mean of the three contacts,
so this distribution directly satisfies `sum(alpha_i * fhat_i) = 0`. For four
or five fingers, the polygon signed-area centroid is generally different from
the vertex mean. The controller therefore keeps the thumb at `alpha1` and
uses a non-negative 3-D force-balance solver to find the solution closest to
the nominal distance-proportional distribution.

If a bounded non-negative solution is unavailable, the controller does not
fall back to the unbalanced legacy policy. It ramps the last valid regular
Cartesian force to zero over `force_balance_error_ramp_sec` while remapping it
through the current Jacobians, latches
`controller_phase=force_balance_error`, and rejects relative-rotation targets
until a grasp type is selected again.

`grasp_type=7` intentionally remains on its legacy policy: thumb-biased `Cv`,
inverse-distance coefficients, and the last active finger as the pivot.
`thumb_centroid_bias` applies only to this legacy type-7 path.

Every inactive finger in regular grasp types 1-5 holds all four joints at the
currently selected pre-grasp pose. Because a newly selected finger is already
waiting there, it joins the Jacobian-transpose grasp immediately without a
separate PD preparation delay or force blend. A 2F-I/2F-M switch still holds
the three-finger bridge for 0.5 s to preserve grasp continuity; relative-
rotation targets are rejected during that bridge.

Regular grasp types also run predictive adjacent-finger link avoidance. Joint
feedback and the XML chain dimensions produce a simplified capsule model for
each moving phalanx. An inactive finger first avoids an adjacent active finger;
while that inactive finger is moving, it becomes an avoidance source for the
next inactive neighbor. This chained propagation handles cases such as active
middle -> avoiding ring -> avoiding pinky without spreading undisturbed
inactive fingers. Both measured-velocity prediction and the previewed avoidance
command are checked. Index, middle, and ring move joint 1; pinky moves joint 2
to avoid ring. The smaller of current and 0.18-second predicted surface
clearance activates avoidance below 9 mm and releases it above 10 mm. Capsules
use a conservative 9 mm radius. Finite-difference FK selects the sign that
increases clearance; only the selected avoidance-joint PD target moves,
with a 0.40 rad offset, 1.2 rad/s target rate, and a damped dedicated command
(`Kp=0.5`, `Kd=0.10`, limit=0.25 N.m). A direction change requires at least
0.1 mm difference between the +/-1 degree FK trials. The other three joints
remain at pre-grasp. The palm-side root segment is ignored because adjacent
roots are naturally close. Debug fields report minimum clearance, five joint
offsets, and five activation flags.

## Relative rotation command

`/dg5f_grasp_control/relative_rotation_deg_cmd` contains a signed angle in
degrees relative to the current fingertip contact constellation. For regular
`grasp_type=1~5`, command time captures the thumb pivot `Pt,0` and every
`Pi,0`. The thumb receives only its ordinary grasp force. For each non-thumb
finger a fixed target is formed as
`Pi,d=Pt+R(theta_ref)(Pi,0-Pt,0)`, where current `Pt` allows common translation
without changing the stored relative geometry. The additional force is
`Fr,i=[kr(Pi,d-Pi)+kd(Pdot_i,d-Pdot_i)]/max(rho_i,rho_min)`. `Pdot_i` comes
from `Ji*qdot`, while `Pdot_i,d` includes the smooth reference-ramp velocity.
The ordinary grasp force remains active, so `Fi=Fg,i+Fr,i` and
`tau_i=Ji.T Fi`. All target coordinates and `rho_i` remain based on the
command-time geometry. Positive commands follow the right-hand rule about
`link_base -X`; the default command limit is +/-45 degrees.

The phase is `rotating`, `rotation_reached`, `rotation_timeout`, or
`rotation_error`. Once every driven fingertip is within the configured
final-position tolerance, the phase becomes `rotation_reached`; Cartesian PD
remains active as a position hold only until the command timeout. At 2 seconds
the additional rotation force is removed even if the target was reached. The
estimated angle uses non-thumb contact vectors relative to the current thumb,
but it is not an object-pose measurement:
rolling or slipping contacts can make it differ from the physical object
angle. There is no timed `Cv -> Cg` transition; the obsolete
centroid-transition stage has been removed.

## Relative task-space translation

`/dg5f_grasp_control/relative_translation_cmd` uses
`geometry_msgs/msg/Vector3Stamped`. Send meters with `frame_id=world` (converted
through the latest hand-to-world rotation) or `frame_id=link_base`. A valid
command for `grasp_type=1~5` captures the current geometric centroid and stores

```text
C_target = C_start + delta_link_base
```

The maximum command norm defaults to `relative_translation_max_m=0.010`. At
command time the controller captures every active fingertip and sets

```text
P_target_i = P_start_i + delta_link_base
```

The Cartesian reference advances with a 0.7-second smoothstep instead of a
position step. X, Y, and Z all use the same centroid Jacobian and damped
least-squares (DLS) inverse:

```text
Jc = [w1 J1  w2 J2  ...  wn Jn]
Jc# = Jc.T (Jc Jc.T + lambda^2 I)^-1
delta_q = Jc# (C_reference - Cg)
tau_position = Kq delta_q - Dq Jc# Jc qdot
```

There is no X-only multiplier or direction-specific force boost in this path.
The existing grasp and relative fingertip-shape torque is treated as a
secondary task and projected into the centroid task null space:

```text
N = I - pinv(Jc) Jc
tau = tau_position + N.T tau_grasp+shape
```

The null-space transition uses the same smooth reference progress so pressing
Move does not create an instantaneous grasp-torque step. The shape force is
constructed with zero resultant:

```text
f_shape_i = g_i - w_i * sum(g)
sum(f_shape_i) = 0
```

It can therefore help preserve the captured contact geometry without adding a
direct object resultant. Joint correction and position-torque limits, DLS
damping, a 3-second timeout, fingertip/centroid settle checks, and normal grasp
torque clipping remain active. Start real-hand commissioning with a 1 mm
command and keep RELEASE ready.

The old direction-dependent `J.T` force normalization parameters remain in
`RuntimeConfig` only so older YAML/launch files still load; they do not affect
the DLS motion controller.

`GraspDebug` reports the start, target, delta, remaining error, centroid
velocity, virtual Cartesian diagnostic force, and per-finger virtual/shape
forces. These Cartesian values are not sensor measurements and do not include
the DLS position torque. It also reports `relative_translation_joint_error`,
`relative_translation_position_torques`, the retained
`relative_translation_nullspace_grasp_torques`, DLS minimum singular value and
condition number, and the exact clipped 20-joint delta in
`translation_torques`. The phase is one of `translating`,
`translation_reached`, `translation_timeout`, or `translation_error`.

## Tuning

Main tuning values are in:

```text
config/grasp_real.yaml
```

Common values to change:

```yaml
use_finger_count: 5
alpha1: 3.0
pose_kp: 0.4
pose_kd: 0.05
pose_pd_limit: 0.25
min_tip_distance: 0.018
collision_repel_gain: 100.0
collision_repel_limit: 0.8
rotation_force_balance_max_alpha_ratio: 10.0
force_balance_error_ramp_sec: 0.5
relative_translation_kp: 600.0
relative_translation_kd: 6.0
relative_translation_hold_kp: 120.0
relative_translation_hold_kd: 1.2
relative_translation_shape_kp: 120.0
relative_translation_shape_kd: 1.2
relative_translation_cross_axis_deadband_m: 0.0003
relative_translation_reference_ramp_sec: 0.7
relative_translation_force_limit: 8.5
relative_translation_per_finger_force_limit: 5.5
relative_translation_dls_damping: 0.005
relative_translation_nullspace_rcond: 0.00001
relative_translation_joint_kp: 1.20
relative_translation_joint_kd: 0.06
relative_translation_joint_correction_limit_rad: 0.30
relative_translation_position_torque_limit: 0.30
relative_translation_nullspace_grasp_gain: 1.0
relative_rotation_max_abs_deg: 45.0
relative_rotation_reference_ramp_sec: 0.5
relative_rotation_position_kp: 48.0
relative_rotation_position_kd: 0.80
relative_rotation_position_error_limit_m: 0.025
relative_rotation_position_tolerance_m: 0.002
relative_rotation_force_limit: 10.00
relative_rotation_radius_min: 0.015
relative_rotation_timeout_sec: 2.0  # always removes Fr after command start
```

## File Roles

```text
dg5f_grasp_control/grasp_real_node.py   ROS 2 node and state machine
dg5f_grasp_control/kinematics.py        FK, tip position, numerical Jacobian
dg5f_grasp_control/grasp_policy.py      alpha, centroid, collision avoidance, J.T force mapping
dg5f_grasp_control/friction.py          friction compensation function
dg5f_grasp_control/friction_params.py   measured friction coefficients
dg5f_grasp_control/poses.py             normal pose and pre-grasp pose
dg5f_grasp_control/mujoco_gravity.py    MuJoCo gravity compensation
dg5f_grasp_control/hand_model.py        joint names and finger index mapping
```

## Shared controller architecture

The grasp equations and state machine are implemented once in:

```text
dg5f_grasp_control/grasp_controller.py
```

Both adapters use this controller:

- `grasp_real_node.py`: ROS joint-state/effort I/O, gravity compensation, and friction compensation.
- `src/mujoco/grasp_sim.py`: MuJoCo state/actuator I/O and simulation gravity compensation.

The shared controller includes pose control, grasp types 1–7, finger switching,
inactive-finger targets, envelop grasp, polygon-centroid groped grasp, collision
repulsion, and grasp-type-7 rotation/transition logic. Therefore changes to
`grasp_policy.py`, `poses.py`, `hand_model.py`, or `grasp_controller.py` are
used by both real hardware and MuJoCo.

Run MuJoCo from the workspace source tree:

```bash
cd ~/hand
source install/setup.bash
python3 src/mujoco/grasp_sim.py
```

The simulator reads `config/grasp_real.yaml` and subscribes to the same
`/grasp_type`, `/pose_type`, alpha1, relative-rotation-degree, and
rotation-matrix topics as the real node. The relative command topic is
`/dg5f_grasp_control/relative_rotation_deg_cmd`; it runs the same thumb-pivot
Cartesian PD rotation and target hold as the real controller.
