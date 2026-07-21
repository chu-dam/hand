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

The zero-resultant statement applies after a regular grasp's finger-composition
blend has settled. A short transient residual can appear while a newly added
finger is blended in; relative-rotation targets are rejected during that
transition.

## Relative rotation command

`/dg5f_grasp_control/relative_rotation_deg_cmd` contains a signed angle in
degrees relative to the current object pose. Because ordinary grasp modes
already use `Cv = Cg` and a balanced force distribution, a valid command for
`grasp_type=1~5` immediately reports `controller_phase=rotation_ready` after
at least one successful force-balance control cycle; there is no timed
`Cv -> Cg` transition. The old `centroid_redistributing` phase and
`rotation_centroid_transition_sec` setting are obsolete.

The current implementation only stores the relative target. Tangential
rotation force is not implemented yet, so `rotation_ready` does not mean that
the object has rotated through the requested angle.

## Relative task-space translation

`/dg5f_grasp_control/relative_translation_cmd` uses
`geometry_msgs/msg/Vector3Stamped`. Send meters with `frame_id=world` (converted
through the latest hand-to-world rotation) or `frame_id=link_base`. A valid
command for `grasp_type=1~5` captures the current geometric centroid and stores

```text
C_target = C_start + delta_link_base
```

The maximum command norm defaults to `relative_translation_max_m=0.010`.
The regular grasp policy remains active and continues to supply the holding
force. At command time the controller captures every active fingertip and sets

```text
P_target_i = P_start_i + delta_link_base
```

The Cartesian reference advances with a 0.7-second smoothstep instead of a
position step. The task force is split into the commanded motion axis, a softer
orthogonal centroid hold, and a relative fingertip-shape term. The shape term
is projected so its sum is exactly zero:

```text
f_shape_i = g_i - w_i * sum(g)
sum(f_shape_i) = 0
```

It can therefore create the restoring moment needed to preserve the captured
contact geometry without shaking the object through an unintended resultant.
The orthogonal hold uses lower gains and a 0.3 mm position deadband; inside the
deadband its position-force resultant is zero, while velocity damping remains
active. Every contact force is mapped through the fingertip Jacobian and added
to the unchanged grasp-force torque.

Resultant/per-finger force limits, a 3-second timeout, fingertip and centroid
position/velocity settle checks, and torque clipping remain active. Start
real-hand commissioning with a 1 mm command.

Because the same Cartesian force can produce very different joint torque in
each direction through `J.T`, weak directions are adaptively boosted toward

```text
translation torque target = min(
    relative_translation_torque_gain_nm_per_m
        * hand_frame_direction_multiplier
        * |Kp * axis_error - Kd * axis_velocity| / Kp,
    relative_translation_torque_limit
)
```

The normalization only increases force components parallel to the commanded
axis when their incremental joint torque is below this target. It therefore
does not amplify the cross-axis resultant, and remains bounded by
`relative_translation_force_limit` and
`relative_translation_per_finger_force_limit`. The current physical hand uses
an X multiplier of `1.30` and Y/Z multipliers of `1.00`. This is evaluated in
`link_base` after a World-frame command is rotated into the hand, so it follows
the physical weak direction when the hand is later mounted on a moving arm.

`GraspDebug` reports the start, target, delta, remaining error, centroid
velocity, commanded translation resultant, per-finger translation forces, and
the exact 20-joint incremental `translation_torques` remaining after the
combined grasp command is clipped. It also reports the adaptive
`relative_translation_torque_target`, `relative_translation_force_scale`, and
one of `translating`, `translation_reached`, `translation_timeout`, or
`translation_error`.

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
relative_translation_torque_normalization_enable: true
relative_translation_torque_gain_nm_per_m: 24.0
relative_translation_torque_axis_multiplier_x: 1.3
relative_translation_torque_axis_multiplier_y: 1.0
relative_translation_torque_axis_multiplier_z: 1.0
relative_translation_torque_limit: 0.17
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
`/dg5f_grasp_control/relative_rotation_deg_cmd`; the current implementation
stores a current-pose-relative target and immediately reports
`rotation_ready`, but does not yet apply tangential rotation force.
