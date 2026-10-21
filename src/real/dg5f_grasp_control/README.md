# dg5f_grasp_control

DG5F-S real-hand grasp controller package.

This package keeps the existing DG5F-S effort controller launch file and moves the custom grasp algorithm from a single experimental script into a ROS 2 Python package.

## Build

Place this package under your workspace `src/` directory, then build:

```bash
cd /home/chu/hand
colcon build --symlink-install
source install/setup.bash
```

## Recommended Execution

Left hand, terminal 1: start the effort controller.

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

For the right hand, use the combined launch below. It selects the right joint
state/effort topics, analytic FK/Jacobian, poses, measured friction parameters,
gain overlay, and `dg5fs_right_w_mount.urdf` gravity model.

## One-Command Execution

If the existing `dg5f_s_driver` package is available in the same workspace,
these launches start both the effort controller and grasp node:

```bash
# Left
ros2 launch dg5f_grasp_control grasp_with_effort.launch.py

# Right (friction scale 1.0)
ros2 launch dg5f_grasp_control grasp_with_effort_right.launch.py
```

Common settings live in `config/grasp_real_common.yaml`; hand-specific gains
live in `config/grasp_real_left_gains.yaml` and
`config/grasp_real_right_gains.yaml`.

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

Every inactive finger in regular grasp types 1-5 holds all four joints at the
currently selected pre-grasp pose. Because a newly selected finger is already
waiting there, it joins the Jacobian-transpose grasp immediately without a
separate PD preparation delay or force blend. A 2F-I/2F-M switch still holds
the three-finger bridge for 0.5 s to preserve grasp continuity; relative-
rotation targets are rejected during that bridge.

Regular grasp types use adjacent-joint following for inactive-finger collision
avoidance. The monitored chain is index J1 -> middle J1 -> ring J1 -> pinky J2.
When adjacent values approach or cross within 2 degrees, their current offset
is stored. The inactive finger then tracks the adjacent joint displacement and
velocity with dedicated PD control. A following inactive finger can become the
source for the next inactive finger, so avoidance propagates from middle to
ring to pinky in the same control cycle. Downstream targets use the propagated
command target instead of waiting for upstream joint feedback. Following ends
when the source moves at least 2 degrees away on the safe side; ordinary pose
PD completes the follower's return to pre-grasp. Joint limits keep a 0.03 rad
margin; defaults
are `Kp=0.5`, `Kd=0.10`, and `limit=0.25 N.m`. The remaining three joints stay
at pre-grasp, and ENV does not use this avoidance path.

## Paper rotation test

The Rotation panel's **Rotate** sends the entered signed angle to
`<relative_rotation_deg_topic>/paper` (`std_msgs/Float64`, degrees), preserving
the simultaneous rotation and 3D centroid-hold controller below.
Right-hand **Paper TEST** sends it to
`<relative_rotation_deg_topic>/paper_sequential`, testing this separate sequence:

1. Capture the command-time FK centroid's world X/Y coordinates.
2. Apply grasp plus angle-controlled rotation, with no center force. After the
   reference ramp (0.5 seconds), require three consecutive estimated angular-speed
   samples below 0.05 rad/s; a faster sample resets the count. Neither angular
   error nor center drift blocks this stage.
3. Enter `centering`: remove rotation force, retain the same grasp/finger set,
   and apply world-X/Y centroid PD. World-Z position and velocity components
   are excluded before mapping the correction back to the hand frame.
4. After XY error is within 2 mm and XY speed within 0.01 m/s for 0.08 seconds,
   report `rotation_reached`/COMPLETE and remove all additional forces, retaining grasp.

Paper TEST uses the same center gains and 3 N total force cap, but center
correction runs only after rotation. The rotation timeout is unchanged;
centering has its own `paper_rotation_center_timeout_sec=3.0` deadline. A
timeout/error removes additional forces and does not report success. The UI
shows ROTATING, CENTERING, and COMPLETE separately. World coordinates use the
published hand-to-world orientation and assume a fixed hand-base origin;
world Z is uncontrolled, not physically locked. The return can change the
physical object angle through rolling/slip; this experiment adds no angle hold
during centering.

The continuous regrasp sequence uses saved grasp
joint targets. The original thumb-pivot controller remains available through
the non-paper relative-rotation ROS topic.
Run an active regular grasp before using the test; it preserves that finger set.

The test uses the tangent direction from equations (15)-(16) at every active
fingertip, including the thumb, with an angle-PD magnitude:
`Fr,i = normalize(axis x (Pi-Cg)) * (Kr*angle_error-Kd*angular_velocity)`.
The inverse-radius factor in equation (17) is deliberately omitted, so equal
angle/velocity errors give equal force magnitudes regardless of contact radius.
The current FK contact centroid defines the tangents. Command-time contacts and
centroid define the signed relative-angle estimate, which removes common
translation. Ordinary grasp forces remain active and `J.T` maps the sum of
grasp, tangent, collision and centroid-hold forces to torques; Human mode
keeps its existing DIP coupling.

Centroid hold saves `C0` at each rotation command and adds
`Fcenter=Kc*(C0-Cg)-Dc*Cdot_g`, shared equally among active fingers.
Defaults are `paper_rotation_center_kp=225` N/m,
`paper_rotation_center_kd=4.5` N*s/m and
`paper_rotation_center_force_limit=3` N for the total correction force.
The right-hand configuration overrides center Kp to 450 N/m; Kd and the force cap are unchanged.
The centroid velocity differentiates the same polygon-centroid function,
so asymmetric 4F/5F rigid rotation does not look like center translation.
The correction is exposed separately as `center_hold_forces`.

`paper_rotation_kr=13.333333333333334` N/rad, `paper_rotation_kd=1.6` N*s/rad and
`paper_rotation_tolerance_deg=1.0` configure the test. The per-finger
rotation-force cap and rotation timeout still apply (right-hand YAML:
3 seconds). The angle reference ramps smoothly over
`paper_rotation_reference_ramp_sec=0.5` seconds. Angular velocity uses the
`relative_rotation_velocity_alpha=0.2` low-pass filter. Inside the final angle
tolerance the P term is disabled and only damping remains. Outside it the
P error is `sign(e)*max(abs(e)-tolerance,0)`, avoiding a force jump at the
boundary. Damping uses filtered velocity clipped between zero and the current
unfiltered velocity, preventing filter lag from assisting a velocity reversal.
Completion also requires center error within
`paper_rotation_center_tolerance_m=0.002` m and center speed at most 0.01 m/s.
After these conditions and angle/velocity tolerances hold for `blind_motion_stable_sec=0.08`
seconds, `rotation_reached` latches and additional rotation forces stay zero
until a new request. Centroid hold remains active after the angle is reached,
until timeout or cancellation.
This also prevents repeated force reversals near the angle goal.
These gains approximate the former inverse-radius controller's 0.5/0.06 gains
at a 37.5 mm radius (75 mm sphere diameter). The old FK contact radius can differ
from the physical sphere radius, so this is a nominal calibration only;
runtime force magnitude remains independent of radius.

Estimated angle is contact-geometry rotation, not measured object orientation.
Slip/rolling can break that correspondence. Tangent forces can have a nonzero
resultant on asymmetric contacts. Centroid feedback reduces the resulting
translation, but holding the FK contact centroid does not guarantee holding
the physical object's center when contacts roll or slip.

## Relative rotation command

`/dg5f_grasp_control/relative_rotation_deg_cmd` contains a signed angle in
degrees relative to the current fingertip contact constellation. For regular
`grasp_type=1~5`, command time captures the thumb pivot `Pt,0` and every
`Pi,0`. The thumb receives only its ordinary grasp force. For each non-thumb
finger a fixed target is formed as
`Pi,d=Pt+R(theta_ref)(Pi,0-Pt,0)`, where current `Pt` allows common translation
without changing the stored relative geometry. The additional force is
`Fr,i=[kr(Pi,d-Pi)+kd(Pdot_i,d-Pdot_i)]/max(rho_i,rho_min)`. The defaults are
`relative_rotation_position_kp=24.0` and `relative_rotation_position_kd=0.0`;
set the latter above zero to use `Pdot_i` from `Ji*qdot` and `Pdot_i,d` from
the smooth reference-ramp velocity.
The ordinary grasp force remains active, so `Fi=Fg,i+Fr,i` and
`tau_i=Ji.T Fi`. All target coordinates and `rho_i` remain based on the
command-time geometry. Positive commands follow the right-hand rule about
`link_base -X`; the default command limit is +/-10 degrees.

The phase is `rotating`, `rotation_reached`, `rotation_timeout`, or
`rotation_error`. Once every driven fingertip is within the configured
final-position tolerance, the phase becomes `rotation_reached`; the Cartesian
position term remains active as a hold only until the command timeout. At 1 second
the additional rotation force is removed even if the target was reached. The
estimated angle uses non-thumb contact vectors relative to the current thumb,
but it is not an object-pose measurement:
rolling or slipping contacts can make it differ from the physical object
angle. There is no timed `Cv -> Cg` transition; the obsolete
centroid-transition stage has been removed.

For the right hand, `/dg5f_grasp_control/right/continuous_rotation_cmd=true`
starts the same five-finger blind regrasp sequence from either `pose_type=5`
or `pose_type=6`. The selected pose keeps its own name and pre-grasp joint
targets; the grasp and rotation state machine is shared.

The command starts a five-finger blind grasp and captures its initial joint
angles and contact-area scale when settling finishes. Middle releases and
regrasps, then index and ring release and regrasp together. During those
regrasps each returning finger holds its measured entry J1 with PD and applies
grasp torque to its other joints. Thumb release leaves index/pinky grasping and
middle/ring holding their entry poses. Thumb J3 release uses
`initial_J3 + 1.1*area_scale*(release_J3-initial_J3)` in both directions;
thumb J1/J2/J4 retain their existing release targets.
When index/middle/ring J1 headroom limits the common release motion, only
thumb J1/J2 targets are interpolated from the initial 5F thumb angles by
`common_J1_delta / area_requested_J1_delta`. Thumb J3 retains its area-scaled,
1.1-multiplied release displacement, and J4 retains its existing release target,
preserving clearance independently of the common sideways rotation limit.
With no common headroom, only thumb J1/J2 stay at their initial grasp angles.
Thumb regrasp uses thumb/index/pinky
grasp for 0.1 seconds, then all five fingers grasp for 0.08 seconds.

Index/middle/ring release use a shared J1 displacement:
`min(area_requested_delta, index_headroom, middle_headroom, ring_headroom)`.
All three J1 reference angles are captured together after initial 5F settling,
again after each rotation/regrasp cycle, and on a direction switch, keeping
the separately executed releases consistent. Their J1 targets use these
references, while J2-J4 still use the current entry angles and release offsets.
The final J1 targets remain clamped to each hand's URDF joint range.
Right-hand J1 ranges are
index [-48,15], middle [-33,42], and ring [-12,48] degrees. The existing
area-scale cap also bounds the nominal 19-degree offset to 23.75 degrees at
the default maximum scale of 1.25. These bounds constrain commanded targets,
not the physical joint angle under external loading.

Pinky release captures its first J1-J4 target for each direction and reuses it
on later cycles, instead of deriving a new target from potentially drifting
measured joints. A new sequence clears these references. The first target uses
the measured entry angles, with the usual area-scaled J1 release offset.

After pinky release, middle also releases. Thumb J1-J4 and index/ring J1-J2 track
the joint angles captured at the initial 5F grasp, adapting the target to the
grasped sphere size without a known diameter. Thumb J1/J2 use pose PD only;
thumb J3/J4 add grasp torque. Index/ring add grasp torque to J1/J2 pose PD, with
their other joints grasp-controlled. Middle and pinky retain release pose PD.
Human-mode non-thumb DIP coupling keeps its normal priority.

The joint-return phase ends after its controlled joints stay within 3 degrees
and below 0.05 rad/s for 0.08 seconds, or the existing
`continuous_rotation_move_sec=0.28` timeout expires. It then enters
`blind_pose_alignment`: thumb/index/middle/ring J1-J4 track the initial saved
grasp angles, while pinky remains released. Thumb/index/ring retain grasp
torques in alignment; middle has pose PD only. Each joint receives one pose
PD term, without stacking middle's inactive pre-grasp PD. Once fingers 1-4
are stopped for 0.08 seconds or another 0.28 seconds expires, 5F regrasp
starts before the next cycle. This sequence does not command a fixed 15-degree object
rotation and does not use Paper TEST's tangent forces or centroid hold.

## Relative task-space translation

`/dg5f_grasp_control/relative_translation_cmd` uses
`geometry_msgs/msg/Vector3Stamped`. Send meters with `frame_id=world` (converted
through the latest hand-to-world rotation) or `frame_id=link_base`. A valid
command for `grasp_type=1~5` captures the current geometric centroid and stores

```text
C_target = C_start + delta_link_base
```

The maximum command norm defaults to `relative_translation_max_m=0.020`. At
command time the controller captures every active fingertip and sets

```text
P_target_i = P_start_i + delta_link_base
```

The Cartesian reference advances with a 0.7-second smoothstep instead of a
position step. Every fingertip uses its own full 3-D target error and velocity:

```text
F_translation_i = (Kp (P_reference_i - P_i)
                  + Kd (Pdot_reference - Pdot_i)) / N_contacts
F_total_i = F_grasp_i + F_translation_i
tau_i = Ji.T F_total_i
```

The per-finger and total translation-force limits, a 3-second timeout, full
3-D fingertip settle checks, and normal grasp torque clipping remain active.
Start real-hand commissioning with a 1 mm command and keep RELEASE ready.

`GraspDebug` reports the start, target, delta, remaining error, centroid
velocity, commanded Cartesian force, per-finger translation forces, and the
exact 20-joint translation contribution in `translation_torques`. These are
controller commands, not force sensor measurements. The phase is one of `translating`,
`translation_reached`, `translation_timeout`, or `translation_error`.

## Floor-card pinch (`grasp_type=7`)

CARD starts only from card pre-grasp (`pose_type=4`). Thumb and index first receive 3 N in
World `-Z`; if both FK fingertip positions stay within 0.2 mm for 0.2 seconds,
the index additionally receives 4 N toward the thumb in the World XY plane.
Both fingers keep World `-Z` contact-height feedback, and a second 0.1-second
stall stores the current index J1–J3 angles and holds them while index J4 moves
toward 80 degrees. J1–J3 use the ordinary pre-grasp PD (`Kp=0.285`, `Kd=0.05`,
limit `0.25 N·m`), while J4 uses `Kp=6.0`, `Kd=1.4`, limit `0.50 N·m`.
The pinch forces remain active and this final phase has no timeout. Pose, grasp,
Teaching, RELEASE,
or stale JointState commands cancel CARD control. The command sequence is:

```bash
ros2 topic pub --once /pose_type std_msgs/msg/Int32 "{data: 4}"
ros2 topic pub --once /grasp_type std_msgs/msg/Int32 "{data: 7}"
```

## Tuning

Common limits/timing and hand-specific gains are in:

```text
config/grasp_real_common.yaml
config/grasp_real_left_gains.yaml
config/grasp_real_right_gains.yaml
```

Current Translation tuning differs by hand:

```yaml
# grasp_real_left_gains.yaml
relative_translation_kp: 600.0
relative_translation_kd: 6.0

# grasp_real_right_gains.yaml
relative_translation_kp: 1200.0
relative_translation_kd: 20.0
relative_translation_velocity_alpha: 0.50
```

## File Roles

```text
dg5f_grasp_control/grasp_real_node.py   ROS 2 node and state machine
dg5f_grasp_control/kinematics.py        left/right FK and analytic Jacobian selector
dg5f_grasp_control/kinematics_*.py      hand-specific FK and analytic Jacobian
dg5f_grasp_control/grasp_policy.py      alpha, centroid, collision avoidance, J.T force mapping
dg5f_grasp_control/friction.py          friction compensation function
dg5f_grasp_control/friction_params_*.py measured left/right friction coefficients
dg5f_grasp_control/poses.py             left/right normal and pre-grasp poses
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

The shared controller includes pose control, grasp types 1–6, finger switching,
inactive-finger targets, envelop grasp, polygon-centroid groped grasp, collision
repulsion, and relative Cartesian manipulation. Therefore changes to
`grasp_policy.py`, `poses.py`, `hand_model.py`, or `grasp_controller.py` are
used by both real hardware and MuJoCo.

Run MuJoCo from the workspace source tree:

```bash
cd ~/hand
source install/setup.bash
python3 src/mujoco/grasp_sim.py
```

The simulator reads `config/grasp_real_common.yaml` and subscribes to the same
`/grasp_type`, `/pose_type`, alpha1, relative-rotation-degree, and
rotation-matrix topics as the real node. The relative command topic is
`/dg5f_grasp_control/relative_rotation_deg_cmd`; it runs the same shared
relative-rotation controller.
