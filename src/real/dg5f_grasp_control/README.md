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
