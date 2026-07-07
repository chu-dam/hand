# DG5F-S Hand Grasp Control Workspace

DG5F-S 5-finger hand를 ROS2에서 실행하고, 토픽 명령으로 사용하는 손가락 개수, grip force 계수, hand rotation matrix를 제어하기 위한 workspace입니다.

## Folder Structure

```text
hand/
├── README.md
├── rb5_850e_payload_1kg.xml
├── rb5_payload_gc_rotation_pub.py
└── src/
    ├── real/
    │   └── dg5f_grasp_control/
    │       ├── config/
    │       │   └── grasp_real.yaml
    │       ├── dg5f_grasp_control/
    │       │   ├── grasp_real_node.py
    │       │   ├── grasp_policy.py
    │       │   ├── kinematics.py
    │       │   ├── mujoco_gravity.py
    │       │   ├── friction.py
    │       │   ├── friction_params.py
    │       │   ├── poses.py
    │       │   ├── hand_model.py
    │       │   └── control_utils.py
    │       ├── launch/
    │       │   ├── grasp_real.launch.py
    │       │   └── grasp_with_effort.launch.py
    │       ├── models/
    │       │   ├── dg5fs_left_w_mount.xml
    │       │   └── meshes/
    │       └── setup.py
    ├── mujoco/
    │   ├── grasp_sim.py
    │   ├── dg5fs_left_w_mount.xml
    │   └── meshes/
    └── vendor/
        ├── dg5f_s_description/
        ├── dg5f_s_driver/
        ├── dg_hardware/
        └── dg_tcp_comm/
```

- `src/real/dg5f_grasp_control`: 실제 DG5F-S hand 제어 코드
- `src/mujoco`: MuJoCo simulation용 코드
- `src/vendor`: 기존 DG5F-S driver/description/hardware 관련 package
- `rb5_payload_gc_rotation_pub.py`: RB5에서 hand rotation matrix topic을 보내기 위한 test code

## Build

새 터미널에서 먼저 ROS2 환경을 source합니다.

```bash
source /opt/ros/humble/setup.bash
```

workspace build:

```bash
cd /home/chu/hand
colcon build --symlink-install
source install/setup.bash
```

이후 새 터미널을 열 때마다 아래 명령을 실행해야 합니다.

```bash
cd /home/chu/hand
source install/setup.bash
```

## Run

### Option 1. 한 번에 실행하기

hand driver와 grasp controller를 한 번에 실행합니다.

```bash
cd /home/chu/hand
source install/setup.bash
ros2 launch dg5f_grasp_control grasp_with_effort.launch.py
```

### Option 2. 두 개로 나누어서 실행하기

Terminal 1: DG5F-S effort controller 실행

```bash
cd /home/chu/hand
source install/setup.bash
ros2 launch dg5f_s_driver dg5f_s_left_effort_controller.launch.py
```

Terminal 2: grasp controller 실행

```bash
cd /home/chu/hand
source install/setup.bash
ros2 launch dg5f_grasp_control grasp_real.launch.py
```

## Topics

### 1. Finger Count Command

사용할 손가락 개수를 보내는 topic입니다.

| Topic | Type |
|---|---|
| `/dg5f_grasp_control/finger_count_cmd` | `std_msgs/msg/Int32` |

명령 값:

| Value | Meaning |
|---:|---|
| `0` | 잡기 직전 자세 |
| `1` | 엄지 + 검지 |
| `2` | 엄지 + 중지 |
| `3` | 엄지 + 검지 + 중지 |
| `4` | 엄지 + 검지 + 중지 + 약지 |
| `5` | 엄지 + 검지 + 중지 + 약지 + 새끼 |

예시:

```bash
ros2 topic pub --once /dg5f_grasp_control/finger_count_cmd std_msgs/msg/Int32 "{data: 0}"
ros2 topic pub --once /dg5f_grasp_control/finger_count_cmd std_msgs/msg/Int32 "{data: 3}"
ros2 topic pub --once /dg5f_grasp_control/finger_count_cmd std_msgs/msg/Int32 "{data: 5}"
```

동작 방식:

- 처음 실행하면 평소 자세로 시작합니다.
- `0`을 보내면 잡기 직전 자세로 이동합니다.
- `1~5`를 보내면 해당 손가락 조합으로 파지합니다.
- 이미 파지 중일 때 다른 값이 들어오면 전체를 풀지 않고 손가락 조합만 바꿉니다.
- 사용하지 않는 손가락은 joint value `0.0` 방향으로 펴집니다.
- `1 -> 2`, `2 -> 1` 전환은 안정성을 위해 내부적으로 `3`을 거쳐서 이동합니다.
- 새로 추가되는 손가락은 먼저 잡기 직전 자세로 이동한 뒤 Jacobian 기반 force control에 들어갑니다.

### 2. Grip Force Command

grip force 크기를 조절하는 계수입니다.

| Topic | Type |
|---|---|
| `/dg5f_grasp_control/alpha1_cmd` | `std_msgs/msg/Float64` |

예시:

```bash
ros2 topic pub --once /dg5f_grasp_control/alpha1_cmd std_msgs/msg/Float64 "{data: 4.0}"
```

값을 크게 하면 grasp force가 커지고, 작게 하면 grasp force가 작아집니다.

### 3. Rotation Matrix Command

hand의 회전 행렬을 보내는 topic입니다.  
로봇팔 끝에 hand가 장착되었을 때 hand frame 기준 gravity compensation을 하기 위해 사용합니다.

| Topic | Type |
|---|---|
| `/dg5f_grasp_control/rotation_matrix_cmd` | `std_msgs/msg/Float64MultiArray` |

보내는 값은 row-major 순서의 3x3 rotation matrix입니다.

```text
R_hand_to_world =
[ r00 r01 r02
  r10 r11 r12
  r20 r21 r22 ]
```

예시: identity rotation matrix

```bash
ros2 topic pub --once /dg5f_grasp_control/rotation_matrix_cmd std_msgs/msg/Float64MultiArray "{data: [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]}"
```

## RB5 Rotation Matrix Publisher

RB5는 hand 제어와 별도로 실행합니다.  
이 코드는 RB5에 1kg payload sphere를 단 상태에서 gravity compensation만 수행하고, hand rotation matrix topic을 publish합니다.

Terminal 1: hand 실행

```bash
cd /home/chu/hand
source install/setup.bash
ros2 launch dg5f_grasp_control grasp_with_effort.launch.py
```

Terminal 2: RB5 rotation publisher 실행

```bash
cd /home/chu/hand
source install/setup.bash
python3 rb5_payload_gc_rotation_pub.py
```

RB5 publisher는 아래 topic으로 rotation matrix를 보냅니다.

```text
/dg5f_grasp_control/rotation_matrix_cmd
```

## Useful Commands

현재 topic 확인:

```bash
ros2 topic list
```

finger count topic echo:

```bash
ros2 topic echo /dg5f_grasp_control/finger_count_cmd
```

effort command 확인:

```bash
ros2 topic echo /effort_controller/commands
```

package 확인:

```bash
ros2 pkg list | grep dg5f
```

launch file 확인:

```bash
ros2 launch dg5f_grasp_control grasp_real.launch.py
ros2 launch dg5f_grasp_control grasp_with_effort.launch.py
```

## Troubleshooting

### `ament_cmake`를 찾지 못하는 경우

ROS2 환경이 source되지 않은 상태일 수 있습니다.

```bash
source /opt/ros/humble/setup.bash
cd /home/chu/hand
colcon build --symlink-install
```

### package tab completion이 안 되는 경우

build 후 workspace setup을 source해야 합니다.

```bash
cd /home/chu/hand
source install/setup.bash
```

### MuJoCo XML에서 mesh file을 못 찾는 경우

`dg5f_grasp_control/setup.py`에 model mesh install 설정이 들어가 있어야 합니다.  
수정 후 다시 build합니다.

```bash
cd /home/chu/hand
colcon build --symlink-install --packages-select dg5f_grasp_control
source install/setup.bash
```

## Git

변경 사항 저장:

```bash
cd /home/chu/hand
git status
git add .
git commit -m "Update hand grasp control README"
git push
```
