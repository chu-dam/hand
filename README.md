# DG5F-S Hand Grasp Control Workspace

DG5F-S 5-finger hand를 ROS2에서 실행하고, 토픽 명령으로 사용하는 손가락 개수, grip force 계수, hand rotation matrix를 제어하기 위한 workspace입니다.

추가로 `finger_count_cmd = 6` 명령을 통해 4손가락과 엄지를 시간 순서대로 닫는 **Sequential Torque-Based Enveloping Grasp** 모션을 실행할 수 있습니다.

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

* `src/real/dg5f_grasp_control`: 실제 DG5F-S hand 제어 코드
* `src/mujoco`: MuJoCo simulation용 코드
* `src/vendor`: 기존 DG5F-S driver/description/hardware 관련 package
* `rb5_payload_gc_rotation_pub.py`: RB5에서 hand rotation matrix topic을 보내기 위한 test code

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

사용할 손가락 개수 또는 grasp mode를 보내는 topic입니다.

| Topic                                  | Type                 |
| -------------------------------------- | -------------------- |
| `/dg5f_grasp_control/finger_count_cmd` | `std_msgs/msg/Int32` |

명령 값:

| Value | Meaning                                  |
| ----: | ---------------------------------------- |
|  `-1` | 평소 대기 자세                                 |
|   `0` | 잡기 직전 자세                                 |
|   `1` | 엄지 + 검지                                  |
|   `2` | 엄지 + 중지                                  |
|   `3` | 엄지 + 검지 + 중지                             |
|   `4` | 엄지 + 검지 + 중지 + 약지                        |
|   `5` | 엄지 + 검지 + 중지 + 약지 + 새끼                   |
|   `6` | Sequential Torque-Based Enveloping Grasp |

예시:

```bash
ros2 topic pub --once /dg5f_grasp_control/finger_count_cmd std_msgs/msg/Int32 "{data: -1}"
ros2 topic pub --once /dg5f_grasp_control/finger_count_cmd std_msgs/msg/Int32 "{data: 0}"
ros2 topic pub --once /dg5f_grasp_control/finger_count_cmd std_msgs/msg/Int32 "{data: 3}"
ros2 topic pub --once /dg5f_grasp_control/finger_count_cmd std_msgs/msg/Int32 "{data: 5}"
ros2 topic pub --once /dg5f_grasp_control/finger_count_cmd std_msgs/msg/Int32 "{data: 6}"
```

동작 방식:

* 처음 실행하면 평소 대기 자세로 시작합니다.
* `-1`을 보내면 평소 대기 자세로 이동합니다.
* `0`을 보내면 잡기 직전 자세로 이동합니다.
* `1~5`를 보내면 해당 손가락 조합으로 파지합니다.
* 이미 파지 중일 때 다른 값이 들어오면 전체를 풀지 않고 손가락 조합만 바꿉니다.
* 사용하지 않는 손가락은 joint value `0.0` 방향으로 펴집니다.
* `1 -> 2`, `2 -> 1` 전환은 안정성을 위해 내부적으로 `3`을 거쳐서 이동합니다.
* 새로 추가되는 손가락은 먼저 잡기 직전 자세로 이동한 뒤 Jacobian 기반 force control에 들어갑니다.
* `6`을 보내면 Sequential Torque-Based Enveloping Grasp 모션을 실행합니다.

### 2. Sequential Torque-Based Enveloping Grasp

`finger_count_cmd = 6` 명령을 보내면 4손가락과 엄지를 시간 순서대로 닫아 물체를 감싸쥐는 grip motion을 실행합니다.

```bash
ros2 topic pub --once /dg5f_grasp_control/finger_count_cmd std_msgs/msg/Int32 "{data: 6}"
```

`6`번 grip에서는 검지, 중지, 약지, 새끼의 1번째 관절은 현재 위치를 PD 제어로 유지합니다.
엄지의 1번째, 2번째 관절도 현재 위치를 PD 제어로 유지합니다.

토크가 들어가는 순서는 다음과 같습니다.

```text
t = 0
→ 검지/중지/약지/새끼의 2번째 관절 토크 시작

t = envelop_joint_delay
→ 검지/중지/약지/새끼의 3번째 관절 토크 시작

t = envelop_joint_delay × 2
→ 검지/중지/약지/새끼의 4번째 관절 토크 시작
→ 엄지 3번째 관절 토크 시작

t = envelop_joint_delay × 3
→ 엄지 4번째 관절 토크 시작
```

즉, 4손가락은 2번째 관절부터 4번째 관절까지 순차적으로 닫히고, 엄지는 한 턴 늦게 시작합니다.
4손가락의 4번째 관절이 접히기 시작할 때 엄지 3번째 관절도 같이 접히기 시작합니다.

관련 설정은 `src/real/dg5f_grasp_control/config/grasp_real.yaml`에서 조절합니다.

```yaml
envelop_tau_scale: 0.025
envelop_joint_delay: 0.20
envelop_non_thumb_tau_sign: 1.0
envelop_thumb_tau_sign: -1.0
```

파라미터 의미:

| Parameter                    | Meaning                        |
| ---------------------------- | ------------------------------ |
| `envelop_tau_scale`          | `6`번 grip에서 사용하는 토크 크기 계수      |
| `envelop_joint_delay`        | 2번째 → 3번째 → 4번째 관절로 넘어가는 시간 간격 |
| `envelop_non_thumb_tau_sign` | 검지/중지/약지/새끼 토크 방향              |
| `envelop_thumb_tau_sign`     | 엄지 토크 방향                       |

`6`번 grip에서 active joint에 들어가는 토크 크기는 다음과 같이 계산됩니다.

```text
torque = alpha1 × envelop_tau_scale
```

예를 들어 `alpha1 = 3.0`, `envelop_tau_scale = 0.025`이면:

```text
torque = 3.0 × 0.025 = 0.075
```

힘을 줄이고 싶으면 `envelop_tau_scale` 값을 낮추면 됩니다.

예시:

```yaml
envelop_tau_scale: 0.025
```

엄지가 반대로 움직이면 아래 값을 반대로 바꿉니다.

```yaml
envelop_thumb_tau_sign: 1.0
```

또는

```yaml
envelop_thumb_tau_sign: -1.0
```

### 3. Grip Force Command

grip force 크기를 조절하는 계수입니다.

| Topic                            | Type                   |
| -------------------------------- | ---------------------- |
| `/dg5f_grasp_control/alpha1_cmd` | `std_msgs/msg/Float64` |

예시:

```bash
ros2 topic pub --once /dg5f_grasp_control/alpha1_cmd std_msgs/msg/Float64 "{data: 4.0}"
```

값을 크게 하면 grasp force가 커지고, 작게 하면 grasp force가 작아집니다.

`1~5`번 일반 grasp에서는 Jacobian 기반 force control의 force 크기에 영향을 줍니다.
`6`번 Sequential Torque-Based Enveloping Grasp에서는 아래 식으로 active joint torque 크기에 영향을 줍니다.

```text
torque = alpha1 × envelop_tau_scale
```

### 4. Rotation Matrix Command

hand의 회전 행렬을 보내는 topic입니다.
로봇팔 끝에 hand가 장착되었을 때 hand frame 기준 gravity compensation을 하기 위해 사용합니다.

| Topic                                     | Type                             |
| ----------------------------------------- | -------------------------------- |
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

rotation matrix topic을 보내지 않으면 controller 내부의 기본 rotation matrix 값을 사용합니다.

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
