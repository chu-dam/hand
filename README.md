# DG5F-S Hand Grasp Control Workspace

DG5F-S 5-finger hand를 ROS 2에서 실행하고, 토픽 명령으로 grasp mode, pre-grasp pose, grasp force 계수, hand rotation matrix를 제어하기 위한 workspace입니다.

현재 구조에서는 실제 DG5F-S 제어와 MuJoCo simulation이 동일한 `GraspController`를 사용합니다. 따라서 centroid 계산, 손가락 선택, force distribution, 손가락 전환, enveloping grasp, rotation/transition 제어를 수정하면 real과 MuJoCo에 동일하게 반영됩니다.

주요 기능은 다음과 같습니다.

- 2~5손가락 groped grasp
- Jacobian transpose 기반 fingertip force control
- 4·5접촉점의 polygon area centroid 계산
- grasp 중 손가락 추가·제거 및 안정적인 조합 전환
- 비사용 손가락 PD 자세 유지
- `grasp_type=6`: Sequential Torque-Based Enveloping Grasp
- `grasp_type=7`: 4손가락 파지 기반 rotation 및 순차 finger transition
- 실제 hand와 MuJoCo가 동일한 공통 제어 코어 사용

---

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
    │       │   ├── grasp_controller.py
    │       │   ├── grasp_real_node.py
    │       │   ├── grasp_policy.py
    │       │   ├── kinematics.py
    │       │   ├── mujoco_gravity.py
    │       │   ├── friction.py
    │       │   ├── friction_params.py
    │       │   ├── poses.py
    │       │   ├── hand_model.py
    │       │   ├── control_utils.py
    │       │   └── config.py
    │       ├── launch/
    │       │   ├── grasp_real.launch.py
    │       │   └── grasp_with_effort.launch.py
    │       ├── models/
    │       │   ├── dg5fs_left_w_mount.xml
    │       │   └── meshes/
    │       ├── package.xml
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

### 주요 파일 역할

| Path | Role |
| --- | --- |
| `src/real/dg5f_grasp_control/dg5f_grasp_control/grasp_controller.py` | real과 MuJoCo가 함께 사용하는 공통 grasp state machine 및 torque controller |
| `src/real/dg5f_grasp_control/dg5f_grasp_control/grasp_policy.py` | centroid, force direction, alpha distribution, rotation force, collision repel 계산 |
| `src/real/dg5f_grasp_control/dg5f_grasp_control/grasp_real_node.py` | 실제 hand의 ROS 2 JointState 수신, 보상 토크 계산, effort publish |
| `src/mujoco/grasp_sim.py` | MuJoCo model과 공통 `GraspController`를 연결하는 simulation adapter |
| `src/real/dg5f_grasp_control/config/grasp_real.yaml` | real과 MuJoCo가 공유하는 controller parameter |
| `src/vendor` | DG5F-S driver, description, hardware interface, TCP communication package |
| `rb5_payload_gc_rotation_pub.py` | RB5 hand rotation matrix topic publisher |

---

## Shared Controller Architecture

real과 MuJoCo는 각각 별도의 grasp 알고리즘을 구현하지 않습니다.

```text
                    ┌─────────────────────────┐
                    │    GraspController      │
                    │                         │
                    │  - grasp state machine  │
                    │  - finger switching     │
                    │  - inactive-finger PD   │
                    │  - envelop grasp        │
                    │  - grasp type 7         │
                    └───────────┬─────────────┘
                                │
                    ┌───────────┴───────────┐
                    │                       │
          ┌─────────▼─────────┐   ┌────────▼─────────┐
          │ grasp_real_node.py│   │  grasp_sim.py    │
          │                   │   │                  │
          │ ROS JointState    │   │ MuJoCo qpos/qvel│
          │ gravity/friction  │   │ MuJoCo physics  │
          │ effort publisher  │   │ actuator torque │
          └───────────────────┘   └──────────────────┘
```

다음 제어 로직은 real과 MuJoCo에서 동일합니다.

- grasp type 및 pose type 상태 전환
- 사용 손가락 선택
- centroid 및 virtual centroid 계산
- `alpha` force distribution
- Jacobian transpose torque
- 비사용 손가락 목표 자세
- 손가락 추가 전 pre-grasp 이동
- `1 ↔ 2` 전환 시 3손가락 경유
- `grasp_type=6` enveloping grasp
- `grasp_type=7` rotation 및 finger transition

단, 물리 환경은 서로 다릅니다.

| Item | Real hand | MuJoCo |
| --- | --- | --- |
| Joint state | 실제 `/joint_states` | MuJoCo `qpos`, `qvel` |
| Gravity compensation | `MujocoGravityCompensator` 결과를 effort에 추가 | MuJoCo `qfrc_bias` 사용 |
| Friction compensation | `calc_friction()` 적용 | 별도 friction compensation 미적용 |
| Contact | 실제 손과 물체 | MuJoCo collision/contact model |
| Command topics | 동일 | 동일 |

---

## Build

### 1. Workspace 확인

`src/vendor` 아래 패키지가 모두 존재해야 합니다.

```bash
cd ~/hand
find src -maxdepth 3 -name package.xml
```

특히 다음 경로가 없으면 전체 build가 실패합니다.

```text
src/vendor/dg_tcp_comm
src/vendor/dg_hardware
src/vendor/dg5f_s_driver
src/vendor/dg5f_s_description
```

> `src` 전체를 교체할 때는 기존 `vendor` 폴더를 삭제하지 마십시오. 제어 코드만 업데이트하는 경우 `src/real/dg5f_grasp_control`과 `src/mujoco`만 교체하는 것이 안전합니다.

### 2. Clean build

```bash
cd ~/hand
rm -rf build install log
colcon build --symlink-install
source install/setup.bash
```

제어 package만 다시 빌드하려면:

```bash
cd ~/hand
colcon build --symlink-install --packages-select dg5f_grasp_control
source install/setup.bash
```

설치된 공통 controller 경로 확인:

```bash
python3 - <<'PY'
import dg5f_grasp_control.grasp_controller as gc
import dg5f_grasp_control.grasp_real_node as rn

print("controller:", gc.__file__)
print("real node:", rn.__file__)
PY
```

---

## Run Real Hand

### Option 1. Driver와 grasp controller를 한 번에 실행

```bash
cd ~/hand
source install/setup.bash
ros2 launch dg5f_grasp_control grasp_with_effort.launch.py
```

### Option 2. Driver와 grasp controller를 따로 실행

Terminal 1: DG5F-S effort controller

```bash
cd ~/hand
source install/setup.bash
ros2 launch dg5f_s_driver dg5f_s_left_effort_controller.launch.py
```

Terminal 2: grasp controller

```bash
cd ~/hand
source install/setup.bash
ros2 launch dg5f_grasp_control grasp_real.launch.py
```

---

## Run MuJoCo Simulation

MuJoCo simulation도 real과 동일한 `GraspController`와 `grasp_real.yaml`을 사용합니다.

```bash
cd ~/hand
source install/setup.bash
python3 src/mujoco/grasp_sim.py
```

실행 후 real과 동일한 토픽으로 명령을 보낼 수 있습니다.

```bash
ros2 topic pub --once /grasp_type std_msgs/msg/Int32 "{data: 4}"
```

MuJoCo를 source tree에서 직접 실행하면 `src/real/dg5f_grasp_control`을 우선 import하므로, controller 코드를 수정한 뒤 package를 다시 설치하지 않아도 source code가 사용됩니다. 다만 real 실행 전에는 `colcon build`와 `source install/setup.bash`를 수행하는 것을 권장합니다.

> real hand와 MuJoCo를 동시에 실행하면 두 노드가 같은 command topic을 구독합니다. 한쪽만 제어하려면 별도의 ROS domain 또는 topic namespace를 사용하십시오.

---

## Topics

| Purpose | Topic | Type |
| --- | --- | --- |
| Grasp mode command | `/grasp_type` | `std_msgs/msg/Int32` |
| Pose command | `/pose_type` | `std_msgs/msg/Int32` |
| Grasp force coefficient | `/dg5f_grasp_control/alpha1_cmd` | `std_msgs/msg/Float64` |
| Hand rotation matrix | `/dg5f_grasp_control/rotation_matrix_cmd` | `std_msgs/msg/Float64MultiArray` |
| Real hand joint state | `/dg5f_s_left/joint_states` | `sensor_msgs/msg/JointState` |
| Real hand effort command | `/dg5f_s_left/effort_controller/commands` | `std_msgs/msg/Float64MultiArray` |

기존 `/dg5f_grasp_control/finger_count_cmd` 대신 `/grasp_type`을 사용합니다.

---

## 1. Grasp Type Command

사용할 손가락 조합 또는 특수 grasp mode를 지정합니다.

| Value | Meaning | Active fingers |
| ---: | --- | --- |
| `-1` | Normal pose | 없음 |
| `0` | 선택된 pre-grasp pose | 없음 |
| `1` | Two-finger grasp | 엄지 + 검지 |
| `2` | Two-finger grasp | 엄지 + 중지 |
| `3` | Three-finger groped grasp | 엄지 + 검지 + 중지 |
| `4` | Four-finger groped grasp | 엄지 + 검지 + 중지 + 약지 |
| `5` | Five-finger groped grasp | 엄지 + 검지 + 중지 + 약지 + 새끼 |
| `6` | Sequential Torque-Based Enveloping Grasp | 5손가락 특수 시퀀스 |
| `7` | Four-finger rotation 및 sequential transition | 엄지 + 검지 + 중지 + 약지, 새끼 PD hold |

예시:

```bash
ros2 topic pub --once /grasp_type std_msgs/msg/Int32 "{data: -1}"
ros2 topic pub --once /grasp_type std_msgs/msg/Int32 "{data: 0}"
ros2 topic pub --once /grasp_type std_msgs/msg/Int32 "{data: 3}"
ros2 topic pub --once /grasp_type std_msgs/msg/Int32 "{data: 5}"
ros2 topic pub --once /grasp_type std_msgs/msg/Int32 "{data: 6}"
ros2 topic pub --once /grasp_type std_msgs/msg/Int32 "{data: 7}"
```

### 일반 동작

- 처음 실행하면 normal pose 상태에서 시작합니다.
- `-1`을 보내면 normal pose로 이동합니다.
- `0`을 보내면 현재 선택된 pre-grasp pose로 이동합니다.
- `1~5`를 보내면 지정된 손가락 조합으로 groped grasp를 수행합니다.
- 파지 중 다른 grasp type이 들어오면 전체를 먼저 풀지 않고 손가락 조합을 전환합니다.
- `1 → 2`, `2 → 1` 전환은 안정성을 위해 내부적으로 grasp type 3을 거칩니다.
- 새로 추가되는 손가락은 먼저 현재 pre-grasp pose로 이동한 뒤 grasp torque로 부드럽게 전환됩니다.

### 비사용 손가락 자세

일반 groped grasp에서 사용하지 않는 손가락은 PD로 다음 자세를 유지합니다.

- 엄지, 검지, 중지, 약지: 첫 번째 관절만 `HAND_PRE_GRASP_POSE` 값 사용
- 새끼: 첫 번째와 두 번째 관절을 `HAND_PRE_GRASP_POSE` 값 사용
- 나머지 관절: `0.0 rad`

비사용 새끼손가락의 목표값은 다음과 같습니다.

```text
[-0.1471, -0.3410, 0.0, 0.0] rad
```

---

## 2. Pose Type Command

normal pose와 두 종류의 pre-grasp pose를 선택합니다.

| Value | Meaning |
| ---: | --- |
| `1` | `HAND_NORMAL_POSE` |
| `2` | 기본 `HAND_PRE_GRASP_POSE` |
| `3` | `HAND_COMPACT_PRE_GRASP_POSE` |

예시:

```bash
ros2 topic pub --once /pose_type std_msgs/msg/Int32 "{data: 1}"
ros2 topic pub --once /pose_type std_msgs/msg/Int32 "{data: 2}"
ros2 topic pub --once /pose_type std_msgs/msg/Int32 "{data: 3}"
```

`pose_type=2` 또는 `3`을 보내면 즉시 해당 pre-grasp 자세로 이동하며, 이후 `grasp_type=0`을 보내도 마지막으로 선택된 pre-grasp 자세를 사용합니다. 새로 추가되는 손가락 역시 선택된 pre-grasp pose를 사용합니다.

---

## 3. Groped Grasp Control

### Geometric centroid

접촉점 개수에 따라 centroid를 다음과 같이 계산합니다.

- 2접촉점: 두 점의 중점
- 3접촉점: 삼각형 꼭짓점 평균
- 4·5접촉점: 3차원 fingertip 위치를 best-fit plane으로 투영한 뒤 polygon signed-area centroid 계산

4·5접촉점에서 단순 산술평균 fallback은 사용하지 않습니다. polygon 면적이 정확히 0이면 계산 오류를 발생시킵니다.

### Thumb centroid bias

- 2손가락 grasp: 편향 없음, `Cv = Cg`
- 3·4·5손가락 grasp: 엄지 방향 virtual centroid 편향 적용

```text
Cv = Cg + thumb_centroid_bias × (P_thumb - Cg)
```

기본값:

```yaml
thumb_centroid_bias: 0.5
```

### Force distribution

`alpha1`은 선택된 첫 번째 손가락에 적용됩니다. 현재 finger selection 순서에서는 첫 번째 손가락이 엄지이므로, 일반적으로 엄지의 force magnitude 기준값입니다.

나머지 손가락은 centroid와의 거리 관계 및 force equilibrium으로 계산합니다. 약지와 새끼에 별도의 force scale은 적용하지 않습니다.

### Fingertip collision repel

현재 collision repel은 다음 fingertip pair에 적용됩니다.

```text
중지 ↔ 약지
약지 ↔ 새끼
```

이는 STL 전체 링크 충돌 제어가 아니라 fingertip 사이 거리가 `min_tip_distance`보다 작아지는 것을 줄이기 위한 추가 force입니다.

```yaml
min_tip_distance: 0.018
collision_repel_gain: 100.0
collision_repel_limit: 0.8
```

---

## 4. Sequential Torque-Based Enveloping Grasp

`grasp_type=6`은 검지, 중지, 약지, 새끼와 엄지를 시간 순서대로 닫아 물체를 감싸 쥐는 mode입니다.

```bash
ros2 topic pub --once /grasp_type std_msgs/msg/Int32 "{data: 6}"
```

`grasp_type=6`이 시작될 때의 전체 관절 위치를 hold pose로 저장합니다.

- 검지·중지·약지·새끼의 첫 번째 관절: 시작 위치를 PD로 유지
- 엄지의 첫 번째·두 번째 관절: 시작 위치를 PD로 유지
- torque가 활성화되지 않은 나머지 관절도 시작 위치를 PD로 유지

토크 시작 순서는 다음과 같습니다.

```text
t = 0
→ 검지/중지/약지/새끼의 2번째 관절 torque 시작

t = envelop_joint_delay
→ 검지/중지/약지/새끼의 3번째 관절 torque 시작

t = envelop_joint_delay × 2
→ 검지/중지/약지/새끼의 4번째 관절 torque 시작
→ 엄지 3번째 관절 torque 시작

t = envelop_joint_delay × 3
→ 엄지 4번째 관절 torque 시작
```

Active joint torque는 다음과 같이 계산됩니다.

```text
torque = alpha1 × envelop_tau_scale
```

기본 설정:

```yaml
envelop_tau_scale: 0.10
envelop_joint_delay: 0.20
envelop_non_thumb_tau_sign: 1.0
envelop_thumb_tau_sign: -1.0
```

| Parameter | Meaning |
| --- | --- |
| `envelop_tau_scale` | `alpha1`에 곱하는 envelop joint torque coefficient |
| `envelop_joint_delay` | joint stage 사이 시간 간격 |
| `envelop_non_thumb_tau_sign` | 검지·중지·약지·새끼 torque 방향 |
| `envelop_thumb_tau_sign` | 엄지 torque 방향 |

예를 들어:

```text
alpha1 = 3.0
envelop_tau_scale = 0.10

torque = 3.0 × 0.10 = 0.30
```

토크가 너무 강하면 `envelop_tau_scale`을 낮추십시오.

---

## 5. Grasp Type 7: Rotation and Finger Transition

`grasp_type=7`은 엄지·검지·중지·약지로 물체를 잡고, 회전 보조력과 순차적인 finger relocation을 수행하는 mode입니다.

```bash
ros2 topic pub --once /grasp_type std_msgs/msg/Int32 "{data: 7}"
```

### Initial grasp

- Active grasp fingers: 엄지, 검지, 중지, 약지
- 새끼손가락: grasp force에 참여하지 않고 PD hold
- 새끼손가락 1·2번 관절 목표:

```text
[0.0, -π/4] rad
```

새끼 3·4번 관절은 `grasp_type=7` 명령을 받은 순간의 위치를 유지합니다.

### Rotation sequence

1. 4손가락 grasp가 안정될 때까지 기다립니다.
2. active joint velocity가 threshold 이하로 일정 시간 유지되면 rotation을 시작합니다.
3. geometric centroid를 rotation center reference로 저장합니다.
4. net force가 거의 0이면서 palm normal 방향 moment를 만드는 `pure_moment` force distribution을 적용합니다.
5. 회전 중 centroid 이동을 줄이기 위해 center-hold force를 추가할 수 있습니다.
6. 회전 종료 조건이 만족되면 finger transition으로 넘어갑니다.

> `rotation_theta_rad`는 실제 물체의 폐루프 목표 각도가 아닙니다. 회전 추가 force 또는 moment의 크기와 방향을 결정하는 command scale입니다.

회전 방향을 반대로 바꾸려면 `rotation_theta_rad`의 부호를 바꿉니다.

```yaml
rotation_theta_rad: 0.174533
```

### Finger transition sequence

회전 종료 후 다음 순서로 손가락을 분리하고 다시 centroid 방향으로 접촉시킵니다.

```text
검지 → 중지 → 엄지 → 약지
```

기본 transition 목표:

| Finger | Detach/relocation target |
| --- | --- |
| 검지 | 첫 번째 관절 `45°` |
| 중지 | 첫 번째 관절 `30°` |
| 엄지 | `[0°, 140°, 0°, -0.234 rad]` |
| 약지 | 첫 번째 관절 `9°` |

각 손가락은 active grasp set에서 먼저 제거된 뒤 PD로 relocation target까지 이동하고, Jacobian transpose torque로 현재 centroid 방향에 재접촉합니다.

`grasp_type7_repeat_transition_cycle=true`이면 transition cycle이 끝난 뒤 다시 rotation stabilization 단계로 돌아가 반복합니다. 다른 grasp 또는 pose 명령을 보내면 cycle이 종료됩니다.

주요 설정:

```yaml
grasp_type7_rotation_mode: pure_moment
grasp_type7_rotation_zero_net_force: true
grasp_type7_rotation_use_radius_compensation: false
rotation_enable_for_grasp_type7: true
rotation_theta_rad: 0.174533
rotation_gain: 0.25
rotation_force_limit: 0.75

grasp_type7_center_hold_enable: true
grasp_type7_center_hold_gain: 1.0
grasp_type7_center_hold_force_limit: 0.10

grasp_type7_repeat_transition_cycle: true
```

---

## 6. Grip Force Command

일반 grasp와 특수 grasp mode의 force 또는 torque 크기를 조절합니다.

| Topic | Type |
| --- | --- |
| `/dg5f_grasp_control/alpha1_cmd` | `std_msgs/msg/Float64` |

예시:

```bash
ros2 topic pub --once /dg5f_grasp_control/alpha1_cmd std_msgs/msg/Float64 "{data: 4.0}"
```

동작 영향:

- `grasp_type=1~5`: groped grasp force distribution의 기준 magnitude
- `grasp_type=6`: `alpha1 × envelop_tau_scale`로 active joint torque 결정
- `grasp_type=7`: 기본 4손가락 grasp force에 반영

값이 클수록 grasp force가 커지고, 작을수록 작아집니다.

---

## 7. Rotation Matrix Command

로봇팔 끝에 hand가 장착된 경우 hand frame 기준 중력 방향을 계산하기 위한 rotation matrix를 전달합니다.

| Topic | Type |
| --- | --- |
| `/dg5f_grasp_control/rotation_matrix_cmd` | `std_msgs/msg/Float64MultiArray` |

보내는 값은 row-major 순서의 `3 × 3` rotation matrix입니다.

```text
R_hand_to_world =
[ r00 r01 r02
  r10 r11 r12
  r20 r21 r22 ]
```

Identity matrix 예시:

```bash
ros2 topic pub --once \
  /dg5f_grasp_control/rotation_matrix_cmd \
  std_msgs/msg/Float64MultiArray \
  "{data: [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]}"
```

real controller에서는 다음 식으로 hand frame의 중력 벡터를 계산합니다.

```text
g_hand = R_hand_to_worldᵀ × [0, 0, -9.81]
```

rotation matrix topic을 보내지 않으면 model 내부 기본 중력 방향을 사용합니다.

MuJoCo에서도 같은 topic을 받아 simulation model의 중력 방향과 gravity compensation 계산에 반영합니다.

---

## RB5 Rotation Matrix Publisher

RB5 제어와 hand 제어는 별도 process로 실행합니다.

Terminal 1: hand controller

```bash
cd ~/hand
source install/setup.bash
ros2 launch dg5f_grasp_control grasp_with_effort.launch.py
```

Terminal 2: RB5 rotation matrix publisher

```bash
cd ~/hand
source install/setup.bash
python3 rb5_payload_gc_rotation_pub.py
```

Publisher topic:

```text
/dg5f_grasp_control/rotation_matrix_cmd
```

`rb5_payload_gc_rotation_pub.py`는 RB5에 payload model을 적용하고 hand의 current rotation matrix를 publish합니다.

---

## Main Parameters

주요 parameter는 다음 파일에서 조절합니다.

```text
src/real/dg5f_grasp_control/config/grasp_real.yaml
```

| Parameter group | Main parameters |
| --- | --- |
| ROS topics | `joint_state_topic`, `effort_topic`, `command_topic`, `pose_topic` |
| Pose PD | `pose_kp`, `pose_kd`, `pose_pd_limit` |
| Friction | `fric_scale`, `fric_tanh_k`, `fric_limit` |
| Groped grasp | `alpha1`, `groped_tau_limit`, `thumb_centroid_bias` |
| Collision repel | `min_tip_distance`, `collision_repel_gain`, `collision_repel_limit` |
| Type 6 | `envelop_tau_scale`, `envelop_joint_delay`, torque signs |
| Type 7 rotation | `rotation_theta_rad`, `rotation_gain`, `rotation_force_limit` |
| Type 7 center hold | `grasp_type7_center_hold_*` |
| Type 7 transition | `grasp_type7_*_transition_*`, attach force 및 torque limit |

MuJoCo도 같은 YAML 파일을 읽으므로 parameter를 한 곳에서 관리할 수 있습니다.

---

## Troubleshooting

### `src/vendor/dg_tcp_comm` does not exist

```text
CMake Error: The source directory ".../src/vendor/dg_tcp_comm" does not exist.
```

`src` 전체를 교체하면서 vendor package가 삭제된 경우입니다. 기존 backup에서 `src/vendor`를 복원한 뒤 clean build하십시오.

```bash
cd ~/hand
rm -rf build install log
colcon build --symlink-install
source install/setup.bash
```

### 변경한 Python 코드가 실행되지 않음

```bash
python3 - <<'PY'
import dg5f_grasp_control.grasp_controller as gc
import dg5f_grasp_control.grasp_real_node as rn

print(gc.__file__)
print(rn.__file__)
PY
```

출력 경로가 현재 `~/hand/install/...` 또는 source tree를 가리키는지 확인합니다.

### Command topic 확인

```bash
ros2 topic list | grep -E "grasp_type|pose_type|alpha1|rotation_matrix"
```

기존 `/dg5f_grasp_control/finger_count_cmd`가 아니라 `/grasp_type`을 사용해야 합니다.