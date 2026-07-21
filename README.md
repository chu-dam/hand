# DG5F-S Hand Grasp Control Workspace

DG5F-S 5-finger hand를 ROS 2에서 실행하고, 토픽 명령으로 grasp mode, pre-grasp pose, grasp force 계수, 상대 회전 목표, hand rotation matrix를 제어하기 위한 workspace입니다.

현재 구조에서는 실제 DG5F-S 제어와 MuJoCo simulation이 동일한 `GraspController`를 사용합니다. 따라서 centroid 계산, 손가락 선택, force distribution, 손가락 전환, enveloping grasp, rotation/transition 제어를 수정하면 real과 MuJoCo에 동일하게 반영됩니다.

주요 기능은 다음과 같습니다.

- 2~5손가락 groped grasp
- Jacobian transpose 기반 fingertip force control
- 4·5접촉점의 polygon area centroid 계산
- grasp 중 손가락 추가·제거 및 안정적인 조합 전환
- 비사용 손가락 PD 자세 유지
- `grasp_type=6`: Sequential Torque-Based Enveloping Grasp
- `grasp_type=7`: 4손가락 파지 기반 rotation 및 순차 finger transition
- `grasp_type=1~5`: `Cv = Cg` 기반 force distribution과 현재 자세 기준 상대 회전 명령
- 실제 hand와 MuJoCo가 동일한 공통 제어 코어 사용

---

## Folder Structure

```text
hand/
├── README.md
├── rb5_850e_payload_1kg.xml
├── rb5_payload_gc_rotation_pub.py
├── web_ui/
│   ├── src/
│   ├── package.json
│   └── README.md
└── src/
    ├── real/
    │   ├── dg5f_grasp_interfaces/
    │   │   └── msg/
    │   │       └── GraspDebug.msg
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
    │       │   ├── ros_debug.py
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
| `src/real/dg5f_grasp_control/dg5f_grasp_control/ros_debug.py` | 공통 제어 결과를 고정된 5손가락 `GraspDebug` 메시지로 변환 |
| `src/real/dg5f_grasp_interfaces/msg/GraspDebug.msg` | 웹/RViz/Foxglove 시각화를 위한 fingertip, centroid, force, torque 인터페이스 |
| `src/mujoco/grasp_sim.py` | MuJoCo model과 공통 `GraspController`를 연결하는 simulation adapter |
| `src/real/dg5f_grasp_control/config/grasp_real.yaml` | real과 MuJoCo가 공유하는 controller parameter |
| `src/vendor` | DG5F-S driver, description, hardware interface, TCP communication package |
| `web_ui` | rosbridge를 통해 JointState/GraspDebug를 시각화하고 고수준 명령을 보내는 React UI |
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
### Clean build

```bash
cd ~/hand
rm -rf build install log
colcon build --symlink-install
source install/setup.bash
```

제어 package만 다시 빌드하려면:

```bash
cd ~/hand
colcon build --symlink-install \
  --packages-select dg5f_grasp_interfaces dg5f_grasp_control
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
source /opt/ros/humble/setup.bash
source install/setup.bash
export ROS_DOMAIN_ID=73
ros2 launch dg5f_grasp_control grasp_with_effort.launch.py
```

### Option 2. Driver와 grasp controller를 따로 실행

Terminal 1: DG5F-S effort controller

```bash
cd ~/hand
source /opt/ros/humble/setup.bash
source install/setup.bash
export ROS_DOMAIN_ID=73
ros2 launch dg5f_s_driver dg5f_s_left_effort_controller.launch.py
```

Terminal 2: grasp controller

```bash
cd ~/hand
source /opt/ros/humble/setup.bash
source install/setup.bash
export ROS_DOMAIN_ID=73
ros2 launch dg5f_grasp_control grasp_real.launch.py
```

### 새 터미널에서 명령 전송

컨트롤러를 실행한 터미널은 그대로 두고 새 터미널에서 아래 블록을 실행합니다.
컨트롤러와 명령 전송 터미널의 `ROS_DOMAIN_ID`는 반드시 같아야 합니다.

```bash
cd ~/hand
source /opt/ros/humble/setup.bash
source install/setup.bash
export ROS_DOMAIN_ID=73

ros2 topic pub --once /grasp_type std_msgs/msg/Int32 "{data: 3}"
```

---

## Run MuJoCo Simulation

MuJoCo simulation도 real과 동일한 `GraspController`와 `grasp_real.yaml`을 사용합니다.

```bash
cd ~/hand
source /opt/ros/humble/setup.bash
source install/setup.bash
export ROS_DOMAIN_ID=73
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
| Relative task-space target | `/dg5f_grasp_control/relative_translation_cmd` | `geometry_msgs/msg/Vector3Stamped` |
| Relative rotation target (degrees) | `/dg5f_grasp_control/relative_rotation_deg_cmd` | `std_msgs/msg/Float64` |
| Hand rotation matrix | `/dg5f_grasp_control/rotation_matrix_cmd` | `std_msgs/msg/Float64MultiArray` |
| Real hand joint state | `/dg5f_s_left/joint_states` | `sensor_msgs/msg/JointState` |
| Real hand effort command | `/dg5f_s_left/effort_controller/commands` | `std_msgs/msg/Float64MultiArray` |
| Control visualization debug | `/dg5f_grasp_control/debug` | `dg5f_grasp_interfaces/msg/GraspDebug` |

기존 `/dg5f_grasp_control/finger_count_cmd` 대신 `/grasp_type`을 사용합니다.

---

## Debug Visualization Topic

real hand와 MuJoCo adapter는 제어기에서 실제로 계산한 Cartesian force와
controller 상태를 다음 topic으로 발행합니다.

```text
/dg5f_grasp_control/debug
```

기본 발행 주기는 `20 Hz`이며 모든 위치와 Cartesian force는
`header.frame_id=link_base` 좌표계입니다. 손가락 기반 배열은 항상
`finger_ids=[1, 2, 3, 4, 5]` 순서와 길이 5를 사용합니다. 비활성 손가락의
`alpha`와 force는 `0`입니다.

주요 필드는 다음과 같습니다.

- 5개 fingertip position
- geometric centroid `Cg`와 virtual centroid `Cv`
- 상대 병진 시작/목표 centroid, 명령 변위, 남은 오차와 phase
- 손가락별 `alpha`
- grasp, rotation, center-hold, collision, total force
- 보상 전 controller torque와 제한 적용 후 최종 commanded effort
- 현재 grasp type, pose type, teaching mode, controller state/phase

확인 명령:

```bash
ros2 interface show dg5f_grasp_interfaces/msg/GraspDebug
ros2 topic echo /dg5f_grasp_control/debug --once
ros2 topic hz /dg5f_grasp_control/debug
```

`debug_publish_hz`를 `0` 이하로 설정하면 debug 발행을 비활성화합니다.
Pose, Teaching, Envelop mode는 joint-space 제어이므로 Cartesian force 배열이
0인 것이 정상입니다.

---

## Web Control UI

`web_ui`에는 실제 `dg5fs_left.urdf`/CAD mesh를 JointState 20축으로 구동하는
3D hand, 고정된 월드 X/Y/Z 축, GraspDebug의 fingertip·centroid·계산 force
overlay, 월드 X/Y/Z 힘 이력 그래프와 시간 초기화, 그리고 Teaching, Pose, Grasp,
상대 회전 목표, Alpha1, rotation matrix 명령을 전송하는 React UI가 있습니다.
파지 후 활성화되는 `04 Task-Space Position`에서 원하는 이동량(mm)과
`±X/±Y/±Z` World 방향을 선택할 수 있습니다. 상대 목표를 ROS로 전달하면
3축 Cartesian impedance와 합모멘트 0 contact-force 분배로 실제 이동 토크가
적용됩니다. 최초 하드웨어 시험은 반드시 `1 mm`로 시작하십시오. Node.js 24 LTS와
rosbridge 설치 후
다음 순서로 실행합니다.

손 컨트롤러를 별도 터미널에서 먼저 실행한 뒤, rosbridge와 웹 UI를 한꺼번에
실행하려면:

```bash
cd ~/hand
export ROS_DOMAIN_ID=73
./start_web.sh
```

`9090 포트가 이미 사용 중`이라고 나오면 이전 rosbridge가 남아 있는지 먼저
확인합니다.

```bash
ss -ltnp 'sport = :9090'
pgrep -af 'rosbridge|ros2 launch rosbridge'
```

출력된 `ros2 launch rosbridge_server ...` 프로세스의 PGID를 확인해 프로세스
그룹을 종료합니다. 예를 들어 PID/PGID가 `34769`인 경우:

```bash
ps -o pid,pgid,cmd -p 34769
kill -INT -- -34769
sleep 1

cd ~/hand
./start_web.sh
```

`34769`는 예시이므로 실제 출력의 PGID로 바꿔야 합니다. 가능하면 기존
`start_web.sh` 터미널에서 `Ctrl+C`로 종료하는 것이 우선입니다.

기본 ROS domain은 `73`입니다. 스크립트는 중복 포트를 검사하고 준비 상태를
기다린 뒤 실행합니다. `Ctrl+C`는 rosbridge와 웹 UI만 종료하며 별도로 실행한
손 컨트롤러는 유지합니다. 별도 실행한 손 컨트롤러도 반드시 같은
`ROS_DOMAIN_ID=73`을 사용해야 합니다. 종료 전 반드시 웹에서 `RELEASE` →
`NORMAL_POSE`를 확인하십시오. 아래는 구성 요소를 각각 실행하는 방법입니다.

```bash
sudo apt update
sudo apt install ros-humble-rosbridge-suite
```

Terminal 1 — hand controller 실행 후 rosbridge:

```bash
source /opt/ros/humble/setup.bash
source ~/hand/install/setup.bash
export ROS_DOMAIN_ID=73
ros2 launch rosbridge_server rosbridge_websocket_launch.xml
```

Terminal 2 — UI:

```bash
cd ~/hand/web_ui
npm ci
npm run dev
```

같은 PC의 브라우저에서 `http://127.0.0.1:8080`을 엽니다. 다른 PC에서 접속할
때만 `npm run dev:lan`을 사용하십시오. UI 명령은 rosbridge 연결뿐 아니라 최근
1초 이내의 JointState와 GraspDebug가 모두 확인되어야 활성화됩니다.

설치, 원격 접속, 안전 조건 및 troubleshooting은
[`web_ui/README.md`](web_ui/README.md)를 참고하십시오.

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
- 새로 추가되는 손가락은 먼저 현재 pre-grasp pose로 이동한 뒤 grasp torque 제어로 전환됩니다.

### 비사용 손가락 자세

일반 groped grasp에서 사용하지 않는 손가락은 PD로 다음 자세를 유지합니다.

- 엄지, 검지, 중지, 약지: 첫 번째 관절만 `HAND_PRE_GRASP_POSE` 값 사용
- 새끼: 첫 번째와 두 번째 관절을 `HAND_PRE_GRASP_POSE` 값 사용
- 나머지 관절: `0.0 rad`

---

## 2. Pose Type Command

normal pose와 두 종류의 pre-grasp pose를 선택합니다.

| Value | Meaning |
| ---: | --- |
| `1` | `HAND_NORMAL_POSE` (평소 자세)|
| `2` | 기본 `HAND_PRE_GRASP_POSE` (큰 물체 잡기 전)|
| `3` | `HAND_COMPACT_PRE_GRASP_POSE` (작은 물체 잡기 전)|

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


### Virtual centroid policy

- 일반 `grasp_type=1~5`: 손가락 수와 관계없이 항상 `Cv = Cg`
- `grasp_type=7`: 기존 thumb-biased virtual centroid와 legacy force heuristic 유지

`thumb_centroid_bias`는 이제 `grasp_type=7`에만 적용됩니다.

```text
Cv = Cg + thumb_centroid_bias × (P_thumb - Cg)   # grasp_type=7 only
```

기본값:

```yaml
thumb_centroid_bias: 0.5
```

### Force distribution

교수님과 확인한 corrected interpretation을 반영하여, 일반
`grasp_type=1~5`는 centroid와의 거리에 **비례**하는 nominal force
coefficient를 사용합니다. Finger ID `1`은 엄지이며, UI와
`/dg5f_grasp_control/alpha1_cmd`의 `alpha1`은 항상 엄지 force magnitude입니다.

```text
d_i       = ||Cg - P_i||
alpha_thumb = alpha1
alpha_i     = alpha1 × d_i / d_thumb
fhat_i      = (Cg - P_i) / d_i
```

- 2F: `Cg`가 두 접촉점의 중점이므로 두 손가락의 alpha가 같습니다.
- 3F: `Cg = (P1 + P2 + P3) / 3`이므로 위 거리 비례식이
  `Σ(alpha_i × fhat_i) = 0`을 직접 만족합니다.
- 4F·5F: polygon signed-area centroid는 일반적으로 꼭지점의 산술평균과
  다르므로 nominal 거리 비례값만으로는 합력 0이 보장되지 않습니다.
  엄지 `alpha1`을 고정한 채, nominal 분포와 가장 가까운 비음수 3차원
  force-balance 해를 계산해 나머지 alpha를 보정합니다.

```text
Σ (alpha_i × fhat_i) ≈ 0,  alpha_i >= 0
```

`rotation_force_balance_max_alpha_ratio`는 3F~5F에서 다른 손가락 alpha가
엄지 `alpha1` 대비 과도하게 커지지 않도록 제한합니다. 제한을 넘거나 비음수
평형 해를 찾지 못하면 legacy 계산으로 전환하지 않습니다. 마지막으로 검증된
평형 Cartesian force를 현재 Jacobian으로 다시 매핑하면서
`force_balance_error_ramp_sec` 동안 0까지 낮춘 뒤 0을 유지하고, Debug phase를
`force_balance_error`로 고정합니다. 손 형상을 확인한 후 grasp type을 다시
선택해야 새 평형 계산을 시작합니다.

`grasp_type=7`은 이 일반 정책의 적용 대상이 아닙니다. 기존 thumb bias,
거리 반비례 분포, 마지막 active finger를 pivot으로 사용하는 legacy
heuristic을 그대로 유지합니다.

위 합력 0 조건은 손가락 구성이 안정된 일반 grasp의 계산값 기준입니다. 손가락을
추가하는 짧은 blend 구간에는 새 손가락 force를 단계적으로 연결하므로 UI 합력에
일시적인 잔차가 보일 수 있으며, 이 동안 상대 회전 명령은 잠깁니다.

### Relative rotation command (immediate ready)

`grasp_type=1~5`로 물체를 잡은 상태에서 다음 topic에 signed degree를 보내면,
현재 물체 자세를 기준으로 한 상대 회전 목표를 저장합니다. 이 값은 절대
각도가 아니며 연속 명령을 누적한 절대 자세도 아닙니다.

```bash
ros2 topic pub --once \
  /dg5f_grasp_control/relative_rotation_deg_cmd \
  std_msgs/msg/Float64 \
  "{data: 30.0}"
```

- 양수: 이후 구현할 회전축 기준 CCW 상대 회전
- 음수: 이후 구현할 회전축 기준 CW 상대 회전
- `0`, `NaN`, `Inf`: 거부
- `grasp_type=1~5`: 최소 한 번의 정상 force-balance control cycle이 확인되면,
  별도의 centroid 이동이나 시간 전환 없이 즉시 `rotation_ready`
- 손가락 추가·제거 전환 중이거나 Teaching Hold 중인 경우 명령 거부
- `force_balance_error` 상태에서는 명령을 거부하며 grasp type 재선택 필요

기존의 `centroid_redistributing` 단계와
`rotation_centroid_transition_sec`에 따른 `Cv → Cg` 전환은 더 이상 사용하지
않습니다. Debug의 `controller_phase`는 유효한 명령을 받으면 즉시
`rotation_ready`로 표시됩니다. pose/grasp 변경 또는 Teaching ON은 저장된 상대
각도와 phase를 초기화합니다.

> 현재는 상대 회전 목표만 저장하며 접선 회전력은 아직 적용하지
> 않습니다. `rotation_ready`는 명령이 수락되었고 centroid/force 조건이
> 이미 준비됐다는 뜻이지, 물체가 입력 각도만큼 회전했다는 뜻이 아닙니다.
> `Cg`는 live joint 위치에서 매 cycle 다시 계산되고 접촉력/물체 pose sensor
> feedback은 없으므로 실제 물체의 완전한 무이동은 보장하지 않습니다.

### Fingertip collision repel (중지<->약지<->새끼 간 충돌 방지용)

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

## 4. Enveloping Grasp

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
torque = min(alpha1 × envelop_tau_scale, groped_tau_limit)
```

기본 설정:

```yaml
envelop_tau_scale: 0.025
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
envelop_tau_scale = 0.025

torque = min(3.0 × 0.025, 3.0) = 0.075
```

토크가 너무 강하면 `envelop_tau_scale`을 낮추십시오.

---

## 5. Grasp Type 7: Rotation and Finger Transition (미완성)

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

- `grasp_type=1~5`: 엄지(Finger ID 1) magnitude이며 거리 비례
  nominal distribution과 4F·5F 평형 보정의 기준값
- `grasp_type=6`: `alpha1 × envelop_tau_scale`로 active joint torque 결정
- `grasp_type=7`: legacy 4손가락 grasp force heuristic에 반영

값이 클수록 grasp force가 커지고, 작을수록 작아집니다.

---

## 7. Rotation Matrix Command

로봇팔 끝에 hand가 장착된 경우 hand frame 기준 중력 방향과 world 기준 명령을
계산하기 위한 rotation matrix를 전달합니다. Hand controller는 특정 로봇팔의
API나 kinematics에 직접 의존하지 않으며, 외부 시스템에서 아래 topic 규약에 맞는
`R_hand_to_world`만 보내면 됩니다.

| 구분 | 값 |
| --- | --- |
| Topic | `/dg5f_grasp_control/rotation_matrix_cmd` |
| Type | `std_msgs/msg/Float64MultiArray` |
| Data | row-major `3 × 3` rotation matrix, 총 9개 값 |
| 의미 | hand (`link_base`) 좌표의 벡터를 world 좌표로 회전하는 행렬 |
| 발행 주체 | RB5 또는 hand가 장착된 외부 로봇팔 측 node |
| 구독 주체 | real hand controller, MuJoCo controller, Web UI |

보내는 값은 row-major 순서의 `3 × 3` rotation matrix입니다.

```text
R_hand_to_world =
[ r00 r01 r02
  r10 r11 r12
  r20 r21 r22 ]
```

따라서 hand frame의 벡터 `v_hand`와 world frame의 벡터 `v_world` 관계는
다음과 같습니다.

```text
v_world = R_hand_to_world × v_hand
v_hand  = R_hand_to_worldᵀ × v_world
```

Identity matrix 예시:

```bash
ros2 topic pub --once \
  /dg5f_grasp_control/rotation_matrix_cmd \
  std_msgs/msg/Float64MultiArray \
  "{data: [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]}"
```

Real controller는 유효한 행렬 9개를 수신할 때마다 저장된 행렬과 hand frame의
중력 벡터를 즉시 갱신합니다.

```text
g_hand = R_hand_to_worldᵀ × [0, 0, -9.81]
```

이 `g_hand`는 이후 모든 control loop에서 MuJoCo hand model의 중력보상 torque를
계산하는 데 자동으로 사용됩니다. World frame으로 들어온 상대 위치 명령도 같은
행렬을 사용하여 `link_base` frame으로 변환합니다.

웹 UI도 같은 행렬을 구독하여 `link_base` 기준 debug 위치와 힘을 월드 좌표로
변환합니다.

```text
F_world = R_hand_to_world × F_link_base
```

행렬을 수신하기 전에는 controller와 UI 모두 identity matrix를 기본값으로
사용합니다. 이때 UI에는 `WORLD · DEFAULT I`로 표시되며, topic message를 수신하면
즉시 실제 행렬로 교체되어 `WORLD · TOPIC`으로 바뀝니다. 따라서 현재처럼 hand를
고정 설치하고 `link_base`와 world의 축이 같다면 identity를 별도로 발행하지 않아도
됩니다.

로봇팔에 장착하여 자세가 변하는 경우에는 로봇팔 측 node가 현재 자세로 계산한
행렬을 계속 발행해야 합니다. `Float64MultiArray`에는 timestamp가 없고 message가
자동 보관되지 않으므로, hand controller와 Web UI를 재시작한 뒤에도 새 행렬이
수신될 수 있도록 일회성이 아닌 주기 발행을 사용합니다. 발행이 중단되면 실행
중인 controller는 마지막으로 수신한 행렬을 계속 사용합니다.

rotation matrix topic을 보내지 않으면 model 내부 기본 중력 방향을 사용합니다.

MuJoCo에서도 같은 topic을 받아 simulation model의 중력 방향과 gravity compensation 계산에 반영합니다.

수신 확인:

```bash
export ROS_DOMAIN_ID=73
ros2 topic info /dg5f_grasp_control/rotation_matrix_cmd --verbose
ros2 topic echo /dg5f_grasp_control/rotation_matrix_cmd --once
```

Real controller terminal에 아래 로그가 출력되면 행렬 수신과 중력 방향 갱신이
완료된 것입니다.

```text
Updated hand gravity vector: [..., ..., ...]
```

---

## RB5 Rotation Matrix Publisher Test Example

최상위의 `rb5_payload_gc_rotation_pub.py`는 위 topic interface를 RB5에서 시험하기
위한 예제 코드입니다. Hand controller의 필수 구성요소는 아니며, 실제 통합에서는
사용할 로봇팔 측 node가 동일한 topic, message type, 행렬 정의만 만족하면 됩니다.

이 테스트 코드는 RB5 joint state로 TCP 자세를 계산하고, hand 고정 장착 회전을
반영한 `R_link_mount_to_world`를 row-major 배열로 주기 발행합니다. 현재 hand
model에서 `link_mount`와 `link_base`는 위치 차이만 있고 축 방향은 같으므로 이
회전행렬을 `R_hand_to_world`로 사용할 수 있습니다.

RB5 테스트에서는 RB5 제어와 hand 제어를 별도 process로 실행합니다.

Terminal 1: hand controller

```bash
cd ~/hand
source /opt/ros/humble/setup.bash
source install/setup.bash
export ROS_DOMAIN_ID=73
ros2 launch dg5f_grasp_control grasp_with_effort.launch.py
```

Terminal 2: RB5 rotation matrix publisher

```bash
cd ~/hand
source /opt/ros/humble/setup.bash
source install/setup.bash
export ROS_DOMAIN_ID=73
python3 rb5_payload_gc_rotation_pub.py
```

테스트 코드가 발행하는 topic:

```text
/dg5f_grasp_control/rotation_matrix_cmd
```

`rb5_payload_gc_rotation_pub.py` 대신 다른 로봇팔을 사용해도 hand 측 코드는 변경할
필요가 없습니다. 해당 로봇팔에서 현재 `R_hand_to_world`를 계산하여 같은 topic으로
발행하면 좌표계 변환과 hand 중력보상이 자동으로 갱신됩니다.

---

## Main Parameters

주요 parameter는 다음 파일에서 조절합니다.

```text
src/real/dg5f_grasp_control/config/grasp_real.yaml
```

| Parameter group | Main parameters |
| --- | --- |
| ROS topics | `joint_state_topic`, `effort_topic`, `command_topic`, `pose_topic`, `relative_rotation_deg_topic`, `debug_topic` |
| Debug visualization | `debug_frame_id`, `debug_publish_hz` |
| Pose PD | `pose_kp`, `pose_kd`, `pose_pd_limit` |
| Friction | `fric_scale`, `fric_tanh_k`, `fric_limit` |
| Groped grasp (type 1~5) | `alpha1`, `groped_tau_limit`, `rotation_force_balance_max_alpha_ratio`, `force_balance_error_ramp_sec` |
| Collision repel | `min_tip_distance`, `collision_repel_gain`, `collision_repel_limit` |
| Type 6 | `envelop_tau_scale`, `envelop_joint_delay`, torque signs |
| Type 7 legacy grasp | `thumb_centroid_bias`, `alpha1` |
| Type 7 rotation | `rotation_theta_rad`, `rotation_gain`, `rotation_force_limit` |
| Type 7 center hold | `grasp_type7_center_hold_*` |
| Type 7 transition | `grasp_type7_*_transition_*`, attach force 및 torque limit |

MuJoCo도 같은 YAML 파일을 읽으므로 parameter를 한 곳에서 관리할 수 있습니다.
