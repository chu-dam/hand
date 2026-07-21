# DG5F-S Web Control UI

React + TypeScript 기반의 DG5F-S 고수준 제어·시각화 화면입니다. 브라우저가
`roslib`으로 rosbridge WebSocket에 연결하여 실제 ROS 2 topic을 구독하고
발행합니다.

```text
DG5F-S controller ── ROS 2 topics ── rosbridge :9090 ── browser UI :8080
```

현재 버전은 다음 기능을 제공합니다.

- `/dg5f_s_left/joint_states` 기반 20관절 상태 표시
- controller와 동일한 `dg5fs_left.urdf`/CAD mesh 기반 실시간 3D hand
- `/dg5f_grasp_control/debug` 기반 3D fingertip, centroid, 계산 force overlay
- 손가락별 `total_forces`와 축별 합력의 X/Y/Z 실시간 이력 그래프 및 시간 초기화
- 마우스 회전·확대/축소·이동 및 force vector 크기 조절
- 파지 후 활성화되는 Task-Space Position UI (`±X/±Y/±Z`, 상대 이동량 mm)
- Teaching, Pose, Grasp Type, 상대 회전 목표, Alpha1, hand rotation matrix 명령
- 연결이 끊기거나 telemetry가 1초 이상 오래되면 모든 제어 명령 자동 잠금
- 회전행렬의 직교성 및 `det(R)≈1` 검증

> 화면의 force는 센서 측정값이 아니라 controller가 계산한 Cartesian force입니다.
> `RELEASE`는 emergency stop이 아니라 `grasp_type=-1` normal-pose 명령입니다.

## 1. 준비

Node.js 24 LTS와 rosbridge가 필요합니다.

```bash
node --version
sudo apt update
sudo apt install ros-humble-rosbridge-suite
```

Node가 없다면 [Node.js 24 LTS 공식 배포본](https://nodejs.org/download/release/latest-v24.x/)을
먼저 설치합니다. 이 프로젝트에는 `.nvmrc`도 포함되어 있어 nvm 사용 시
`nvm install && nvm use`로 버전을 맞출 수 있습니다.

프런트엔드 의존성 설치:

```bash
cd ~/hand/web_ui
npm ci
```

## 2. 실행

### rosbridge와 웹 UI 한 번에 실행

손 컨트롤러를 별도 터미널에서 먼저 실행한 뒤 다음 명령을 사용합니다.

```bash
cd ~/hand
export ROS_DOMAIN_ID=73
./start_web.sh
```

`9090 포트가 이미 사용 중`이라고 나오면 이전 rosbridge 프로세스를 확인합니다.

```bash
ss -ltnp 'sport = :9090'
pgrep -af 'rosbridge|ros2 launch rosbridge'
```

`ros2 launch rosbridge_server ...`의 PGID가 `34769`로 확인된 사례라면 다음처럼
종료한 뒤 다시 실행합니다.

```bash
kill -INT -- -34769
sleep 1

cd ~/hand
./start_web.sh
```

PID/PGID는 실행할 때마다 달라지므로 `34769`는 실제 조회된 PGID로 바꿔야 합니다.
기존 `start_web.sh` 터미널이 남아 있다면 그 터미널에서 `Ctrl+C`를 누르는 것이
가장 안전합니다.

스크립트는 기본적으로 `ROS_DOMAIN_ID=73`과 로컬 주소
`127.0.0.1:9090/8080`을 사용합니다. 별도로 실행한 손 컨트롤러도 반드시 같은
domain을 사용해야 합니다. 다른 domain이 필요하면 다음처럼 실행합니다.

```bash
ROS_DOMAIN_ID=42 ./start_web.sh
```

실행한 구성 요소의 로그는 한 터미널에 함께 표시됩니다. 종료 전 웹에서
`RELEASE`를 누르고 `NORMAL_POSE`를 확인한 다음 `Ctrl+C`를 누르십시오.
`start_web.sh`는 자신이 실행한 rosbridge와 웹 UI만 종료하여 기존 손 컨트롤러는
유지합니다.

아래 내용은 각 구성 요소를 별도 터미널에서 직접 실행하는 방법입니다.

실제 hand controller가 이미 실행 중인 상태에서 별도 terminal을 엽니다.

Terminal A — rosbridge:

```bash
source /opt/ros/humble/setup.bash
source ~/hand/install/setup.bash
export ROS_DOMAIN_ID=73
ros2 launch rosbridge_server rosbridge_websocket_launch.xml
```

workspace를 source한 terminal에서 rosbridge를 실행해야 custom
`dg5f_grasp_interfaces/msg/GraspDebug` type을 찾을 수 있습니다.

Terminal B — 같은 PC에서만 UI 사용:

```bash
cd ~/hand/web_ui
npm run dev
```

브라우저에서 `http://127.0.0.1:8080`을 엽니다. 기본 WebSocket 주소는
`ws://127.0.0.1:9090`입니다.

다른 PC에서 접속해야 할 때만 다음처럼 LAN 공개 모드를 사용합니다.

```bash
cd ~/hand/web_ui
npm run dev:lan
```

제어 PC에서 `http://ROBOT_PC_IP:8080`을 열면 UI가 같은 hostname의
`ws://ROBOT_PC_IP:9090`에 자동 연결합니다. WebSocket 주소가 다르면 화면 상단
입력창에서 변경하거나 다음 query parameter를 사용할 수 있습니다.

```text
http://ROBOT_PC_IP:8080/?rosbridge=ws%3A%2F%2FROSBRIDGE_IP%3A9090
```

rosbridge는 ROS topic publish 권한을 브라우저에 노출합니다. `dev:lan`과 9090
포트는 격리된 연구실 LAN에서만 열고, 공용 Wi-Fi나 인터넷에는 공개하지 마십시오.

## 3. 정상 상태 확인

화면 상단이 다음 상태가 되면 명령 버튼이 활성화됩니다.

```text
ROS CONNECTED · HAND LIVE · DEBUG < 1.0s · Commands READY
```

`ROS CONNECTED`인데 `HAND` 또는 `DEBUG`가 `WAITING`이면 다음을 확인합니다.

```bash
source /opt/ros/humble/setup.bash
source ~/hand/install/setup.bash
export ROS_DOMAIN_ID=73
ros2 topic hz /dg5f_s_left/joint_states
ros2 topic hz /dg5f_grasp_control/debug
```

`GraspDebug`만 보이지 않으면 rosbridge를 종료한 뒤 반드시 workspace를 source하고
다시 실행합니다.

## 4. 제어 안전 조건

- `JointState`와 `GraspDebug`가 모두 최근 1초 안에 수신될 때만 명령 가능
- Task-Space Position은 일반 `grasp_type=1~5`에서만 활성화되며,
  `Vector3Stamped(world)` 상대 목표를 ROS로 보내 3축 Cartesian impedance로 이동함
- 실제 하드웨어 이동 명령이므로 최초 시험은 `1 mm`로 하고 RELEASE를 준비할 것
- Teaching Mode 중 Pose, Grasp, Alpha1, Release 명령 잠금
- Relative Rotation은 `controller_state=GROPED_GRASP`인 `grasp_type=1~5`에서만
  활성화되며 Teaching Hold 또는 `force_balance_error` 중에는 거부하고 0이 아닌
  signed degree만 허용
- hand orientation matrix는 `NORMAL_POSE`에서만 적용 가능
- matrix 입력은 빈칸·비유한값·비직교행렬·반사행렬을 거부
- 명령 전송 메시지는 적용 완료를 의미하지 않으므로 화면의 Debug echo를 확인

Task-Space Position의 `Set Target`은 선택한 World 축과 0~10 mm 이동량을 다음
topic에 SI 단위(m)로 발행합니다.

```text
/dg5f_grasp_control/relative_translation_cmd (geometry_msgs/msg/Vector3Stamped)
```

제어기는 최신 hand-to-world 회전행렬로 명령을 `link_base`로 변환하고, 명령 순간의
`Cg`에 상대 변위를 더해 목표를 저장합니다. X/Y/Z 오차를 동시에 PD 제어하며,
손가락별 이동 힘은 합력은 목표 이동력을 만들고 합모멘트는 0이 되도록 분배합니다.
기본 timeout은 3초이고, timeout/error 시 이동 추가 힘을 제거합니다.

Rotation 입력은 현재 물체 자세 기준의 상대 각도입니다. 양수는 CCW, 음수는 CW이며
다음 topic으로 degree 단위 그대로 발행합니다.

```text
/dg5f_grasp_control/relative_rotation_deg_cmd (std_msgs/msg/Float64)
```

일반 `grasp_type=1~5`는 항상 `Cv=Cg`를 사용합니다. 엄지(Finger ID 1)의
`alpha1`을 기준으로 centroid 거리 비례 nominal force를 만들며, 4F·5F는 비음수
3차원 평형 해로 보정합니다. 최소 한 번의 정상 control cycle 후 유효한 상대 회전
명령은 별도 centroid 전환 없이 즉시 `controller_phase=rotation_ready`가 됩니다.
평형 계산이 안전 제한을 넘으면 `force_balance_error`가 표시되고 grasp type을 다시
선택할 때까지 회전 명령을 거부합니다. 현재는 상대 목표만 저장하고 접선 회전력은 아직
적용하지 않으므로, 이 상태는 물체가 입력 각도만큼 회전했다는 뜻이 아닙니다. 센서 기반
물체 pose hold도 아니므로 실제 무이동은 별도로 확인해야 합니다.

## 5. 빌드

```bash
cd ~/hand/web_ui
npm run build
npm run preview
```

production 결과는 `web_ui/dist/`에 생성됩니다.

## 6. 실시간 3D 모델

3D 화면은 실제 controller가 사용하는 mount 없는 `dg5fs_left.urdf`와 DAE visual
mesh를 사용합니다. `/dg5f_s_left/joint_states`의 `name[i]`와 `position[i]`를 이름
기준으로 대응시켜 20개 관절에 radian 값을 그대로 적용하므로, 배열 순서가 바뀌어도
동일하게 동작합니다. 별도의 `/tf` 구독은 필요하지 않습니다.

`GraspDebug`의 fingertip, total force, `Cg`, `Cv` 원본은 모두 `link_base`
기준입니다. UI는 `/dg5f_grasp_control/rotation_matrix_cmd`에서 row-major
`R_hand_to_world`를 함께 구독하여 위치와 힘을 월드 좌표로 회전하고, 3D 화면의
고정된 X/Y/Z 월드 축과 힘 이력 그래프에 표시합니다.

```text
p_world = R_hand_to_world × p_link_base
F_world = R_hand_to_world × F_link_base
```

컨트롤러의 model 기본 중력 `[0, 0, -9.81]`과 동일하게, 행렬을 아직 수신하지
않았으면 UI는 identity matrix를 기본값으로 사용하고 `WORLD · DEFAULT I`로
표시합니다. `/dg5f_grasp_control/rotation_matrix_cmd`가 들어오면 즉시 해당 행렬로
교체되고 상태가 `WORLD · TOPIC`으로 바뀝니다. 따라서 고정 설치로 두 좌표계가
같으면 별도 입력 없이 바로 사용할 수 있습니다. RB5 사용 시에는
`rb5_payload_gc_rotation_pub.py`를 계속 실행합니다. 이 topic에는 timestamp가
없으므로 UI는 각 `GraspDebug` 수신 시점의 최신 행렬을 사용합니다.

Force scale의 단위는 `mm/N`이고 화면의 화살표 길이만 바뀌며 controller 계산에는
영향을 주지 않습니다.

- 왼쪽 drag: 회전
- wheel 또는 pinch: 확대/축소
- 오른쪽 drag: 이동
- `Reset view`: 초기 카메라 복원

브라우저용 URDF와 mesh는 `public/robot/dg5f_s_description/`에 포함되어 production
build에서도 ROS package 경로와 무관하게 로드됩니다.
