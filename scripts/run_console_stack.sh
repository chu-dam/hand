#!/usr/bin/env bash

set -Eeuo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
WEB_DIR="${ROOT_DIR}/web_ui"
CONTROLLER_MANAGER="/dg5f_s_left/controller_manager"

CHECK_ONLY=false
if [[ "${1:-}" == "--check" ]]; then
  CHECK_ONLY=true
  shift
fi

if [[ -n "${1:-}" ]]; then
  cat <<'EOF'
Usage:
  scripts/run_console_stack.sh [--check]
EOF
  exit 2
fi

BRIDGE_PID=""
WEB_PID=""
CLEANUP_STARTED=false

log() {
  printf '[DG5F] %s\n' "$*"
}

warn() {
  printf '[DG5F] WARNING: %s\n' "$*" >&2
}

die() {
  printf '[DG5F] ERROR: %s\n' "$*" >&2
  exit 1
}

process_alive() {
  local pid="${1:-}"
  [[ -n "${pid}" ]] && kill -0 "${pid}" 2>/dev/null
}

port_listening() {
  local port="$1"
  [[ -n "$(ss -H -ltn "sport = :${port}" 2>/dev/null)" ]]
}

wait_for_process_exit() {
  local pid="$1"
  local attempts="$2"
  local index

  for ((index = 0; index < attempts; index += 1)); do
    if ! process_alive "${pid}"; then
      return 0
    fi
    sleep 0.1
  done
  return 1
}

signal_process_group() {
  local signal="$1"
  local pid="$2"
  local pgid

  pgid="$(ps -o pgid= -p "${pid}" 2>/dev/null | tr -d '[:space:]')"
  if [[ "${pgid}" == "${pid}" ]]; then
    kill -s "${signal}" -- "-${pid}" 2>/dev/null || true
  else
    kill -s "${signal}" "${pid}" 2>/dev/null || true
  fi
}

stop_process() {
  local label="$1"
  local pid="${2:-}"
  local graceful_attempts="$3"

  [[ -n "${pid}" ]] || return 0
  if ! process_alive "${pid}"; then
    wait "${pid}" 2>/dev/null || true
    return 0
  fi

  log "${label} 종료 중..."
  signal_process_group INT "${pid}"
  if ! wait_for_process_exit "${pid}" "${graceful_attempts}"; then
    warn "${label}가 SIGINT에 응답하지 않아 SIGTERM을 보냅니다."
    signal_process_group TERM "${pid}"
    if ! wait_for_process_exit "${pid}" 50; then
      warn "${label} 잔여 프로세스를 강제 정리합니다."
      signal_process_group KILL "${pid}"
    fi
  fi
  wait "${pid}" 2>/dev/null || true
}

cleanup() {
  local status=$?
  trap - EXIT INT TERM HUP

  if [[ "${CLEANUP_STARTED}" == true ]]; then
    return "${status}"
  fi
  CLEANUP_STARTED=true

  if [[ -n "${WEB_PID}${BRIDGE_PID}" ]]; then
    printf '\n'
    log "실행한 프로세스를 역순으로 정리합니다."
  fi
  stop_process "웹 UI" "${WEB_PID}" 50
  stop_process "rosbridge" "${BRIDGE_PID}" 100
  log "정리 완료"

  return "${status}"
}

trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
trap 'exit 129' HUP

[[ -r /opt/ros/humble/setup.bash ]] \
  || die "/opt/ros/humble/setup.bash를 찾을 수 없습니다."
[[ -r "${ROOT_DIR}/install/setup.bash" ]] \
  || die "${ROOT_DIR}/install/setup.bash를 찾을 수 없습니다. 먼저 colcon build를 실행하세요."

# ROS/colcon setup files may read optional unset variables.
set +u
source /opt/ros/humble/setup.bash
source "${ROOT_DIR}/install/setup.bash"
set -u

export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-73}"

if [[ -x "${WEB_DIR}/.node/bin/node" ]]; then
  NODE_BIN="${WEB_DIR}/.node/bin/node"
  export PATH="${WEB_DIR}/.node/bin:${PATH}"
else
  NODE_BIN="$(command -v node || true)"
fi
VITE_ENTRY="${WEB_DIR}/node_modules/vite/bin/vite.js"

for command_name in ros2 timeout ss setsid ps tr; do
  command -v "${command_name}" >/dev/null 2>&1 \
    || die "필수 명령을 찾을 수 없습니다: ${command_name}"
done
[[ -n "${NODE_BIN}" && -x "${NODE_BIN}" ]] \
  || die "Node.js를 찾을 수 없습니다. web_ui/.node 설치 상태를 확인하세요."
[[ -f "${VITE_ENTRY}" ]] \
  || die "Vite를 찾을 수 없습니다. cd web_ui && npm ci 를 먼저 실행하세요."
ros2 pkg prefix dg5f_grasp_control >/dev/null 2>&1 \
  || die "dg5f_grasp_control package를 찾을 수 없습니다. workspace를 다시 빌드하세요."
ros2 pkg prefix rosbridge_server >/dev/null 2>&1 \
  || die "rosbridge_server를 찾을 수 없습니다. ros-humble-rosbridge-suite를 설치하세요."

if port_listening 9090; then
  die "9090 포트가 이미 사용 중입니다. 기존 rosbridge를 먼저 종료하세요."
fi
if port_listening 8080; then
  die "8080 포트가 이미 사용 중입니다. 기존 웹 UI를 먼저 종료하세요."
fi

controller_manager_available() {
  timeout 4 ros2 control list_controllers \
    -c "${CONTROLLER_MANAGER}" >/dev/null 2>&1
}

if controller_manager_available; then
  log "외부에서 실행 중인 손 컨트롤러를 확인했습니다."
else
  warn "손 컨트롤러가 아직 보이지 않습니다. 웹은 실행되지만 HAND/DEBUG는 대기 상태가 됩니다."
fi

if [[ "${CHECK_ONLY}" == true ]]; then
  log "사전 점검 통과 · ROS_DOMAIN_ID=${ROS_DOMAIN_ID}"
  exit 0
fi

wait_for_port() {
  local label="$1"
  local port="$2"
  local pid="$3"
  local timeout_seconds="$4"
  local deadline=$((SECONDS + timeout_seconds))

  while ((SECONDS < deadline)); do
    process_alive "${pid}" || die "${label}가 준비 전에 종료되었습니다."
    if port_listening "${port}"; then
      log "${label} 준비 완료 · 127.0.0.1:${port}"
      return 0
    fi
    sleep 0.25
  done

  die "${label}가 ${timeout_seconds}초 안에 ${port} 포트를 열지 못했습니다."
}

log "rosbridge 시작"
setsid ros2 launch rosbridge_server rosbridge_websocket_launch.xml \
  address:=127.0.0.1 port:=9090 &
BRIDGE_PID=$!
wait_for_port "rosbridge" 9090 "${BRIDGE_PID}" 30

log "웹 UI 시작"
(
  cd "${WEB_DIR}"
  exec setsid "${NODE_BIN}" "${VITE_ENTRY}" \
    --host 127.0.0.1 --port 8080 --strictPort
) &
WEB_PID=$!
wait_for_port "웹 UI" 8080 "${WEB_PID}" 20

printf '\n'
log "전체 준비 완료: http://127.0.0.1:8080"
log "이 터미널의 Ctrl+C는 웹 UI와 rosbridge만 종료하며, 외부 손 컨트롤러는 유지합니다."
warn "종료 전 웹에서 RELEASE를 누르고 NORMAL_POSE를 확인한 뒤 Ctrl+C를 누르세요."
printf '\n'

while true; do
  process_alive "${WEB_PID}" || die "웹 UI가 예기치 않게 종료되었습니다."
  process_alive "${BRIDGE_PID}" || die "rosbridge가 예기치 않게 종료되었습니다."
  sleep 1
done
