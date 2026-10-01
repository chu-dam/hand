#!/usr/bin/env python3
"""왼손 DG-5F-S20: ROS/SDK 없이 Modbus TCP로 자세 실행.

손을 Operator 모드로 설정하고 DGManager/ROS 연결을 종료한 뒤 실행:
    python3 left_hand_aging.py --ip 169.254.186.73
    python3 left_hand_aging.py --self-test  # 장비 연결 없는 검사

연결 직후 0도 이동 후 pose_1 → pose_2 → pose_3 → pose_4 → 0도를 반복한다.
사이클별 작동시간은 1.0 → 0.8 → 0.6 → 0.4초로 반복한다.
목표 도달 여부와 관계없이 작동시간 + 0.2초 뒤 다음 자세로 넘어간다.
전원만 켰을 때 자동 실행되는 펌웨어 설정은 변경하지 않는다.
새 자세는 POSES와 SEQUENCE에 J1~J20 순서로 추가한다. 각도 단위는 degree.
종료 시 System Stop을 전송한다(자세 유지 해제). 통신 단절 시 정지 보장 불가.

근거: DG-5F-S User Manual EN v1.0.2, 6.2 (주소는 0-based).
Holding: 0 start/stop, 2 teach, 7..26 목표(0.1도), 27..46 시간(ms),
286 blend, 292 grasp, 389 current mode. Input: 0 모델, 66..85 온도(0.1°C).
"""

import argparse
import math
import socket
import struct
import sys
import time


POSES = {
    "0": [0] * 20,
    "pose_1": [
        0, 150, 90, -90,    # FINGER 1: J1~J4
        0, 120, -90, -90,   # FINGER 2: J5~J8
        0, 120, -90, -90,   # FINGER 3: J9~J12
        0, 120, -90, -90,   # FINGER 4: J13~J16
        0, 0, -90, -90,     # FINGER 5: J17~J20
    ],
    "pose_2": [
        0, 0, -90, 90,
        0, 0, 90, 90,
        0, 0, 90, 90,
        0, 0, 90, 90,
        0, 0, 90, 90,
    ],
    "pose_3": [
        -90, 90, 0, 0,
        48, 0, 0, 0,
        32.7, 0, 0, 0,
        12, 0, 0, 0,
        -90, -90, 0, 0,
    ],
    "pose_4": [
        90, 90, 0, 0,
        -15, 0, 0, 0,
        -42, 0, 0, 0,
        -48, 0, 0, 0,
        -90, -90, 0, 0,
    ],
}
SEQUENCE = ("pose_1", "pose_2", "pose_3", "pose_4", "0")
MOTION_TIMES = (1.0, 0.8, 0.6, 0.4)
TIMEOUT_MARGIN = 0.2


def encode_pose(pose):
    if len(pose) != 20:
        raise ValueError("pose에는 J1~J20 각도 20개가 필요합니다.")
    for joint, angle in enumerate(pose, 1):
        if not math.isfinite(angle) or not -180 <= angle <= 180:
            raise ValueError(f"J{joint}: {angle}도는 Modbus 허용 범위 -180~180 밖입니다.")
    return [round(angle * 10) & 0xFFFF for angle in pose]


class Modbus:
    def __init__(self, sock, unit=1):
        self.sock = sock
        self.unit = unit
        self.transaction = 0

    def recv_exact(self, size):
        data = bytearray()
        while len(data) < size:
            chunk = self.sock.recv(size - len(data))
            if not chunk:
                raise ConnectionError("손과의 TCP 연결이 끊어졌습니다.")
            data.extend(chunk)
        return bytes(data)

    def request(self, pdu):
        self.transaction = (self.transaction + 1) & 0xFFFF
        self.sock.sendall(struct.pack(">HHHB", self.transaction, 0, len(pdu) + 1,
                                      self.unit) + pdu)
        transaction, protocol, length, unit = struct.unpack(">HHHB", self.recv_exact(7))
        if (transaction, protocol, unit) != (self.transaction, 0, self.unit):
            raise ValueError("Modbus 응답 헤더 불일치: Operator 모드를 확인하세요.")
        if not 2 <= length <= 254:
            raise ValueError(f"잘못된 Modbus 응답 길이: {length}")
        response = self.recv_exact(length - 1)
        if response[0] == (pdu[0] | 0x80):
            raise RuntimeError(f"Modbus 예외 응답: {response.hex(' ')}")
        if response[0] != pdu[0]:
            raise ValueError("Modbus 응답 function 불일치")
        return response

    def write(self, address, values):
        payload = struct.pack(f">{len(values)}H", *values)
        pdu = struct.pack(">BHHB", 16, address, len(values), len(payload)) + payload
        if self.request(pdu) != pdu[:5]:
            raise ValueError("Modbus 쓰기 확인 응답 불일치")

    def read_input(self, address, count):
        response = self.request(struct.pack(">BHH", 4, address, count))
        if len(response) != 2 + count * 2 or response[1] != count * 2:
            raise ValueError("Modbus 읽기 응답 길이 불일치")
        return struct.unpack(f">{count}H", response[2:])


def initialize(hand, time_ms):
    model = hand.read_input(0, 1)[0]
    if model != 0x5F14:
        raise ValueError(f"왼손 DG-5F-S20(0x5F14)이 아닙니다: 0x{model:04X}")
    hand.write(0, [0])                 # 설정 중 정지
    hand.write(286, [0])               # 기존 blend 정지
    hand.write(292, [0])               # 기존 grasp 해제
    hand.write(2, [0])                 # teach 해제
    hand.write(389, [0])               # 전류 모드 해제 → 위치 제어
    hand.write(27, [time_ms] * 20)
    hand.write(7, encode_pose(POSES["0"]))  # 시작 전에 0도 목표 지정
    hand.write(0, [1])


def log_temperatures(hand, pose_name):
    raw = hand.read_input(66, 20)
    temperatures = [(value if value < 32768 else value - 65536) / 10 for value in raw]
    highest = max(temperatures)
    print(f"  {pose_name} 온도: 최고 {highest:.1f}°C (J{temperatures.index(highest) + 1}), "
          f"평균 {sum(temperatures) / len(temperatures):.1f}°C", flush=True)


def self_test():
    """분할 TCP 수신, 부호/배율, 초기화 순서 및 오류 처리를 장비 없이 검사."""
    class FakeSocket:
        def __init__(self):
            self.buffer = b""
            self.writes = []
            self.model = 0x5F14
            self.fail = False

        def sendall(self, data):
            transaction, protocol, length, unit = struct.unpack(">HHHB", data[:7])
            assert protocol == 0 and unit == 1 and length == len(data) - 6
            function, address, count = struct.unpack(">BHH", data[7:12])
            if self.fail:
                pdu = bytes([function | 0x80, 2])
            elif function == 16:
                assert data[12] == count * 2
                self.writes.append((address, struct.unpack(f">{count}H", data[13:])))
                pdu = data[7:12]
            else:
                assert function == 4
                values = [self.model] if address == 0 else list(range(200, 400, 10))
                assert len(values) == count
                pdu = struct.pack(f">BB{count}H", 4, count * 2, *values)
            self.buffer = struct.pack(">HHHB", transaction, 0, len(pdu) + 1, unit) + pdu

        def recv(self, size):
            chunk = self.buffer[:min(size, 3)]
            self.buffer = self.buffer[len(chunk):]
            return chunk

    for pose in POSES.values():
        encode_pose(pose)
    encoded = encode_pose(POSES["pose_1"])
    assert encoded[:4] == [0, 1500, 900, 64636]
    sock = FakeSocket()
    hand = Modbus(sock)
    initialize(hand, 1000)
    assert sock.writes == [
        (0, (0,)), (286, (0,)), (292, (0,)), (2, (0,)), (389, (0,)),
        (27, (1000,) * 20), (7, (0,) * 20), (0, (1,)),
    ]
    hand.write(7, encoded)
    assert sock.writes[-1] == (7, tuple(encoded))
    log_temperatures(hand, "test")
    for invalid in ([0] * 19, [float("nan")] * 20, [181] * 20):
        try:
            encode_pose(invalid)
        except ValueError:
            pass
        else:
            raise AssertionError("잘못된 pose를 거부해야 합니다.")
    sock.model = 0x5F24
    before = len(sock.writes)
    try:
        initialize(hand, 1000)
    except ValueError:
        pass
    else:
        raise AssertionError("오른손 모델을 거부해야 합니다.")
    assert len(sock.writes) == before
    sock.fail = True
    try:
        hand.read_input(0, 1)
    except RuntimeError:
        pass
    else:
        raise AssertionError("Modbus 예외를 거부해야 합니다.")
    print("self-test OK (장비 연결 없음)")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ip", default="169.254.186.73", help="새 왼손 기본 IP")
    parser.add_argument("--port", type=int, default=502)
    parser.add_argument("--unit", type=int, default=1, help="Modbus slave ID")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return 0
    if not 1 <= args.port <= 65535 or not 1 <= args.unit <= 255:
        parser.error("port: 1~65535, unit: 1~255")
    for pose in POSES.values():
        encode_pose(pose)
    with socket.create_connection((args.ip, args.port), timeout=2) as sock:
        hand = Modbus(sock, args.unit)
        # 모델 확인 전에는 정지 명령을 포함한 쓰기 명령을 보내지 않는다.
        if hand.read_input(0, 1)[0] != 0x5F14:
            raise ValueError("연결 대상이 왼손 DG-5F-S20이 아닙니다.")
        try:
            print(f"연결 {args.ip}:{args.port} → 전 관절 0도 이동", flush=True)
            initialize(hand, round(MOTION_TIMES[0] * 1000))
            time.sleep(MOTION_TIMES[0] + TIMEOUT_MARGIN)
            log_temperatures(hand, "초기 0")
            cycle = 1
            while True:
                motion_time = MOTION_TIMES[(cycle - 1) % len(MOTION_TIMES)]
                timeout = motion_time + TIMEOUT_MARGIN
                hand.write(27, [round(motion_time * 1000)] * 20)
                print(f"cycle {cycle}: 작동 {motion_time:.1f}초 / timeout {timeout:.1f}초",
                      flush=True)
                for name in SEQUENCE:
                    print(f"  {name}", flush=True)
                    hand.write(7, encode_pose(POSES[name]))
                    time.sleep(timeout)
                    log_temperatures(hand, name)
                cycle += 1
        finally:
            try:
                hand.write(0, [0])
                print("System Stop 전송 완료")
            except (OSError, ValueError, RuntimeError) as exc:
                print(f"정지 확인 실패: {exc}. 손의 상태를 직접 확인하세요.", file=sys.stderr)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (KeyboardInterrupt, EOFError):
        print("\n종료")
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"오류: {exc}", file=sys.stderr)
        sys.exit(1)
