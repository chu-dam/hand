"""Delto 촉각센서 1~5 접촉 위치(x, y)와 힘(fx, fy, fz) 출력.

  송신 : 00 04 01 05                     Get Data, 타입 0x05 = 센서
  반환 : 00 3F 01 <12 B x 5 = 60 B>      타입 0x05 만 요청 → 관절 ID 블록 없음

프레임 (FDCAN_protocol_spec.pdf, big endian, 12 B):
    0-1   x       int16   0.01 mm
    2-3   y       int16   0.01 mm
    4-5   fx      int16   0.001 N
    6-7   fy      int16   0.001 N
    8-9   fz      int16   0.001 N      <- 수직력
    10-11 status  uint16  비트필드

    py -3 sensor_xyf.py             # 20 Hz 연속
    py -3 sensor_xyf.py --once
    py -3 sensor_xyf.py --hz 100
    py -3 sensor_xyf.py --csv out.csv
    py -3 sensor_xyf.py --scroll    # 덮어쓰지 않고 계속 아래로 출력
    py -3 sensor_xyf.py --status         # status 비트필드도 출력
    py -3 sensor_xyf.py --left

읽기 전용. Set Duty 없음.
"""

import argparse
import csv
import socket
import struct
import sys
import time

DEFAULT_IP = "169.254.186.72"   # 오른손
LEFT_IP = "169.254.186.73"
PORT = 502

REQUEST = bytes.fromhex("00040105")
FRAME_BYTES = 12
SENSOR_COUNT = 5
FINGERS = ["엄지", "검지", "중지", "약지", "소지"]


def recv_packet(sock, buf):
    """Length(2 B, 자기 포함) 프레이밍으로 한 패킷 수신."""
    while len(buf) < 2:
        chunk = sock.recv(4096)
        if not chunk:
            raise ConnectionError("그리퍼가 연결을 종료했습니다.")
        buf += chunk
    length = struct.unpack(">H", buf[:2])[0]
    while len(buf) < length:
        chunk = sock.recv(4096)
        if not chunk:
            raise ConnectionError("그리퍼가 연결을 종료했습니다.")
        buf += chunk
    packet = bytes(buf[:length])
    del buf[:length]
    return packet


def read_sensors(sock, buf):
    """센서 5 개의 (x[mm], y[mm], fx/fy/fz[N], status) 리스트."""
    sock.sendall(REQUEST)
    packet = recv_packet(sock, buf)
    data = packet[3:]                      # Length 2 B + CMD 1 B 제거

    need = FRAME_BYTES * SENSOR_COUNT
    if len(data) < need:
        raise ValueError(
            f"센서 블록이 짧습니다: {len(data)} B < {need} B. "
            "Sensor 펌웨어와 옵션코드를 확인하세요 (00 03 08)."
        )

    out = []
    for i in range(SENSOR_COUNT):
        frame = data[i * FRAME_BYTES:(i + 1) * FRAME_BYTES]
        x, y, fx, fy, fz, status = struct.unpack(">5hH", frame)
        out.append((
            x / 100.0,
            y / 100.0,
            fx / 1000.0,
            fy / 1000.0,
            fz / 1000.0,
            status,
        ))
    return out


def main():
    parser = argparse.ArgumentParser(description="센서 1~5 접촉 위치와 수직력")
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--hz", type=float, default=20.0)
    parser.add_argument("--csv", metavar="PATH", help="CSV 로 기록")
    parser.add_argument("--scroll", action="store_true",
                        help="한 프레임씩 아래로 계속 출력")
    parser.add_argument("--left", action="store_true")
    parser.add_argument("--status", action="store_true",
                        help="status 비트필드(hex)도 출력")
    parser.add_argument("--ip")
    args = parser.parse_args()

    ip = args.ip or (LEFT_IP if args.left else DEFAULT_IP)
    period = 1.0 / args.hz if args.hz > 0 else 0.0

    writer = None
    csv_file = None
    if args.csv:
        csv_file = open(args.csv, "w", newline="", encoding="utf-8")
        writer = csv.writer(csv_file)
        header = ["t"]
        for i in range(1, SENSOR_COUNT + 1):
            header += [
                f"s{i}_x_mm", f"s{i}_y_mm",
                f"s{i}_fx_N", f"s{i}_fy_N", f"s{i}_fz_N",
            ]
            if args.status:
                header += [f"s{i}_status"]
        writer.writerow(header)

    sock = socket.create_connection((ip, PORT), timeout=1.0)
    sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    buf = bytearray()
    print(f"연결 {ip}:{PORT}")
    if args.status:
        header = "  ".join(
            f"S{i + 1}({FINGERS[i]}): {'x':>7} {'y':>7} "
            f"{'fx':>7} {'fy':>7} {'fz':>7} {'st':>6}"
            for i in range(SENSOR_COUNT))
    else:
        header = "  ".join(f"S{i + 1}({FINGERS[i]}): {'x':>7} {'y':>7} "
                           f"{'fx':>7} {'fy':>7} {'fz':>7}"
                           for i in range(SENSOR_COUNT))
    print(header)

    start = time.monotonic()
    try:
        while True:
            sensors = read_sensors(sock, buf)
            elapsed = time.monotonic() - start

            cells = []
            for i, (x, y, fx, fy, fz, status) in enumerate(sensors):
                if args.status:
                    cells.append(
                        f"S{i + 1}: {x:7.2f} {y:7.2f} "
                        f"{fx:7.3f} {fy:7.3f} {fz:7.3f} 0x{status:04X}")
                else:
                    cells.append(
                        f"S{i + 1}: {x:7.2f} {y:7.2f} "
                        f"{fx:7.3f} {fy:7.3f} {fz:7.3f}")
            line = "  ".join(cells)

            if args.scroll or args.once:
                print(f"t={elapsed:7.2f}s  {line}")
            else:
                sys.stdout.write(f"\rt={elapsed:7.2f}s  {line}")
                sys.stdout.flush()

            if writer:
                row = [f"{elapsed:.4f}"]
                for x, y, fx, fy, fz, status in sensors:
                    row += [
                        f"{x:.2f}", f"{y:.2f}",
                        f"{fx:.3f}", f"{fy:.3f}", f"{fz:.3f}",
                    ]
                    if args.status:
                        row += [f"0x{status:04X}"]
                writer.writerow(row)

            if args.once:
                break
            if period:
                time.sleep(period)
    except KeyboardInterrupt:
        print()
    except (ConnectionError, ValueError) as exc:
        print(f"\n오류: {exc}", file=sys.stderr)
        return 1
    finally:
        sock.close()
        if csv_file:
            csv_file.close()
            print(f"CSV 저장: {args.csv}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
