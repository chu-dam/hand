#!/usr/bin/env python3

import json
import os
import signal
import subprocess
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from urllib.parse import urlparse


ROOT_DIR = Path(__file__).resolve().parent.parent
LAUNCH_FILES = {
    "left": "grasp_with_effort.launch.py",
    "right": "grasp_with_effort_right.launch.py",
}


class ControllerManager:
    def __init__(self):
        self.processes = {side: None for side in LAUNCH_FILES}
        self.lock = threading.Lock()

    def status(self):
        with self.lock:
            return {
                side: {
                    "running": process is not None and process.poll() is None,
                    "pid": process.pid if process is not None and process.poll() is None else None,
                }
                for side, process in self.processes.items()
            }

    @staticmethod
    def _stop_process(process):
        if process is None or process.poll() is not None:
            return
        os.killpg(process.pid, signal.SIGINT)
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()

    def start(self, side):
        if side not in LAUNCH_FILES:
            raise ValueError("hand side must be left or right")
        with self.lock:
            current = self.processes[side]
            if current is not None and current.poll() is None:
                return

            for other_side, process in self.processes.items():
                if other_side != side:
                    self._stop_process(process)
                    self.processes[other_side] = None

            self.processes[side] = subprocess.Popen(
                ["ros2", "launch", "dg5f_grasp_control", LAUNCH_FILES[side]],
                cwd=ROOT_DIR,
                start_new_session=True,
            )

    def stop(self, side):
        if side not in LAUNCH_FILES:
            raise ValueError("hand side must be left or right")
        with self.lock:
            self._stop_process(self.processes[side])
            self.processes[side] = None

    def stop_all(self):
        with self.lock:
            for side, process in self.processes.items():
                self._stop_process(process)
                self.processes[side] = None


class ControllerHandler(BaseHTTPRequestHandler):
    manager = None

    def _reply(self, status, payload):
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if urlparse(self.path).path != "/api/controllers":
            self._reply(404, {"error": "not found"})
            return
        self._reply(200, self.manager.status())

    def do_POST(self):
        parts = urlparse(self.path).path.strip("/").split("/")
        if len(parts) != 4 or parts[:2] != ["api", "controllers"]:
            self._reply(404, {"error": "not found"})
            return

        side, action = parts[2], parts[3]
        try:
            if action == "start":
                self.manager.start(side)
            elif action == "stop":
                self.manager.stop(side)
            else:
                self._reply(404, {"error": "not found"})
                return
        except (OSError, ValueError) as error:
            self._reply(400, {"error": str(error)})
            return
        self._reply(200, self.manager.status())

    def log_message(self, _format, *_args):
        return


def main():
    manager = ControllerManager()
    ControllerHandler.manager = manager
    server = HTTPServer(("127.0.0.1", 8081), ControllerHandler)

    def stop_server(_signum, _frame):
        raise KeyboardInterrupt

    signal.signal(signal.SIGINT, stop_server)
    signal.signal(signal.SIGTERM, stop_server)
    print("[DG5F] controller API ready · 127.0.0.1:8081", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        manager.stop_all()


if __name__ == "__main__":
    main()
