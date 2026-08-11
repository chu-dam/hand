import unittest
from unittest.mock import patch

from scripts.controller_api import ControllerManager


class FakeProcess:
    def __init__(self, pid):
        self.pid = pid
        self.returncode = None

    def poll(self):
        return self.returncode

    def wait(self, timeout=None):
        self.returncode = 0
        return 0


class ControllerManagerTest(unittest.TestCase):
    @patch("scripts.controller_api.os.killpg")
    @patch("scripts.controller_api.subprocess.Popen")
    def test_starting_other_hand_stops_current_hand(self, popen, killpg):
        left = FakeProcess(101)
        right = FakeProcess(202)
        popen.side_effect = [left, right]
        manager = ControllerManager()

        manager.start("left")
        manager.start("right")

        self.assertEqual(manager.status()["left"], {"running": False, "pid": None})
        self.assertEqual(manager.status()["right"], {"running": True, "pid": 202})
        killpg.assert_called_once()


if __name__ == "__main__":
    unittest.main()
