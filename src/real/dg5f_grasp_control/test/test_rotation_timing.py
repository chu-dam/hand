import csv
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np

from dg5f_grasp_control.hand_model import FINGER_JOINT_INDEX
from dg5f_grasp_control.rotation_timing import RotationPhaseTimer, phase_metrics


class RotationPhaseTimerTest(unittest.TestCase):
    def test_records_each_completed_phase_and_appends(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "logs" / "rotation.csv"
            timer = RotationPhaseTimer(path)
            timer.record("blind_middle_release", 10.0)
            timer.record("blind_middle_release", 10.1)
            timer.record("blind_middle_regrasp", 10.18, 6.0, 0.07)
            timer.record(None, 10.46, None, 0.03)
            timer.record(None, 10.50)
            timer.record("blind_thumb_release", 11.0)
            timer.record(None, 11.12, 2.0, 0.04)

            with path.open(newline="", encoding="utf-8") as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(
                [
                    (
                        row["phase"], row["duration_sec"],
                        row["max_target_error_deg"], row["max_joint_velocity_rad_s"],
                    )
                    for row in rows
                ],
                [
                    ("blind_middle_release", "0.1800", "6.00", "0.070"),
                    ("blind_middle_regrasp", "0.2800", "", "0.030"),
                    ("blind_thumb_release", "0.1200", "2.00", "0.040"),
                ],
            )

    def test_metrics_use_only_fingers_in_completed_phase(self):
        q = np.zeros(20)
        qdot = np.zeros(20)
        target = np.full(20, np.nan)
        middle = FINGER_JOINT_INDEX[3][0]
        thumb = FINGER_JOINT_INDEX[1][0]
        target[middle] = np.deg2rad(6)
        target[thumb] = np.deg2rad(90)
        qdot[middle] = 0.07
        qdot[thumb] = 2.0

        error, speed = phase_metrics("blind_middle_release", q, qdot, target)
        self.assertAlmostEqual(error, 6.0)
        self.assertAlmostEqual(speed, 0.07)
        error, speed = phase_metrics(
            "blind_middle_regrasp", q, qdot, np.full(20, np.nan)
        )
        self.assertIsNone(error)
        self.assertAlmostEqual(speed, 0.07)


if __name__ == "__main__":
    unittest.main()
