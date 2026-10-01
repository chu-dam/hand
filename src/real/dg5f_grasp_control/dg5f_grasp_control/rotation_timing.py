import csv
from datetime import datetime

import numpy as np

from dg5f_grasp_control.hand_model import FINGER_JOINT_INDEX


def phase_metrics(phase, q, qdot, pd_target):
    if phase.startswith("blind_middle_"):
        fingers = (3,)
    elif phase.startswith("blind_index_ring_"):
        fingers = (2, 4)
    elif phase.startswith("blind_thumb_"):
        fingers = (1,)
    elif phase == "blind_pinky_release":
        fingers = (5,)
    elif phase in ("blind_pinky_regrasp", "blind_reverse_pinky_regrasp"):
        fingers = (3, 5)
    elif phase.startswith("blind_pose_"):
        fingers = (1, 2, 3, 4)
    else:
        fingers = range(1, 6)
    indices = np.concatenate([FINGER_JOINT_INDEX[finger] for finger in fingers])
    speed = float(np.max(np.abs(qdot[indices])))
    target = pd_target[indices]
    controlled = np.isfinite(target)
    error_deg = (
        float(np.rad2deg(np.max(np.abs(target[controlled] - q[indices][controlled]))))
        if np.any(controlled)
        else None
    )
    return error_deg, speed


class RotationPhaseTimer:
    def __init__(self, path):
        self.path = path
        self.phase = None
        self.started_at = None

    def record(self, phase, now, target_error_deg=None, joint_velocity_rad_s=None):
        if phase == self.phase:
            return
        if self.phase is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", newline="", encoding="utf-8") as stream:
                writer = csv.writer(stream)
                if stream.tell() == 0:
                    writer.writerow((
                        "ended_at", "phase", "duration_sec",
                        "max_target_error_deg", "max_joint_velocity_rad_s",
                    ))
                writer.writerow((
                    datetime.now().astimezone().isoformat(timespec="milliseconds"),
                    self.phase,
                    f"{now - self.started_at:.4f}",
                    "" if target_error_deg is None else f"{target_error_deg:.2f}",
                    "" if joint_velocity_rad_s is None else f"{joint_velocity_rad_s:.3f}",
                ))
        self.phase = phase
        self.started_at = now if phase is not None else None
