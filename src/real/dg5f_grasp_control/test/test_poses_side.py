import unittest

import numpy as np

from dg5f_grasp_control.config import RuntimeConfig
from dg5f_grasp_control.grasp_controller import GraspController
from dg5f_grasp_control.poses import (
    LEFT_POSE_TYPE_TARGETS,
    RIGHT_POSE_TYPE_TARGETS,
    get_pose_type_targets,
)


class PoseSideTest(unittest.TestCase):
    def test_controller_selects_pose_table_for_hand_side(self):
        left = GraspController(RuntimeConfig(hand_side="left"), log=None)
        right = GraspController(RuntimeConfig(hand_side="right"), log=None)

        self.assertIs(left.pose_type_targets, LEFT_POSE_TYPE_TARGETS)
        self.assertIs(right.pose_type_targets, RIGHT_POSE_TYPE_TARGETS)
        self.assertFalse(np.shares_memory(
            LEFT_POSE_TYPE_TARGETS[2],
            RIGHT_POSE_TYPE_TARGETS[2],
        ))

    def test_invalid_hand_side_is_rejected(self):
        with self.assertRaises(ValueError):
            get_pose_type_targets("invalid")


if __name__ == "__main__":
    unittest.main()
