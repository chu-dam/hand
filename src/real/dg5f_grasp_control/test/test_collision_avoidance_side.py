import unittest

import numpy as np

from dg5f_grasp_control.config import RuntimeConfig
from dg5f_grasp_control.grasp_controller import GraspController
from dg5f_grasp_control.hand_model import FINGER_JOINT_INDEX
from dg5f_grasp_control.poses import RIGHT_HAND_PRE_GRASP_POSE


class CollisionAvoidanceSideTest(unittest.TestCase):
    def test_right_ring_follows_middle_in_mirrored_positive_direction(self):
        controller = GraspController(RuntimeConfig(hand_side="right"), log=None)
        controller.apply_grasp_type(2, now=1.0)
        q = RIGHT_HAND_PRE_GRASP_POSE.copy()
        middle_joint = FINGER_JOINT_INDEX[3][0]
        ring_joint = FINGER_JOINT_INDEX[4][0]

        q[middle_joint] = 0.18
        controller.step(q, np.zeros(20), now=1.1)
        q[middle_joint] = 0.19
        qdot = np.zeros(20)
        qdot[middle_joint] = 0.2
        triggered = controller.step(q, qdot, now=1.2)
        trigger_target = triggered.inactive_pd_target[ring_joint]

        q[middle_joint] = 0.24
        output = controller.step(q, qdot, now=1.3)

        self.assertTrue(output.inactive_collision_avoidance_active[3])
        self.assertAlmostEqual(
            output.inactive_pd_target[ring_joint] - trigger_target,
            0.05,
        )


if __name__ == "__main__":
    unittest.main()
