import unittest
from types import SimpleNamespace
import numpy as np

from dg5f_grasp_control.config import RuntimeConfig
from dg5f_grasp_control.grasp_real_node import GraspRealRunner
from dg5f_grasp_control.hand_model import HAND_JOINT_NAMES, JOINT_COUNT


class JointVelocityTest(unittest.TestCase):
    def test_uses_only_joint_state_velocity(self):
        runner = GraspRealRunner.__new__(GraspRealRunner)
        runner.cfg = RuntimeConfig(qdot_alpha=1.0)
        runner.hand_q = np.zeros(JOINT_COUNT)
        runner.hand_qdot = np.zeros(JOINT_COUNT)
        runner.got_state = False
        runner.last_joint_state_time = None

        names = list(reversed(HAND_JOINT_NAMES))
        velocity_by_name = {
            name: 0.01 * (index + 1)
            for index, name in enumerate(HAND_JOINT_NAMES)
        }
        first = SimpleNamespace(
            name=names,
            position=[0.0] * JOINT_COUNT,
            velocity=[velocity_by_name[name] for name in names],
        )
        second = SimpleNamespace(
            name=names,
            position=[0.03] * JOINT_COUNT,
            velocity=[],
        )

        runner.joint_cb(first)
        np.testing.assert_allclose(
            runner.hand_qdot,
            [velocity_by_name[name] for name in HAND_JOINT_NAMES],
        )
        runner.joint_cb(second)

        np.testing.assert_allclose(runner.hand_qdot, 0.0)

    def test_right_hand_uses_joint_state_velocity(self):
        runner = GraspRealRunner.__new__(GraspRealRunner)
        runner.cfg = RuntimeConfig(hand_side="right", qdot_alpha=1.0)
        runner.hand_q = np.zeros(JOINT_COUNT)
        runner.hand_qdot = np.zeros(JOINT_COUNT)
        runner.got_state = False
        runner.last_joint_state_time = None

        message = SimpleNamespace(
            name=list(HAND_JOINT_NAMES),
            position=[0.0] * JOINT_COUNT,
            velocity=[0.2] * JOINT_COUNT,
        )
        runner.joint_cb(message)

        np.testing.assert_allclose(runner.hand_qdot, 0.2)


if __name__ == "__main__":
    unittest.main()
