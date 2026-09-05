import unittest
from types import SimpleNamespace

import numpy as np

from dg5f_grasp_control.config import RuntimeConfig
from dg5f_grasp_control.grasp_controller import GraspController
from dg5f_grasp_control.grasp_real_node import GraspRealRunner
from dg5f_grasp_control.hand_model import FINGER_JOINT_INDEX
from dg5f_grasp_control.poses import (
    LEFT_POSE_TYPE_TARGETS,
    RIGHT_HAND_BLIND_GRASP_INITIAL_POSE,
    RIGHT_HAND_BLIND_GRASP_PRE_ROTATION_POSE,
    RIGHT_HAND_CONTINUOUS_ROTATION_POSE,
    RIGHT_HAND_PRE_ROTATION_POSE,
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
        self.assertNotIn(5, LEFT_POSE_TYPE_TARGETS)
        self.assertIs(
            RIGHT_POSE_TYPE_TARGETS[5],
            RIGHT_HAND_PRE_ROTATION_POSE,
        )
        self.assertIs(
            RIGHT_POSE_TYPE_TARGETS[6],
            RIGHT_HAND_BLIND_GRASP_PRE_ROTATION_POSE,
        )
        self.assertEqual(RIGHT_HAND_PRE_ROTATION_POSE.shape, (20,))
        self.assertTrue(np.all(np.isfinite(RIGHT_HAND_PRE_ROTATION_POSE)))
        np.testing.assert_allclose(
            RIGHT_HAND_BLIND_GRASP_PRE_ROTATION_POSE,
            [
                0.3533, -1.2919, 0.0707, 0.8334,
                -0.3433, 0.6871, 0.5681, 0.4410,
                0.0562, 0.4845, 0.6138, 0.5281,
                0.3669, 0.7437, 0.4677, 0.5086,
                0.8076, 1.3137, 0.7222, 0.5203,
            ],
        )
        np.testing.assert_allclose(
            RIGHT_HAND_BLIND_GRASP_INITIAL_POSE,
            [
                0.3533, -1.2903, 0.0696, 0.8306,
                -0.4128, 0.6676, 0.5573, 0.7308,
                0.0559, 0.4929, 0.6075, 0.5377,
                0.3557, 0.7126, 0.5145, 0.8327,
                0.6713, 1.3387, 0.9833, 0.8310,
            ],
        )
        self.assertEqual(RIGHT_HAND_CONTINUOUS_ROTATION_POSE.shape, (20,))
        self.assertTrue(np.all(np.isfinite(RIGHT_HAND_CONTINUOUS_ROTATION_POSE)))
        np.testing.assert_allclose(
            RIGHT_HAND_PRE_ROTATION_POSE,
            [
                0.7384, -1.0917, 0.2936, 0.8437,
                -0.0799, 0.7665, 0.5812, 0.2939,
                0.1504, 0.5627, 0.7418, 0.2431,
                0.2705, 0.7821, 0.6339, 0.2834,
                0.7327, 0.9755, 1.0877, 0.0398,
            ],
        )

    def test_invalid_hand_side_is_rejected(self):
        with self.assertRaises(ValueError):
            get_pose_type_targets("invalid")

    def test_right_pre_rotation_command_is_accepted(self):
        runner = GraspRealRunner.__new__(GraspRealRunner)
        runner.teaching_mode = False
        runner.pending_teaching_mode = None
        runner.pending_pose_type = None
        runner.controller = SimpleNamespace(pose_type_targets=RIGHT_POSE_TYPE_TARGETS)

        runner.pose_type_cb(SimpleNamespace(data=5))

        self.assertEqual(runner.pending_pose_type, 5)

    def test_pre_rotation_uses_its_own_pose_pd_gains(self):
        controller = GraspController(
            RuntimeConfig(
                hand_side="right",
                pose_kp=0.10,
                pose_kd=0.0,
                pose_pd_limit=1.0,
                pre_rotation_pose_kp=0.20,
                pre_rotation_pose_kd=0.0,
                pre_rotation_pose_pd_limit=1.0,
                pre_rotation_pinky_j1_kp=0.40,
                pre_rotation_pinky_j1_kd=0.0,
                pre_rotation_pinky_j1_tau_limit=2.0,
            ),
            log=None,
        )
        controller.apply_pose_type(5, now=1.0)
        target = RIGHT_HAND_PRE_ROTATION_POSE

        output = controller.step(
            target - 0.1,
            np.zeros(20),
            now=1.1,
        )

        expected = np.full(20, 0.02)
        expected[FINGER_JOINT_INDEX[5][0]] = 0.04
        np.testing.assert_allclose(output.tau, expected, atol=1e-12)

    def test_blind_grasp_pre_rotation_uses_its_own_pose_pd_gains(self):
        controller = GraspController(
            RuntimeConfig(
                hand_side="right",
                blind_grasp_pre_rotation_pose_kp=3.0,
                blind_grasp_pre_rotation_pose_kd=5.0,
                pre_rotation_pose_pd_limit=10.0,
            ),
            log=None,
        )
        controller.apply_pose_type(6, now=1.0)
        target = RIGHT_HAND_BLIND_GRASP_PRE_ROTATION_POSE

        output = controller.step(
            target - 0.1,
            np.full(20, 0.01),
            now=1.1,
        )

        np.testing.assert_allclose(output.tau, np.full(20, 0.25), atol=1e-12)

if __name__ == "__main__":
    unittest.main()
