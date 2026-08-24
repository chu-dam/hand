import unittest

import numpy as np

from dg5f_grasp_control.config import RuntimeConfig
from dg5f_grasp_control.grasp_controller import GraspController
from dg5f_grasp_control.hand_model import FINGER_JOINT_INDEX
from dg5f_grasp_control.poses import RIGHT_HAND_CONTINUOUS_ROTATION_POSE


class ContinuousRotationPoseSequenceTest(unittest.TestCase):
    def make_controller(self):
        controller = GraspController(RuntimeConfig(hand_side="right"), log=None)
        controller.apply_pose_type(5, now=1.0)
        controller.sync_joint_state(controller.pose_type_targets[5])
        return controller

    def test_starts_only_from_right_pre_rotation(self):
        controller = self.make_controller()

        self.assertTrue(controller.start_continuous_rotation(now=2.0))
        self.assertEqual(
            controller.continuous_rotation_phase,
            "continuous_release_middle",
        )

        left = GraspController(RuntimeConfig(hand_side="left"), log=None)
        self.assertFalse(left.start_continuous_rotation(now=2.0))

    def test_supplied_pose_values(self):
        np.testing.assert_allclose(
            RIGHT_HAND_CONTINUOUS_ROTATION_POSE,
            [
                0.7660, -1.3130, 0.5590, 0.4587,
                -0.2552, 0.9154, 0.5842, 0.2882,
                -0.1162, 0.6428, 0.5979, 0.2505,
                0.0005, 0.6575, 0.5683, 0.2827,
                0.8777, 0.7625, 0.5538, 0.2868,
            ],
        )

    def test_release_joint_targets_move_open(self):
        controller = self.make_controller()
        pre_rotation = controller.pose_type_targets[5].copy()
        controller.start_continuous_rotation(now=2.0)

        middle_j1 = FINGER_JOINT_INDEX[3][0]
        middle_j2 = FINGER_JOINT_INDEX[3][1]
        middle_j3 = FINGER_JOINT_INDEX[3][2]
        self.assertAlmostEqual(
            controller.continuous_rotation_pose_target[middle_j1],
            RIGHT_HAND_CONTINUOUS_ROTATION_POSE[middle_j1],
        )
        self.assertAlmostEqual(
            controller.continuous_rotation_pose_target[middle_j2],
            pre_rotation[middle_j2] - np.deg2rad(15.0),
        )
        self.assertAlmostEqual(
            controller.continuous_rotation_pose_target[middle_j3],
            pre_rotation[middle_j3] + np.deg2rad(15.0),
        )

        controller.continuous_rotation_group_index = 1
        controller.continuous_rotation_pose_target = pre_rotation.copy()
        controller._start_continuous_release(now=2.5)
        for finger in (4, 2):
            joint_1 = FINGER_JOINT_INDEX[finger][0]
            joint_2 = FINGER_JOINT_INDEX[finger][1]
            joint_3 = FINGER_JOINT_INDEX[finger][2]
            joint_2_release_deg = 30.0 if finger == 4 else 20.0
            self.assertAlmostEqual(
                controller.continuous_rotation_pose_target[joint_1],
                RIGHT_HAND_CONTINUOUS_ROTATION_POSE[joint_1],
            )
            self.assertAlmostEqual(
                controller.continuous_rotation_pose_target[joint_2],
                pre_rotation[joint_2] - np.deg2rad(joint_2_release_deg),
            )
            self.assertAlmostEqual(
                controller.continuous_rotation_pose_target[joint_3],
                pre_rotation[joint_3] + np.deg2rad(20.0),
            )

        controller.continuous_rotation_group_index = 2
        controller.continuous_rotation_pose_target = pre_rotation.copy()
        controller._start_continuous_release(now=3.0)
        thumb_j2 = FINGER_JOINT_INDEX[1][1]
        thumb_j3 = FINGER_JOINT_INDEX[1][2]
        self.assertAlmostEqual(
            controller.continuous_rotation_pose_target[thumb_j2],
            pre_rotation[thumb_j2] - np.deg2rad(20.0),
        )
        self.assertAlmostEqual(
            controller.continuous_rotation_pose_target[thumb_j3],
            pre_rotation[thumb_j3] - np.deg2rad(15.0),
        )

        controller.continuous_rotation_group_index = 3
        before_pinky = RIGHT_HAND_CONTINUOUS_ROTATION_POSE.copy()
        controller.continuous_rotation_pose_target = before_pinky.copy()
        controller._start_continuous_release(now=4.0)
        pinky_j1 = FINGER_JOINT_INDEX[5][0]
        pinky_j2 = FINGER_JOINT_INDEX[5][1]
        self.assertAlmostEqual(
            controller.continuous_rotation_pose_target[pinky_j1],
            before_pinky[pinky_j1] - np.deg2rad(15.0),
        )
        self.assertAlmostEqual(
            controller.continuous_rotation_pose_target[pinky_j2],
            before_pinky[pinky_j2],
        )
        for finger in range(1, 5):
            indices = np.asarray(FINGER_JOINT_INDEX[finger], dtype=int)
            np.testing.assert_allclose(
                controller.continuous_rotation_pose_target[indices],
                pre_rotation[indices],
            )

    def test_sequence_moves_groups_then_returns_and_repeats(self):
        controller = self.make_controller()
        pre_rotation = controller.pose_type_targets[5].copy()
        controller.start_continuous_rotation(now=2.0)

        schedule = (
            (2.21, "continuous_move_middle"),
            (2.52, "continuous_release_ring_index"),
            (2.73, "continuous_move_ring_index"),
            (3.04, "continuous_release_thumb"),
            (3.25, "continuous_move_thumb"),
            (3.56, "continuous_release_pinky"),
            (3.77, "continuous_release_middle"),
        )
        for now, phase in schedule:
            controller._process_continuous_rotation(now)
            self.assertEqual(controller.continuous_rotation_phase, phase)

        pinky_j1 = FINGER_JOINT_INDEX[5][0]
        self.assertAlmostEqual(
            controller.continuous_rotation_pose_target[pinky_j1],
            pre_rotation[pinky_j1],
        )

    def test_each_move_uses_supplied_pose_for_that_group(self):
        controller = self.make_controller()
        controller.start_continuous_rotation(now=2.0)

        for group_index, fingers in enumerate(((3,), (4, 2), (1,))):
            controller.continuous_rotation_group_index = group_index
            controller._start_continuous_move(now=3.0 + group_index)
            for finger in fingers:
                indices = np.asarray(FINGER_JOINT_INDEX[finger], dtype=int)
                np.testing.assert_allclose(
                    controller.continuous_rotation_pose_target[indices],
                    RIGHT_HAND_CONTINUOUS_ROTATION_POSE[indices],
                )

if __name__ == "__main__":
    unittest.main()
