import unittest
from unittest.mock import patch

import numpy as np

from dg5f_grasp_control.config import RuntimeConfig
from dg5f_grasp_control.grasp_controller import (
    GraspController,
    _estimate_sphere_from_points,
)
from dg5f_grasp_control.hand_model import FINGER_JOINT_INDEX
from dg5f_grasp_control.poses import (
    RIGHT_HAND_BLIND_GRASP_INITIAL_POSE,
    RIGHT_HAND_BLIND_GRASP_REVERSE_ROTATION_POSE,
    RIGHT_HAND_CONTINUOUS_ROTATION_POSE,
)


class ContinuousRotationPoseSequenceTest(unittest.TestCase):
    def test_sphere_estimate_recovers_known_center(self):
        center = np.array([0.08, -0.01, 0.12])
        radius = 0.0455
        directions = np.array([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [-1.0, -1.0, -1.0],
        ])
        directions /= np.linalg.norm(directions, axis=1, keepdims=True)

        estimate = _estimate_sphere_from_points(
            center + radius * directions,
            radius,
        )

        self.assertIsNotNone(estimate)
        estimated_center, fit_error = estimate
        np.testing.assert_allclose(estimated_center, center, atol=1e-12)
        self.assertAlmostEqual(fit_error, 0.0, places=12)

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

    def test_blind_rotation_starts_five_finger_grasp(self):
        controller = GraspController(RuntimeConfig(hand_side="right"), log=None)
        controller.apply_pose_type(6, now=1.0)

        self.assertTrue(controller.start_continuous_rotation(now=2.0))
        self.assertEqual(controller.state, "GROPED_GRASP")
        self.assertEqual(controller.active_finger_count, 5)
        self.assertEqual(controller.use_fingers, [1, 2, 3, 4, 5])
        self.assertEqual(
            controller.continuous_rotation_phase,
            "blind_grasp_settle",
        )
        output = controller.step(
            controller.pose_type_targets[6],
            np.zeros(20),
            now=2.1,
        )
        self.assertTrue(np.all(np.isnan(output.inactive_pd_target)))
        np.testing.assert_allclose(output.inactive_pd, 0.0)

    def test_blind_release_sequence_ends_with_pinky_released(self):
        logs = []
        controller = GraspController(RuntimeConfig(hand_side="right"), log=logs.append)
        controller.apply_pose_type(6, now=1.0)
        initial = RIGHT_HAND_BLIND_GRASP_INITIAL_POSE.copy()
        controller.start_continuous_rotation(now=2.0)

        controller._process_continuous_rotation(2.501)
        middle = np.asarray(FINGER_JOINT_INDEX[3], dtype=int)
        target = controller.continuous_rotation_pose_target
        self.assertEqual(controller.use_fingers, [1, 2, 4, 5])
        self.assertEqual(controller.continuous_rotation_phase, "blind_middle_release")
        j1_delta = -0.8110 - initial[FINGER_JOINT_INDEX[2][0]]
        self.assertAlmostEqual(target[middle[0]], initial[middle[0]] + j1_delta)
        self.assertAlmostEqual(target[middle[1]], initial[middle[1]] - np.deg2rad(3.0))
        self.assertAlmostEqual(target[middle[2]], initial[middle[2]] - np.deg2rad(3.0))
        self.assertAlmostEqual(target[middle[3]], initial[middle[3]])

        controller._process_continuous_rotation(2.682)
        self.assertEqual(controller.use_fingers, [1, 2, 3, 4, 5])
        self.assertEqual(controller.continuous_rotation_phase, "blind_middle_regrasp")

        controller._process_continuous_rotation(2.963)
        index = np.asarray(FINGER_JOINT_INDEX[2], dtype=int)
        ring = np.asarray(FINGER_JOINT_INDEX[4], dtype=int)
        target = controller.continuous_rotation_pose_target
        self.assertEqual(controller.use_fingers, [1, 3, 5])
        self.assertEqual(controller.continuous_rotation_phase, "blind_index_ring_release")
        self.assertAlmostEqual(target[index[0]], -0.8110)
        np.testing.assert_allclose(target[index[1:]], initial[index[1:]])
        self.assertAlmostEqual(target[ring[0]], initial[ring[0]] + j1_delta)
        self.assertAlmostEqual(target[ring[1]], initial[ring[1]] - np.deg2rad(6.0))
        self.assertAlmostEqual(target[ring[2]], initial[ring[2]] - np.deg2rad(6.0))
        self.assertAlmostEqual(target[ring[3]], initial[ring[3]])

        controller._process_continuous_rotation(3.144)
        self.assertEqual(controller.use_fingers, [1, 2, 3, 4, 5])
        self.assertEqual(controller.continuous_rotation_phase, "blind_index_ring_regrasp")

        controller._process_continuous_rotation(3.425)
        thumb = np.asarray(FINGER_JOINT_INDEX[1], dtype=int)
        target = controller.continuous_rotation_pose_target
        self.assertEqual(controller.use_fingers, [2, 3, 4, 5])
        self.assertEqual(controller.continuous_rotation_phase, "blind_thumb_down")
        np.testing.assert_allclose(target[thumb], [0.2436, -1.5139, 0.0359, 0.8207])

        controller._process_continuous_rotation(3.606)
        self.assertEqual(controller.use_fingers, [1, 2, 3, 4, 5])
        self.assertEqual(controller.continuous_rotation_phase, "blind_thumb_regrasp")

        controller._process_continuous_rotation(3.887)
        pinky = np.asarray(FINGER_JOINT_INDEX[5], dtype=int)
        target = controller.continuous_rotation_pose_target
        self.assertEqual(controller.use_fingers, [1, 2, 3, 4])
        self.assertEqual(controller.continuous_rotation_phase, "blind_pinky_release")
        self.assertAlmostEqual(target[pinky[0]], initial[pinky[0]] - np.deg2rad(10.0))
        np.testing.assert_allclose(target[pinky[1:]], initial[pinky[1:]])

        controller._process_continuous_rotation(4.068)
        self.assertEqual(controller.use_fingers, [1, 2, 3, 4])
        self.assertEqual(controller.continuous_rotation_phase, "blind_pose_rotation")

        output = controller.step(controller.pose_type_targets[6], np.zeros(20), 4.069)
        controlled = np.arange(0, 16)
        np.testing.assert_allclose(
            output.inactive_pd_target[controlled],
            RIGHT_HAND_BLIND_GRASP_INITIAL_POSE[controlled],
        )
        np.testing.assert_allclose(output.inactive_pd_target[pinky], target[pinky])

        controller._process_continuous_rotation(4.349)
        self.assertEqual(controller.use_fingers, [1, 2, 4, 5])
        self.assertEqual(controller.continuous_rotation_phase, "blind_middle_release")
        self.assertTrue(
            any(
                line.startswith(
                    "[BLIND_ROTATION] four_finger_polygon_area="
                )
                and line.endswith(" mm^2")
                for line in logs
            )
        )

    def test_blind_direction_change_during_release_regrasp_then_reverses(self):
        controller = GraspController(RuntimeConfig(hand_side="right"), log=None)
        controller.apply_pose_type(6, now=1.0)
        controller.start_continuous_rotation(now=2.0)

        controller._process_continuous_rotation(2.501)
        controller.request_blind_direction_change()
        self.assertEqual(controller.continuous_rotation_phase, "blind_middle_release")

        controller._process_continuous_rotation(2.55)
        self.assertEqual(controller.continuous_rotation_phase, "blind_middle_regrasp")
        controller._process_continuous_rotation(3.051)
        self.assertEqual(controller.continuous_rotation_phase, "blind_pinky_release")
        self.assertEqual(controller.blind_rotation_direction, -1)
        self.assertTrue(controller.continuous_rotation_active)

        controller._process_continuous_rotation(3.552)
        controller._process_continuous_rotation(4.053)
        self.assertEqual(controller.continuous_rotation_phase, "blind_thumb_down")

        controller._process_continuous_rotation(4.554)
        controller._process_continuous_rotation(5.055)
        self.assertEqual(controller.continuous_rotation_phase, "blind_index_ring_release")

    def test_blind_direction_change_after_regrasp_repeats_same_group(self):
        controller = GraspController(RuntimeConfig(hand_side="right"), log=None)
        controller.apply_pose_type(6, now=1.0)
        controller.start_continuous_rotation(now=2.0)

        controller._process_continuous_rotation(2.501)
        controller._process_continuous_rotation(2.682)
        controller.request_blind_direction_change()
        controller._process_continuous_rotation(2.963)

        self.assertEqual(controller.continuous_rotation_phase, "blind_middle_release")
        self.assertEqual(controller.blind_rotation_direction, -1)

        controller._process_continuous_rotation(3.464)
        controller._process_continuous_rotation(3.965)
        self.assertEqual(controller.continuous_rotation_phase, "blind_pinky_release")

    def test_reverse_release_uses_rotation_j1_and_opens_j2_j3(self):
        controller = GraspController(RuntimeConfig(hand_side="right"), log=None)
        controller.blind_rotation_direction = -1
        initial = RIGHT_HAND_BLIND_GRASP_INITIAL_POSE

        for finger, release_deg in ((2, 6.0), (3, 3.0), (4, 6.0)):
            indices = np.asarray(FINGER_JOINT_INDEX[finger], dtype=int)
            target = controller._blind_release_target(finger)
            self.assertAlmostEqual(target[0], initial[indices[0]])
            np.testing.assert_allclose(
                target[1:3],
                initial[indices[1:3]] - np.deg2rad(release_deg),
            )

    def test_reverse_thumb_release_uses_supplied_target(self):
        controller = GraspController(RuntimeConfig(hand_side="right"), log=None)
        controller.blind_rotation_direction = -1
        thumb = np.asarray(FINGER_JOINT_INDEX[1], dtype=int)

        controller._start_blind_thumb_release(now=2.0)
        target = controller.continuous_rotation_pose_target[thumb]

        np.testing.assert_allclose(target, [0.7660, -1.1010, -0.2536, 0.6646])

    def test_reverse_pinky_release_uses_fifteen_degrees(self):
        controller = GraspController(RuntimeConfig(hand_side="right"), log=None)
        controller.blind_rotation_direction = -1
        pinky = np.asarray(FINGER_JOINT_INDEX[5], dtype=int)

        controller._start_blind_pinky_release(now=2.0)
        target = controller.continuous_rotation_pose_target[pinky]

        self.assertAlmostEqual(
            target[0],
            RIGHT_HAND_BLIND_GRASP_INITIAL_POSE[pinky[0]] - np.deg2rad(15.0),
        )
        np.testing.assert_allclose(
            target[1:],
            RIGHT_HAND_BLIND_GRASP_INITIAL_POSE[pinky[1:]],
        )

    def test_reverse_pose_rotation_uses_supplied_non_pinky_target(self):
        controller = GraspController(RuntimeConfig(hand_side="right"), log=None)
        controller.apply_pose_type(6, now=1.0)
        controller.start_continuous_rotation(now=2.0)
        controller.blind_rotation_direction = -1
        controller._start_blind_pinky_release(now=3.0)
        controller._process_continuous_rotation(3.501)

        output = controller.step(
            controller.pose_type_targets[6],
            np.zeros(20),
            now=3.502,
        )

        np.testing.assert_allclose(
            output.inactive_pd_target[:16],
            RIGHT_HAND_BLIND_GRASP_REVERSE_ROTATION_POSE[:16],
        )

    def test_low_sphere_x_uses_thumb_lift_once(self):
        controller = GraspController(RuntimeConfig(hand_side="right"), log=None)
        controller.apply_pose_type(6, now=1.0)
        controller.start_continuous_rotation(now=2.0)
        controller.blind_thumb_lift_pending = True
        controller._set_continuous_rotation_phase(
            "blind_index_ring_regrasp",
            3.0,
        )

        controller._process_continuous_rotation(3.281)

        thumb = np.asarray(FINGER_JOINT_INDEX[1], dtype=int)
        np.testing.assert_allclose(
            controller.continuous_rotation_pose_target[thumb],
            [0.2436, -1.5139, 0.0359, 0.8207],
        )
        self.assertTrue(controller.blind_thumb_lift_pending)
        self.assertEqual(controller.continuous_rotation_phase, "blind_thumb_down")

        controller._process_continuous_rotation(3.462)

        target = controller.continuous_rotation_pose_target[thumb]
        np.testing.assert_allclose(
            target,
            [0.2436, -1.5139, -0.0852, 0.9840],
        )
        self.assertFalse(controller.blind_thumb_lift_pending)
        self.assertEqual(controller.continuous_rotation_phase, "blind_thumb_release")

    def test_post_rotation_estimate_arms_lift_after_three_low_samples(self):
        controller = GraspController(RuntimeConfig(hand_side="right"), log=None)
        controller.apply_pose_type(6, now=1.0)
        controller.start_continuous_rotation(now=2.0)
        controller.use_fingers = [1, 2, 3, 4]
        controller.active_finger_count = 4
        controller._set_continuous_rotation_phase("blind_pose_rotation", 2.0)
        center = np.array([0.09, 0.0, 0.11])
        radius = controller.cfg.blind_sphere_effective_radius_m
        directions = np.array([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [-1.0, -1.0, -1.0],
        ])
        directions /= np.linalg.norm(directions, axis=1, keepdims=True)
        tips = center + radius * directions

        with patch(
            "dg5f_grasp_control.grasp_controller.tip_position",
            side_effect=lambda _q, finger: tips[finger - 1],
        ):
            for _ in range(3):
                controller._update_blind_sphere_geometry(2.3)

        self.assertTrue(controller.blind_sphere_estimate_valid)
        np.testing.assert_allclose(controller.blind_sphere_center, center, atol=1e-9)
        self.assertTrue(controller.blind_thumb_lift_pending)

    def test_out_of_range_sphere_returns_to_blind_pre_rotation_pose(self):
        controller = GraspController(RuntimeConfig(hand_side="right"), log=None)
        controller.apply_pose_type(6, now=1.0)
        controller.start_continuous_rotation(now=2.0)
        controller.use_fingers = [1, 2, 3, 4]
        controller.active_finger_count = 4
        controller._set_continuous_rotation_phase("blind_pose_rotation", 2.0)
        center = np.array([0.087, 0.0, 0.11])
        radius = controller.cfg.blind_sphere_effective_radius_m
        directions = np.array([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [-1.0, -1.0, -1.0],
        ])
        directions /= np.linalg.norm(directions, axis=1, keepdims=True)
        tips = center + radius * directions

        with patch(
            "dg5f_grasp_control.grasp_controller.tip_position",
            side_effect=lambda _q, finger: tips[finger - 1],
        ):
            stopped = controller._update_blind_sphere_geometry(2.3)

        self.assertTrue(stopped)
        self.assertFalse(controller.continuous_rotation_active)
        self.assertEqual(controller.state, "PRE_GRASP_POSE")
        self.assertEqual(controller.pose_type, 6)

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
