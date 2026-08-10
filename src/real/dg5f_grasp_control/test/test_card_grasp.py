import unittest
from unittest.mock import patch

import numpy as np

from dg5f_grasp_control.config import RuntimeConfig
from dg5f_grasp_control.grasp_controller import GraspController
from dg5f_grasp_control.hand_model import FINGER_JOINT_INDEX
from dg5f_grasp_control.poses import HAND_CARD_PRE_GRASP_POSE


def fake_tip_position(q, finger):
    indices = np.asarray(FINGER_JOINT_INDEX[finger], dtype=int)
    return np.asarray(q, dtype=np.float64)[indices[:3]].copy()


def fake_tip_jacobian(q, finger, eps=1e-6):
    del q, finger, eps
    return np.column_stack((np.eye(3), np.zeros(3)))


class CardGraspTest(unittest.TestCase):
    def test_card_pre_grasp_pose_matches_requested_joint_angles(self):
        np.testing.assert_allclose(
            HAND_CARD_PRE_GRASP_POSE,
            [
                0.0, np.pi / 2, -0.3536, -0.7756,
                0.0, 0.3751, 0.8587, 0.5213,
                *([0.0] * 12),
            ],
        )

    @patch("dg5f_grasp_control.grasp_policy.tip_jacobian", fake_tip_jacobian)
    @patch("dg5f_grasp_control.grasp_policy.tip_position", fake_tip_position)
    @patch("dg5f_grasp_control.grasp_controller.tip_jacobian", fake_tip_jacobian)
    @patch("dg5f_grasp_control.grasp_controller.tip_position", fake_tip_position)
    def test_timeout_log_reports_stall_diagnostics(self):
        logs = []
        controller = GraspController(
            RuntimeConfig(card_pinch_timeout_sec=0.05),
            log=logs.append,
        )
        q = np.zeros(20, dtype=np.float64)
        q[FINGER_JOINT_INDEX[2][0]] = 0.05
        qdot = np.zeros(20, dtype=np.float64)
        controller.step(q, qdot, now=0.5)
        controller.apply_pose_type(4, now=0.6)
        controller.apply_grasp_type(7, now=1.0)
        controller.step(q, qdot, now=1.0)

        controller.step(q, qdot, now=1.051)

        self.assertEqual(controller.state, "PRE_GRASP_POSE")
        timeout_log = next(message for message in logs if "timeout" in message)
        self.assertIn("stall_motion_mm=", timeout_log)
        self.assertIn("threshold_mm=", timeout_log)

    @patch("dg5f_grasp_control.grasp_policy.tip_jacobian", fake_tip_jacobian)
    @patch("dg5f_grasp_control.grasp_policy.tip_position", fake_tip_position)
    @patch("dg5f_grasp_control.grasp_controller.tip_jacobian", fake_tip_jacobian)
    @patch("dg5f_grasp_control.grasp_controller.tip_position", fake_tip_position)
    def test_card_start_applies_constant_world_down_force(self):
        controller = GraspController(RuntimeConfig(), log=None)
        q = np.zeros(20, dtype=np.float64)
        q[FINGER_JOINT_INDEX[2][1]] = 0.05
        controller.step(q, np.zeros(20), now=0.5)
        controller.apply_pose_type(4, now=0.6)
        controller.set_rotation_hand_to_world(
            np.array(
                [
                    [0.0, 0.0, 1.0],
                    [0.0, 1.0, 0.0],
                    [-1.0, 0.0, 0.0],
                ]
            )
        )
        controller.apply_grasp_type(7, now=1.0)

        output = controller.step(q, np.zeros(20), now=1.0)

        self.assertEqual(controller.card_phase, "card_pinch")
        np.testing.assert_allclose(output.total_forces[1], [0.0, 0.0, 0.0])
        np.testing.assert_allclose(output.total_forces[2], [3.0, -4.0, 0.0])

    @patch("dg5f_grasp_control.grasp_policy.tip_jacobian", fake_tip_jacobian)
    @patch("dg5f_grasp_control.grasp_policy.tip_position", fake_tip_position)
    @patch("dg5f_grasp_control.grasp_controller.tip_jacobian", fake_tip_jacobian)
    @patch("dg5f_grasp_control.grasp_controller.tip_position", fake_tip_position)
    def test_pinch_direction_is_projected_onto_world_xy(self):
        controller = GraspController(RuntimeConfig(), log=None)
        q = np.zeros(20, dtype=np.float64)
        q[FINGER_JOINT_INDEX[1][2]] = 0.04
        q[FINGER_JOINT_INDEX[2][0]] = 0.05
        q[FINGER_JOINT_INDEX[2][3]] = HAND_CARD_PRE_GRASP_POSE[
            FINGER_JOINT_INDEX[2][3]
        ]
        qdot = np.zeros(20, dtype=np.float64)
        controller.step(q, qdot, now=0.5)
        controller.apply_pose_type(4, now=0.6)
        controller.apply_grasp_type(7, now=1.0)
        pinch = controller.step(q, qdot, now=1.0)

        np.testing.assert_allclose(pinch.total_forces[1], [0.0, 0.0, 0.0])
        np.testing.assert_allclose(pinch.total_forces[2], [-4.0, 0.0, -3.0])
        np.testing.assert_allclose(
            pinch.inactive_pd[FINGER_JOINT_INDEX[2]],
            np.zeros(4),
        )

    @patch("dg5f_grasp_control.grasp_policy.tip_jacobian", fake_tip_jacobian)
    @patch("dg5f_grasp_control.grasp_policy.tip_position", fake_tip_position)
    @patch("dg5f_grasp_control.grasp_controller.tip_jacobian", fake_tip_jacobian)
    @patch("dg5f_grasp_control.grasp_controller.tip_position", fake_tip_position)
    def test_pinch_down_force_is_constant_when_index_moves(self):
        controller = GraspController(RuntimeConfig(), log=None)
        q = np.zeros(20, dtype=np.float64)
        q[FINGER_JOINT_INDEX[2][0]] = 0.05
        qdot = np.zeros(20, dtype=np.float64)
        controller.step(q, qdot, now=0.5)
        controller.apply_pose_type(4, now=0.6)
        controller.apply_grasp_type(7, now=1.0)
        controller.step(q, qdot, now=1.0)

        lifted = q.copy()
        lifted[FINGER_JOINT_INDEX[2][2]] = 0.001
        pinch = controller.step(lifted, qdot, now=1.01)

        np.testing.assert_allclose(pinch.total_forces[1], [0.0, 0.0, 0.0])
        np.testing.assert_allclose(pinch.total_forces[2], [-4.0, 0.0, -3.0])

    @patch("dg5f_grasp_control.grasp_policy.tip_jacobian", fake_tip_jacobian)
    @patch("dg5f_grasp_control.grasp_policy.tip_position", fake_tip_position)
    @patch("dg5f_grasp_control.grasp_controller.tip_jacobian", fake_tip_jacobian)
    @patch("dg5f_grasp_control.grasp_controller.tip_position", fake_tip_position)
    def test_card_sequence_flexes_index_tip_after_pinch(self):
        controller = GraspController(RuntimeConfig(), log=None)
        q = HAND_CARD_PRE_GRASP_POSE.copy()
        index_joints = np.asarray(FINGER_JOINT_INDEX[2], dtype=int)
        q[FINGER_JOINT_INDEX[3][0]] = 0.08
        qdot = np.zeros(20, dtype=np.float64)
        controller.step(q, qdot, now=0.5)

        controller.apply_grasp_type(7, now=0.6)
        self.assertEqual(controller.state, "NORMAL_POSE")

        controller.apply_pose_type(4, now=0.7)
        controller.apply_grasp_type(7, now=1.0)
        self.assertEqual(controller.card_phase, "card_pinch")
        pinch = controller.step(q, qdot, now=1.0)
        np.testing.assert_allclose(pinch.total_forces[1], [0.0, 0.0, 0.0])
        self.assertAlmostEqual(pinch.total_forces[2][2], -3.0)

        waiting = controller.step(q, qdot, now=1.101)
        self.assertEqual(controller.card_phase, "card_post_pinch_wait")
        np.testing.assert_allclose(waiting.total_forces[1], [0.0, 0.0, 0.0])
        self.assertAlmostEqual(np.linalg.norm(waiting.total_forces[2][:2]), 4.0)
        self.assertAlmostEqual(waiting.total_forces[2][2], -3.0)
        index_tip_joint = index_joints[-1]
        np.testing.assert_allclose(
            controller.card_index_flex_hold_target,
            q[index_joints[:-1]],
        )

        holding = controller.step(q, qdot, now=3.102)
        self.assertEqual(controller.card_phase, "card_index_tip_flex")
        self.assertGreater(holding.grasp_tau[index_tip_joint], 0.0)
        self.assertGreater(holding.err[index_tip_joint], 0.0)

        at_target = q.copy()
        at_target[index_tip_joint] = np.deg2rad(80.0)
        at_target[index_joints[0]] += 0.01
        still_holding = controller.step(at_target, qdot, now=3.2)
        self.assertEqual(controller.card_phase, "card_index_tip_flex")
        self.assertAlmostEqual(
            np.linalg.norm(still_holding.total_forces[2][:2]),
            4.0,
        )
        self.assertAlmostEqual(still_holding.total_forces[2][2], -3.0)
        self.assertAlmostEqual(still_holding.total_forces[1][2], -3.0)
        self.assertAlmostEqual(still_holding.err[index_joints[0]], -0.01)
        self.assertAlmostEqual(still_holding.grasp_tau[index_tip_joint], 0.0)
        self.assertEqual(controller.use_fingers, [1, 2])
        self.assertEqual(controller.active_finger_count, 7)

        returning = controller.step(at_target, qdot, now=3.301)
        self.assertEqual(controller.card_phase, "card_index_tip_return")
        self.assertLess(returning.grasp_tau[index_tip_joint], 0.0)

        returned = at_target.copy()
        returned[index_tip_joint] = 0.0
        final = controller.step(returned, qdot, now=3.4)
        self.assertEqual(final.state, "GROPED_GRASP")
        self.assertEqual(controller.card_phase, "idle")
        self.assertEqual(controller.use_fingers, [1, 2])
        self.assertEqual(controller.active_finger_count, 1)


if __name__ == "__main__":
    unittest.main()
