import unittest

import numpy as np
from builtin_interfaces.msg import Time

from dg5f_grasp_control.config import RuntimeConfig
from dg5f_grasp_control.grasp_controller import GraspController
from dg5f_grasp_control.grasp_policy import GraspPolicy
from dg5f_grasp_control.ros_debug import build_grasp_debug_message


FINGER_IDS = [1, 2, 3, 4, 5]


class GraspDebugTest(unittest.TestCase):
    def test_policy_force_decomposition_preserves_reference_torque(self):
        q = np.linspace(-0.15, 0.2, 20)
        policy = GraspPolicy([1, 2, 3, 4], RuntimeConfig())

        result = policy.calc_grasp_tau(
            q,
            rotation_enabled=True,
            rotation_center=np.array([0.02, 0.0, 0.1]),
            center_hold_target=np.array([0.01, 0.0, 0.1]),
            center_hold_enabled=True,
        )

        expected_tau = np.array([
            -0.305856319700,
            -0.003875673590,
            -0.197250620090,
            -0.096891325694,
            0.060455305690,
            0.000521910037,
            -0.001192314423,
            -0.000851189786,
            0.068698550241,
            0.003388712662,
            0.003653810188,
            0.002339934277,
            0.005788819987,
            0.005137482073,
            0.004827550571,
            0.002675409584,
            0.0,
            0.0,
            0.0,
            0.0,
        ])
        np.testing.assert_allclose(
            result.tau,
            expected_tau,
            rtol=0.0,
            atol=1e-11,
        )

        for finger in [1, 2, 3, 4]:
            component_sum = (
                result.grasp_forces[finger]
                + result.rotation_forces[finger]
                + result.center_hold_forces[finger]
                + result.collision_forces[finger]
            )
            np.testing.assert_allclose(
                result.total_forces[finger],
                component_sum,
                rtol=0.0,
                atol=1e-12,
            )

    def test_debug_message_uses_fixed_five_finger_layout(self):
        q = np.linspace(-0.05, 0.1, 20)
        controller = GraspController(RuntimeConfig(), log=None)
        controller.apply_grasp_type(4, now=1.0)
        output = controller.step(q, np.zeros(20), now=1.1)
        commanded_efforts = np.clip(output.tau, -7.5, 7.5)

        message = build_grasp_debug_message(
            controller=controller,
            q=q,
            output=output,
            controller_torques=output.tau,
            commanded_efforts=commanded_efforts,
            stamp=Time(sec=12, nanosec=34),
            frame_id="link_base",
        )

        self.assertEqual(list(message.finger_ids), FINGER_IDS)
        self.assertEqual(message.header.frame_id, "link_base")
        self.assertEqual(message.header.stamp.sec, 12)
        self.assertEqual(len(message.fingertip_positions), 5)
        self.assertEqual(len(message.alpha), 5)
        self.assertEqual(len(message.grasp_forces), 5)
        self.assertEqual(len(message.translation_forces), 5)
        self.assertEqual(len(message.rotation_forces), 5)
        self.assertEqual(len(message.center_hold_forces), 5)
        self.assertEqual(len(message.collision_forces), 5)
        self.assertEqual(len(message.total_forces), 5)
        self.assertEqual(len(message.translation_torques), 20)
        self.assertEqual(len(message.controller_torques), 20)
        self.assertEqual(len(message.commanded_efforts), 20)
        self.assertEqual(message.grasp_type, 4)
        self.assertEqual(message.pose_type, 1)
        self.assertFalse(message.teaching_mode)
        self.assertEqual(message.controller_state, "GROPED_GRASP")

        np.testing.assert_allclose(
            [
                message.total_forces[4].x,
                message.total_forces[4].y,
                message.total_forces[4].z,
            ],
            np.zeros(3),
        )

    def test_debug_message_reports_immediate_relative_rotation_ready_phase(self):
        q = np.linspace(-0.05, 0.1, 20)
        controller = GraspController(RuntimeConfig(), log=None)
        controller.apply_grasp_type(3, now=1.0)
        controller.step(q, np.zeros(20), now=1.0)
        self.assertTrue(
            controller.prepare_relative_rotation(np.pi / 6.0, now=1.0)
        )
        output = controller.step(q, np.zeros(20), now=1.1)

        message = build_grasp_debug_message(
            controller=controller,
            q=q,
            output=output,
            controller_torques=output.tau,
            commanded_efforts=output.tau,
            stamp=Time(),
            frame_id="link_base",
        )

        self.assertEqual(message.controller_phase, "rotation_ready")

    def test_debug_message_reports_active_translation_target_and_force(self):
        q = np.linspace(-0.05, 0.1, 20)
        controller = GraspController(RuntimeConfig(), log=None)
        controller.apply_grasp_type(3, now=1.0)
        initial = controller.step(q, np.zeros(20), now=1.0)
        delta = np.array([0.0, -0.004, 0.0], dtype=np.float64)
        self.assertTrue(
            controller.prepare_relative_translation(delta, now=1.1)
        )
        output = controller.step(q, np.zeros(20), now=1.2)

        message = build_grasp_debug_message(
            controller=controller,
            q=q,
            output=output,
            controller_torques=output.tau,
            commanded_efforts=output.tau,
            stamp=Time(),
            frame_id="link_base",
        )

        self.assertEqual(
            message.relative_translation_phase,
            "translating",
        )
        self.assertEqual(message.controller_phase, "translating")
        np.testing.assert_allclose(
            [
                message.relative_translation_target_centroid.x,
                message.relative_translation_target_centroid.y,
                message.relative_translation_target_centroid.z,
            ],
            initial.cg + delta,
            rtol=0.0,
            atol=1e-12,
        )
        self.assertGreater(
            np.linalg.norm(
                [
                    message.relative_translation_command_force.x,
                    message.relative_translation_command_force.y,
                    message.relative_translation_command_force.z,
                ]
            ),
            0.0,
        )
        self.assertEqual(len(message.translation_torques), 20)
        self.assertGreater(
            np.linalg.norm(message.translation_torques),
            0.0,
        )
        self.assertGreater(message.relative_translation_torque_target, 0.0)
        self.assertGreaterEqual(message.relative_translation_force_scale, 1.0)
        np.testing.assert_allclose(
            [
                message.relative_translation_error.x,
                message.relative_translation_error.y,
                message.relative_translation_error.z,
            ],
            delta,
            rtol=0.0,
            atol=1e-12,
        )
