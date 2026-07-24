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
        self.assertEqual(
            len(message.relative_rotation_center_joint_error),
            20,
        )
        self.assertEqual(
            len(message.relative_rotation_center_position_torques),
            20,
        )
        self.assertEqual(
            len(message.relative_rotation_nullspace_torques),
            20,
        )
        self.assertEqual(len(message.inactive_collision_avoidance_offsets_rad), 5)
        self.assertEqual(len(message.inactive_collision_avoidance_active), 5)
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

    def test_debug_message_reports_active_relative_rotation(self):
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

        self.assertEqual(message.controller_phase, "rotating")
        self.assertEqual(message.relative_rotation_phase, "rotating")
        self.assertAlmostEqual(
            message.relative_rotation_target_rad,
            np.pi / 6.0,
        )
        self.assertAlmostEqual(message.relative_rotation_current_rad, 0.0)
        self.assertAlmostEqual(message.relative_rotation_error_rad, np.pi / 6.0)
        self.assertGreater(message.relative_rotation_command_moment, 0.0)
        self.assertEqual(
            message.relative_rotation_control_mode,
            "cartesian_thumb_pivot_jacobian_transpose",
        )
        np.testing.assert_allclose(
            [
                message.relative_rotation_pivot.x,
                message.relative_rotation_pivot.y,
                message.relative_rotation_pivot.z,
            ],
            output.fingertip_positions[1],
            rtol=0.0,
            atol=0.0,
        )
        self.assertEqual(message.relative_rotation_dls_sigma_min, 0.0)
        self.assertEqual(message.relative_rotation_dls_condition, 0.0)
        self.assertEqual(
            len(message.relative_rotation_center_joint_error),
            20,
        )
        self.assertEqual(
            len(message.relative_rotation_center_position_torques),
            20,
        )
        self.assertEqual(
            len(message.relative_rotation_nullspace_torques),
            20,
        )
        np.testing.assert_allclose(
            [
                message.relative_rotation_axis.x,
                message.relative_rotation_axis.y,
                message.relative_rotation_axis.z,
            ],
            [-1.0, 0.0, 0.0],
            rtol=0.0,
            atol=1e-12,
        )

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
        self.assertEqual(
            message.relative_translation_control_mode,
            "axis_centroid_dls_nullspace",
        )
        self.assertGreater(message.relative_translation_dls_sigma_min, 0.0)
        self.assertTrue(np.isfinite(message.relative_translation_dls_condition))
        self.assertEqual(len(message.relative_translation_joint_error), 20)
        self.assertEqual(len(message.relative_translation_position_torques), 20)
        self.assertEqual(
            len(message.relative_translation_nullspace_grasp_torques),
            20,
        )
        self.assertGreater(
            np.linalg.norm(message.relative_translation_position_torques),
            0.0,
        )
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
