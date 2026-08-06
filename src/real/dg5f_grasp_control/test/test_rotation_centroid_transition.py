import unittest
from dataclasses import replace

import numpy as np

from dg5f_grasp_control.config import RuntimeConfig
from dg5f_grasp_control.grasp_controller import GraspController
from dg5f_grasp_control.grasp_policy import GraspPolicy
from dg5f_grasp_control.hand_model import FINGER_JOINT_INDEX
from dg5f_grasp_control.kinematics import (
    tip_jacobian,
    tip_position,
)
from dg5f_grasp_control.poses import (
    HAND_COMPACT_PRE_GRASP_POSE,
    HAND_PRE_GRASP_POSE,
)


Q_FIXED = np.linspace(-0.12, 0.18, 20)
QDOT_ZERO = np.zeros(20, dtype=np.float64)
THUMB_DISTANCE_MODE = "thumb_distance_proportional"


def assert_vector_dict_allclose(test_case, actual, expected, *, atol=1e-12):
    test_case.assertEqual(set(actual), set(expected))
    for finger in actual:
        np.testing.assert_allclose(
            actual[finger],
            expected[finger],
            rtol=0.0,
            atol=atol,
        )


def grasp_resultant(output):
    return sum(
        output.grasp_forces.values(),
        np.zeros(3, dtype=np.float64),
    )


def grasp_moment_about_cg(output):
    return sum(
        (
            np.cross(
                output.fingertip_positions[finger] - output.cg,
                force,
            )
            for finger, force in output.grasp_forces.items()
        ),
        np.zeros(3, dtype=np.float64),
    )


def assert_output_is_finite(test_case, output):
    for value in (
        output.tau,
        output.err,
        output.grasp_tau,
        output.inactive_pd,
        output.cg,
        output.cv,
    ):
        test_case.assertTrue(np.all(np.isfinite(value)))

    test_case.assertTrue(all(np.isfinite(value) for value in output.alpha.values()))
    for vector_map in (
        output.fingertip_positions,
        output.grasp_forces,
        output.rotation_forces,
        output.center_hold_forces,
        output.collision_forces,
        output.total_forces,
    ):
        for value in vector_map.values():
            test_case.assertTrue(np.all(np.isfinite(value)))


class ThumbDistanceProportionalPolicyTest(unittest.TestCase):
    def test_default_mode_is_explicit_legacy_mode(self):
        cfg = RuntimeConfig(thumb_centroid_bias=0.35)
        policy = GraspPolicy([1, 2, 3], cfg)

        default = policy.calc_grasp_tau(Q_FIXED)
        explicit_legacy = policy.calc_grasp_tau(
            Q_FIXED,
            alpha_distribution_mode="legacy",
        )

        np.testing.assert_allclose(
            default.tau,
            explicit_legacy.tau,
            rtol=0.0,
            atol=1e-12,
        )
        np.testing.assert_allclose(
            default.cg,
            explicit_legacy.cg,
            rtol=0.0,
            atol=1e-12,
        )
        np.testing.assert_allclose(
            default.cv,
            explicit_legacy.cv,
            rtol=0.0,
            atol=1e-12,
        )
        self.assertEqual(default.alpha, explicit_legacy.alpha)
        assert_vector_dict_allclose(
            self,
            default.grasp_forces,
            explicit_legacy.grasp_forces,
        )

    def test_two_finger_grasps_use_midpoint_and_equal_alpha(self):
        cfg = RuntimeConfig(alpha1=2.75, thumb_centroid_bias=0.8)

        for fingers in ([1, 2], [1, 3], [2, 1], [3, 1]):
            with self.subTest(fingers=fingers):
                result = GraspPolicy(fingers, cfg).calc_grasp_tau(
                    Q_FIXED,
                    alpha_distribution_mode=THUMB_DISTANCE_MODE,
                )

                np.testing.assert_allclose(
                    result.cv,
                    result.cg,
                    rtol=0.0,
                    atol=1e-12,
                )
                self.assertEqual(set(result.alpha), set(fingers))
                for finger in fingers:
                    self.assertAlmostEqual(result.alpha[finger], cfg.alpha1)
                np.testing.assert_allclose(
                    grasp_resultant(result),
                    np.zeros(3),
                    rtol=0.0,
                    atol=1e-10,
                )

    def test_three_finger_alpha_is_thumb_referenced_and_distance_proportional(self):
        cfg = RuntimeConfig(alpha1=3.25, thumb_centroid_bias=0.7)
        policy = GraspPolicy([1, 2, 3], cfg)
        alpha, fhat, cg, cv, tip_positions = policy.calc_alpha_and_forces(
            Q_FIXED,
            alpha_distribution_mode=THUMB_DISTANCE_MODE,
        )
        result = policy.calc_grasp_tau(
            Q_FIXED,
            alpha_distribution_mode=THUMB_DISTANCE_MODE,
        )

        self.assertEqual(alpha, result.alpha)
        np.testing.assert_allclose(cg, result.cg, rtol=0.0, atol=1e-12)
        np.testing.assert_allclose(cv, result.cv, rtol=0.0, atol=1e-12)
        assert_vector_dict_allclose(
            self,
            tip_positions,
            result.fingertip_positions,
        )
        for finger in alpha:
            np.testing.assert_allclose(
                alpha[finger] * fhat[finger],
                result.grasp_forces[finger],
                rtol=0.0,
                atol=1e-12,
            )

        np.testing.assert_allclose(result.cv, result.cg, rtol=0.0, atol=1e-12)
        self.assertAlmostEqual(result.alpha[1], cfg.alpha1)

        distances = {
            finger: np.linalg.norm(result.cg - position)
            for finger, position in result.fingertip_positions.items()
        }
        for finger in (2, 3):
            expected = cfg.alpha1 * distances[finger] / distances[1]
            self.assertAlmostEqual(result.alpha[finger], expected, places=10)

        np.testing.assert_allclose(
            grasp_resultant(result),
            np.zeros(3),
            rtol=0.0,
            atol=1e-9,
        )
        np.testing.assert_allclose(
            grasp_moment_about_cg(result),
            np.zeros(3),
            rtol=0.0,
            atol=1e-9,
        )

    def test_three_finger_thumb_reference_does_not_depend_on_input_order(self):
        cfg = RuntimeConfig(alpha1=2.4)
        reference = GraspPolicy([1, 2, 3], cfg).calc_grasp_tau(
            Q_FIXED,
            alpha_distribution_mode=THUMB_DISTANCE_MODE,
        )

        for fingers in ([2, 1, 3], [3, 2, 1], [2, 3, 1]):
            with self.subTest(fingers=fingers):
                result = GraspPolicy(fingers, cfg).calc_grasp_tau(
                    Q_FIXED,
                    alpha_distribution_mode=THUMB_DISTANCE_MODE,
                )

                self.assertAlmostEqual(result.alpha[1], cfg.alpha1)
                np.testing.assert_allclose(
                    result.cg,
                    reference.cg,
                    rtol=0.0,
                    atol=1e-12,
                )
                np.testing.assert_allclose(
                    result.cv,
                    reference.cv,
                    rtol=0.0,
                    atol=1e-12,
                )
                for finger in (1, 2, 3):
                    self.assertAlmostEqual(
                        result.alpha[finger],
                        reference.alpha[finger],
                        places=10,
                    )
                np.testing.assert_allclose(
                    grasp_resultant(result),
                    np.zeros(3),
                    rtol=0.0,
                    atol=1e-9,
                )

    def test_four_and_five_finger_modes_are_centered_nonnegative_and_balanced(self):
        cfg = RuntimeConfig(alpha1=3.0, thumb_centroid_bias=0.9)

        for fingers in ([1, 2, 3, 4], [1, 2, 3, 4, 5]):
            with self.subTest(fingers=fingers):
                result = GraspPolicy(fingers, cfg).calc_grasp_tau(
                    Q_FIXED,
                    alpha_distribution_mode=THUMB_DISTANCE_MODE,
                )

                np.testing.assert_allclose(
                    result.cv,
                    result.cg,
                    rtol=0.0,
                    atol=1e-12,
                )
                self.assertAlmostEqual(result.alpha[1], cfg.alpha1)
                self.assertTrue(
                    all(
                        np.isfinite(value) and value >= 0.0
                        for value in result.alpha.values()
                    )
                )
                np.testing.assert_allclose(
                    grasp_resultant(result),
                    np.zeros(3),
                    rtol=0.0,
                    atol=1e-8,
                )
                np.testing.assert_allclose(
                    grasp_moment_about_cg(result),
                    np.zeros(3),
                    rtol=0.0,
                    atol=1e-8,
                )

    def test_alpha_ratio_limit_applies_to_non_thumb_fingers_only(self):
        cfg = RuntimeConfig(
            alpha1=3.0,
            rotation_force_balance_max_alpha_ratio=0.6,
        )
        result = GraspPolicy([1, 2, 3], cfg).calc_grasp_tau(
            Q_FIXED,
            alpha_distribution_mode=THUMB_DISTANCE_MODE,
        )

        self.assertAlmostEqual(result.alpha[1], cfg.alpha1)
        self.assertLessEqual(result.alpha[2], 0.6 * cfg.alpha1)
        self.assertLessEqual(result.alpha[3], 0.6 * cfg.alpha1)


class EnvelopingGraspTest(unittest.TestCase):
    def test_pinky_starts_with_joint3_while_other_fingers_start_with_joint2(self):
        controller = GraspController(RuntimeConfig(), log=None)
        controller.step(Q_FIXED, QDOT_ZERO, now=0.5)
        controller.apply_grasp_type(6, now=1.0)

        output = controller.step(Q_FIXED, QDOT_ZERO, now=1.0)
        active = set(np.flatnonzero(output.grasp_tau))

        self.assertEqual(
            active,
            {
                FINGER_JOINT_INDEX[2][1],
                FINGER_JOINT_INDEX[3][1],
                FINGER_JOINT_INDEX[4][1],
                FINGER_JOINT_INDEX[5][2],
            },
        )
        self.assertEqual(output.grasp_tau[FINGER_JOINT_INDEX[5][1]], 0.0)

    def test_thumb_joint3_starts_one_stage_before_joint4(self):
        cfg = RuntimeConfig(envelop_joint_delay=0.20)
        controller = GraspController(cfg, log=None)
        q = np.zeros(20, dtype=np.float64)
        controller.step(q, QDOT_ZERO, now=0.5)
        controller.apply_grasp_type(6, now=1.0)

        joint3_started = controller.step(q, QDOT_ZERO, now=1.20)
        thumb_joint3 = FINGER_JOINT_INDEX[1][2]
        thumb_joint4 = FINGER_JOINT_INDEX[1][3]
        self.assertNotEqual(joint3_started.grasp_tau[thumb_joint3], 0.0)
        self.assertEqual(joint3_started.grasp_tau[thumb_joint4], 0.0)

        joint4_started = controller.step(q, QDOT_ZERO, now=1.40)
        self.assertNotEqual(joint4_started.grasp_tau[thumb_joint4], 0.0)


class InactiveFingerPreGraspTest(unittest.TestCase):
    def test_unknown_grasp_type_8_is_rejected(self):
        controller = GraspController(RuntimeConfig(), log=None)
        with self.assertRaises(ValueError):
            controller.apply_grasp_type(8, now=1.0)

    def _assert_inactive_targets(self, grasp_type, expected_pose):
        controller = GraspController(RuntimeConfig(), log=None)
        if expected_pose is HAND_COMPACT_PRE_GRASP_POSE:
            controller.apply_pose_type(3, now=0.5)
        else:
            controller.apply_pose_type(2, now=0.5)
        controller.apply_grasp_type(grasp_type, now=1.0)
        output = controller.step(expected_pose, QDOT_ZERO, now=1.1)

        active = set(controller.use_fingers)
        for finger, indices in FINGER_JOINT_INDEX.items():
            indices = np.asarray(indices, dtype=int)
            if finger in active:
                self.assertTrue(np.all(np.isnan(output.inactive_pd_target[indices])))
            else:
                np.testing.assert_allclose(
                    output.inactive_pd_target[indices],
                    expected_pose[indices],
                    rtol=0.0,
                    atol=0.0,
                )

    def test_all_unused_finger_joints_hold_default_pre_grasp(self):
        for grasp_type in (1, 2, 3, 4):
            with self.subTest(grasp_type=grasp_type):
                self._assert_inactive_targets(grasp_type, HAND_PRE_GRASP_POSE)

    def test_unused_fingers_follow_selected_compact_pre_grasp(self):
        self._assert_inactive_targets(1, HAND_COMPACT_PRE_GRASP_POSE)

    def test_added_finger_joins_grasp_without_pd_preparation_delay(self):
        controller = GraspController(RuntimeConfig(), log=None)
        controller.apply_grasp_type(1, now=1.0)

        controller.apply_grasp_type(3, now=1.1)

        self.assertEqual(controller.active_finger_count, 3)
        self.assertEqual(controller.use_fingers, [1, 2, 3])
        self.assertIsNone(controller.deferred_finger_count)

    def test_two_finger_switch_keeps_three_finger_bridge_without_pd_delay(self):
        controller = GraspController(RuntimeConfig(), log=None)
        controller.apply_grasp_type(1, now=1.0)

        controller.apply_grasp_type(2, now=1.1)

        self.assertEqual(controller.active_finger_count, 3)
        self.assertEqual(controller.use_fingers, [1, 2, 3])
        self.assertEqual(controller.deferred_finger_count, 2)
        self.assertIsNotNone(controller.deferred_finger_count_at)

        output = controller.step(Q_FIXED, QDOT_ZERO, now=1.7)
        self.assertEqual(output.active_finger_count, 2)
        self.assertEqual(output.use_fingers, [1, 3])

    def test_inactive_ring_matches_middle_displacement_after_early_trigger(self):
        controller = GraspController(RuntimeConfig(), log=None)
        controller.apply_grasp_type(2, now=1.0)
        q = HAND_PRE_GRASP_POSE.copy()
        q[FINGER_JOINT_INDEX[3][0]] = -0.18
        controller.step(q, QDOT_ZERO, now=1.1)

        q[FINGER_JOINT_INDEX[3][0]] = -0.19
        qdot = QDOT_ZERO.copy()
        qdot[FINGER_JOINT_INDEX[3][0]] = -0.2
        trigger_output = controller.step(q, qdot, now=1.2)

        self.assertTrue(trigger_output.inactive_collision_avoidance_active[3])
        self.assertEqual(controller.inactive_collision_follow_source[3], 3)
        trigger_target = trigger_output.inactive_pd_target[
            FINGER_JOINT_INDEX[4][0]
        ]

        q[FINGER_JOINT_INDEX[3][0]] = -0.24
        output = controller.step(q, qdot, now=1.3)
        self.assertAlmostEqual(
            output.inactive_pd_target[FINGER_JOINT_INDEX[4][0]]
            - trigger_target,
            -0.05,
        )

    def test_large_joint_value_crossing_does_not_trigger_inactive_index(self):
        controller = GraspController(RuntimeConfig(), log=None)
        controller.apply_grasp_type(2, now=1.0)
        q = HAND_PRE_GRASP_POSE.copy()
        q[FINGER_JOINT_INDEX[3][0]] = -0.10
        controller.step(q, QDOT_ZERO, now=1.1)

        q[FINGER_JOINT_INDEX[3][0]] = 0.40
        output = controller.step(q, QDOT_ZERO, now=1.2)

        self.assertFalse(output.inactive_collision_avoidance_active[1])

    def test_ring_and_pinky_targets_start_and_propagate_together(self):
        controller = GraspController(RuntimeConfig(), log=None)
        controller.apply_grasp_type(2, now=1.0)
        q = HAND_PRE_GRASP_POSE.copy()
        q[FINGER_JOINT_INDEX[3][0]] = -0.18
        controller.step(q, QDOT_ZERO, now=1.1)
        q[FINGER_JOINT_INDEX[3][0]] = -0.19
        qdot = QDOT_ZERO.copy()
        qdot[FINGER_JOINT_INDEX[3][0]] = -0.2
        trigger_output = controller.step(q, qdot, now=1.2)

        self.assertTrue(trigger_output.inactive_collision_avoidance_active[3])
        self.assertTrue(trigger_output.inactive_collision_avoidance_active[4])
        self.assertEqual(controller.inactive_collision_follow_source[4], 4)
        ring_trigger_target = trigger_output.inactive_pd_target[
            FINGER_JOINT_INDEX[4][0]
        ]
        trigger_target = trigger_output.inactive_pd_target[
            FINGER_JOINT_INDEX[5][1]
        ]

        q[FINGER_JOINT_INDEX[3][0]] = -0.24
        output = controller.step(q, qdot, now=1.3)
        ring_delta = (
            output.inactive_pd_target[FINGER_JOINT_INDEX[4][0]]
            - ring_trigger_target
        )
        pinky_delta = (
            output.inactive_pd_target[FINGER_JOINT_INDEX[5][1]]
            - trigger_target
        )
        self.assertAlmostEqual(
            ring_delta,
            -0.05,
        )
        self.assertAlmostEqual(pinky_delta, ring_delta)

    def test_following_releases_before_follower_finishes_returning(self):
        controller = GraspController(RuntimeConfig(), log=None)
        controller.apply_grasp_type(2, now=1.0)
        q = HAND_PRE_GRASP_POSE.copy()
        q[FINGER_JOINT_INDEX[3][0]] = -0.18
        controller.step(q, QDOT_ZERO, now=1.1)
        q[FINGER_JOINT_INDEX[3][0]] = -0.205
        controller.step(q, QDOT_ZERO, now=1.2)

        q[FINGER_JOINT_INDEX[3][0]] = 0.0
        q[FINGER_JOINT_INDEX[4][0]] = -0.35
        output = controller.step(q, QDOT_ZERO, now=1.3)

        self.assertFalse(output.inactive_collision_avoidance_active[3])
        self.assertEqual(controller.inactive_collision_follow_source[3], 0)
        self.assertEqual(
            output.inactive_pd_target[FINGER_JOINT_INDEX[4][0]],
            HAND_PRE_GRASP_POSE[FINGER_JOINT_INDEX[4][0]],
        )


class GeneralGraspRotationPreparationTest(unittest.TestCase):
    def test_relative_rotation_waits_for_first_successful_balance_cycle(self):
        controller = GraspController(RuntimeConfig(), log=None)
        controller.apply_grasp_type(3, now=1.0)

        self.assertFalse(
            controller.prepare_relative_rotation(np.deg2rad(10.0), now=1.0)
        )
        controller.step(Q_FIXED, QDOT_ZERO, now=1.0)
        self.assertTrue(
            controller.prepare_relative_rotation(np.deg2rad(10.0), now=1.01)
        )

    def test_regular_grasp_controllers_always_use_centered_balanced_distribution(self):
        cfg = RuntimeConfig(alpha1=3.0, thumb_centroid_bias=0.75)

        for grasp_type in (1, 2, 3, 4, 5):
            with self.subTest(grasp_type=grasp_type):
                controller = GraspController(cfg, log=None)
                controller.apply_grasp_type(grasp_type, now=1.0)
                output = controller.step(Q_FIXED, QDOT_ZERO, now=1.0)

                np.testing.assert_allclose(
                    output.cv,
                    output.cg,
                    rtol=0.0,
                    atol=1e-12,
                )
                self.assertAlmostEqual(output.alpha[1], cfg.alpha1)
                self.assertTrue(all(value >= 0.0 for value in output.alpha.values()))
                np.testing.assert_allclose(
                    grasp_resultant(output),
                    np.zeros(3),
                    rtol=0.0,
                    atol=1e-8,
                )

    def test_relative_rotation_tracks_fixed_cartesian_targets_with_jacobian_transpose(self):
        angle = np.deg2rad(10.0)

        for grasp_type in (1, 2, 3, 4, 5):
            with self.subTest(grasp_type=grasp_type):
                controller = GraspController(RuntimeConfig(), log=None)
                controller.apply_grasp_type(grasp_type, now=2.0)
                controller.step(Q_FIXED, QDOT_ZERO, now=2.0)

                self.assertTrue(controller.prepare_relative_rotation(angle, now=2.0))
                self.assertEqual(controller.relative_rotation_phase, "rotating")
                self.assertAlmostEqual(controller.relative_rotation_target_rad, angle)

                output = controller.step(Q_FIXED, QDOT_ZERO, now=2.5)
                self.assertEqual(output.relative_rotation_phase, "rotating")
                self.assertEqual(
                    output.relative_rotation_control_mode,
                    "cartesian_thumb_pivot_jacobian_transpose",
                )
                np.testing.assert_allclose(
                    output.relative_rotation_pivot,
                    controller.relative_rotation_start_fingertips[1],
                    rtol=0.0,
                    atol=0.0,
                )
                np.testing.assert_allclose(
                    output.cv,
                    output.cg,
                    rtol=0.0,
                    atol=1e-12,
                )
                np.testing.assert_allclose(
                    grasp_resultant(output),
                    np.zeros(3),
                    rtol=0.0,
                    atol=1e-8,
                )
                axis = output.relative_rotation_axis
                axis_cross = np.array(
                    [
                        [0.0, -axis[2], axis[1]],
                        [axis[2], 0.0, -axis[0]],
                        [-axis[1], axis[0], 0.0],
                    ],
                    dtype=np.float64,
                )
                rotation = (
                    np.eye(3)
                    + np.sin(angle) * axis_cross
                    + (1.0 - np.cos(angle)) * (axis_cross @ axis_cross)
                )
                for finger, start_position in (
                    controller.relative_rotation_start_fingertips.items()
                ):
                    if finger == 1:
                        np.testing.assert_allclose(
                            output.rotation_forces[finger],
                            np.zeros(3),
                            rtol=0.0,
                            atol=0.0,
                        )
                        np.testing.assert_allclose(
                            output.total_forces[finger],
                            output.grasp_forces[finger],
                            rtol=0.0,
                            atol=1e-12,
                        )
                        continue
                    start_radius = (
                        start_position
                        - controller.relative_rotation_pivot
                    )
                    target = (
                        controller.relative_rotation_pivot
                        + rotation @ start_radius
                    )
                    error = target - output.fingertip_positions[finger]
                    error_norm = float(np.linalg.norm(error))
                    if (
                        error_norm
                        > controller.cfg.relative_rotation_position_error_limit_m
                    ):
                        error *= (
                            controller.cfg.relative_rotation_position_error_limit_m
                            / error_norm
                        )
                    expected_force = (
                        controller.cfg.relative_rotation_position_kp
                        / max(
                            np.linalg.norm(start_radius),
                            controller.cfg.relative_rotation_radius_min,
                        )
                    ) * error
                    force_norm = float(np.linalg.norm(expected_force))
                    if (
                        force_norm
                        > controller.cfg.relative_rotation_force_limit
                    ):
                        expected_force *= (
                            controller.cfg.relative_rotation_force_limit
                            / force_norm
                        )
                    np.testing.assert_allclose(
                        output.rotation_forces[finger],
                        expected_force,
                        rtol=0.0,
                        atol=1e-12,
                    )
                    np.testing.assert_allclose(
                        output.total_forces[finger],
                        output.grasp_forces[finger] + expected_force,
                        rtol=0.0,
                        atol=1e-12,
                    )

                rotation_moment = sum(
                    (
                        np.dot(
                            np.cross(
                                output.fingertip_positions[finger]
                                - controller.relative_rotation_pivot,
                                force,
                            ),
                            output.relative_rotation_axis,
                        )
                        for finger, force in output.rotation_forces.items()
                    ),
                    0.0,
                )
                self.assertGreater(rotation_moment, 0.0)
                self.assertAlmostEqual(
                    output.relative_rotation_command_moment,
                    rotation_moment,
                )
                self.assertLessEqual(
                    max(np.linalg.norm(f) for f in output.rotation_forces.values()),
                    controller.cfg.relative_rotation_force_limit + 1e-12,
                )
                np.testing.assert_allclose(
                    output.grasp_tau,
                    controller.policy.calc_tau_from_total_forces(
                        controller.hand_q,
                        output.total_forces,
                    ),
                    rtol=0.0,
                    atol=1e-12,
                )

    def test_new_relative_command_replaces_instead_of_accumulating_angle(self):
        controller = GraspController(RuntimeConfig(), log=None)
        controller.apply_grasp_type(3, now=3.0)
        controller.step(Q_FIXED, QDOT_ZERO, now=3.0)

        self.assertTrue(
            controller.prepare_relative_rotation(np.deg2rad(5.0), now=3.0)
        )
        self.assertTrue(
            controller.prepare_relative_rotation(np.deg2rad(-10.0), now=3.1)
        )

        self.assertEqual(controller.relative_rotation_phase, "rotating")
        self.assertAlmostEqual(
            controller.relative_rotation_target_rad,
            np.deg2rad(-10.0),
        )

    def test_negative_rotation_command_generates_negative_moment(self):
        controller = GraspController(RuntimeConfig(), log=None)
        controller.apply_grasp_type(3, now=1.0)
        controller.step(Q_FIXED, QDOT_ZERO, now=1.0)
        self.assertTrue(
            controller.prepare_relative_rotation(np.deg2rad(-10.0), now=1.0)
        )

        output = controller.step(Q_FIXED, QDOT_ZERO, now=1.25)

        self.assertLess(output.relative_rotation_command_moment, 0.0)

    def test_rotation_targets_keep_start_geometry_and_follow_thumb_translation(self):
        cfg = RuntimeConfig(
            relative_rotation_reference_ramp_sec=0.0,
            relative_rotation_position_kp=1.0,
            relative_rotation_position_error_limit_m=1.0,
            relative_rotation_position_tolerance_m=0.0,
            relative_rotation_force_limit=100.0,
            relative_rotation_radius_min=1e-6,
        )
        controller = GraspController(cfg, log=None)
        controller.apply_grasp_type(3, now=1.0)
        initial = controller.step(Q_FIXED, QDOT_ZERO, now=1.0)
        target_angle = np.deg2rad(10.0)
        self.assertTrue(
            controller.prepare_relative_rotation(target_angle, now=1.0)
        )
        offset = np.array([0.0, 0.001, 0.0], dtype=np.float64)
        shifted_tips = {
            finger: position + offset
            for finger, position in initial.fingertip_positions.items()
        }

        rotation_forces = controller._calc_relative_rotation_forces(
            shifted_tips,
            now=1.1,
        )
        axis = controller.relative_rotation_axis
        axis_cross = np.array(
            [
                [0.0, -axis[2], axis[1]],
                [axis[2], 0.0, -axis[0]],
                [-axis[1], axis[0], 0.0],
            ],
            dtype=np.float64,
        )
        target_rotation = (
            np.eye(3)
            + np.sin(target_angle) * axis_cross
            + (1.0 - np.cos(target_angle)) * (axis_cross @ axis_cross)
        )
        for finger, start_position in (
            controller.relative_rotation_start_fingertips.items()
        ):
            if finger == 1:
                np.testing.assert_allclose(
                    rotation_forces[finger],
                    np.zeros(3),
                    rtol=0.0,
                    atol=0.0,
                )
                continue
            start_radius = start_position - controller.relative_rotation_pivot
            fixed_target = (
                shifted_tips[1]
                + target_rotation @ start_radius
            )
            expected_force = (
                fixed_target - shifted_tips[finger]
            ) / np.linalg.norm(start_radius)
            np.testing.assert_allclose(
                rotation_forces[finger],
                expected_force,
                rtol=0.0,
                atol=1e-12,
            )

    def test_rotation_damping_opposes_fingertip_velocity(self):
        cfg = RuntimeConfig(
            relative_rotation_reference_ramp_sec=0.0,
            relative_rotation_position_kp=0.0,
            relative_rotation_position_kd=1.0,
            relative_rotation_velocity_alpha=1.0,
            relative_rotation_position_error_limit_m=1.0,
            relative_rotation_position_tolerance_m=0.0,
            relative_rotation_force_limit=100.0,
            relative_rotation_radius_min=1e-6,
        )
        controller = GraspController(cfg, log=None)
        controller.apply_grasp_type(3, now=1.0)
        initial = controller.step(Q_FIXED, QDOT_ZERO, now=1.0)
        self.assertTrue(
            controller.prepare_relative_rotation(np.deg2rad(10.0), now=1.0)
        )
        qdot = np.zeros(20, dtype=np.float64)
        moving_finger = next(
            finger for finger in controller.use_fingers if finger != 1
        )
        indices = np.asarray(FINGER_JOINT_INDEX[moving_finger], dtype=int)
        qdot[indices[1]] = 0.25

        rotation_forces = controller._calc_relative_rotation_forces(
            initial.fingertip_positions,
            now=1.1,
            qdot=qdot,
        )

        fingertip_velocity = (
            tip_jacobian(
                controller.hand_q,
                moving_finger,
                eps=cfg.jacobian_eps,
            )
            @ qdot[indices]
        )
        rho = np.linalg.norm(
            initial.fingertip_positions[moving_finger]
            - controller.relative_rotation_pivot
        )
        np.testing.assert_allclose(
            rotation_forces[moving_finger],
            -fingertip_velocity / rho,
            rtol=0.0,
            atol=1e-12,
        )
        axis = controller.relative_rotation_axis
        angular_numerator = 0.0
        angular_denominator = 0.0
        for finger in controller.use_fingers:
            if finger == 1:
                continue
            radius = (
                initial.fingertip_positions[finger]
                - initial.fingertip_positions[1]
            )
            planar_radius = radius - np.dot(radius, axis) * axis
            finger_indices = np.asarray(FINGER_JOINT_INDEX[finger], dtype=int)
            velocity = (
                tip_jacobian(Q_FIXED, finger, eps=cfg.jacobian_eps)
                @ qdot[finger_indices]
            )
            angular_numerator += np.dot(
                axis,
                np.cross(planar_radius, velocity),
            )
            angular_denominator += np.dot(planar_radius, planar_radius)
        expected_angular_velocity = angular_numerator / angular_denominator
        self.assertAlmostEqual(
            controller.relative_rotation_angular_velocity,
            expected_angular_velocity,
        )

    def test_invalid_angle_and_non_regular_grasps_are_rejected(self):
        controller = GraspController(RuntimeConfig(), log=None)
        controller.apply_grasp_type(3, now=1.0)

        for value in (0.0, np.nan, np.inf, -np.inf):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    controller.prepare_relative_rotation(value, now=1.0)

        controller.step(Q_FIXED, QDOT_ZERO, now=1.1)
        with self.assertRaises(ValueError):
            controller.prepare_relative_rotation(np.deg2rad(10.1), now=1.1)

        for grasp_type in (-1, 6):
            with self.subTest(grasp_type=grasp_type):
                controller.apply_grasp_type(grasp_type, now=2.0)
                self.assertFalse(
                    controller.prepare_relative_rotation(
                        np.deg2rad(10.0),
                        now=2.0,
                    )
                )

    def test_pose_or_grasp_restart_cancels_active_rotation_request(self):
        controller = GraspController(RuntimeConfig(), log=None)
        controller.apply_grasp_type(3, now=10.0)
        controller.step(Q_FIXED, QDOT_ZERO, now=10.0)
        self.assertTrue(
            controller.prepare_relative_rotation(np.deg2rad(10.0), now=10.0)
        )
        self.assertEqual(controller.relative_rotation_phase, "rotating")

        controller.apply_pose_type(1, now=10.1)
        self.assertEqual(controller.relative_rotation_phase, "idle")

        controller.apply_grasp_type(3, now=11.0)
        controller.step(Q_FIXED, QDOT_ZERO, now=11.0)
        self.assertTrue(
            controller.prepare_relative_rotation(np.deg2rad(10.0), now=11.0)
        )
        controller.apply_grasp_type(-1, now=11.1)
        self.assertEqual(controller.relative_rotation_phase, "idle")

    def test_rotation_reaches_thumb_pivot_targets_and_holds_them(self):
        cfg = RuntimeConfig(
            relative_rotation_reference_ramp_sec=0.0,
            relative_rotation_velocity_alpha=1.0,
        )
        controller = GraspController(cfg, log=None)
        controller.apply_grasp_type(3, now=1.0)
        initial = controller.step(Q_FIXED, QDOT_ZERO, now=1.0)
        target = np.deg2rad(10.0)
        self.assertTrue(controller.prepare_relative_rotation(target, now=1.0))

        axis = controller.relative_rotation_axis
        skew = np.array(
            [
                [0.0, -axis[2], axis[1]],
                [axis[2], 0.0, -axis[0]],
                [-axis[1], axis[0], 0.0],
            ],
            dtype=np.float64,
        )
        rotation = (
            np.eye(3)
            + np.sin(target) * skew
            + (1.0 - np.cos(target)) * (skew @ skew)
        )
        pivot = initial.fingertip_positions[1]
        rotated_tips = {1: pivot.copy()}
        for finger, position in initial.fingertip_positions.items():
            if finger == 1:
                continue
            rotated_tips[finger] = pivot + rotation @ (position - pivot)

        rotation_forces = controller._calc_relative_rotation_forces(
            rotated_tips,
            now=1.5,
        )

        self.assertEqual(controller.relative_rotation_phase, "rotation_reached")
        self.assertAlmostEqual(controller.relative_rotation_current_rad, target)
        self.assertAlmostEqual(controller.relative_rotation_error_rad, 0.0)
        for force in rotation_forces.values():
            np.testing.assert_allclose(force, np.zeros(3), rtol=0.0, atol=0.0)

        disturbed_tips = {
            finger: position.copy()
            for finger, position in rotated_tips.items()
        }
        driven_finger = next(
            finger for finger in controller.use_fingers if finger != 1
        )
        disturbed_tips[driven_finger] += np.array([0.0, 0.001, 0.0])
        hold_forces = controller._calc_relative_rotation_forces(
            disturbed_tips,
            now=1.6,
        )
        self.assertEqual(controller.relative_rotation_phase, "rotation_reached")
        self.assertGreater(np.linalg.norm(hold_forces[driven_finger]), 0.0)
        np.testing.assert_allclose(
            hold_forces[1],
            np.zeros(3),
            rtol=0.0,
            atol=0.0,
        )

        timed_out_forces = controller._calc_relative_rotation_forces(
            disturbed_tips,
            now=3.01,
        )
        self.assertEqual(controller.relative_rotation_phase, "rotation_timeout")
        for force in timed_out_forces.values():
            np.testing.assert_allclose(
                force,
                np.zeros(3),
                rtol=0.0,
                atol=0.0,
            )

    def test_rotation_timeout_removes_tangential_force(self):
        cfg = RuntimeConfig(relative_rotation_timeout_sec=0.1)
        controller = GraspController(cfg, log=None)
        controller.apply_grasp_type(3, now=1.0)
        baseline = controller.step(Q_FIXED, QDOT_ZERO, now=1.0)
        self.assertTrue(
            controller.prepare_relative_rotation(np.deg2rad(10.0), now=1.0)
        )

        timed_out = controller.step(Q_FIXED, QDOT_ZERO, now=1.11)

        self.assertEqual(timed_out.relative_rotation_phase, "rotation_timeout")
        for force in timed_out.rotation_forces.values():
            np.testing.assert_allclose(force, np.zeros(3), rtol=0.0, atol=0.0)
        np.testing.assert_allclose(
            timed_out.grasp_tau,
            baseline.grasp_tau,
            rtol=0.0,
            atol=1e-12,
        )

    def test_balance_limit_failure_fades_to_zero_and_latches_until_regrasp(self):
        controller = GraspController(
            RuntimeConfig(
                rotation_force_balance_max_alpha_ratio=10.0,
                force_balance_error_ramp_sec=0.5,
            ),
            log=None,
        )
        controller.apply_grasp_type(3, now=1.0)
        valid = controller.step(Q_FIXED, QDOT_ZERO, now=1.0)
        self.assertTrue(
            controller.prepare_relative_rotation(np.deg2rad(10.0), now=1.01)
        )

        controller.update_config(
            replace(
                controller.cfg,
                rotation_force_balance_max_alpha_ratio=0.1,
            )
        )
        shifted_q = Q_FIXED + 0.02
        failed = controller.step(shifted_q, QDOT_ZERO, now=1.1)

        self.assertEqual(failed.relative_rotation_phase, "force_balance_error")
        self.assertEqual(failed.relative_rotation_target_rad, 0.0)
        np.testing.assert_allclose(failed.cv, failed.cg, rtol=0.0, atol=1e-12)
        expected_geometry = controller.policy.calc_zero_grasp_result(shifted_q)
        assert_vector_dict_allclose(
            self,
            failed.fingertip_positions,
            expected_geometry.fingertip_positions,
        )
        np.testing.assert_allclose(
            failed.grasp_tau,
            controller.policy.calc_tau_from_total_forces(
                shifted_q,
                valid.total_forces,
            ),
            rtol=0.0,
            atol=1e-12,
        )
        np.testing.assert_allclose(
            grasp_resultant(failed),
            np.zeros(3),
            rtol=0.0,
            atol=1e-8,
        )
        self.assertFalse(
            controller.prepare_relative_rotation(np.deg2rad(10.0), now=1.1)
        )

        halfway = controller.step(shifted_q, QDOT_ZERO, now=1.35)
        stopped = controller.step(shifted_q, QDOT_ZERO, now=1.6)
        halfway_forces = {
            finger: 0.5 * force
            for finger, force in valid.total_forces.items()
        }
        np.testing.assert_allclose(
            halfway.grasp_tau,
            controller.policy.calc_tau_from_total_forces(
                shifted_q,
                halfway_forces,
            ),
            rtol=0.0,
            atol=1e-12,
        )
        np.testing.assert_allclose(
            stopped.grasp_tau,
            np.zeros(20),
            rtol=0.0,
            atol=1e-12,
        )

        controller.update_config(
            replace(
                controller.cfg,
                rotation_force_balance_max_alpha_ratio=10.0,
            )
        )
        still_latched = controller.step(shifted_q, QDOT_ZERO, now=1.7)
        self.assertEqual(
            still_latched.relative_rotation_phase,
            "force_balance_error",
        )
        np.testing.assert_allclose(
            still_latched.grasp_tau,
            np.zeros(20),
            rtol=0.0,
            atol=1e-12,
        )

        controller.apply_grasp_type(3, now=2.0)
        recovered = controller.step(Q_FIXED, QDOT_ZERO, now=2.0)
        self.assertEqual(recovered.relative_rotation_phase, "idle")
        np.testing.assert_allclose(
            recovered.cv,
            recovered.cg,
            rtol=0.0,
            atol=1e-12,
        )
        np.testing.assert_allclose(
            grasp_resultant(recovered),
            np.zeros(3),
            rtol=0.0,
            atol=1e-8,
        )
        self.assertTrue(
            controller.prepare_relative_rotation(np.deg2rad(10.0), now=2.0)
        )


class RelativeTranslationTargetTest(unittest.TestCase):
    def test_cartesian_velocity_uses_received_joint_velocity(self):
        cfg = RuntimeConfig(relative_translation_velocity_alpha=1.0)
        controller = GraspController(cfg, log=None)
        controller.apply_grasp_type(3, now=1.0)
        controller.step(Q_FIXED, QDOT_ZERO, now=1.1)
        self.assertTrue(
            controller.prepare_relative_translation(
                np.array([0.001, 0.0, 0.0]),
                now=1.2,
            )
        )
        qdot = np.zeros(20, dtype=np.float64)
        qdot[1] = 0.2
        qdot[6] = -0.15
        output = controller.step(Q_FIXED, qdot, now=1.3)

        expected_fingertip_velocities = {}
        for finger in controller.use_fingers:
            indices = np.asarray(FINGER_JOINT_INDEX[finger], dtype=int)
            expected_fingertip_velocities[finger] = (
                tip_jacobian(Q_FIXED, finger, eps=cfg.jacobian_eps)
                @ qdot[indices]
            )
            np.testing.assert_allclose(
                controller.relative_translation_fingertip_velocities[finger],
                expected_fingertip_velocities[finger],
                rtol=0.0,
                atol=1e-12,
            )

        expected_centroid_velocity = sum(
            (
                controller.relative_translation_contact_weights[finger]
                * velocity
                for finger, velocity in expected_fingertip_velocities.items()
            ),
            np.zeros(3, dtype=np.float64),
        )
        np.testing.assert_allclose(
            output.relative_translation_centroid_velocity,
            expected_centroid_velocity,
            rtol=0.0,
            atol=1e-12,
        )

    def test_target_is_relative_and_adds_zero_moment_translation_wrench(self):
        controller = GraspController(RuntimeConfig(), log=None)
        controller.apply_grasp_type(3, now=1.0)
        before = controller.step(Q_FIXED, QDOT_ZERO, now=1.1)
        delta = np.array([0.010, 0.0, 0.0], dtype=np.float64)

        self.assertTrue(
            controller.prepare_relative_translation(delta, now=1.2)
        )
        after = controller.step(Q_FIXED, QDOT_ZERO, now=1.3)

        self.assertEqual(after.relative_translation_phase, "translating")
        np.testing.assert_allclose(
            after.relative_translation_start_centroid,
            before.cg,
            rtol=0.0,
            atol=1e-12,
        )
        np.testing.assert_allclose(
            after.relative_translation_target_centroid,
            before.cg + delta,
            rtol=0.0,
            atol=1e-12,
        )
        np.testing.assert_allclose(
            after.relative_translation_error,
            delta,
            rtol=0.0,
            atol=1e-12,
        )
        for finger, start_position in before.fingertip_positions.items():
            np.testing.assert_allclose(
                controller.relative_translation_start_fingertips[finger],
                start_position,
                rtol=0.0,
                atol=1e-12,
            )
            np.testing.assert_allclose(
                controller.relative_translation_target_fingertips[finger],
                start_position + delta,
                rtol=0.0,
                atol=1e-12,
            )
        self.assertAlmostEqual(
            sum(controller.relative_translation_contact_weights.values()),
            1.0,
        )
        resultant = sum(
            after.translation_forces.values(),
            np.zeros(3, dtype=np.float64),
        )
        moment = sum(
            (
                np.cross(
                    after.fingertip_positions[finger] - after.cg,
                    force,
                )
                for finger, force in after.translation_forces.items()
            ),
            np.zeros(3, dtype=np.float64),
        )
        self.assertGreater(resultant[0], 0.0)
        np.testing.assert_allclose(resultant[1:], np.zeros(2), rtol=0.0, atol=1e-10)
        np.testing.assert_allclose(moment, np.zeros(3), rtol=0.0, atol=1e-10)
        np.testing.assert_allclose(
            after.relative_translation_command_force,
            resultant,
            rtol=0.0,
            atol=1e-12,
        )
        self.assertGreater(np.linalg.norm(after.tau - before.tau), 1e-8)
        assert_vector_dict_allclose(
            self,
            after.total_forces,
            before.total_forces,
        )
        np.testing.assert_allclose(
            after.translation_torques,
            after.grasp_tau - before.grasp_tau,
            rtol=0.0,
            atol=1e-12,
        )
        self.assertGreater(np.max(np.abs(after.translation_torques)), 0.0)
        self.assertEqual(
            after.relative_translation_control_mode,
            "axis_centroid_dls_nullspace",
        )

    def test_reference_ramp_avoids_a_cartesian_force_step(self):
        cfg = RuntimeConfig(
            relative_translation_kd=0.0,
            relative_translation_shape_kd=0.0,
            relative_translation_reference_ramp_sec=0.5,
        )
        controller = GraspController(cfg, log=None)
        controller.apply_grasp_type(3, now=1.0)
        controller.step(Q_FIXED, QDOT_ZERO, now=1.1)
        delta = np.array([0.005, 0.0, 0.0], dtype=np.float64)
        self.assertTrue(controller.prepare_relative_translation(delta, now=1.2))

        early = controller.step(Q_FIXED, QDOT_ZERO, now=1.3)
        expected_progress = 0.2 * 0.2 * (3.0 - 2.0 * 0.2)
        self.assertAlmostEqual(
            controller.relative_translation_reference_progress,
            expected_progress,
        )
        np.testing.assert_allclose(
            early.relative_translation_command_force,
            cfg.relative_translation_kp * expected_progress * delta,
            rtol=0.0,
            atol=1e-12,
        )

        settled_reference = controller.step(Q_FIXED, QDOT_ZERO, now=1.7)
        np.testing.assert_allclose(
            settled_reference.relative_translation_command_force,
            cfg.relative_translation_kp * delta,
            rtol=0.0,
            atol=1e-12,
        )

    def test_cross_axis_offset_does_not_create_centroid_drive_force(self):
        cfg = RuntimeConfig(
            relative_translation_kd=0.0,
            relative_translation_shape_kd=0.0,
            relative_translation_velocity_alpha=0.0,
            relative_translation_reference_ramp_sec=0.0,
        )
        controller = GraspController(cfg, log=None)
        controller.apply_grasp_type(3, now=1.0)
        baseline = controller.step(Q_FIXED, QDOT_ZERO, now=1.1)
        self.assertTrue(
            controller.prepare_relative_translation(
                np.array([0.001, 0.0, 0.0]),
                now=1.2,
            )
        )
        cross_offset = np.array([0.0, 0.005, -0.004], dtype=np.float64)
        shifted_tips = {
            finger: position + cross_offset
            for finger, position in baseline.fingertip_positions.items()
        }
        forces = controller._calc_relative_translation_forces(
            baseline.cg + cross_offset,
            shifted_tips,
            now=1.3,
        )
        resultant = sum(forces.values(), np.zeros(3, dtype=np.float64))
        self.assertGreater(resultant[0], 0.0)
        np.testing.assert_allclose(
            resultant[1:],
            np.zeros(2),
            rtol=0.0,
            atol=1e-12,
        )

    def test_independent_fingertip_error_creates_shape_restoring_moment(self):
        cfg = RuntimeConfig(
            relative_translation_kd=0.0,
            relative_translation_shape_kd=0.0,
            relative_translation_reference_ramp_sec=0.0,
            relative_translation_force_limit=100.0,
            relative_translation_per_finger_force_limit=100.0,
        )
        controller = GraspController(cfg, log=None)
        controller.apply_grasp_type(3, now=1.0)
        controller.step(Q_FIXED, QDOT_ZERO, now=1.1)
        self.assertTrue(
            controller.prepare_relative_translation(
                np.array([0.001, 0.0, 0.0]),
                now=1.2,
            )
        )

        shifted_q = Q_FIXED.copy()
        shifted_q[4] += 0.02
        output = controller.step(shifted_q, QDOT_ZERO, now=1.3)
        resultant = sum(
            output.translation_forces.values(),
            np.zeros(3, dtype=np.float64),
        )
        shape_forces = {
            finger: force
            - controller.relative_translation_contact_weights[finger] * resultant
            for finger, force in output.translation_forces.items()
        }
        np.testing.assert_allclose(
            sum(shape_forces.values(), np.zeros(3, dtype=np.float64)),
            np.zeros(3),
            rtol=0.0,
            atol=1e-12,
        )

        restoring_moment = sum(
            (
                np.cross(
                    output.fingertip_positions[finger] - output.cg,
                    force,
                )
                for finger, force in shape_forces.items()
            ),
            np.zeros(3, dtype=np.float64),
        )
        self.assertGreater(np.linalg.norm(restoring_moment), 1e-4)

    def test_target_rejects_out_of_range_distance(self):
        controller = GraspController(
            RuntimeConfig(relative_translation_max_m=0.010),
            log=None,
        )
        controller.apply_grasp_type(3, now=1.0)
        controller.step(Q_FIXED, QDOT_ZERO, now=1.1)

        with self.assertRaises(ValueError):
            controller.prepare_relative_translation(
                np.array([0.0101, 0.0, 0.0]),
                now=1.2,
            )

    def test_axis_dls_joint_position_control_is_bounded_for_all_xyz_axes(self):
        cfg = RuntimeConfig(
            relative_translation_reference_ramp_sec=0.0,
            relative_translation_joint_correction_limit_rad=0.20,
            relative_translation_position_torque_limit=0.15,
        )
        for axis in range(3):
            controller = GraspController(cfg, log=None)
            controller.apply_grasp_type(2, now=1.0)
            controller.step(HAND_PRE_GRASP_POSE, QDOT_ZERO, now=1.1)
            delta = np.zeros(3, dtype=np.float64)
            delta[axis] = 0.005
            self.assertTrue(
                controller.prepare_relative_translation(delta, now=1.2)
            )
            output = controller.step(HAND_PRE_GRASP_POSE, QDOT_ZERO, now=1.3)

            self.assertEqual(
                output.relative_translation_control_mode,
                "axis_centroid_dls_nullspace",
            )
            self.assertGreater(
                np.max(np.abs(output.relative_translation_joint_error)),
                0.0,
            )
            self.assertLessEqual(
                np.max(np.abs(output.relative_translation_joint_error)),
                0.20 + 1e-12,
            )
            self.assertGreater(
                np.max(np.abs(output.relative_translation_position_torques)),
                0.0,
            )
            self.assertLessEqual(
                np.max(np.abs(output.relative_translation_position_torques)),
                0.15 + 1e-12,
            )
            self.assertGreater(output.relative_translation_dls_sigma_min, 0.0)
            self.assertTrue(
                np.isfinite(output.relative_translation_dls_condition)
            )

    def test_grasp_torque_is_projected_out_of_command_axis_task(self):
        cfg = RuntimeConfig(relative_translation_reference_ramp_sec=0.0)
        controller = GraspController(cfg, log=None)
        controller.apply_grasp_type(3, now=1.0)
        controller.step(HAND_PRE_GRASP_POSE, QDOT_ZERO, now=1.1)
        self.assertTrue(
            controller.prepare_relative_translation(
                np.array([0.001, 0.0, 0.0]),
                now=1.2,
            )
        )
        output = controller.step(HAND_PRE_GRASP_POSE, QDOT_ZERO, now=1.3)

        active_indices = np.concatenate(
            [
                np.asarray(FINGER_JOINT_INDEX[finger], dtype=int)
                for finger in controller.use_fingers
            ]
        )
        centroid_jacobian = np.hstack(
            [
                controller.relative_translation_contact_weights[finger]
                * tip_jacobian(HAND_PRE_GRASP_POSE, finger)
                for finger in controller.use_fingers
            ]
        )
        translation_axis = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        axis_jacobian = translation_axis.reshape(1, 3) @ centroid_jacobian
        leakage = axis_jacobian @ (
            output.relative_translation_nullspace_grasp_torques[active_indices]
        )
        np.testing.assert_allclose(
            leakage,
            np.zeros(1),
            rtol=0.0,
            atol=1e-10,
        )

    def test_cross_axis_centroid_error_does_not_change_position_torque(self):
        cfg = RuntimeConfig(relative_translation_reference_ramp_sec=0.0)
        controller = GraspController(cfg, log=None)
        controller.apply_grasp_type(3, now=1.0)
        baseline = controller.step(HAND_PRE_GRASP_POSE, QDOT_ZERO, now=1.1)
        self.assertTrue(
            controller.prepare_relative_translation(
                np.array([0.001, 0.0, 0.0]),
                now=1.2,
            )
        )
        controller.relative_translation_reference_progress = 1.0

        controller.relative_translation_error = np.array(
            [0.001, 0.0, 0.0],
            dtype=np.float64,
        )
        axis_only_torque = controller._calc_relative_translation_joint_control(
            base_total_forces=baseline.total_forces,
            base_grasp_tau=baseline.grasp_tau,
            qdot=QDOT_ZERO,
        )[2]

        controller.relative_translation_error = np.array(
            [0.001, 0.050, -0.040],
            dtype=np.float64,
        )
        cross_offset_torque = controller._calc_relative_translation_joint_control(
            base_total_forces=baseline.total_forces,
            base_grasp_tau=baseline.grasp_tau,
            qdot=QDOT_ZERO,
        )[2]

        np.testing.assert_allclose(
            cross_offset_torque,
            axis_only_torque,
            rtol=0.0,
            atol=1e-12,
        )

    def test_pose_or_grasp_change_cancels_target(self):
        controller = GraspController(RuntimeConfig(), log=None)
        controller.apply_grasp_type(3, now=1.0)
        controller.step(Q_FIXED, QDOT_ZERO, now=1.1)
        self.assertTrue(
            controller.prepare_relative_translation(
                np.array([0.001, 0.0, 0.0]),
                now=1.2,
            )
        )

        controller.apply_grasp_type(-1, now=1.3)

        self.assertEqual(controller.relative_translation_phase, "idle")
        np.testing.assert_allclose(
            controller.relative_translation_target_centroid,
            np.zeros(3),
            rtol=0.0,
            atol=0.0,
        )

    def test_two_through_five_finger_translation_has_no_cross_axis_resultant_or_moment(self):
        for grasp_type in (1, 2, 3, 4, 5):
            for axis in range(3):
                with self.subTest(grasp_type=grasp_type, axis=axis):
                    controller = GraspController(RuntimeConfig(), log=None)
                    controller.apply_grasp_type(grasp_type, now=1.0)
                    controller.step(Q_FIXED, QDOT_ZERO, now=1.1)
                    delta = np.zeros(3, dtype=np.float64)
                    delta[axis] = 0.001
                    self.assertTrue(
                        controller.prepare_relative_translation(
                            delta,
                            now=1.2,
                        )
                    )
                    output = controller.step(Q_FIXED, QDOT_ZERO, now=1.21)
                    resultant = sum(
                        output.translation_forces.values(),
                        np.zeros(3, dtype=np.float64),
                    )
                    moment = sum(
                        (
                            np.cross(
                                output.fingertip_positions[finger] - output.cg,
                                force,
                            )
                            for finger, force in output.translation_forces.items()
                        ),
                        np.zeros(3, dtype=np.float64),
                    )

                    self.assertGreater(resultant[axis], 0.0)
                    cross_axes = np.delete(resultant, axis)
                    np.testing.assert_allclose(
                        cross_axes,
                        np.zeros(2),
                        rtol=0.0,
                        atol=1e-10,
                    )
                    np.testing.assert_allclose(
                        moment,
                        np.zeros(3),
                        rtol=0.0,
                        atol=1e-10,
                    )

    def test_timeout_removes_translation_force(self):
        cfg = RuntimeConfig(
            relative_translation_timeout_sec=0.1,
            inactive_collision_avoidance_enable=False,
        )
        controller = GraspController(cfg, log=None)
        controller.apply_grasp_type(3, now=1.0)
        baseline = controller.step(Q_FIXED, QDOT_ZERO, now=1.1)
        self.assertTrue(
            controller.prepare_relative_translation(
                np.array([0.001, 0.0, 0.0]),
                now=1.2,
            )
        )

        timed_out = controller.step(Q_FIXED, QDOT_ZERO, now=1.31)

        self.assertEqual(timed_out.relative_translation_phase, "translation_timeout")
        for force in timed_out.translation_forces.values():
            np.testing.assert_allclose(force, np.zeros(3), rtol=0.0, atol=0.0)
        np.testing.assert_allclose(
            timed_out.translation_torques,
            np.zeros(20),
            rtol=0.0,
            atol=0.0,
        )
        np.testing.assert_allclose(timed_out.tau, baseline.tau, rtol=0.0, atol=1e-12)

    def test_force_balance_failure_removes_translation_from_fallback_cache(self):
        controller = GraspController(RuntimeConfig(), log=None)
        controller.apply_grasp_type(3, now=1.0)
        controller.step(Q_FIXED, QDOT_ZERO, now=1.1)
        self.assertTrue(
            controller.prepare_relative_translation(
                np.array([0.001, 0.0, 0.0]),
                now=1.2,
            )
        )
        moving = controller.step(Q_FIXED, QDOT_ZERO, now=1.21)
        self.assertGreater(
            np.linalg.norm(moving.relative_translation_command_force),
            0.0,
        )

        controller.update_config(
            replace(
                controller.cfg,
                rotation_force_balance_max_alpha_ratio=0.1,
            )
        )
        failed = controller.step(Q_FIXED + 0.02, QDOT_ZERO, now=1.3)

        self.assertEqual(failed.relative_translation_phase, "idle")
        for force in failed.translation_forces.values():
            np.testing.assert_allclose(force, np.zeros(3), rtol=0.0, atol=0.0)
        resultant = sum(
            failed.total_forces.values(),
            np.zeros(3, dtype=np.float64),
        )
        np.testing.assert_allclose(resultant, np.zeros(3), rtol=0.0, atol=1e-8)


if __name__ == "__main__":
    unittest.main()
