import unittest
from dataclasses import replace

import numpy as np

from dg5f_grasp_control.config import RuntimeConfig
from dg5f_grasp_control.grasp_controller import GraspController
from dg5f_grasp_control.grasp_policy import GraspPolicy


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


class GeneralGraspRotationPreparationTest(unittest.TestCase):
    def test_relative_rotation_waits_for_first_successful_balance_cycle(self):
        controller = GraspController(RuntimeConfig(), log=None)
        controller.apply_grasp_type(3, now=1.0)

        self.assertFalse(
            controller.prepare_relative_rotation(np.pi / 6.0, now=1.0)
        )
        controller.step(Q_FIXED, QDOT_ZERO, now=1.0)
        self.assertTrue(
            controller.prepare_relative_rotation(np.pi / 6.0, now=1.01)
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

    def test_relative_rotation_request_is_immediately_ready_without_rotation_force(self):
        angle = np.pi / 6.0

        for grasp_type in (1, 2, 3, 4, 5):
            with self.subTest(grasp_type=grasp_type):
                controller = GraspController(RuntimeConfig(), log=None)
                controller.apply_grasp_type(grasp_type, now=2.0)
                controller.step(Q_FIXED, QDOT_ZERO, now=2.0)

                self.assertTrue(controller.prepare_relative_rotation(angle, now=2.0))
                self.assertEqual(controller.relative_rotation_phase, "rotation_ready")
                self.assertAlmostEqual(controller.relative_rotation_target_rad, angle)

                output = controller.step(Q_FIXED, QDOT_ZERO, now=2.0)
                self.assertEqual(output.relative_rotation_phase, "rotation_ready")
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
                for force in output.rotation_forces.values():
                    np.testing.assert_allclose(
                        force,
                        np.zeros(3),
                        rtol=0.0,
                        atol=1e-12,
                    )

    def test_new_relative_command_replaces_instead_of_accumulating_angle(self):
        controller = GraspController(RuntimeConfig(), log=None)
        controller.apply_grasp_type(3, now=3.0)
        controller.step(Q_FIXED, QDOT_ZERO, now=3.0)

        self.assertTrue(controller.prepare_relative_rotation(np.pi / 6.0, now=3.0))
        self.assertTrue(controller.prepare_relative_rotation(-np.pi / 4.0, now=3.1))

        self.assertEqual(controller.relative_rotation_phase, "rotation_ready")
        self.assertAlmostEqual(
            controller.relative_rotation_target_rad,
            -np.pi / 4.0,
        )

    def test_invalid_angle_and_non_regular_grasps_are_rejected(self):
        controller = GraspController(RuntimeConfig(), log=None)
        controller.apply_grasp_type(3, now=1.0)

        for value in (0.0, np.nan, np.inf, -np.inf):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    controller.prepare_relative_rotation(value, now=1.0)

        for grasp_type in (-1, 6, 7):
            with self.subTest(grasp_type=grasp_type):
                controller.apply_grasp_type(grasp_type, now=2.0)
                self.assertFalse(
                    controller.prepare_relative_rotation(np.pi / 6.0, now=2.0)
                )

    def test_pose_or_grasp_restart_cancels_ready_rotation_request(self):
        controller = GraspController(RuntimeConfig(), log=None)
        controller.apply_grasp_type(3, now=10.0)
        controller.step(Q_FIXED, QDOT_ZERO, now=10.0)
        self.assertTrue(controller.prepare_relative_rotation(np.pi / 6.0, now=10.0))
        self.assertEqual(controller.relative_rotation_phase, "rotation_ready")

        controller.apply_pose_type(1, now=10.1)
        self.assertEqual(controller.relative_rotation_phase, "idle")

        controller.apply_grasp_type(3, now=11.0)
        controller.step(Q_FIXED, QDOT_ZERO, now=11.0)
        self.assertTrue(controller.prepare_relative_rotation(np.pi / 6.0, now=11.0))
        controller.apply_grasp_type(-1, now=11.1)
        self.assertEqual(controller.relative_rotation_phase, "idle")

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
            controller.prepare_relative_rotation(np.pi / 6.0, now=1.01)
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
            controller.prepare_relative_rotation(np.pi / 6.0, now=1.1)
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
            controller.prepare_relative_rotation(np.pi / 6.0, now=2.0)
        )


class RelativeTranslationTargetTest(unittest.TestCase):
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
        cfg = RuntimeConfig(relative_translation_timeout_sec=0.1)
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


class GraspType7LegacyCompatibilityTest(unittest.TestCase):
    def test_type7_controller_keeps_legacy_alpha_and_virtual_centroid(self):
        cfg = RuntimeConfig(
            alpha1=3.0,
            thumb_centroid_bias=0.5,
            rotation_enable_for_grasp_type7=False,
        )
        controller = GraspController(cfg, log=None)
        controller.apply_grasp_type(7, now=1.0)
        output = controller.step(Q_FIXED, QDOT_ZERO, now=1.0)

        expected = GraspPolicy([1, 2, 3, 4], cfg).calc_grasp_tau(
            Q_FIXED,
            alpha_distribution_mode="legacy",
        )

        self.assertEqual(output.alpha, expected.alpha)
        np.testing.assert_allclose(output.cg, expected.cg, rtol=0.0, atol=1e-12)
        np.testing.assert_allclose(output.cv, expected.cv, rtol=0.0, atol=1e-12)
        assert_vector_dict_allclose(
            self,
            output.grasp_forces,
            expected.grasp_forces,
        )
        self.assertGreater(np.linalg.norm(output.cv - output.cg), 1e-6)

    def test_type7_thumb_detach_legacy_path_remains_finite(self):
        cfg = RuntimeConfig(rotation_enable_for_grasp_type7=False)
        controller = GraspController(cfg, log=None)
        controller.apply_grasp_type(7, now=1.0)
        controller._start_grasp_type7_thumb_detach_pregrasp()

        output = controller.step(Q_FIXED, QDOT_ZERO, now=1.1)
        expected = GraspPolicy([2, 3, 4], cfg).calc_grasp_tau(
            Q_FIXED,
            alpha_distribution_mode="legacy",
        )

        self.assertEqual(output.g7_phase, "thumb_pose_0_140_0_0")
        self.assertEqual(output.use_fingers, [2, 3, 4])
        self.assertEqual(output.alpha, expected.alpha)
        assert_vector_dict_allclose(
            self,
            output.grasp_forces,
            expected.grasp_forces,
        )
        assert_output_is_finite(self, output)


if __name__ == "__main__":
    unittest.main()
