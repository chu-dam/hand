import unittest
from unittest.mock import patch

import numpy as np

from dg5f_grasp_control.config import RuntimeConfig
from dg5f_grasp_control.grasp_controller import GraspController
from dg5f_grasp_control.grasp_policy import polygon_centroid_3d
from dg5f_grasp_control.kinematics import set_hand_side


class PaperRotationTest(unittest.TestCase):
    def make_controller(self, angle=0.1, *, sequential=False, **config):
        set_hand_side("right")
        config.setdefault("paper_rotation_reference_ramp_sec", 0.0)
        controller = GraspController(RuntimeConfig(hand_side="right", **config), log=None)
        controller.apply_grasp_type(3, now=0.5)
        controller.step(np.zeros(20), np.zeros(20), now=0.6)
        self.assertTrue(controller.prepare_relative_rotation(
            angle, now=1.0, paper=True, center_after_rotation=sequential,
        ))
        return controller

    def test_ui_request_drives_all_contacts_and_maps_added_forces(self):
        controller = self.make_controller()
        output = controller.step(np.zeros(20), np.zeros(20), now=1.01)
        self.assertEqual(output.relative_rotation_control_mode, "paper_centroid_tangential")
        self.assertGreater(np.linalg.norm(output.rotation_forces[1]), 0.0)
        np.testing.assert_allclose(
            output.relative_rotation_pivot,
            polygon_centroid_3d(np.array(list(output.fingertip_positions.values()))),
        )
        for finger in output.use_fingers:
            np.testing.assert_allclose(
                output.total_forces[finger],
                output.grasp_forces[finger] + output.collision_forces[finger]
                + output.rotation_forces[finger] + output.center_hold_forces[finger],
            )
        np.testing.assert_allclose(
            output.grasp_tau,
            controller.policy.calc_tau_from_total_forces(controller.hand_q, output.total_forces),
        )

    def test_paper_force_is_tangent_radius_independent_and_signed(self):
        points = {
            1: np.array([0.0, 0.04, 0.0]),
            2: np.array([0.0, -0.01, 0.03]),
            4: np.array([0.0, -0.03, -0.03]),
        }
        for angle in (0.1, -0.1):
            for scale in (0.5, 1.0, 2.0):
                with self.subTest(angle=angle, scale=scale):
                    controller = self.make_controller(angle)
                    scaled = {f: scale * p for f, p in points.items()}
                    controller.relative_rotation_pivot = np.zeros(3)
                    controller.relative_rotation_start_fingertips = scaled
                    forces = controller._calc_relative_rotation_forces(scaled, 1.01, np.zeros(20))
                    for finger, radius in scaled.items():
                        tangent = np.cross(controller.relative_rotation_axis, radius)
                        expected = controller.cfg.paper_rotation_kr * np.sign(angle) * (
                            abs(angle) - np.deg2rad(controller.cfg.paper_rotation_tolerance_deg)
                        )
                        np.testing.assert_allclose(forces[finger], expected * tangent / np.linalg.norm(tangent))
                        self.assertAlmostEqual(float(np.dot(forces[finger], radius)), 0.0)
                    self.assertGreater(controller.relative_rotation_command_moment * angle, 0.0)

    def test_angle_estimate_ignores_translation_but_completion_requires_center_return(self):
        controller = self.make_controller()
        axis = controller.relative_rotation_axis
        cross = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
        angle = controller.relative_rotation_target_rad
        rotation = np.eye(3) + np.sin(angle) * cross + (1 - np.cos(angle)) * cross @ cross
        center = controller.relative_rotation_pivot
        shift = np.array([0.005, -0.003, 0.002])
        translated = {f: p + shift for f, p in controller.relative_rotation_start_fingertips.items()}
        self.assertAlmostEqual(controller._estimate_relative_rotation_angle(translated), 0.0)
        rotated = {
            f: center + rotation @ (p - center) + shift
            for f, p in controller.relative_rotation_start_fingertips.items()
        }
        controller._calc_relative_rotation_forces(rotated, 1.01, np.zeros(20))
        self.assertEqual(controller.relative_rotation_phase, "rotating")
        forces = controller._calc_relative_rotation_forces(rotated, 1.10, np.zeros(20))
        self.assertEqual(controller.relative_rotation_phase, "rotating")
        self.assertAlmostEqual(controller.relative_rotation_current_rad, angle)
        for force in forces.values():
            np.testing.assert_allclose(force, 0.0, atol=1e-12)
        np.testing.assert_allclose(
            sum(controller.paper_rotation_center_forces.values()),
            -controller.cfg.paper_rotation_center_kp * shift,
        )
        centered = {f: p - shift for f, p in rotated.items()}
        controller._calc_relative_rotation_forces(centered, 1.11, np.zeros(20))
        controller._calc_relative_rotation_forces(centered, 1.20, np.zeros(20))
        self.assertEqual(controller.relative_rotation_phase, "rotation_reached")

    def test_center_force_is_shared_capped_and_cleared_on_timeout(self):
        controller = self.make_controller()
        points = controller.relative_rotation_start_fingertips
        shift = np.array([0.1, 0.0, 0.0])
        translated = {f: p + shift for f, p in points.items()}
        controller._calc_relative_rotation_forces(translated, 1.01, np.zeros(20))
        for force in controller.paper_rotation_center_forces.values():
            np.testing.assert_allclose(
                force,
                [-controller.cfg.paper_rotation_center_force_limit / len(points), 0.0, 0.0],
                atol=1e-12,
            )
        controller._calc_relative_rotation_forces(translated, 2.01, np.zeros(20))
        self.assertEqual(controller.paper_rotation_center_forces, {})

    def test_center_damping_opposes_common_translation(self):
        controller = self.make_controller()
        points = controller.relative_rotation_start_fingertips
        velocity = np.array([0.01, -0.02, 0.0])
        jacobian = np.column_stack([velocity, np.zeros((3, 3))])
        with patch("dg5f_grasp_control.grasp_controller.tip_jacobian", return_value=jacobian):
            controller._calc_relative_rotation_forces(points, 1.01, np.ones(20))
        np.testing.assert_allclose(
            sum(controller.paper_rotation_center_forces.values()),
            -controller.cfg.paper_rotation_center_kd * velocity, atol=1e-10,
        )

    def test_asymmetric_five_finger_rigid_rotation_does_not_trigger_center_damping(self):
        controller = self.make_controller()
        points = {
            1: np.array([0.0, 0.06, 0.0]),
            2: np.array([0.0, 0.01, 0.05]),
            3: np.array([0.0, -0.04, 0.02]),
            4: np.array([0.0, -0.03, -0.02]),
            5: np.array([0.0, 0.0, -0.04]),
        }
        center = polygon_centroid_3d(np.array(list(points.values())))
        controller.relative_rotation_start_fingertips = points
        controller.relative_rotation_pivot = center.copy()

        def jacobian(q, finger, **kwargs):
            velocity = 0.5 * np.cross(controller.relative_rotation_axis, points[finger] - center)
            return np.column_stack([velocity, np.zeros((3, 3))])

        with patch("dg5f_grasp_control.grasp_controller.tip_jacobian", side_effect=jacobian):
            controller._calc_relative_rotation_forces(points, 1.01, np.ones(20))
        for force in controller.paper_rotation_center_forces.values():
            np.testing.assert_allclose(force, 0.0, atol=1e-10)

    def test_reference_ramp_softens_initial_force(self):
        controller = self.make_controller(paper_rotation_reference_ramp_sec=0.5)
        points = controller.relative_rotation_start_fingertips
        start = controller._calc_relative_rotation_forces(points, 1.0, np.zeros(20))
        half = controller._calc_relative_rotation_forces(points, 1.25, np.zeros(20))
        full = controller._calc_relative_rotation_forces(points, 1.5, np.zeros(20))
        for finger in points:
            np.testing.assert_allclose(start[finger], 0.0)
            tolerance = np.deg2rad(controller.cfg.paper_rotation_tolerance_deg)
            ratio = (0.05 - tolerance) / (0.1 - tolerance)
            np.testing.assert_allclose(half[finger], ratio * full[finger])

    def test_angle_deadband_removes_small_reversing_force_and_latches_stop(self):
        controller = self.make_controller()
        controller._estimate_relative_rotation_angle = lambda points: 0.1 + np.deg2rad(0.5)
        points = controller.relative_rotation_start_fingertips
        first = controller._calc_relative_rotation_forces(points, 1.01, np.zeros(20))
        self.assertTrue(all(np.all(force == 0.0) for force in first.values()))
        controller._calc_relative_rotation_forces(points, 1.10, np.zeros(20))
        self.assertEqual(controller.relative_rotation_phase, "rotation_reached")
        controller._estimate_relative_rotation_angle = lambda points: -0.1
        stopped = controller._calc_relative_rotation_forces(points, 1.11, np.zeros(20))
        self.assertTrue(all(np.all(force == 0.0) for force in stopped.values()))

    def test_angular_velocity_step_is_filtered_before_damping(self):
        controller = self.make_controller(relative_rotation_velocity_alpha=0.2)
        points = controller.relative_rotation_start_fingertips
        center = controller.relative_rotation_pivot
        qdot = np.ones(20)

        def jacobian(q, finger, **kwargs):
            velocity = 0.5 * np.cross(controller.relative_rotation_axis, points[finger] - center)
            return np.column_stack([velocity, np.zeros((3, 3))])

        with patch("dg5f_grasp_control.grasp_controller.tip_jacobian", side_effect=jacobian):
            controller._calc_relative_rotation_forces(points, 1.01, qdot)
        self.assertAlmostEqual(controller.relative_rotation_angular_velocity, 0.1)

    def test_proportional_force_is_continuous_at_tolerance_boundary(self):
        controller = self.make_controller()
        points = controller.relative_rotation_start_fingertips
        tolerance = np.deg2rad(controller.cfg.paper_rotation_tolerance_deg)
        epsilon = 1e-6
        controller._estimate_relative_rotation_angle = lambda points: 0.1 - tolerance - epsilon
        forces = controller._calc_relative_rotation_forces(points, 1.01, np.zeros(20))
        for force in forces.values():
            self.assertAlmostEqual(np.linalg.norm(force), controller.cfg.paper_rotation_kr * epsilon)

    def test_delayed_damping_does_not_drive_velocity_reversal(self):
        controller = self.make_controller(paper_rotation_kr=0.0)
        points = controller.relative_rotation_start_fingertips
        center = controller.relative_rotation_pivot
        omega = 0.5

        def jacobian(q, finger, **kwargs):
            velocity = omega * np.cross(controller.relative_rotation_axis, points[finger] - center)
            return np.column_stack([velocity, np.zeros((3, 3))])

        with patch("dg5f_grasp_control.grasp_controller.tip_jacobian", side_effect=jacobian):
            for now in (1.01, 1.02, 1.03):
                controller._calc_relative_rotation_forces(points, now, np.ones(20))
            omega = -0.1
            forces = controller._calc_relative_rotation_forces(points, 1.04, np.ones(20))
        self.assertGreater(controller.relative_rotation_angular_velocity, 0.0)
        for force in forces.values():
            np.testing.assert_allclose(force, 0.0)

    def test_sequential_rotation_has_no_center_force_then_returns_only_world_xy(self):
        controller = self.make_controller(sequential=True)
        rotation = np.array([[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]])
        controller.set_rotation_hand_to_world(rotation)
        self.assertTrue(controller.prepare_relative_rotation(
            0.1, now=1.0, paper=True, center_after_rotation=True,
        ))
        start = controller.relative_rotation_start_fingertips
        shift_world = np.array([0.005, -0.003, 0.02])
        shifted = {f: p + rotation.T @ shift_world for f, p in start.items()}
        forces = controller._calc_relative_rotation_forces(shifted, 1.01, np.zeros(20))
        self.assertEqual(controller.relative_rotation_phase, "rotating")
        self.assertEqual(controller.paper_rotation_center_forces, {})
        self.assertTrue(any(np.linalg.norm(force) > 0.0 for force in forces.values()))
        controller._estimate_relative_rotation_angle = lambda points: 0.1
        controller._calc_relative_rotation_forces(shifted, 1.02, np.zeros(20))
        controller._calc_relative_rotation_forces(shifted, 1.03, np.zeros(20))
        forces = controller._calc_relative_rotation_forces(shifted, 1.11, np.zeros(20))
        self.assertEqual(controller.relative_rotation_phase, "centering")
        self.assertTrue(all(np.all(force == 0.0) for force in forces.values()))
        forces = controller._calc_relative_rotation_forces(shifted, 1.12, np.zeros(20))
        self.assertTrue(all(np.all(force == 0.0) for force in forces.values()))
        world_force = rotation @ sum(controller.paper_rotation_center_forces.values())
        np.testing.assert_allclose(world_force, [-1.125, 0.675, 0.0], atol=1e-10)
        only_world_z_shift = {f: p + rotation.T @ np.array([0.0, 0.0, 0.02]) for f, p in start.items()}
        controller._calc_relative_rotation_forces(only_world_z_shift, 1.13, np.zeros(20))
        controller._calc_relative_rotation_forces(only_world_z_shift, 1.22, np.zeros(20))
        self.assertEqual(controller.relative_rotation_phase, "rotation_reached")
        self.assertEqual(controller.paper_rotation_center_forces, {})
        controller._calc_relative_rotation_forces(shifted, 1.23, np.zeros(20))
        self.assertEqual(controller.paper_rotation_center_forces, {})

    def test_sequential_centering_has_its_own_timeout_and_keeps_grasp(self):
        controller = self.make_controller(sequential=True, paper_rotation_center_timeout_sec=0.5)
        controller._estimate_relative_rotation_angle = lambda points: 0.1
        controller.step(np.zeros(20), np.zeros(20), now=1.81)
        controller.step(np.zeros(20), np.zeros(20), now=1.82)
        controller.step(np.zeros(20), np.zeros(20), now=1.90)
        self.assertEqual(controller.relative_rotation_phase, "centering")
        controller.paper_center_target_world[0] += 0.005
        output = controller.step(np.zeros(20), np.zeros(20), now=2.05)
        self.assertEqual(output.relative_rotation_phase, "centering")
        self.assertEqual(output.relative_rotation_control_mode, "paper_rotate_then_center_xy")
        self.assertEqual(set(output.grasp_forces), {1, 2, 3})
        for finger in output.use_fingers:
            np.testing.assert_allclose(output.rotation_forces[finger], 0.0)
            np.testing.assert_allclose(
                output.total_forces[finger], output.grasp_forces[finger]
                + output.collision_forces[finger] + output.center_hold_forces[finger],
            )
        output = controller.step(np.zeros(20), np.zeros(20), now=2.41)
        self.assertEqual(output.relative_rotation_phase, "rotation_timeout")
        self.assertEqual(controller.paper_rotation_center_forces, {})
        np.testing.assert_allclose(
            output.grasp_tau,
            controller.policy.calc_tau_from_total_forces(controller.hand_q, output.total_forces),
        )
        controller.cancel_relative_rotation()
        self.assertFalse(controller.paper_center_after_rotation)
        self.assertEqual(controller.paper_rotation_stopped_count, 0)

    def test_sequential_transition_needs_three_slow_samples_after_ramp_not_angle(self):
        controller = self.make_controller(
            sequential=True, paper_rotation_reference_ramp_sec=0.5,
        )
        points = controller.relative_rotation_start_fingertips
        for now in (1.1, 1.2, 1.49):
            controller._calc_relative_rotation_forces(points, now, np.zeros(20))
        self.assertEqual(controller.paper_rotation_stopped_count, 0)
        for now in (1.5, 1.501):
            controller._calc_relative_rotation_forces(points, now, np.zeros(20))
        self.assertEqual(controller.relative_rotation_phase, "rotating")
        self.assertEqual(controller.paper_rotation_stopped_count, 2)
        controller.relative_rotation_angular_velocity = 0.1
        controller._calc_relative_rotation_forces(points, 1.502, np.zeros(20))
        self.assertEqual(controller.paper_rotation_stopped_count, 0)
        controller.relative_rotation_angular_velocity = 0.0
        for now in (1.503, 1.504):
            controller._calc_relative_rotation_forces(points, now, np.zeros(20))
            self.assertEqual(controller.relative_rotation_phase, "rotating")
        controller._calc_relative_rotation_forces(points, 1.505, np.zeros(20))
        self.assertEqual(controller.relative_rotation_phase, "centering")
        self.assertAlmostEqual(controller.relative_rotation_error_rad, 0.1)

    def test_force_cap_timeout_and_cancel(self):
        controller = self.make_controller(paper_rotation_kr=100.0, relative_rotation_force_limit=0.2)
        points = controller.relative_rotation_start_fingertips
        forces = controller._calc_relative_rotation_forces(points, 1.01, np.zeros(20))
        self.assertTrue(all(np.linalg.norm(force) <= 0.2 + 1e-12 for force in forces.values()))
        forces = controller._calc_relative_rotation_forces(points, 2.01, np.zeros(20))
        self.assertEqual(controller.relative_rotation_phase, "rotation_timeout")
        self.assertTrue(all(np.all(force == 0.0) for force in forces.values()))
        controller.cancel_relative_rotation()
        self.assertFalse(controller.paper_rotation_active)


if __name__ == "__main__":
    unittest.main()
