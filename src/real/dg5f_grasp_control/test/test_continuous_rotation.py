import unittest
from pathlib import Path

import numpy as np

from dg5f_grasp_control.config import RuntimeConfig
from dg5f_grasp_control.grasp_controller import (
    BLIND_POSE_ROTATION_MIDDLE_KP_SCALE,
    BLIND_SEQUENCE_ROTATION_PD_INDICES,
    GraspController,
    _estimate_sphere_from_contacts,
)
from dg5f_grasp_control.hand_model import FINGER_JOINT_INDEX
from dg5f_grasp_control.kinematics import set_hand_side, tip_position
from dg5f_grasp_control.mujoco_gravity import MujocoGravityCompensator
from dg5f_grasp_control.poses import (
    RIGHT_HAND_BLIND_GRASP_INITIAL_POSE,
    RIGHT_HAND_BLIND_GRASP_REVERSE_ROTATION_POSE,
    RIGHT_HAND_CONTINUOUS_ROTATION_POSE,
)


class ContinuousRotationPoseSequenceTest(unittest.TestCase):
    def setUp(self):
        set_hand_side("right")

    def finish_blind_motion(self, controller, fingers, now):
        indices = np.concatenate([
            np.asarray(FINGER_JOINT_INDEX[finger], dtype=int)
            for finger in fingers
        ])
        controller.hand_qdot[indices] = 0.1
        controller._process_continuous_rotation(now)
        controller.hand_qdot[indices] = 0.0
        controller._process_continuous_rotation(now + 0.001)
        controller._process_continuous_rotation(now + 0.082)

    def test_out_of_range_sphere_hand_x_returns_to_blind_pre_rotation(self):
        for sphere_x in (0.0837876, 0.1146891):
            controller = GraspController(
                RuntimeConfig(hand_side="right"),
                log=None,
            )
            controller.apply_pose_type(6, now=1.0)
            self.assertTrue(controller.start_continuous_rotation(now=2.0))
            controller.blind_sphere_sampled_hand_x = sphere_x

            controller.step(np.zeros(20), np.zeros(20), now=2.01)

            self.assertFalse(controller.continuous_rotation_active)
            self.assertEqual(controller.state, "PRE_GRASP_POSE")
            self.assertEqual(controller.pose_type, 6)

    def test_joint_angles_no_longer_trigger_ball_drop(self):
        controller = GraspController(RuntimeConfig(hand_side="right"), log=None)
        controller.apply_pose_type(6, now=1.0)
        self.assertTrue(controller.start_continuous_rotation(now=2.0))
        controller.blind_sphere_sampled_hand_x = 0.100
        q = np.zeros(20)
        for finger in (2, 3, 4):
            q[int(FINGER_JOINT_INDEX[finger][1])] = np.deg2rad(90.0)

        controller.step(q, np.zeros(20), now=2.01)

        self.assertTrue(controller.continuous_rotation_active)

    def test_joint_rotation_test_leaves_non_thumb_j4_grasp_only(self):
        controller = GraspController(RuntimeConfig(hand_side="right"), log=None)
        controller.apply_grasp_type(5, now=1.0)
        q = RIGHT_HAND_BLIND_GRASP_INITIAL_POSE - 0.05
        controller.sync_joint_state(q)

        self.assertTrue(controller.start_joint_rotation_test(now=2.0))
        output = controller.step(q, np.zeros(20), now=2.001)

        self.assertEqual(controller.use_fingers, [1, 2, 3, 4])
        controlled = np.array([
            *FINGER_JOINT_INDEX[1],
            *FINGER_JOINT_INDEX[2][:3],
            *FINGER_JOINT_INDEX[3][:3],
            *FINGER_JOINT_INDEX[4][:3],
        ])
        np.testing.assert_allclose(
            output.inactive_pd_target[controlled],
            RIGHT_HAND_BLIND_GRASP_INITIAL_POSE[controlled],
        )
        j4 = np.array([FINGER_JOINT_INDEX[finger][3] for finger in (2, 3, 4)])
        self.assertTrue(np.all(np.isnan(output.inactive_pd_target[j4])))
        np.testing.assert_allclose(output.tau[j4], output.grasp_tau[j4])

    def test_restart_clears_stale_ball_drop_height(self):
        controller = GraspController(RuntimeConfig(hand_side="right"), log=None)
        controller.apply_pose_type(6, now=1.0)
        self.assertTrue(controller.start_continuous_rotation(now=2.0))
        controller.blind_sphere_sampled_hand_x = 0.08
        controller.step(np.zeros(20), np.zeros(20), now=2.01)
        self.assertFalse(controller.continuous_rotation_active)

        self.assertTrue(controller.start_continuous_rotation(now=3.0))
        self.assertIsNone(controller.blind_sphere_sampled_hand_x)
        controller.step(np.zeros(20), np.zeros(20), now=3.01)

        self.assertTrue(controller.continuous_rotation_active)

    def test_right_urdf_fused_tactile_tips_produce_contact_points(self):
        model_path = (
            Path(__file__).parents[3]
            / "vendor/dg5f_s_description/urdf/dg5fs_right_w_mount.urdf"
        )
        gravity = MujocoGravityCompensator(model_path)

        contacts = [
            gravity.tactile_contact_geometry(
                RIGHT_HAND_BLIND_GRASP_INITIAL_POSE, finger, 1.0, 1.0
            )
            for finger in range(1, 6)
        ]

        self.assertTrue(all(contact is not None for contact in contacts))

    def test_tactile_filter_warms_up_and_resets_for_released_finger(self):
        controller = GraspController(RuntimeConfig(hand_side="right"), log=None)
        contacts = np.zeros((5, 5))

        for x in (10.0, 100.0, 12.0, 14.0):
            contacts[4] = [x, 2.0, 3.0, 4.0, 5.0]
            controller.set_tactile_contacts(contacts)
        np.testing.assert_allclose(controller.tactile_contacts[4], 0.0)

        contacts[4] = [13.0, 2.0, 3.0, 4.0, 5.0]
        controller.set_tactile_contacts(contacts)
        np.testing.assert_allclose(
            controller.tactile_contacts[4],
            [13.0, 2.0, 3.0, 4.0, 5.0],
        )
        contacts[4] = [15.0, 2.0, 3.0, 4.0, 5.0]
        controller.set_tactile_contacts(contacts)
        self.assertAlmostEqual(controller.tactile_contacts[4, 0], 13.4)

        controller._set_continuous_rotation_phase("blind_pinky_release", 1.0)
        controller.set_tactile_contacts(contacts)
        np.testing.assert_allclose(controller.tactile_contacts[4], 0.0)
        self.assertEqual(len(controller.tactile_sample_queues[4]), 0)

        controller._set_continuous_rotation_phase("blind_pinky_regrasp", 2.0)
        for x in (20.0, 22.0, 21.0, 24.0, 23.0):
            contacts[4] = [x, 2.0, 3.0, 4.0, 5.0]
            controller.set_tactile_contacts(contacts)
        np.testing.assert_allclose(
            controller.tactile_contacts[4],
            [22.0, 2.0, 3.0, 4.0, 5.0],
        )

        contacts[4] = [0.0, 0.0, 3.0, 4.0, 5.0]
        controller.set_tactile_contacts(contacts)
        np.testing.assert_allclose(controller.tactile_contacts[4], 0.0)
        self.assertEqual(len(controller.tactile_sample_queues[4]), 0)

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

        estimate = _estimate_sphere_from_contacts(
            center + radius * directions,
            radius,
        )

        self.assertIsNotNone(estimate)
        estimated_center, fit_error = estimate
        np.testing.assert_allclose(estimated_center, center, atol=1e-12)
        self.assertAlmostEqual(fit_error, 0.0, places=12)

    def test_sphere_estimate_accepts_three_contact_points(self):
        center = np.array([0.08, -0.01, 0.12])
        radius = 0.0375

        estimated_center, fit_error = _estimate_sphere_from_contacts(
            center + radius * np.eye(3), radius
        )

        np.testing.assert_allclose(estimated_center, center, atol=1e-12)
        self.assertAlmostEqual(fit_error, 0.0, places=12)

    def test_sphere_estimate_approximates_non_intersecting_contact_spheres(self):
        center = np.array([0.08, -0.01, 0.12])
        radius = 0.0375
        directions = np.array([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [-1.0, -1.0, -1.0],
        ])
        directions /= np.linalg.norm(directions, axis=1, keepdims=True)
        points = center + radius * directions
        points[0, 0] += 0.001

        estimated_center, fit_error = _estimate_sphere_from_contacts(points, radius)

        self.assertLess(np.linalg.norm(estimated_center - center), 0.002)
        self.assertGreater(fit_error, 0.0)

    def test_normal_4f_updates_75mm_sphere_from_tactile_fk_points(self):
        controller = GraspController(RuntimeConfig(hand_side="right"), log=None)
        controller.set_blind_use_tactile_estimation(True)
        controller.apply_grasp_type(4, now=1.0)
        center = np.array([0.08, -0.01, 0.12])
        radius = 0.0375
        normals = {
            1: np.array([1.0, 0.0, 0.0]),
            2: np.array([0.0, 1.0, 0.0]),
            3: np.array([0.0, 0.0, 1.0]),
            4: np.array([-1.0, 0.0, 0.0]),
            5: np.array([0.0, -1.0, 0.0]),
        }
        controller.set_tactile_contact_geometry_provider(
            lambda _q, finger, _x, _y: (
                center - radius * normals[finger],
                normals[finger],
            )
        )
        contacts = np.ones((5, 5), dtype=np.float64)
        for _ in range(5):
            controller.set_tactile_contacts(contacts)

        controller.step(
            np.zeros(20),
            np.zeros(20),
            now=1.01,
        )

        self.assertTrue(controller.blind_sphere_estimate_valid)
        np.testing.assert_allclose(controller.blind_sphere_center, center)
        self.assertAlmostEqual(controller.blind_sphere_effective_radius_m, 0.0375)

    def test_sphere_skip_reason_is_logged_without_spam(self):
        logs = []
        controller = GraspController(
            RuntimeConfig(hand_side="right"),
            log=logs.append,
        )
        controller.set_blind_use_tactile_estimation(True)

        controller.step(np.zeros(20), np.zeros(20), now=1.0)
        controller.step(np.zeros(20), np.zeros(20), now=1.1)
        controller.step(np.zeros(20), np.zeros(20), now=2.1)

        sphere_logs = [line for line in logs if line.startswith("[BLIND_SPHERE]")]
        self.assertEqual(len(sphere_logs), 2)
        self.assertIn("state=NORMAL_POSE", sphere_logs[0])
        self.assertIn("estimate_fingers=2/3", sphere_logs[0])
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

    def test_blind_rotation_also_starts_from_active_five_finger_grasp(self):
        controller = GraspController(RuntimeConfig(hand_side="right"), log=None)
        controller.apply_pose_type(6, now=1.0)
        controller.apply_grasp_type(5, now=1.5)

        self.assertTrue(controller.start_continuous_rotation(now=2.0))
        self.assertEqual(controller.use_fingers, [1, 2, 3, 4, 5])
        self.assertEqual(controller.continuous_rotation_phase, "blind_grasp_settle")

    def test_blind_sequence_rotates_without_middle_or_pinky(self):
        logs = []
        controller = GraspController(RuntimeConfig(hand_side="right"), log=logs.append)
        controller.apply_pose_type(6, now=1.0)
        initial = RIGHT_HAND_BLIND_GRASP_INITIAL_POSE.copy()
        controller.sync_joint_state(initial)
        controller.start_continuous_rotation(now=2.0)

        controller._process_continuous_rotation(2.501)
        self.assertEqual(
            controller.continuous_rotation_phase,
            "blind_forward_sphere_estimate",
        )
        for x in (0.0974, 0.0975, 0.0973, 0.0976, 0.0972):
            controller.set_ui_sphere_center_hand([x, 0.0, 0.05])
        controller._process_continuous_rotation(3.002)

        middle = np.asarray(FINGER_JOINT_INDEX[3], dtype=int)
        target = controller.continuous_rotation_pose_target
        self.assertEqual(controller.use_fingers, [1, 2, 4, 5])
        self.assertEqual(controller.continuous_rotation_phase, "blind_middle_release")
        self.assertTrue(np.all(np.isfinite(target[middle])))

        self.finish_blind_motion(controller, (3,), 3.003)
        self.assertEqual(controller.use_fingers, [1, 2, 3, 4, 5])
        self.assertEqual(controller.continuous_rotation_phase, "blind_middle_regrasp")

        self.finish_blind_motion(controller, (3,), 3.184)
        self.assertEqual(controller.use_fingers, [1, 3, 5])
        self.assertEqual(controller.continuous_rotation_phase, "blind_index_ring_release")

        pinky = np.asarray(FINGER_JOINT_INDEX[5], dtype=int)
        measured = initial.copy()
        measured[pinky[1:]] = [1.21, 0.91, 0.37]
        controller.sync_joint_state(measured)
        controller._start_blind_pinky_release(4.0)
        target = controller.continuous_rotation_pose_target
        self.assertEqual(controller.use_fingers, [1, 2, 3, 4])
        self.assertEqual(controller.continuous_rotation_phase, "blind_pinky_release")
        np.testing.assert_allclose(
            target[pinky],
            measured[pinky] + np.deg2rad([-5.0, 0.0, -5.0, 0.0]),
        )

        self.finish_blind_motion(controller, (5,), 4.001)
        self.assertEqual(controller.use_fingers, [1, 2, 4])
        self.assertEqual(controller.continuous_rotation_phase, "blind_pose_rotation")

        moving_qdot = np.zeros(20)
        moving_qdot[:16] = 0.1
        output = controller.step(
            controller.pose_type_targets[6], moving_qdot, 4.182
        )
        controlled = np.array([
            *FINGER_JOINT_INDEX[1],
            *FINGER_JOINT_INDEX[2][:3],
            *FINGER_JOINT_INDEX[4][:3],
        ])
        np.testing.assert_allclose(
            output.inactive_pd_target[controlled],
            RIGHT_HAND_BLIND_GRASP_INITIAL_POSE[controlled],
        )
        middle_target = RIGHT_HAND_BLIND_GRASP_INITIAL_POSE[middle].copy()
        middle_target[1:3] -= np.deg2rad(2.0)
        np.testing.assert_allclose(
            output.inactive_pd_target[middle],
            middle_target,
        )
        j4 = np.array([FINGER_JOINT_INDEX[finger][3] for finger in (2, 4)])
        self.assertTrue(np.all(np.isnan(output.inactive_pd_target[j4])))
        np.testing.assert_allclose(output.tau[j4], output.grasp_tau[j4])
        np.testing.assert_allclose(output.inactive_pd_target[pinky], target[pinky])

        controller._start_blind_regrasp(3, now=4.3)
        self.assertEqual(controller.use_fingers, [1, 2, 3, 4, 5])

    def test_blind_direction_change_during_release_switches_same_release(self):
        controller = GraspController(RuntimeConfig(hand_side="right"), log=None)
        controller.apply_pose_type(6, now=1.0)
        controller.sync_joint_state(RIGHT_HAND_BLIND_GRASP_INITIAL_POSE)
        controller.start_continuous_rotation(now=2.0)

        controller._start_blind_pinky_release(now=2.5)
        controller.request_blind_direction_change()
        controller._process_continuous_rotation(2.55)

        self.assertEqual(controller.blind_rotation_direction, -1)
        self.assertEqual(controller.continuous_rotation_phase, "blind_pinky_release")
        self.assertEqual(controller.use_fingers, [1, 2, 3, 4])

        controller.blind_rotation_direction = 1
        controller._start_blind_index_ring_release(now=3.0)
        controller.request_blind_direction_change()
        controller._process_continuous_rotation(3.01)

        self.assertEqual(controller.blind_rotation_direction, -1)
        self.assertEqual(
            controller.continuous_rotation_phase,
            "blind_index_ring_release",
        )

    def test_direction_change_defers_center_estimate_until_reverse_cycle_end(self):
        for initial_direction in (1, -1):
            controller = GraspController(RuntimeConfig(hand_side="right"), log=None)
            controller.apply_pose_type(6, now=1.0)
            controller.start_continuous_rotation(now=2.0)
            controller.blind_rotation_direction = initial_direction
            controller._start_blind_thumb_release(now=3.0)

            controller.request_blind_direction_change()
            controller._process_continuous_rotation(3.01)

            self.assertEqual(controller.blind_rotation_direction, -initial_direction)
            self.assertEqual(controller.continuous_rotation_phase, "blind_thumb_release")

    def test_direction_change_during_pose_rotation_reverses_in_place(self):
        controller = GraspController(RuntimeConfig(hand_side="right"), log=None)
        controller.apply_pose_type(6, now=1.0)
        controller.start_continuous_rotation(now=2.0)
        controller._start_blind_pinky_release(now=2.1)
        controller._start_blind_pose_rotation(2.2)

        controller.request_blind_direction_change()
        controller._process_continuous_rotation(2.25)

        self.assertEqual(controller.blind_rotation_direction, -1)
        self.assertEqual(controller.continuous_rotation_phase, "blind_pose_rotation")
        self.assertEqual(controller.continuous_rotation_phase_started_at, 2.25)
        self.assertEqual(controller.use_fingers, [1, 2, 4])
        output = controller.step(np.zeros(20), np.zeros(20), now=2.251)
        controlled = np.array([
            *FINGER_JOINT_INDEX[1],
            *FINGER_JOINT_INDEX[2][:3],
            *FINGER_JOINT_INDEX[4][:3],
        ])
        np.testing.assert_allclose(
            output.inactive_pd_target[controlled],
            RIGHT_HAND_BLIND_GRASP_REVERSE_ROTATION_POSE[controlled],
        )
        middle = np.asarray(FINGER_JOINT_INDEX[3], dtype=int)
        middle_target = RIGHT_HAND_BLIND_GRASP_REVERSE_ROTATION_POSE[middle].copy()
        middle_target[1:3] -= np.deg2rad(2.0)
        np.testing.assert_allclose(output.inactive_pd_target[middle], middle_target)
        j4 = np.array([FINGER_JOINT_INDEX[finger][3] for finger in (2, 4)])
        self.assertTrue(np.all(np.isnan(output.inactive_pd_target[j4])))

    def test_blind_direction_change_after_regrasp_repeats_same_group(self):
        controller = GraspController(RuntimeConfig(hand_side="right"), log=None)
        controller.apply_pose_type(6, now=1.0)
        controller.start_continuous_rotation(now=2.0)
        controller.continuous_rotation_pose_target = (
            RIGHT_HAND_BLIND_GRASP_INITIAL_POSE.copy()
        )

        controller._start_blind_regrasp(3, now=2.5)
        controller.request_blind_direction_change()
        self.finish_blind_motion(controller, (5,), 2.501)

        self.assertEqual(controller.continuous_rotation_phase, "blind_pinky_release")
        self.assertEqual(controller.blind_rotation_direction, -1)

    def test_reverse_estimates_after_middle_regrasp(self):
        controller = GraspController(RuntimeConfig(hand_side="right"), log=None)
        controller.continuous_rotation_active = True
        controller.blind_rotation_direction = -1
        controller.continuous_rotation_pose_target = np.zeros(20)
        controller._start_blind_regrasp(3, now=2.0)

        self.assertEqual(
            controller.continuous_rotation_phase,
            "blind_reverse_pinky_regrasp",
        )
        self.finish_blind_motion(controller, (5,), 2.001)
        self.assertEqual(controller.continuous_rotation_phase, "blind_thumb_release")

        controller._start_blind_regrasp(0, now=3.0)
        self.assertEqual(controller.continuous_rotation_phase, "blind_reverse_sphere_estimate")
        for x in (0.0966, 0.0970, 0.0974, 0.0968, 0.0972):
            controller.set_ui_sphere_center_hand([x, 0.0, 0.05])
        controller._process_continuous_rotation(3.501)

        self.assertFalse(controller.blind_thumb_lift_pending)
        self.assertAlmostEqual(controller.blind_sphere_sampled_hand_x, 0.097)
        self.assertEqual(controller.continuous_rotation_phase, "blind_pinky_release")

    def test_release_is_relative_to_current_joint_state(self):
        controller = GraspController(RuntimeConfig(hand_side="right"), log=None)
        measured = np.linspace(-0.4, 0.4, 20)
        controller.sync_joint_state(measured)

        for direction in (1, -1):
            controller.blind_rotation_direction = direction
            for finger, delta_deg in (
                (2, [-2.0, 2.0, 2.0] if direction > 0 else [-15.0, 2.0, 2.0]),
                (3, [0.0, -5.0, 0.0]),
                (4, [-15.0, 2.0, 2.0] if direction > 0 else [-2.0, 2.0, 2.0]),
            ):
                indices = np.asarray(FINGER_JOINT_INDEX[finger], dtype=int)
                target = controller._blind_release_target(finger)
                self.assertAlmostEqual(
                    target[0],
                    (
                        measured[indices[0]] - np.deg2rad(
                            11.4 if finger == 4 else 19.0
                        )
                        if direction > 0
                        else measured[indices[0]] + np.deg2rad(
                            13.3 if finger == 2 else 19.0
                        )
                    ),
                )
                np.testing.assert_allclose(
                    target[1:4],
                    measured[indices[1:4]] + np.deg2rad(delta_deg),
                )

    def test_forward_j1_release_scales_with_estimated_diameter(self):
        controller = GraspController(RuntimeConfig(hand_side="right"), log=None)
        measured = np.linspace(-0.4, 0.4, 20)
        controller.sync_joint_state(measured)
        for diameter_mm, expected_deg in (
            (45.0, 8.0),
            (55.0, 8.0),
            (65.0, 13.5),
            (75.0, 19.0),
            (85.0, 19.0),
        ):
            controller.sphere_estimator.radius = diameter_mm / 2000.0
            self.assertAlmostEqual(
                controller._blind_finger_j1_release_deg(), expected_deg
            )
            for finger in (2, 3, 4):
                indices = np.asarray(FINGER_JOINT_INDEX[finger], dtype=int)
                target = controller._blind_release_target(finger)
                self.assertAlmostEqual(
                    target[0],
                    measured[indices[0]] - np.deg2rad(
                        0.6 * expected_deg if finger == 4 else expected_deg
                    ),
                )

    def test_reverse_index_j1_moves_seventy_percent_of_ring(self):
        controller = GraspController(RuntimeConfig(hand_side="right"), log=None)
        measured = np.linspace(-0.4, 0.4, 20)
        controller.sync_joint_state(measured)
        controller.blind_rotation_direction = -1
        for diameter_mm, release_deg in ((55.0, 8.0), (65.0, 13.5), (75.0, 19.0)):
            controller.sphere_estimator.radius = diameter_mm / 2000.0
            index = np.asarray(FINGER_JOINT_INDEX[2], dtype=int)
            ring = np.asarray(FINGER_JOINT_INDEX[4], dtype=int)
            index_target = controller._blind_release_target(2)
            ring_target = controller._blind_release_target(4)
            self.assertAlmostEqual(
                index_target[0] - measured[index[0]], np.deg2rad(0.7 * release_deg)
            )
            self.assertAlmostEqual(
                ring_target[0] - measured[ring[0]], np.deg2rad(release_deg)
            )

    def test_reverse_thumb_release_uses_supplied_target(self):
        controller = GraspController(RuntimeConfig(hand_side="right"), log=None)
        controller.blind_rotation_direction = -1
        thumb = np.asarray(FINGER_JOINT_INDEX[1], dtype=int)

        controller._start_blind_thumb_release(now=2.0)
        target = controller.continuous_rotation_pose_target[thumb]

        np.testing.assert_allclose(
            target,
            [0.3238, -1.3102, -np.deg2rad(4.0), -np.deg2rad(4.0)],
        )

    def test_forward_thumb_release_uses_supplied_target(self):
        controller = GraspController(RuntimeConfig(hand_side="right"), log=None)
        thumb = np.asarray(FINGER_JOINT_INDEX[1], dtype=int)
        middle = np.asarray(FINGER_JOINT_INDEX[3], dtype=int)
        ring = np.asarray(FINGER_JOINT_INDEX[4], dtype=int)
        measured = np.linspace(-0.2, 0.2, 20)
        controller.sync_joint_state(measured)

        controller._start_blind_thumb_release(now=2.0)

        self.assertEqual(controller.use_fingers, [2, 5])
        np.testing.assert_allclose(
            controller.continuous_rotation_pose_target[thumb],
            [
                0.1168,
                -1.6596,
                measured[thumb[2]] - np.deg2rad(4.0),
                measured[thumb[3]] - np.deg2rad(4.0),
            ],
        )
        np.testing.assert_allclose(
            controller.continuous_rotation_pose_target[middle],
            measured[middle],
        )
        np.testing.assert_allclose(
            controller.continuous_rotation_pose_target[ring],
            measured[ring],
        )
        controller._start_blind_regrasp(2, now=2.2)
        self.assertEqual(controller.use_fingers, [1, 2, 3, 4, 5])

    def test_reverse_pinky_release_offsets_current_grasp_without_snapshot(self):
        controller = GraspController(RuntimeConfig(hand_side="right"), log=None)
        controller.blind_rotation_direction = -1
        pinky = np.asarray(FINGER_JOINT_INDEX[5], dtype=int)
        measured = np.zeros(20)
        measured[pinky] = [0.61, 1.22, 0.93, 0.38]
        controller.sync_joint_state(measured)

        controller._start_blind_pinky_release(now=2.0)
        target = controller.continuous_rotation_pose_target[pinky]

        np.testing.assert_allclose(
            target,
            measured[pinky] + np.deg2rad([-5.0, 0.0, -5.0, 0.0]),
        )

    def test_forward_pinky_release_offsets_saved_initial_grasp(self):
        controller = GraspController(RuntimeConfig(hand_side="right"), log=None)
        pinky = np.asarray(FINGER_JOINT_INDEX[5], dtype=int)
        measured = np.zeros(20)
        measured[pinky] = [0.70, 1.22, 0.93, 0.38]
        controller.sync_joint_state(measured)
        initial_grasp = measured.copy()
        initial_grasp[pinky] = [0.68, 1.27, 0.98, 0.91]
        controller.blind_initial_sphere_q = initial_grasp

        controller._start_blind_pinky_release(now=2.0)
        target = controller.continuous_rotation_pose_target[pinky]

        np.testing.assert_allclose(
            target,
            initial_grasp[pinky] + np.deg2rad([-5.0, 0.0, -5.0, 0.0]),
        )

    def test_pinky_regrasp_uses_blind_grasp_without_pose_pd(self):
        controller = GraspController(RuntimeConfig(hand_side="right"), log=None)
        controller.continuous_rotation_active = True
        controller.continuous_rotation_pose_target = np.zeros(20)
        controller._start_blind_regrasp(3, now=2.0)
        pinky = np.asarray(FINGER_JOINT_INDEX[5], dtype=int)

        output = controller.step(np.zeros(20), np.zeros(20), now=2.001)

        self.assertEqual(controller.use_fingers, [1, 2, 3, 4, 5])
        self.assertEqual(controller.policy.cfg.alpha1, controller.cfg.alpha1)
        self.assertTrue(np.all(np.isnan(output.inactive_pd_target[pinky])))
        np.testing.assert_allclose(output.inactive_pd[pinky], 0.0)
        self.assertTrue(np.any(np.abs(output.grasp_tau[pinky]) > 0.0))
        middle = np.asarray(FINGER_JOINT_INDEX[3], dtype=int)
        self.assertTrue(np.all(np.isnan(output.inactive_pd_target[middle])))
        self.assertTrue(np.any(np.abs(output.grasp_tau[middle]) > 0.0))
        self.finish_blind_motion(controller, (5,), 2.201)
        self.assertEqual(
            controller.continuous_rotation_phase,
            "blind_forward_sphere_estimate",
        )

    def test_reverse_pose_rotation_uses_supplied_non_pinky_target(self):
        controller = GraspController(RuntimeConfig(hand_side="right"), log=None)
        np.testing.assert_allclose(
            RIGHT_HAND_BLIND_GRASP_REVERSE_ROTATION_POSE,
            [
                0.1600, -1.5956, -0.0031, 0.9427,
                -0.6014, 0.8062, 0.6217, 0.4442,
                -0.2934, 0.3152, 1.4038, -0.4492,
                -0.0522, 0.6980, 0.3691, 0.5243,
                0.8022, 1.2627, 0.7718, 0.5074,
            ],
        )
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

        controlled = np.array([
            *FINGER_JOINT_INDEX[1],
            *FINGER_JOINT_INDEX[2][:3],
            *FINGER_JOINT_INDEX[4][:3],
        ])
        np.testing.assert_allclose(
            output.inactive_pd_target[controlled],
            RIGHT_HAND_BLIND_GRASP_REVERSE_ROTATION_POSE[controlled],
        )

    def test_low_world_sphere_z_uses_direction_specific_thumb_lift_once(self):
        expected_by_direction = {
            1: [0.1136, -1.6558, -np.deg2rad(4.0), -np.deg2rad(4.0)],
            -1: [0.3506, -1.3271, -np.deg2rad(4.0), -np.deg2rad(4.0)],
        }
        for direction, expected in expected_by_direction.items():
            controller = GraspController(RuntimeConfig(hand_side="right"), log=None)
            controller.set_blind_use_tactile_estimation(True)
            controller.apply_pose_type(6, now=1.0)
            controller.start_continuous_rotation(now=2.0)
            controller.blind_rotation_direction = direction
            controller.blind_thumb_lift_pending = True
            controller._start_blind_thumb_release(3.0)

            thumb = np.asarray(FINGER_JOINT_INDEX[1], dtype=int)
            np.testing.assert_allclose(
                controller.continuous_rotation_pose_target[thumb], expected
            )
            self.assertFalse(controller.blind_thumb_lift_pending)
            self.assertEqual(controller.continuous_rotation_phase, "blind_thumb_release")

    def test_forward_sphere_estimate_uses_hand_x_directly(self):
        controller = GraspController(RuntimeConfig(hand_side="right"), log=None)
        controller.apply_pose_type(6, now=1.0)
        controller.start_continuous_rotation(now=2.0)
        controller.continuous_rotation_pose_target = (
            RIGHT_HAND_BLIND_GRASP_INITIAL_POSE.copy()
        )
        controller._start_blind_forward_sphere_estimate(now=2.1)
        first_samples = (0.0970, 0.0974, 0.0978, 0.0972, 0.0976)
        for x in first_samples:
            controller.set_ui_sphere_center_hand([x, 0.0, 0.05])
        controller._process_continuous_rotation(2.601)
        self.assertFalse(controller.blind_thumb_lift_pending)
        self.assertAlmostEqual(
            controller.blind_sphere_sampled_hand_x,
            np.mean(first_samples),
        )

        controller._start_blind_forward_sphere_estimate(now=3.0)
        for x in (0.0966, 0.0970, 0.0974, 0.0968, 0.0972):
            controller.set_ui_sphere_center_hand([x, 0.0, 0.05])
        controller._process_continuous_rotation(3.501)
        self.assertFalse(controller.blind_thumb_lift_pending)
        self.assertAlmostEqual(
            controller.blind_sphere_sampled_hand_x,
            0.097,
        )

    def test_thumb_release_runs_fk_from_current_j3_j4_minus_four_degrees(self):
        controller = GraspController(RuntimeConfig(hand_side="right"), log=None)
        controller.blind_reverse_base_q = np.zeros(20)
        thumb = np.asarray(FINGER_JOINT_INDEX[1], dtype=int)
        measured = np.zeros(20)
        measured[thumb[2:4]] = [0.31, 0.72]
        controller.sync_joint_state(measured)
        controller.blind_sphere_hand_x_samples = [0.095]
        self.assertTrue(controller._finish_blind_sphere_sampling())
        base_target = controller._blind_thumb_release_target()
        base_target[2:4] = measured[thumb[2:4]] - np.deg2rad(4.0)
        q = np.zeros(20)
        q[thumb] = base_target
        base_tip = tip_position(q, 1)

        controller._start_blind_thumb_release(now=1.0)
        target = controller.continuous_rotation_pose_target[thumb]
        q[thumb] = target
        adjusted_tip = tip_position(q, 1)

        self.assertEqual(target[0], base_target[0])
        self.assertAlmostEqual(adjusted_tip[0], base_tip[0] - 0.004, delta=5e-4)
        np.testing.assert_allclose(adjusted_tip[1:], base_tip[1:], atol=5e-4)
        self.assertEqual(controller.continuous_rotation_phase, "blind_thumb_release")

    def test_fk_target_x_drops_one_mm_when_local_x_faces_world_down(self):
        controller = GraspController(RuntimeConfig(hand_side="right"), log=None)
        upward = np.array([[0.0, 0.0, -1.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]])
        downward = np.array([[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]])

        controller.set_rotation_hand_to_world(upward)
        self.assertAlmostEqual(controller._blind_fk_sphere_target_hand_x(), 0.100)
        controller.set_rotation_hand_to_world(downward)
        self.assertAlmostEqual(controller._blind_fk_sphere_target_hand_x(), 0.099)

    def test_fk_target_x_interpolates_from_ball_diameter(self):
        controller = GraspController(
            RuntimeConfig(
                hand_side="right",
                blind_fk_sphere_target_hand_x_m=0.097,
                blind_fk_downward_target_x_offset_m=-0.001,
            ),
            log=None,
        )
        demo = np.array([
            [0.4695, 0.0, -0.8829],
            [0.0, 1.0, 0.0],
            [0.8829, 0.0, 0.4695],
        ])
        downward = np.array([
            [0.0, 0.0, 1.0],
            [0.0, 1.0, 0.0],
            [-1.0, 0.0, 0.0],
        ])
        controller.set_rotation_hand_to_world(demo)

        for diameter_mm, target_x in ((55.0, 0.099), (65.0, 0.098), (75.0, 0.097)):
            controller.sphere_estimator.radius = diameter_mm / 2000.0
            self.assertAlmostEqual(
                controller._blind_fk_sphere_target_hand_x(), target_x
            )

        controller.sphere_estimator.radius = 0.055 / 2.0
        controller.set_rotation_hand_to_world(downward)
        self.assertAlmostEqual(controller._blind_fk_sphere_target_hand_x(), 0.098)

    def test_fk_height_offset_applies_to_index_and_ring_releases(self):
        controller = GraspController(RuntimeConfig(hand_side="right"), log=None)
        controller.blind_reverse_base_q = np.zeros(20)
        controller.sync_joint_state(RIGHT_HAND_BLIND_GRASP_INITIAL_POSE)
        controller.blind_sphere_sampled_hand_x = 0.095
        base_targets = {
            finger: controller._blind_release_target(finger)
            for finger in (2, 4)
        }

        middle = np.asarray(FINGER_JOINT_INDEX[3], dtype=int)
        controller._start_blind_index_ring_release(now=2.0)
        adjusted_targets = {}
        for finger in (2, 4):
            indices = np.asarray(FINGER_JOINT_INDEX[finger], dtype=int)
            adjusted_targets[finger] = (
                controller.continuous_rotation_pose_target[indices].copy()
            )

        for finger in (2, 4):
            indices = np.asarray(FINGER_JOINT_INDEX[finger], dtype=int)
            q = np.zeros(20)
            q[indices] = base_targets[finger]
            base_tip = tip_position(q, finger)
            q[indices] = adjusted_targets[finger]
            adjusted_tip = tip_position(q, finger)
            np.testing.assert_allclose(
                adjusted_tip[1:],
                base_tip[1:],
                atol=5e-4,
            )
            self.assertAlmostEqual(
                adjusted_tip[0],
                base_tip[0] - 0.004,
                delta=5e-4,
            )

        controller._start_blind_pinky_release(now=3.0)
        np.testing.assert_allclose(
            controller.continuous_rotation_pose_target[middle],
            RIGHT_HAND_BLIND_GRASP_INITIAL_POSE[middle],
        )

    def test_fk_height_offset_uses_maximum_reachable_distance(self):
        controller = GraspController(RuntimeConfig(hand_side="right"), log=None)
        controller.sync_joint_state(RIGHT_HAND_BLIND_GRASP_INITIAL_POSE)
        controller.blind_sphere_sampled_hand_x = 0.110

        for finger in (1, 2, 4):
            base = (
                controller._blind_thumb_release_target()
                if finger == 1
                else controller._blind_release_target(finger)
            )
            adjusted = (
                controller._blind_fk_thumb_release_target()
                if finger == 1
                else controller._blind_fk_release_target(finger, base)
            )
            indices = np.asarray(FINGER_JOINT_INDEX[finger], dtype=int)
            base_q = np.zeros(20)
            adjusted_q = np.zeros(20)
            base_q[indices] = base
            adjusted_q[indices] = adjusted
            base_tip = tip_position(base_q, finger)
            adjusted_tip = tip_position(adjusted_q, finger)

            self.assertAlmostEqual(
                adjusted_tip[0] - base_tip[0], 0.004, delta=5e-4
            )
            np.testing.assert_allclose(
                adjusted_tip[1:], base_tip[1:], atol=5e-4
            )

    def test_middle_release_ignores_fk_height(self):
        controller = GraspController(RuntimeConfig(hand_side="right"), log=None)
        controller.sync_joint_state(RIGHT_HAND_BLIND_GRASP_INITIAL_POSE)
        controller.blind_sphere_sampled_hand_x = 0.0945
        middle = np.asarray(FINGER_JOINT_INDEX[3], dtype=int)
        base_target = controller._blind_release_target(3)

        controller._start_blind_middle_release(now=2.0)

        np.testing.assert_allclose(
            controller.continuous_rotation_pose_target[middle],
            base_target,
        )

    def test_fk_thumb_ik_clamps_invalid_j1_target_to_urdf_limit(self):
        controller = GraspController(
            RuntimeConfig(hand_side="right", blind_thumb_j1_target_rad=1.0),
            log=None,
        )
        controller.blind_sphere_sampled_hand_x = 0.095

        target = controller._blind_fk_thumb_release_target()

        self.assertEqual(target[0], 0.9250245036)

    def test_blind_release_advances_after_joint_velocity_stays_low(self):
        controller = GraspController(RuntimeConfig(hand_side="right"), log=None)
        controller.continuous_rotation_active = True
        controller._start_blind_pinky_release(now=1.0)
        pinky = np.asarray(FINGER_JOINT_INDEX[5], dtype=int)

        controller._process_continuous_rotation(1.001)
        controller.hand_qdot[pinky] = 0.1
        controller._process_continuous_rotation(1.002)
        self.assertEqual(controller.continuous_rotation_phase, "blind_pinky_release")

        controller.hand_qdot[pinky] = 0.0
        controller._process_continuous_rotation(1.003)
        controller._process_continuous_rotation(1.084)

        self.assertEqual(controller.continuous_rotation_phase, "blind_pose_rotation")

    def test_regrasp_advances_after_eighty_percent_speed_drop(self):
        controller = GraspController(RuntimeConfig(hand_side="right"), log=None)
        thumb = np.asarray(FINGER_JOINT_INDEX[1], dtype=int)
        controller._set_continuous_rotation_phase("blind_thumb_regrasp", 1.0)
        controller.hand_qdot[thumb] = 0.2
        self.assertFalse(controller._blind_motion_ready(1.01, (1,), 1.0))
        controller.hand_qdot[thumb] = 0.039
        self.assertFalse(controller._blind_motion_ready(1.09, (1,), 1.0))

        controller.blind_sphere_sampled_hand_x = 0.097
        controller._set_continuous_rotation_phase("blind_thumb_regrasp", 2.0)
        controller.hand_qdot[thumb] = 0.2
        self.assertFalse(controller._blind_motion_ready(2.01, (1,), 1.0))
        controller.hand_qdot[thumb] = 0.039
        self.assertTrue(controller._blind_motion_ready(2.09, (1,), 1.0))

    def test_pose_rotation_aligns_all_four_fingers_before_pinky_regrasp(self):
        controller = GraspController(RuntimeConfig(hand_side="right"), log=None)
        controller.apply_grasp_type(5, now=0.5, internal=True)
        controller.continuous_rotation_active = True
        controller.continuous_rotation_pose_target = np.zeros(20)
        controller._start_blind_pose_rotation(now=1.0)

        self.finish_blind_motion(controller, (1, 2, 3, 4), 1.001)
        self.assertEqual(controller.continuous_rotation_phase, "blind_pose_alignment")

        output = controller.step(np.zeros(20), np.zeros(20), now=1.085)
        np.testing.assert_allclose(
            output.inactive_pd_target[:16],
            RIGHT_HAND_BLIND_GRASP_INITIAL_POSE[:16],
        )

        self.finish_blind_motion(controller, (1, 2, 3, 4), 1.086)
        self.assertEqual(controller.continuous_rotation_phase, "blind_pinky_regrasp")

    def test_pose_rotation_uses_weaker_middle_kp_only(self):
        controller = GraspController(
            RuntimeConfig(
                hand_side="right",
                blind_grasp_pre_rotation_pose_kp=1.0,
                blind_grasp_pre_rotation_pose_kd=0.0,
                pre_rotation_pose_pd_limit=10.0,
            ),
            log=None,
        )
        controller.apply_grasp_type(5, now=0.5, internal=True)
        controller.continuous_rotation_active = True
        controller.continuous_rotation_pose_target = np.zeros(20)
        controller._start_blind_pose_rotation(now=1.0)
        middle = np.asarray(FINGER_JOINT_INDEX[3], dtype=int)

        output = controller.step(np.zeros(20), np.zeros(20), now=1.001)

        np.testing.assert_allclose(
            output.inactive_pd[middle],
            BLIND_POSE_ROTATION_MIDDLE_KP_SCALE
            * controller.continuous_rotation_pose_target[middle],
        )

    def test_pose_rotation_uses_first_five_finger_sphere_joint_snapshot(self):
        controller = GraspController(RuntimeConfig(hand_side="right"), log=None)
        controller.apply_pose_type(6, now=1.0)
        controller.apply_grasp_type(5, now=1.1)
        captured = RIGHT_HAND_BLIND_GRASP_INITIAL_POSE + np.linspace(-0.02, 0.02, 20)
        controller.sphere_estimator.radius = 0.0325
        controller.sphere_estimator.initial_center = np.array([0.1, 0.0, 0.08])
        controller.step(captured, np.zeros(20), now=1.2)
        np.testing.assert_allclose(controller.blind_initial_sphere_q, captured)

        controller.step(captured + 0.1, np.zeros(20), now=1.3)
        np.testing.assert_allclose(controller.blind_initial_sphere_q, captured)

        controller.continuous_rotation_pose_target = np.zeros(20)
        controller._start_blind_pose_rotation(now=1.4)
        controlled = BLIND_SEQUENCE_ROTATION_PD_INDICES
        output = controller.step(captured, np.zeros(20), now=1.41)
        np.testing.assert_allclose(
            output.inactive_pd_target[controlled], captured[controlled]
        )

        controller._start_blind_pose_alignment(now=1.5)
        output = controller.step(captured, np.zeros(20), now=1.51)
        np.testing.assert_allclose(output.inactive_pd_target[:16], captured[:16])

        controller.apply_grasp_type(5, now=2.0)
        self.assertIsNone(controller.blind_initial_sphere_q)

    def test_first_forward_thumb_regrasp_captures_reverse_base_pose(self):
        controller = GraspController(RuntimeConfig(hand_side="right"), log=None)
        controller.apply_pose_type(6, now=1.0)
        controller.continuous_rotation_active = True
        controller.continuous_rotation_pose_target = np.zeros(20)
        controller._start_blind_regrasp(2, now=1.1)
        captured = RIGHT_HAND_BLIND_GRASP_INITIAL_POSE + np.linspace(-0.01, 0.01, 20)
        controller.sync_joint_state(captured)

        controller._process_continuous_rotation(1.381)

        np.testing.assert_allclose(controller.blind_reverse_base_q, captured)
        self.assertEqual(controller.continuous_rotation_phase, "blind_pinky_release")
        controller.blind_rotation_direction = -1
        np.testing.assert_allclose(controller._blind_rotation_pose_target(), captured)

    def test_first_forward_release_skips_x_feedback_until_reverse_base_exists(self):
        controller = GraspController(RuntimeConfig(hand_side="right"), log=None)
        controller.sync_joint_state(RIGHT_HAND_BLIND_GRASP_INITIAL_POSE)
        controller.blind_sphere_sampled_hand_x = 0.095
        expected = {
            finger: controller._blind_release_target(finger)
            for finger in (2, 4)
        }

        controller._start_blind_index_ring_release(now=1.0)

        for finger in (2, 4):
            indices = np.asarray(FINGER_JOINT_INDEX[finger], dtype=int)
            np.testing.assert_allclose(
                controller.continuous_rotation_pose_target[indices],
                expected[finger],
            )

    def test_reverse_estimate_advances_after_five_stable_centers(self):
        controller = GraspController(RuntimeConfig(hand_side="right"), log=None)
        controller.continuous_rotation_active = True
        controller.blind_rotation_direction = -1
        controller.continuous_rotation_pose_target = np.zeros(20)
        controller._start_blind_regrasp(0, now=2.0)
        controller._process_continuous_rotation(2.001)
        centers = (
            [0.0500, 0.0000, 0.1360],
            [0.0505, 0.0002, 0.1365],
            [0.0498, -0.0003, 0.1358],
            [0.0502, 0.0001, 0.1362],
            [0.0499, -0.0001, 0.1359],
        )
        for center in centers[:4]:
            controller.set_ui_sphere_center_hand(center)

        controller._process_continuous_rotation(2.09)
        self.assertEqual(
            controller.continuous_rotation_phase,
            "blind_reverse_sphere_estimate",
        )

        controller.set_ui_sphere_center_hand(centers[4])
        controller._process_continuous_rotation(2.091)

        self.assertEqual(controller.continuous_rotation_phase, "blind_pinky_release")

    def test_reverse_estimate_survives_pinky_regrasp(self):
        controller = GraspController(RuntimeConfig(hand_side="right"), log=None)
        controller.blind_rotation_direction = -1
        controller.continuous_rotation_pose_target = np.zeros(20)
        controller.blind_sphere_sampled_hand_x = 0.093

        controller._start_blind_regrasp(3, now=2.0)

        self.assertEqual(
            controller.continuous_rotation_phase,
            "blind_reverse_pinky_regrasp",
        )
        self.assertEqual(controller.blind_sphere_sampled_hand_x, 0.093)

        controller._start_blind_forward_sphere_estimate(now=3.0)
        self.assertIsNone(controller.blind_sphere_sampled_hand_x)

    def test_one_thumb_missed_rotation_uses_thumb_down_pose(self):
        controller = GraspController(RuntimeConfig(hand_side="right"), log=None)
        controller.set_blind_use_tactile_estimation(True)
        controller.apply_pose_type(6, now=1.0)
        controller.start_continuous_rotation(now=2.0)
        no_contacts = np.zeros((5, 5))

        controller._set_continuous_rotation_phase("blind_pose_rotation", 3.0)
        controller.set_tactile_contacts(no_contacts)
        controller._set_continuous_rotation_phase("blind_pinky_regrasp", 3.3)

        controller.blind_thumb_lift_pending = False
        controller._start_blind_thumb_release(now=5.0)
        thumb = np.asarray(FINGER_JOINT_INDEX[1], dtype=int)
        np.testing.assert_allclose(
            controller.continuous_rotation_pose_target[thumb],
            [0.0799, -1.6336, -np.deg2rad(4.0), -np.deg2rad(4.0)],
        )
        controller.blind_rotation_direction = -1
        controller._start_blind_thumb_release(now=6.0)
        np.testing.assert_allclose(
            controller.continuous_rotation_pose_target[thumb],
            [0.3368, -1.3441, -np.deg2rad(4.0), -np.deg2rad(4.0)],
        )

    def test_fk_estimation_mode_ignores_missing_thumb_tactile(self):
        controller = GraspController(RuntimeConfig(hand_side="right"), log=None)
        self.assertFalse(controller.blind_use_tactile_estimation)
        controller.apply_pose_type(6, now=1.0)
        controller.start_continuous_rotation(now=2.0)

        controller._set_continuous_rotation_phase("blind_pose_rotation", 3.0)
        controller._set_continuous_rotation_phase("blind_pinky_regrasp", 3.3)
        self.assertEqual(controller.blind_thumb_missed_rotation_count, 0)

        expected = controller._blind_thumb_release_target()
        controller.blind_thumb_missed_rotation_count = 1
        controller._start_blind_thumb_release(now=5.0)
        thumb = np.asarray(FINGER_JOINT_INDEX[1], dtype=int)
        target = controller.continuous_rotation_pose_target[thumb]
        np.testing.assert_allclose(target[:2], expected[:2])
        np.testing.assert_allclose(target[2:], -np.deg2rad([4.0, 4.0]))

    def test_any_thumb_contact_during_rotation_resets_missed_count(self):
        controller = GraspController(RuntimeConfig(hand_side="right"), log=None)
        controller.set_blind_use_tactile_estimation(True)
        controller.blind_thumb_missed_rotation_count = 1
        controller._set_continuous_rotation_phase("blind_pose_rotation", 1.0)
        controller.set_tactile_contacts(np.zeros((5, 5)))
        contacts = np.zeros((5, 5))
        contacts[0, 0] = 1.0
        controller.set_tactile_contacts(contacts)
        controller._set_continuous_rotation_phase("blind_pinky_regrasp", 1.3)

        self.assertEqual(controller.blind_thumb_missed_rotation_count, 0)

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
