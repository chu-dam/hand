import unittest
import numpy as np

from dg5f_grasp_control.config import RuntimeConfig
from dg5f_grasp_control.grasp_controller import GraspController
from dg5f_grasp_control.grasp_policy import contact_polygon_area_3d
from dg5f_grasp_control.hand_model import FINGER_JOINT_INDEX
from dg5f_grasp_control.kinematics import set_hand_side, tip_position
from dg5f_grasp_control.poses import (
    RIGHT_HAND_BLIND_GRASP_INITIAL_POSE,
    RIGHT_HAND_BLIND_GRASP_REVERSE_ROTATION_POSE,
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
                RuntimeConfig(
                    hand_side="right",
                    blind_sphere_adaptation_enabled=True,
                ),
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
        controller = GraspController(
            RuntimeConfig(
                hand_side="right",
                blind_sphere_adaptation_enabled=True,
            ),
            log=None,
        )
        controller.apply_pose_type(6, now=1.0)
        self.assertTrue(controller.start_continuous_rotation(now=2.0))
        controller.blind_sphere_sampled_hand_x = 0.08
        controller.step(np.zeros(20), np.zeros(20), now=2.01)
        self.assertFalse(controller.continuous_rotation_active)

        self.assertTrue(controller.start_continuous_rotation(now=3.0))
        self.assertIsNone(controller.blind_sphere_sampled_hand_x)
        controller.step(np.zeros(20), np.zeros(20), now=3.01)

        self.assertTrue(controller.continuous_rotation_active)

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
            "blind_grasp_settle",
        )
        self.assertEqual(controller.state, "GROPED_GRASP")
        self.assertEqual(controller.use_fingers, [1, 2, 3, 4, 5])

        left = GraspController(RuntimeConfig(hand_side="left"), log=None)
        self.assertFalse(left.start_continuous_rotation(now=2.0))

    def test_both_pre_rotation_poses_start_same_blind_sequence(self):
        for pose_type in (5, 6):
            controller = GraspController(
                RuntimeConfig(hand_side="right"),
                log=None,
            )
            controller.apply_pose_type(pose_type, now=1.0)
            np.testing.assert_allclose(
                controller._current_pre_grasp_pose(),
                controller.pose_type_targets[pose_type],
            )

            self.assertTrue(controller.start_continuous_rotation(now=2.0))
            self.assertEqual(controller.pose_type, pose_type)
            self.assertEqual(controller.state, "GROPED_GRASP")
            self.assertEqual(controller.active_finger_count, 5)
            self.assertEqual(controller.use_fingers, [1, 2, 3, 4, 5])
            self.assertEqual(
                controller.continuous_rotation_phase,
                "blind_grasp_settle",
            )
            output = controller.step(
                controller.pose_type_targets[pose_type],
                np.zeros(20),
                now=2.1,
            )
            self.assertTrue(np.all(np.isnan(output.inactive_pd_target)))
            np.testing.assert_allclose(output.inactive_pd, 0.0)

    def test_unknown_object_rotation_skips_sphere_estimation(self):
        controller = GraspController(RuntimeConfig(hand_side="right"), log=None)
        controller.apply_pose_type(6, now=1.0)
        controller.start_continuous_rotation(now=2.0)

        self.finish_blind_motion(controller, range(1, 6), 2.001)

        self.assertEqual(
            controller.continuous_rotation_phase,
            "blind_middle_release",
        )
        self.assertIsNone(controller.blind_sphere_sampled_hand_x)

    def test_blind_j1_release_scales_linearly_with_contact_polygon_area(self):
        q = RIGHT_HAND_BLIND_GRASP_INITIAL_POSE.copy()
        points = np.asarray([tip_position(q, finger) for finger in range(1, 6)])
        area = contact_polygon_area_3d(points)
        controller = GraspController(
            RuntimeConfig(
                hand_side="right",
                blind_j1_area_reference_m2=2.0 * area,
                blind_j1_area_scale_min=0.1,
                blind_j1_area_scale_max=2.0,
            ),
            log=None,
        )
        controller.sync_joint_state(q)

        controller._capture_blind_j1_release_scale()

        self.assertAlmostEqual(controller.blind_j1_release_scale, 0.5)
        middle = np.asarray(FINGER_JOINT_INDEX[3], dtype=int)
        target = controller._blind_release_target(3)
        self.assertAlmostEqual(
            target[0],
            q[middle[0]] - np.deg2rad(0.5 * 19.0),
        )
        controller._start_blind_pinky_release(now=2.0)
        pinky = np.asarray(FINGER_JOINT_INDEX[5], dtype=int)
        self.assertAlmostEqual(
            controller.continuous_rotation_pose_target[pinky[0]],
            q[pinky[0]] - np.deg2rad(0.5 * 8.0),
        )

    def test_blind_rotation_also_starts_from_active_five_finger_grasp(self):
        controller = GraspController(RuntimeConfig(hand_side="right"), log=None)
        controller.apply_pose_type(6, now=1.0)
        controller.apply_grasp_type(5, now=1.5)

        self.assertTrue(controller.start_continuous_rotation(now=2.0))
        self.assertEqual(controller.use_fingers, [1, 2, 3, 4, 5])
        self.assertEqual(controller.continuous_rotation_phase, "blind_grasp_settle")

    def test_blind_sequence_rotates_without_middle_or_pinky(self):
        logs = []
        controller = GraspController(
            RuntimeConfig(
                hand_side="right",
                blind_sphere_adaptation_enabled=True,
            ),
            log=logs.append,
        )
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
        self.assertAlmostEqual(
            target[pinky[0]], initial[pinky[0]] - np.deg2rad(8.0)
        )
        np.testing.assert_allclose(target[pinky[1:]], measured[pinky[1:]])

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
            *FINGER_JOINT_INDEX[2][:2],
            *FINGER_JOINT_INDEX[4][:2],
        ])
        np.testing.assert_allclose(output.inactive_pd_target[controlled], initial[controlled])
        np.testing.assert_allclose(
            output.inactive_pd_target[middle],
            controller.continuous_rotation_pose_target[middle],
        )
        np.testing.assert_allclose(output.inactive_pd_target[pinky], target[pinky])
        uncontrolled = np.setdiff1d(
            np.arange(20), np.concatenate([controlled, middle, pinky])
        )
        self.assertTrue(np.all(np.isnan(output.inactive_pd_target[uncontrolled])))
        np.testing.assert_allclose(
            output.tau[uncontrolled], output.grasp_tau[uncontrolled]
        )

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

    def test_direction_change_reestimates_before_thumb_release(self):
        for initial_direction in (1, -1):
            controller = GraspController(
                RuntimeConfig(
                    hand_side="right",
                    blind_sphere_adaptation_enabled=True,
                ),
                log=None,
            )
            controller.apply_pose_type(6, now=1.0)
            controller.start_continuous_rotation(now=2.0)
            controller.blind_rotation_direction = initial_direction
            controller._start_blind_thumb_release(now=3.0)

            controller.request_blind_direction_change()
            controller._process_continuous_rotation(3.01)

            self.assertEqual(controller.blind_rotation_direction, -initial_direction)
            self.assertEqual(
                controller.continuous_rotation_phase,
                "blind_direction_sphere_estimate",
            )
            controller._process_continuous_rotation(3.511)
            self.assertEqual(
                controller.continuous_rotation_phase,
                "blind_direction_sphere_estimate",
            )

            for x in (0.0961, 0.0965, 0.0969, 0.0963, 0.0967):
                controller.set_ui_sphere_center_hand([x, 0.0, 0.05])
            controller._process_continuous_rotation(3.512)

            thumb = np.asarray(FINGER_JOINT_INDEX[1], dtype=int)
            self.assertEqual(controller.continuous_rotation_phase, "blind_thumb_release")
            q = np.zeros(20)
            q[thumb] = controller.continuous_rotation_pose_target[thumb]
            tip = tip_position(q, 1)
            base_tip = controller.blind_thumb_release_tip_hand[-initial_direction]
            np.testing.assert_allclose(tip[1:], base_tip[1:], atol=5e-4)
            self.assertAlmostEqual(tip[0], base_tip[0] - 0.0035, delta=5e-4)

    def test_direction_change_during_pose_rotation_reverses_in_place(self):
        controller = GraspController(RuntimeConfig(hand_side="right"), log=None)
        controller.apply_pose_type(6, now=1.0)
        controller.start_continuous_rotation(now=2.0)
        anchor = np.linspace(-0.4, 0.4, 20)
        controller.sync_joint_state(anchor)
        controller._start_blind_pinky_release(now=2.1)
        controller._start_blind_pose_rotation(2.2)
        middle = np.asarray(FINGER_JOINT_INDEX[3], dtype=int)
        middle_target = controller.continuous_rotation_pose_target[middle].copy()

        controller.request_blind_direction_change()
        controller._process_continuous_rotation(2.25)

        self.assertEqual(controller.blind_rotation_direction, -1)
        self.assertEqual(controller.continuous_rotation_phase, "blind_pose_rotation")
        self.assertEqual(controller.continuous_rotation_phase_started_at, 2.25)
        self.assertEqual(controller.use_fingers, [1, 2, 4])
        np.testing.assert_allclose(
            controller.continuous_rotation_pose_target[middle],
            middle_target,
        )
        output = controller.step(np.zeros(20), np.zeros(20), now=2.251)
        controlled = np.array([
            *FINGER_JOINT_INDEX[1],
            *FINGER_JOINT_INDEX[2][:2],
            *FINGER_JOINT_INDEX[4][:2],
        ])
        np.testing.assert_allclose(output.inactive_pd_target[controlled], anchor[controlled])
        self.assertFalse(controller.paper_rotation_active)
        pinky = np.asarray(FINGER_JOINT_INDEX[5], dtype=int)
        uncontrolled = np.setdiff1d(
            np.arange(20), np.concatenate([controlled, middle, pinky])
        )
        self.assertTrue(np.all(np.isnan(output.inactive_pd_target[uncontrolled])))

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
        controller = GraspController(
            RuntimeConfig(
                hand_side="right",
                blind_sphere_adaptation_enabled=True,
            ),
            log=None,
        )
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

        self.assertAlmostEqual(controller.blind_sphere_sampled_hand_x, 0.097)
        self.assertEqual(controller.continuous_rotation_phase, "blind_pinky_release")

    def test_release_is_relative_to_current_joint_positions(self):
        controller = GraspController(RuntimeConfig(hand_side="right"), log=None)
        measured = np.linspace(-0.4, 0.4, 20)
        for finger, j1 in ((2, -0.2), (3, 0.0), (4, 0.3)):
            measured[FINGER_JOINT_INDEX[finger][0]] = j1
        controller.sync_joint_state(measured)

        for direction in (1, -1):
            controller.blind_rotation_direction = direction
            for finger, release_deg in (
                (2, [0.0, 0.0] if direction > 0 else [6.0, 6.0]),
                (3, [0.0, 5.0]),
                (4, [6.0, 6.0] if direction > 0 else [0.0, 0.0]),
            ):
                indices = np.asarray(FINGER_JOINT_INDEX[finger], dtype=int)
                target = controller._blind_release_target(finger)
                self.assertAlmostEqual(
                    target[0],
                    measured[indices[0]]
                    + (-1.0 if direction > 0 else 1.0) * np.deg2rad(19.0),
                )
                np.testing.assert_allclose(
                    target[1:3],
                    measured[indices[1:3]] - np.deg2rad(release_deg),
                )
                self.assertEqual(target[3], measured[indices[3]])

    def test_release_j1_stops_at_each_finger_limit_in_both_directions(self):
        controller = GraspController(RuntimeConfig(hand_side="right"), log=None)
        for finger in (2, 3, 4):
            indices = np.asarray(FINGER_JOINT_INDEX[finger], dtype=int)
            lower, upper = controller.finger_avoidance_joint_limits[finger]
            for direction, start, expected in ((1, lower + 0.01, lower), (-1, upper - 0.01, upper)):
                with self.subTest(finger=finger, direction=direction):
                    measured = RIGHT_HAND_BLIND_GRASP_INITIAL_POSE.copy()
                    measured[indices[0]] = start
                    controller.sync_joint_state(measured)
                    controller.blind_rotation_direction = direction
                    controller.blind_j1_release_scale = 1.25
                    target = controller._blind_release_target(finger)
                    self.assertAlmostEqual(target[0], expected)
                    self.assertEqual(target[3], measured[indices[3]])
                    if finger == 3:
                        controller._start_blind_middle_release(now=1.0)
                    else:
                        controller._start_blind_index_ring_release(now=1.0)
                    self.assertAlmostEqual(controller.continuous_rotation_pose_target[indices[0]], expected)

    def test_common_j1_motion_and_thumb_release_share_the_limiting_ratio(self):
        for direction in (1, -1):
            with self.subTest(direction=direction):
                controller = GraspController(RuntimeConfig(hand_side="right"), log=None)
                controller.blind_rotation_direction = direction
                reference = RIGHT_HAND_BLIND_GRASP_INITIAL_POSE.copy()
                for finger, margin_deg in ((2, 20.0), (3, 15.0), (4, 8.0)):
                    lower, upper = controller.finger_avoidance_joint_limits[finger]
                    reference[FINGER_JOINT_INDEX[finger][0]] = (
                        lower + np.deg2rad(margin_deg) if direction > 0
                        else upper - np.deg2rad(margin_deg)
                    )
                controller.sync_joint_state(reference)
                controller.blind_initial_grasp_q = reference.copy()
                controller.blind_j1_release_reference_q = reference.copy()
                for finger in (2, 3, 4):
                    j1 = FINGER_JOINT_INDEX[finger][0]
                    target = controller._blind_release_target(finger)
                    self.assertAlmostEqual(target[0] - reference[j1], -direction * np.deg2rad(8.0))
                prefix = "blind_thumb" if direction > 0 else "blind_reverse_thumb"
                original_thumb_target = np.array([
                    getattr(controller.cfg, f"{prefix}_j{joint}_target_rad")
                    for joint in range(1, 5)
                ])
                initial_thumb = reference[FINGER_JOINT_INDEX[1]]
                original_thumb_target[2] = initial_thumb[2] + 1.1 * (
                    original_thumb_target[2] - initial_thumb[2]
                )
                expected_thumb = original_thumb_target.copy()
                expected_thumb[:2] = initial_thumb[:2] + (8.0 / 19.0) * (
                    original_thumb_target[:2] - initial_thumb[:2]
                )
                np.testing.assert_allclose(controller._blind_thumb_release_target(), expected_thumb)
                drifted = reference.copy()
                drifted[FINGER_JOINT_INDEX[2][0]] += 0.02
                controller.sync_joint_state(drifted)
                self.assertAlmostEqual(
                    controller._blind_release_target(2)[0],
                    reference[FINGER_JOINT_INDEX[2][0]] - direction * np.deg2rad(8.0),
                )

    def test_no_common_j1_headroom_holds_thumb_j1_j2_but_preserves_release_clearance(self):
        controller = GraspController(RuntimeConfig(hand_side="right"), log=None)
        reference = RIGHT_HAND_BLIND_GRASP_INITIAL_POSE.copy()
        reference[FINGER_JOINT_INDEX[4][0]] = controller.finger_avoidance_joint_limits[4][0]
        controller.sync_joint_state(reference)
        controller.blind_initial_grasp_q = reference.copy()
        controller.blind_j1_release_reference_q = reference.copy()
        for finger in (2, 3, 4):
            self.assertAlmostEqual(
                controller._blind_release_target(finger)[0],
                reference[FINGER_JOINT_INDEX[finger][0]],
            )
        initial_thumb = reference[FINGER_JOINT_INDEX[1]]
        target = controller._blind_thumb_release_target()
        np.testing.assert_allclose(target[:2], initial_thumb[:2])
        self.assertAlmostEqual(target[2], initial_thumb[2] + 1.1 * (
            controller.cfg.blind_thumb_j3_target_rad - initial_thumb[2]
        ))
        self.assertAlmostEqual(target[3], controller.cfg.blind_thumb_j4_target_rad)

    def test_common_j1_reference_refreshes_for_next_cycle_and_clears_on_stop(self):
        controller = GraspController(RuntimeConfig(hand_side="right"), log=None)
        controller.apply_pose_type(6, now=0.5)
        controller.start_continuous_rotation(now=1.0)
        controller._process_continuous_rotation(1.501)
        np.testing.assert_allclose(controller.blind_j1_release_reference_q, controller.hand_q)
        updated = controller.hand_q.copy() + 0.01
        controller.sync_joint_state(updated)
        controller._advance_blind_rotation(3, now=2.0)
        np.testing.assert_allclose(controller.blind_j1_release_reference_q, updated)
        controller.stop_continuous_rotation(now=2.1)
        self.assertIsNone(controller.blind_j1_release_reference_q)

    def test_reverse_thumb_release_uses_supplied_target(self):
        controller = GraspController(RuntimeConfig(hand_side="right"), log=None)
        controller.blind_rotation_direction = -1
        thumb = np.asarray(FINGER_JOINT_INDEX[1], dtype=int)

        controller._start_blind_thumb_release(now=2.0)
        target = controller.continuous_rotation_pose_target[thumb]

        np.testing.assert_allclose(target, [0.3238, -1.3102, -0.1991, 0.8927])

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
            [0.1168, -1.6596, -0.1108, 0.9184],
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
        self.assertEqual(controller.use_fingers, [1, 2, 5])

    def test_thumb_regrasp_adds_fingers_in_stages_before_pinky_release(self):
        controller = GraspController(RuntimeConfig(hand_side="right"), log=None)
        controller.apply_pose_type(6, now=0.5)
        controller.sync_joint_state(RIGHT_HAND_BLIND_GRASP_INITIAL_POSE)
        controller.start_continuous_rotation(now=1.0)
        controller._start_blind_thumb_release(now=2.0)
        held = np.concatenate([FINGER_JOINT_INDEX[3], FINGER_JOINT_INDEX[4]])
        held_target = controller.continuous_rotation_pose_target[held].copy()
        controller._start_blind_regrasp(2, now=2.2)

        output = controller.step(controller.hand_q, np.zeros(20), now=2.299)
        self.assertEqual(set(output.grasp_forces), {1, 2, 5})
        np.testing.assert_allclose(output.grasp_tau[held], 0.0)
        np.testing.assert_allclose(output.inactive_pd_target[held], held_target)
        self.assertTrue(np.all(np.isnan(
            output.inactive_pd_target[FINGER_JOINT_INDEX[1]]
        )))

        output = controller.step(controller.hand_q, np.zeros(20), now=2.301)
        self.assertEqual(controller.continuous_rotation_phase, "blind_thumb_regrasp_settle")
        self.assertEqual(set(output.grasp_forces), {1, 2, 3, 4, 5})
        self.assertTrue(np.all(np.isnan(output.inactive_pd_target[held])))
        controller._process_continuous_rotation(2.380)
        self.assertEqual(controller.continuous_rotation_phase, "blind_thumb_regrasp_settle")
        controller._process_continuous_rotation(2.382)
        self.assertEqual(controller.continuous_rotation_phase, "blind_pinky_release")
        self.assertEqual(controller.use_fingers, [1, 2, 3, 4])

    def test_middle_index_ring_regrasp_holds_measured_j1_and_keeps_distal_grasp(self):
        for group, fingers in ((0, (3,)), (1, (2, 4))):
            for direction in (1, -1):
                with self.subTest(group=group, direction=direction):
                    controller = GraspController(RuntimeConfig(hand_side="right"), log=None)
                    controller.apply_pose_type(6, now=0.5)
                    controller.sync_joint_state(RIGHT_HAND_BLIND_GRASP_INITIAL_POSE)
                    controller.start_continuous_rotation(now=1.0)
                    controller.blind_rotation_direction = direction
                    if group == 0:
                        controller._start_blind_middle_release(2.0)
                    else:
                        controller._start_blind_index_ring_release(2.0)
                    j1 = np.array([FINGER_JOINT_INDEX[f][0] for f in fingers])
                    measured = controller.hand_q.copy()
                    measured[j1] += 0.03
                    controller.sync_joint_state(measured)
                    controller._start_blind_regrasp(group, now=2.1)
                    np.testing.assert_allclose(
                        controller.continuous_rotation_pose_target[j1], measured[j1],
                    )
                    displaced = measured.copy()
                    displaced[j1] += 0.02
                    output = controller.step(displaced, np.zeros(20), now=2.11)
                    self.assertEqual(set(output.grasp_forces), {1, 2, 3, 4, 5})
                    np.testing.assert_allclose(output.grasp_tau[j1], 0.0)
                    np.testing.assert_allclose(output.inactive_pd_target[j1], measured[j1])
                    np.testing.assert_allclose(
                        output.tau[j1], -0.02 * controller.cfg.blind_grasp_pre_rotation_pose_kp,
                    )
                    distal = np.concatenate([FINGER_JOINT_INDEX[f][1:] for f in fingers])
                    expected = controller.policy.calc_tau_from_total_forces(
                        displaced, output.total_forces,
                    )
                    np.testing.assert_allclose(output.grasp_tau[distal], expected[distal])
                    np.testing.assert_allclose(output.tau[distal], expected[distal])
                    self.assertTrue(np.all(np.isnan(output.inactive_pd_target[distal])))

    def test_thumb_release_scales_only_j3_from_initial_grasp(self):
        controller = GraspController(RuntimeConfig(hand_side="right"), log=None)
        thumb = np.asarray(FINGER_JOINT_INDEX[1], dtype=int)
        initial = RIGHT_HAND_BLIND_GRASP_INITIAL_POSE.copy()
        controller.blind_initial_grasp_q = initial
        controller.blind_j1_release_scale = 0.5

        target = controller._blind_thumb_release_target()

        np.testing.assert_allclose(target[[0, 1, 3]], [
            controller.cfg.blind_thumb_j1_target_rad,
            controller.cfg.blind_thumb_j2_target_rad,
            controller.cfg.blind_thumb_j4_target_rad,
        ])
        self.assertAlmostEqual(
            target[2],
            initial[thumb[2]] + 1.1 * 0.5 * (
                controller.cfg.blind_thumb_j3_target_rad - initial[thumb[2]]
            ),
        )

    def test_reverse_pinky_release_matches_forward_eight_degrees(self):
        controller = GraspController(RuntimeConfig(hand_side="right"), log=None)
        controller.blind_rotation_direction = -1
        pinky = np.asarray(FINGER_JOINT_INDEX[5], dtype=int)
        measured = np.zeros(20)
        measured[pinky] = [0.61, 1.22, 0.93, 0.38]
        controller.sync_joint_state(measured)

        controller._start_blind_pinky_release(now=2.0)
        target = controller.continuous_rotation_pose_target[pinky]

        self.assertAlmostEqual(
            target[0], measured[pinky[0]] - np.deg2rad(8.0)
        )
        np.testing.assert_allclose(target[1:], measured[pinky[1:]])

    def test_pinky_release_reuses_first_target_despite_joint_drift(self):
        controller = GraspController(RuntimeConfig(hand_side="right"), log=None)
        pinky = np.asarray(FINGER_JOINT_INDEX[5], dtype=int)
        original = RIGHT_HAND_BLIND_GRASP_INITIAL_POSE.copy()
        first_targets = {}
        for direction in (1, -1):
            controller.blind_rotation_direction = direction
            controller.sync_joint_state(original)
            controller._start_blind_pinky_release(now=1.0)
            first_targets[direction] = controller.continuous_rotation_pose_target[pinky].copy()
        drifted = original.copy()
        drifted[pinky] += [0.2, -0.1, 0.15, -0.1]
        controller.sync_joint_state(drifted)
        controller.blind_j1_release_scale = 0.5
        for direction in (1, -1, 1):
            controller.blind_rotation_direction = direction
            controller._start_blind_pinky_release(now=2.0)
            np.testing.assert_allclose(
                controller.continuous_rotation_pose_target[pinky], first_targets[direction],
            )

    def test_pinky_release_reference_is_reset_for_a_new_sequence(self):
        controller = GraspController(RuntimeConfig(hand_side="right"), log=None)
        controller.apply_pose_type(6, now=0.5)
        controller.start_continuous_rotation(now=1.0)
        controller._start_blind_pinky_release(now=1.1)
        self.assertTrue(controller.blind_pinky_release_targets)
        controller.stop_continuous_rotation(now=1.2)
        self.assertEqual(controller.blind_pinky_release_targets, {})
        pinky = np.asarray(FINGER_JOINT_INDEX[5], dtype=int)
        measured = RIGHT_HAND_BLIND_GRASP_INITIAL_POSE.copy()
        measured[pinky] += 0.1
        controller.sync_joint_state(measured)
        controller.apply_grasp_type(5, now=1.9)
        self.assertTrue(controller.start_continuous_rotation(now=2.0))
        controller._start_blind_pinky_release(now=2.1)
        expected = measured[pinky].copy()
        expected[0] -= np.deg2rad(controller.cfg.blind_pinky_release_deg)
        np.testing.assert_allclose(controller.continuous_rotation_pose_target[pinky], expected)

    def test_forward_pinky_release_keeps_measured_j2_to_j4(self):
        controller = GraspController(RuntimeConfig(hand_side="right"), log=None)
        pinky = np.asarray(FINGER_JOINT_INDEX[5], dtype=int)
        measured = np.zeros(20)
        measured[pinky] = [0.70, 1.22, 0.93, 0.38]
        controller.sync_joint_state(measured)

        controller._start_blind_pinky_release(now=2.0)
        target = controller.continuous_rotation_pose_target[pinky]

        self.assertAlmostEqual(
            target[0], measured[pinky[0]] - np.deg2rad(8.0)
        )
        np.testing.assert_allclose(target[1:], measured[pinky[1:]])

    def test_pinky_regrasp_combines_target_pd_with_grasp_force(self):
        controller = GraspController(
            RuntimeConfig(
                hand_side="right",
                blind_sphere_adaptation_enabled=True,
            ),
            log=None,
        )
        controller.continuous_rotation_active = True
        controller.continuous_rotation_pose_target = np.zeros(20)
        controller._start_blind_regrasp(3, now=2.0)
        pinky = np.asarray(FINGER_JOINT_INDEX[5], dtype=int)

        output = controller.step(np.zeros(20), np.zeros(20), now=2.001)

        self.assertEqual(controller.use_fingers, [1, 2, 3, 4, 5])
        self.assertEqual(controller.policy.cfg.alpha1, 5.0)
        self.assertEqual(controller.cfg.alpha1, 3.0)
        np.testing.assert_allclose(
            output.inactive_pd_target[pinky],
            [0.6948, 1.1952, 0.7905, 0.8992],
        )
        self.assertTrue(np.any(np.abs(output.inactive_pd[pinky]) > 0.0))
        self.assertTrue(np.any(np.abs(output.grasp_tau[pinky]) > 0.0))
        middle = np.asarray(FINGER_JOINT_INDEX[3], dtype=int)
        self.assertTrue(np.all(np.isnan(output.inactive_pd_target[middle])))
        self.assertTrue(np.any(np.abs(output.grasp_tau[middle]) > 0.0))
        self.finish_blind_motion(controller, (5,), 2.201)
        self.assertEqual(
            controller.continuous_rotation_phase,
            "blind_forward_sphere_estimate",
        )
        self.assertEqual(controller.policy.cfg.alpha1, 3.0)

    def test_unknown_object_pinky_regrasp_has_no_fixed_target(self):
        controller = GraspController(RuntimeConfig(hand_side="right"), log=None)
        controller.blind_rotation_direction = -1
        controller.continuous_rotation_pose_target = np.zeros(20)
        controller._start_blind_regrasp(3, now=2.0)
        pinky = np.asarray(FINGER_JOINT_INDEX[5], dtype=int)

        np.testing.assert_allclose(
            controller.continuous_rotation_pose_target[pinky],
            0.0,
        )

    def test_reverse_pose_rotation_uses_initial_grasp_target(self):
        controller = GraspController(RuntimeConfig(hand_side="right"), log=None)
        controller.apply_pose_type(6, now=1.0)
        controller.start_continuous_rotation(now=2.0)
        anchor = np.linspace(-0.3, 0.3, 20)
        controller.sync_joint_state(anchor)
        controller.blind_initial_grasp_q = anchor.copy()
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
            *FINGER_JOINT_INDEX[2][:2],
            *FINGER_JOINT_INDEX[4][:2],
        ])
        self.assertEqual(controller.use_fingers, [1, 2, 4])
        np.testing.assert_allclose(output.inactive_pd_target[controlled], anchor[controlled])
        self.assertFalse(controller.paper_rotation_active)

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
        self.assertAlmostEqual(
            controller.blind_sphere_sampled_hand_x,
            np.mean(first_samples),
        )

        controller._start_blind_forward_sphere_estimate(now=3.0)
        for x in (0.0966, 0.0970, 0.0974, 0.0968, 0.0972):
            controller.set_ui_sphere_center_hand([x, 0.0, 0.05])
        controller._process_continuous_rotation(3.501)
        self.assertAlmostEqual(
            controller.blind_sphere_sampled_hand_x,
            0.097,
        )

    def test_fk_sphere_hand_x_continuously_offsets_thumb_release_tip(self):
        controller = GraspController(
            RuntimeConfig(
                hand_side="right",
                blind_sphere_adaptation_enabled=True,
            ),
            log=None,
        )
        base_tip = controller.blind_thumb_release_tip_hand[1].copy()
        controller.blind_sphere_hand_x_samples = [0.095]
        self.assertTrue(controller._finish_blind_sphere_sampling())
        controller._start_blind_thumb_release(now=1.0)
        thumb = np.asarray(FINGER_JOINT_INDEX[1], dtype=int)
        q = np.zeros(20)
        q[thumb] = controller.continuous_rotation_pose_target[thumb]
        adjusted_tip = tip_position(q, 1)
        np.testing.assert_allclose(adjusted_tip[1:], base_tip[1:], atol=5e-4)
        self.assertAlmostEqual(adjusted_tip[0], base_tip[0] - 0.005, delta=5e-4)
        self.assertEqual(
            q[thumb[0]], controller._blind_thumb_release_target()[0]
        )
        self.assertGreaterEqual(q[thumb[0]], -0.2792526803)
        self.assertLessEqual(q[thumb[0]], 0.9250245036)
        self.assertEqual(controller.continuous_rotation_phase, "blind_thumb_release")

    def test_fk_height_offset_applies_to_index_and_ring_releases(self):
        controller = GraspController(
            RuntimeConfig(
                hand_side="right",
                blind_sphere_adaptation_enabled=True,
            ),
            log=None,
        )
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
                base_tip[0] - 0.005,
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

            self.assertGreater(adjusted_tip[0] - base_tip[0], 0.0005)
            self.assertLess(adjusted_tip[0] - base_tip[0], 0.020)
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

    def test_pose_rotation_tracks_initial_grasp_joints_then_regrasps(self):
        controller = GraspController(
            RuntimeConfig(
                hand_side="right",
                continuous_rotation_move_sec=2.0,
            ),
            log=None,
        )
        controller.apply_grasp_type(5, now=0.5, internal=True)
        controller.continuous_rotation_active = True
        controller.continuous_rotation_pose_target = np.zeros(20)
        anchor = np.linspace(-0.2, 0.2, 20)
        controller.blind_initial_grasp_q = anchor.copy()
        controller._start_blind_pose_rotation(now=1.0)

        output = controller.step(np.zeros(20), np.zeros(20), now=1.085)
        self.assertFalse(controller.paper_rotation_active)
        controlled = np.array([
            *FINGER_JOINT_INDEX[1],
            *FINGER_JOINT_INDEX[2][:2],
            *FINGER_JOINT_INDEX[4][:2],
        ])
        self.assertEqual(controller.use_fingers, [1, 2, 4])
        self.assertEqual(set(output.grasp_forces), {1, 2, 4})
        expected_grasp_tau = controller.policy.calc_tau_from_total_forces(
            controller.hand_q, output.total_forces,
        )
        expected_grasp_tau[FINGER_JOINT_INDEX[1][:2]] = 0.0
        np.testing.assert_allclose(
            output.grasp_tau, expected_grasp_tau,
        )
        for finger in (1, 2, 4):
            np.testing.assert_allclose(output.rotation_forces[finger], 0.0)
            np.testing.assert_allclose(output.center_hold_forces[finger], 0.0)
        np.testing.assert_allclose(
            output.tau[FINGER_JOINT_INDEX[1]],
            (output.inactive_pd + output.grasp_tau)[FINGER_JOINT_INDEX[1]],
        )
        self.assertTrue(np.any(
            np.abs(output.grasp_tau[FINGER_JOINT_INDEX[2]]) > 0.0
        ))
        self.assertTrue(np.any(
            np.abs(output.grasp_tau[FINGER_JOINT_INDEX[4]]) > 0.0
        ))
        np.testing.assert_allclose(output.inactive_pd_target[controlled], anchor[controlled])
        middle = np.asarray(FINGER_JOINT_INDEX[3], dtype=int)
        pinky = np.asarray(FINGER_JOINT_INDEX[5], dtype=int)
        self.assertEqual(
            output.inactive_pd_target[middle[0]],
            anchor[middle[0]],
        )
        np.testing.assert_allclose(
            output.inactive_pd_target[middle],
            controller.continuous_rotation_pose_target[middle],
        )
        uncontrolled = np.setdiff1d(
            np.arange(20), np.concatenate([controlled, middle, pinky])
        )
        self.assertTrue(np.all(np.isnan(output.inactive_pd_target[uncontrolled])))

        controller.step(np.zeros(20), np.zeros(20), now=1.7)
        self.assertEqual(controller.continuous_rotation_phase, "blind_pose_rotation")

        current = np.zeros(20)
        current[controlled] = anchor[controlled]
        controller.step(current, np.zeros(20), now=1.8)
        output = controller.step(current, np.zeros(20), now=1.89)
        self.assertEqual(controller.continuous_rotation_phase, "blind_pose_alignment")
        self.assertEqual(controller.use_fingers, [1, 2, 4])
        np.testing.assert_allclose(output.inactive_pd_target[:16], anchor[:16])
        np.testing.assert_allclose(output.inactive_pd_target[pinky], 0.0)
        expected_middle_pd = np.clip(
            controller.cfg.blind_grasp_pre_rotation_pose_kp * (anchor[middle] - current[middle]),
            -controller.cfg.pre_rotation_pose_pd_limit, controller.cfg.pre_rotation_pose_pd_limit,
        )
        np.testing.assert_allclose(output.inactive_pd[middle], expected_middle_pd)
        controller.step(current, np.zeros(20), now=1.90)
        controller.step(current, np.zeros(20), now=1.99)
        self.assertEqual(controller.continuous_rotation_phase, "blind_pinky_regrasp")
        self.assertEqual(controller.use_fingers, [1, 2, 3, 4, 5])

        output = controller.step(controller.hand_q, np.zeros(20), now=2.00)
        self.assertTrue(np.all(np.isnan(output.inactive_pd_target[pinky])))
        self.assertTrue(np.any(
            np.abs(output.grasp_tau[FINGER_JOINT_INDEX[1]]) > 0.0
        ))

    def test_pose_rotation_timeout_starts_alignment_before_pinky_regrasp(self):
        controller = GraspController(RuntimeConfig(hand_side="right"), log=None)
        controller.apply_grasp_type(5, now=0.5, internal=True)
        controller.continuous_rotation_active = True
        controller.continuous_rotation_pose_target = np.zeros(20)
        controller.blind_initial_grasp_q = np.ones(20)
        controller._start_blind_pose_rotation(now=1.0)

        controller.step(np.zeros(20), np.zeros(20), now=1.01)
        controller.step(np.zeros(20), np.zeros(20), now=1.02)
        controller.step(np.zeros(20), np.zeros(20), now=1.281)
        self.assertEqual(controller.continuous_rotation_phase, "blind_pose_alignment")
        controller.hand_qdot[:16] = 0.1
        controller._process_continuous_rotation(1.4)
        self.assertEqual(controller.continuous_rotation_phase, "blind_pose_alignment")
        controller._process_continuous_rotation(1.562)
        self.assertEqual(controller.continuous_rotation_phase, "blind_pinky_regrasp")
        self.assertTrue(controller.continuous_rotation_active)
        self.assertEqual(controller.use_fingers, [1, 2, 3, 4, 5])

    def test_direction_change_during_alignment_preserves_saved_target_and_pinky_release(self):
        controller = GraspController(RuntimeConfig(hand_side="right"), log=None)
        controller.apply_pose_type(6, now=0.5)
        controller.start_continuous_rotation(now=1.0)
        controller._start_blind_pinky_release(now=1.1)
        controller._start_blind_pose_rotation(now=1.2)
        saved = controller.blind_initial_grasp_q.copy()
        pinky = FINGER_JOINT_INDEX[5]
        pinky_target = controller.continuous_rotation_pose_target[pinky].copy()
        controller._start_blind_pose_alignment(now=1.3)
        controller.request_blind_direction_change()
        controller._process_continuous_rotation(1.31)
        self.assertEqual(controller.continuous_rotation_phase, "blind_pose_alignment")
        self.assertEqual(controller.blind_rotation_direction, -1)
        self.assertEqual(controller.use_fingers, [1, 2, 4])
        np.testing.assert_allclose(controller.continuous_rotation_pose_target[:16], saved[:16])
        np.testing.assert_allclose(controller.continuous_rotation_pose_target[pinky], pinky_target)

    def test_sequence_rotation_keeps_paper_controller_inactive(self):
        controller = GraspController(RuntimeConfig(hand_side="right"), log=None)
        controller.apply_pose_type(6, now=0.5)
        controller.start_continuous_rotation(now=1.0)
        controller._start_blind_pinky_release(now=1.1)
        controller._start_blind_pose_rotation(now=1.2)
        controller.step(np.zeros(20), np.zeros(20), now=1.21)
        controller.step(np.zeros(20), np.zeros(20), now=1.22)
        self.assertFalse(controller.paper_rotation_active)
        controller.stop_continuous_rotation(now=1.3)
        output = controller.step(np.zeros(20), np.zeros(20), now=1.31)
        self.assertFalse(controller.paper_rotation_active)
        self.assertTrue(all(np.all(force == 0.0) for force in output.rotation_forces.values()))

    def test_reverse_estimate_advances_after_five_stable_centers(self):
        controller = GraspController(
            RuntimeConfig(
                hand_side="right",
                blind_sphere_adaptation_enabled=True,
            ),
            log=None,
        )
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


if __name__ == "__main__":
    unittest.main()
