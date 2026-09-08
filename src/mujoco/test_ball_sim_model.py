import unittest

import mujoco
import numpy as np

from ball_sim_model import (
    BALL_CENTER_FINGERS,
    BALL_INITIAL_Z_OFFSET_M,
    BALL_MASS_KG,
    BALL_RADIUS_M,
    SOFT_TIP_MARGIN_M,
    SILICONE_PLA_SLIDING_FRICTION,
    HAND_TO_WORLD_ROTATION,
    JOINT_FRICTION_COMPENSATION_FRACTION,
    WORLD_GRAVITY,
    BLIND_GRASP_ALPHA1_N,
    BALL_RELEASE_MIN_CONTACT_FINGERS,
    apply_gravity_compensated_pose_hold,
    apply_gravity_compensated_grasp,
    ball_contact_fingers,
    ball_touches_hand,
    build_model,
    hold_ball_pose,
    reset_at_grasp_pose,
    set_ball_suspended,
)
from dg5f_grasp_control.friction_params_right import (
    HAND_FRIC_B,
    HAND_FRIC_COULOMB_SCALE,
    HAND_FRIC_FC,
)
from dg5f_grasp_control.config import RuntimeConfig
from dg5f_grasp_control.grasp_controller import GraspController
from dg5f_grasp_control.grasp_policy import polygon_centroid_3d
from dg5f_grasp_control.kinematics import tip_position
from dg5f_grasp_control import kinematics, kinematics_right


class BallSimModelTest(unittest.TestCase):
    def test_tactile_tip_ball_reset(self):
        model = build_model()
        data = mujoco.MjData(model)
        center, tip_centers = reset_at_grasp_pose(model, data)

        for finger in range(1, 6):
            np.testing.assert_allclose(
                kinematics.tip_position(data.qpos[:20], finger),
                kinematics_right.tip_position(data.qpos[:20], finger),
            )

        controller_tips_world = np.asarray(
            [
                HAND_TO_WORLD_ROTATION @ tip_position(data.qpos[:20], finger)
                for finger in BALL_CENTER_FINGERS
            ]
        )
        expected_center = polygon_centroid_3d(controller_tips_world)
        expected_center[2] += BALL_INITIAL_Z_OFFSET_M
        np.testing.assert_allclose(center, expected_center)
        self.assertEqual(tip_centers.shape, (len(BALL_CENTER_FINGERS), 3))

        geom_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_GEOM, "ball_geom"
        )
        self.assertEqual(model.geom_type[geom_id], mujoco.mjtGeom.mjGEOM_MESH)
        ball_mesh = int(model.geom_dataid[geom_id])
        start = int(model.mesh_vertadr[ball_mesh])
        ball_vertices = model.mesh_vert[
            start : start + int(model.mesh_vertnum[ball_mesh])
        ]
        self.assertAlmostEqual(
            np.max(np.linalg.norm(ball_vertices, axis=1)), BALL_RADIUS_M
        )
        self.assertAlmostEqual(model.body_mass[model.geom_bodyid[geom_id]], BALL_MASS_KG)
        self.assertAlmostEqual(
            model.geom_friction[geom_id, 0], SILICONE_PLA_SLIDING_FRICTION
        )
        self.assertAlmostEqual(model.geom_margin[geom_id], SOFT_TIP_MARGIN_M)
        self.assertEqual(model.geom_priority[geom_id], 1)
        self.assertEqual(model.opt.solver, mujoco.mjtSolver.mjSOL_NEWTON)
        finger_1_geoms = np.flatnonzero(
            model.geom_bodyid
            == mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "link_1_4")
        )
        finger_2_geoms = np.flatnonzero(
            model.geom_bodyid
            == mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "link_2_4")
        )
        geom_1 = int(finger_1_geoms[0])
        geom_2 = int(finger_2_geoms[0])
        tactile_geom = next(
            int(candidate)
            for candidate in finger_1_geoms
            if mujoco.mj_id2name(
                model,
                mujoco.mjtObj.mjOBJ_MESH,
                int(model.geom_dataid[candidate]),
            ) == "tactile_tip"
        )
        self.assertAlmostEqual(
            model.geom_friction[tactile_geom, 0],
            SILICONE_PLA_SLIDING_FRICTION,
        )
        self.assertTrue(
            model.opt.enableflags & int(mujoco.mjtEnableBit.mjENBL_MULTICCD)
        )
        self.assertEqual(
            model.geom_contype[geom_1] & model.geom_conaffinity[geom_1], 0
        )
        self.assertNotEqual(
            model.geom_contype[geom_1] & model.geom_conaffinity[geom_2], 0
        )
        self.assertNotEqual(
            model.geom_contype[geom_1] & model.geom_conaffinity[geom_id], 0
        )
        np.testing.assert_allclose(model.opt.gravity, WORLD_GRAVITY)
        np.testing.assert_allclose(
            model.dof_frictionloss[:20],
            JOINT_FRICTION_COMPENSATION_FRACTION
            * HAND_FRIC_COULOMB_SCALE
            * HAND_FRIC_FC,
        )
        np.testing.assert_allclose(
            model.dof_damping[:20],
            JOINT_FRICTION_COMPENSATION_FRACTION * HAND_FRIC_B,
        )
        ball_body = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_BODY, "ball"
        )
        self.assertEqual(model.body_gravcomp[ball_body], 1.0)
        self.assertTrue(ball_touches_hand(model, data))
        self.assertLess(
            len(ball_contact_fingers(model, data)),
            BALL_RELEASE_MIN_CONTACT_FINGERS,
        )
        set_ball_suspended(model, False)
        self.assertEqual(model.body_gravcomp[ball_body], 0.0)

        ball_joint = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_JOINT, "ball_free"
        )
        ball_qpos_address = int(model.jnt_qposadr[ball_joint])
        pinned_pose = data.qpos[ball_qpos_address : ball_qpos_address + 7].copy()
        for _ in range(200):
            hold_ball_pose(model, data, pinned_pose)
            apply_gravity_compensated_pose_hold(model, data)
            mujoco.mj_step(model, data)
        hold_ball_pose(model, data, pinned_pose)
        np.testing.assert_allclose(
            data.qpos[ball_qpos_address : ball_qpos_address + 7], pinned_pose
        )

        reset_at_grasp_pose(model, data)
        mujoco.mj_forward(model, data)
        expected_gravity_compensation = data.qfrc_bias.copy()
        apply_gravity_compensated_pose_hold(model, data)
        hand_dofs = np.arange(20)
        np.testing.assert_allclose(
            data.qfrc_applied[hand_dofs], expected_gravity_compensation[hand_dofs]
        )

        controller = GraspController(
            RuntimeConfig(
                hand_side="right", use_finger_count=5, alpha1=BLIND_GRASP_ALPHA1_N
            ),
            log=None,
        )
        controller.sync_joint_state(data.qpos[:20])
        controller.apply_grasp_type(5, data.time)
        output = apply_gravity_compensated_grasp(
            model, data, controller, data.time
        )
        self.assertEqual(output.active_finger_count, 5)
        self.assertEqual(output.use_fingers, [1, 2, 3, 4, 5])
        self.assertAlmostEqual(controller.cfg.alpha1, 3.0)


if __name__ == "__main__":
    unittest.main()
