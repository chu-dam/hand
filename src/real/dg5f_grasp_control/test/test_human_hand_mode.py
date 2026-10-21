import unittest

import numpy as np

from dg5f_grasp_control.config import RuntimeConfig
from dg5f_grasp_control.grasp_controller import (
    GraspController,
    HUMAN_DIP_COUPLING_RATIO,
)
from dg5f_grasp_control.grasp_policy import GraspPolicy
from dg5f_grasp_control.hand_model import FINGER_JOINT_INDEX
from dg5f_grasp_control.kinematics import set_hand_side, tip_jacobian


class HumanHandModeTest(unittest.TestCase):
    def setUp(self):
        set_hand_side("right")
        self.cfg = RuntimeConfig(
            hand_side="right",
            pose_kp=0.29,
            pose_kd=0.05,
            human_dip_coupling_tau_limit=10.0,
        )

    def test_human_jacobian_uses_first_three_joints_only(self):
        q = np.linspace(-0.2, 0.3, 20)
        force = np.array([1.0, -2.0, 0.5])
        forces = {2: force}
        current = GraspPolicy([2], self.cfg).calc_tau_from_total_forces(q, forces)
        human = GraspPolicy(
            [2], self.cfg, human_hand_mode=True
        ).calc_tau_from_total_forces(q, forces)
        indices = np.asarray(FINGER_JOINT_INDEX[2], dtype=int)
        jacobian = tip_jacobian(q, 2)

        np.testing.assert_allclose(human[indices[:3]], jacobian[:, :3].T @ force)
        self.assertEqual(human[indices[3]], 0.0)
        self.assertNotEqual(current[indices[3]], 0.0)

    def test_human_dip_uses_only_pip_coupling_in_pre_grasp(self):
        controller = GraspController(self.cfg, log=None)
        controller.apply_pose_type(2, now=1.0)
        controller.set_human_hand_mode(True)
        q = np.zeros(20)
        qdot = np.zeros(20)
        for finger in HUMAN_DIP_COUPLING_RATIO:
            pip, dip = FINGER_JOINT_INDEX[finger][-2:]
            q[pip] = 0.4
            q[dip] = 0.1
            qdot[pip] = 0.2
            qdot[dip] = -0.1

        output = controller.step(q, qdot, now=1.01)

        for finger, ratio in HUMAN_DIP_COUPLING_RATIO.items():
            dip = FINGER_JOINT_INDEX[finger][-1]
            self.assertAlmostEqual(
                output.tau[dip],
                self.cfg.pose_kp * (ratio * 0.4 - 0.1)
                + self.cfg.pose_kd * (ratio * 0.2 + 0.1),
            )

    def test_human_mode_survives_grasp_policy_rebuild(self):
        controller = GraspController(self.cfg, log=None)
        controller.set_human_hand_mode(True)

        controller.apply_grasp_type(5, now=1.0)

        self.assertTrue(controller.human_hand_mode)
        self.assertTrue(controller.policy.human_hand_mode)


if __name__ == "__main__":
    unittest.main()
