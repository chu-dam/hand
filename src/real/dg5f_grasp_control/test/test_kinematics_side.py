import unittest

import numpy as np

from dg5f_grasp_control import kinematics
from dg5f_grasp_control import kinematics_left, kinematics_right


class KinematicsSideTest(unittest.TestCase):
    def tearDown(self):
        kinematics.set_hand_side("left")

    def test_side_selection_and_right_analytic_jacobian(self):
        q = np.linspace(-0.25, 0.30, 20)

        kinematics.set_hand_side("left")
        np.testing.assert_allclose(
            kinematics.tesollo_forward_kinematics(q),
            kinematics_left.tesollo_forward_kinematics(q),
        )

        kinematics.set_hand_side("right")
        np.testing.assert_allclose(
            kinematics.tesollo_forward_kinematics(q),
            kinematics_right.tesollo_forward_kinematics(q),
        )

        eps = 1e-7
        for finger in range(1, 6):
            expected = np.empty((3, 4), dtype=np.float64)
            for column, q_index in enumerate(range(4 * (finger - 1), 4 * finger)):
                q_plus = q.copy()
                q_minus = q.copy()
                q_plus[q_index] += eps
                q_minus[q_index] -= eps
                expected[:, column] = (
                    kinematics_right.tip_position(q_plus, finger)
                    - kinematics_right.tip_position(q_minus, finger)
                ) / (2.0 * eps)

            np.testing.assert_allclose(
                kinematics.tip_jacobian(q, finger),
                expected,
                rtol=0.0,
                atol=1e-9,
            )

    def test_invalid_side_is_rejected(self):
        with self.assertRaises(ValueError):
            kinematics.set_hand_side("invalid")


if __name__ == "__main__":
    unittest.main()
