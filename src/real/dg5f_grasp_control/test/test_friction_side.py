import unittest

import numpy as np

from dg5f_grasp_control import friction
from dg5f_grasp_control import friction_params_left, friction_params_right


class FrictionSideTest(unittest.TestCase):
    def tearDown(self):
        friction.set_hand_side("left")

    def test_left_parameters_are_selected(self):
        qdot = np.linspace(-0.2, 0.2, 20)
        friction.set_hand_side("left")
        expected = (
            friction_params_left.HAND_FRIC_COULOMB_SCALE
            * friction_params_left.HAND_FRIC_FC
            * np.tanh(17.0 * qdot)
            + friction_params_left.HAND_FRIC_B * qdot
        )
        np.testing.assert_allclose(friction.calc_friction(qdot), expected)

    def test_right_measured_parameters_are_selected(self):
        qdot = np.linspace(-0.2, 0.2, 20)
        friction.set_hand_side("right")
        expected = (
            friction_params_right.HAND_FRIC_FC
            * np.tanh(20.0 * qdot)
            + friction_params_right.HAND_FRIC_B * qdot
        )
        np.testing.assert_allclose(
            friction.calc_friction(qdot, tanh_k=20.0),
            expected,
        )


if __name__ == "__main__":
    unittest.main()
