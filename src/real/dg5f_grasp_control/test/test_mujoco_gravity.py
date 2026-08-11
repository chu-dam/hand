import unittest

import numpy as np
from ament_index_python.packages import get_package_share_directory

from dg5f_grasp_control.mujoco_gravity import MujocoGravityCompensator


class MujocoGravityTest(unittest.TestCase):
    def test_right_urdf_computes_finite_gravity_torque(self):
        model_path = (
            get_package_share_directory("dg5f_s_description")
            + "/urdf/dg5fs_right_w_mount.urdf"
        )
        compensator = MujocoGravityCompensator(model_path)

        torque = compensator.compute(
            np.zeros(20, dtype=np.float64),
            gravity=np.array([0.0, 0.0, -9.81]),
        )

        self.assertEqual(torque.shape, (20,))
        self.assertTrue(np.all(np.isfinite(torque)))
        self.assertGreater(np.max(np.abs(torque)), 0.0)


if __name__ == "__main__":
    unittest.main()
