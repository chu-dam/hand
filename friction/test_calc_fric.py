import unittest

import numpy as np
import pandas as pd

from calc_fric import fit_joint_model


class SymmetricFrictionFitTest(unittest.TestCase):
    def test_directional_average_removes_constant_gravity_error(self):
        fc = 0.04
        viscous = 0.09
        gravity_error = 0.012
        velocities = np.array([0.2, 0.4, 0.8, 1.2])
        rows = []
        for direction in (1, -1):
            for velocity in velocities:
                qdot = direction * velocity
                rows.append({
                    "direction": direction,
                    "qdot_fit": qdot,
                    "q_delta": qdot,
                    "test_effort": (
                        gravity_error
                        + direction * fc
                        + viscous * qdot
                    ),
                    "stop_reason": (
                        "upper_limit" if direction > 0 else "lower_limit"
                    ),
                })

        result, _, _ = fit_joint_model(pd.DataFrame(rows))

        self.assertAlmostEqual(result["Fc"], fc)
        self.assertAlmostEqual(result["B"], viscous)
        self.assertAlmostEqual(result["gravity_offset"], gravity_error)


if __name__ == "__main__":
    unittest.main()
