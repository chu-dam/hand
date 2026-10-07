import unittest

import numpy as np
import pandas as pd

from calc_fric import fit_joint_model, make_trial_table


class SymmetricFrictionFitTest(unittest.TestCase):
    def test_parallel_joints_are_separate_trials(self):
        rows = []
        for joint_index in (5, 9, 13):
            for sample in range(5):
                rows.append({
                    "source_file": "parallel.csv",
                    "trial_id": 1,
                    "joint_index": joint_index,
                    "joint_name": f"joint_{joint_index}",
                    "direction": 1,
                    "test_effort": 0.1,
                    "time": sample,
                    "sample_time": sample,
                    "q": joint_index + sample * 0.1,
                    "qdot": 0.1,
                    "q_start": joint_index,
                    "q_delta": sample * 0.1,
                    "stop_reason": "upper_limit" if sample == 4 else "",
                })

        trials = make_trial_table(pd.DataFrame(rows))

        self.assertEqual(set(trials["joint_index"]), {5, 9, 13})
        self.assertEqual(len(trials), 3)

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
