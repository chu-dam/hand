import unittest
from pathlib import Path

import yaml

from dg5f_grasp_control.config import RuntimeConfig


class RightGainConfigTest(unittest.TestCase):
    def test_right_gain_overlay_contains_only_known_parameters(self):
        config_dir = Path(__file__).resolve().parent.parent / "config"
        left = yaml.safe_load(
            (config_dir / "grasp_real_left_gains.yaml").read_text()
        )
        right = yaml.safe_load(
            (config_dir / "grasp_real_right_gains.yaml").read_text()
        )
        left_params = left["grasp_real"]["ros__parameters"]
        right_params = right["grasp_real"]["ros__parameters"]

        self.assertLessEqual(
            set(right_params),
            set(RuntimeConfig.__dataclass_fields__),
        )
        self.assertEqual(left_params["envelop_thumb_tau_sign"], -1.0)
        self.assertEqual(right_params["envelop_thumb_tau_sign"], 1.0)


if __name__ == "__main__":
    unittest.main()
