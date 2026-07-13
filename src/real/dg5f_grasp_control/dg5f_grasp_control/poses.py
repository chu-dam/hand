import numpy as np

HAND_NORMAL_POSE = np.array([
     0.0288,  0.3665, -0.7102, -0.2981,  # thumb
     0.2100, -0.0094,  0.5128,  0.4074,  # index
     0.0000,  0.0054,  0.4328,  0.5548,  # middle
    -0.2100, -0.0157,  0.4594,  0.5142,  # ring
    -0.1471, -0.3410, -0.1508,  0.8662,  # pinky
], dtype=np.float64)

HAND_PRE_GRASP_POSE = np.array([
    -0.0431,  1.3725, -0.7102, -0.2981,  # thumb
     0.2100, -0.0094,  0.5128,  0.4074,  # index
    -0.0351,  0.0054,  0.4328,  0.5548,  # middle
    -0.2100, -0.0157,  0.4594,  0.5142,  # ring
    -0.1471, -0.3410, -0.1508,  0.8662,  # pinky
], dtype=np.float64)

HAND_COMPACT_PRE_GRASP_POSE = np.array([
     0.0000,  1.5710, -0.5240, -0.2340,  # thumb
     0.1562,  0.7800,  0.3459,  0.4384,  # index
     0.0000,  0.7800,  0.3780,  0.3988,  # middle
    -0.1562,  0.7800,  0.2384,  0.5344,  # ring
     0.0000, -0.3410,  0.5236,  0.4873,  # pinky
], dtype=np.float64)

POSE_TYPE_TARGETS = {
    1: HAND_NORMAL_POSE,
    2: HAND_PRE_GRASP_POSE,
    3: HAND_COMPACT_PRE_GRASP_POSE,
}
