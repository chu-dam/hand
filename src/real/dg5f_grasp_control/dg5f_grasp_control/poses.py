import numpy as np


# Left-hand poses
LEFT_HAND_NORMAL_POSE = np.array([
     0.0288,  0.3665, -0.7102, -0.2981,  # thumb
     0.2100, -0.0094,  0.5128,  0.4074,  # index
     0.0000,  0.0054,  0.4328,  0.5548,  # middle
    -0.2100, -0.0157,  0.4594,  0.5142,  # ring
    -0.1471, -0.3410, -0.1508,  0.8662,  # pinky
], dtype=np.float64)

LEFT_HAND_PRE_GRASP_POSE = np.array([
     0.0342,  1.6153, -0.3260, -0.4194,  # thumb
     0.2100,  0.0930,  0.8035,  0.2777,  # index
     0.0000,  0.2477,  0.4854,  0.4978,  # middle
    -0.2100,  0.2576,  0.4238,  0.4777,  # ring
    -0.1292, -0.3042,  0.3047,  0.6637,  # pinky
], dtype=np.float64)

LEFT_HAND_COMPACT_PRE_GRASP_POSE = np.array([
     0.0342,  1.6153, -0.5240, -0.2340,  # thumb
     0.1562,  0.7800,  0.3459,  0.4384,  # index
     0.0000,  0.7800,  0.3780,  0.3988,  # middle
    -0.1562,  0.7800,  0.2384,  0.5344,  # ring
    -0.1709, -0.5348,  1.1196,  0.4133,  # pinky
], dtype=np.float64)

LEFT_HAND_CARD_PRE_GRASP_POSE = np.array([
     0.0000,  np.pi / 2, -0.3536, -0.7756,  # thumb
     0.0000,  0.3751,    0.8587,  0.5213,  # index
     0.0000,  0.0000,    0.0000,  0.0000,  # middle
     0.0000,  0.0000,    0.0000,  0.0000,  # ring
     0.0000,  0.0000,    0.0000,  0.0000,  # pinky
], dtype=np.float64)


# Right-hand poses. Start from the verified left-hand values and tune here.
RIGHT_HAND_NORMAL_POSE = np.array([
     0.0288, -0.3665,  0.7102,  0.2981,  # thumb
    -0.2100, -0.0094,  0.5128,  0.4074,  # index
     0.0000,  0.0054,  0.4328,  0.5548,  # middle
     0.2100, -0.0157,  0.4594,  0.5142,  # ring
     0.1471,  0.3410, -0.1508,  0.8662,  # pinky
], dtype=np.float64)

RIGHT_HAND_PRE_GRASP_POSE = np.array([
     0.0342, -1.6153,  0.3260,  0.4194,  # thumb
    -0.2100,  0.0930,  0.8035,  0.2777,  # index
     0.0000,  0.2477,  0.4854,  0.4978,  # middle
     0.2100,  0.2576,  0.4238,  0.4777,  # ring
     0.1292,  0.3042,  0.3047,  0.6637,  # pinky
], dtype=np.float64)

RIGHT_HAND_COMPACT_PRE_GRASP_POSE = np.array([
     0.0342, -1.6153,  0.5240,  0.2340,  # thumb
    -0.1562,  0.7800,  0.3459,  0.4384,  # index
     0.0000,  0.7800,  0.3780,  0.3988,  # middle
     0.1562,  0.7800,  0.2384,  0.5344,  # ring
     0.1709,  0.5348,  1.1196,  0.4133,  # pinky
], dtype=np.float64)

RIGHT_HAND_CARD_PRE_GRASP_POSE = np.array([
     0.0000, -np.pi / 2, 0.3536,  0.7756,  # thumb
     0.0000,  0.3751,    0.8587,  0.5213,  # index
     0.0000,  0.0000,    0.0000,  0.0000,  # middle
     0.0000,  0.0000,    0.0000,  0.0000,  # ring
     0.0000,  0.0000,    0.0000,  0.0000,  # pinky
], dtype=np.float64)


LEFT_POSE_TYPE_TARGETS = {
    1: LEFT_HAND_NORMAL_POSE,
    2: LEFT_HAND_PRE_GRASP_POSE,
    3: LEFT_HAND_COMPACT_PRE_GRASP_POSE,
    4: LEFT_HAND_CARD_PRE_GRASP_POSE,
}

RIGHT_POSE_TYPE_TARGETS = {
    1: RIGHT_HAND_NORMAL_POSE,
    2: RIGHT_HAND_PRE_GRASP_POSE,
    3: RIGHT_HAND_COMPACT_PRE_GRASP_POSE,
    4: RIGHT_HAND_CARD_PRE_GRASP_POSE,
}


def get_pose_type_targets(hand_side):
    if hand_side == "left":
        return LEFT_POSE_TYPE_TARGETS
    if hand_side == "right":
        return RIGHT_POSE_TYPE_TARGETS
    raise ValueError("hand_side must be 'left' or 'right'")


# Backward-compatible left-hand names.
HAND_NORMAL_POSE = LEFT_HAND_NORMAL_POSE
HAND_PRE_GRASP_POSE = LEFT_HAND_PRE_GRASP_POSE
HAND_COMPACT_PRE_GRASP_POSE = LEFT_HAND_COMPACT_PRE_GRASP_POSE
HAND_CARD_PRE_GRASP_POSE = LEFT_HAND_CARD_PRE_GRASP_POSE
POSE_TYPE_TARGETS = LEFT_POSE_TYPE_TARGETS
