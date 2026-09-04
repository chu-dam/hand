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

RIGHT_HAND_PRE_ROTATION_POSE = np.array([
     0.7384, -1.0917, 0.2936, 0.8437,  # thumb
    -0.0799,  0.7665, 0.5812, 0.2939,  # index
     0.1504,  0.5627, 0.7418, 0.2431,  # middle
     0.2705,  0.7821, 0.6339, 0.2834,  # ring
     0.7327,  0.9755, 1.0877, 0.0398,  # pinky
], dtype=np.float64)

RIGHT_HAND_BLIND_GRASP_PRE_ROTATION_POSE = np.array([
     0.7123, -1.1078, -0.4318, 1.3542,  # thumb
    -0.4388,  0.5618,  0.5831, 1.1058,  # index
     0.0220,  0.2747,  0.6414, 1.0463,  # middle
     0.4079,  0.3859,  0.7892, 0.9531,  # ring
     0.9088,  1.4821,  0.9477, 0.3524,  # pinky
], dtype=np.float64)

RIGHT_HAND_BLIND_GRASP_INITIAL_POSE = np.array([
     0.7632, -1.0491, -0.3667, 1.3799,  # thumb
    -0.4147,  0.5814,  0.6002, 1.1100,  # index
     0.0033,  0.2796,  0.6386, 1.0521,  # middle
     0.3749,  0.3904,  0.7976, 0.9589,  # ring
     0.6962,  1.1519,  0.9589, 0.3608,  # pinky
], dtype=np.float64)

RIGHT_HAND_BLIND_GRASP_REVERSE_ROTATION_POSE = np.array([
     0.1696, -1.6570, -0.2011, 1.3533,  # thumb
    -0.8377580409572782, 0.7569, 0.6754, 1.0774,  # index (J1 clamped to -48 deg)
    -0.4679,  0.3540,  0.6517, 1.0371,  # middle
    -0.0960,  0.2946,  0.6718, 0.9289,  # ring
     0.6058,  1.0894,  1.0210, 0.3850,  # pinky
], dtype=np.float64)

RIGHT_HAND_CONTINUOUS_ROTATION_POSE = np.array([
     0.7660, -1.3130, 0.5590, 0.4587,  # thumb
    -0.2552,  0.9154, 0.5842, 0.2882,  # index
    -0.1162,  0.6428, 0.5979, 0.2505,  # middle
     0.0005,  0.6575, 0.5683, 0.2827,  # ring
     0.8777,  0.7625, 0.5538, 0.2868,  # pinky
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
    5: RIGHT_HAND_PRE_ROTATION_POSE,
    6: RIGHT_HAND_BLIND_GRASP_PRE_ROTATION_POSE,
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
