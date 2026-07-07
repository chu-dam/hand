import numpy as np

JOINT_COUNT = 20
FINGER_COUNT = 5
HAND_JOINT_NAMES = [f"joint_{f}_{j}" for f in range(1, 6) for j in range(1, 5)]

FINGER_SELECTIONS = {
    1: [1, 2],
    2: [1, 3],
    3: [1, 2, 3],
    4: [1, 2, 3, 4],
    5: [1, 2, 3, 4, 5],
}

FINGER_JOINT_INDEX = {
    1: [0, 1, 2, 3],
    2: [4, 5, 6, 7],
    3: [8, 9, 10, 11],
    4: [12, 13, 14, 15],
    5: [16, 17, 18, 19],
}

GRASP_TAU_SIGN = np.ones(JOINT_COUNT, dtype=np.float64)


def selected_fingers(use_finger_count):
    if use_finger_count not in FINGER_SELECTIONS:
        raise ValueError("use_finger_count must be one of: 1, 2, 3, 4, 5")
    return FINGER_SELECTIONS[use_finger_count]
