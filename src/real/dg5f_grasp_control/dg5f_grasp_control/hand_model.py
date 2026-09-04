import numpy as np

JOINT_COUNT = 20
FINGER_COUNT = 5
TACTILE_Y_ORIGIN_OFFSET_M = 0.0032
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

# Per-finger joint used by inactive-finger collision avoidance. Index, middle,
# and ring spread with joint 1. Pinky instead flexes joint 2 to move away from
# ring in the physical hand geometry.
FINGER_AVOIDANCE_JOINT_LOCAL_INDEX = {
    1: 0,
    2: 0,
    3: 0,
    4: 0,
    5: 1,
}

# Limits copied from each hand URDF for the selected avoidance joints. The
# right-hand joint coordinates are mirrored, so its limits are negated.
LEFT_FINGER_AVOIDANCE_JOINT_LIMITS = {
    1: (-1.57079632679, 1.57079632679),
    2: (-0.261799387799, 0.837758040957),
    3: (-0.733038285838, 0.575958653158),
    4: (-0.837758040957, 0.209439510239),
    5: (-1.57079632679, 0.645771823238),
}

RIGHT_FINGER_AVOIDANCE_JOINT_LIMITS = {
    finger: (-upper, -lower)
    for finger, (lower, upper) in LEFT_FINGER_AVOIDANCE_JOINT_LIMITS.items()
}
RIGHT_FINGER_AVOIDANCE_JOINT_LIMITS[1] = (
    -0.2792526803190927,
    0.9250245035569946,
)


def get_finger_avoidance_joint_limits(hand_side):
    if hand_side == "left":
        return LEFT_FINGER_AVOIDANCE_JOINT_LIMITS
    if hand_side == "right":
        return RIGHT_FINGER_AVOIDANCE_JOINT_LIMITS
    raise ValueError("hand_side must be 'left' or 'right'")


# Backward-compatible left-hand name.
FINGER_AVOIDANCE_JOINT_LIMITS = LEFT_FINGER_AVOIDANCE_JOINT_LIMITS

GRASP_TAU_SIGN = np.ones(JOINT_COUNT, dtype=np.float64)


def selected_fingers(use_finger_count):
    if use_finger_count not in FINGER_SELECTIONS:
        raise ValueError("use_finger_count must be one of: 1, 2, 3, 4, 5")
    return FINGER_SELECTIONS[use_finger_count]
