from dg5f_grasp_control import kinematics_left, kinematics_right


_implementation = kinematics_left


def set_hand_side(side):
    global _implementation
    if side not in ("left", "right"):
        raise ValueError("hand_side must be 'left' or 'right'")
    _implementation = kinematics_left if side == "left" else kinematics_right


def tesollo_forward_kinematics(q):
    return _implementation.tesollo_forward_kinematics(q)


def tip_position(q, finger):
    return _implementation.tip_position(q, finger)


def tip_jacobian(q, finger, eps=1e-6):
    return _implementation.tip_jacobian(q, finger, eps=eps)
