import numpy as np

from dg5f_grasp_control import friction_params_left, friction_params_right


_params = friction_params_left


def set_hand_side(side):
    global _params
    if side not in ("left", "right"):
        raise ValueError("hand_side must be 'left' or 'right'")
    _params = friction_params_left if side == "left" else friction_params_right


def calc_friction(qdot, scale=1.0, tanh_k=17.0, limit=1.0):
    qdot = np.asarray(qdot, dtype=np.float64)
    tau = (
        _params.HAND_FRIC_COULOMB_SCALE
        * _params.HAND_FRIC_FC
        * np.tanh(tanh_k * qdot)
        + _params.HAND_FRIC_B * qdot
    )
    return np.clip(scale * tau, -limit, limit)
