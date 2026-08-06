import numpy as np

from dg5f_grasp_control.friction_params import HAND_FRIC_B, HAND_FRIC_FC


def calc_friction(qdot, scale=1.0, tanh_k=17.0, limit=1.0):
    qdot = np.asarray(qdot, dtype=np.float64)
    tau = HAND_FRIC_FC * np.tanh(tanh_k * qdot) * 0.7 + HAND_FRIC_B * qdot
    return np.clip(scale * tau, -limit, limit)
