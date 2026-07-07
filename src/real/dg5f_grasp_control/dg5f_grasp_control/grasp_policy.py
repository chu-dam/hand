import numpy as np

from dg5f_grasp_control.hand_model import (
    FINGER_JOINT_INDEX,
    GRASP_TAU_SIGN,
)
from dg5f_grasp_control.kinematics import tip_jacobian, tip_position

FINGER_FORCE_SCALE = {
    1: 1.0,
    2: 1.0,
    3: 0.9,
    4: 0.55,
    5: 0.35,
}

COLLISION_AVOID_PAIRS = [
    (3, 4),
    (4, 5),
]

BASE_JOINT_TAU_LIMIT = {
    12: 0.8,
    16: 0.5,
}


class GraspPolicy:
    def __init__(self, use_fingers, cfg):
        self.use_fingers = list(use_fingers)
        self.cfg = cfg

    def calc_alpha_and_forces(self, q):
        tip_pos = {
            finger: tip_position(q, finger)
            for finger in self.use_fingers
        }

        points = np.array([tip_pos[finger] for finger in self.use_fingers])
        cg = np.mean(points, axis=0)

        if 1 in tip_pos:
            thumb_pos = tip_pos[1]
            cv = cg + self.cfg.thumb_centroid_bias * (thumb_pos - cg)
        else:
            cv = cg

        fhat = {}
        dist = {}
        for finger in self.use_fingers:
            diff = cv - tip_pos[finger]
            dist[finger] = max(np.linalg.norm(diff), 1e-6)
            fhat[finger] = self.cfg.groped_force_direction_sign * diff / dist[finger]

        alpha = {}

        if len(self.use_fingers) == 2:
            alpha[self.use_fingers[0]] = self.cfg.alpha1
            alpha[self.use_fingers[1]] = self.cfg.alpha1
            return alpha, fhat, cg, cv, tip_pos

        first_finger = self.use_fingers[0]
        pivot_finger = self.use_fingers[-1]
        alpha[first_finger] = self.cfg.alpha1

        for finger in self.use_fingers[1:-1]:
            alpha[finger] = dist[first_finger] / dist[finger] * self.cfg.alpha1

        force_sum = np.zeros(3, dtype=np.float64)
        for finger in self.use_fingers[:-1]:
            force_sum += alpha[finger] * fhat[finger]

        alpha[pivot_finger] = np.linalg.norm(force_sum)

        return alpha, fhat, cg, cv, tip_pos

    def calc_collision_avoidance_forces(self, tip_pos):
        repel = {
            finger: np.zeros(3, dtype=np.float64)
            for finger in tip_pos
        }

        for finger_a, finger_b in COLLISION_AVOID_PAIRS:
            if finger_a not in tip_pos or finger_b not in tip_pos:
                continue

            diff = tip_pos[finger_a] - tip_pos[finger_b]
            dist = np.linalg.norm(diff)
            if dist < 1e-9 or dist >= self.cfg.min_tip_distance:
                continue

            direction = diff / dist
            mag = self.cfg.collision_repel_gain * (self.cfg.min_tip_distance - dist)
            mag = min(mag, self.cfg.collision_repel_limit)

            repel[finger_a] += mag * direction
            repel[finger_b] -= mag * direction

        return repel

    def calc_grasp_tau(self, q):
        alpha, fhat, cg, cv, tip_pos = self.calc_alpha_and_forces(q)
        repel = self.calc_collision_avoidance_forces(tip_pos)

        tau = np.zeros(20, dtype=np.float64)

        for finger in self.use_fingers:
            idxs = FINGER_JOINT_INDEX[finger]
            J = tip_jacobian(q, finger, eps=self.cfg.jacobian_eps)
            grasp_force = FINGER_FORCE_SCALE[finger] * alpha[finger] * fhat[finger]
            total_force = grasp_force + repel[finger]
            tau_finger = J.T @ total_force
            tau[idxs] = tau_finger * GRASP_TAU_SIGN[idxs]

        tau = np.clip(tau, -self.cfg.groped_tau_limit, self.cfg.groped_tau_limit)

        for joint_idx, limit in BASE_JOINT_TAU_LIMIT.items():
            tau[joint_idx] = np.clip(tau[joint_idx], -limit, limit)

        return tau, alpha, cg, cv, tip_pos
