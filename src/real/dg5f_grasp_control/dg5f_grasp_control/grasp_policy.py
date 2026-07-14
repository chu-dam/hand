from dataclasses import dataclass
from typing import Dict

import numpy as np

from dg5f_grasp_control.hand_model import (
    FINGER_JOINT_INDEX,
    GRASP_TAU_SIGN,
)
from dg5f_grasp_control.kinematics import tip_jacobian, tip_position

COLLISION_AVOID_PAIRS = [
    (3, 4),
    (4, 5),
]

BASE_JOINT_TAU_LIMIT = {
    12: 0.8,
    16: 0.5,
}


@dataclass
class GraspPolicyResult:
    """Cartesian force decomposition and the torque produced from it."""

    tau: np.ndarray
    alpha: Dict[int, float]
    cg: np.ndarray
    cv: np.ndarray
    fingertip_positions: Dict[int, np.ndarray]
    grasp_forces: Dict[int, np.ndarray]
    rotation_forces: Dict[int, np.ndarray]
    center_hold_forces: Dict[int, np.ndarray]
    collision_forces: Dict[int, np.ndarray]
    total_forces: Dict[int, np.ndarray]


def _normalize(v, eps=1e-9):
    n = np.linalg.norm(v)
    if n < eps:
        return np.zeros_like(v), 0.0
    return v / n, n


def polygon_centroid_3d(points):
    """Return the groped-shape centroid in 3-D.

    Two contacts use the midpoint and three contacts use the triangle
    centroid. For four or more contacts, fingertip positions are projected
    onto their best-fit plane, ordered as a polygon, evaluated with the
    paper's signed-area centroid equations, and mapped back to 3-D.

    No arithmetic-mean fallback is used for the N-contact polygon case.
    """
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or points.shape[0] == 0:
        raise ValueError("points must have shape (N, 3) with N >= 1")

    projection_origin = np.mean(points, axis=0)
    if points.shape[0] <= 3:
        return projection_origin

    centered = points - projection_origin
    _, _, basis = np.linalg.svd(centered, full_matrices=False)

    axis_u = basis[0]
    axis_v = basis[1]
    projected = np.column_stack((centered @ axis_u, centered @ axis_v))

    # The centroid equations require vertices ordered around the polygon.
    angles = np.arctan2(projected[:, 1], projected[:, 0])
    polygon = projected[np.argsort(angles)]
    polygon_next = np.roll(polygon, -1, axis=0)

    cross = (
        polygon[:, 0] * polygon_next[:, 1]
        - polygon_next[:, 0] * polygon[:, 1]
    )
    twice_signed_area = float(np.sum(cross))
    if abs(twice_signed_area) <= np.finfo(np.float64).eps:
        raise ValueError("contact polygon has zero signed area")

    centroid_2d = np.array([
        np.sum((polygon[:, 0] + polygon_next[:, 0]) * cross),
        np.sum((polygon[:, 1] + polygon_next[:, 1]) * cross),
    ], dtype=np.float64) / (3.0 * twice_signed_area)

    return (
        projection_origin
        + centroid_2d[0] * axis_u
        + centroid_2d[1] * axis_v
    )


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
        cg = polygon_centroid_3d(points)

        if len(self.use_fingers) == 2:
            # Pinch grasp: keep the virtual centroid at the geometric midpoint
            # so the two equal-magnitude contact forces remain opposite.
            cv = cg
        elif 1 in tip_pos:
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


    def calc_pure_moment_rotation_forces(self, rotation_center, tip_pos):
        """Least-norm internal force distribution for in-place rotation.

        The earlier tangent-force method can still produce an unwanted
        translation/slip because the fingertips are not symmetrically placed.
        This method solves for planar contact forces whose net force is zero
        while their moment about z_c is the requested rotation moment.

        The returned vectors are the actual contact forces applied by
        calc_grasp_tau; no per-finger force scaling is added afterward.
        """
        rotation_force = {
            finger: np.zeros(3, dtype=np.float64)
            for finger in tip_pos
        }

        if len(tip_pos) < 2 or abs(float(self.cfg.rotation_theta_rad)) < 1e-12:
            return rotation_force

        z_c = np.array([
            self.cfg.rotation_palm_normal_x,
            self.cfg.rotation_palm_normal_y,
            self.cfg.rotation_palm_normal_z,
        ], dtype=np.float64)
        z_c, z_norm = _normalize(z_c)
        if z_norm <= 0.0:
            return rotation_force

        # Build a stable orthonormal basis u, v on the plane perpendicular to z_c.
        ref = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        if abs(float(np.dot(ref, z_c))) > 0.90:
            ref = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        u, u_norm = _normalize(np.cross(z_c, ref))
        if u_norm <= 0.0:
            return rotation_force
        v = np.cross(z_c, u)
        v, v_norm = _normalize(v)
        if v_norm <= 0.0:
            return rotation_force

        fingers = list(tip_pos.keys())
        n = len(fingers)
        A = np.zeros((3, 2 * n), dtype=np.float64)

        for k, finger in enumerate(fingers):
            r = np.asarray(tip_pos[finger], dtype=np.float64) - np.asarray(rotation_center, dtype=np.float64)
            # Variables are actual planar forces: f_i = a_i*u + b_i*v.
            A[0, 2 * k] = 1.0       # sum a_i = 0
            A[1, 2 * k + 1] = 1.0   # sum b_i = 0
            A[2, 2 * k] = np.dot(np.cross(r, u), z_c)
            A[2, 2 * k + 1] = np.dot(np.cross(r, v), z_c)

        # Desired moment scale is chosen to be comparable to the old tangent-force
        # method: each finger contributed roughly rotation_gain*theta moment.
        command = float(self.cfg.rotation_gain) * float(self.cfg.rotation_theta_rad)
        desired_moment = command * float(n)
        b = np.array([0.0, 0.0, desired_moment], dtype=np.float64)

        # Minimum-norm solution of A x = b. If the contact geometry is nearly
        # singular, fall back to the old tangent method.
        try:
            x = A.T @ np.linalg.solve(A @ A.T + 1e-8 * np.eye(3), b)
        except np.linalg.LinAlgError:
            return self.calc_tangent_rotation_forces(rotation_center, tip_pos)

        actual_forces = {}
        max_norm = 0.0
        for k, finger in enumerate(fingers):
            f_actual = x[2 * k] * u + x[2 * k + 1] * v
            actual_forces[finger] = f_actual
            max_norm = max(max_norm, float(np.linalg.norm(f_actual)))

        force_limit = abs(float(self.cfg.rotation_force_limit))
        if force_limit > 0.0 and max_norm > force_limit:
            scale = force_limit / max(max_norm, 1e-9)
            for finger in actual_forces:
                actual_forces[finger] *= scale

        for finger, f_actual in actual_forces.items():
            rotation_force[finger] = f_actual

        return rotation_force

    def calc_tangent_rotation_forces(self, rotation_center, tip_pos):
        rotation_force = {
            finger: np.zeros(3, dtype=np.float64)
            for finger in tip_pos
        }

        if abs(float(self.cfg.rotation_theta_rad)) < 1e-12:
            return rotation_force

        z_c = np.array([
            self.cfg.rotation_palm_normal_x,
            self.cfg.rotation_palm_normal_y,
            self.cfg.rotation_palm_normal_z,
        ], dtype=np.float64)
        z_c, z_norm = _normalize(z_c)
        if z_norm <= 0.0:
            return rotation_force

        radius_min = max(float(self.cfg.rotation_radius_min), 1e-6)
        nominal_radius = max(
            float(getattr(self.cfg, "grasp_type7_rotation_nominal_radius", 0.060)),
            radius_min,
        )
        use_radius_compensation = bool(
            getattr(self.cfg, "grasp_type7_rotation_use_radius_compensation", False)
        )
        force_limit = abs(float(self.cfg.rotation_force_limit))
        command = float(self.cfg.rotation_gain) * float(self.cfg.rotation_theta_rad)

        for finger, pos in tip_pos.items():
            r = pos - rotation_center
            tangent = np.cross(z_c, r)
            tangent_hat, tangent_norm = _normalize(tangent)
            if tangent_norm <= 0.0:
                continue

            radius = max(np.linalg.norm(r), radius_min)
            denom = radius if use_radius_compensation else nominal_radius
            mag = command / denom
            mag = float(np.clip(mag, -force_limit, force_limit))
            rotation_force[finger] = mag * tangent_hat

        # Remove the net translation component of the rotation assist so that
        # the rotation forces sum to zero without per-finger weighting.
        if bool(getattr(self.cfg, "grasp_type7_rotation_zero_net_force", True)):
            forces = list(rotation_force.values())
            if forces:
                mean_force = np.mean(forces, axis=0)
                for finger in rotation_force:
                    rotation_force[finger] = rotation_force[finger] - mean_force

                    # Keep each corrected force bounded after balancing.
                    norm = np.linalg.norm(rotation_force[finger])
                    if force_limit > 0.0 and norm > force_limit:
                        rotation_force[finger] *= force_limit / max(norm, 1e-9)

        return rotation_force

    def calc_rotation_forces(self, rotation_center, tip_pos):
        mode = str(getattr(self.cfg, "grasp_type7_rotation_mode", "pure_moment")).strip().lower()
        if mode in ("pure_moment", "pure-moment", "moment", "couple"):
            return self.calc_pure_moment_rotation_forces(rotation_center, tip_pos)
        return self.calc_tangent_rotation_forces(rotation_center, tip_pos)

    def calc_center_hold_forces(self, cg_current, cg_ref, tip_pos):
        center_force = {
            finger: np.zeros(3, dtype=np.float64)
            for finger in tip_pos
        }

        if cg_ref is None or not bool(getattr(self.cfg, "grasp_type7_center_hold_enable", False)):
            return center_force

        err = np.asarray(cg_ref, dtype=np.float64) - np.asarray(cg_current, dtype=np.float64)
        force = float(getattr(self.cfg, "grasp_type7_center_hold_gain", 0.0)) * err

        if bool(getattr(self.cfg, "grasp_type7_center_hold_project_to_rotation_plane", True)):
            z_c = np.array([
                self.cfg.rotation_palm_normal_x,
                self.cfg.rotation_palm_normal_y,
                self.cfg.rotation_palm_normal_z,
            ], dtype=np.float64)
            z_c, z_norm = _normalize(z_c)
            if z_norm > 0.0:
                force = force - np.dot(force, z_c) * z_c

        limit = abs(float(getattr(self.cfg, "grasp_type7_center_hold_force_limit", 0.0)))
        norm = np.linalg.norm(force)
        if limit > 0.0 and norm > limit:
            force = force * (limit / max(norm, 1e-9))

        for finger in center_force:
            center_force[finger] = force.copy()

        return center_force

    def calc_grasp_tau(
        self,
        q,
        rotation_enabled=False,
        rotation_center=None,
        center_hold_target=None,
        center_hold_enabled=False,
    ):
        alpha, fhat, cg, cv, tip_pos = self.calc_alpha_and_forces(q)
        repel = self.calc_collision_avoidance_forces(tip_pos)
        rotation_ref = cg if rotation_center is None else np.asarray(rotation_center, dtype=np.float64)
        rotation_force = (
            self.calc_rotation_forces(rotation_ref, tip_pos)
            if rotation_enabled and self.cfg.rotation_enable_for_grasp_type7
            else {finger: np.zeros(3, dtype=np.float64) for finger in tip_pos}
        )
        center_hold_force = (
            self.calc_center_hold_forces(cg, center_hold_target, tip_pos)
            if center_hold_enabled and center_hold_target is not None
            else {finger: np.zeros(3, dtype=np.float64) for finger in tip_pos}
        )

        tau = np.zeros(20, dtype=np.float64)
        grasp_forces = {}
        total_forces = {}

        for finger in self.use_fingers:
            idxs = FINGER_JOINT_INDEX[finger]
            J = tip_jacobian(q, finger, eps=self.cfg.jacobian_eps)
            grasp_force = alpha[finger] * fhat[finger]
            total_force = (
                grasp_force
                + rotation_force[finger]
                + center_hold_force[finger]
                + repel[finger]
            )
            grasp_forces[finger] = grasp_force.copy()
            total_forces[finger] = total_force.copy()
            tau_finger = J.T @ total_force
            tau[idxs] = tau_finger * GRASP_TAU_SIGN[idxs]

        tau = np.clip(tau, -self.cfg.groped_tau_limit, self.cfg.groped_tau_limit)

        for joint_idx, limit in BASE_JOINT_TAU_LIMIT.items():
            tau[joint_idx] = np.clip(tau[joint_idx], -limit, limit)

        return GraspPolicyResult(
            tau=tau,
            alpha={int(finger): float(value) for finger, value in alpha.items()},
            cg=cg.copy(),
            cv=cv.copy(),
            fingertip_positions={
                int(finger): np.asarray(position, dtype=np.float64).copy()
                for finger, position in tip_pos.items()
            },
            grasp_forces=grasp_forces,
            rotation_forces={
                int(finger): np.asarray(force, dtype=np.float64).copy()
                for finger, force in rotation_force.items()
            },
            center_hold_forces={
                int(finger): np.asarray(force, dtype=np.float64).copy()
                for finger, force in center_hold_force.items()
            },
            collision_forces={
                int(finger): np.asarray(force, dtype=np.float64).copy()
                for finger, force in repel.items()
            },
            total_forces=total_forces,
        )
