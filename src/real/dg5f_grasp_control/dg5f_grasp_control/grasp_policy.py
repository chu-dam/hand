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

ALPHA_DISTRIBUTION_LEGACY = "legacy"
ALPHA_DISTRIBUTION_THUMB_DISTANCE_PROPORTIONAL = (
    "thumb_distance_proportional"
)


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

    def _calc_force_balanced_alpha(
        self,
        fhat,
        reference_alpha,
        fixed_finger=None,
    ):
        """Project reference coefficients onto a non-negative force balance.

        Alpha for ``fixed_finger`` remains fixed. With at most four remaining
        fingers, enumerating their active subsets is small and avoids adding a
        SciPy dependency. Among exact non-negative solutions, keep the one
        closest to the supplied reference distribution.
        """

        fingers = list(self.use_fingers)
        if len(fingers) <= 2:
            return dict(reference_alpha)

        fixed_finger = fingers[0] if fixed_finger is None else int(fixed_finger)
        if fixed_finger not in fingers:
            raise ValueError("fixed force-balance finger must be active")
        variable_fingers = [
            finger for finger in fingers if finger != fixed_finger
        ]
        first_alpha = float(reference_alpha[fixed_finger])
        if not np.isfinite(first_alpha) or first_alpha < 0.0:
            raise ValueError("alpha1 must be finite and non-negative")
        matrix = np.column_stack([fhat[finger] for finger in variable_fingers])
        target = -first_alpha * fhat[fixed_finger]
        reference = np.array(
            [reference_alpha[finger] for finger in variable_fingers],
            dtype=np.float64,
        )
        if not np.all(np.isfinite(matrix)) or not np.all(np.isfinite(target)):
            raise ValueError("force directions must be finite")
        if not np.all(np.isfinite(reference)):
            raise ValueError("reference alpha values must be finite")

        max_ratio = float(self.cfg.rotation_force_balance_max_alpha_ratio)
        if not np.isfinite(max_ratio) or max_ratio <= 0.0:
            raise ValueError(
                "rotation force-balance alpha ratio must be finite and > 0"
            )
        alpha_limit = max_ratio * first_alpha

        alpha_tolerance = 1e-10 * max(1.0, float(np.max(np.abs(reference))))
        residual_tolerance = 1e-8 * max(1.0, float(np.linalg.norm(target)))
        best = None

        # A five-finger grasp has only four free coefficients: at most 15
        # non-empty active subsets need to be checked.
        for mask in range(1, 1 << len(variable_fingers)):
            active = [
                index
                for index in range(len(variable_fingers))
                if mask & (1 << index)
            ]
            active_matrix = matrix[:, active]
            active_reference = reference[active]

            try:
                correction = np.linalg.lstsq(
                    active_matrix,
                    target - active_matrix @ active_reference,
                    rcond=None,
                )[0]
            except np.linalg.LinAlgError:
                continue
            active_alpha = active_reference + correction
            if not np.all(np.isfinite(active_alpha)):
                continue
            candidate_force = active_matrix @ active_alpha
            residual = float(np.linalg.norm(candidate_force - target))
            if residual > residual_tolerance:
                continue
            if np.any(active_alpha < -alpha_tolerance):
                continue

            candidate = np.zeros(len(variable_fingers), dtype=np.float64)
            candidate[active] = np.maximum(active_alpha, 0.0)
            if not np.all(np.isfinite(candidate)):
                continue
            if candidate.size and float(np.max(candidate)) > alpha_limit:
                continue
            clipped_residual = float(np.linalg.norm(matrix @ candidate - target))
            if clipped_residual > residual_tolerance:
                continue
            score = float(np.linalg.norm(candidate - reference))
            if best is None or score < best[0]:
                best = (score, candidate)

        if best is None:
            raise ValueError(
                "no non-negative alpha solution satisfies force balance"
            )

        balanced = {fixed_finger: float(reference_alpha[fixed_finger])}
        for finger, value in zip(variable_fingers, best[1]):
            balanced[finger] = float(value)
        return balanced

    def calc_alpha_and_forces(
        self,
        q,
        thumb_centroid_bias_override=None,
        force_balance_blend=0.0,
        alpha_distribution_mode=ALPHA_DISTRIBUTION_LEGACY,
    ):
        alpha_distribution_mode = str(alpha_distribution_mode).strip().lower()
        supported_modes = {
            ALPHA_DISTRIBUTION_LEGACY,
            ALPHA_DISTRIBUTION_THUMB_DISTANCE_PROPORTIONAL,
        }
        if alpha_distribution_mode not in supported_modes:
            raise ValueError(
                "alpha distribution mode must be 'legacy' or "
                "'thumb_distance_proportional'"
            )

        tip_pos = {
            finger: tip_position(q, finger)
            for finger in self.use_fingers
        }

        points = np.array([tip_pos[finger] for finger in self.use_fingers])
        cg = polygon_centroid_3d(points)

        use_thumb_distance_proportional = (
            alpha_distribution_mode
            == ALPHA_DISTRIBUTION_THUMB_DISTANCE_PROPORTIONAL
        )

        if use_thumb_distance_proportional:
            if 1 not in tip_pos:
                raise ValueError(
                    "thumb-distance-proportional distribution requires "
                    "active thumb finger ID 1"
                )
            # Regular grasp types 1..5 use the geometric centroid directly.
            # The optional legacy policy retains its thumb-biased centroid.
            cv = cg
        elif len(self.use_fingers) == 2:
            # Pinch grasp: keep the virtual centroid at the geometric midpoint
            # so the two equal-magnitude contact forces remain opposite.
            cv = cg
        elif 1 in tip_pos:
            thumb_pos = tip_pos[1]
            thumb_centroid_bias = (
                self.cfg.thumb_centroid_bias
                if thumb_centroid_bias_override is None
                else float(thumb_centroid_bias_override)
            )
            if not np.isfinite(thumb_centroid_bias):
                raise ValueError("thumb centroid bias must be finite")
            cv = cg + thumb_centroid_bias * (thumb_pos - cg)
        else:
            cv = cg

        fhat = {}
        dist = {}
        for finger in self.use_fingers:
            diff = cv - tip_pos[finger]
            dist[finger] = max(np.linalg.norm(diff), 1e-6)
            fhat[finger] = self.cfg.groped_force_direction_sign * diff / dist[finger]

        alpha1 = float(self.cfg.alpha1)
        if not np.isfinite(alpha1) or alpha1 < 0.0:
            raise ValueError("alpha1 must be finite and non-negative")

        alpha = {}

        if len(self.use_fingers) == 2:
            alpha[self.use_fingers[0]] = alpha1
            alpha[self.use_fingers[1]] = alpha1
            return alpha, fhat, cg, cv, tip_pos

        if use_thumb_distance_proportional:
            thumb_finger = 1
            thumb_distance = dist[thumb_finger]
            alpha[thumb_finger] = alpha1
            for finger in self.use_fingers:
                if finger == thumb_finger:
                    continue
                alpha[finger] = alpha1 * dist[finger] / thumb_distance

            max_ratio = float(self.cfg.rotation_force_balance_max_alpha_ratio)
            if not np.isfinite(max_ratio) or max_ratio <= 0.0:
                raise ValueError(
                    "force-balance alpha ratio must be finite and > 0"
                )
            alpha_limit = max_ratio * alpha1
            alpha_tolerance = 1e-10 * max(1.0, alpha_limit)
            if any(
                not np.isfinite(value)
                or value < 0.0
                or value > alpha_limit + alpha_tolerance
                for finger, value in alpha.items()
                if finger != thumb_finger
            ):
                raise ValueError(
                    "thumb-distance-proportional alpha exceeds the configured "
                    "force-balance ratio"
                )

            # For three contacts Cg is the arithmetic vertex mean, so the
            # distance-proportional radial forces cancel analytically. For
            # four/five contacts the signed-area polygon centroid generally is
            # not the arithmetic mean; retain the proportional values as the
            # nominal distribution, then enforce exact non-negative 3-D force
            # balance while keeping the thumb coefficient fixed at alpha1.
            if len(self.use_fingers) >= 4:
                alpha = self._calc_force_balanced_alpha(
                    fhat,
                    alpha,
                    fixed_finger=thumb_finger,
                )

            return alpha, fhat, cg, cv, tip_pos

        first_finger = self.use_fingers[0]
        pivot_finger = self.use_fingers[-1]
        alpha[first_finger] = alpha1

        for finger in self.use_fingers[1:-1]:
            alpha[finger] = dist[first_finger] / dist[finger] * alpha1

        force_sum = np.zeros(3, dtype=np.float64)
        for finger in self.use_fingers[:-1]:
            force_sum += alpha[finger] * fhat[finger]

        alpha[pivot_finger] = np.linalg.norm(force_sum)

        balance_blend = float(force_balance_blend)
        if not np.isfinite(balance_blend) or not 0.0 <= balance_blend <= 1.0:
            raise ValueError("force balance blend must be finite and within [0, 1]")
        if balance_blend > 0.0:
            balanced_alpha = self._calc_force_balanced_alpha(fhat, alpha)
            alpha = {
                finger: (
                    (1.0 - balance_blend) * alpha[finger]
                    + balance_blend * balanced_alpha[finger]
                )
                for finger in self.use_fingers
            }

        return alpha, fhat, cg, cv, tip_pos

    def calc_zero_grasp_result(self, q):
        """Return current contact geometry with every Cartesian force at zero.

        This is the fail-closed result used when the regular proportional
        distribution cannot satisfy its bounded non-negative constraints and
        no previous valid command remains to ramp down.
        """

        tip_pos = {
            finger: tip_position(q, finger)
            for finger in self.use_fingers
        }
        points = np.array([tip_pos[finger] for finger in self.use_fingers])
        try:
            cg = polygon_centroid_3d(points)
        except ValueError:
            # Force output is zero in this path; an arithmetic mean keeps the
            # diagnostic geometry finite even if the contact polygon itself
            # is degenerate and has no signed-area centroid.
            cg = np.mean(points, axis=0)
        zero_forces = {
            finger: np.zeros(3, dtype=np.float64)
            for finger in self.use_fingers
        }

        return GraspPolicyResult(
            tau=np.zeros(20, dtype=np.float64),
            alpha={finger: 0.0 for finger in self.use_fingers},
            cg=cg.copy(),
            cv=cg.copy(),
            fingertip_positions={
                int(finger): np.asarray(position, dtype=np.float64).copy()
                for finger, position in tip_pos.items()
            },
            grasp_forces={
                finger: force.copy() for finger, force in zero_forces.items()
            },
            rotation_forces={
                finger: force.copy() for finger, force in zero_forces.items()
            },
            center_hold_forces={
                finger: force.copy() for finger, force in zero_forces.items()
            },
            collision_forces={
                finger: force.copy() for finger, force in zero_forces.items()
            },
            total_forces={
                finger: force.copy() for finger, force in zero_forces.items()
            },
        )

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

    def calc_tau_from_total_forces(self, q, total_forces):
        """Map supplied Cartesian fingertip forces through the current Jacobians."""

        tau = np.zeros(20, dtype=np.float64)
        for finger in self.use_fingers:
            force = np.asarray(
                total_forces.get(finger, np.zeros(3)),
                dtype=np.float64,
            )
            if force.shape != (3,) or not np.all(np.isfinite(force)):
                raise ValueError("each fingertip force must be a finite 3-vector")
            idxs = FINGER_JOINT_INDEX[finger]
            jacobian = tip_jacobian(q, finger, eps=self.cfg.jacobian_eps)
            tau[idxs] = jacobian.T @ force * GRASP_TAU_SIGN[idxs]

        tau = np.clip(
            tau,
            -self.cfg.groped_tau_limit,
            self.cfg.groped_tau_limit,
        )
        for joint_idx, limit in BASE_JOINT_TAU_LIMIT.items():
            tau[joint_idx] = np.clip(tau[joint_idx], -limit, limit)
        return tau


    def calc_grasp_tau(
        self,
        q,
        thumb_centroid_bias_override=None,
        force_balance_blend=0.0,
        alpha_distribution_mode=ALPHA_DISTRIBUTION_LEGACY,
    ):
        alpha, fhat, cg, cv, tip_pos = self.calc_alpha_and_forces(
            q,
            thumb_centroid_bias_override=thumb_centroid_bias_override,
            force_balance_blend=force_balance_blend,
            alpha_distribution_mode=alpha_distribution_mode,
        )
        repel = self.calc_collision_avoidance_forces(tip_pos)
        rotation_force = {
            finger: np.zeros(3, dtype=np.float64) for finger in tip_pos
        }
        center_hold_force = {
            finger: np.zeros(3, dtype=np.float64) for finger in tip_pos
        }

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
