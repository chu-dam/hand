"""Sphere geometry estimation, with a once-per-grasp radius lock."""

from collections import deque
from itertools import combinations

import numpy as np


RADIUS_FIT_TIP_RADIUS_M = 0.010
CENTER_TIP_RADIUS_M = 0.009
CENTER_CONTINUITY_WEIGHT = 0.05


def fit_sphere_radius(points):
    points = np.asarray(points, dtype=float)
    if points.shape != (5, 3) or not np.all(np.isfinite(points)):
        return None
    origin = points.mean(axis=0)
    offsets = points - origin
    _, spread, _ = np.linalg.svd(offsets, full_matrices=False)
    if spread[0] <= 1e-12 or spread[-1] / spread[0] < 0.05:
        return None
    try:
        center = origin + np.linalg.solve(
            offsets.T @ offsets,
            0.5 * offsets.T @ np.sum(offsets ** 2, axis=1),
        )
    except np.linalg.LinAlgError:
        return None
    distances = np.linalg.norm(points - center, axis=1)
    radius = distances.mean() - RADIUS_FIT_TIP_RADIUS_M
    rms = np.sqrt(np.mean((distances - distances.mean()) ** 2))
    if not np.isfinite(radius) or radius <= 0 or rms > 0.002:
        return None
    return float(radius), float(rms)


def center_from_radius(points, radius, reference_center=None):
    """Use three thumb-based triangles; pinky is not used."""
    points = np.asarray(points, dtype=float)[:4]
    if points.shape != (4, 3) or not np.all(np.isfinite(points)):
        return None
    if not np.isfinite(radius) or radius <= 0:
        return None
    if reference_center is not None:
        reference_center = np.asarray(reference_center, dtype=float)
        if (
            reference_center.shape != (3,)
            or not np.all(np.isfinite(reference_center))
        ):
            reference_center = None
    distance = radius + CENTER_TIP_RADIUS_M
    centers = []
    for first, second in combinations((1, 2, 3), 2):
        p = points[0]
        u, v = points[first] - p, points[second] - p
        normal = np.cross(u, v)
        squared = normal @ normal
        if squared < 1e-12:
            continue
        circumcenter = p + (
            (u @ u) * np.cross(v, normal) + (v @ v) * np.cross(normal, u)
        ) / (2 * squared)
        rho = np.linalg.norm(circumcenter - p)
        if rho - distance > 0.002 + 1e-8:
            continue
        offset = normal / np.sqrt(squared) * np.sqrt(
            max(0, distance ** 2 - rho ** 2)
        )
        candidates = [circumcenter + offset, circumcenter - offset]
        if reference_center is not None:
            errors = [
                np.linalg.norm(candidate - reference_center)
                for candidate in candidates
            ]
        else:
            other = next(i for i in (1, 2, 3) if i not in (first, second))
            errors = [
                abs(np.linalg.norm(candidate - points[other]) - distance)
                for candidate in candidates
            ]
        centers.append(candidates[1] if errors[1] < errors[0] else candidates[0])
    return np.mean(centers, axis=0) if centers else None


def center_from_radius_five_points(points, radius, reference_center=None):
    """Fit a fixed-radius center from all five fingertips."""
    points = np.asarray(points, dtype=float)
    if points.shape != (5, 3) or not np.all(np.isfinite(points)):
        return None
    if not np.isfinite(radius) or radius <= 0:
        return None
    reference = None
    if reference_center is not None:
        reference = np.asarray(reference_center, dtype=float)
        if reference.shape != (3,) or not np.all(np.isfinite(reference)):
            reference = None
    if reference is None:
        matrix = 2.0 * (points[1:] - points[0])
        values = (
            np.sum(points[1:] ** 2, axis=1)
            - np.sum(points[0] ** 2)
        )
        center, _, rank, _ = np.linalg.lstsq(matrix, values, rcond=None)
        if rank < 3:
            center = center_from_radius(points, radius)
            if center is None:
                return None
    else:
        center = reference.copy()

    distance = radius + CENTER_TIP_RADIUS_M
    for _ in range(10):
        vectors = center - points
        lengths = np.linalg.norm(vectors, axis=1)
        if np.any(lengths <= 1e-12):
            return None
        jacobian = vectors / lengths[:, None]
        residual = lengths - distance
        if reference is not None:
            weight = np.sqrt(CENTER_CONTINUITY_WEIGHT)
            jacobian = np.vstack((jacobian, weight * np.eye(3)))
            residual = np.concatenate((residual, weight * (center - reference)))
        step, _, rank, _ = np.linalg.lstsq(jacobian, -residual, rcond=None)
        if rank < 3 or not np.all(np.isfinite(step)):
            return None
        center += step
        if np.linalg.norm(step) <= 1e-6:
            break

    residual = np.linalg.norm(points - center, axis=1) - distance
    if np.sqrt(np.mean(residual ** 2)) > 0.003:
        return None
    return center


class SphereEstimator:
    def __init__(self, *, use_five_point_center=False):
        self.use_five_point_center = bool(use_five_point_center)
        self.reset()

    def reset(self):
        self.radius = None
        self.initial_center = None
        self.center = None
        self.last_center = None
        self.rms = 0.0
        self.state = "radius_pending"
        self.stable_since = None
        self.samples = deque(maxlen=5)
        self.last_sample_at = None
        self.last_update_at = None

    def update(self, points, qdot, active_fingers, now, *, eligible=True):
        self.center = None
        self.state = "center_waiting" if self.radius is not None else "radius_pending"
        if not np.isfinite(now):
            self.stable_since = None
            self.samples.clear()
            return
        if self.last_update_at is not None and now < self.last_update_at:
            self.stable_since = None
            self.last_sample_at = None
            self.samples.clear()
        self.last_update_at = now
        needed = (
            {1, 2, 3, 4, 5}
            if self.radius is None or self.use_five_point_center
            else {1, 2, 3, 4}
        )
        qdot = np.asarray(qdot, dtype=float)
        count = 20 if self.radius is None or self.use_five_point_center else 16
        ready = (
            eligible
            and needed.issubset(active_fingers)
            and qdot.shape == (20,)
            and np.all(np.isfinite(qdot[:count]))
            and np.max(np.abs(qdot[:count])) < 0.05
        )
        if not ready:
            self.stable_since = None
            self.samples.clear()
            self.last_sample_at = None
            return
        if self.stable_since is None:
            self.stable_since = now
        if self.radius is None:
            fit = fit_sphere_radius(points)
            if fit is None:
                self.state = "radius_invalid"
                self.stable_since = None
                self.samples.clear()
                self.last_sample_at = None
                return
            self.state = "radius_settling"
            if self.last_sample_at is None or now - self.last_sample_at >= 0.05:
                self.samples.append(fit)
                self.last_sample_at = now
            if now - self.stable_since < 0.3 or len(self.samples) < 5:
                return
            radii = np.array([sample[0] for sample in self.samples])
            radius = float(np.median(radii))
            if np.max(np.abs(radii - radius)) > 0.001:
                return
            center = (
                center_from_radius_five_points(points, radius)
                if self.use_five_point_center
                else center_from_radius(points, radius)
            )
            if center is None:
                self.state = "center_invalid"
                return
            self.radius = radius
            self.rms = float(np.median([sample[1] for sample in self.samples]))
            self.initial_center = center.copy()
            self.center = center
            self.last_center = center.copy()
            self.state = "tracking"
            return
        if now - self.stable_since < 0.08:
            return
        self.center = (
            center_from_radius_five_points(points, self.radius, self.last_center)
            if self.use_five_point_center
            else center_from_radius(points, self.radius, self.last_center)
        )
        if self.center is not None:
            self.last_center = self.center.copy()
        self.state = "tracking" if self.center is not None else "center_invalid"
