import unittest

import numpy as np

from dg5f_grasp_control.sphere_estimation import (
    SphereEstimator,
    center_from_radius,
    center_from_radius_five_points,
    fit_sphere_radius,
)


def contacts(radius=0.0375, center=(0.1, 0.01, 0.08), tip_radius=0.010):
    directions = np.array([
        [1, 0, 0],
        [-1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
        [0, -1, -1],
    ], dtype=float)
    directions /= np.linalg.norm(directions, axis=1)[:, None]
    return np.asarray(center) + (radius + tip_radius) * directions


class SphereEstimationTest(unittest.TestCase):
    def test_fit_returns_object_radius_after_tip_correction(self):
        for radius in (0.026, 0.0375):
            fit = fit_sphere_radius(contacts(radius))
            self.assertIsNotNone(fit)
            self.assertAlmostEqual(fit[0], radius)

    def test_planar_geometry_is_rejected(self):
        angles = np.linspace(0, 2 * np.pi, 5, endpoint=False)
        points = np.column_stack([
            np.cos(angles), np.sin(angles), np.zeros(5)
        ]) * 0.05
        self.assertIsNone(fit_sphere_radius(points))

    def test_center_uses_estimated_radius(self):
        center = center_from_radius(contacts(tip_radius=0.009), 0.0375)
        np.testing.assert_allclose(center, [0.1, 0.01, 0.08], atol=1e-10)

    def test_five_point_center_uses_all_fingertips(self):
        center = center_from_radius_five_points(
            contacts(tip_radius=0.009),
            0.0375,
        )
        np.testing.assert_allclose(center, [0.1, 0.01, 0.08], atol=1e-10)

    def test_five_point_estimator_locks_from_five_finger_grasp(self):
        estimator = SphereEstimator(use_five_point_center=True)
        for now in np.linspace(1, 1.6, 13):
            estimator.update(
                contacts(), np.zeros(20), {1, 2, 3, 4, 5}, float(now)
            )
        self.assertAlmostEqual(estimator.radius, 0.0375)
        np.testing.assert_allclose(
            estimator.center,
            [0.1, 0.01, 0.08],
            atol=2e-4,
        )

    def test_tracking_keeps_center_on_reference_side(self):
        radius = 0.040
        contact_radius = 0.030
        height = np.sqrt((radius + 0.009) ** 2 - contact_radius ** 2)
        points = np.array([
            [contact_radius, 0.0, 0.0],
            [0.0, contact_radius, 0.0],
            [-contact_radius, 0.0, 0.0],
            [0.0, -contact_radius, 0.0],
            [0.0, 0.0, 0.0],
        ])
        estimator = SphereEstimator()
        estimator.radius = radius
        estimator.last_center = np.array([0.0, 0.0, height])

        estimator.update(points, np.zeros(20), {1, 2, 3, 4}, 1.0)
        estimator.update(points, np.zeros(20), {1, 2, 3, 4}, 1.081)

        np.testing.assert_allclose(
            estimator.center,
            [0.0, 0.0, height],
            atol=1e-10,
        )

    def test_radius_locks_after_stable_five_sample_window(self):
        estimator = SphereEstimator()
        for now in np.linspace(1, 1.6, 13):
            estimator.update(
                contacts(), np.zeros(20), {1, 2, 3, 4, 5}, float(now)
            )
        self.assertAlmostEqual(estimator.radius, 0.0375)
        self.assertIsNotNone(estimator.center)
        np.testing.assert_allclose(estimator.center, estimator.last_center)

    def test_motion_prevents_radius_lock(self):
        estimator = SphereEstimator()
        for now in np.linspace(1, 1.6, 13):
            estimator.update(
                contacts(), np.ones(20), {1, 2, 3, 4, 5}, float(now)
            )
        self.assertIsNone(estimator.radius)


if __name__ == "__main__":
    unittest.main()
