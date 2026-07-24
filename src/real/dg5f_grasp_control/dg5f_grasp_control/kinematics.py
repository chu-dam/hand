import numpy as np

from dg5f_grasp_control.hand_model import FINGER_JOINT_INDEX


# Fixed transforms of the four joint bodies in dg5fs_left_w_mount.xml.
# Quaternions use MuJoCo's (w, x, y, z) convention. Keeping these compact
# constants here avoids parsing meshes/XML in the 200 Hz control loop while
# preserving the same link_base frame as tesollo_forward_kinematics().
_FINGER_CHAIN_POSITIONS = {
    1: (
        (0.0017, -0.018, 0.0298),
        (0.0, 0.0, 0.0262),
        (0.0381, 0.0, 0.0),
        (0.0334, 0.0, 0.0),
    ),
    2: (
        (0.0017, -0.028, 0.08365),
        (0.02415, 0.0, 0.0126),
        (0.0334, 0.0, 0.0),
        (0.0334, 0.0, 0.0),
    ),
    3: (
        (0.0017, -0.005, 0.08865),
        (0.02415, 0.0, 0.0126),
        (0.0334, 0.0, 0.0),
        (0.0334, 0.0, 0.0),
    ),
    4: (
        (0.0017, 0.018, 0.08065),
        (0.02415, 0.0, 0.0126),
        (0.0334, 0.0, 0.0),
        (0.0334, 0.0, 0.0),
    ),
    5: (
        (0.013, 0.01805, 0.0364),
        (0.02445, 0.0, 0.0307),
        (0.0, 0.0272, 0.0),
        (0.0334, 0.0, 0.0),
    ),
}

_FINGER_CHAIN_QUATERNIONS = {
    1: (
        (0.499998163397, -0.499999999997, 0.500001836603, -0.499999999997),
        (0.707105482511, 0.707108079859, 0.0, 0.0),
        (0.707105482511, -0.707108079859, 0.0, 0.0),
        (1.0, 0.0, 0.0, 0.0),
    ),
    2: (
        (-2.59734346696e-6, 0.707105482506, 2.59735300756e-6, 0.707108079855),
        (0.707105482511, 0.707108079859, 0.0, 0.0),
        (1.0, 0.0, 0.0, 0.0),
        (1.0, 0.0, 0.0, 0.0),
    ),
    3: (
        (-2.59734346696e-6, 0.707105482506, 2.59735300756e-6, 0.707108079855),
        (0.707105482511, 0.707108079859, 0.0, 0.0),
        (1.0, 0.0, 0.0, 0.0),
        (1.0, 0.0, 0.0, 0.0),
    ),
    4: (
        (-2.59734346696e-6, 0.707105482506, 2.59735300756e-6, 0.707108079855),
        (0.707105482511, 0.707108079859, 0.0, 0.0),
        (1.0, 0.0, 0.0, 0.0),
        (1.0, 0.0, 0.0, 0.0),
    ),
    5: (
        (0.707105482511, 0.0, 0.0, 0.707108079859),
        (0.707105482511, 0.707108079859, 0.0, 0.0),
        (0.499998163397, 0.499999999997, 0.500001836603, 0.499999999997),
        (1.0, 0.0, 0.0, 0.0),
    ),
}

_FINGER_TIP_LENGTHS = {
    1: 0.0318,
    2: 0.01978,
    3: 0.0209,
    4: 0.0209,
    5: 0.0318,
}


def _quaternion_matrix(quaternion):
    w, x, y, z = np.asarray(quaternion, dtype=np.float64)
    return np.array(
        [
            [
                1.0 - 2.0 * (y * y + z * z),
                2.0 * (x * y - z * w),
                2.0 * (x * z + y * w),
            ],
            [
                2.0 * (x * y + z * w),
                1.0 - 2.0 * (x * x + z * z),
                2.0 * (y * z - x * w),
            ],
            [
                2.0 * (x * z - y * w),
                2.0 * (y * z + x * w),
                1.0 - 2.0 * (x * x + y * y),
            ],
        ],
        dtype=np.float64,
    )


def _rotation_z(angle):
    sine = float(np.sin(angle))
    cosine = float(np.cos(angle))
    return np.array(
        [
            [cosine, -sine, 0.0],
            [sine, cosine, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


_FINGER_CHAIN_ROTATIONS = {
    finger: tuple(
        _quaternion_matrix(quaternion)
        for quaternion in quaternions
    )
    for finger, quaternions in _FINGER_CHAIN_QUATERNIONS.items()
}


def tesollo_forward_kinematics(q):
    s = np.sin(q)
    c = np.cos(q)
    x = np.zeros(15, dtype=np.float64)

    x[0] = (
        -0.0318 * s[1] * s[2] * s[3]
        + 0.0318 * s[1] * c[2] * c[3]
        + 0.0334 * s[1] * c[2]
        + 0.0381 * s[1]
        + 0.0279
    )
    x[1] = (
        0.0318 * (s[0] * s[2] - c[0] * c[1] * c[2]) * c[3]
        + 0.0318 * (s[0] * c[2] + s[2] * c[0] * c[1]) * s[3]
        + 0.0334 * s[0] * s[2]
        - 0.0334 * c[0] * c[1] * c[2]
        - 0.0381 * c[0] * c[1]
        - 0.018
    )
    x[2] = (
        0.0318 * (s[0] * s[2] * c[1] - c[0] * c[2]) * s[3]
        + 0.0318 * (-s[0] * c[1] * c[2] - s[2] * c[0]) * c[3]
        - 0.0334 * s[0] * c[1] * c[2]
        - 0.0381 * s[0] * c[1]
        - 0.0334 * s[2] * c[0]
        + 0.0298
    )

    x[3] = (
        0.01978 * (-s[5] * s[6] + c[5] * c[6]) * s[7]
        + 0.01978 * (s[5] * c[6] + s[6] * c[5]) * c[7]
        + 0.0334 * s[5] * c[6]
        + 0.0334 * s[5]
        + 0.0334 * s[6] * c[5]
        + 0.0143
    )
    x[4] = (
        0.01978 * (s[4] * s[5] * s[6] - s[4] * c[5] * c[6]) * c[7]
        + 0.01978 * (s[4] * s[5] * c[6] + s[4] * s[6] * c[5]) * s[7]
        + 0.0334 * s[4] * s[5] * s[6]
        - 0.0334 * s[4] * c[5] * c[6]
        - 0.0334 * s[4] * c[5]
        - 0.02415 * s[4]
        - 0.028
    )
    x[5] = (
        0.01978 * (-s[5] * s[6] * c[4] + c[4] * c[5] * c[6]) * c[7]
        + 0.01978 * (-s[5] * c[4] * c[6] - s[6] * c[4] * c[5]) * s[7]
        - 0.0334 * s[5] * s[6] * c[4]
        + 0.0334 * c[4] * c[5] * c[6]
        + 0.0334 * c[4] * c[5]
        + 0.02415 * c[4]
        + 0.08365
    )

    x[6] = (
        0.0209 * (-s[10] * s[9] + c[10] * c[9]) * s[11]
        + 0.0209 * (s[10] * c[9] + s[9] * c[10]) * c[11]
        + 0.0334 * s[10] * c[9]
        + 0.0334 * s[9] * c[10]
        + 0.0334 * s[9]
        + 0.0143
    )
    x[7] = (
        0.0209 * (s[10] * s[8] * s[9] - s[8] * c[10] * c[9]) * c[11]
        + 0.0209 * (s[10] * s[8] * c[9] + s[8] * s[9] * c[10]) * s[11]
        + 0.0334 * s[10] * s[8] * s[9]
        - 0.0334 * s[8] * c[10] * c[9]
        - 0.0334 * s[8] * c[9]
        - 0.02415 * s[8]
        - 0.005
    )
    x[8] = (
        0.0209 * (-s[10] * s[9] * c[8] + c[10] * c[8] * c[9]) * c[11]
        + 0.0209 * (-s[10] * c[8] * c[9] - s[9] * c[10] * c[8]) * s[11]
        - 0.0334 * s[10] * s[9] * c[8]
        + 0.0334 * c[10] * c[8] * c[9]
        + 0.0334 * c[8] * c[9]
        + 0.02415 * c[8]
        + 0.08865
    )

    x[9] = (
        0.0209 * (-s[13] * s[14] + c[13] * c[14]) * s[15]
        + 0.0209 * (s[13] * c[14] + s[14] * c[13]) * c[15]
        + 0.0334 * s[13] * c[14]
        + 0.0334 * s[13]
        + 0.0334 * s[14] * c[13]
        + 0.0143
    )
    x[10] = (
        0.0209 * (s[12] * s[13] * s[14] - s[12] * c[13] * c[14]) * c[15]
        + 0.0209 * (s[12] * s[13] * c[14] + s[12] * s[14] * c[13]) * s[15]
        + 0.0334 * s[12] * s[13] * s[14]
        - 0.0334 * s[12] * c[13] * c[14]
        - 0.0334 * s[12] * c[13]
        - 0.02415 * s[12]
        + 0.018
    )
    x[11] = (
        0.0209 * (-s[13] * s[14] * c[12] + c[12] * c[13] * c[14]) * c[15]
        + 0.0209 * (-s[13] * c[12] * c[14] - s[14] * c[12] * c[13]) * s[15]
        - 0.0334 * s[13] * s[14] * c[12]
        + 0.0334 * c[12] * c[13] * c[14]
        + 0.0334 * c[12] * c[13]
        + 0.02415 * c[12]
        + 0.08065
    )

    x[12] = (
        0.0318 * (-s[16] * s[17] * s[18] + c[16] * c[18]) * s[19]
        + 0.0318 * (s[16] * s[17] * c[18] + s[18] * c[16]) * c[19]
        + 0.0334 * s[16] * s[17] * c[18]
        + 0.0272 * s[16] * s[17]
        - 0.02445 * s[16]
        + 0.0334 * s[18] * c[16]
        + 0.013
    )
    x[13] = (
        0.0318 * (s[16] * s[18] - s[17] * c[16] * c[18]) * c[19]
        + 0.0318 * (s[16] * c[18] + s[17] * s[18] * c[16]) * s[19]
        + 0.0334 * s[16] * s[18]
        - 0.0334 * s[17] * c[16] * c[18]
        - 0.0272 * s[17] * c[16]
        + 0.02445 * c[16]
        + 0.01805
    )
    x[14] = (
        -0.0318 * s[18] * s[19] * c[17]
        + 0.0318 * c[17] * c[18] * c[19]
        + 0.0334 * c[17] * c[18]
        + 0.0272 * c[17]
        + 0.0671
    )

    return x.reshape(5, 3)



def tip_position(q, finger):
    return tesollo_forward_kinematics(q)[finger - 1].copy()


def finger_link_points(q, finger):
    """Return joint 1..4 and fingertip centerline points in link_base."""

    q = np.asarray(q, dtype=np.float64)
    if q.shape != (20,) or not np.all(np.isfinite(q)):
        raise ValueError("q must be a finite 20-vector")
    finger = int(finger)
    if finger not in FINGER_JOINT_INDEX:
        raise ValueError("finger must be one of 1, 2, 3, 4, or 5")

    rotation = np.eye(3, dtype=np.float64)
    position = np.zeros(3, dtype=np.float64)
    points = []
    joint_indices = FINGER_JOINT_INDEX[finger]
    for local_index, (translation, fixed_rotation) in enumerate(
        zip(
            _FINGER_CHAIN_POSITIONS[finger],
            _FINGER_CHAIN_ROTATIONS[finger],
        )
    ):
        position = position + rotation @ np.asarray(
            translation,
            dtype=np.float64,
        )
        rotation = rotation @ fixed_rotation
        points.append(position.copy())
        rotation = rotation @ _rotation_z(q[joint_indices[local_index]])

    fingertip = position + rotation @ np.array(
        [_FINGER_TIP_LENGTHS[finger], 0.0, 0.0],
        dtype=np.float64,
    )
    points.append(fingertip)
    return np.asarray(points, dtype=np.float64)


def finger_capsule_segments(q, finger):
    """Return four centerline segments for one simplified finger model."""

    points = finger_link_points(q, finger)
    return np.stack((points[:-1], points[1:]), axis=1)


def point_segment_distance(point, segment_start, segment_end):
    point = np.asarray(point, dtype=np.float64)
    segment_start = np.asarray(segment_start, dtype=np.float64)
    segment_end = np.asarray(segment_end, dtype=np.float64)
    direction = segment_end - segment_start
    length_squared = float(np.dot(direction, direction))
    if length_squared <= 1e-18:
        return float(np.linalg.norm(point - segment_start))
    parameter = float(
        np.clip(
            np.dot(point - segment_start, direction) / length_squared,
            0.0,
            1.0,
        )
    )
    closest = segment_start + parameter * direction
    return float(np.linalg.norm(point - closest))


def segment_segment_distance(start_a, end_a, start_b, end_b):
    """Return the Euclidean minimum distance between two 3-D segments."""

    start_a = np.asarray(start_a, dtype=np.float64)
    end_a = np.asarray(end_a, dtype=np.float64)
    start_b = np.asarray(start_b, dtype=np.float64)
    end_b = np.asarray(end_b, dtype=np.float64)
    direction_a = end_a - start_a
    direction_b = end_b - start_b
    relative = start_a - start_b
    aa = float(np.dot(direction_a, direction_a))
    ab = float(np.dot(direction_a, direction_b))
    bb = float(np.dot(direction_b, direction_b))
    ar = float(np.dot(direction_a, relative))
    br = float(np.dot(direction_b, relative))

    distances = [
        point_segment_distance(start_a, start_b, end_b),
        point_segment_distance(end_a, start_b, end_b),
        point_segment_distance(start_b, start_a, end_a),
        point_segment_distance(end_b, start_a, end_a),
    ]
    determinant = aa * bb - ab * ab
    if determinant > 1e-18:
        parameter_a = (ab * br - bb * ar) / determinant
        parameter_b = (aa * br - ab * ar) / determinant
        if 0.0 <= parameter_a <= 1.0 and 0.0 <= parameter_b <= 1.0:
            point_a = start_a + parameter_a * direction_a
            point_b = start_b + parameter_b * direction_b
            distances.append(float(np.linalg.norm(point_a - point_b)))
    return min(distances)


def finger_capsule_clearance(
    q,
    finger_a,
    finger_b,
    capsule_radius,
    *,
    first_segment=0,
):
    """Return minimum surface clearance between two uniform-radius fingers."""

    capsule_radius = float(capsule_radius)
    if not np.isfinite(capsule_radius) or capsule_radius < 0.0:
        raise ValueError("capsule_radius must be finite and non-negative")
    first_segment = int(first_segment)
    if first_segment < 0 or first_segment >= 4:
        raise ValueError("first_segment must be between 0 and 3")
    segments_a = finger_capsule_segments(q, finger_a)[first_segment:]
    segments_b = finger_capsule_segments(q, finger_b)[first_segment:]
    return capsule_segments_clearance(
        segments_a,
        segments_b,
        capsule_radius,
    )


def capsule_segments_clearance(segments_a, segments_b, capsule_radius):
    """Return uniform-radius capsule clearance for precomputed segments."""

    capsule_radius = float(capsule_radius)
    if not np.isfinite(capsule_radius) or capsule_radius < 0.0:
        raise ValueError("capsule_radius must be finite and non-negative")
    segments_a = np.asarray(segments_a, dtype=np.float64)
    segments_b = np.asarray(segments_b, dtype=np.float64)
    if (
        segments_a.ndim != 3
        or segments_a.shape[1:] != (2, 3)
        or segments_b.ndim != 3
        or segments_b.shape[1:] != (2, 3)
        or len(segments_a) == 0
        or len(segments_b) == 0
    ):
        raise ValueError("segments must have non-empty shape (N, 2, 3)")
    start_a = segments_a[:, None, 0, :]
    end_a = segments_a[:, None, 1, :]
    start_b = segments_b[None, :, 0, :]
    end_b = segments_b[None, :, 1, :]
    direction_a = end_a - start_a
    direction_b = end_b - start_b
    relative = start_a - start_b
    aa = np.sum(direction_a * direction_a, axis=2)
    ab = np.sum(direction_a * direction_b, axis=2)
    bb = np.sum(direction_b * direction_b, axis=2)
    ar = np.sum(direction_a * relative, axis=2)
    br = np.sum(direction_b * relative, axis=2)

    def points_to_b(points):
        numerator = np.sum((points - start_b) * direction_b, axis=2)
        parameter = np.divide(
            numerator,
            bb,
            out=np.zeros_like(numerator),
            where=bb > 1e-18,
        )
        parameter = np.clip(parameter, 0.0, 1.0)
        closest = start_b + parameter[:, :, None] * direction_b
        return np.linalg.norm(points - closest, axis=2)

    def points_to_a(points):
        numerator = np.sum((points - start_a) * direction_a, axis=2)
        parameter = np.divide(
            numerator,
            aa,
            out=np.zeros_like(numerator),
            where=aa > 1e-18,
        )
        parameter = np.clip(parameter, 0.0, 1.0)
        closest = start_a + parameter[:, :, None] * direction_a
        return np.linalg.norm(points - closest, axis=2)

    determinant = aa * bb - ab * ab
    parameter_a = np.divide(
        ab * br - bb * ar,
        determinant,
        out=np.zeros_like(determinant),
        where=determinant > 1e-18,
    )
    parameter_b = np.divide(
        aa * br - ab * ar,
        determinant,
        out=np.zeros_like(determinant),
        where=determinant > 1e-18,
    )
    interior_valid = (
        (determinant > 1e-18)
        & (parameter_a >= 0.0)
        & (parameter_a <= 1.0)
        & (parameter_b >= 0.0)
        & (parameter_b <= 1.0)
    )
    interior_a = start_a + parameter_a[:, :, None] * direction_a
    interior_b = start_b + parameter_b[:, :, None] * direction_b
    interior_distance = np.where(
        interior_valid,
        np.linalg.norm(interior_a - interior_b, axis=2),
        np.inf,
    )
    centerline_distance = float(
        np.min(
            np.stack(
                (
                    points_to_b(start_a),
                    points_to_b(end_a),
                    points_to_a(start_b),
                    points_to_a(end_b),
                    interior_distance,
                )
            )
        )
    )
    return float(centerline_distance - 2.0 * capsule_radius)


def tip_jacobian(q, finger, eps=1e-6):
    idxs = FINGER_JOINT_INDEX[finger]
    J = np.zeros((3, 4), dtype=np.float64)

    for col, qidx in enumerate(idxs):
        q_plus = q.copy()
        q_minus = q.copy()
        q_plus[qidx] += eps
        q_minus[qidx] -= eps
        J[:, col] = (
            tip_position(q_plus, finger)
            - tip_position(q_minus, finger)
        ) / (2.0 * eps)

    return J
