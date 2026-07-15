import type { Point3, RotationMatrix3 } from "./types";

const ROTATION_MATRIX_SIZE = 9;
const ROTATION_TOLERANCE = 0.05;

export const IDENTITY_ROTATION_MATRIX: RotationMatrix3 = [
  1, 0, 0,
  0, 1, 0,
  0, 0, 1,
];

export function decodeRotationMatrix(values: unknown): RotationMatrix3 | null {
  if (!Array.isArray(values) || values.length !== ROTATION_MATRIX_SIZE) return null;

  const matrix = values.map(Number);
  if (matrix.some((value) => !Number.isFinite(value))) return null;

  const rowDot = (left: number, right: number) => (
    matrix[left * 3] * matrix[right * 3]
    + matrix[left * 3 + 1] * matrix[right * 3 + 1]
    + matrix[left * 3 + 2] * matrix[right * 3 + 2]
  );
  let maximumOrthogonalityError = 0;
  for (let left = 0; left < 3; left += 1) {
    for (let right = 0; right < 3; right += 1) {
      const expected = left === right ? 1 : 0;
      maximumOrthogonalityError = Math.max(
        maximumOrthogonalityError,
        Math.abs(rowDot(left, right) - expected),
      );
    }
  }

  const determinant = (
    matrix[0] * (matrix[4] * matrix[8] - matrix[5] * matrix[7])
    - matrix[1] * (matrix[3] * matrix[8] - matrix[5] * matrix[6])
    + matrix[2] * (matrix[3] * matrix[7] - matrix[4] * matrix[6])
  );
  if (
    maximumOrthogonalityError > ROTATION_TOLERANCE
    || Math.abs(determinant - 1) > ROTATION_TOLERANCE
  ) {
    return null;
  }

  return matrix as unknown as RotationMatrix3;
}

export function rotateVectorToWorld(
  vector: Point3,
  handToWorld: RotationMatrix3,
): Point3 {
  return {
    x: handToWorld[0] * vector.x + handToWorld[1] * vector.y + handToWorld[2] * vector.z,
    y: handToWorld[3] * vector.x + handToWorld[4] * vector.y + handToWorld[5] * vector.z,
    z: handToWorld[6] * vector.x + handToWorld[7] * vector.y + handToWorld[8] * vector.z,
  };
}
